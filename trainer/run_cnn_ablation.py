# SPDX-License-Identifier: MIT
"""Run a fixed-data Transformer/CNN/hybrid architecture ablation.

Defaults reproduce the 2026-08-07 ~48k-sample corpus:
  - gen4_bootstrap_20260807.part00..07
  - gen4_search32_[a-f]_20260807.part00..07

All architectures use the exact same dataset order, game-level split, sampled batch
indices, AdamW hyperparameters, number of steps, batch size, and seed.  The primary
comparison metric is final held-out policy cross-entropy.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import tetra_dataset
from ablation_models import build_ablation_model, checkpoint_config
from tetraformer import losses
from train import split_indices_by_game


LOSS_WEIGHTS = {"policy": 1.0, "value": 1.0, "aux": 0.1, "timing_pair": 0.0, "timing_rank": 0.0}


def default_dataset_paths() -> list[str]:
    paths = sorted(glob.glob("data/production/gen4_bootstrap_20260807.part*.tetradat"))
    paths += sorted(glob.glob("data/production/gen4_search32_[a-f]_20260807.part*.tetradat"))
    return paths


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    x = (value + 0x9E3779B97F4A7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    return (x ^ (x >> 31)) & mask


def split_indices_by_game_hashed(ds: tetra_dataset.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic leakage-safe 80/20 split distributed across seed ranges."""
    seeds = np.asarray(ds.game_seed, dtype=np.uint64)
    unique = sorted({int(seed) for seed in seeds.tolist() if int(seed) != 0})
    if len(unique) < 2:
        return split_indices_by_game(ds)
    validation_seeds = {seed for seed in unique if _splitmix64(seed) % 5 == 0}
    if not validation_seeds:
        validation_seeds.add(unique[-1])
    if len(validation_seeds) == len(unique):
        validation_seeds.remove(unique[0])
    validation = np.asarray([int(seed) in validation_seeds for seed in seeds], dtype=bool)
    return np.flatnonzero(~validation), np.flatnonzero(validation)


def save_model(path: Path, architecture: str, model: torch.nn.Module, step: int,
               args: argparse.Namespace, metrics: dict[str, float],
               optimizer: torch.optim.Optimizer | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(
        {
            "format_version": 1,
            "architecture": architecture,
            "config": checkpoint_config(model),
            "step": step,
            "seed": args.seed,
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch": args.batch,
            "loss_weights": dict(LOSS_WEIGHTS),
            "metrics": dict(metrics),
            "state_dict": state,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        },
        path,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--dataset-result-json", default=None,
                    help="reuse the exact dataset path list stored in a prior ablation result JSON")
    ap.add_argument("--replace-dataset-prefix", nargs=2, action="append", default=[],
                    metavar=("OLD", "NEW"),
                    help="replace a dataset path prefix after loading --dataset-result-json; repeatable")
    ap.add_argument(
        "--architectures", nargs="+", choices=("transformer", "cnn", "hybrid", "spatial_hybrid", "pooled_spatial_hybrid", "augmented_pooled_hybrid", "split_hybrid", "fusion_hybrid", "dual_policy_hybrid"),
        default=("transformer", "cnn", "hybrid"),
    )
    ap.add_argument("--expect-samples", type=int, default=47693)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--microbatch", type=int, default=0,
                    help="split each sampled training batch for gradient accumulation; 0 uses --batch")
    ap.add_argument("--eval-batch", type=int, default=0,
                    help="validation batch size; 0 uses --batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--timing-pair-weight", type=float, default=0.0,
                    help="experimental soft FASTEST-vs-WAIT conditional policy loss; 0 preserves baseline")
    ap.add_argument("--timing-rank-weight", type=float, default=0.0,
                    help="experimental margin-weighted FASTEST-vs-WAIT ranking loss; 0 preserves baseline")
    ap.add_argument("--factor-timing-policy", action="store_true",
                    help="factor FASTEST/WAIT logits so timing only splits base-placement prior mass")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=("ordered", "hashed"), default="ordered",
                    help="ordered preserves train.py behaviour; hashed distributes games across seed ranges")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--output-dir", default="models/ablation_cnn_20260808")
    ap.add_argument("--result-json", default="models/ablation_cnn_20260808/results.json")
    ap.add_argument("--resume-training", default=None,
                    help="resume model+AdamW state and continue from checkpoint step to --steps")
    ap.add_argument("--pooled-cnn-channels", type=int, default=192,
                    help="CNN width for pooled_spatial_hybrid")
    ap.add_argument("--pooled-cnn-blocks", type=int, default=10,
                    help="residual CNN blocks for pooled_spatial_hybrid")
    args = ap.parse_args()
    if args.timing_pair_weight < 0.0 or args.timing_rank_weight < 0.0:
        raise SystemExit("timing loss weights must be non-negative")
    LOSS_WEIGHTS["timing_pair"] = args.timing_pair_weight
    LOSS_WEIGHTS["timing_rank"] = args.timing_rank_weight

    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("GPU requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.type == "cuda":
        print(f"device        {torch.cuda.get_device_name(device)}", flush=True)

    if args.dataset_result_json:
        if args.datasets:
            raise SystemExit("--dataset-result-json and --datasets are mutually exclusive")
        prior = json.loads(Path(args.dataset_result_json).read_text(encoding="utf-8"))
        paths = list(prior["dataset"]["paths"])
    else:
        paths = list(args.datasets) if args.datasets else default_dataset_paths()
    for old_prefix, new_prefix in args.replace_dataset_prefix:
        replaced = sum(old_prefix in p for p in paths)
        if replaced == 0:
            raise SystemExit(f"dataset prefix not found: {old_prefix}")
        paths = [p.replace(old_prefix, new_prefix, 1) if old_prefix in p else p for p in paths]
    if not paths:
        raise SystemExit("no ablation datasets found")
    print(f"inputs        {len(paths)} shards", flush=True)
    loaded = []
    for path in paths:
        ds = tetra_dataset.load(path)
        loaded.append(ds)
        print(f"  {path}: {len(ds)}", flush=True)
    dataset = tetra_dataset.Dataset.concatenate(loaded)
    del loaded
    dataset.sanity_check()
    if args.expect_samples > 0 and len(dataset) != args.expect_samples:
        raise SystemExit(
            f"sample-count mismatch: expected {args.expect_samples}, got {len(dataset)}"
        )
    print(
        f"dataset       {len(dataset)} samples, {dataset.tokens.shape[1]} tokens, "
        f"{dataset.actions.shape[1]} actions",
        flush=True,
    )

    if args.split == "hashed":
        train_np, val_np = split_indices_by_game_hashed(dataset)
    else:
        train_np, val_np = split_indices_by_game(dataset)
    train_idx = torch.as_tensor(train_np, dtype=torch.long)
    val_idx = torch.as_tensor(val_np, dtype=torch.long)
    print(
        f"split         train {len(train_idx)} / validation {len(val_idx)} "
        f"({len(set(map(int, dataset.game_seed[train_np].tolist())))} / "
        f"{len(set(map(int, dataset.game_seed[val_np].tolist())))} game seeds)",
        flush=True,
    )
    data = dataset.torch()
    header = dataset.header

    def batch_at(indices: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = {key: value[indices] for key, value in data.items()}
        if device.type != "cpu":
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        return batch

    def evaluate(model: torch.nn.Module) -> dict[str, float]:
        model.eval()
        totals = {
            "policy": 0.0,
            "value": 0.0,
            "value_accuracy": 0.0,
            "value_scalar_mse": 0.0,
            "timing_pair": 0.0,
            "timing_rank": 0.0,
        }
        count_total = 0
        aux_num = 0.0
        aux_den = 0.0
        eval_batch = args.eval_batch if args.eval_batch > 0 else args.batch
        with torch.inference_mode():
            for offset in range(0, len(val_idx), eval_batch):
                idx = val_idx[offset:offset + eval_batch]
                batch = batch_at(idx)
                _, parts = losses(model, batch, weights=LOSS_WEIGHTS)
                count = len(idx)
                count_total += count
                for key in totals:
                    totals[key] += float(parts[key]) * count
                valid = float(batch["aux_valid_mask"].sum().item())
                aux_num += float(parts["aux"]) * valid
                aux_den += valid
        model.train()
        result = {key: value / max(1, count_total) for key, value in totals.items()}
        result["aux"] = aux_num / aux_den if aux_den else 0.0
        result["total"] = (
            LOSS_WEIGHTS["policy"] * result["policy"]
            + LOSS_WEIGHTS["value"] * result["value"]
            + LOSS_WEIGHTS["aux"] * result["aux"]
            + LOSS_WEIGHTS["timing_pair"] * result["timing_pair"]
            + LOSS_WEIGHTS["timing_rank"] * result["timing_rank"]
        )
        return result

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "dataset": {
            "paths": paths,
            "samples": len(dataset),
            "train_samples": len(train_idx),
            "validation_samples": len(val_idx),
        },
        "training": {
            "optimizer": "AdamW",
            "steps": args.steps,
            "batch": args.batch,
            "microbatch": args.microbatch if args.microbatch > 0 else args.batch,
            "eval_batch": args.eval_batch if args.eval_batch > 0 else args.batch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "split": args.split,
            "loss_weights": dict(LOSS_WEIGHTS),
            "device": str(device),
            "pooled_cnn_channels": args.pooled_cnn_channels,
            "pooled_cnn_blocks": args.pooled_cnn_blocks,
            "factor_timing_policy": args.factor_timing_policy,
        },
        "architectures": {},
    }

    # Pre-generate the exact sampled training indices once.  Every architecture
    # receives the same minibatches in the same order, not merely the same RNG seed.
    generator = torch.Generator().manual_seed(args.seed + 1)
    sample_width = min(args.batch, len(train_idx))
    batch_schedule = torch.randint(
        0, len(train_idx), (args.steps, sample_width), generator=generator
    )

    for architecture in args.architectures:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = build_ablation_model(
            architecture, header.token_features, header.action_features, header.aux_targets,
            pooled_cnn_channels=args.pooled_cnn_channels,
            pooled_cnn_blocks=args.pooled_cnn_blocks,
            factor_timing_policy=args.factor_timing_policy,
        ).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        start_step = 0
        if args.resume_training is not None:
            if len(args.architectures) != 1:
                raise SystemExit("--resume-training requires exactly one architecture")
            resume = torch.load(args.resume_training, map_location="cpu", weights_only=False)
            if resume.get("architecture") != architecture:
                raise SystemExit(
                    f"resume architecture mismatch: {resume.get('architecture')} != {architecture}"
                )
            if int(resume.get("seed", args.seed)) != args.seed:
                raise SystemExit("resume seed mismatch")
            model.load_state_dict(resume["state_dict"])
            optimizer_state = resume.get("optimizer_state_dict")
            if optimizer_state is None:
                raise SystemExit("resume checkpoint has no optimizer_state_dict")
            optimizer.load_state_dict(optimizer_state)
            start_step = int(resume.get("step", 0))
            if start_step >= args.steps:
                raise SystemExit(
                    f"resume step {start_step} must be smaller than target --steps {args.steps}"
                )
            print(f"resume        {args.resume_training} @ step {start_step}", flush=True)
        start = evaluate(model)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        print(
            f"\n[{architecture}] params {parameter_count:,}  "
            f"val-start policy {start['policy']:.6f} total {start['total']:.6f}",
            flush=True,
        )
        best_policy = start["policy"]
        best_step = start_step
        best_metrics = dict(start)
        save_model(
            output_dir / f"{architecture}.best.pt", architecture, model,
            start_step, args, start, optimizer
        )

        model.train()
        t0 = time.time()
        remaining_steps = args.steps - start_step
        log_every = max(1, remaining_steps // 10)
        microbatch = args.microbatch if args.microbatch > 0 else sample_width
        microbatch = min(max(1, microbatch), sample_width)
        for local_step in range(start_step + 1, args.steps + 1):
            pick = train_idx[batch_schedule[local_step - 1]]
            optimizer.zero_grad(set_to_none=True)
            parts_accum: dict[str, float] = {}
            pick_count = len(pick)
            for micro_offset in range(0, pick_count, microbatch):
                micro_pick = pick[micro_offset:micro_offset + microbatch]
                total, micro_parts = losses(model, batch_at(micro_pick), weights=LOSS_WEIGHTS)
                scale = len(micro_pick) / max(1, pick_count)
                (total * scale).backward()
                for key, value in micro_parts.items():
                    if key.startswith("_") or not isinstance(value, (int, float)):
                        continue
                    parts_accum[key] = parts_accum.get(key, 0.0) + float(value) * scale
            parts = parts_accum
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if local_step % log_every == 0 or local_step == args.steps:
                print(
                    f"[{architecture}] step {local_step:4d}/{args.steps} "
                    f"train policy {parts['policy']:.5f} total {parts['total']:.5f}",
                    flush=True,
                )
            if args.eval_every > 0 and (
                local_step % args.eval_every == 0 or local_step == args.steps
            ):
                metrics = evaluate(model)
                print(
                    f"[{architecture}] val step {local_step:4d} "
                    f"policy {metrics['policy']:.6f} total {metrics['total']:.6f}",
                    flush=True,
                )
                if metrics["policy"] < best_policy:
                    best_policy = metrics["policy"]
                    best_step = local_step
                    best_metrics = dict(metrics)
                    save_model(
                        output_dir / f"{architecture}.best.pt",
                        architecture,
                        model,
                        local_step,
                        args,
                        metrics,
                        optimizer,
                    )

        elapsed = time.time() - t0
        peak_allocated = 0.0
        peak_reserved = 0.0
        if device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        final = evaluate(model)
        save_model(
            output_dir / f"{architecture}.final.pt",
            architecture,
            model,
            args.steps,
            args,
            final,
            optimizer,
        )
        print(
            f"[{architecture}] FINAL policy {final['policy']:.6f} total {final['total']:.6f} "
            f"best-policy {best_policy:.6f}@{best_step}  {elapsed:.1f}s "
            f"peak-alloc {peak_allocated:.2f}GiB peak-reserved {peak_reserved:.2f}GiB",
            flush=True,
        )
        results["architectures"][architecture] = {
            "parameters": parameter_count,
            "start": start,
            "final": final,
            "best": best_metrics,
            "best_step": best_step,
            "seconds": elapsed,
            "peak_allocated_gib": peak_allocated,
            "peak_reserved_gib": peak_reserved,
            "final_checkpoint": str(output_dir / f"{architecture}.final.pt"),
            "best_checkpoint": str(output_dir / f"{architecture}.best.pt"),
        }
        del optimizer, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults       {result_path}", flush=True)

    ranking = sorted(
        (
            (name, info["final"]["policy"])
            for name, info in results["architectures"].items()
        ),
        key=lambda item: item[1],
    )
    print("held-out policy ranking:", flush=True)
    for rank, (name, policy) in enumerate(ranking, 1):
        print(f"  {rank}. {name:11s} {policy:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
