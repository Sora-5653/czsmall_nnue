# SPDX-License-Identifier: MIT
"""Train a controlled TetraFormer size ablation on the fixed Gen-4 corpus.

The experiment varies only Transformer width/depth/FFN size.  Dataset order,
hashed game split, optimizer, sampled minibatches, update count and seed are
identical across sizes so later Arena runs can isolate network capacity from
search budget.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import time
from pathlib import Path

import torch

import tetra_dataset
from run_cnn_ablation import default_dataset_paths, split_indices_by_game_hashed
from tetraformer import TetraFormer, TetraFormerConfig, losses


MODEL_SPECS = {
    # Approximately logarithmic parameter spacing: 0.13M / 0.95M / 7.18M.
    "xs": dict(width=64, layers=2, heads=4, ffn=192),
    "m": dict(width=128, layers=4, heads=4, ffn=384),
    "s": dict(width=256, layers=8, heads=8, ffn=768),
}

LOSS_WEIGHTS = {
    "policy": 1.0,
    "value": 1.0,
    "aux": 0.1,
    "timing_pair": 0.0,
    "timing_rank": 0.0,
}


def make_model(size: str, header: tetra_dataset.Header) -> TetraFormer:
    spec = MODEL_SPECS[size]
    return TetraFormer(
        TetraFormerConfig(
            token_features=header.token_features,
            action_features=header.action_features,
            aux_targets=header.aux_targets,
            **spec,
        )
    )


def checkpoint_payload(
    model: TetraFormer,
    *,
    size: str,
    seed: int,
    step: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "architecture": "transformer",
        "size_ablation": size,
        "config": asdict(model.cfg),
        "step": step,
        "seed": seed,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch": args.batch,
        "loss_weights": dict(LOSS_WEIGHTS),
        "metrics": dict(metrics),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--sizes", nargs="+", choices=tuple(MODEL_SPECS), default=("xs", "m", "s"))
    ap.add_argument("--expect-samples", type=int, default=47693)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--microbatch", type=int, default=0)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--result-json", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("GPU requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.type == "cuda":
        print(f"device        {torch.cuda.get_device_name(device)}", flush=True)

    paths = list(args.datasets) if args.datasets else default_dataset_paths()
    if not paths:
        raise SystemExit("no datasets found")
    print(f"inputs        {len(paths)} shards", flush=True)
    loaded = [tetra_dataset.load(path) for path in paths]
    dataset = tetra_dataset.Dataset.concatenate(loaded)
    del loaded
    dataset.sanity_check()
    if args.expect_samples > 0 and len(dataset) != args.expect_samples:
        raise SystemExit(
            f"sample-count mismatch: expected {args.expect_samples}, got {len(dataset)}"
        )

    train_np, val_np = split_indices_by_game_hashed(dataset)
    train_idx = torch.as_tensor(train_np, dtype=torch.long)
    val_idx = torch.as_tensor(val_np, dtype=torch.long)
    data = dataset.torch()
    header = dataset.header
    print(
        f"dataset       {len(dataset)} samples; train {len(train_idx)} / validation {len(val_idx)}",
        flush=True,
    )

    def batch_at(indices: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = {key: value[indices] for key, value in data.items()}
        if device.type != "cpu":
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        return batch

    def evaluate(model: TetraFormer) -> dict[str, float]:
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
        with torch.inference_mode():
            for offset in range(0, len(val_idx), args.eval_batch):
                idx = val_idx[offset : offset + args.eval_batch]
                batch = batch_at(idx)
                _, parts = losses(model, batch, weights=LOSS_WEIGHTS)
                count = len(idx)
                count_total += count
                for key in totals:
                    totals[key] += float(parts[key]) * count
                valid = float(batch["aux_valid_mask"].sum().item())
                aux_num += float(parts["aux"]) * valid
                aux_den += valid
        result = {key: value / max(1, count_total) for key, value in totals.items()}
        result["aux"] = aux_num / aux_den if aux_den else 0.0
        result["total"] = (
            result["policy"] + result["value"] + LOSS_WEIGHTS["aux"] * result["aux"]
        )
        model.train()
        return result

    generator = torch.Generator().manual_seed(args.seed + 1)
    sample_width = min(args.batch, len(train_idx))
    batch_schedule = torch.randint(
        0, len(train_idx), (args.steps, sample_width), generator=generator
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "dataset": {
            "paths": paths,
            "samples": len(dataset),
            "train_samples": len(train_idx),
            "validation_samples": len(val_idx),
            "split": "hashed-game-80/20",
        },
        "training": {
            "steps": args.steps,
            "batch": args.batch,
            "microbatch": args.microbatch if args.microbatch > 0 else args.batch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "loss_weights": dict(LOSS_WEIGHTS),
            "device": str(device),
        },
        "sizes": {},
    }

    for size in args.sizes:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = make_model(size, header).to(device)
        params = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        initial = evaluate(model)
        best_policy = initial["policy"]
        best_step = 0
        best_metrics = dict(initial)
        print(
            f"\n[{size}] params {params:,}  val-start policy {initial['policy']:.6f}",
            flush=True,
        )

        microbatch = args.microbatch if args.microbatch > 0 else sample_width
        microbatch = max(1, min(microbatch, sample_width))
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        model.train()
        for step in range(1, args.steps + 1):
            pick = train_idx[batch_schedule[step - 1]]
            optimizer.zero_grad(set_to_none=True)
            pick_count = len(pick)
            for offset in range(0, pick_count, microbatch):
                micro_pick = pick[offset : offset + microbatch]
                total, _ = losses(model, batch_at(micro_pick), weights=LOSS_WEIGHTS)
                (total * (len(micro_pick) / pick_count)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps):
                metrics = evaluate(model)
                print(
                    f"[{size}] step {step:4d}: policy {metrics['policy']:.6f} "
                    f"value {metrics['value']:.6f} total {metrics['total']:.6f}",
                    flush=True,
                )
                if metrics["policy"] < best_policy:
                    best_policy = metrics["policy"]
                    best_step = step
                    best_metrics = dict(metrics)

        elapsed = time.perf_counter() - t0
        final = evaluate(model)
        final_path = output_dir / f"transformer_{size}.final.pt"
        torch.save(
            checkpoint_payload(
                model,
                size=size,
                seed=args.seed,
                step=args.steps,
                args=args,
                metrics=final,
                optimizer=optimizer,
            ),
            final_path,
        )
        peak_allocated = 0.0
        peak_reserved = 0.0
        if device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        print(
            f"[{size}] FINAL policy {final['policy']:.6f} total {final['total']:.6f}; "
            f"{elapsed:.1f}s; peak {peak_allocated:.2f}/{peak_reserved:.2f} GiB",
            flush=True,
        )
        results["sizes"][size] = {
            "config": asdict(model.cfg),
            "parameters": params,
            "initial": initial,
            "final": final,
            "best_policy": best_policy,
            "best_step": best_step,
            "best_metrics": best_metrics,
            "seconds": elapsed,
            "peak_allocated_gib": peak_allocated,
            "peak_reserved_gib": peak_reserved,
            "checkpoint": str(final_path),
        }
        del optimizer, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults       {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
