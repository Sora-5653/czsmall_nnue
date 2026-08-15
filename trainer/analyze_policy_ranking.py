# SPDX-License-Identifier: MIT
"""Diagnose policy CE vs action-ranking quality on the fixed CNN ablation split.

This evaluates trained checkpoints on the exact game-level held-out split used by
`run_cnn_ablation.py`.  It treats the search visit distribution as the teacher and
reports both distributional metrics (cross-entropy/KL-style quantities) and the
ranking metrics that matter more directly for policy-only play.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

import tetra_dataset
from ablation_models import load_ablation_checkpoint
from train import split_indices_by_game


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    x = (value + 0x9E3779B97F4A7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    return (x ^ (x >> 31)) & mask


def split_indices_by_game_hashed(ds: tetra_dataset.Dataset) -> tuple[np.ndarray, np.ndarray]:
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


def default_dataset_paths() -> list[str]:
    paths = sorted(glob.glob("data/production/gen4_bootstrap_20260807.part*.tetradat"))
    paths += sorted(glob.glob("data/production/gen4_search32_[a-f]_20260807.part*.tetradat"))
    return paths


def default_checkpoints() -> list[str]:
    root = r"C:\Users\eddyf\AppData\Local\Temp\czsmall_ablation_400"
    return [
        root + r"\transformer.final.pt",
        root + r"\cnn.final.pt",
        root + r"\hybrid.final.pt",
    ]


def checkpoint_name(path: str) -> str:
    name = Path(path).name
    if "." in name:
        name = name.split(".", 1)[0]
    return name


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--checkpoints", nargs="*", default=None)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--expect-samples", type=int, default=47693)
    ap.add_argument("--split", choices=("ordered", "hashed"), default="ordered")
    ap.add_argument("--output", default="models/ablation_cnn_20260808/policy_ranking.json")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("GPU requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.type == "cuda":
        print(f"device        {torch.cuda.get_device_name(device)}", flush=True)

    paths = list(args.datasets) if args.datasets else default_dataset_paths()
    checkpoints = list(args.checkpoints) if args.checkpoints else default_checkpoints()
    loaded = [tetra_dataset.load(path) for path in paths]
    dataset = tetra_dataset.Dataset.concatenate(loaded)
    dataset.sanity_check()
    if args.expect_samples > 0 and len(dataset) != args.expect_samples:
        raise SystemExit(
            f"sample-count mismatch: expected {args.expect_samples}, got {len(dataset)}"
        )
    if args.split == "hashed":
        _, val_np = split_indices_by_game_hashed(dataset)
    else:
        _, val_np = split_indices_by_game(dataset)
    val_idx = torch.as_tensor(val_np, dtype=torch.long)
    data = dataset.torch()
    print(f"held-out      {len(val_idx)} samples", flush=True)

    # Compute teacher entropy once so every model uses the exact same strata.
    target = data["policy_target"][val_idx]
    mask = data["action_mask"][val_idx]
    teacher_log = torch.where(target > 0, target.log(), torch.zeros_like(target))
    teacher_entropy = -(target * teacher_log).sum(dim=-1)
    legal_count = mask.sum(dim=-1).clamp(min=1)
    normalized_entropy = teacher_entropy / legal_count.log().clamp(min=1e-6)
    quantiles = torch.quantile(normalized_entropy, torch.tensor([0.25, 0.5, 0.75]))
    entropy_edges = [float("-inf"), *[float(x) for x in quantiles.tolist()], float("inf")]
    print(
        "entropy bins  " + " / ".join(f"{x:.4f}" for x in quantiles.tolist()),
        flush=True,
    )

    results: dict[str, object] = {
        "dataset": {
            "samples": len(dataset),
            "held_out_samples": len(val_idx),
            "held_out_game_seeds": len(set(map(int, dataset.game_seed[val_np].tolist()))),
        },
        "entropy_quantiles_normalized": [float(x) for x in quantiles.tolist()],
        "models": {},
    }

    for checkpoint in checkpoints:
        name = checkpoint_name(checkpoint)
        model = load_ablation_checkpoint(checkpoint, device)
        total_count = 0
        ce_sum = 0.0
        teacher_entropy_sum = 0.0
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        reciprocal_rank_sum = 0.0
        teacher_best_rank_sum = 0.0
        teacher_mass_at_model_argmax_sum = 0.0
        teacher_best_mass_sum = 0.0
        teacher_regret_sum = 0.0
        model_top1_prob_sum = 0.0
        model_prob_teacher_best_sum = 0.0
        chosen_valid_count = 0
        chosen_top1_correct = 0
        chosen_rank_sum = 0.0
        model_prob_chosen_sum = 0.0
        entropy_bins = [
            {"count": 0, "ce_sum": 0.0, "top1": 0, "rank_sum": 0.0,
             "argmax_teacher_mass_sum": 0.0}
            for _ in range(4)
        ]

        with torch.inference_mode():
            for offset in range(0, len(val_idx), args.batch):
                idx = val_idx[offset:offset + args.batch]
                batch = {key: value[idx].to(device, non_blocking=True)
                         for key, value in data.items()}
                logits, _, _ = model(
                    batch["tokens"], batch["token_mask"],
                    batch["actions"], batch["action_mask"]
                )
                logp = torch.log_softmax(logits.float(), dim=-1)
                prob = logp.exp()
                target_b = batch["policy_target"]
                legal_b = batch["action_mask"] > 0.5
                count = target_b.shape[0]

                ce = -(target_b * logp.nan_to_num(neginf=0.0)).sum(dim=-1)
                teacher_best = target_b.argmax(dim=-1)
                model_best = logits.argmax(dim=-1)
                teacher_best_mass = target_b.gather(1, teacher_best[:, None]).squeeze(1)
                chosen_teacher_mass = target_b.gather(1, model_best[:, None]).squeeze(1)
                chosen_model_prob = prob.gather(1, model_best[:, None]).squeeze(1)
                teacher_best_model_prob = prob.gather(1, teacher_best[:, None]).squeeze(1)

                teacher_best_logit = logits.gather(1, teacher_best[:, None])
                rank = 1 + ((logits > teacher_best_logit) & legal_b).sum(dim=-1)
                agreement = model_best == teacher_best

                chosen = batch.get("chosen_action")
                if chosen is not None:
                    chosen = chosen.long()
                    chosen_valid = (chosen >= 0) & (chosen < logits.shape[1])
                    safe_chosen = chosen.clamp(0, max(0, logits.shape[1] - 1))
                    chosen_valid = chosen_valid & legal_b.gather(
                        1, safe_chosen[:, None]
                    ).squeeze(1)
                    if chosen_valid.any():
                        chosen_logit = logits.gather(1, safe_chosen[:, None])
                        chosen_rank = 1 + ((logits > chosen_logit) & legal_b).sum(dim=-1)
                        chosen_prob = prob.gather(1, safe_chosen[:, None]).squeeze(1)
                        chosen_valid_count += int(chosen_valid.sum().item())
                        chosen_top1_correct += int(
                            ((model_best == safe_chosen) & chosen_valid).sum().item()
                        )
                        chosen_rank_sum += float(
                            chosen_rank[chosen_valid].float().sum().item()
                        )
                        model_prob_chosen_sum += float(chosen_prob[chosen_valid].sum().item())

                # top-k uses logits directly. k is clipped to the rectangular action
                # dimension; padded actions are already -inf in every ablation model.
                max_k = min(5, logits.shape[1])
                topk = logits.topk(max_k, dim=-1).indices
                top3 = (topk[:, :min(3, max_k)] == teacher_best[:, None]).any(dim=1)
                top5 = (topk[:, :max_k] == teacher_best[:, None]).any(dim=1)

                total_count += count
                ce_sum += float(ce.sum().item())
                top1_correct += int(agreement.sum().item())
                top3_correct += int(top3.sum().item())
                top5_correct += int(top5.sum().item())
                reciprocal_rank_sum += float((1.0 / rank.float()).sum().item())
                teacher_best_rank_sum += float(rank.float().sum().item())
                teacher_mass_at_model_argmax_sum += float(chosen_teacher_mass.sum().item())
                teacher_best_mass_sum += float(teacher_best_mass.sum().item())
                teacher_regret_sum += float((teacher_best_mass - chosen_teacher_mass).sum().item())
                model_top1_prob_sum += float(chosen_model_prob.sum().item())
                model_prob_teacher_best_sum += float(teacher_best_model_prob.sum().item())

                target_log = torch.where(target_b > 0, target_b.log(), torch.zeros_like(target_b))
                entropy = -(target_b * target_log).sum(dim=-1)
                nlegal = legal_b.sum(dim=-1).float().clamp(min=1)
                norm_entropy = entropy / nlegal.log().clamp(min=1e-6)
                teacher_entropy_sum += float(norm_entropy.sum().item())

                for bin_id in range(4):
                    lower = entropy_edges[bin_id]
                    upper = entropy_edges[bin_id + 1]
                    selected = (norm_entropy > lower) & (norm_entropy <= upper)
                    if not selected.any():
                        continue
                    item = entropy_bins[bin_id]
                    selected_count = int(selected.sum().item())
                    item["count"] += selected_count
                    item["ce_sum"] += float(ce[selected].sum().item())
                    item["top1"] += int(agreement[selected].sum().item())
                    item["rank_sum"] += float(rank[selected].float().sum().item())
                    item["argmax_teacher_mass_sum"] += float(
                        chosen_teacher_mass[selected].sum().item()
                    )

        metrics = {
            "cross_entropy": ce_sum / total_count,
            "top1_teacher_argmax_accuracy": top1_correct / total_count,
            "top3_teacher_best_recall": top3_correct / total_count,
            "top5_teacher_best_recall": top5_correct / total_count,
            "mean_teacher_best_rank": teacher_best_rank_sum / total_count,
            "mean_reciprocal_rank_teacher_best": reciprocal_rank_sum / total_count,
            "mean_teacher_mass_at_model_argmax": teacher_mass_at_model_argmax_sum / total_count,
            "mean_teacher_best_mass": teacher_best_mass_sum / total_count,
            "mean_teacher_mass_regret": teacher_regret_sum / total_count,
            "mean_model_top1_probability": model_top1_prob_sum / total_count,
            "mean_model_probability_teacher_best": model_prob_teacher_best_sum / total_count,
            "chosen_action_valid_count": chosen_valid_count,
            "chosen_action_top1_accuracy": chosen_top1_correct / max(1, chosen_valid_count),
            "mean_chosen_action_rank": chosen_rank_sum / max(1, chosen_valid_count),
            "mean_model_probability_chosen_action": model_prob_chosen_sum / max(1, chosen_valid_count),
            "mean_normalized_teacher_entropy": teacher_entropy_sum / total_count,
            "entropy_bins": [],
        }
        for bin_id, item in enumerate(entropy_bins):
            count = item["count"]
            metrics["entropy_bins"].append({
                "bin": bin_id,
                "count": count,
                "cross_entropy": item["ce_sum"] / max(1, count),
                "top1_teacher_argmax_accuracy": item["top1"] / max(1, count),
                "mean_teacher_best_rank": item["rank_sum"] / max(1, count),
                "mean_teacher_mass_at_model_argmax": (
                    item["argmax_teacher_mass_sum"] / max(1, count)
                ),
            })

        results["models"][name] = metrics
        print(
            f"{name:11s} CE {metrics['cross_entropy']:.6f}  "
            f"top1 {metrics['top1_teacher_argmax_accuracy']:.4f}  "
            f"top3 {metrics['top3_teacher_best_recall']:.4f}  "
            f"MRR {metrics['mean_reciprocal_rank_teacher_best']:.4f}  "
            f"chosen1 {metrics['chosen_action_top1_accuracy']:.4f}  "
            f"teacher-mass@argmax {metrics['mean_teacher_mass_at_model_argmax']:.4f}  "
            f"regret {metrics['mean_teacher_mass_regret']:.4f}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"results       {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
