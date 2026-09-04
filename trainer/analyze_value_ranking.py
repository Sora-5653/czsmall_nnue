# SPDX-License-Identifier: MIT
"""Diagnose whether a value head ranks held-out positions usefully.

Cross-entropy can improve simply by shrinking overconfident predictions toward
50/50. MCTS also needs ordinal information: positions from winning trajectories
should receive larger scalar values than positions from losing trajectories.
This tool reports both calibration losses and ranking/separation metrics on the
same game-level held-out split used by train.py.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tetra_dataset
from gpu_match import load_model
from train import split_indices_by_game


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties; labels are 0/1."""
    pos = labels > 0.5
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        average_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = average_rank
        i = j
    rank_sum_pos = float(ranks[pos].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def analyze(model_path: str, dataset: tetra_dataset.Dataset, val_idx: np.ndarray,
            device: torch.device, batch_size: int) -> dict[str, float]:
    model = load_model(model_path, device)
    model.eval()
    data = dataset.torch()
    scalars: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    ce_sum = 0.0
    count = 0

    with torch.no_grad():
        for offset in range(0, len(val_idx), batch_size):
            idx_np = val_idx[offset:offset + batch_size]
            idx = torch.as_tensor(idx_np, dtype=torch.long)
            batch = {key: value[idx] for key, value in data.items()}
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            _, wdl, _ = model(
                batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"]
            )
            z = batch["value_target"]
            win = (z > 0.5).float()
            loss = (z < -0.5).float()
            draw = 1.0 - win - loss
            target = torch.stack([win, draw, loss], dim=-1)
            ce = -(target * torch.log_softmax(wdl, dim=-1)).sum(-1)
            prob = torch.softmax(wdl, dim=-1)
            scalar = prob[:, 0] - prob[:, 2]
            ce_sum += float(ce.sum().item())
            count += len(idx_np)
            scalars.append(scalar.cpu().numpy())
            targets.append(z.cpu().numpy())

    s = np.concatenate(scalars).astype(np.float64)
    z = np.concatenate(targets).astype(np.float64)
    decisive = np.abs(z) > 0.5
    sd = s[decisive]
    zd = z[decisive]
    labels = (zd > 0.5).astype(np.float64)
    auc = binary_auc(sd, labels)
    wins = sd[labels > 0.5]
    losses = sd[labels < 0.5]
    correlation = float(np.corrcoef(sd, zd)[0, 1]) if len(sd) > 1 and np.std(sd) > 0 else float("nan")
    accuracy = float(((sd > 0.0) == (zd > 0.0)).mean()) if len(sd) else float("nan")
    mse = float(np.mean((s - z) ** 2))
    remaining_norm = dataset.aux_target[val_idx, 2].astype(np.float64)
    remaining = 64.0 * remaining_norm / np.clip(1.0 - remaining_norm, 1e-8, None)
    buckets: dict[str, tuple[float, float, int]] = {}
    for name, mask in (
        ("<=8", remaining <= 8.5),
        ("9-32", (remaining > 8.5) & (remaining <= 32.5)),
        (">32", remaining > 32.5),
    ):
        mask = mask & decisive
        if int(mask.sum()) == 0:
            buckets[name] = (float("nan"), float("nan"), 0)
            continue
        bs = s[mask]
        bz = z[mask]
        bl = (bz > 0.5).astype(np.float64)
        bauc = binary_auc(bs, bl)
        bw = bs[bl > 0.5]
        blo = bs[bl < 0.5]
        bsep = float(bw.mean() - blo.mean()) if len(bw) and len(blo) else float("nan")
        buckets[name] = (bauc, bsep, int(mask.sum()))
    return {
        "cross_entropy": ce_sum / max(1, count),
        "scalar_mse": mse,
        "decisive_accuracy": accuracy,
        "auc": auc,
        "correlation": correlation,
        "mean_win_scalar": float(wins.mean()) if len(wins) else float("nan"),
        "mean_loss_scalar": float(losses.mean()) if len(losses) else float("nan"),
        "separation": float(wins.mean() - losses.mean()) if len(wins) and len(losses) else float("nan"),
        "scalar_std": float(sd.std()) if len(sd) else float("nan"),
        "samples": float(len(s)),
        "decisive_samples": float(len(sd)),
        "buckets": buckets,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    loaded = [tetra_dataset.load(path) for path in args.datasets]
    dataset = tetra_dataset.Dataset.concatenate(loaded)
    dataset.sanity_check()
    _, val_idx = split_indices_by_game(dataset)
    device = torch.device(args.device)
    print(f"dataset       {len(dataset)} samples / held-out {len(val_idx)}")
    print("model                                      CE      MSE    acc    AUC   corr    sep   std")
    for model_path in args.models:
        metrics = analyze(model_path, dataset, val_idx, device, max(1, args.batch))
        name = Path(model_path).name
        print(
            f"{name[:40]:40s}  {metrics['cross_entropy']:.4f}  {metrics['scalar_mse']:.4f}  "
            f"{metrics['decisive_accuracy']:.3f}  {metrics['auc']:.3f}  "
            f"{metrics['correlation']:.3f}  {metrics['separation']:+.3f}  "
            f"{metrics['scalar_std']:.3f}"
        )
        print(
            f"  win_mean={metrics['mean_win_scalar']:+.3f} "
            f"loss_mean={metrics['mean_loss_scalar']:+.3f} "
            f"decisive={int(metrics['decisive_samples'])}"
        )
        bucket_text = "  by remaining: " + "  ".join(
            f"{name} AUC={auc:.3f} sep={sep:+.3f} n={count}"
            for name, (auc, sep, count) in metrics["buckets"].items()
        )
        print(bucket_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
