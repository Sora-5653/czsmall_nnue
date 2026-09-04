# SPDX-License-Identifier: MIT
"""Probe whether a frozen TetraFormer trunk exposes cumulative 8s top-out risk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tetra_dataset
from analyze_value_ranking import binary_auc
from gpu_match import load_model
from train import split_indices_by_game


@torch.no_grad()
def pooled_features(model, data: dict[str, torch.Tensor], indices: np.ndarray,
                    device: torch.device, batch_size: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    for offset in range(0, len(indices), batch_size):
        idx = torch.as_tensor(indices[offset:offset + batch_size], dtype=torch.long)
        tokens = data["tokens"][idx].to(device)
        token_mask = data["token_mask"][idx].to(device)
        pad = token_mask < 0.5
        x = model.token_in(tokens)
        for block in model.blocks:
            x = block(x, pad)
        x = model.norm(x)
        w = token_mask.unsqueeze(-1)
        pooled = (x * w).sum(1) / w.sum(1).clamp(min=1.0)
        parts.append(pooled.float().cpu().numpy())
    return np.concatenate(parts)


def auc_or_nan(scores: np.ndarray, labels: np.ndarray) -> float:
    return binary_auc(scores.astype(np.float64), labels.astype(np.float64))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    loaded = [tetra_dataset.load(path) for path in args.datasets]
    ds = tetra_dataset.Dataset.concatenate(loaded)
    ds.sanity_check()
    train_idx, val_idx = split_indices_by_game(ds)
    device = torch.device(args.device)
    base = load_model(args.model, device)
    base.eval()
    data = ds.torch()

    train_x = pooled_features(base, data, train_idx, device, max(1, args.batch))
    val_x = pooled_features(base, data, val_idx, device, max(1, args.batch))
    self_indices = [6, 10, 14, 18]
    opp_indices = [7, 11, 15, 19]
    train_y = np.stack([
        ds.aux_target[train_idx][:, self_indices].max(axis=1),
        ds.aux_target[train_idx][:, opp_indices].max(axis=1),
    ], axis=1).astype(np.float32)
    val_y = np.stack([
        ds.aux_target[val_idx][:, self_indices].max(axis=1),
        ds.aux_target[val_idx][:, opp_indices].max(axis=1),
    ], axis=1).astype(np.float32)
    val_z = ds.value_target[val_idx].astype(np.float64)

    x_train = torch.from_numpy(train_x).to(device)
    y_train = torch.from_numpy(train_y).to(device)
    x_val = torch.from_numpy(val_x).to(device)
    width = train_x.shape[1]
    probe = nn.Sequential(
        nn.Linear(width, width // 2), nn.SiLU(), nn.Linear(width // 2, 2)
    ).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(1234)

    for step in range(1, args.steps + 1):
        pick = torch.randint(0, len(x_train), (min(args.batch, len(x_train)),),
                             generator=generator).to(device)
        logits = probe(x_train[pick])
        loss = F.binary_cross_entropy_with_logits(logits, y_train[pick])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, args.steps // 5) == 0:
            print(f"step {step:5d} train_bce={loss.item():.4f}")

    probe.eval()
    with torch.no_grad():
        prob = torch.sigmoid(probe(x_val)).cpu().numpy()
        val_bce = F.binary_cross_entropy_with_logits(
            probe(x_val), torch.from_numpy(val_y).to(device)
        ).item()
    self_auc = auc_or_nan(prob[:, 0], val_y[:, 0])
    opp_auc = auc_or_nan(prob[:, 1], val_y[:, 1])
    balance = prob[:, 1] - prob[:, 0]
    outcome_auc = auc_or_nan(balance, (val_z > 0.5).astype(np.float64))
    print(f"validation   bce={val_bce:.4f}")
    print(f"self_topout  AUC={self_auc:.3f} positive_rate={val_y[:,0].mean():.3f}")
    print(f"opp_topout   AUC={opp_auc:.3f} positive_rate={val_y[:,1].mean():.3f}")
    print(f"balance->win AUC={outcome_auc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
