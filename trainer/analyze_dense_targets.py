# SPDX-License-Identifier: MIT
"""Measure how short-horizon tactical targets rank eventual game outcomes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tetra_dataset
from analyze_value_ranking import binary_auc
from train import split_indices_by_game


def unsquash_count(x: np.ndarray, scale: float = 8.0) -> np.ndarray:
    x = np.clip(x.astype(np.float64), 0.0, 1.0 - 1e-8)
    return scale * x / (1.0 - x)


def metric(score: np.ndarray, z: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    mask = mask & (np.abs(z) > 0.5)
    if int(mask.sum()) == 0:
        return float("nan"), float("nan"), 0
    s = score[mask]
    labels = (z[mask] > 0.5).astype(np.float64)
    auc = binary_auc(s, labels)
    wins = s[labels > 0.5]
    losses = s[labels < 0.5]
    sep = float(wins.mean() - losses.mean()) if len(wins) and len(losses) else float("nan")
    return auc, sep, int(mask.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    args = ap.parse_args()

    loaded = [tetra_dataset.load(path) for path in args.datasets]
    ds = tetra_dataset.Dataset.concatenate(loaded)
    ds.sanity_check()
    _, val_idx = split_indices_by_game(ds)
    a = ds.aux_target[val_idx]
    z = ds.value_target[val_idx].astype(np.float64)

    # Schema v3 preserves the original 36 targets and appends 8 garbage-clear
    # counts. Real-time interval channels occupy indices 4..19.
    attack_bins = np.stack([unsquash_count(a[:, 4 + 4 * h]) for h in range(4)], axis=1)
    received_bins = np.stack([unsquash_count(a[:, 5 + 4 * h]) for h in range(4)], axis=1)
    self_topout_bins = np.stack([a[:, 6 + 4 * h] for h in range(4)], axis=1)
    opp_topout_bins = np.stack([a[:, 7 + 4 * h] for h in range(4)], axis=1)
    gclear_bins = np.stack([unsquash_count(a[:, 36 + h]) for h in range(4)], axis=1)

    attack_8 = attack_bins.sum(axis=1)
    received_8 = received_bins.sum(axis=1)
    gclear_8 = gclear_bins.sum(axis=1)
    vs_activity_8 = attack_8 + gclear_8
    net_pressure_8 = vs_activity_8 - received_8
    terminal_balance_8 = opp_topout_bins.max(axis=1) - self_topout_bins.max(axis=1)

    remaining_norm = a[:, 2].astype(np.float64)
    remaining = 64.0 * remaining_norm / np.clip(1.0 - remaining_norm, 1e-8, None)
    buckets = {
        "all": np.ones(len(z), dtype=bool),
        "<=8": remaining <= 8.5,
        "9-32": (remaining > 8.5) & (remaining <= 32.5),
        ">32": remaining > 32.5,
    }
    signals = {
        "attack_8s": attack_8,
        "garbage_clear_8s": gclear_8,
        "vs_activity_8s": vs_activity_8,
        "received_8s_negative": -received_8,
        "net_pressure_8s": net_pressure_8,
        "terminal_balance_8s": terminal_balance_8,
    }

    print(f"dataset {len(ds)} / held-out {len(val_idx)}")
    for name, score in signals.items():
        parts = []
        for bucket, mask in buckets.items():
            auc, sep, count = metric(score, z, mask)
            parts.append(f"{bucket}:AUC={auc:.3f},sep={sep:+.3f},n={count}")
        print(f"{name:22s}  " + "  ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
