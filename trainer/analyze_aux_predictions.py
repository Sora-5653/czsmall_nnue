# SPDX-License-Identifier: MIT
"""Measure whether predicted auxiliary heads provide useful search-value ranking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tetra_dataset
from analyze_value_ranking import binary_auc
from gpu_match import load_model
from train import split_indices_by_game


def score_metric(score: np.ndarray, z: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    mask = mask & (np.abs(z) > 0.5) & np.isfinite(score)
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
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    loaded = [tetra_dataset.load(path) for path in args.datasets]
    ds = tetra_dataset.Dataset.concatenate(loaded)
    ds.sanity_check()
    _, val_idx = split_indices_by_game(ds)
    data = ds.torch()
    z = ds.value_target[val_idx].astype(np.float64)
    remaining_norm = ds.aux_target[val_idx, 2].astype(np.float64)
    remaining = 64.0 * remaining_norm / np.clip(1.0 - remaining_norm, 1e-8, None)
    buckets = {
        "all": np.ones(len(val_idx), dtype=bool),
        "<=8": remaining <= 8.5,
        "9-32": (remaining > 8.5) & (remaining <= 32.5),
        ">32": remaining > 32.5,
    }
    device = torch.device(args.device)

    for model_path in args.models:
        model = load_model(model_path, device)
        model.eval()
        aux_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        with torch.no_grad():
            for offset in range(0, len(val_idx), max(1, args.batch)):
                idx_np = val_idx[offset:offset + max(1, args.batch)]
                idx = torch.as_tensor(idx_np, dtype=torch.long)
                batch = {key: value[idx].to(device, non_blocking=True) for key, value in data.items()}
                _, wdl, aux = model(
                    batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"]
                )
                prob = torch.softmax(wdl, dim=-1)
                value_parts.append((prob[:, 0] - prob[:, 2]).cpu().numpy())
                aux_parts.append(aux.float().cpu().numpy())
        aux = np.concatenate(aux_parts).astype(np.float64)
        wdl_scalar = np.concatenate(value_parts).astype(np.float64)

        self_raw = aux[:, [6, 10, 14, 18]]
        opp_raw = aux[:, [7, 11, 15, 19]]
        if bool(getattr(model, "topout_aux_logits", False)):
            self_raw = 1.0 / (1.0 + np.exp(-np.clip(self_raw, -40.0, 40.0)))
            opp_raw = 1.0 / (1.0 + np.exp(-np.clip(opp_raw, -40.0, 40.0)))
        else:
            self_raw = np.clip(self_raw, 0.0, 1.0)
            opp_raw = np.clip(opp_raw, 0.0, 1.0)
        self_topout = self_raw.sum(axis=1).clip(0.0, 1.0)
        opp_topout = opp_raw.sum(axis=1).clip(0.0, 1.0)
        terminal_balance = opp_topout - self_topout
        terminal_confidence = np.maximum(self_topout, opp_topout)
        attack_8 = aux[:, [4, 8, 12, 16]].sum(axis=1)
        received_8 = aux[:, [5, 9, 13, 17]].sum(axis=1)
        net_attack = attack_8 - received_8

        signals = {
            "wdl": wdl_scalar,
            "pred_terminal_balance_8s": terminal_balance,
            "pred_attack_minus_received_8s": net_attack,
        }
        if aux.shape[1] >= 44:
            gclear_8 = aux[:, 36:40].sum(axis=1)
            signals["pred_vs_minus_received_8s"] = attack_8 + gclear_8 - received_8

        print(f"\n{Path(model_path).name} aux={aux.shape[1]}")
        quantiles = np.quantile(terminal_confidence, [0.5, 0.75, 0.9, 0.95, 0.99])
        print(
            "topout confidence quantiles "
            + " ".join(f"q{q:g}={value:.3f}" for q, value in zip((50, 75, 90, 95, 99), quantiles))
        )
        decisive = np.abs(z) > 0.5
        for threshold in (0.2, 0.3, 0.5, 0.7):
            high = decisive & (terminal_confidence >= threshold)
            count = int(high.sum())
            if count:
                sign_accuracy = float(
                    ((terminal_balance[high] > 0.0) == (z[high] > 0.0)).mean()
                )
                auc = binary_auc(terminal_balance[high], (z[high] > 0.0).astype(np.float64))
            else:
                sign_accuracy = float("nan")
                auc = float("nan")
            print(
                f"  confidence>={threshold:.1f}: n={count} "
                f"sign_acc={sign_accuracy:.3f} AUC={auc:.3f}"
            )
        for name, score in signals.items():
            pieces = []
            for bucket, mask in buckets.items():
                auc, sep, count = score_metric(score, z, mask)
                pieces.append(f"{bucket}:AUC={auc:.3f},sep={sep:+.3f},n={count}")
            print(f"{name:31s}  " + "  ".join(pieces))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
