#!/usr/bin/env python3
"""Measure whether checkpoints recognize teacher Quad/T-spin placements.

The dataset's executed action is treated as the behavioral teacher only on states
where that chosen placement is a Quad or a T-spin clear.  For each checkpoint we
report the chosen placement's model rank/probability and top-1 agreement.  This is
a cheap localization diagnostic before spending GPU time on full search rollouts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tetra_dataset
from train import load_checkpoint_model


def classify_action(row: np.ndarray) -> str | None:
    piece = int(np.argmax(row[:7]))
    lines = int(round(float(row[14]) * 4.0))
    spin = int(np.clip(round(float(row[16]) * 2.0), 0, 2))
    if spin > 0 and piece == 5 and lines > 0:  # T is index 5 in IJLOSTZ
        return "tspin"
    if lines >= 4:
        return "quad"
    return None


def name(path: str) -> str:
    return Path(path).stem


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lookback", type=int, default=8,
                    help="include up to this many preceding placements as setup states")
    args = ap.parse_args()

    loaded = [tetra_dataset.load(path) for path in args.datasets]
    ds = tetra_dataset.Dataset.concatenate(loaded)
    ds.sanity_check()
    selected: dict[str, list[int]] = {"quad": [], "tspin": []}
    trajectories: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i in range(len(ds)):
        key = (int(ds.game_seed[i]), int(ds.player_perspective[i]))
        trajectories.setdefault(key, []).append((int(ds.move_number[i]), i))
        chosen = int(ds.chosen_action[i])
        if chosen < 0 or chosen >= ds.actions.shape[1] or ds.action_mask[i, chosen] < 0.5:
            continue
        kind = classify_action(ds.actions[i, chosen])
        if kind is not None:
            selected[kind].append(i)

    sample_position: dict[int, tuple[list[tuple[int, int]], int]] = {}
    for seq in trajectories.values():
        seq.sort()
        for pos, (_, sample_index) in enumerate(seq):
            sample_position[sample_index] = (seq, pos)

    lookback = max(0, args.lookback)
    for kind in ("quad", "tspin"):
        setup: set[int] = set()
        for event_index in selected[kind]:
            seq, pos = sample_position[event_index]
            for _, sample_index in seq[max(0, pos - lookback):pos]:
                setup.add(sample_index)
        selected[f"{kind}_pre{lookback}"] = sorted(setup)

    print(f"samples       {len(ds)}")
    print("special       " + ", ".join(f"{k}={len(v)}" for k, v in selected.items()))
    data = ds.torch()
    device = torch.device(args.device)

    for model_path in args.models:
        model = load_checkpoint_model(model_path, str(device))
        print(f"\n{name(model_path)}")
        for kind, indices in selected.items():
            if not indices:
                print(f"  {kind:6s} count=0")
                continue
            total = 0
            top1 = 0
            rank_sum = 0.0
            prob_sum = 0.0
            margin_sum = 0.0
            for offset in range(0, len(indices), max(1, args.batch)):
                chosen_indices = indices[offset:offset + max(1, args.batch)]
                idx = torch.as_tensor(chosen_indices, dtype=torch.long)
                batch = {k: v[idx].to(device, non_blocking=True) for k, v in data.items()}
                with torch.inference_mode():
                    logits, _, _ = model(
                        batch["tokens"], batch["token_mask"],
                        batch["actions"], batch["action_mask"],
                    )
                    chosen = batch["chosen_action"].long()
                    legal = batch["action_mask"] > 0.5
                    chosen_logit = logits.gather(1, chosen[:, None]).squeeze(1)
                    rank = 1 + ((logits > chosen_logit[:, None]) & legal).sum(dim=1)
                    probs = torch.softmax(logits.float(), dim=-1)
                    chosen_prob = probs.gather(1, chosen[:, None]).squeeze(1)
                    masked_other = logits.masked_fill(~legal, float("-inf")).clone()
                    masked_other.scatter_(1, chosen[:, None], float("-inf"))
                    best_other = masked_other.max(dim=1).values
                    margin = chosen_logit - best_other
                    top1 += int((rank == 1).sum().item())
                    rank_sum += float(rank.float().sum().item())
                    prob_sum += float(chosen_prob.sum().item())
                    margin_sum += float(margin.float().sum().item())
                    total += len(chosen_indices)
            print(
                f"  {kind:6s} count={total} top1={top1/max(1,total):.3f} "
                f"mean_rank={rank_sum/max(1,total):.3f} "
                f"mean_prob={prob_sum/max(1,total):.4f} "
                f"margin={margin_sum/max(1,total):+.4f}"
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
