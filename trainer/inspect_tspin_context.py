#!/usr/bin/env python3
"""Print compact board evolution around full T-spin clears in a tetradat file."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tetra_dataset

PIECES = "IJLOSTZ"
ROTS = ("N", "R", "2", "L")


def action_desc(a: np.ndarray) -> tuple[str, int, str, int, int, int, int]:
    piece = PIECES[int(np.argmax(a[:7]))]
    rot = ROTS[int(np.argmax(a[7:11]))]
    x = int(round(float(a[12]) * 10))
    y = int(round(float(a[13]) * 20))
    lines = int(round(float(a[14]) * 4))
    spin = int(round(float(a[16]) * 2))
    return piece, spin, rot, x, y, lines, int(a[18] > 0.5)


def rows(ds, i: int, n: int = 6) -> list[str]:
    out = []
    for y in range(n - 1, -1, -1):
        bits = ds.tokens[i, y, 8:18]
        out.append(f"{y:02d}|" + "".join("#" if v > 0.5 else "." for v in bits) + "|")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--before", type=int, default=10)
    args = ap.parse_args()
    ds = tetra_dataset.load(args.dataset)
    ds.sanity_check()

    for target in range(len(ds)):
        c = int(ds.chosen_action[target])
        if c < 0 or ds.action_mask[target, c] < 0.5:
            continue
        piece, spin, rot, x, y, lines, ac = action_desc(ds.actions[target, c])
        if not (piece == "T" and spin == 2 and lines > 0):
            continue
        seed = int(ds.game_seed[target])
        player = int(ds.player_perspective[target])
        move = int(ds.move_number[target])
        same = [i for i in range(target + 1)
                if int(ds.game_seed[i]) == seed and int(ds.player_perspective[i]) == player]
        pos = same.index(target)
        same = same[max(0, pos - args.before):pos + 1]
        print(f"FULL T-SPIN CLEAR seed={seed} player={player:+d} target_move={move}")
        for i in same:
            ci = int(ds.chosen_action[i])
            if ci < 0 or ds.action_mask[i, ci] < 0.5:
                continue
            p, s, r, xx, yy, ll, aa = action_desc(ds.actions[i, ci])
            mark = "  <<< T-SPIN" if i == target else ""
            print(f"\nmove={int(ds.move_number[i])} {p}{r}@{xx},{yy} lines={ll} spin={s} AC={aa}{mark}")
            print("\n".join(rows(ds, i)))


if __name__ == "__main__":
    main()
