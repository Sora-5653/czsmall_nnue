#!/usr/bin/env python3
"""Rank clean no-attack-delivery player trajectories for stacking curricula.

APP is reconstructed from the placement attack auxiliary target (index 20 in
append-only schemas v2+).  Clear structure comes from the exact chosen action.
The output is deliberately per player rather than per game: one side can be a
useful firepower teacher even when the paired trajectory is weak or short.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tetra_dataset
from analyze_clear_patterns import decode_action


@dataclass
class Row:
    seed: int
    player: int
    samples: int
    attack: float
    app: float
    quads: int
    tspin_clears: int
    full_tspins: int
    mini_tspins: int
    b2b_continuations: int
    max_b2b: int
    max_combo: int
    difficult: int
    last_move: int


def unsquash_attack(y: np.ndarray) -> np.ndarray:
    y = np.clip(y.astype(np.float64), 0.0, 1.0 - 1e-9)
    return 8.0 * y / (1.0 - y)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--min-app", type=float, default=0.0)
    ap.add_argument("--min-samples", type=int, default=0)
    args = ap.parse_args()

    loaded = [tetra_dataset.load(path) for path in args.datasets]
    ds = tetra_dataset.Dataset.concatenate(loaded)
    ds.sanity_check()
    if ds.header.aux_targets <= 20:
        raise SystemExit("placement attack auxiliary target 20 is unavailable")

    attack_valid = ds.aux_valid_mask[:, 20] > 0.5
    attack = unsquash_attack(ds.aux_target[:, 20])
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in range(len(ds)):
        seed = int(ds.game_seed[i])
        player = int(ds.player_perspective[i])
        if seed != 0:
            groups[(seed, player)].append(i)

    rows: list[Row] = []
    for (seed, player), indices in sorted(groups.items()):
        indices.sort(key=lambda i: int(ds.move_number[i]))
        valid_indices = [i for i in indices if attack_valid[i]]
        total_attack = float(attack[valid_indices].sum()) if valid_indices else 0.0
        denom = len(valid_indices)
        app = total_attack / denom if denom else 0.0

        quads = tspin_clears = full_tspins = mini_tspins = 0
        b2b = max_b2b = b2b_continuations = difficult = 0
        combo = -1
        max_combo = 0
        for i in indices:
            chosen = int(ds.chosen_action[i])
            if chosen < 0 or chosen >= ds.actions.shape[1] or ds.action_mask[i, chosen] < 0.5:
                continue
            piece, _rot, _x, _y, lines, spin, _ac = decode_action(ds.actions[i, chosen])
            is_t = piece == "T" and spin != "none"
            is_difficult = lines >= 4 or (is_t and lines > 0)
            if lines >= 4 and spin == "none":
                quads += 1
            if is_t:
                if spin == "full":
                    full_tspins += 1
                else:
                    mini_tspins += 1
                if lines > 0:
                    tspin_clears += 1
            if lines > 0:
                combo += 1
                max_combo = max(max_combo, combo)
                if is_difficult:
                    difficult += 1
                    if b2b > 0:
                        b2b_continuations += 1
                    b2b += 1
                    max_b2b = max(max_b2b, b2b)
                else:
                    b2b = 0
            else:
                combo = -1
        rows.append(Row(
            seed=seed,
            player=player,
            samples=denom,
            attack=total_attack,
            app=app,
            quads=quads,
            tspin_clears=tspin_clears,
            full_tspins=full_tspins,
            mini_tspins=mini_tspins,
            b2b_continuations=b2b_continuations,
            max_b2b=max_b2b,
            max_combo=max_combo,
            difficult=difficult,
            last_move=max((int(ds.move_number[i]) for i in indices), default=0),
        ))

    rows.sort(key=lambda r: (r.app, r.quads, r.b2b_continuations, r.samples), reverse=True)
    shown = [r for r in rows if r.app >= args.min_app and r.samples >= args.min_samples]
    print(f"files={len(args.datasets)} samples={len(ds)} trajectories={len(rows)} shown={len(shown)}")
    print("seed      pl  samples  attack    APP  Quad  Tclr full mini B2Bc maxB2 diff last")
    for r in shown:
        print(
            f"{r.seed:8d} {r.player:+3d} {r.samples:8d} {r.attack:7.1f} {r.app:6.3f} "
            f"{r.quads:5d} {r.tspin_clears:5d} {r.full_tspins:4d} {r.mini_tspins:4d} "
            f"{r.b2b_continuations:4d} {r.max_b2b:4d} {r.difficult:4d} {r.last_move:4d}"
        )

    if rows:
        apps = np.asarray([r.app for r in rows], dtype=np.float64)
        print(
            "\nAPP distribution "
            f"mean={apps.mean():.3f} median={np.median(apps):.3f} "
            f"q75={np.quantile(apps, 0.75):.3f} q90={np.quantile(apps, 0.9):.3f} "
            f"max={apps.max():.3f}"
        )
        for threshold in (0.10, 0.15, 0.20, 0.25, 0.30):
            selected = [r for r in rows if r.app >= threshold]
            print(
                f"APP>={threshold:.2f}: {len(selected):2d} trajectories, "
                f"samples={sum(r.samples for r in selected):5d}, "
                f"Quad={sum(r.quads for r in selected):3d}, "
                f"Tclear={sum(r.tspin_clears for r in selected):2d}, "
                f"B2Bcont={sum(r.b2b_continuations for r in selected):3d}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
