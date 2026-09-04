#!/usr/bin/env python3
"""Inspect chosen placements for Quad / T-spin / B2B structure.

This is a diagnostic for exported rectangular `.tetradat` self-play files. It
uses the chosen action embedding (the exact action search executed) and decodes
the pre-placement self board from the tokenizer's row tokens. No model
inference is performed here.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tetra_dataset

PIECES = "IJLOSTZ"
ROTS = ("N", "R", "2", "L")
SPINS = ("none", "mini", "full")


@dataclass
class Event:
    sample: int
    seed: int
    player: int
    move: int
    piece: str
    rot: str
    x: int
    y: int
    lines: int
    spin: str
    all_clear: bool
    board: list[str]

    @property
    def kind(self) -> str:
        if self.spin != "none" and self.piece == "T":
            suffix = {0: "0", 1: "S", 2: "D", 3: "T"}.get(self.lines, str(self.lines))
            prefix = "TSM" if self.spin == "mini" else "TS"
            return prefix + suffix
        if self.lines >= 4:
            return "Quad"
        return {0: "none", 1: "Single", 2: "Double", 3: "Triple"}.get(self.lines, str(self.lines))


def decode_action(row: np.ndarray) -> tuple[str, str, int, int, int, str, bool]:
    piece_idx = int(np.argmax(row[:7]))
    rot_idx = int(np.argmax(row[7:11]))
    piece = PIECES[piece_idx]
    rot = ROTS[rot_idx]
    # League board is 10x20 visible; embeddings are normalized by these values.
    x = int(round(float(row[12]) * 10.0))
    y = int(round(float(row[13]) * 20.0))
    lines = int(round(float(row[14]) * 4.0))
    spin_idx = int(np.clip(round(float(row[16]) * 2.0), 0, 2))
    return piece, rot, x, y, lines, SPINS[spin_idx], bool(row[18] > 0.5)


def decode_board(tokens: np.ndarray, token_mask: np.ndarray) -> list[str]:
    """Decode self row tokens. Tokenizer order starts with 24 self rows."""
    rows: list[str] = []
    for y in range(min(24, tokens.shape[0])):
        if token_mask[y] < 0.5:
            break
        bits = tokens[y, 8:18]
        rows.append("".join("#" if v > 0.5 else "." for v in bits))
    return rows


def board_text(rows: list[str], event_y: int, max_rows: int = 20) -> str:
    if not rows:
        return "  <board unavailable>"
    occupied = [i for i, row in enumerate(rows[:max_rows]) if "#" in row]
    top = max(occupied, default=0)
    top = min(max_rows - 1, max(top + 2, min(max_rows - 1, event_y + 3)))
    out = []
    for y in range(top, -1, -1):
        out.append(f"  {y:02d} |{rows[y]}|")
    out.append("     +----------+")
    out.append("      0123456789")
    return "\n".join(out)


def analyze(paths: list[str], label: str, examples: int) -> None:
    datasets = [tetra_dataset.load(p) for p in paths]
    ds = tetra_dataset.Dataset.concatenate(datasets)
    ds.sanity_check()

    events: list[Event] = []
    trajectories: dict[tuple[int, int], list[Event]] = defaultdict(list)
    chosen_total = 0
    invalid_chosen = 0

    for i in range(len(ds)):
        chosen = int(ds.chosen_action[i])
        if chosen < 0:
            continue
        if chosen >= ds.actions.shape[1] or ds.action_mask[i, chosen] < 0.5:
            invalid_chosen += 1
            continue
        chosen_total += 1
        piece, rot, x, y, lines, spin, all_clear = decode_action(ds.actions[i, chosen])
        ev = Event(
            sample=i,
            seed=int(ds.game_seed[i]),
            player=int(ds.player_perspective[i]),
            move=int(ds.move_number[i]),
            piece=piece,
            rot=rot,
            x=x,
            y=y,
            lines=lines,
            spin=spin,
            all_clear=all_clear,
            board=decode_board(ds.tokens[i], ds.token_mask[i]),
        )
        events.append(ev)
        trajectories[(ev.seed, ev.player)].append(ev)

    counts = Counter(ev.kind for ev in events)
    full_t = [ev for ev in events if ev.piece == "T" and ev.spin == "full"]
    mini_t = [ev for ev in events if ev.piece == "T" and ev.spin == "mini"]
    quad = [ev for ev in events if ev.lines >= 4 and ev.spin == "none"]
    attack_clears = [ev for ev in events if ev.lines >= 4 or (ev.spin != "none" and ev.lines > 0)]

    max_b2b = 0
    max_b2b_sequence: list[Event] = []
    b2b_continuations = 0
    max_combo = 0
    difficult_by_traj: list[int] = []
    for seq in trajectories.values():
        seq.sort(key=lambda e: e.move)
        streak = 0
        streak_events: list[Event] = []
        combo = -1
        difficult = 0
        for ev in seq:
            if ev.lines > 0:
                combo += 1
                max_combo = max(max_combo, combo)
                hard = ev.lines >= 4 or ev.spin != "none"
                if hard:
                    difficult += 1
                    if streak > 0:
                        b2b_continuations += 1
                    streak += 1
                    streak_events.append(ev)
                    if streak > max_b2b:
                        max_b2b = streak
                        max_b2b_sequence = list(streak_events)
                else:
                    streak = 0
                    streak_events = []
            else:
                combo = -1
        difficult_by_traj.append(difficult)

    print(f"=== {label} ===")
    print(f"files={len(paths)} samples={len(ds)} chosen={chosen_total} invalid_chosen={invalid_chosen}")
    print(f"trajectories={len(trajectories)}")
    print("clear counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"Quad={len(quad)} full_T_spin_actions={len(full_t)} mini_T_spin_actions={len(mini_t)}")
    if quad:
        quad_shapes = Counter(f"{ev.piece}{ev.rot}" for ev in quad)
        print("Quad placements: " + ", ".join(f"{k}={v}" for k, v in quad_shapes.most_common()))
    t_clear_counts = Counter(ev.kind for ev in events if ev.kind.startswith("TS") and ev.lines > 0)
    if t_clear_counts:
        print("T-spin clears: " + ", ".join(f"{k}={v}" for k, v in sorted(t_clear_counts.items())))
    print(f"difficult_attack_clears={len(attack_clears)} max_B2B_clear_streak={max_b2b} B2B_continuations={b2b_continuations} max_combo={max_combo}")
    if max_b2b_sequence:
        print("max B2B sequence: " + " -> ".join(
            f"m{ev.move}:{ev.kind}:{ev.piece}{ev.rot}@{ev.x},{ev.y}" for ev in max_b2b_sequence
        ))
    if difficult_by_traj:
        print(f"difficult clears/trajectory mean={np.mean(difficult_by_traj):.2f} max={max(difficult_by_traj)}")

    history_by_event: dict[tuple[int, int, int], list[Event]] = {}
    for seq in trajectories.values():
        for j, ev in enumerate(seq):
            history_by_event[(ev.seed, ev.player, ev.move)] = seq[max(0, j - 8):j]

    interesting = [ev for ev in events if ev.kind.startswith("TS") or ev.kind == "Quad"]
    interesting.sort(key=lambda e: (0 if ev_kind_priority(e.kind) == 0 else 1, e.seed, e.player, e.move))
    print(f"\n--- visual examples ({min(examples, len(interesting))}/{len(interesting)}) ---")
    for ev in interesting[:examples]:
        print(
            f"\n{ev.kind}: seed={ev.seed} player={ev.player:+d} move={ev.move} "
            f"piece={ev.piece} rot={ev.rot} x={ev.x} y={ev.y} lines={ev.lines} spin={ev.spin} AC={int(ev.all_clear)}"
        )
        history = history_by_event.get((ev.seed, ev.player, ev.move), [])
        if history:
            print("  preceding: " + " | ".join(
                f"m{h.move}:{h.piece}{h.rot}@{h.x},{h.y}:{h.kind}" for h in history
            ))
        print(board_text(ev.board, ev.y))


def ev_kind_priority(kind: str) -> int:
    return 0 if kind.startswith("TS") else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--label", default="dataset")
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()
    analyze(args.datasets, args.label, max(0, args.examples))


if __name__ == "__main__":
    main()
