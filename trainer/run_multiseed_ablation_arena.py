# SPDX-License-Identifier: MIT
"""Run ablation Arena over many widely separated base seeds and aggregate results."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from ablation_models import load_ablation_checkpoint
from gpu_arena import evaluate


def wilson(wins: int, games: int, z: float = 1.96) -> tuple[float, float, float]:
    if games <= 0:
        return 0.0, 0.0, 1.0
    p = wins / games
    zz = z * z
    den = 1.0 + zz / games
    center = (p + zz / (2.0 * games)) / den
    half = z * math.sqrt(p * (1.0 - p) / games + zz / (4.0 * games * games)) / den
    return p, max(0.0, center - half), min(1.0, center + half)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidate")
    ap.add_argument("champion")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--pairs", type=int, default=1)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--pieces", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--determinizations", type=int, default=1)
    ap.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--engine", default="")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("GPU requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    candidate = load_ablation_checkpoint(args.candidate, device)
    champion = load_ablation_checkpoint(args.champion, device)
    root = Path(__file__).resolve().parents[1]
    engine = args.engine or str(root / ("build/tetra_cli.exe" if os.name == "nt" else "build/tetra_cli"))
    if not os.path.exists(engine):
        raise SystemExit(f"engine not found: {engine}; run make tools first")

    trials: list[dict[str, int | float]] = []
    wins = losses = draws = games = 0
    infer_seconds = 0.0
    search_sims = max(0, args.sims)
    policy_only = search_sims == 0
    for seed in args.seeds:
        result, elapsed = evaluate(
            candidate, champion, device, engine,
            max(1, args.pairs), max(1, search_sims), max(1, args.pieces), max(1, args.batch),
            max(1, args.determinizations), False, args.precision, max(0, seed),
            0 if policy_only else -1,
            0 if policy_only else -1,
        )
        row = {
            "seed": seed,
            "games": int(result["games_played"]),
            "candidate_wins": int(result["candidate_wins"]),
            "champion_wins": int(result["champion_wins"]),
            "draws": int(result["draws"]),
            "win_rate": float(result["win_rate"]),
            "inference_seconds": elapsed,
        }
        trials.append(row)
        wins += row["candidate_wins"]
        losses += row["champion_wins"]
        draws += row["draws"]
        games += row["games"]
        infer_seconds += elapsed
        print(
            f"seed {seed:10d}: {row['candidate_wins']}-{row['champion_wins']}"
            f"-{row['draws']}  {100.0 * row['win_rate']:.1f}%",
            flush=True,
        )

    # Arena draws count as half a point for aggregate score rate.
    score = wins + 0.5 * draws
    rate, lower, upper = wilson(round(score), games)
    summary = {
        "candidate": args.candidate,
        "champion": args.champion,
        "pairs_per_seed": args.pairs,
        "sims": args.sims,
        "pieces": args.pieces,
        "precision": args.precision,
        "seeds": args.seeds,
        "games": games,
        "candidate_wins": wins,
        "champion_wins": losses,
        "draws": draws,
        "score_rate": score / games if games else 0.0,
        "wilson_lower": lower,
        "wilson_upper": upper,
        "inference_seconds": infer_seconds,
        "trials": trials,
    }
    print(
        f"AGGREGATE: {wins}-{losses}-{draws} over {games} games, "
        f"score {100.0 * summary['score_rate']:.1f}% "
        f"(Wilson 95% {100.0 * lower:.1f}-{100.0 * upper:.1f}%)",
        flush=True,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"results       {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
