# SPDX-License-Identifier: MIT
"""Compare TetraFormer size against search budget under matched Arena rules.

Two comparisons are produced for each XS/M candidate against the S reference:

* same_sims: both sides receive the same simulation count, isolating network size;
* equal_inference: candidate simulations are scaled by measured fp16 throughput so
  each side receives approximately the same neural-inference time budget.

The second mode is deliberately called equal *inference* time, not exact equal
wall-clock time: tree bookkeeping and engine work are also measured and written to
the result so the approximation can be audited.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from gpu_arena import evaluate
from gpu_match import load_model


STAT_KEYS = (
    "candidate_vs",
    "champion_vs",
    "candidate_apm",
    "champion_apm",
    "candidate_app",
    "champion_app",
    "candidate_pps",
    "champion_pps",
    "candidate_avg_pieces",
    "champion_avg_pieces",
    "candidate_avg_seconds",
    "champion_avg_seconds",
    "candidate_survival_rate",
    "champion_survival_rate",
    "candidate_sent_per_game",
    "champion_sent_per_game",
    "candidate_garbage_cleared_per_game",
    "champion_garbage_cleared_per_game",
    "candidate_received_per_game",
    "champion_received_per_game",
    "candidate_blockout_rate",
    "champion_blockout_rate",
    "candidate_lockout_rate",
    "champion_lockout_rate",
    "candidate_garbageout_rate",
    "champion_garbageout_rate",
)


def wilson(score: float, games: int, z: float = 1.96) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 1.0
    p = score / games
    zz = z * z
    den = 1.0 + zz / games
    center = (p + zz / (2.0 * games)) / den
    half = z * math.sqrt(p * (1.0 - p) / games + zz / (4.0 * games * games)) / den
    return max(0.0, center - half), min(1.0, center + half)


def benchmark(
    path: str,
    device: torch.device,
    *,
    batch: int,
    tokens: int,
    actions: int,
    features: int,
    warmup: int,
    reps: int,
) -> dict[str, float]:
    model = load_model(path, device)
    token_x = torch.rand(batch, tokens, features, device=device)
    token_mask = torch.ones(batch, tokens, device=device)
    action_x = torch.rand(batch, actions, features, device=device)
    action_mask = torch.ones(batch, actions, device=device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(warmup):
            model(token_x, token_mask, action_x, action_mask)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(reps):
            model(token_x, token_mask, action_x, action_mask)
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    params = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "parameters": params,
        "ms_per_batch": 1000.0 * elapsed / reps,
        "positions_per_second": reps * batch / elapsed,
    }
    del model, token_x, token_mask, action_x, action_mask
    torch.cuda.empty_cache()
    return result


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    games = sum(int(row["games_played"]) for row in rows)
    wins = sum(int(row["candidate_wins"]) for row in rows)
    losses = sum(int(row["champion_wins"]) for row in rows)
    draws = sum(int(row["draws"]) for row in rows)
    score = wins + 0.5 * draws
    lower, upper = wilson(score, games)
    weighted: dict[str, float] = {}
    for key in STAT_KEYS:
        weighted[key] = (
            sum(float(row[key]) * int(row["games_played"]) for row in rows) / games
            if games
            else 0.0
        )
    return {
        "games": games,
        "candidate_wins": wins,
        "champion_wins": losses,
        "draws": draws,
        "score_rate": score / games if games else 0.0,
        "wilson_lower": lower,
        "wilson_upper": upper,
        "gpu_inference_seconds": sum(float(row["gpu_inference_seconds"]) for row in rows),
        "wall_seconds": sum(float(row["wall_seconds"]) for row in rows),
        **weighted,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xs", required=True)
    ap.add_argument("--m", required=True)
    ap.add_argument("--s", required=True)
    ap.add_argument("--base-sims", nargs="+", type=int, default=(8, 16, 32))
    ap.add_argument("--candidate-sizes", nargs="+", choices=("xs", "m"), default=("xs", "m"))
    ap.add_argument(
        "--modes",
        nargs="+",
        choices=("same_sims", "equal_inference"),
        default=("same_sims", "equal_inference"),
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=(42, 1337))
    ap.add_argument("--pairs", type=int, default=1)
    ap.add_argument("--pieces", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--determinizations", type=int, default=1)
    ap.add_argument("--precision", choices=("fp16", "fp32", "bf16"), default="fp16")
    ap.add_argument("--search-mode", choices=("gumbel", "puct"), default="gumbel")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--benchmark-warmup", type=int, default=20)
    ap.add_argument("--benchmark-reps", type=int, default=200)
    ap.add_argument("--equal-scale-xs", type=float, default=0.0,
                    help="override XS/S sim scale for equal_inference; 0 uses forward throughput")
    ap.add_argument("--equal-scale-m", type=float, default=0.0,
                    help="override M/S sim scale for equal_inference; 0 uses forward throughput")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("GPU requested but torch.cuda.is_available() is false")
    if not Path(args.engine).exists():
        raise SystemExit(f"engine not found: {args.engine}")
    for sims in args.base_sims:
        if sims < 0:
            raise SystemExit("--base-sims entries must be non-negative")
        if sims == 0 and "equal_inference" in args.modes:
            raise SystemExit("base sims 0 is policy-only and supports --modes same_sims only")

    device = torch.device(args.device)
    paths = {"xs": args.xs, "m": args.m, "s": args.s}
    throughput: dict[str, dict[str, float]] = {}
    for size, path in paths.items():
        throughput[size] = benchmark(
            path,
            device,
            batch=args.batch,
            tokens=102,
            actions=106,
            features=24,
            warmup=args.benchmark_warmup,
            reps=args.benchmark_reps,
        )
        row = throughput[size]
        print(
            f"bench {size:>2}: {int(row['parameters']):,} params, "
            f"{row['ms_per_batch']:.3f} ms/batch, "
            f"{row['positions_per_second']:.1f} pos/s",
            flush=True,
        )

    s_rate = throughput["s"]["positions_per_second"]
    ratios = {
        size: throughput[size]["positions_per_second"] / s_rate for size in ("xs", "m", "s")
    }
    sim_scales = dict(ratios)
    if args.equal_scale_xs > 0.0:
        sim_scales["xs"] = args.equal_scale_xs
    if args.equal_scale_m > 0.0:
        sim_scales["m"] = args.equal_scale_m
    budgets: dict[str, dict[str, int]] = {}
    for base in args.base_sims:
        budgets[str(base)] = {
            size: max(1, int(round(base * sim_scales[size]))) for size in ("xs", "m", "s")
        }
        budgets[str(base)]["s"] = base

    output: dict[str, object] = {
        "checkpoints": paths,
        "protocol": {
            "search_mode": args.search_mode,
            "gumbel": args.search_mode == "gumbel",
            "gumbel_c_scale": 0.01,
            "gumbel_noise_scale": 0.05,
            "timing_actions": False,
            "determinizations": args.determinizations,
            "policy_temperature": 1.0,
            "precision": args.precision,
            "pieces": args.pieces,
            "pairs_per_seed": args.pairs,
            "seeds": args.seeds,
        },
        "throughput": throughput,
        "throughput_ratio_vs_s": ratios,
        "equal_inference_sim_scale_vs_s": sim_scales,
        "equal_inference_scale_source": {
            "xs": "override" if args.equal_scale_xs > 0.0 else "forward_throughput",
            "m": "override" if args.equal_scale_m > 0.0 else "forward_throughput",
            "s": "reference",
        },
        "equal_inference_budgets": budgets,
        "conditions": [],
    }

    s_model = load_model(args.s, device)
    try:
        for base in args.base_sims:
            for mode in args.modes:
                for size in args.candidate_sizes:
                    candidate_sims = base if mode == "same_sims" else budgets[str(base)][size]
                    champion_sims = base
                    candidate = load_model(paths[size], device)
                    rows: list[dict[str, object]] = []
                    try:
                        print(
                            f"\n{mode} base={base}: {size}({candidate_sims}) vs s({champion_sims})",
                            flush=True,
                        )
                        for seed in args.seeds:
                            wall_start = time.perf_counter()
                            result, infer_seconds = evaluate(
                                candidate,
                                s_model,
                                device,
                                args.engine,
                                max(1, args.pairs),
                                max(1, base),
                                max(1, args.pieces),
                                max(1, args.batch),
                                max(1, args.determinizations),
                                args.search_mode == "gumbel",
                                args.precision,
                                max(0, seed),
                                candidate_sims,
                                champion_sims,
                                -1,
                                -1,
                                0.01,
                                0.05,
                                -1,
                                -1,
                            )
                            wall_seconds = time.perf_counter() - wall_start
                            row = {key: value for key, value in result.items() if key != "promoted"}
                            row.update(
                                {
                                    "seed": seed,
                                    "gpu_inference_seconds": infer_seconds,
                                    "wall_seconds": wall_seconds,
                                }
                            )
                            rows.append(row)
                            print(
                                f"  seed {seed}: {int(result['candidate_wins'])}-"
                                f"{int(result['champion_wins'])}-{int(result['draws'])}, "
                                f"score {100.0 * float(result['win_rate']):.1f}%, "
                                f"wall {wall_seconds:.2f}s",
                                flush=True,
                            )
                    finally:
                        del candidate
                        torch.cuda.empty_cache()
                    summary = aggregate(rows)
                    condition = {
                        "mode": mode,
                        "base_sims": base,
                        "candidate_size": size,
                        "candidate_sims": candidate_sims,
                        "champion_size": "s",
                        "champion_sims": champion_sims,
                        "trials": rows,
                        "summary": summary,
                    }
                    output["conditions"].append(condition)
                    print(
                        f"  aggregate {summary['candidate_wins']}-{summary['champion_wins']}-"
                        f"{summary['draws']} / {summary['games']} games; "
                        f"score {100.0 * float(summary['score_rate']):.1f}% "
                        f"CI {100.0 * float(summary['wilson_lower']):.1f}-"
                        f"{100.0 * float(summary['wilson_upper']):.1f}%",
                        flush=True,
                    )
    finally:
        del s_model
        torch.cuda.empty_cache()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nresults       {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
