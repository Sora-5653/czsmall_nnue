#!/usr/bin/env python3
"""Summarize recorded Arena JSON rows without changing raw evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments" / "search_throughput_vs_capacity_2026-08-18"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * q) - 1))
    return ordered[index]


def summarize_row(path: Path) -> dict[str, object]:
    row = json.loads(path.read_text(encoding="utf-8"))
    diagnostics = row.get("diagnostics", {}).get("arena_diagnostics", {})
    output: dict[str, object] = {
        "file": str(path.relative_to(ROOT)),
        "condition_id": row.get("condition_id", path.stem),
        "variant": row.get("variant", ""),
        "environment": row.get("environment", ""),
        "budget_ms": row.get("budget_ms", 0),
        "seed": row.get("seed", 0),
        "games": row.get("games", 0),
        "wins": row.get("wins", 0),
        "losses": row.get("losses", 0),
        "draws": row.get("draws", 0),
        "win_rate": row.get("win_rate", 0.0),
        "ci95_low": row.get("ci95_low", 0.0),
        "ci95_high": row.get("ci95_high", 1.0),
        "wall_seconds": row.get("wall_seconds", 0.0),
        "warmup_seconds": row.get("warmup_seconds", 0.0),
    }
    for side_name in ("candidate", "champion"):
        side = diagnostics.get(side_name, {})
        prefix = "candidate" if side_name == "candidate" else "champion"
        decisions = max(1, int(side.get("decisions", 0)))
        depth_samples = max(1, int(side.get("depth_samples", 0)))
        latencies = [float(value) for value in side.get("decision_latencies_ms", [])]
        output[f"{prefix}_decisions"] = int(side.get("decisions", 0))
        output[f"{prefix}_simulations"] = int(side.get("simulations", 0))
        output[f"{prefix}_nodes_per_decision"] = float(side.get("nodes", 0)) / decisions
        output[f"{prefix}_positions_per_decision"] = float(side.get("positions_evaluated", 0)) / decisions
        output[f"{prefix}_eval_calls_per_decision"] = float(side.get("evaluator_calls", 0)) / decisions
        output[f"{prefix}_mean_depth"] = float(side.get("depth_sum", 0.0)) / depth_samples
        output[f"{prefix}_decision_p50_ms"] = statistics.median(latencies) if latencies else 0.0
        output[f"{prefix}_decision_p95_ms"] = percentile(latencies, 0.95)
        output[f"{prefix}_elapsed_per_decision_ms"] = float(side.get("elapsed_ms", 0.0)) / decisions
        output[f"{prefix}_evaluator_elapsed_per_decision_ms"] = float(side.get("evaluator_elapsed_ms", 0.0)) / decisions
        output[f"{prefix}_overshoot_p95_ms"] = float(side.get("overshoot_ms", 0.0)) / decisions
        output[f"{prefix}_time_budget_exhaustions"] = int(side.get("time_budget_exhaustions", 0))
        output[f"{prefix}_raw_policy_matches"] = int(side.get("raw_policy_matches", 0))
        output[f"{prefix}_searched_action_changes"] = int(side.get("searched_action_changes", 0))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(EXPERIMENT / "results" / "summary.json"))
    parser.add_argument("--csv-output", default=str(EXPERIMENT / "results" / "summary.csv"))
    parser.add_argument("inputs", nargs="*", help="Arena JSON files; defaults to short and control rows")
    args = parser.parse_args()

    if args.inputs:
        paths = [Path(value).resolve() for value in args.inputs]
    else:
        paths = sorted(EXPERIMENT.joinpath("results").glob("short-*.json"))
        paths += sorted(EXPERIMENT.joinpath("results").glob("A0-B0-*.json"))
        paths += sorted(EXPERIMENT.joinpath("results").glob("D-equal-*.json"))
        paths += sorted(EXPERIMENT.joinpath("results").glob("smoke-e3-10ms-pieces50-b16.json"))
        paths += sorted(EXPERIMENT.joinpath("results").glob("debug-warmup-five-piece.json"))
    paths = [path for path in paths if path.is_file()]
    rows = [summarize_row(path) for path in paths]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_output = Path(args.csv_output).resolve()
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "json": str(output), "csv": str(csv_output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
