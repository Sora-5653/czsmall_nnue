#!/usr/bin/env python3
"""Aggregate the diagnostic Arena rows and Python bridge traces.

This script is intentionally descriptive.  It never interprets win rate and
does not alter checkpoints or the production experiment manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    fraction = rank - lo
    return values[lo] + (values[hi] - values[lo]) * fraction


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def trace_stats(rows: list[dict[str, Any]], model_id: int) -> dict[str, Any]:
    selected = [row for row in rows if int(row.get("model_id", -1)) == model_id]
    batches = [int(row.get("served_batch", row.get("batch", 0))) for row in selected]
    positions = sum(batches)

    def values(key: str) -> list[float]:
        return [float(row.get(key, 0.0)) for row in selected]

    forward = [
        float(row.get("model_forward_us", 0.0))
        + float(row.get("device_sync_us", 0.0))
        for row in selected
    ]
    service = values("total_us")
    per_position_forward = [
        forward[i] / batches[i] for i in range(len(selected)) if batches[i] > 0
    ]
    per_position_service = [
        service[i] / batches[i] for i in range(len(selected)) if batches[i] > 0
    ]
    histogram: dict[str, int] = {}
    for batch in batches:
        histogram[str(batch)] = histogram.get(str(batch), 0) + 1
    return {
        "requests": len(selected),
        "positions": positions,
        "batch_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        "request_fraction": len(selected) / len(rows) if rows else 0.0,
        "position_fraction": positions / sum(
            int(row.get("served_batch", row.get("batch", 0))) for row in rows
        ) if rows else 0.0,
        "mean_batch": statistics.fmean(batches) if batches else 0.0,
        "forward_us_mean": statistics.fmean(forward) if forward else 0.0,
        "forward_us_p95": percentile(forward, 0.95),
        "forward_us_per_position_mean": (
            statistics.fmean(per_position_forward) if per_position_forward else 0.0
        ),
        "forward_us_per_position_p95": percentile(per_position_forward, 0.95),
        "service_us_mean": statistics.fmean(service) if service else 0.0,
        "service_us_p95": percentile(service, 0.95),
        "service_us_per_position_mean": (
            statistics.fmean(per_position_service) if per_position_service else 0.0
        ),
        "service_us_per_position_p95": percentile(per_position_service, 0.95),
        "first_shape_count": sum(1 for row in selected if not row.get("shape_seen_before", True)),
        "first_shape_forward_us": [
            forward[i] for i, row in enumerate(selected)
            if not row.get("shape_seen_before", True)
        ],
    }


def side_timing(row: dict[str, Any], side_name: str) -> dict[str, Any]:
    side = row.get("diagnostics", {}).get("arena_diagnostics", {}).get(side_name, {})
    elapsed_ms = float(side.get("elapsed_ms", 0.0))
    evaluator_ms = float(side.get("evaluator_elapsed_ms", 0.0))
    total_us = elapsed_ms * 1000.0
    known = {
        "root_setup_us": float(side.get("root_setup_us", 0.0)),
        "gather_us": float(side.get("gather_us", 0.0)),
        "evaluator_us": evaluator_ms * 1000.0,
        "backup_us": float(side.get("backup_us", 0.0)),
        "finalize_us": float(side.get("finalize_us", 0.0)),
    }
    accounted = sum(known.values())
    result = {
        "decisions": int(side.get("decisions", 0)),
        "nodes": int(side.get("nodes", 0)),
        "positions_evaluated": int(side.get("positions_evaluated", 0)),
        "evaluator_calls": int(side.get("evaluator_calls", 0)),
        "elapsed_ms": elapsed_ms,
        "evaluator_elapsed_ms": evaluator_ms,
        "nodes_per_decision": (
            int(side.get("nodes", 0)) / int(side.get("decisions", 1))
            if int(side.get("decisions", 0)) else 0.0
        ),
        "nodes_per_ms": (
            int(side.get("nodes", 0)) / elapsed_ms if elapsed_ms > 0 else 0.0
        ),
        "overshoot_ms": float(side.get("overshoot_ms", 0.0)),
        "max_depth": int(side.get("max_depth", 0)),
        "timing_us": known,
        "unaccounted_us": total_us - accounted,
        "timing_percent": {
            key.removesuffix("_us"): (value / total_us * 100.0 if total_us else 0.0)
            for key, value in known.items()
        },
        "cpu_search_percent": (
            max(0.0, total_us - evaluator_ms * 1000.0) / total_us * 100.0
            if total_us else 0.0
        ),
    }
    return result


def analyze(result_path: Path) -> list[dict[str, Any]]:
    row = json.loads(result_path.read_text(encoding="utf-8"))
    trace_path = Path(row.get("trace_path", ""))
    if not trace_path.is_absolute():
        trace_path = ROOT / trace_path
    traces = read_jsonl(trace_path)
    out: list[dict[str, Any]] = []
    for side_name, model_id, model_name in (
        ("candidate", 0, "B"),
        ("champion", 1, "A"),
    ):
        timing = side_timing(row, side_name)
        bridge = trace_stats(traces, model_id)
        out.append({
            "result": str(result_path),
            "condition_id": row.get("condition_id", result_path.stem),
            "environment": row.get("environment", ""),
            "budget_ms": row.get("budget_ms", 0.0),
            "side": side_name,
            "model": model_name,
            "diagnostic_evaluator": traces[0].get("diagnostic_evaluator", "model") if traces else "unknown",
            "timing": timing,
            "bridge": bridge,
        })
    return out


def flatten(entry: dict[str, Any]) -> dict[str, Any]:
    timing = entry["timing"]
    bridge = entry["bridge"]
    return {
        "condition_id": entry["condition_id"],
        "environment": entry["environment"],
        "budget_ms": entry["budget_ms"],
        "side": entry["side"],
        "model": entry["model"],
        "diagnostic_evaluator": entry["diagnostic_evaluator"],
        "decisions": timing["decisions"],
        "nodes": timing["nodes"],
        "nodes_per_decision": timing["nodes_per_decision"],
        "nodes_per_ms": timing["nodes_per_ms"],
        "elapsed_ms": timing["elapsed_ms"],
        "evaluator_elapsed_ms": timing["evaluator_elapsed_ms"],
        "cpu_search_percent": timing["cpu_search_percent"],
        "root_setup_percent": timing["timing_percent"]["root_setup"],
        "gather_percent": timing["timing_percent"]["gather"],
        "evaluator_percent": timing["timing_percent"]["evaluator"],
        "backup_percent": timing["timing_percent"]["backup"],
        "finalize_percent": timing["timing_percent"]["finalize"],
        "unaccounted_us": timing["unaccounted_us"],
        "requests": bridge["requests"],
        "positions": bridge["positions"],
        "mean_batch": bridge["mean_batch"],
        "batch_histogram": json.dumps(bridge["batch_histogram"], sort_keys=True),
        "forward_us_mean": bridge["forward_us_mean"],
        "forward_us_p95": bridge["forward_us_p95"],
        "forward_us_per_position_mean": bridge["forward_us_per_position_mean"],
        "forward_us_per_position_p95": bridge["forward_us_per_position_p95"],
        "service_us_per_position_mean": bridge["service_us_per_position_mean"],
        "service_us_per_position_p95": bridge["service_us_per_position_p95"],
        "first_shape_count": bridge["first_shape_count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path,
                        help="diagnostic result JSON files")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    entries: list[dict[str, Any]] = []
    for path in args.results:
        entries.extend(analyze(path.resolve()))
    flat = [flatten(entry) for entry in entries]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]) if flat else ["condition_id"])
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"rows": len(flat), "output_json": str(args.output_json),
                      "output_csv": str(args.output_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
