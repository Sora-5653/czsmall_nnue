#!/usr/bin/env python3
"""Aggregate restartable paired pilot blocks without changing raw rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments" / "search_throughput_vs_capacity_2026-08-18"


def wilson(score: float, games: int) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    p = score / games
    denominator = 1.0 + z * z / games
    center = (p + z * z / (2.0 * games)) / denominator
    half = z * math.sqrt(
        p * (1.0 - p) / games + z * z / (4.0 * games * games)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def weighted(rows: list[dict[str, Any]], key: str, games: int) -> float:
    return sum(float(row["arena"].get(key, 0.0)) * int(row["games"]) for row in rows) / games


def aggregate_side(rows: list[dict[str, Any]], side_name: str) -> dict[str, Any]:
    side = "candidate" if side_name == "candidate" else "champion"
    decisions = 0
    nodes = 0
    positions = 0
    evaluator_calls = 0
    elapsed_ms = 0.0
    evaluator_ms = 0.0
    histogram: dict[str, int] = {}
    for row in rows:
        diagnostics = row.get("diagnostics", {}).get("arena_diagnostics", {}).get(side, {})
        decisions += int(diagnostics.get("decisions", 0))
        nodes += int(diagnostics.get("nodes", 0))
        positions += int(diagnostics.get("positions_evaluated", 0))
        evaluator_calls += int(diagnostics.get("evaluator_calls", 0))
        elapsed_ms += float(diagnostics.get("elapsed_ms", 0.0))
        evaluator_ms += float(diagnostics.get("evaluator_elapsed_ms", 0.0))
        for batch, count in row.get("actual_batch_histogram", {}).items():
            histogram[str(batch)] = histogram.get(str(batch), 0) + int(count)
    return {
        "decisions": decisions,
        "nodes": nodes,
        "positions_evaluated": positions,
        "evaluator_calls": evaluator_calls,
        "elapsed_ms": elapsed_ms,
        "evaluator_elapsed_ms": evaluator_ms,
        "nodes_per_decision": nodes / decisions if decisions else 0.0,
        "nodes_per_ms": nodes / elapsed_ms if elapsed_ms else 0.0,
        "evaluator_percent": evaluator_ms * 100.0 / elapsed_ms if elapsed_ms else 0.0,
        "cpu_search_percent": (elapsed_ms - evaluator_ms) * 100.0 / elapsed_ms if elapsed_ms else 0.0,
        "mean_batch": (
            sum(int(batch) * count for batch, count in histogram.items()) / sum(histogram.values())
            if histogram else 0.0
        ),
        "batch_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("block-*.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    valid = [row for row in rows if "arena" in row and int(row.get("games", 0)) > 0]
    failures = [row for row in rows if row not in valid]
    games = sum(int(row["games"]) for row in valid)
    wins = sum(int(row["wins"]) for row in valid)
    losses = sum(int(row["losses"]) for row in valid)
    draws = sum(int(row["draws"]) for row in valid)
    score = wins + 0.5 * draws
    ci_low, ci_high = wilson(score, games)
    arena_keys = (
        "candidate_apm", "champion_apm", "candidate_app", "champion_app",
        "candidate_pps", "champion_pps", "candidate_avg_pieces", "champion_avg_pieces",
        "candidate_avg_seconds", "champion_avg_seconds", "candidate_survival_rate",
        "champion_survival_rate", "candidate_sent_per_game", "champion_sent_per_game",
        "candidate_received_per_game", "champion_received_per_game",
        "candidate_blockout_rate", "champion_blockout_rate", "candidate_lockout_rate",
        "champion_lockout_rate", "candidate_garbageout_rate", "champion_garbageout_rate",
    )
    performance = {key: weighted(valid, key, games) for key in arena_keys} if games else {}
    summary: dict[str, Any] = {
        "condition_id": (
            valid[0].get("condition_id", "").rsplit("-block-", 1)[0]
            if valid else ""
        ),
        "blocks_expected": 50,
        "blocks_found": len(paths),
        "blocks_valid": len(valid),
        "blocks_failed": len(failures),
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": score / games if games else 0.0,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "performance": performance,
        "candidate_search": aggregate_side(valid, "candidate"),
        "champion_search": aggregate_side(valid, "champion"),
        "total_wall_seconds": sum(float(row.get("wall_seconds", 0.0)) for row in valid),
        "total_stderr_bytes": sum(int(row.get("stderr_bytes", 0)) for row in valid),
        "stderr_modes": sorted({row.get("stderr_mode", "") for row in valid}),
        "timeout_or_error_rows": [
            {"condition_id": row.get("condition_id"), "status": row.get("status")} for row in failures
        ],
        "source_files": [str(path) for path in paths],
        "interpretation": (
            "200-game E3 10 ms performance evaluation; independent of the four-game diagnosis rows. "
            "This is a strength/performance observation for the frozen A/B pair, not a promotion decision."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    p = summary["performance"]
    c = summary["candidate_search"]
    a = summary["champion_search"]
    lines = [
        "# 200局Pilot性能評価", "",
        "## 条件", "",
        "E3・10 ms・Gumbel・batch16・fp16・`cuda:1`・最大300手。"
        "固定seed列50個を1 paired block 4局として、blockごとに子プロセスを再起動し、"
        "標準エラー出力排出とバッチ1/2/4/8/16事前ウォームアップを有効にした。", "",
        "## 結果", "",
        f"- 完了: {summary['blocks_valid']}/{summary['blocks_expected']} block、{summary['games']}局。"
        f"失敗/timeout: {summary['blocks_failed']}。",
        f"- B勝数/敗数/引分: {summary['wins']}/{summary['losses']}/{summary['draws']}。",
        f"- B勝率（引分0.5）: {summary['win_rate'] * 100:.1f}% "
        f"（95% CI {summary['ci95_low'] * 100:.1f}–{summary['ci95_high'] * 100:.1f}%）。",
        "",
        "| 指標 | B | A |",
        "|---|---:|---:|",
        f"| APM | {p['candidate_apm']:.2f} | {p['champion_apm']:.2f} |",
        f"| APP | {p['candidate_app']:.4f} | {p['champion_app']:.4f} |",
        f"| PPS | {p['candidate_pps']:.3f} | {p['champion_pps']:.3f} |",
        f"| 平均手数 | {p['candidate_avg_pieces']:.2f} | {p['champion_avg_pieces']:.2f} |",
        f"| 平均生存時間(s) | {p['candidate_avg_seconds']:.3f} | {p['champion_avg_seconds']:.3f} |",
        f"| 生存率 | {p['candidate_survival_rate'] * 100:.1f}% | {p['champion_survival_rate'] * 100:.1f}% |",
        f"| 送信garbage/局 | {p['candidate_sent_per_game']:.3f} | {p['champion_sent_per_game']:.3f} |",
        f"| 受信garbage/局 | {p['candidate_received_per_game']:.3f} | {p['champion_received_per_game']:.3f} |",
        "",
        "## 探索・実行量", "",
        f"- B: {c['nodes_per_decision']:.2f} nodes/decision、{c['nodes_per_ms']:.3f} nodes/ms、"
        f"評価器割合 {c['evaluator_percent']:.2f}%、平均batch {c['mean_batch']:.2f}。",
        f"- A: {a['nodes_per_decision']:.2f} nodes/decision、{a['nodes_per_ms']:.3f} nodes/ms、"
        f"評価器割合 {a['evaluator_percent']:.2f}%、平均batch {a['mean_batch']:.2f}。",
        f"- 標準エラー出力: {', '.join(summary['stderr_modes'])}、総 {summary['total_stderr_bytes']} bytes。",
        "",
        "## 解釈", "",
        "この200局は、4倍条件の判定とは別に、実ゲームでの性能を測るための評価である。"
        "評価器単体ではBが大バッチで4倍以上だが、実ゲームの勝率・nodes/ms・PPSは"
        "別の指標であり、今回の結果だけから実ゲーム4倍とは言えない。Bの採用・昇格は"
        "このPilotの勝率だけで決めず、事前に定めたpromotion条件と必要ならMainで判断する。", "",
    ]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
        "games": games,
        "blocks_valid": len(valid),
        "blocks_failed": len(failures),
        "win_rate": summary["win_rate"],
        "ci95": [ci_low, ci_high],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
