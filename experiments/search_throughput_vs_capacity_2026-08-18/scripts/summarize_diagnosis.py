#!/usr/bin/env python3
"""Create the final machine-readable and Markdown diagnosis summary.

The script only reads recorded evidence. It never reruns Arena, changes a
checkpoint, or treats a four-game row as a strength result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments" / "search_throughput_vs_capacity_2026-08-18"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def timing_row(rows: list[dict[str, Any]], condition: str, model: str,
              evaluator: str = "model") -> dict[str, Any]:
    for row in rows:
        if (row.get("condition_id") == condition
                and row.get("model") == model
                and row.get("diagnostic_evaluator") == evaluator):
            return row
    raise KeyError(f"timing row not found: {condition}/{model}/{evaluator}")


def compact_timing(row: dict[str, Any]) -> dict[str, Any]:
    timing = row["timing"]
    bridge = row["bridge"]
    return {
        "model": row["model"],
        "diagnostic_evaluator": row["diagnostic_evaluator"],
        "decisions": timing["decisions"],
        "nodes": timing["nodes"],
        "nodes_per_decision": timing["nodes_per_decision"],
        "nodes_per_ms": timing["nodes_per_ms"],
        "elapsed_ms": timing["elapsed_ms"],
        "evaluator_elapsed_ms": timing["evaluator_elapsed_ms"],
        "cpu_search_percent": timing["cpu_search_percent"],
        "evaluator_percent": timing["timing_percent"]["evaluator"],
        "gather_percent": timing["timing_percent"]["gather"],
        "mean_batch": bridge["mean_batch"],
        "batch_histogram": bridge["batch_histogram"],
        "forward_us_per_position_mean": bridge["forward_us_per_position_mean"],
        "service_us_per_position_mean": bridge["service_us_per_position_mean"],
        "forward_us_per_position_p95": bridge["forward_us_per_position_p95"],
        "service_us_per_position_p95": bridge["service_us_per_position_p95"],
        "first_shape_count": bridge["first_shape_count"],
    }


def hang_evidence(hang_dir: Path) -> dict[str, Any]:
    pipe_prefix = hang_dir / "external-variable-pipe-300x4-10ms-b16-v4"
    pipe_monitor = read_json(pipe_prefix.with_suffix(".monitor.json"))
    pipe_hang = read_json(pipe_prefix.with_suffix(".hang.json"))
    pipe_state = read_json(pipe_prefix.with_suffix(".state.json"))
    repeats: list[dict[str, Any]] = []
    for index in (1, 2, 3):
        prefix = hang_dir / f"external-variable-drain-300x4-10ms-b16-repeat{index}"
        monitor = read_json(prefix.with_suffix(".monitor.json"))
        result_path = prefix.with_suffix(".result.json")
        state = read_json(prefix.with_suffix(".state.json"))
        result = read_json(result_path)
        repeats.append({
            "index": index,
            "status": monitor.get("status"),
            "returncode": monitor.get("returncode"),
            "result_exists": result_path.is_file(),
            "games": result.get("games", 0),
            "stderr_mode": result.get("stderr_mode"),
            "stderr_bytes": result.get("stderr_bytes", 0),
            "request_count": result.get("request_count", 0),
            "final_phase": state.get("phase"),
            "trace_exists": Path(result.get("trace_path", "")).is_file(),
        })
    fixed_prefix = hang_dir / "external-fixed-drain-300x4-10ms-b16"
    fixed_monitor = read_json(fixed_prefix.with_suffix(".monitor.json"))
    fixed_result_path = fixed_prefix.with_suffix(".result.json")
    fixed_result = read_json(fixed_result_path)
    return {
        "pipe_control": {
            "status": pipe_monitor.get("status"),
            "returncode": pipe_monitor.get("returncode"),
            "reason": pipe_monitor.get("reason"),
            "last_phase": pipe_hang.get("last_state", {}).get("phase"),
            "last_completed_request": pipe_hang.get("last_state", {}).get(
                "last_completed_request_id"
            ),
            "result_exists": pipe_prefix.with_suffix(".result.json").is_file(),
            "stack_registered": pipe_state.get("stack_signal_registered", False),
        },
        "drain_repeats": repeats,
        "three_repeats_completed": all(
            item["status"] == "completed"
            and item["returncode"] == 0
            and item["result_exists"]
            and item["games"] == 4
            and item["final_phase"] == "child_exited"
            and item["trace_exists"]
            for item in repeats
        ),
        "fixed_shape": {
            "status": fixed_monitor.get("status"),
            "returncode": fixed_monitor.get("returncode"),
            "result_exists": fixed_result_path.is_file(),
            "games": fixed_result.get("games", 0),
            "warmup_batch_shapes": fixed_result.get("warmup_batch_shapes", []),
            "final_phase": read_json(fixed_prefix.with_suffix(".state.json")).get("phase"),
        },
    }


def local_dummy_summary(path: Path) -> dict[str, Any]:
    rows = read_json(path)
    compact: dict[str, Any] = {}
    for entry in rows:
        diagnostics = entry["local_dummy_diagnostics"]
        key = f"{entry['environment']}-{diagnostics['budget_ms']}ms"
        compact[key] = {
            "candidate_nodes_per_ms": (
                diagnostics["candidate"]["nodes"] / diagnostics["candidate"]["elapsed_ms"]
            ),
            "champion_nodes_per_ms": (
                diagnostics["champion"]["nodes"] / diagnostics["champion"]["elapsed_ms"]
            ),
            "candidate_evaluator_elapsed_ms": diagnostics["candidate"]["evaluator_elapsed_ms"],
            "champion_evaluator_elapsed_ms": diagnostics["champion"]["evaluator_elapsed_ms"],
        }
    return compact


def render_markdown(summary: dict[str, Any]) -> str:
    e3_40 = summary["per_decision"]["E3-40ms"]
    e3_160 = summary["per_decision"]["E3-160ms"]
    e1 = summary["per_decision"]["E1-40ms"]
    e2 = summary["per_decision"]["E2-40ms"]
    stable = summary["stability"]
    pilot = summary.get("pilot")
    pilot_lines: list[str] = []
    if pilot:
        performance = pilot["performance"]
        pilot_lines = [
            "## 性能評価（別枠）", "",
            f"- E3・10 ms・batch16・最大300手の200局Pilotは "
            f"{pilot['blocks_valid']}/{pilot['blocks_expected']} block 完了、timeout/error {pilot['blocks_failed']}。",
            f"- B勝率: {pilot['win_rate'] * 100:.1f}% "
            f"（95% CI {pilot['ci95_low'] * 100:.1f}–{pilot['ci95_high'] * 100:.1f}%）、"
            f"勝敗引分 {pilot['wins']}/{pilot['losses']}/{pilot['draws']}。",
            f"- PPS: B {performance['candidate_pps']:.3f}、A {performance['champion_pps']:.3f}; "
            f"平均生存手数: B {performance['candidate_avg_pieces']:.2f}、A {performance['champion_avg_pieces']:.2f}。",
            "- この結果は固定A/Bの性能・強さ評価であり、4倍達成判定やcheckpoint昇格とは別である。",
            "",
        ]
    lines = [
        "# 追加診断結果", "",
        "## 結論", "",
        "長尺Arenaの停止原因は、結果フレーム受信後に子プロセスの終了を待つ間、"
        "子プロセスが標準エラー出力へ書く最終診断JSONで詰まる標準エラー出力パイプの"
        "枯渇と判定した。標準エラー出力を常時排出すると同一シードで3回連続完走し、"
        "固定形状・バッチ事前ウォームアップでも完走した。", "",
        "4倍については、評価器単体のバッチ128/256では達成済みである。一方、実ゲームの"
        "nodes/msや勝率が4倍になったわけではなく、実ゲーム中のforward差はおおむね"
        "2.1〜2.4倍で、40/160 ms E3のnodes差は小さい。したがって、4倍のゲーム実効達成は"
        "今回の証拠からは確認できず、探索・プロトコルを含む実ゲーム4倍は未達として扱う。", "",
        "## per-decision内訳", "",
        "| 条件 | B evaluator | B search | A evaluator | A search |",
        "|---|---:|---:|---:|---:|",
        f"| E3 40 ms | {e3_40['B']['evaluator_percent']:.2f}% | {e3_40['B']['cpu_search_percent']:.2f}% | {e3_40['A']['evaluator_percent']:.2f}% | {e3_40['A']['cpu_search_percent']:.2f}% |",
        f"| E3 160 ms | {e3_160['B']['evaluator_percent']:.2f}% | {e3_160['B']['cpu_search_percent']:.2f}% | {e3_160['A']['evaluator_percent']:.2f}% | {e3_160['A']['cpu_search_percent']:.2f}% |",
        f"| E1 40 ms | {e1['B']['evaluator_percent']:.2f}% | {e1['B']['cpu_search_percent']:.2f}% | {e1['A']['evaluator_percent']:.2f}% | {e1['A']['cpu_search_percent']:.2f}% |",
        f"| E2 40 ms | {e2['B']['evaluator_percent']:.2f}% | {e2['B']['cpu_search_percent']:.2f}% | {e2['A']['evaluator_percent']:.2f}% | {e2['A']['cpu_search_percent']:.2f}% |", "",
        "E3 40/160 msでは両モデルとも評価器が大半だが、時間予算が同じため、速いBは"
        "同じ決定時間内により多くの探索を行う方向へ差が吸収される。E1ではB側の探索・"
        "シミュレータ割合が大きく、Bのnodes優位が見えやすい。E2はE3に近い。", "",
        "## dummyと橋渡し", "",
        "protocol dummyはモデル前向きを抜いてもC++との要求応答を残し、local dummyは"
        "同じC++ Search/Arenaを完全にローカル評価器で動かした。protocol dummyからlocal"
        " dummyへさらに改善する差が、要求応答・Python境界のコストである。", "",
        "## 長尺安定性", "",
        f"- 未排出pipe制御: 最終状態は `{stable['pipe_control']['last_phase']}`、"
        f"最後の完了要求は {stable['pipe_control']['last_completed_request']}、結果ファイルなし。",
        f"- 排出あり3回: `three_repeats_completed={stable['three_repeats_completed']}`。各回4局、"
        "終了コード0、トレースと結果を保存。",
        f"- 固定形状版: `{stable['fixed_shape']}`。",
        "- Windows上ではPythonスタック登録APIが利用できず、スタックファイル取得は未達。"
        "代わりに外部監視の状態遷移と排出あり／なしの再現差を根拠にした。", "",
        *pilot_lines,
        "## 次の一手", "",
        "今回の200局PilotはE3・10 msだけを実施した。40/160 msやE1/E2の200局評価、"
        "およびMainは別条件のため未実施である。実ゲーム4倍を主張するには、探索・"
        "プロトコルを含めた別の最適化と、同一条件の十分な性能・強さ評価が必要である。", "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-summary", type=Path,
                        default=EXPERIMENT / "results/diagnosis/timing_summary_complete.json")
    parser.add_argument("--local-dummy", type=Path,
                        default=EXPERIMENT / "results/diagnosis/local_dummy_summary.json")
    parser.add_argument("--benchmark", type=Path,
                        default=EXPERIMENT / "results/evaluator_benchmark.json")
    parser.add_argument("--hang-dir", type=Path,
                        default=EXPERIMENT / "results/diagnosis/hangs")
    parser.add_argument("--pilot-summary", type=Path,
                        default=EXPERIMENT / "results/pilot/pilot-e3-10ms-pieces300-drain-prewarm-summary.json")
    parser.add_argument("--make-test-passed", action="store_true")
    parser.add_argument("--output-json", type=Path,
                        default=EXPERIMENT / "results/diagnosis/diagnosis_summary.json")
    parser.add_argument("--output-md", type=Path,
                        default=EXPERIMENT / "notes/DIAGNOSIS_RESULT.md")
    args = parser.parse_args()

    rows = read_json(args.timing_summary)
    per_decision: dict[str, dict[str, Any]] = {}
    for label, condition in (
        ("E3-40ms", "diagnosis-e3-40ms-20"),
        ("E3-160ms", "diagnosis-e3-160ms-5"),
        ("E1-40ms", "diagnosis-e1-40ms-20"),
        ("E2-40ms", "diagnosis-e2-40ms-20"),
    ):
        per_decision[label] = {
            "B": compact_timing(timing_row(rows, condition, "B")),
            "A": compact_timing(timing_row(rows, condition, "A")),
        }

    benchmark = read_json(args.benchmark)
    ratios = benchmark.get("throughput_ratio_B_over_A", {})
    real_e3_40_b = per_decision["E3-40ms"]["B"]
    real_e3_40_a = per_decision["E3-40ms"]["A"]
    real_e3_160_b = per_decision["E3-160ms"]["B"]
    real_e3_160_a = per_decision["E3-160ms"]["A"]
    pilot = read_json(args.pilot_summary) if args.pilot_summary.is_file() else None
    summary: dict[str, Any] = {
        "conclusion": {
            "hang_class": "I: stderr pipe saturation after result frame",
            "evaluator_only_4x_at_batch_128": float(ratios.get("128", 0.0)) >= 4.0,
            "evaluator_only_4x_at_batch_256": float(ratios.get("256", 0.0)) >= 4.0,
            "gameplay_4x_claim": False,
            "models_changed": False,
        },
        "evaluator_benchmark": {
            "batch_128_B_over_A": ratios.get("128"),
            "batch_256_B_over_A": ratios.get("256"),
            "device": benchmark.get("device"),
            "states_per_batch_measurement": benchmark.get("states_per_batch_measurement"),
        },
        "per_decision": per_decision,
        "forward_ratio_real_game": {
            "E3-40ms_forward_A_over_B": real_e3_40_a["forward_us_per_position_mean"] / real_e3_40_b["forward_us_per_position_mean"],
            "E3-160ms_forward_A_over_B": real_e3_160_a["forward_us_per_position_mean"] / real_e3_160_b["forward_us_per_position_mean"],
            "E3-40ms_service_A_over_B": real_e3_40_a["service_us_per_position_mean"] / real_e3_40_b["service_us_per_position_mean"],
        },
        "protocol_dummy": {
            row["condition_id"] + "/" + row["model"]: compact_timing(row)
            for row in rows if row.get("diagnostic_evaluator") == "protocol-dummy"
        },
        "local_dummy": local_dummy_summary(args.local_dummy),
        "stability": hang_evidence(args.hang_dir),
        "pilot": pilot,
        "make_test_passed": args.make_test_passed,
    }
    if pilot and int(pilot.get("games", 0)) >= 200:
        summary["pilot_recommendation"] = "e3_10ms_200_game_pilot_completed_remaining_matrix_pending"
    else:
        summary["pilot_recommendation"] = (
            "reopen_200_game_pilot_after_unchanged_conditions"
            if summary["stability"]["three_repeats_completed"] and args.make_test_passed
            else "hold_until_stability_and_make_test_pass"
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
        "three_repeats_completed": summary["stability"]["three_repeats_completed"],
        "make_test_passed": args.make_test_passed,
        "pilot_games": pilot.get("games", 0) if pilot else 0,
        "pilot_recommendation": summary["pilot_recommendation"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
