#!/usr/bin/env python3
"""Run restartable paired pilot blocks with a per-block timeout.

The default block is one four-game Arena factorial unit. Restarting after each
unit bounds ROCm state and leaves a JSON failure record instead of silently
mixing an interrupted block into the score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments" / "search_throughput_vs_capacity_2026-08-18"
RUN_CONDITION = Path(__file__).with_name("run_condition.py")


def load_seeds() -> list[int]:
    return [
        int(line.strip())
        for line in (EXPERIMENT / "manifests" / "seeds.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--condition-prefix", required=True)
    parser.add_argument("--variant", default="B")
    parser.add_argument("--environment", default="E3")
    parser.add_argument("--budget-ms", type=float, required=True)
    parser.add_argument("--blocks", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--pieces", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--stderr-mode", choices=("drain", "file", "devnull"), default="drain")
    parser.add_argument("--prewarm-batches", action="store_true",
                        help="prewarm batch shapes 1, 2, 4, 8, and 16")
    parser.add_argument("--garbage-style", type=int, default=1)
    parser.add_argument("--garbage-period", type=int, default=8)
    parser.add_argument("--garbage-lines", type=int, default=2)
    args = parser.parse_args()

    seeds = load_seeds()
    if args.start_index < 0 or args.start_index + args.blocks > len(seeds):
        raise SystemExit("requested pilot block range is outside manifests/seeds.txt")
    output_dir = EXPERIMENT / "results" / "pilot" / args.condition_prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(args.start_index, args.start_index + args.blocks):
        seed = seeds[index]
        output = output_dir / f"block-{index:03d}.json"
        if output.is_file():
            continue
        command = [
            sys.executable, str(RUN_CONDITION),
            "--candidate", args.candidate,
            "--champion", args.champion,
            "--condition-id", f"{args.condition_prefix}-block-{index:03d}",
            "--variant", args.variant,
            "--environment", args.environment,
            "--budget-ms", str(args.budget_ms),
            "--pairs", "1",
            "--pieces", str(args.pieces),
            "--sims", "100000",
            "--batch", str(args.batch),
            "--determinizations", "1",
            "--precision", "fp16",
            "--seed", str(seed),
            "--candidate-time-ms", str(args.budget_ms),
            "--champion-time-ms", str(args.budget_ms),
            "--garbage-style", str(args.garbage_style),
            "--garbage-period", str(args.garbage_period),
            "--garbage-lines", str(args.garbage_lines),
            "--fixed-token-count", "0",
            "--fixed-action-count", "0",
            "--stderr-mode", args.stderr_mode,
            "--output", str(output),
        ]
        if args.prewarm_batches:
            command.append("--prewarm-batches")
        stdout_path = output.with_suffix(".stdout")
        stderr_path = output.with_suffix(".stderr")
        try:
            completed = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True,
                timeout=args.timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            failure = {
                "status": "timeout",
                "condition_id": f"{args.condition_prefix}-block-{index:03d}",
                "seed": seed,
                "timeout_seconds": args.timeout_seconds,
            }
            output.write_text(json.dumps(failure, indent=2), encoding="utf-8")
            stdout_path.write_text(str(exc.stdout or ""), encoding="utf-8")
            stderr_path.write_text(str(exc.stderr or ""), encoding="utf-8")
            continue
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0 and not output.exists():
            output.write_text(json.dumps({
                "status": "error",
                "condition_id": f"{args.condition_prefix}-block-{index:03d}",
                "seed": seed,
                "returncode": completed.returncode,
            }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
