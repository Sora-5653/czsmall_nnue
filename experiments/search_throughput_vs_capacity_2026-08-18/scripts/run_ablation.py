#!/usr/bin/env python3
"""Run the same restartable short pilot across E1/E2/E3 environments."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
PILOT = Path(__file__).with_name("run_pilot.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--budget-ms", type=float, default=40.0)
    parser.add_argument("--blocks", type=int, default=50)
    parser.add_argument("--pieces", type=int, default=50)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    environments = (("E1", 0, 8, 0), ("E2", 2, 8, 2), ("E3", 1, 8, 2))
    for environment, garbage_style, period, lines in environments:
        command = [
            sys.executable, str(PILOT),
            "--candidate", args.candidate, "--champion", args.champion,
            "--condition-prefix", f"ablation-{environment}-{args.budget_ms:g}ms",
            "--environment", environment, "--budget-ms", str(args.budget_ms),
            "--blocks", str(args.blocks), "--pieces", str(args.pieces),
            "--timeout-seconds", str(args.timeout_seconds),
            "--garbage-style", str(garbage_style),
            "--garbage-period", str(period), "--garbage-lines", str(lines),
        ]
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
