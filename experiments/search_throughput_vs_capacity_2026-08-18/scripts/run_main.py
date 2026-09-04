#!/usr/bin/env python3
"""Entry point for a main run; delegates to the restartable pilot driver."""

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
    parser.add_argument("--condition-prefix", required=True)
    parser.add_argument("--variant", default="B")
    parser.add_argument("--environment", default="E3")
    parser.add_argument("--budget-ms", type=float, required=True)
    parser.add_argument("--blocks", type=int, default=500,
                        help="500 blocks = 2,000 factorial games")
    parser.add_argument("--pieces", type=int, default=300)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()
    command = [sys.executable, str(PILOT)]
    for key in ("candidate", "champion", "condition_prefix", "variant", "environment", "budget_ms", "blocks", "pieces", "timeout_seconds"):
        command.extend([f"--{key.replace('_', '-')}", str(getattr(args, key))])
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
