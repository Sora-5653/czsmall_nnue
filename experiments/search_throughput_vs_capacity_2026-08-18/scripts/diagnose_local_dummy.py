#!/usr/bin/env python3
"""Run the same C++ Arena/search with a local uniform evaluator.

The command is a diagnostic ceiling, not a gameplay or strength benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def run_case(engine: Path, environment: str, budget: int, pieces: int, seed: int) -> dict:
    garbage_style = {"E1": 0, "E2": 2, "E3": 1}[environment]
    command = [
        str(engine), "arena-diagnostic", "uniform", "uniform", "1", "100000",
        str(pieces), str(budget), str(garbage_style), str(seed),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=180)
    value = None
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            value = json.loads(line)
            break
    if value is None:
        raise RuntimeError(f"local dummy did not return JSON: {completed.stdout!r}")
    value["environment"] = environment
    value["pieces"] = pieces
    value["seed"] = seed
    value["command"] = command
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        ("E3", 10, 5, 2026081836),
        ("E3", 40, 20, 2026081832),
        ("E3", 160, 5, 2026081837),
        ("E1", 40, 20, 2026081838),
    ]
    results = [run_case(args.engine.resolve(), *case) for case in cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(results), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
