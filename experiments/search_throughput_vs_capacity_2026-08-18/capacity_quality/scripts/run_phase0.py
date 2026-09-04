#!/usr/bin/env python3
"""Run one Phase 0 A-vs-S400 condition with an explicit ROCm GPU gate.

The existing experiment runner deliberately has no ``--require-gpu`` option;
this wrapper makes the gate mandatory before invoking it and records both the
wrapper command and the child runner command.  It never falls back to CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = ROOT / "experiments" / "search_throughput_vs_capacity_2026-08-18"
CAPACITY = EXPERIMENT / "capacity_quality"
RUNNER = EXPERIMENT / "scripts" / "run_condition.py"
ENGINE = ROOT / "build" / "tetra_cli.exe"
CANDIDATE = ROOT / "models" / "gen14_rank100_100_20260814.best.pt"
REFERENCE = ROOT / "models" / "size_search_ablation_20260816" / "seed42" / "transformer_s.final.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_gate(device_name: str) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type != "cuda":
        raise SystemExit(f"GPU gate rejected non-CUDA device: {device_name}")
    if not torch.cuda.is_available():
        raise SystemExit("GPU gate rejected: torch.cuda.is_available() is false")
    index = 0 if device.index is None else int(device.index)
    if index < 0 or index >= torch.cuda.device_count():
        raise SystemExit(f"GPU gate rejected unavailable device index: {index}")
    torch.cuda.set_device(index)
    name = torch.cuda.get_device_name(index)
    hip = torch.version.hip
    if not hip:
        raise SystemExit("GPU gate rejected: torch.version.hip is empty")
    if "RX 9070 XT" not in name:
        raise SystemExit(f"GPU gate rejected unexpected device: {name}")
    return {
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_version_hip": hip,
        "torch_cuda_is_available": bool(torch.cuda.is_available()),
        "device": device_name,
        "device_index": index,
        "device_name": name,
        "device_count": torch.cuda.device_count(),
        "gcn_arch_name": getattr(torch.cuda.get_device_properties(index), "gcnArchName", "unknown"),
        "require_gpu": True,
        "cpu_fallback_allowed": False,
    }


def next_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}-rerun{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate a non-overwriting output path for {path}")


def mode_args(mode: str) -> tuple[str, dict[str, int]]:
    if mode == "no_search":
        return "no-search", {
            "candidate_sims": 0,
            "champion_sims": 0,
            "candidate_node_budget": -1,
            "champion_node_budget": -1,
        }
    if mode == "equal_node64":
        return "equal-node64", {
            "candidate_sims": -1,
            "champion_sims": -1,
            "candidate_node_budget": 64,
            "champion_node_budget": 64,
        }
    raise ValueError(mode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("no_search", "equal_node64"), required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--require-gpu", action="store_true", help="mandatory explicit GPU gate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--pieces", type=int, default=300)
    parser.add_argument("--watchdog-seconds", type=float, default=300.0)
    args = parser.parse_args()
    if not args.require_gpu:
        raise SystemExit("run_phase0 requires --require-gpu; CPU fallback is forbidden")
    if args.seed != 42:
        raise SystemExit("primary Phase 0 requires --seed 42")
    if args.pairs != 50 or args.pieces != 300:
        raise SystemExit("primary Phase 0 requires --pairs 50 --pieces 300")
    for path in (RUNNER, ENGINE, CANDIDATE, REFERENCE):
        if not path.is_file():
            raise SystemExit(f"required input is missing: {path}")

    mode_label, overrides = mode_args(args.mode)
    gpu = gpu_gate(args.device)
    run_dir = CAPACITY / "results" / "phase0"
    log_dir = CAPACITY / "results" / "logs"
    record_dir = CAPACITY / "notes" / "run_records"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    stem = f"A-vs-S400-{mode_label}-seed42-pairs50"
    output = next_path(run_dir / f"{stem}.json")
    stdout_path = next_path(log_dir / f"{stem}.stdout.log")
    stderr_path = next_path(log_dir / f"{stem}.stderr.log")
    request_path = next_path(log_dir / f"{stem}.requests.jsonl")
    trace_path = next_path(log_dir / f"{stem}.trace.jsonl")
    state_path = next_path(log_dir / f"{stem}.state.json")
    hang_path = next_path(log_dir / f"{stem}.hang.json")

    child_command = [
        sys.executable,
        str(RUNNER),
        "--candidate", str(CANDIDATE),
        "--champion", str(REFERENCE),
        "--engine", str(ENGINE),
        "--device", args.device,
        "--condition-id", f"phase0-A-vs-S400-{mode_label}-seed42-pairs50",
        "--variant", "A-vs-S400",
        "--environment", "E3",
        "--budget-ms", "0",
        "--pairs", str(args.pairs),
        "--pieces", str(args.pieces),
        "--sims", "100000",
        "--batch", "16",
        "--determinizations", "1",
        "--precision", "fp16",
        "--seed", str(args.seed),
        "--candidate-sims", str(overrides["candidate_sims"]),
        "--champion-sims", str(overrides["champion_sims"]),
        "--candidate-node-budget", str(overrides["candidate_node_budget"]),
        "--champion-node-budget", str(overrides["champion_node_budget"]),
        "--garbage-style", "1",
        "--garbage-period", "8",
        "--garbage-lines", "2",
        "--fixed-token-count", "0",
        "--fixed-action-count", "0",
        "--prewarm-batches",
        "--watchdog-seconds", str(args.watchdog_seconds),
        "--stderr-mode", "drain",
        "--request-log", str(request_path),
        "--diagnostic-trace", str(trace_path),
        "--state-output", str(state_path),
        "--hang-output", str(hang_path),
        "--output", str(output),
    ]
    started_at = time.time()
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            child_command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    elapsed = time.perf_counter() - started
    result_exists = output.is_file()
    result: dict[str, object] | None = None
    if result_exists:
        try:
            result = json.loads(output.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = None
    status = "complete" if completed.returncode == 0 and result and int(result.get("games", 0)) == 200 else "incomplete"
    record = {
        "status": status,
        "mode": args.mode,
        "mode_label": mode_label,
        "started_at_unix": started_at,
        "elapsed_seconds": elapsed,
        "exit_code": completed.returncode,
        "wrapper_command": [sys.executable, str(Path(__file__).resolve()), "--mode", args.mode, "--device", args.device, "--require-gpu", "--seed", "42", "--pairs", "50", "--pieces", "300"],
        "child_command": child_command,
        "gpu_gate": gpu,
        "checkpoint_sha256_before": {
            "A": sha256(CANDIDATE),
            "S400": sha256(REFERENCE),
        },
        "output": str(output),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "request_log": str(request_path),
        "trace_log": str(trace_path),
        "state_log": str(state_path),
        "hang_log": str(hang_path),
        "result": result,
    }
    record_path = next_path(record_dir / f"{stem}.record.json")
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "exit_code": completed.returncode,
        "games": result.get("games") if result else None,
        "output": str(output),
        "record": str(record_path),
        "elapsed_seconds": elapsed,
    }, indent=2, ensure_ascii=False), flush=True)
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
