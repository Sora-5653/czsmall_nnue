#!/usr/bin/env python3
"""Run a long GPU Arena under an external progress watchdog.

The monitor is a separate OS process from ``run_condition.py``.  This matters
on ROCm because a Python thread cannot reliably interrupt a long native GPU
call that is holding the interpreter.  The monitor records the last state and
trace before terminating only its own child process tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
RUNNER = EXPERIMENT / "scripts" / "run_condition.py"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_trace_tail(path: Path, limit: int = 100) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    events: list[dict] = []
    for line in lines[-limit:]:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def drain(stream, path: Path) -> None:
    with path.open("w", encoding="utf-8") as output:
        for line in iter(stream.readline, ""):
            output.write(line)
            output.flush()


def terminate_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        return
    subprocess.run(["kill", "-TERM", str(pid)], capture_output=True, check=False)


def request_stack_dump(pid: int, stack_path: Path) -> dict[str, object]:
    result: dict[str, object] = {"requested": False, "signal": ""}
    if sys.platform == "win32" and hasattr(signal, "SIGBREAK"):
        try:
            # run_condition registers faulthandler for SIGBREAK when a hang
            # path is supplied.  This is best effort: a native call may still
            # defer signal handling until it returns.
            import os
            os.kill(pid, signal.SIGBREAK)
            result.update({"requested": True, "signal": "SIGBREAK"})
        except OSError as exc:
            result["error"] = str(exc)
    elif sys.platform != "win32" and hasattr(signal, "SIGUSR1"):
        try:
            import os
            os.kill(pid, signal.SIGUSR1)
            result.update({"requested": True, "signal": "SIGUSR1"})
        except OSError as exc:
            result["error"] = str(exc)
    result["stack_file_exists_after_request"] = stack_path.is_file()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--pieces", type=int, default=300)
    parser.add_argument("--budget-ms", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--stderr-mode", choices=("pipe", "drain", "file", "devnull"), default="pipe")
    parser.add_argument("--fixed-shape", action="store_true")
    parser.add_argument("--prewarm-batches", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    trace_path = prefix.with_suffix(".trace.jsonl")
    state_path = prefix.with_suffix(".state.json")
    hang_path = prefix.with_suffix(".hang.json")
    stack_path = prefix.with_suffix(".stacks.txt")
    result_path = prefix.with_suffix(".result.json")
    stdout_path = prefix.with_suffix(".stdout.log")
    stderr_path = prefix.with_suffix(".stderr.log")

    for path in (trace_path, state_path, hang_path, stack_path, result_path,
                 stdout_path, stderr_path):
        if path.exists():
            path.unlink()

    fixed_args = ["--fixed-token-count", "128", "--fixed-action-count", "128"] if args.fixed_shape else [
        "--fixed-token-count", "0", "--fixed-action-count", "0"
    ]
    command = [
        sys.executable, str(RUNNER),
        "--candidate", args.candidate,
        "--champion", args.champion,
        "--engine", str(args.engine.resolve()),
        "--device", args.device,
        "--condition-id", prefix.stem,
        "--variant", "B",
        "--environment", "E3",
        "--budget-ms", str(args.budget_ms),
        "--pairs", "1",
        "--pieces", str(args.pieces),
        "--sims", "100000",
        "--batch", str(args.batch),
        "--determinizations", "1",
        "--precision", "fp16",
        "--seed", str(args.seed),
        "--candidate-time-ms", str(args.budget_ms),
        "--champion-time-ms", str(args.budget_ms),
        "--warmup-passes", "2",
        "--diagnostic-trace", str(trace_path),
        "--hang-output", str(hang_path),
        "--state-output", str(state_path),
        "--watchdog-seconds", "0",
        "--stderr-mode", args.stderr_mode,
        "--output", str(result_path),
        *fixed_args,
    ]
    if args.prewarm_batches:
        command.append("--prewarm-batches")
    if args.stderr_mode == "file":
        command.extend(["--stderr-output", str(stderr_path)])

    started = time.monotonic()
    child = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert child.stdout is not None and child.stderr is not None
    stdout_thread = threading.Thread(target=drain, args=(child.stdout, stdout_path), daemon=True)
    stderr_thread = threading.Thread(target=drain, args=(child.stderr, stderr_path), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    status = "completed"
    monitor_reason = "child exited"
    stack_request: dict[str, object] = {"requested": False}
    while child.poll() is None:
        time.sleep(0.25)
        now = time.monotonic()
        state = read_json(state_path)
        try:
            state_mtime = state_path.stat().st_mtime
            stale_seconds = max(0.0, time.time() - state_mtime)
        except FileNotFoundError:
            stale_seconds = now - started
        if stale_seconds <= args.timeout_seconds:
            continue

        status = "watchdog_timeout"
        monitor_reason = f"state file unchanged for {stale_seconds:.3f}s"
        snapshot = {
            "status": status,
            "reason": monitor_reason,
            "wall_time": time.time(),
            "elapsed_seconds": now - started,
            "parent_pid": child.pid,
            "parent_returncode_before_kill": child.poll(),
            "stderr_mode": args.stderr_mode,
            "fixed_shape": args.fixed_shape,
            "last_state": state,
            "last_trace_events": read_trace_tail(trace_path),
            "trace_path": str(trace_path),
            "state_path": str(state_path),
            "child_command": command,
        }
        hang_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
        stack_request = request_stack_dump(child.pid, stack_path)
        snapshot["stack_request"] = stack_request
        hang_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
        terminate_tree(child.pid)
        break

    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        terminate_tree(child.pid)
        child.wait(timeout=10)
    stdout_thread.join(timeout=2)
    stderr_thread.join(timeout=2)
    monitor = {
        "status": status,
        "reason": monitor_reason,
        "returncode": child.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "command": command,
        "trace_path": str(trace_path),
        "state_path": str(state_path),
        "hang_path": str(hang_path),
        "stack_path": str(stack_path),
        "stack_request": stack_request,
        "result_path": str(result_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    prefix.with_suffix(".monitor.json").write_text(
        json.dumps(monitor, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(monitor, indent=2, ensure_ascii=False))
    return 0 if status == "completed" and child.returncode == 0 else 124


if __name__ == "__main__":
    raise SystemExit(main())
