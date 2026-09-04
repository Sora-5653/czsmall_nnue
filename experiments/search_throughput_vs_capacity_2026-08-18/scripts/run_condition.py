#!/usr/bin/env python3
"""Run one or more fixed-condition GPU Arena blocks for the experiment.

The C++ child remains authoritative for rules, move generation, search, and
game scoring. This wrapper only serves the two checkpoints over the existing
binary GPU bridge and records the child diagnostics emitted on stderr.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "trainer"))

from gpu_arena import ARENA_MAGIC, REQUEST_MAGIC, _read_arena_result  # noqa: E402
from gpu_match import (  # noqa: E402
    _serve_request_batch,
    load_model,
    read_exact,
    read_request_frame,
)


EXPERIMENT = ROOT / "experiments" / "search_throughput_vs_capacity_2026-08-18"
DEFAULT_ENGINE = ROOT / ("build/tetra_cli.exe" if os.name == "nt" else "build/tetra_cli")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def parse_diagnostics(stderr: str) -> dict[str, object]:
    for line in reversed(stderr.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "arena_diagnostics" in value:
            return value
    return {"parse_error": "arena diagnostics JSON was not found", "stderr": stderr}


def warmup_model(model: torch.nn.Module, device: torch.device, precision: str,
                 batch: int, passes: int, token_count: int,
                 action_count: int, prewarm_batches: bool = False) -> float:
    """Pay the one-time ROCm/kernel setup cost before measured game play.

    The child protocol can request a variable number of positions on its first
    frame.  Warming both the single-position and configured batch shapes keeps
    that implementation detail from turning the first measured decision into
    a multi-second compiler/load outlier.  The elapsed time is returned and is
    recorded separately from the game wall clock.
    """
    if passes <= 0:
        return 0.0
    warm_token_count = token_count if token_count > 0 else 102
    warm_action_count = action_count if action_count > 0 else 106
    if prewarm_batches:
        batch_values = sorted({1, 2, 4, 8, 16, max(1, min(int(batch), 16))})
    else:
        batch_values = sorted({1, max(1, min(int(batch), 16))})
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    started = time.perf_counter()
    with torch.inference_mode():
        for warm_batch in batch_values:
            token = torch.rand(warm_batch, warm_token_count, 24, device=device)
            token_mask = torch.ones(warm_batch, warm_token_count, device=device)
            action = torch.rand(warm_batch, warm_action_count, 24, device=device)
            action_mask = torch.ones(warm_batch, warm_action_count, device=device)
            for _ in range(passes):
                if autocast_dtype is not None and device.type == "cuda":
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                        model(token, token_mask, action, action_mask)
                else:
                    model(token, token_mask, action, action_mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            del token, token_mask, action, action_mask
    return time.perf_counter() - started


def pad_request(request, token_count: int, action_count: int):
    """Use one masked tensor shape for all variable-length game requests.

    ROCm pays a substantial one-time dispatch/setup cost for every new tensor
    shape. The model masks token and action padding, so fixed padding preserves
    the legal-action semantics while preventing a long Arena from compiling a
    new kernel for every action count and token count encountered in play.
    """
    if token_count <= 0 or action_count <= 0:
        return request
    tokens = request.token_np
    actions = request.action_np
    if tokens.shape[1] > token_count or actions.shape[1] > action_count:
        raise RuntimeError(
            f"request shape exceeds fixed padding: tokens={tokens.shape[1]}/{token_count}, "
            f"actions={actions.shape[1]}/{action_count}"
        )
    if tokens.shape[1] == token_count and actions.shape[1] == action_count:
        return request

    token_padded = np.zeros(
        (tokens.shape[0], token_count, tokens.shape[2]), dtype=np.float32
    )
    token_mask_padded = np.zeros(
        (request.token_mask_np.shape[0], token_count), dtype=np.float32
    )
    action_padded = np.zeros(
        (actions.shape[0], action_count, actions.shape[2]), dtype=np.float32
    )
    action_mask_padded = np.zeros(
        (request.action_mask_np.shape[0], action_count), dtype=np.float32
    )
    token_padded[:, :tokens.shape[1], :] = tokens
    token_mask_padded[:, :request.token_mask_np.shape[1]] = request.token_mask_np
    action_padded[:, :actions.shape[1], :] = actions
    action_mask_padded[:, :request.action_mask_np.shape[1]] = request.action_mask_np
    request.token_np = token_padded
    request.token_mask_np = token_mask_padded
    request.action_np = action_padded
    request.action_mask_np = action_mask_padded
    return request


def run_block(
    candidate: torch.nn.Module,
    champion: torch.nn.Module,
    device: torch.device,
    engine: Path,
    *,
    pairs: int,
    sims: int,
    pieces: int,
    batch: int,
    determinizations: int,
    use_gumbel: bool,
    precision: str,
    seed: int,
    candidate_sims: int = -1,
    champion_sims: int = -1,
    candidate_time_ms: float = -1.0,
    champion_time_ms: float = -1.0,
    candidate_node_budget: int = -1,
    champion_node_budget: int = -1,
    garbage_style: int = 1,
    garbage_period: int = 8,
    garbage_lines: int = 2,
    request_log: Path | None = None,
    trace_path: Path | None = None,
    hang_path: Path | None = None,
    state_path: Path | None = None,
    watchdog_seconds: float = 0.0,
    stderr_mode: str = "drain",
    stderr_path: Path | None = None,
    diagnostic_evaluator: str = "model",
    fixed_token_count: int = 128,
    fixed_action_count: int = 128,
) -> dict[str, object]:
    command = [
        str(engine),
        "gpu-arena-protocol",
        str(max(1, pairs)),
        str(max(0, sims)),
        str(max(1, pieces)),
        str(max(1, batch)),
        str(max(1, determinizations)),
        "1" if use_gumbel else "0",
        str(max(0, seed)),
        str(candidate_sims),
        str(champion_sims),
        "-1",
        "-1",
        "0.01",
        "0.05",
        "-1",
        "-1",
        "-1",
        "-1",
        f"{candidate_time_ms:.9g}",
        f"{champion_time_ms:.9g}",
        str(candidate_node_budget),
        str(champion_node_budget),
        str(garbage_style),
        str(max(1, garbage_period)),
        str(max(0, garbage_lines)),
    ]
    started = time.perf_counter()
    stderr_handle = None
    stderr_target = subprocess.PIPE
    if stderr_mode == "devnull":
        stderr_target = subprocess.DEVNULL
    elif stderr_mode == "file":
        if stderr_path is None:
            raise ValueError("stderr_path is required when stderr_mode=file")
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_handle = stderr_path.open("wb")
        stderr_target = stderr_handle
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_target,
        bufsize=0,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    if stderr_mode in {"pipe", "drain"}:
        assert proc.stderr is not None

    stderr_chunks: list[bytes] = []
    stderr_thread: threading.Thread | None = None
    if stderr_mode == "drain":
        def drain_stderr() -> None:
            assert proc.stderr is not None
            while True:
                chunk = proc.stderr.read(64 * 1024)
                if not chunk:
                    return
                stderr_chunks.append(chunk)

        stderr_thread = threading.Thread(
            target=drain_stderr, name="gpu-arena-stderr-drain", daemon=True
        )
        stderr_thread.start()

    trace_stream = None
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_stream = trace_path.open("w", encoding="utf-8", buffering=1)
    stack_signal_stream = None
    stack_signal = getattr(signal, "SIGBREAK", None) if os.name == "nt" else getattr(signal, "SIGUSR1", None)
    stack_signal_registered = False
    if hang_path is not None and stack_signal is not None:
        hang_path.parent.mkdir(parents=True, exist_ok=True)
        stack_signal_stream = hang_path.with_suffix(".stacks.txt").open(
            "a", encoding="utf-8", buffering=1
        )
        try:
            if not hasattr(faulthandler, "register"):
                raise AttributeError("faulthandler.register is unavailable on this platform")
            faulthandler.register(
                stack_signal, file=stack_signal_stream, all_threads=True, chain=False
            )
            stack_signal_registered = True
        except (AttributeError, RuntimeError, ValueError):
            stack_signal_stream.close()
            stack_signal_stream = None
    recent_trace: list[dict[str, object]] = []
    shape_first_seen: dict[str, int] = {}
    batch_histogram_all: dict[str, int] = {}
    shape_histogram_all: dict[str, int] = {}
    state: dict[str, object] = {
        "started_at": time.time(),
        "child_pid": proc.pid,
        "phase": "started",
        "request_count": 0,
        "last_completed_request_id": -1,
        "current_request_id": -1,
        "last_completed_decision_id": -1,
        "current_game_id": None,
        "current_model_id": None,
        "batch": None,
        "tokens": None,
        "actions": None,
        "last_progress_monotonic": time.monotonic(),
        "child_returncode": None,
        "stack_signal": str(stack_signal) if stack_signal is not None else "",
        "stack_signal_registered": stack_signal_registered,
    }

    def persist_state() -> None:
        if state_path is None:
            return
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    persist_state()
    watchdog_stop = threading.Event()
    watchdog_triggered = threading.Event()
    watchdog_thread: threading.Thread | None = None

    def emit_trace(event: dict[str, object]) -> None:
        recent_trace.append(dict(event))
        if len(recent_trace) > 100:
            del recent_trace[:-100]
        if trace_stream is not None:
            trace_stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            trace_stream.flush()

    def save_hang(reason: str) -> None:
        if watchdog_triggered.is_set():
            return
        watchdog_triggered.set()
        now = time.time()
        snapshot = dict(state)
        snapshot.update({
            "reason": reason,
            "wall_time": now,
            "child_alive": proc.poll() is None,
            "child_returncode": proc.poll(),
            "stderr_mode": stderr_mode,
            "recent_trace": recent_trace[-100:],
        })
        if hang_path is None:
            return
        hang_path.parent.mkdir(parents=True, exist_ok=True)
        hang_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
        stack_path = hang_path.with_suffix(".stacks.txt")
        with stack_path.open("w", encoding="utf-8") as stream:
            stream.write(f"watchdog reason: {reason}\n")
            faulthandler.dump_traceback(file=stream, all_threads=True)

    if watchdog_seconds > 0.0:
        def watchdog() -> None:
            while not watchdog_stop.wait(0.25):
                last_progress = float(state["last_progress_monotonic"])
                if time.monotonic() - last_progress <= watchdog_seconds:
                    continue
                save_hang(f"progress timeout after {watchdog_seconds:g}s")
                if proc.poll() is None:
                    proc.kill()
                return

        watchdog_thread = threading.Thread(
            target=watchdog, name="gpu-arena-watchdog", daemon=True
        )
        watchdog_thread.start()

    inference_seconds = 0.0
    request_index = 0
    result: dict[str, object] | None = None
    models = {0: candidate, 1: champion}
    try:
        while True:
            state["phase"] = "reading_frame_magic"
            state["last_progress_monotonic"] = time.monotonic()
            persist_state()
            magic = read_exact(proc.stdout, 4)
            if magic == REQUEST_MAGIC:
                state["phase"] = "reading_request_frame"
                request_started = time.perf_counter()
                request = read_request_frame(proc, request_index)
                request_read_finished = time.perf_counter()
                raw_batch = int(request.token_np.shape[0])
                raw_tokens = int(request.token_np.shape[1])
                raw_actions = int(request.action_np.shape[1])
                shape_key = f"{raw_batch}x{raw_tokens}x{raw_actions}"
                shape_seen_before = shape_key in shape_first_seen
                shape_first_seen.setdefault(shape_key, request_index)
                batch_histogram_all[str(raw_batch)] = (
                    batch_histogram_all.get(str(raw_batch), 0) + 1
                )
                shape_histogram_all[shape_key] = shape_histogram_all.get(shape_key, 0) + 1
                state.update({
                    "phase": "request_received",
                    "current_request_id": request_index,
                    "current_model_id": request.model_id,
                    "batch": raw_batch,
                    "tokens": raw_tokens,
                    "actions": raw_actions,
                    "last_progress_monotonic": time.monotonic(),
                })
                persist_state()
                pad_request(request, fixed_token_count, fixed_action_count)
                event: dict[str, object] = {
                    "event": "inference_complete",
                    "request_id": request_index,
                    "model_id": int(request.model_id),
                    "wire_batch": raw_batch,
                    "wire_tokens": raw_tokens,
                    "wire_actions": raw_actions,
                    "served_batch": int(request.token_np.shape[0]),
                    "served_tokens": int(request.token_np.shape[1]),
                    "served_actions": int(request.action_np.shape[1]),
                    "shape_seen_before": shape_seen_before,
                    "shape_first_seen_index": shape_first_seen[shape_key],
                    "request_read_us": (request_read_finished - request_started) * 1e6,
                }
                if request_log is not None:
                    request_log.parent.mkdir(parents=True, exist_ok=True)
                    with request_log.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({
                            "request_index": request_index,
                            "model_id": request.model_id,
                            "batch": raw_batch,
                            "tokens": raw_tokens,
                            "actions": raw_actions,
                            "served_batch": int(request.token_np.shape[0]),
                            "served_tokens": int(request.token_np.shape[1]),
                            "served_actions": int(request.action_np.shape[1]),
                            "shape_seen_before": shape_seen_before,
                            "started_at": time.time(),
                        }) + "\n")
                        stream.flush()
                state["phase"] = "serving_request"
                inference_seconds += _serve_request_batch(
                    models, device, [request], precision,
                    trace=event if trace_path is not None else None,
                    diagnostic_dummy=diagnostic_evaluator == "protocol-dummy",
                )
                event["completed_at"] = time.time()
                event["request_total_us"] = (time.perf_counter() - request_started) * 1e6
                event["diagnostic_evaluator"] = diagnostic_evaluator
                emit_trace(event)
                state.update({
                    "phase": "response_complete",
                    "last_completed_request_id": request_index,
                    "request_count": request_index + 1,
                    "last_progress_monotonic": time.monotonic(),
                })
                persist_state()
                request_index += 1
                continue
            if magic == ARENA_MAGIC:
                state["phase"] = "reading_result_frame"
                persist_state()
                result = _read_arena_result(proc)
                state["phase"] = "result_complete"
                state["last_progress_monotonic"] = time.monotonic()
                persist_state()
                break
            raise RuntimeError(f"unexpected GPU Arena frame: {magic!r}")
    except BaseException:
        if watchdog_triggered.is_set():
            raise RuntimeError(f"GPU Arena watchdog stopped the child; see {hang_path}")
        proc.kill()
        proc.wait()
        raise
    finally:
        watchdog_stop.set()
        if watchdog_thread is not None:
            watchdog_thread.join(timeout=1.0)
    state["phase"] = "waiting_child_exit"
    state["last_progress_monotonic"] = time.monotonic()
    persist_state()
    proc.stdin.close()
    return_code = proc.wait()
    state["child_returncode"] = return_code
    state["phase"] = "child_exited"
    state["last_progress_monotonic"] = time.monotonic()
    persist_state()
    if stderr_mode == "pipe":
        assert proc.stderr is not None
        stderr_bytes = proc.stderr.read()
    elif stderr_mode == "drain":
        assert proc.stderr is not None
        if stderr_thread is not None:
            stderr_thread.join(timeout=2.0)
        stderr_bytes = b"".join(stderr_chunks)
    elif stderr_mode == "file":
        assert stderr_path is not None
        if stderr_handle is not None:
            stderr_handle.close()
        stderr_bytes = stderr_path.read_bytes()
    else:
        stderr_bytes = b""
    if trace_stream is not None:
        trace_stream.close()
    if stack_signal_registered and stack_signal is not None:
        try:
            faulthandler.unregister(stack_signal)
        except (RuntimeError, ValueError):
            pass
    if stack_signal_stream is not None:
        stack_signal_stream.close()
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    if return_code != 0:
        raise RuntimeError(f"tetra_cli failed with {return_code}: {stderr}")
    assert result is not None
    return {
        "arena": result,
        "diagnostics": parse_diagnostics(stderr),
        "gpu_inference_seconds": inference_seconds,
        "wall_seconds": time.perf_counter() - started,
        "request_count": request_index,
        "actual_batch_histogram": batch_histogram_all,
        "wire_shape_histogram": shape_histogram_all,
        "stderr_mode": stderr_mode,
        "stderr_bytes": len(stderr_bytes),
        "trace_path": str(trace_path) if trace_path is not None else "",
        "hang_path": str(hang_path) if hang_path is not None else "",
        "state_path": str(state_path) if state_path is not None else "",
        "command": command,
    }


def wilson(score: float, games: int) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    p = score / games
    den = 1.0 + z * z / games
    center = (p + z * z / (2.0 * games)) / den
    half = z * math.sqrt(p * (1.0 - p) / games + z * z / (4.0 * games * games)) / den
    return max(0.0, center - half), min(1.0, center + half)


def check_device(device: torch.device) -> str:
    if device.type != "cuda":
        return str(device)
    if not torch.cuda.is_available():
        raise SystemExit("GPU requested but torch.cuda.is_available() is false")
    index = 0 if device.index is None else device.index
    if index < 0 or index >= torch.cuda.device_count():
        raise SystemExit(f"requested CUDA device {index} is not available")
    return torch.cuda.get_device_name(index)


def make_row(args: argparse.Namespace, block: dict[str, object],
             warmup_seconds: float) -> dict[str, object]:
    arena = block["arena"]
    assert isinstance(arena, dict)
    diagnostics = block["diagnostics"]
    assert isinstance(diagnostics, dict)
    games = int(arena["games_played"])
    score = int(arena["candidate_wins"]) + 0.5 * int(arena["draws"])
    ci_low, ci_high = wilson(score, games)
    return {
        "condition_id": args.condition_id,
        "variant": args.variant,
        "budget_ms": args.budget_ms,
        "environment": args.environment,
        "seed": args.seed,
        "pairs": args.pairs,
        "games": games,
        "wins": int(arena["candidate_wins"]),
        "losses": int(arena["champion_wins"]),
        "draws": int(arena["draws"]),
        "win_rate": score / games if games else 0.0,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "arena": arena,
        "diagnostics": diagnostics,
        "warmup_seconds": warmup_seconds,
        "warmup_passes": args.warmup_passes,
        "warmup_batch_shapes": (
            sorted({1, 2, 4, 8, 16, max(1, min(args.batch, 16))})
            if args.prewarm_batches
            else sorted({1, max(1, min(args.batch, 16))})
        ),
        "gpu_inference_seconds": block["gpu_inference_seconds"],
        "wall_seconds": block["wall_seconds"],
        "request_count": block.get("request_count", 0),
        "actual_batch_histogram": block.get("actual_batch_histogram", {}),
        "wire_shape_histogram": block.get("wire_shape_histogram", {}),
        "stderr_mode": block.get("stderr_mode", "pipe"),
        "stderr_bytes": block.get("stderr_bytes", 0),
        "trace_path": block.get("trace_path", ""),
        "hang_path": block.get("hang_path", ""),
        "state_path": block.get("state_path", ""),
        "command": block["command"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--engine", default=str(DEFAULT_ENGINE))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--variant", default="B")
    parser.add_argument("--environment", default="E3")
    parser.add_argument("--budget-ms", type=float, default=0.0)
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--pieces", type=int, default=100)
    parser.add_argument("--sims", type=int, default=100000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--determinizations", type=int, default=1)
    parser.add_argument("--precision", choices=("fp16", "fp32", "bf16"), default="fp16")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--no-gumbel", action="store_true")
    parser.add_argument("--candidate-sims", type=int, default=-1)
    parser.add_argument("--champion-sims", type=int, default=-1)
    parser.add_argument("--candidate-time-ms", type=float, default=-1.0)
    parser.add_argument("--champion-time-ms", type=float, default=-1.0)
    parser.add_argument("--candidate-node-budget", type=int, default=-1)
    parser.add_argument("--champion-node-budget", type=int, default=-1)
    parser.add_argument("--garbage-style", type=int, default=1)
    parser.add_argument("--garbage-period", type=int, default=8)
    parser.add_argument("--garbage-lines", type=int, default=2)
    parser.add_argument("--warmup-passes", type=int, default=2)
    parser.add_argument(
        "--prewarm-batches", action="store_true",
        help="warm batch shapes 1, 2, 4, 8, and 16 before measured play",
    )
    parser.add_argument("--request-log", default="")
    parser.add_argument("--diagnostic-trace", default="",
                        help="write per-request bridge timing JSONL")
    parser.add_argument("--hang-output", default="",
                        help="write watchdog state JSON and Python stacks here")
    parser.add_argument("--state-output", default="",
                        help="write the current request state for an external watchdog")
    parser.add_argument("--watchdog-seconds", type=float, default=0.0,
                        help="stop and dump state after this much progress silence")
    parser.add_argument("--stderr-mode", choices=("pipe", "drain", "file", "devnull"),
                        default="drain",
                        help="drain is the stable default; pipe reproduces the old hang-prone control")
    parser.add_argument("--stderr-output", default="",
                        help="stderr file for --stderr-mode file")
    parser.add_argument("--diagnostic-evaluator", choices=("model", "protocol-dummy"),
                        default="model",
                        help="protocol-dummy skips model forward; never use for strength")
    parser.add_argument("--fixed-token-count", type=int, default=128,
                        help="fixed masked token width; <=0 keeps protocol shapes")
    parser.add_argument("--fixed-action-count", type=int, default=128,
                        help="fixed masked action width; <=0 keeps protocol shapes")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    candidate_path = Path(args.candidate).resolve()
    champion_path = Path(args.champion).resolve()
    engine = Path(args.engine).resolve()
    if not candidate_path.is_file() or not champion_path.is_file():
        raise SystemExit("candidate/champion checkpoint is missing")
    if not engine.is_file():
        raise SystemExit(f"engine is missing: {engine}")
    device = torch.device(args.device)
    device_name = check_device(device)
    candidate = load_model(str(candidate_path), device)
    champion = load_model(str(champion_path), device)
    warmup_started = time.perf_counter()
    try:
        candidate_warmup_seconds = warmup_model(
            candidate, device, args.precision, args.batch, args.warmup_passes,
            args.fixed_token_count, args.fixed_action_count, args.prewarm_batches
        )
        champion_warmup_seconds = warmup_model(
            champion, device, args.precision, args.batch, args.warmup_passes,
            args.fixed_token_count, args.fixed_action_count, args.prewarm_batches
        )
        warmup_seconds = time.perf_counter() - warmup_started
        block = run_block(
            candidate,
            champion,
            device,
            engine,
            pairs=args.pairs,
            sims=args.sims,
            pieces=args.pieces,
            batch=args.batch,
            determinizations=args.determinizations,
            use_gumbel=not args.no_gumbel,
            precision=args.precision,
            seed=args.seed,
            candidate_sims=args.candidate_sims,
            champion_sims=args.champion_sims,
            candidate_time_ms=args.candidate_time_ms,
            champion_time_ms=args.champion_time_ms,
            candidate_node_budget=args.candidate_node_budget,
            champion_node_budget=args.champion_node_budget,
            garbage_style=args.garbage_style,
            garbage_period=args.garbage_period,
            garbage_lines=args.garbage_lines,
            request_log=Path(args.request_log).resolve() if args.request_log else None,
            trace_path=Path(args.diagnostic_trace).resolve() if args.diagnostic_trace else None,
            hang_path=Path(args.hang_output).resolve() if args.hang_output else None,
            state_path=Path(args.state_output).resolve() if args.state_output else None,
            watchdog_seconds=max(0.0, args.watchdog_seconds),
            stderr_mode=args.stderr_mode,
            stderr_path=Path(args.stderr_output).resolve() if args.stderr_output else None,
            diagnostic_evaluator=args.diagnostic_evaluator,
            fixed_token_count=args.fixed_token_count,
            fixed_action_count=args.fixed_action_count,
        )
    finally:
        del candidate, champion
        torch.cuda.empty_cache()

    row = make_row(args, block, warmup_seconds)
    row["model_warmup_seconds"] = {
        "candidate": candidate_warmup_seconds,
        "champion": champion_warmup_seconds,
    }
    row["device"] = device_name
    row["git_commit"] = git_value("rev-parse", "HEAD")
    row["git_status"] = git_value("status", "--short", "--branch")
    row["checkpoint_sha256"] = {
        "candidate": sha256(candidate_path),
        "champion": sha256(champion_path),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "device": device_name,
        "games": row["games"],
        "win_rate": row["win_rate"],
        "ci95": [row["ci95_low"], row["ci95_high"]],
        "wall_seconds": row["wall_seconds"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
