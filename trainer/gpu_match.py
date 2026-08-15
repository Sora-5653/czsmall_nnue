# SPDX-License-Identifier: MIT
"""Run the C++ game/search loop with TetraFormer inference on a GPU.

The C++ child owns the deterministic rules, move generation and search.  This
process owns PyTorch and answers batched evaluator frames on the requested GPU.
The final report uses the simulator's counters and exposes the compact Tetr.io
style metrics APM (attack per minute) and APP (attack per placed piece).

Example::

    python trainer/gpu_match.py models/gpu_gen_20260805.pt \
        --device cuda:1 --games 4 --pieces 200 --sims 32
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import queue
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from tetraformer import TetraFormer, TetraFormerConfig


REQUEST_MAGIC = b"TGPU"
RESULT_MAGIC = b"TGST"
EXPORT_MAGIC = b"TGED"
ARENA_MAGIC = b"TGAR"
TOKEN_FEATURES = 24
ACTION_FEATURES = 24


@dataclass
class GpuRequest:
    index: int
    proc: subprocess.Popen
    model_id: int
    token_np: np.ndarray
    token_mask_np: np.ndarray
    action_np: np.ndarray
    action_mask_np: np.ndarray


@dataclass
class ChildFinished:
    index: int
    result: dict[str, int | float]


@dataclass
class ChildFailed:
    index: int
    error: BaseException


@dataclass
class StreamingInferenceTelemetry:
    """Observability for the LC3-style shared inference queue."""

    wire_requests: int = 0
    positions: int = 0
    batches: int = 0
    wait_seconds: float = 0.0
    inference_seconds: float = 0.0
    max_positions: int = 0
    max_wire_requests: int = 0
    deadline_flushes: int = 0

    def fill_ratio(self, target_positions: int) -> float:
        if self.batches <= 0 or target_positions <= 0:
            return 0.0
        return self.positions / (self.batches * target_positions)


class StreamingInferenceQueue:
    """Merge evaluator requests from independent searches into GPU batches.

    LC3's useful architectural idea here is that search workers do not own an
    evaluator call. They submit requests to one streaming queue, which gathers
    for a bounded window and then evaluates all compatible requests together.
    This class deliberately does not change search order or labels; it only
    centralises scheduling and exposes queue telemetry.
    """

    def __init__(self, model: torch.nn.Module | dict[int, torch.nn.Module],
                 device: torch.device, precision: str,
                 target_positions: int, window_ms: float):
        self.model = model
        self.device = device
        self.precision = precision
        self.target_positions = max(1, int(target_positions))
        self.window_ms = max(0.0, float(window_ms))
        self.telemetry = StreamingInferenceTelemetry()

    def collect(self, first: GpuRequest, events: queue.Queue,
                handle_control) -> list[GpuRequest]:
        pending = [first]
        pending_positions = first.token_np.shape[0]
        started = time.perf_counter()
        deadline = started + self.window_ms / 1000.0
        hit_deadline = False
        while pending_positions < self.target_positions:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                hit_deadline = True
                break
            try:
                item = events.get(timeout=remaining)
            except queue.Empty:
                hit_deadline = True
                break
            if isinstance(item, GpuRequest):
                pending.append(item)
                pending_positions += item.token_np.shape[0]
            else:
                handle_control(item)
        self.telemetry.wait_seconds += time.perf_counter() - started
        if hit_deadline and pending_positions < self.target_positions:
            self.telemetry.deadline_flushes += 1
        return pending

    def serve(self, pending: list[GpuRequest]) -> list[tuple[list[GpuRequest], float]]:
        grouped: dict[int, list[GpuRequest]] = {}
        for request in pending:
            grouped.setdefault(request.model_id, []).append(request)

        served: list[tuple[list[GpuRequest], float]] = []
        for group in grouped.values():
            positions = sum(request.token_np.shape[0] for request in group)
            elapsed = _serve_request_batch(self.model, self.device, group, self.precision)
            self.telemetry.wire_requests += len(group)
            self.telemetry.positions += positions
            self.telemetry.batches += 1
            self.telemetry.inference_seconds += elapsed
            self.telemetry.max_positions = max(self.telemetry.max_positions, positions)
            self.telemetry.max_wire_requests = max(self.telemetry.max_wire_requests, len(group))
            served.append((group, elapsed))
        return served


def read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        part = stream.read(remaining)
        if not part:
            raise RuntimeError("tetra_cli closed the GPU protocol before a complete frame")
        chunks.append(part)
        remaining -= len(part)
    return b"".join(chunks)


class TacticalValueWrapper(torch.nn.Module):
    """Gate short-horizon top-out predictions into the WDL leaf value.

    The base checkpoint remains untouched.  Only high-confidence tactical
    predictions affect the scalar value; policy logits and auxiliary outputs
    are returned bit-for-bit from the base model.  Keeping the draw mass fixed
    makes this deliberately conservative: the wrapper can re-rank decisive
    probability but cannot manufacture certainty the WDL head did not have.
    """

    SELF_TOPOUT_INDICES = (6, 10, 14, 18)
    OPP_TOPOUT_INDICES = (7, 11, 15, 19)

    def __init__(self, base: torch.nn.Module, weight: float, threshold: float):
        super().__init__()
        self.base = base
        self.weight = float(weight)
        self.threshold = float(threshold)
        self.cfg = getattr(base, "cfg", None)

    def forward(self, tokens, token_mask, actions, action_mask):
        policy_logits, wdl_logits, aux = self.base(tokens, token_mask, actions, action_mask)
        if aux.shape[-1] <= max(self.OPP_TOPOUT_INDICES):
            raise RuntimeError("tactical value requires the interval top-out auxiliary targets")

        # The four targets are disjoint time intervals.  MSE heads are not
        # constrained to [0, 1], so clamp each interval before summing and cap
        # the resulting event probability at one.
        self_raw = aux[..., list(self.SELF_TOPOUT_INDICES)]
        opp_raw = aux[..., list(self.OPP_TOPOUT_INDICES)]
        if bool(getattr(self.base, "topout_aux_logits", False)):
            self_raw = torch.sigmoid(self_raw)
            opp_raw = torch.sigmoid(opp_raw)
        else:
            self_raw = self_raw.clamp(0.0, 1.0)
            opp_raw = opp_raw.clamp(0.0, 1.0)
        self_p = self_raw.sum(-1).clamp(0.0, 1.0)
        opp_p = opp_raw.sum(-1).clamp(0.0, 1.0)
        tactical_scalar = opp_p - self_p
        confidence = torch.maximum(self_p, opp_p)
        denominator = max(1e-6, 1.0 - self.threshold)
        gate = ((confidence - self.threshold) / denominator).clamp(0.0, 1.0)
        blend = (self.weight * gate).clamp(0.0, 1.0)

        wdl_prob = torch.softmax(wdl_logits.float(), dim=-1)
        original_scalar = wdl_prob[..., 0] - wdl_prob[..., 2]
        draw = wdl_prob[..., 1]
        decisive_mass = (1.0 - draw).clamp(0.0, 1.0)
        blended_scalar = (
            (1.0 - blend) * original_scalar + blend * tactical_scalar.float()
        )
        blended_scalar = torch.maximum(
            torch.minimum(blended_scalar, decisive_mass), -decisive_mass
        )
        win = 0.5 * (decisive_mass + blended_scalar)
        loss = 0.5 * (decisive_mass - blended_scalar)
        new_prob = torch.stack([win, draw, loss], dim=-1).clamp_min(1e-7)
        new_prob = new_prob / new_prob.sum(dim=-1, keepdim=True)
        synthetic_wdl_logits = torch.log(new_prob).to(wdl_logits.dtype)
        return policy_logits, synthetic_wdl_logits, aux


def wrap_tactical_value(model: torch.nn.Module, weight: float,
                        threshold: float) -> torch.nn.Module:
    if weight <= 0.0:
        return model
    if not 0.0 < weight <= 1.0:
        raise ValueError("tactical value weight must be in (0, 1]")
    if not 0.0 <= threshold < 1.0:
        raise ValueError("tactical value threshold must be in [0, 1)")
    return TacticalValueWrapper(model, weight, threshold)


class AttackValueWrapper(torch.nn.Module):
    """Add a small, centered short-horizon attack preference to leaf WDL.

    The interval attack target at auxiliary index 4 predicts attack produced in
    the next 0--1 seconds.  It is only a weak ranking signal, so this wrapper is
    deliberately bounded and intended as a search tie-breaker rather than a
    replacement for the win/loss value.  Centering avoids granting every leaf a
    constant positive bonus merely because attack is non-negative.
    """

    ATTACK_AUX_INDEX = 4

    def __init__(self, base: torch.nn.Module, weight: float,
                 center: float, scale: float):
        super().__init__()
        self.base = base
        self.weight = float(weight)
        self.center = float(center)
        self.scale = float(scale)
        self.cfg = getattr(base, "cfg", None)
        self.topout_aux_logits = bool(getattr(base, "topout_aux_logits", False))

    def forward(self, tokens, token_mask, actions, action_mask):
        policy_logits, wdl_logits, aux = self.base(tokens, token_mask, actions, action_mask)
        if aux.shape[-1] <= self.ATTACK_AUX_INDEX:
            raise RuntimeError("attack-value search requires interval attack auxiliary target 4")

        attack = aux[..., self.ATTACK_AUX_INDEX].float()
        signal = torch.tanh((attack - self.center) / self.scale)
        wdl_prob = torch.softmax(wdl_logits.float(), dim=-1)
        original_scalar = wdl_prob[..., 0] - wdl_prob[..., 2]
        draw = wdl_prob[..., 1]
        decisive_mass = (1.0 - draw).clamp(0.0, 1.0)
        shifted_scalar = original_scalar + self.weight * signal
        shifted_scalar = torch.maximum(
            torch.minimum(shifted_scalar, decisive_mass), -decisive_mass
        )
        win = 0.5 * (decisive_mass + shifted_scalar)
        loss = 0.5 * (decisive_mass - shifted_scalar)
        new_prob = torch.stack([win, draw, loss], dim=-1).clamp_min(1e-7)
        new_prob = new_prob / new_prob.sum(dim=-1, keepdim=True)
        synthetic_wdl_logits = torch.log(new_prob).to(wdl_logits.dtype)
        return policy_logits, synthetic_wdl_logits, aux


def wrap_attack_value(model: torch.nn.Module, weight: float,
                      center: float = 0.135, scale: float = 0.1) -> torch.nn.Module:
    if weight <= 0.0:
        return model
    if not 0.0 < weight <= 0.25:
        raise ValueError("attack value weight must be in (0, 0.25]")
    if not np.isfinite(center):
        raise ValueError("attack value center must be finite")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("attack value scale must be finite and positive")
    return AttackValueWrapper(model, weight, center, scale)


class PolicyTemperatureWrapper(torch.nn.Module):
    """Rescale policy logits without changing their argmax or model weights."""

    def __init__(self, base: torch.nn.Module, temperature: float):
        super().__init__()
        self.base = base
        self.temperature = float(temperature)
        self.cfg = getattr(base, "cfg", None)
        self.topout_aux_logits = bool(getattr(base, "topout_aux_logits", False))

    def forward(self, tokens, token_mask, actions, action_mask):
        policy_logits, wdl_logits, aux = self.base(tokens, token_mask, actions, action_mask)
        return policy_logits / self.temperature, wdl_logits, aux


def wrap_policy_temperature(model: torch.nn.Module, temperature: float) -> torch.nn.Module:
    if abs(temperature - 1.0) <= 1e-9:
        return model
    if not 0.0 < temperature <= 4.0:
        raise ValueError("policy temperature must be in (0, 4]")
    return PolicyTemperatureWrapper(model, temperature)


def load_model(path: str, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    elif checkpoint.get("architecture", "transformer") != "transformer":
        # Keep the production GPU bridge architecture-agnostic so ablation
        # checkpoints can generate their own self-play distribution.  Import
        # lazily to avoid making the normal TetraFormer path depend on the
        # experimental model module at import time.
        from ablation_models import load_ablation_checkpoint
        model = load_ablation_checkpoint(path, "cpu")
    else:
        model = TetraFormer(TetraFormerConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
    loss_weights = checkpoint.get("loss_weights", {}) if isinstance(checkpoint, dict) else {}
    model.topout_aux_logits = float(loss_weights.get("topout_aux", 0.0)) > 0.0
    model.to(device)
    model.eval()
    return model


def read_request_frame(proc: subprocess.Popen, index: int = -1) -> GpuRequest:
    """Read one request and own its arrays until a response is written."""
    assert proc.stdout is not None
    model_id = struct.unpack("<I", read_exact(proc.stdout, 4))[0]
    n, tokens, actions = struct.unpack("<III", read_exact(proc.stdout, 12))
    if n <= 0 or tokens <= 0 or actions <= 0:
        raise RuntimeError("invalid dimensions in GPU evaluator frame")

    token_np = np.frombuffer(
        read_exact(proc.stdout, n * tokens * TOKEN_FEATURES * 4), dtype="<f4"
    ).reshape(n, tokens, TOKEN_FEATURES).copy()
    token_mask_np = np.frombuffer(
        read_exact(proc.stdout, n * tokens * 4), dtype="<f4"
    ).reshape(n, tokens).copy()
    action_np = np.frombuffer(
        read_exact(proc.stdout, n * actions * ACTION_FEATURES * 4), dtype="<f4"
    ).reshape(n, actions, ACTION_FEATURES).copy()
    action_mask_np = np.frombuffer(
        read_exact(proc.stdout, n * actions * 4), dtype="<f4"
    ).reshape(n, actions).copy()
    return GpuRequest(index, proc, model_id, token_np, token_mask_np, action_np, action_mask_np)


def _select_model(model: TetraFormer | dict[int, TetraFormer], model_id: int) -> TetraFormer:
    if isinstance(model, dict):
        try:
            return model[model_id]
        except KeyError as exc:
            raise RuntimeError(f"unknown GPU evaluator model id: {model_id}") from exc
    if model_id != 0:
        raise RuntimeError(f"unexpected GPU evaluator model id: {model_id}")
    return model


def _serve_request_batch(models: TetraFormer | dict[int, TetraFormer],
                         device: torch.device, requests: list[GpuRequest],
                         precision: str) -> float:
    """Infer several wire requests together, padding only their rectangular edges."""
    if not requests:
        return 0.0

    total = sum(r.token_np.shape[0] for r in requests)
    max_tokens = max(r.token_np.shape[1] for r in requests)
    max_actions = max(r.action_np.shape[1] for r in requests)
    token_np = np.zeros((total, max_tokens, TOKEN_FEATURES), dtype=np.float32)
    token_mask_np = np.zeros((total, max_tokens), dtype=np.float32)
    action_np = np.zeros((total, max_actions, ACTION_FEATURES), dtype=np.float32)
    action_mask_np = np.zeros((total, max_actions), dtype=np.float32)
    offsets: list[tuple[GpuRequest, int, int]] = []
    offset = 0
    for request in requests:
        count = request.token_np.shape[0]
        token_np[offset:offset + count, :request.token_np.shape[1]] = request.token_np
        token_mask_np[offset:offset + count, :request.token_mask_np.shape[1]] = request.token_mask_np
        action_np[offset:offset + count, :request.action_np.shape[1]] = request.action_np
        action_mask_np[offset:offset + count, :request.action_mask_np.shape[1]] = request.action_mask_np
        offsets.append((request, offset, count))
        offset += count

    token = torch.from_numpy(token_np).to(device)
    token_mask = torch.from_numpy(token_mask_np).to(device)
    action = torch.from_numpy(action_np).to(device)
    action_mask = torch.from_numpy(action_mask_np).to(device)
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    autocast_enabled = device.type == "cuda" and autocast_dtype is not None
    # A mixed request batch is only useful when all requests select the same
    # checkpoint. The caller groups model IDs before reaching this function.
    active_model = _select_model(models, requests[0].model_id)
    t0 = time.perf_counter()
    with torch.inference_mode():
        if autocast_enabled:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                logits, wdl, aux = active_model(token, token_mask, action, action_mask)
        else:
            logits, wdl, aux = active_model(token, token_mask, action, action_mask)
        # Keep probability arithmetic and the wire format in fp32 even when
        # the transformer itself runs under fp16/bf16 autocast.
        policy = torch.softmax(logits.float(), dim=-1)
        wdl = torch.softmax(wdl.float(), dim=-1)
        aux = aux.float()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    policy_np = policy.detach().cpu().numpy()
    wdl_np = wdl.detach().cpu().numpy()
    aux_np = aux.detach().cpu().numpy()

    for request, start, count in offsets:
        assert request.proc.stdin is not None
        payload: list[bytes] = [struct.pack("<4sI", b"TGPR", count)]
        for i in range(start, start + count):
            actions = int(np.count_nonzero(action_mask_np[i] >= 0.5))
            payload.append(struct.pack("<I", actions))
            payload.append(np.asarray(policy_np[i, :actions], dtype="<f4").tobytes())
            payload.append(np.asarray(wdl_np[i, :3], dtype="<f4").tobytes())
            payload.append(np.asarray(aux_np[i, :4], dtype="<f4").tobytes())
        request.proc.stdin.write(b"".join(payload))
        request.proc.stdin.flush()
    return elapsed


def answer_request(model: TetraFormer | dict[int, TetraFormer], device: torch.device,
                   proc: subprocess.Popen, precision: str = "fp32") -> float:
    """Read one C++ evaluator request, run it on the GPU, and answer it."""
    request = read_request_frame(proc)
    return _serve_request_batch(model, device, [request], precision)


def serve_game(model: TetraFormer, device: torch.device, engine: str, pieces: int,
               sims: int, batch_size: int, seed: int, determinizations: int,
               use_gumbel: bool, precision: str) -> tuple[dict[str, int | float], float]:
    proc = subprocess.Popen(
        [engine, "gpu-play-protocol", str(pieces), str(sims), str(batch_size), str(seed),
         str(max(1, determinizations)), "1" if use_gumbel else "0"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None

    inference_seconds = 0.0
    try:
        while True:
            magic = read_exact(proc.stdout, 4)
            if magic == RESULT_MAGIC:
                payload = read_exact(proc.stdout, struct.calcsize("<7IIfqQQ"))
                values = struct.unpack("<7IIfqQQ", payload)
                result = {
                    "pieces": values[0],
                    "cleared": values[1],
                    "garbage_cleared": values[2],
                    "sent": values[3],
                    "received": values[4],
                    "survived": values[5],
                    "topout": values[6],
                    "tick_rate": values[7],
                    "outcome": values[8],
                    "duration": values[9],
                    "positions": values[10],
                    "batches": values[11],
                }
                break
            if magic != REQUEST_MAGIC:
                raise RuntimeError(f"unexpected GPU protocol frame: {magic!r}")
            inference_seconds += answer_request(model, device, proc, precision)
    except Exception:
        proc.kill()
        proc.wait()
        raise

    proc.stdin.close()
    return_code = proc.wait()
    error = proc.stderr.read().decode("utf-8", errors="replace").strip()
    if return_code != 0:
        raise RuntimeError(f"tetra_cli GPU child failed with {return_code}: {error}")
    result["inference_seconds"] = inference_seconds
    result["gpu_server_positions"] = result["positions"]
    result["gpu_server_batches"] = result["batches"]
    return result, inference_seconds


def _read_game_result(proc: subprocess.Popen) -> dict[str, int | float]:
    assert proc.stdout is not None
    values = struct.unpack("<7IIfqQQ", read_exact(proc.stdout, struct.calcsize("<7IIfqQQ")))
    return {
        "pieces": values[0],
        "cleared": values[1],
        "garbage_cleared": values[2],
        "sent": values[3],
        "received": values[4],
        "survived": values[5],
        "topout": values[6],
        "tick_rate": values[7],
        "outcome": values[8],
        "duration": values[9],
        "positions": values[10],
        "batches": values[11],
    }


def serve_games_parallel(model: TetraFormer, device: torch.device, engine: str,
                         games: int, pieces: int, sims: int, batch_size: int,
                         seed: int, determinizations: int, use_gumbel: bool,
                         precision: str, workers: int, window_ms: float) -> list[dict[str, int | float]]:
    """Serve several C++ games through a short-window GPU micro-batcher.

    A single search cannot always fill its requested batch, especially with
    Gumbel rounds. Independent games provide the missing parallelism without
    changing any C++ search decision or replay label.
    """
    total_games = max(1, games)
    max_workers = max(1, min(total_games, workers))
    requests: queue.Queue[GpuRequest | ChildFinished | ChildFailed] = queue.Queue()
    results: list[dict[str, int | float] | None] = [None] * total_games
    inference_by_game = [0.0] * total_games
    batcher = StreamingInferenceQueue(
        model, device, precision,
        target_positions=max(1, batch_size * max_workers),
        window_ms=window_ms,
    )
    processes: dict[int, subprocess.Popen] = {}
    threads: list[threading.Thread] = []
    next_index = 0
    running = 0
    finished = 0

    def launch(index: int) -> None:
        nonlocal running
        proc = subprocess.Popen(
            [engine, "gpu-play-protocol", str(pieces), str(sims), str(batch_size),
             str(seed + index), str(max(1, determinizations)), "1" if use_gumbel else "0"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0,
        )
        processes[index] = proc
        running += 1

        def reader() -> None:
            try:
                while True:
                    assert proc.stdout is not None
                    magic = read_exact(proc.stdout, 4)
                    if magic == REQUEST_MAGIC:
                        requests.put(read_request_frame(proc, index))
                        continue
                    if magic == RESULT_MAGIC:
                        result = _read_game_result(proc)
                        return_code = proc.wait()
                        error = proc.stderr.read().decode("utf-8", errors="replace").strip()
                        if return_code != 0:
                            raise RuntimeError(
                                f"tetra_cli GPU child failed with {return_code}: {error}"
                            )
                        requests.put(ChildFinished(index, result))
                        return
                    raise RuntimeError(f"unexpected GPU protocol frame: {magic!r}")
            except BaseException as exc:
                requests.put(ChildFailed(index, exc))

        thread = threading.Thread(target=reader, name=f"gpu-game-reader-{index}", daemon=True)
        thread.start()
        threads.append(thread)

    while next_index < max_workers:
        launch(next_index)
        next_index += 1

    def handle_control(item) -> None:
        nonlocal finished, running, next_index
        if isinstance(item, ChildFailed):
            raise item.error
        if not isinstance(item, ChildFinished):
            raise RuntimeError(f"unexpected GPU queue control item: {type(item).__name__}")
        results[item.index] = item.result
        finished += 1
        running -= 1
        if next_index < total_games:
            launch(next_index)
            next_index += 1

    try:
        while finished < total_games:
            first = requests.get()
            if not isinstance(first, GpuRequest):
                handle_control(first)
                continue

            pending = batcher.collect(first, requests, handle_control)
            for group, elapsed in batcher.serve(pending):
                group_total = sum(r.token_np.shape[0] for r in group)
                for request in group:
                    inference_by_game[request.index] += elapsed * (
                        request.token_np.shape[0] / max(1, group_total)
                    )
    except BaseException:
        for proc in processes.values():
            if proc.poll() is None:
                proc.kill()
        raise
    finally:
        for thread in threads:
            thread.join(timeout=1.0)

    completed = []
    for index, result in enumerate(results):
        if result is None:
            raise RuntimeError(f"GPU game {index} did not return a result")
        result["inference_seconds"] = inference_by_game[index]
        result["gpu_server_positions"] = batcher.telemetry.positions
        result["gpu_server_batches"] = batcher.telemetry.batches
        result["gpu_queue_requests"] = batcher.telemetry.wire_requests
        result["gpu_queue_wait_ms"] = batcher.telemetry.wait_seconds * 1000.0
        result["gpu_queue_deadline_flushes"] = batcher.telemetry.deadline_flushes
        result["gpu_queue_fill_ratio"] = batcher.telemetry.fill_ratio(batcher.target_positions)
        completed.append(result)
    return completed


def report(results: list[dict[str, int | float]], device: torch.device, model_path: str,
           policy_temperature: float = 1.0) -> None:
    print(f"model        {model_path}")
    print(f"policy_T     {policy_temperature:g}")
    print(f"device       {torch.cuda.get_device_name(device) if device.type == 'cuda' else device}")
    print(f"games        {len(results)}")
    print("game  pieces  cleared  gclear  sent  received  seconds  PPS   APM   APP     VS  result")

    total_pieces = total_cleared = total_garbage_cleared = total_sent = total_received = 0
    total_seconds = total_inference = 0.0
    total_positions = total_batches = 0
    for i, r in enumerate(results):
        seconds = float(r["duration"]) / max(1, int(r["tick_rate"]))
        pieces = int(r["pieces"])
        sent = int(r["sent"])
        garbage_cleared = int(r["garbage_cleared"])
        pps = pieces / seconds if seconds > 0 else 0.0
        apm = sent * 60.0 / seconds if seconds > 0 else 0.0
        app = sent / pieces if pieces > 0 else 0.0
        vs = ((sent + garbage_cleared) / pieces) * pps * 100.0 if pieces > 0 else 0.0
        outcome = float(r["outcome"])
        if outcome > 0.5:
            result_name = "win"
        elif outcome < -0.5:
            result_name = f"loss/topout#{int(r['topout'])}"
        else:
            result_name = "draw/truncated"
        print(f"{i:4d}  {pieces:7d}  {int(r['cleared']):7d}  {garbage_cleared:6d}  {sent:4d}  "
              f"{int(r['received']):8d}  {seconds:7.2f}  {pps:4.1f}  {apm:5.1f}  "
              f"{app:5.3f}  {vs:5.1f}  {result_name}")
        total_pieces += pieces
        total_cleared += int(r["cleared"])
        total_garbage_cleared += garbage_cleared
        total_sent += sent
        total_received += int(r["received"])
        total_seconds += seconds
        total_inference += float(r["inference_seconds"])
        total_positions += int(r["positions"])
        total_batches += int(r["batches"])

    print("\naggregate")
    print(f"  pieces       {total_pieces}")
    print(f"  cleared      {total_cleared}")
    print(f"  gclear       {total_garbage_cleared}")
    print(f"  sent         {total_sent}")
    print(f"  received     {total_received}")
    print(f"  seconds      {total_seconds:.2f}")
    print(f"  PPS          {total_pieces / total_seconds if total_seconds else 0.0:.2f}")
    print(f"  APM          {total_sent * 60.0 / total_seconds if total_seconds else 0.0:.2f}")
    print(f"  APP          {total_sent / total_pieces if total_pieces else 0.0:.3f}")
    print(f"  VS           {100.0 * (total_sent + total_garbage_cleared) / total_seconds if total_seconds else 0.0:.2f}")
    print(f"  GPU infer    {total_inference:.2f}s / {total_positions} positions / "
          f"{total_batches} batches")
    print(f"  mean batch   {total_positions / total_batches if total_batches else 0.0:.2f}")
    if results and int(results[0].get("gpu_server_batches", 0)) > 0:
        server_positions = int(results[0]["gpu_server_positions"])
        server_batches = int(results[0]["gpu_server_batches"])
        print(f"  GPU microbatch {server_positions} positions / {server_batches} forwards "
              f"(mean {server_positions / server_batches:.2f})")
        print(
            f"  queue        requests={int(results[0].get('gpu_queue_requests', 0))} "
            f"wait={float(results[0].get('gpu_queue_wait_ms', 0.0)):.1f}ms "
            f"deadline_flushes={int(results[0].get('gpu_queue_deadline_flushes', 0))} "
            f"fill={float(results[0].get('gpu_queue_fill_ratio', 0.0)):.3f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--engine", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--pieces", type=int, default=100)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--determinizations", type=int, default=1,
                    help="root futures averaged per move; 1 is the low-latency default")
    search_mode = ap.add_mutually_exclusive_group()
    search_mode.add_argument("--gumbel", dest="gumbel", action="store_true",
                             help="use Gumbel sequential halving (default)")
    search_mode.add_argument("--puct", dest="gumbel", action="store_false",
                             help="use batched PUCT instead of Gumbel")
    ap.set_defaults(gumbel=True)
    ap.add_argument("--policy-temperature", type=float, default=1.0,
                    help="temperature applied to policy logits before search; below 1 sharpens the prior")
    ap.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16",
                    help="GPU inference arithmetic; fp16 is the default low-latency mode")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel C++ games feeding the GPU micro-batcher; 0 selects up to 4")
    ap.add_argument("--batch-window-ms", type=float, default=2.0,
                    help="maximum time to wait for another game request")
    args = ap.parse_args()

    if not 0.0 < args.policy_temperature <= 4.0:
        raise SystemExit("--policy-temperature must be in (0, 4]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("GPU requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    model = wrap_policy_temperature(load_model(args.checkpoint, device), args.policy_temperature)

    root = Path(__file__).resolve().parents[1]
    engine = args.engine or str(root / ("build/tetra_cli.exe" if os.name == "nt" else "build/tetra_cli"))
    if not os.path.exists(engine):
        raise SystemExit(f"engine not found: {engine}; run make tools first")

    games = max(1, args.games)
    workers = min(games, 4) if args.workers <= 0 else max(1, args.workers)
    if games > 1 and workers > 1:
        results = serve_games_parallel(
            model, device, engine, games, max(1, args.pieces), max(1, args.sims),
            max(1, args.batch), args.seed, max(1, args.determinizations), args.gumbel,
            args.precision, workers, max(0.0, args.batch_window_ms)
        )
    else:
        results = []
        for i in range(games):
            result, _ = serve_game(
                model, device, engine, max(1, args.pieces), max(1, args.sims),
                max(1, args.batch), args.seed + i, max(1, args.determinizations),
                args.gumbel, args.precision
            )
            results.append(result)
    report(results, device, args.checkpoint, args.policy_temperature)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
