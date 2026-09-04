# SPDX-License-Identifier: MIT
"""Parallel GPU self-play dataset generation.

Several independent C++ ``gpu-export-protocol`` children feed one PyTorch model.
Requests arriving within a short window are padded and evaluated together, using
the same GPU micro-batcher as :mod:`gpu_match`.  Each child writes a separate
``.tetradat`` shard; pass all shards to ``train.py`` rather than byte-concatenating
them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import queue
import struct
import subprocess
import threading
from pathlib import Path

import torch

from gpu_match import (
    EXPORT_MAGIC,
    REQUEST_MAGIC,
    RESULT_MAGIC,
    AsyncStreamingInferenceDispatcher,
    ChildFailed,
    ChildStderrDrainer,
    GpuRequest,
    StreamingInferenceQueue,
    StreamingInferenceTelemetry,
    load_model,
    read_exact,
    read_request_frame,
    wrap_policy_temperature,
)
from gpu_selfplay import EXPORT_FORMAT, GAME_FORMAT


@dataclass
class ShardFinished:
    index: int
    path: str
    results: list[dict[str, int | float]]
    summary: dict[str, int]


def _game_result(stream) -> dict[str, int | float]:
    values = struct.unpack(GAME_FORMAT, read_exact(stream, struct.calcsize(GAME_FORMAT)))
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


def _shard_path(output: str, index: int, shard_count: int) -> str:
    path = Path(output)
    if shard_count <= 1:
        return str(path)
    suffix = path.suffix or ".tetradat"
    stem = path.name[:-len(path.suffix)] if path.suffix else path.name
    return str(path.with_name(f"{stem}.part{index:02d}{suffix}"))


def generate_parallel(
    model: torch.nn.Module,
    device: torch.device,
    engine: str,
    output: str,
    games: int,
    pieces: int,
    sims: int,
    batch_size: int,
    seed: int,
    model_version: int,
    determinizations: int,
    use_gumbel: bool,
    root_noise_fraction: float,
    precision: str,
    workers: int,
    window_ms: float,
    target_positions: int = 512,
    inflight_batches: int = 2,
    gpu_workers: int = 2,
    enable_timing_actions: bool = False,
    no_attack_delivery: bool = False,
) -> tuple[
    list[dict[str, int | float]],
    list[dict[str, int | str]],
    StreamingInferenceTelemetry,
]:
    total_games = max(1, games)
    worker_count = max(1, min(total_games, workers))
    base = total_games // worker_count
    extra = total_games % worker_count
    games_per_worker = [base + (1 if i < extra else 0) for i in range(worker_count)]

    events: queue.Queue[GpuRequest | ShardFinished | ChildFailed] = queue.Queue()
    processes: dict[int, subprocess.Popen] = {}
    threads: list[threading.Thread] = []
    shard_meta: list[dict[str, int | str] | None] = [None] * worker_count
    all_results: list[dict[str, int | float]] = []
    batcher = StreamingInferenceQueue(
        model, device, precision,
        target_positions=max(1, min(batch_size * worker_count, target_positions)),
        window_ms=window_ms,
    )
    dispatcher = AsyncStreamingInferenceDispatcher(
        batcher,
        events,
        max_queued_batches=max(1, inflight_batches),
        gpu_workers=max(1, gpu_workers),
    )

    offset = 0
    for index, shard_games in enumerate(games_per_worker):
        shard = _shard_path(output, index, worker_count)
        Path(shard).parent.mkdir(parents=True, exist_ok=True)
        shard_seed = seed + offset
        offset += shard_games
        try:
            proc = subprocess.Popen(
                [
                    engine,
                    "gpu-export-protocol",
                    shard,
                    str(shard_games),
                    str(pieces),
                    str(sims),
                    str(batch_size),
                    str(shard_seed),
                    str(model_version),
                    str(max(1, determinizations)),
                    "1" if use_gumbel else "0",
                    f"{root_noise_fraction:.9g}",
                    "1" if enable_timing_actions else "0",
                    "1" if no_attack_delivery else "0",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except BaseException:
            for running_proc in processes.values():
                if running_proc.poll() is None:
                    running_proc.kill()
            for running_proc in processes.values():
                running_proc.wait()
            dispatcher.close(raise_worker_error=False)
            for thread in threads:
                thread.join(timeout=1.0)
            raise
        processes[index] = proc
        stderr = ChildStderrDrainer(
            proc, f"gpu-selfplay-stderr-drain-{index}"
        )

        def reader(idx: int = index, p: subprocess.Popen = proc,
                   pth: str = shard, expected_games: int = shard_games,
                   first_seed: int = shard_seed,
                   diagnostics: ChildStderrDrainer = stderr) -> None:
            results: list[dict[str, int | float]] = []
            try:
                assert p.stdout is not None
                assert p.stdin is not None
                assert p.stderr is not None
                while True:
                    magic = read_exact(p.stdout, 4)
                    if magic == REQUEST_MAGIC:
                        events.put(read_request_frame(p, idx))
                        continue
                    if magic == RESULT_MAGIC:
                        results.append(_game_result(p.stdout))
                        continue
                    if magic == EXPORT_MAGIC:
                        values = struct.unpack(
                            EXPORT_FORMAT,
                            read_exact(p.stdout, struct.calcsize(EXPORT_FORMAT)),
                        )
                        summary = {
                            "games": values[0],
                            "samples": values[1],
                            "positions": values[2],
                            "batches": values[3],
                            "seed": first_seed,
                            "expected_games": expected_games,
                        }
                        p.stdin.close()
                        return_code = p.wait()
                        error = diagnostics.finish()
                        if return_code != 0:
                            raise RuntimeError(
                                f"tetra_cli GPU self-play shard {idx} failed with "
                                f"{return_code}: {error}"
                            )
                        events.put(ShardFinished(idx, pth, results, summary))
                        return
                    raise RuntimeError(
                        f"unexpected GPU self-play protocol frame from shard {idx}: {magic!r}"
                    )
            except BaseException as exc:
                events.put(ChildFailed(idx, exc))

        thread = threading.Thread(
            target=reader, name=f"gpu-selfplay-reader-{index}", daemon=True
        )
        thread.start()
        threads.append(thread)

    finished = 0

    def handle_control(item) -> None:
        nonlocal finished
        if isinstance(item, ChildFailed):
            raise item.error
        if not isinstance(item, ShardFinished):
            raise RuntimeError(
                f"unexpected self-play inference queue control item: {type(item).__name__}"
            )
        shard_meta[item.index] = {"path": item.path, **item.summary}
        all_results.extend(item.results)
        finished += 1

    failed = False
    try:
        while finished < worker_count:
            first = events.get()
            if not isinstance(first, GpuRequest):
                handle_control(first)
                continue

            pending = batcher.collect(first, events, handle_control)
            dispatcher.submit(pending)
    except BaseException:
        failed = True
        for proc in processes.values():
            if proc.poll() is None:
                proc.kill()
        for proc in processes.values():
            proc.wait()
        raise
    finally:
        dispatcher.close(raise_worker_error=not failed)
        for thread in threads:
            thread.join(timeout=1.0)

    completed_meta = [m for m in shard_meta if m is not None]
    if len(completed_meta) != worker_count:
        raise RuntimeError("not every GPU self-play shard completed")
    return all_results, completed_meta, batcher.telemetry


def report(results: list[dict[str, int | float]], shards: list[dict[str, int | str]],
           telemetry: StreamingInferenceTelemetry,
           device: torch.device, model_path: str,
           target_positions: int) -> None:
    print(f"model        {model_path}")
    print(f"device       {torch.cuda.get_device_name(device) if device.type == 'cuda' else device}")
    print(f"games        {len(results)}")
    print(f"shards       {len(shards)}")
    total_samples = sum(int(s["samples"]) for s in shards)
    total_pieces = sum(int(r["pieces"]) for r in results)
    total_sent = sum(int(r["sent"]) for r in results)
    total_garbage_cleared = sum(int(r["garbage_cleared"]) for r in results)
    total_ticks = sum(int(r["duration"]) for r in results)
    tick_rate = int(results[0]["tick_rate"]) if results else 1
    seconds = total_ticks / max(1, tick_rate)
    print(f"samples      {total_samples}")
    print(f"pieces       {total_pieces}")
    print(f"game_seconds {seconds:.2f}")
    print(f"PPS          {total_pieces / seconds if seconds else 0.0:.2f}")
    print(f"APM          {total_sent * 60.0 / seconds if seconds else 0.0:.2f}")
    print(f"APP          {total_sent / total_pieces if total_pieces else 0.0:.3f}")
    print(f"VS           {100.0 * (total_sent + total_garbage_cleared) / seconds if seconds else 0.0:.2f}")
    print(f"GPU infer    {telemetry.inference_seconds:.2f}s / {telemetry.positions} positions / {telemetry.batches} batches")
    print(f"mean batch   {telemetry.positions / max(1, telemetry.batches):.2f}")
    print(
        f"queue        requests={telemetry.wire_requests} "
        f"wait={telemetry.wait_seconds * 1000.0:.1f}ms "
        f"deadline_flushes={telemetry.deadline_flushes} "
        f"fill={telemetry.fill_ratio(target_positions):.3f} "
        f"max_positions={telemetry.max_positions} max_requests={telemetry.max_wire_requests}"
    )
    print("dataset shards")
    for shard in sorted(shards, key=lambda x: str(x["path"])):
        print(
            f"  {shard['path']} games={shard['games']} samples={shard['samples']} "
            f"seed={shard['seed']}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("output", help="base output name; multiple workers create .partNN shards")
    ap.add_argument("--engine", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--pieces", type=int, default=300)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model-version", type=int, default=1)
    ap.add_argument("--determinizations", type=int, default=2)
    ap.add_argument("--no-gumbel", action="store_true")
    ap.add_argument("--policy-temperature", type=float, default=1.0,
                    help="temperature applied to policy logits before search; below 1 sharpens the prior")
    ap.add_argument("--root-noise-fraction", type=float, default=0.25,
                    help="Dirichlet-style root exploration mix; use 0 for teacher/distillation data")
    ap.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    ap.add_argument("--timing-actions", action="store_true",
                    help="branch FASTEST vs WAIT_FOR_EVENT when pending garbage exists")
    ap.add_argument("--no-attack-delivery", action="store_true",
                    help="compute and record attacks but do not deliver them to the opponent")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument(
        "--batch-window-ms", type=float, default=20.0,
        help="micro-batching window across independent C++ exporter children",
    )
    ap.add_argument(
        "--target-positions", type=int, default=256,
        help="cap the shared GPU batch target so many workers can prepare the next batch asynchronously",
    )
    ap.add_argument(
        "--inflight-batches", type=int, default=2,
        help="bounded number of GPU batches queued ahead of the active inference batch",
    )
    ap.add_argument(
        "--gpu-workers", type=int, default=2,
        help="concurrent ROCm/CUDA inference streams serving independent request batches",
    )
    args = ap.parse_args()

    if not 0.0 <= args.root_noise_fraction <= 1.0:
        raise SystemExit("--root-noise-fraction must be in [0, 1]")
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

    results, shards, telemetry = generate_parallel(
        model,
        device,
        engine,
        args.output,
        max(1, args.games),
        max(1, args.pieces),
        max(0, args.sims),
        max(1, args.batch),
        args.seed,
        max(0, args.model_version),
        max(1, args.determinizations),
        not args.no_gumbel,
        args.root_noise_fraction,
        args.precision,
        max(1, args.workers),
        max(0.0, args.batch_window_ms),
        max(1, args.target_positions),
        max(1, args.inflight_batches),
        max(1, args.gpu_workers),
        args.timing_actions,
        args.no_attack_delivery,
    )
    print(f"policy_T     {args.policy_temperature:g}")
    report(
        results, shards, telemetry, device, args.checkpoint,
        target_positions=max(
            1,
            min(
                args.batch * max(1, min(max(1, args.games), args.workers)),
                max(1, args.target_positions),
            ),
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
