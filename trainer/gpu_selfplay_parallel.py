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
import time
from pathlib import Path

import torch

from gpu_match import (
    EXPORT_MAGIC,
    REQUEST_MAGIC,
    RESULT_MAGIC,
    ChildFailed,
    GpuRequest,
    _serve_request_batch,
    load_model,
    read_exact,
    read_request_frame,
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
        "sent": values[2],
        "received": values[3],
        "survived": values[4],
        "topout": values[5],
        "tick_rate": values[6],
        "outcome": values[7],
        "duration": values[8],
        "positions": values[9],
        "batches": values[10],
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
) -> tuple[list[dict[str, int | float]], list[dict[str, int | str]], float, int, int]:
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
    inference_seconds = 0.0
    server_positions = 0
    server_batches = 0

    offset = 0
    for index, shard_games in enumerate(games_per_worker):
        shard = _shard_path(output, index, worker_count)
        Path(shard).parent.mkdir(parents=True, exist_ok=True)
        shard_seed = seed + offset
        offset += shard_games
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
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        processes[index] = proc

        def reader(idx: int = index, p: subprocess.Popen = proc,
                   pth: str = shard, expected_games: int = shard_games,
                   first_seed: int = shard_seed) -> None:
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
                        error = p.stderr.read().decode("utf-8", errors="replace").strip()
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
    try:
        while finished < worker_count:
            first = events.get()
            if isinstance(first, ChildFailed):
                raise first.error
            if isinstance(first, ShardFinished):
                shard_meta[first.index] = {
                    "path": first.path,
                    **first.summary,
                }
                all_results.extend(first.results)
                finished += 1
                continue

            pending = [first]
            pending_positions = first.token_np.shape[0]
            deadline = time.perf_counter() + max(0.0, window_ms) / 1000.0
            target_positions = max(1, batch_size * worker_count)
            while pending_positions < target_positions:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                try:
                    item = events.get(timeout=remaining)
                except queue.Empty:
                    break
                if isinstance(item, GpuRequest):
                    pending.append(item)
                    pending_positions += item.token_np.shape[0]
                elif isinstance(item, ShardFinished):
                    shard_meta[item.index] = {"path": item.path, **item.summary}
                    all_results.extend(item.results)
                    finished += 1
                else:
                    raise item.error

            grouped: dict[int, list[GpuRequest]] = {}
            for request in pending:
                grouped.setdefault(request.model_id, []).append(request)
            for group in grouped.values():
                positions = sum(r.token_np.shape[0] for r in group)
                server_positions += positions
                server_batches += 1
                inference_seconds += _serve_request_batch(model, device, group, precision)
    except BaseException:
        for proc in processes.values():
            if proc.poll() is None:
                proc.kill()
        raise
    finally:
        for thread in threads:
            thread.join(timeout=1.0)

    completed_meta = [m for m in shard_meta if m is not None]
    if len(completed_meta) != worker_count:
        raise RuntimeError("not every GPU self-play shard completed")
    return all_results, completed_meta, inference_seconds, server_positions, server_batches


def report(results: list[dict[str, int | float]], shards: list[dict[str, int | str]],
           inference_seconds: float, server_positions: int, server_batches: int,
           device: torch.device, model_path: str) -> None:
    print(f"model        {model_path}")
    print(f"device       {torch.cuda.get_device_name(device) if device.type == 'cuda' else device}")
    print(f"games        {len(results)}")
    print(f"shards       {len(shards)}")
    total_samples = sum(int(s["samples"]) for s in shards)
    total_pieces = sum(int(r["pieces"]) for r in results)
    total_sent = sum(int(r["sent"]) for r in results)
    total_ticks = sum(int(r["duration"]) for r in results)
    tick_rate = int(results[0]["tick_rate"]) if results else 1
    seconds = total_ticks / max(1, tick_rate)
    print(f"samples      {total_samples}")
    print(f"pieces       {total_pieces}")
    print(f"game_seconds {seconds:.2f}")
    print(f"PPS          {total_pieces / seconds if seconds else 0.0:.2f}")
    print(f"APM          {total_sent * 60.0 / seconds if seconds else 0.0:.2f}")
    print(f"GPU infer    {inference_seconds:.2f}s / {server_positions} positions / {server_batches} batches")
    print(f"mean batch   {server_positions / max(1, server_batches):.2f}")
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
    ap.add_argument("--root-noise-fraction", type=float, default=0.25,
                    help="Dirichlet-style root exploration mix; use 0 for teacher/distillation data")
    ap.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--batch-window-ms", type=float, default=2.0,
        help="micro-batching window across independent C++ exporter children",
    )
    args = ap.parse_args()

    if not 0.0 <= args.root_noise_fraction <= 1.0:
        raise SystemExit("--root-noise-fraction must be in [0, 1]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("GPU requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    root = Path(__file__).resolve().parents[1]
    engine = args.engine or str(root / ("build/tetra_cli.exe" if os.name == "nt" else "build/tetra_cli"))
    if not os.path.exists(engine):
        raise SystemExit(f"engine not found: {engine}; run make tools first")

    results, shards, inference_seconds, positions, batches = generate_parallel(
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
    )
    report(results, shards, inference_seconds, positions, batches, device, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
