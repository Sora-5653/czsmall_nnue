# SPDX-License-Identifier: MIT
"""Parallel selective Reanalyse over several dataset shards.

Independent C++ ``gpu-reanalyse-protocol`` children reconstruct/search their
own historical roots while one PyTorch teacher and one shared streaming GPU
queue serve all evaluator requests.  Outputs remain non-destructive: every
source produces a separate refreshed shard, audit JSONL, and provenance
manifest, published only after all workers complete successfully.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import queue
import struct
import subprocess
import tempfile
import threading
from pathlib import Path

import torch

from gpu_match import (
    REQUEST_MAGIC,
    AsyncStreamingInferenceDispatcher,
    ChildFailed,
    ChildStderrDrainer,
    GpuRequest,
    StreamingInferenceQueue,
    StreamingInferenceTelemetry,
    load_model,
    read_exact,
    read_request_frame,
)
from reanalyze import DONE_FORMAT, DONE_MAGIC, engine_commit, sha256_file
from tetra_dataset import load as load_dataset


@dataclass
class ReanalyseFinished:
    index: int
    summary: dict[str, int]
    diagnostics: str


@dataclass
class Job:
    index: int
    source: Path
    output: Path
    audit: Path
    manifest: Path
    temp_output: Path
    temp_audit: Path
    source_rows: int
    original_model_version: int
    select_count: int
    proc: subprocess.Popen | None = None


def _shard_path(base: Path, index: int, count: int) -> Path:
    if count <= 1:
        return base
    suffix = base.suffix or ".tetradat"
    stem = base.name[:-len(base.suffix)] if base.suffix else base.name
    return base.with_name(f"{stem}.part{index:02d}{suffix}")


def _make_temp(parent: Path, suffix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=parent, suffix=suffix)
    os.close(fd)
    return Path(name)


def run_parallel(
    model: torch.nn.Module,
    device: torch.device,
    precision: str,
    engine: Path,
    checkpoint: Path,
    sources: list[Path],
    output_base: Path,
    select_fraction: float,
    original_sims: int,
    sims: int,
    batch: int,
    determinizations: int,
    use_gumbel: bool,
    no_attack_delivery: bool,
    timing_actions: bool,
    teacher_model_version: int,
    workers: int,
    window_ms: float,
    target_positions: int,
    inflight_batches: int,
    gpu_workers: int,
    root: Path,
) -> tuple[list[Path], StreamingInferenceTelemetry]:
    worker_count = max(1, min(len(sources), workers))
    if worker_count != len(sources):
        # Keep implementation simple and deterministic: the caller should
        # provide no more source shards than desired concurrent workers.
        # iterate.py does exactly that for self-play/Reanalyse parity.
        raise ValueError("parallel Reanalyse currently requires workers >= number of source shards")

    jobs: list[Job] = []
    for index, source in enumerate(sources):
        loaded = load_dataset(str(source))
        source_rows = len(loaded)
        original_model_version = int(loaded.header.model_version)
        del loaded
        select_count = max(1, round(source_rows * select_fraction))
        select_count = min(source_rows, select_count)
        if select_count >= source_rows:
            raise ValueError(
                f"selection for {source} would refresh every row; reduce --select-fraction"
            )
        output = _shard_path(output_base, index, len(sources))
        audit = output.with_suffix(output.suffix + ".audit.jsonl")
        manifest = output.with_suffix(output.suffix + ".reanalyze.json")
        for path in (output, audit, manifest):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite existing Reanalyse artifact: {path}")
        jobs.append(Job(
            index=index,
            source=source,
            output=output,
            audit=audit,
            manifest=manifest,
            temp_output=_make_temp(output.parent, ".tetradat"),
            temp_audit=_make_temp(output.parent, ".audit.jsonl"),
            source_rows=source_rows,
            original_model_version=original_model_version,
            select_count=select_count,
        ))

    events: queue.Queue[GpuRequest | ReanalyseFinished | ChildFailed] = queue.Queue()
    threads: list[threading.Thread] = []
    telemetry_queue = StreamingInferenceQueue(
        model,
        device,
        precision,
        target_positions=max(1, min(batch * worker_count, target_positions)),
        window_ms=max(0.0, window_ms),
    )
    dispatcher = AsyncStreamingInferenceDispatcher(
        telemetry_queue,
        events,
        max_queued_batches=max(1, inflight_batches),
        gpu_workers=max(1, gpu_workers),
    )
    finished: list[ReanalyseFinished | None] = [None] * len(jobs)

    def launch(job: Job) -> None:
        proc = subprocess.Popen(
            [
                str(engine), "gpu-reanalyse-protocol", str(job.source),
                str(job.temp_output), str(job.temp_audit), str(job.select_count),
                str(sims), str(batch), str(determinizations),
                "1" if use_gumbel else "0",
                "1" if no_attack_delivery else "0",
                "1" if timing_actions else "0",
                str(teacher_model_version),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        job.proc = proc
        stderr = ChildStderrDrainer(
            proc, f"gpu-reanalyse-stderr-drain-{job.index}"
        )

        def reader() -> None:
            try:
                assert proc.stdout is not None and proc.stdin is not None and proc.stderr is not None
                while True:
                    magic = read_exact(proc.stdout, 4)
                    if magic == REQUEST_MAGIC:
                        events.put(read_request_frame(proc, job.index))
                        continue
                    if magic == DONE_MAGIC:
                        values = struct.unpack(
                            DONE_FORMAT,
                            read_exact(proc.stdout, struct.calcsize(DONE_FORMAT)),
                        )
                        summary = {
                            "source_rows": values[0],
                            "selected_rows": values[1],
                            "token_rows_verified": values[2],
                            "action_rows_verified": values[3],
                            "positions_evaluated": values[4],
                            "batches_issued": values[5],
                        }
                        proc.stdin.close()
                        return_code = proc.wait()
                        diagnostics = stderr.finish()
                        if return_code != 0:
                            raise RuntimeError(
                                f"reanalyse shard {job.index} failed with {return_code}: {diagnostics}"
                            )
                        events.put(ReanalyseFinished(job.index, summary, diagnostics))
                        return
                    raise RuntimeError(
                        f"unexpected Reanalyse protocol frame from shard {job.index}: {magic!r}"
                    )
            except BaseException as exc:
                events.put(ChildFailed(job.index, exc))

        thread = threading.Thread(
            target=reader,
            name=f"gpu-reanalyse-reader-{job.index}",
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    completed = 0

    def handle_control(item) -> None:
        nonlocal completed
        if isinstance(item, ChildFailed):
            raise item.error
        if not isinstance(item, ReanalyseFinished):
            raise RuntimeError(
                f"unexpected Reanalyse queue control item: {type(item).__name__}"
            )
        finished[item.index] = item
        completed += 1

    failed = False
    try:
        for job in jobs:
            launch(job)

        while completed < len(jobs):
            first = events.get()
            if not isinstance(first, GpuRequest):
                handle_control(first)
                continue
            pending = telemetry_queue.collect(first, events, handle_control)
            dispatcher.submit(pending)

        for job, result in zip(jobs, finished):
            if result is None:
                raise RuntimeError(f"Reanalyse shard {job.index} did not return a result")
            summary = result.summary
            if summary["source_rows"] != job.source_rows:
                raise RuntimeError(f"source row mismatch for shard {job.index}: {summary}")
            if summary["selected_rows"] != job.select_count:
                raise RuntimeError(f"selected row mismatch for shard {job.index}: {summary}")
            if (summary["token_rows_verified"] != job.source_rows or
                    summary["action_rows_verified"] != job.source_rows):
                raise RuntimeError(f"reconstruction parity incomplete for shard {job.index}: {summary}")

        # Publish only after every worker has passed all validation.  This keeps
        # a failed generation retryable without leaving a half-published set.
        created = datetime.now(timezone.utc).isoformat()
        commit = engine_commit(root)
        checkpoint_hash = sha256_file(checkpoint)
        for job, result in zip(jobs, finished):
            assert result is not None
            os.replace(job.temp_output, job.output)
            os.replace(job.temp_audit, job.audit)
            record = {
                "format": "tetraformer-reanalyse-v1",
                "created_utc": created,
                "parallel_group_size": len(jobs),
                "parallel_worker_index": job.index,
                "source_dataset": str(job.source),
                "source_sha256": sha256_file(job.source),
                "source_rows": job.source_rows,
                "original_model_version": job.original_model_version,
                "original_search_budget": {"simulations": original_sims},
                "teacher_checkpoint": str(checkpoint),
                "teacher_checkpoint_sha256": checkpoint_hash,
                "teacher_model_version": teacher_model_version,
                "engine": str(engine),
                "engine_commit": commit,
                "rules": {
                    "attack_delivery": not no_attack_delivery,
                    "timing_actions": timing_actions,
                },
                "selection": {
                    "method": "top historical-policy KL vs current raw policy",
                    "requested_rows": job.select_count,
                    "fraction": job.select_count / job.source_rows,
                },
                "reanalysis_search": {
                    "simulations": sims,
                    "batch": batch,
                    "determinizations": determinizations,
                    "algorithm": "gumbel" if use_gumbel else "puct",
                    "root_noise_fraction": 0.0,
                },
                "label_contract": {
                    "refreshed": ["policy_target", "search_value_in_audit"],
                    "preserved": ["chosen_action", "terminal_value_target", "aux_targets"],
                    "note": (
                        "rectangular v4 has no search-value training column; the refreshed "
                        "search value is retained losslessly in the audit sidecar"
                    ),
                },
                "output_dataset": str(job.output),
                "output_sha256": sha256_file(job.output),
                "audit_jsonl": str(job.audit),
                "audit_sha256": sha256_file(job.audit),
                "summary": result.summary,
                "child_diagnostics": result.diagnostics,
            }
            job.manifest.write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    except BaseException:
        failed = True
        for job in jobs:
            if job.proc is not None and job.proc.poll() is None:
                job.proc.kill()
        for job in jobs:
            if job.proc is not None:
                job.proc.wait()
        raise
    finally:
        dispatcher.close(raise_worker_error=not failed)
        for thread in threads:
            thread.join(timeout=1.0)
        for job in jobs:
            job.temp_output.unlink(missing_ok=True)
            job.temp_audit.unlink(missing_ok=True)

    return [job.output for job in jobs], telemetry_queue.telemetry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("output", help="base output name; several sources create .partNN shards")
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--engine", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    ap.add_argument("--select-fraction", type=float, default=0.05)
    ap.add_argument("--original-sims", type=int, required=True)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--determinizations", type=int, default=2)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--batch-window-ms", type=float, default=20.0)
    ap.add_argument("--target-positions", type=int, default=256)
    ap.add_argument("--inflight-batches", type=int, default=2)
    ap.add_argument("--gpu-workers", type=int, default=2)
    ap.add_argument("--no-gumbel", action="store_true")
    ap.add_argument("--no-attack-delivery", action="store_true")
    ap.add_argument("--timing-actions", action="store_true")
    ap.add_argument("--teacher-model-version", type=int, default=0)
    args = ap.parse_args()

    if not 0.0 < args.select_fraction < 1.0:
        raise SystemExit("--select-fraction must be in (0, 1)")
    if args.original_sims < 1 or args.sims <= args.original_sims:
        raise SystemExit("--sims must be materially larger than positive --original-sims")
    if args.workers < len(args.sources):
        raise SystemExit("--workers must be at least the number of source shards")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("GPU requested but torch.cuda.is_available() is false")

    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    sources = [Path(source).resolve() for source in args.sources]
    if not checkpoint.is_file() or any(not source.is_file() for source in sources):
        raise SystemExit("checkpoint and every source dataset must exist")
    engine = Path(args.engine).resolve() if args.engine else root / (
        "build/tetra_cli.exe" if os.name == "nt" else "build/tetra_cli"
    )
    if not engine.is_file():
        raise SystemExit(f"engine not found: {engine}; run make tools first")

    device = torch.device(args.device)
    model = load_model(str(checkpoint), device)
    outputs, telemetry = run_parallel(
        model=model,
        device=device,
        precision=args.precision,
        engine=engine,
        checkpoint=checkpoint,
        sources=sources,
        output_base=output,
        select_fraction=args.select_fraction,
        original_sims=args.original_sims,
        sims=args.sims,
        batch=max(1, args.batch),
        determinizations=max(1, args.determinizations),
        use_gumbel=not args.no_gumbel,
        no_attack_delivery=args.no_attack_delivery,
        timing_actions=args.timing_actions,
        teacher_model_version=args.teacher_model_version,
        workers=max(1, args.workers),
        window_ms=max(0.0, args.batch_window_ms),
        target_positions=max(1, args.target_positions),
        inflight_batches=max(1, args.inflight_batches),
        gpu_workers=max(1, args.gpu_workers),
        root=root,
    )

    print(f"teacher      {checkpoint}")
    print(f"sources      {len(sources)}")
    print(f"outputs      {len(outputs)}")
    print(f"workers      {len(sources)}")
    print(f"search       {args.original_sims} -> {args.sims} simulations, root noise 0")
    print(
        f"GPU infer    {telemetry.inference_seconds:.2f}s / {telemetry.positions} positions / "
        f"{telemetry.batches} forwards"
    )
    print(
        f"queue        requests={telemetry.wire_requests} "
        f"wait={telemetry.wait_seconds * 1000.0:.1f}ms "
        f"deadline_flushes={telemetry.deadline_flushes} "
        f"fill={telemetry.fill_ratio(max(1, args.batch * len(sources))):.3f} "
        f"max_positions={telemetry.max_positions} max_requests={telemetry.max_wire_requests}"
    )
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
