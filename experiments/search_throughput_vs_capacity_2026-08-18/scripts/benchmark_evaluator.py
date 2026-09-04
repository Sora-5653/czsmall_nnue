#!/usr/bin/env python3
"""Benchmark the frozen A/B evaluators on representative token/action states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "trainer"))
from gpu_match import load_model  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_model(
    path: Path, device: torch.device, batches: list[int], states: int, warmup: int
) -> dict[str, object]:
    model = load_model(str(path), device)
    params = sum(parameter.numel() for parameter in model.parameters())
    torch.cuda.reset_peak_memory_stats(device)
    rows: list[dict[str, float | int]] = []
    for batch in batches:
        token_x = torch.rand(batch, 102, 24, device=device)
        token_mask = torch.ones(batch, 102, device=device)
        action_x = torch.rand(batch, 106, 24, device=device)
        action_mask = torch.ones(batch, 106, device=device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            for _ in range(warmup):
                model(token_x, token_mask, action_x, action_mask)
            torch.cuda.synchronize(device)
            repetitions = max(1, (states + batch - 1) // batch)
            samples: list[float] = []
            started = time.perf_counter()
            for _ in range(repetitions):
                call_started = time.perf_counter()
                model(token_x, token_mask, action_x, action_mask)
                torch.cuda.synchronize(device)
                samples.append((time.perf_counter() - call_started) * 1000.0)
            elapsed = time.perf_counter() - started
        samples.sort()
        p50 = samples[len(samples) // 2]
        p95 = samples[min(len(samples) - 1, max(0, int(len(samples) * 0.95) - 1))]
        rows.append({
            "batch": batch,
            "states": repetitions * batch,
            "repetitions": repetitions,
            "median_latency_ms": p50,
            "p95_latency_ms": p95,
            "states_per_second": repetitions * batch / elapsed,
            "elapsed_seconds": elapsed,
        })
        del token_x, token_mask, action_x, action_mask
    peak = torch.cuda.max_memory_allocated(device)
    del model
    torch.cuda.empty_cache()
    return {"path": str(path), "sha256": sha256(path), "parameters": params,
            "peak_memory_bytes": peak, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--states", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batches", nargs="+", type=int,
                        default=(1, 2, 4, 8, 16, 32, 64, 128))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("benchmark requires a visible CUDA/ROCm device")
    index = 0 if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise SystemExit(f"device is unavailable: {device}")
    if args.states < 10000:
        raise SystemExit("HANDOFF requires at least 10,000 measured states per evaluator/batch")
    batches = [max(1, value) for value in args.batches]
    result = {
        "device": torch.cuda.get_device_name(index),
        "device_index": index,
        "states_per_batch_measurement": args.states,
        "autocast": "float16",
        "evaluators": {
            "A": benchmark_model(Path(args.a).resolve(), device, batches, args.states, args.warmup),
            "B": benchmark_model(Path(args.b).resolve(), device, batches, args.states, args.warmup),
        },
    }
    a_rows = {int(row["batch"]): row for row in result["evaluators"]["A"]["rows"]}
    b_rows = {int(row["batch"]): row for row in result["evaluators"]["B"]["rows"]}
    result["throughput_ratio_B_over_A"] = {
        str(batch): float(b_rows[batch]["states_per_second"]) /
        float(a_rows[batch]["states_per_second"])
        for batch in batches
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
