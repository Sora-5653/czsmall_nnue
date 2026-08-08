# SPDX-License-Identifier: MIT
"""Generate a compact self-play dataset with PyTorch inference on the GPU.

The C++ child owns the rules, Cobra move generation, search and dataset
serialization. This process owns the checkpoint and serves the same batched
GPU evaluator protocol as :mod:`gpu_match`.

Example::

    python trainer/gpu_selfplay.py models/gen1.pt data/gen2.tetradat \
        --device cuda --games 32 --pieces 300 --sims 64 --model-version 2
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
from pathlib import Path

import torch

from gpu_match import EXPORT_MAGIC, REQUEST_MAGIC, RESULT_MAGIC, answer_request
from gpu_match import load_model, read_exact


GAME_FORMAT = "<6IIfqQQ"
EXPORT_FORMAT = "<IIQQ"


def generate(model: torch.nn.Module, device: torch.device, engine: str, output: str,
             games: int, pieces: int, sims: int, batch_size: int, seed: int,
             model_version: int, determinizations: int,
             use_gumbel: bool, precision: str) -> tuple[list[dict[str, int | float]], dict[str, int], float]:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            engine,
            "gpu-export-protocol",
            str(output_path),
            str(games),
            str(pieces),
            str(sims),
            str(batch_size),
            str(seed),
            str(model_version),
            str(max(1, determinizations)),
            "1" if use_gumbel else "0",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None

    results: list[dict[str, int | float]] = []
    summary: dict[str, int] = {}
    inference_seconds = 0.0
    try:
        while True:
            magic = read_exact(proc.stdout, 4)
            if magic == REQUEST_MAGIC:
                inference_seconds += answer_request(model, device, proc, precision)
                continue
            if magic == RESULT_MAGIC:
                values = struct.unpack(GAME_FORMAT, read_exact(proc.stdout, struct.calcsize(GAME_FORMAT)))
                results.append({
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
                })
                continue
            if magic == EXPORT_MAGIC:
                values = struct.unpack(EXPORT_FORMAT, read_exact(proc.stdout, struct.calcsize(EXPORT_FORMAT)))
                summary = {
                    "games": values[0],
                    "samples": values[1],
                    "positions": values[2],
                    "batches": values[3],
                }
                break
            raise RuntimeError(f"unexpected GPU self-play protocol frame: {magic!r}")
    except Exception:
        proc.kill()
        proc.wait()
        raise

    proc.stdin.close()
    return_code = proc.wait()
    error = proc.stderr.read().decode("utf-8", errors="replace").strip()
    if return_code != 0:
        raise RuntimeError(f"tetra_cli GPU self-play child failed with {return_code}: {error}")
    return results, summary, inference_seconds


def report(results: list[dict[str, int | float]], summary: dict[str, int],
           inference_seconds: float, device: torch.device, model_path: str,
           output: str) -> None:
    print(f"model        {model_path}")
    print(f"device       {torch.cuda.get_device_name(device) if device.type == 'cuda' else device}")
    print(f"dataset      {output}")
    print(f"games        {len(results)}")
    print("game  pieces  cleared  sent  received  seconds  PPS   APM   APP  result")

    total_pieces = total_cleared = total_sent = total_received = 0
    total_seconds = 0.0
    for i, r in enumerate(results):
        seconds = float(r["duration"]) / max(1, int(r["tick_rate"]))
        pieces = int(r["pieces"])
        sent = int(r["sent"])
        pps = pieces / seconds if seconds > 0 else 0.0
        apm = sent * 60.0 / seconds if seconds > 0 else 0.0
        app = sent / pieces if pieces > 0 else 0.0
        outcome = float(r["outcome"])
        if outcome > 0.5:
            result_name = "win"
        elif outcome < -0.5:
            result_name = f"loss/topout#{int(r['topout'])}"
        else:
            result_name = "draw/truncated"
        print(f"{i:4d}  {pieces:7d}  {int(r['cleared']):7d}  {sent:4d}  "
              f"{int(r['received']):8d}  {seconds:7.2f}  {pps:4.1f}  {apm:5.1f}  "
              f"{app:5.3f}  {result_name}")
        total_pieces += pieces
        total_cleared += int(r["cleared"])
        total_sent += sent
        total_received += int(r["received"])
        total_seconds += seconds

    print("\nexport")
    print(f"  samples      {summary.get('samples', 0)}")
    print(f"  cleared      {total_cleared}")
    print(f"  sent         {total_sent}")
    print(f"  received     {total_received}")
    print(f"  seconds      {total_seconds:.2f}")
    print(f"  PPS          {total_pieces / total_seconds if total_seconds else 0.0:.2f}")
    print(f"  APM          {total_sent * 60.0 / total_seconds if total_seconds else 0.0:.2f}")
    print(f"  APP          {total_sent / total_pieces if total_pieces else 0.0:.3f}")
    print(f"  GPU infer    {inference_seconds:.2f}s / {summary.get('positions', 0)} positions / "
          f"{summary.get('batches', 0)} batches")
    print(f"  mean batch   {summary.get('positions', 0) / summary.get('batches', 1):.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("output")
    ap.add_argument("--engine", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--pieces", type=int, default=300)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model-version", type=int, default=1)
    ap.add_argument("--determinizations", type=int, default=2,
                    help="root futures averaged per move; self-play defaults to 2")
    ap.add_argument("--no-gumbel", action="store_true",
                    help="use batched PUCT instead of Gumbel sequential halving")
    ap.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("GPU requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    root = Path(__file__).resolve().parents[1]
    engine = args.engine or str(root / ("build/tetra_cli.exe" if os.name == "nt" else "build/tetra_cli"))
    if not os.path.exists(engine):
        raise SystemExit(f"engine not found: {engine}; run make tools first")

    results, summary, inference_seconds = generate(
        model, device, engine, args.output, max(1, args.games), max(1, args.pieces),
        max(1, args.sims), max(1, args.batch), args.seed, max(0, args.model_version),
        max(1, args.determinizations), not args.no_gumbel, args.precision
    )
    report(results, summary, inference_seconds, device, args.checkpoint, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
