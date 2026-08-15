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
import sys
from pathlib import Path

import torch

from gpu_match import EXPORT_MAGIC, REQUEST_MAGIC, RESULT_MAGIC, answer_request
from gpu_match import load_model, read_exact, wrap_attack_value, wrap_policy_temperature


GAME_FORMAT = "<7IIfqQQ"
EXPORT_FORMAT = "<IIQQ"


def generate(model: torch.nn.Module, device: torch.device, engine: str, output: str,
             games: int, pieces: int, sims: int, batch_size: int, seed: int,
             model_version: int, determinizations: int,
             use_gumbel: bool, root_noise_fraction: float,
             precision: str, enable_timing_actions: bool = False,
             no_attack_delivery: bool = False
             ) -> tuple[list[dict[str, int | float]], dict[str, int], float]:
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
            f"{root_noise_fraction:.9g}",
            "1" if enable_timing_actions else "0",
            "1" if no_attack_delivery else "0",
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
    if error:
        print(error, file=sys.stderr, flush=True)
    return results, summary, inference_seconds


def report(results: list[dict[str, int | float]], summary: dict[str, int],
           inference_seconds: float, device: torch.device, model_path: str,
           output: str) -> None:
    print(f"model        {model_path}")
    print(f"device       {torch.cuda.get_device_name(device) if device.type == 'cuda' else device}")
    print(f"dataset      {output}")
    print(f"games        {len(results)}")
    print("game  pieces  cleared  gclear  sent  received  seconds  PPS   APM   APP     VS  result")

    total_pieces = total_cleared = total_garbage_cleared = total_sent = total_received = 0
    total_seconds = 0.0
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

    print("\nexport")
    print(f"  samples      {summary.get('samples', 0)}")
    print(f"  cleared      {total_cleared}")
    print(f"  gclear       {total_garbage_cleared}")
    print(f"  sent         {total_sent}")
    print(f"  received     {total_received}")
    print(f"  seconds      {total_seconds:.2f}")
    print(f"  PPS          {total_pieces / total_seconds if total_seconds else 0.0:.2f}")
    print(f"  APM          {total_sent * 60.0 / total_seconds if total_seconds else 0.0:.2f}")
    print(f"  APP          {total_sent / total_pieces if total_pieces else 0.0:.3f}")
    print(f"  VS           {100.0 * (total_sent + total_garbage_cleared) / total_seconds if total_seconds else 0.0:.2f}")
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
    ap.add_argument("--policy-temperature", type=float, default=1.0,
                    help="temperature applied to policy logits before search; below 1 sharpens the prior")
    ap.add_argument("--attack-value-weight", type=float, default=0.0,
                    help="bounded WDL tie-break from predicted 0-1s attack (0 disables)")
    ap.add_argument("--attack-value-center", type=float, default=0.135,
                    help="center of the predicted attack signal before tanh scaling")
    ap.add_argument("--attack-value-scale", type=float, default=0.1,
                    help="positive tanh scale for the predicted attack signal")
    ap.add_argument("--root-noise-fraction", type=float, default=0.25,
                    help="Dirichlet-style root exploration mix; use 0 for teacher/distillation data")
    ap.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    ap.add_argument("--timing-actions", action="store_true",
                    help="branch FASTEST vs WAIT_FOR_EVENT when pending garbage exists")
    ap.add_argument("--no-attack-delivery", action="store_true",
                    help="benchmark/curriculum mode: compute attacks but do not deliver them between players")
    ap.add_argument("--factor-timing-policy", action="store_true",
                    help="preserve base-placement prior mass across FASTEST/WAIT variants")
    ap.add_argument("--timing-wait-bias", type=float, default=0.0,
                    help="diagnostic WAIT logit bias used with --factor-timing-policy")
    args = ap.parse_args()

    if not 0.0 <= args.root_noise_fraction <= 1.0:
        raise SystemExit("--root-noise-fraction must be in [0, 1]")
    if not 0.0 < args.policy_temperature <= 4.0:
        raise SystemExit("--policy-temperature must be in (0, 4]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("GPU requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    if abs(args.timing_wait_bias) > 1e-12 and not args.factor_timing_policy:
        raise SystemExit("--timing-wait-bias requires --factor-timing-policy")
    if args.factor_timing_policy:
        cfg = getattr(model, "cfg", None)
        if (cfg is None or not hasattr(cfg, "factor_timing_policy") or
                not hasattr(cfg, "timing_wait_logit_bias")):
            raise SystemExit("checkpoint architecture does not support timing factorization")
        cfg.factor_timing_policy = True
        cfg.timing_wait_logit_bias = float(args.timing_wait_bias)
    try:
        model = wrap_attack_value(
            model, args.attack_value_weight,
            args.attack_value_center, args.attack_value_scale,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    model = wrap_policy_temperature(model, args.policy_temperature)

    root = Path(__file__).resolve().parents[1]
    engine = args.engine or str(root / ("build/tetra_cli.exe" if os.name == "nt" else "build/tetra_cli"))
    if not os.path.exists(engine):
        raise SystemExit(f"engine not found: {engine}; run make tools first")

    results, summary, inference_seconds = generate(
        model, device, engine, args.output, max(1, args.games), max(1, args.pieces),
        max(0, args.sims), max(1, args.batch), args.seed, max(0, args.model_version),
        max(1, args.determinizations), not args.no_gumbel,
        args.root_noise_fraction, args.precision, args.timing_actions,
        args.no_attack_delivery
    )
    print(f"policy_T     {args.policy_temperature:g}")
    if args.attack_value_weight > 0.0:
        print(
            f"attack_value  weight={args.attack_value_weight:g} "
            f"center={args.attack_value_center:g} scale={args.attack_value_scale:g}"
        )
    report(results, summary, inference_seconds, device, args.checkpoint, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
