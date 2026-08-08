# SPDX-License-Identifier: MIT
"""Run the paired Candidate-vs-Champion Arena with GPU inference.

The C++ child remains the authority for the simulator, mirrored paired games,
and promotion arithmetic.  It tags each evaluator request with model 0
(candidate) or model 1 (champion); this process keeps both PyTorch checkpoints
resident on the selected GPU and serves the requests over one binary pipe.
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
from pathlib import Path

import torch

from gpu_match import ARENA_MAGIC, REQUEST_MAGIC, answer_request, load_model, read_exact


ARENA_FORMAT = "<4I3fI"


def evaluate(candidate: torch.nn.Module, champion: torch.nn.Module, device: torch.device,
             engine: str, pairs: int, sims: int, pieces: int, batch_size: int,
             determinizations: int, use_gumbel: bool, precision: str, seed: int,
             candidate_sims: int = -1, champion_sims: int = -1,
             candidate_gumbel: int = -1, champion_gumbel: int = -1,
             gumbel_c_scale: float = 0.01
             ) -> tuple[dict[str, int | float], float]:
    proc = subprocess.Popen(
        [
            engine,
            "gpu-arena-protocol",
            str(max(1, pairs)),
            str(max(1, sims)),
            str(max(1, pieces)),
            str(max(1, batch_size)),
            str(max(1, determinizations)),
            "1" if use_gumbel else "0",
            str(max(0, seed)),
            str(candidate_sims),
            str(champion_sims),
            str(candidate_gumbel),
            str(champion_gumbel),
            f"{gumbel_c_scale:.9g}",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None

    models = {0: candidate, 1: champion}
    inference_seconds = 0.0
    try:
        while True:
            magic = read_exact(proc.stdout, 4)
            if magic == REQUEST_MAGIC:
                inference_seconds += answer_request(models, device, proc, precision)
                continue
            if magic == ARENA_MAGIC:
                values = struct.unpack(
                    ARENA_FORMAT, read_exact(proc.stdout, struct.calcsize(ARENA_FORMAT))
                )
                result = {
                    "games_played": values[0],
                    "candidate_wins": values[1],
                    "champion_wins": values[2],
                    "draws": values[3],
                    "win_rate": values[4],
                    "ci_lower": values[5],
                    "ci_upper": values[6],
                    "promoted": bool(values[7]),
                }
                break
            raise RuntimeError(f"unexpected GPU Arena protocol frame: {magic!r}")
    except Exception:
        proc.kill()
        proc.wait()
        raise

    proc.stdin.close()
    return_code = proc.wait()
    error = proc.stderr.read().decode("utf-8", errors="replace").strip()
    if return_code != 0:
        raise RuntimeError(f"tetra_cli GPU Arena child failed with {return_code}: {error}")
    return result, inference_seconds


def report(result: dict[str, int | float], candidate: str, champion: str,
           inference_seconds: float, pairs: int, sims: int, pieces: int,
           determinizations: int, use_gumbel: bool, precision: str, seed: int,
           candidate_sims: int = -1, champion_sims: int = -1,
           candidate_gumbel: int = -1, champion_gumbel: int = -1,
           gumbel_c_scale: float = 0.01) -> None:
    games = int(result["games_played"])
    print(f"Arena: Candidate ({candidate}) vs Champion ({champion})")
    candidate_budget = sims if candidate_sims < 0 else candidate_sims
    champion_budget = sims if champion_sims < 0 else champion_sims
    candidate_uses_gumbel = use_gumbel if candidate_gumbel < 0 else candidate_gumbel != 0
    champion_uses_gumbel = use_gumbel if champion_gumbel < 0 else champion_gumbel != 0
    print(
        f"Running {pairs} paired games ({games} games total, "
        f"sims={candidate_budget}/{champion_budget} (candidate/champion), "
        f"max_pieces={pieces}, determinizations={determinizations}, "
        f"gumbel={'on' if candidate_uses_gumbel else 'off'}/"
        f"{'on' if champion_uses_gumbel else 'off'} (candidate/champion), "
        f"gumbel_c_scale={gumbel_c_scale:g}, precision={precision}, seed={seed})...\n"
    )
    print(f"Result over {games} games:")
    print(f"  Candidate wins : {int(result['candidate_wins'])}")
    print(f"  Champion wins  : {int(result['champion_wins'])}")
    print(f"  Draws          : {int(result['draws'])}")
    print(
        f"  Win Rate       : {float(result['win_rate']) * 100.0:.1f}% "
        f"(95% CI: {float(result['ci_lower']) * 100.0:.1f}% - "
        f"{float(result['ci_upper']) * 100.0:.1f}%)"
    )
    print("  Threshold      : 55.0%")
    print(f"  Status         : {'PROMOTED' if result['promoted'] else 'RETAINED (no promotion)'}")
    print(f"  GPU infer      : {inference_seconds:.2f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("champion")
    ap.add_argument("--engine", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--candidate-sims", type=int, default=-1,
                    help="diagnostic override; 0 means policy-only")
    ap.add_argument("--champion-sims", type=int, default=-1,
                    help="diagnostic override; 0 means policy-only")
    ap.add_argument("--pieces", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--determinizations", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42,
                    help="base seed for paired games; vary it for independent Arena trials")
    ap.add_argument("--gumbel", action="store_true")
    ap.add_argument("--candidate-gumbel", action="store_true",
                    help="diagnostic override: enable Gumbel only for the candidate")
    ap.add_argument("--champion-gumbel", action="store_true",
                    help="diagnostic override: enable Gumbel only for the champion")
    ap.add_argument("--gumbel-c-scale", type=float, default=0.01,
                    help="diagnostic scale on Gumbel root Q ranking")
    ap.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    args = ap.parse_args()

    if args.gumbel_c_scale < 0.0:
        raise SystemExit("--gumbel-c-scale must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("GPU requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    candidate = load_model(args.candidate, device)
    champion = load_model(args.champion, device)

    root = Path(__file__).resolve().parents[1]
    engine = args.engine or str(root / ("build/tetra_cli.exe" if os.name == "nt" else "build/tetra_cli"))
    if not os.path.exists(engine):
        raise SystemExit(f"engine not found: {engine}; run make tools first")

    result, inference_seconds = evaluate(
        candidate, champion, device, engine, max(1, args.pairs), max(1, args.sims),
        max(1, args.pieces), max(1, args.batch), max(1, args.determinizations),
        args.gumbel, args.precision, max(0, args.seed),
        args.candidate_sims, args.champion_sims,
        1 if args.candidate_gumbel else -1, 1 if args.champion_gumbel else -1,
        args.gumbel_c_scale
    )
    report(
        result, args.candidate, args.champion, inference_seconds, max(1, args.pairs),
        max(1, args.sims), max(1, args.pieces), max(1, args.determinizations), args.gumbel,
        args.precision, max(0, args.seed), args.candidate_sims, args.champion_sims,
        1 if args.candidate_gumbel else -1, 1 if args.champion_gumbel else -1,
        args.gumbel_c_scale
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
