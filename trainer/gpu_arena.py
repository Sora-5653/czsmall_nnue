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

from gpu_match import (
    ARENA_MAGIC,
    REQUEST_MAGIC,
    answer_request,
    load_model,
    read_exact,
    wrap_policy_temperature,
    wrap_tactical_value,
)


ARENA_FORMAT = "<4I29fI"


def evaluate(candidate: torch.nn.Module, champion: torch.nn.Module, device: torch.device,
             engine: str, pairs: int, sims: int, pieces: int, batch_size: int,
             determinizations: int, use_gumbel: bool, precision: str, seed: int,
             candidate_sims: int = -1, champion_sims: int = -1,
             candidate_gumbel: int = -1, champion_gumbel: int = -1,
             gumbel_c_scale: float = 0.01, gumbel_noise_scale: float = 0.05,
             candidate_timing_actions: int = -1, champion_timing_actions: int = -1,
             candidate_gumbel_noise_scale: float = -1.0,
             champion_gumbel_noise_scale: float = -1.0
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
            f"{gumbel_noise_scale:.9g}",
            str(candidate_timing_actions),
            str(champion_timing_actions),
            f"{candidate_gumbel_noise_scale:.9g}",
            f"{champion_gumbel_noise_scale:.9g}",
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
                    "candidate_vs": values[7],
                    "champion_vs": values[8],
                    "candidate_apm": values[9],
                    "champion_apm": values[10],
                    "candidate_app": values[11],
                    "champion_app": values[12],
                    "candidate_pps": values[13],
                    "champion_pps": values[14],
                    "candidate_avg_pieces": values[15],
                    "champion_avg_pieces": values[16],
                    "candidate_avg_seconds": values[17],
                    "champion_avg_seconds": values[18],
                    "candidate_survival_rate": values[19],
                    "champion_survival_rate": values[20],
                    "candidate_sent_per_game": values[21],
                    "champion_sent_per_game": values[22],
                    "candidate_garbage_cleared_per_game": values[23],
                    "champion_garbage_cleared_per_game": values[24],
                    "candidate_received_per_game": values[25],
                    "champion_received_per_game": values[26],
                    "candidate_blockout_rate": values[27],
                    "champion_blockout_rate": values[28],
                    "candidate_lockout_rate": values[29],
                    "champion_lockout_rate": values[30],
                    "candidate_garbageout_rate": values[31],
                    "champion_garbageout_rate": values[32],
                    "promoted": bool(values[33]),
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
           gumbel_c_scale: float = 0.01, gumbel_noise_scale: float = 0.05,
           candidate_timing_actions: int = -1, champion_timing_actions: int = -1,
           candidate_tactical_value_weight: float = 0.0,
           champion_tactical_value_weight: float = 0.0,
           candidate_tactical_value_threshold: float = 0.7,
           champion_tactical_value_threshold: float = 0.7,
           candidate_policy_temperature: float = 1.0,
           champion_policy_temperature: float = 1.0,
           candidate_gumbel_noise_scale: float = -1.0,
           champion_gumbel_noise_scale: float = -1.0) -> None:
    games = int(result["games_played"])
    print(f"Arena: Candidate ({candidate}) vs Champion ({champion})")
    candidate_budget = sims if candidate_sims < 0 else candidate_sims
    champion_budget = sims if champion_sims < 0 else champion_sims
    candidate_uses_gumbel = use_gumbel if candidate_gumbel < 0 else candidate_gumbel != 0
    champion_uses_gumbel = use_gumbel if champion_gumbel < 0 else champion_gumbel != 0
    candidate_uses_timing = False if candidate_timing_actions < 0 else candidate_timing_actions != 0
    champion_uses_timing = False if champion_timing_actions < 0 else champion_timing_actions != 0
    candidate_noise = (gumbel_noise_scale if candidate_gumbel_noise_scale < 0.0
                       else candidate_gumbel_noise_scale)
    champion_noise = (gumbel_noise_scale if champion_gumbel_noise_scale < 0.0
                      else champion_gumbel_noise_scale)
    print(
        f"Running {pairs} paired games ({games} games total, "
        f"sims={candidate_budget}/{champion_budget} (candidate/champion), "
        f"max_pieces={pieces}, determinizations={determinizations}, "
        f"gumbel={'on' if candidate_uses_gumbel else 'off'}/"
        f"{'on' if champion_uses_gumbel else 'off'} (candidate/champion), "
        f"gumbel_c_scale={gumbel_c_scale:g}, "
        f"gumbel_noise_scale={candidate_noise:g}/{champion_noise:g} (candidate/champion), "
        f"timing={'on' if candidate_uses_timing else 'off'}/"
        f"{'on' if champion_uses_timing else 'off'} (candidate/champion), "
        f"tactical_value={candidate_tactical_value_weight:g}@{candidate_tactical_value_threshold:g}/"
        f"{champion_tactical_value_weight:g}@{champion_tactical_value_threshold:g} "
        f"(candidate/champion), policy_T={candidate_policy_temperature:g}/"
        f"{champion_policy_temperature:g} (candidate/champion), "
        f"precision={precision}, seed={seed})...\n"
    )
    print(f"Result over {games} games:")
    print(f"  Candidate wins : {int(result['candidate_wins'])}")
    print(f"  Champion wins  : {int(result['champion_wins'])}")
    print(f"  Draws          : {int(result['draws'])}")
    print(
        f"  VS             : {float(result['candidate_vs']):.1f} / "
        f"{float(result['champion_vs']):.1f} (candidate/champion)"
    )
    print(
        f"  APM / APP / PPS: {float(result['candidate_apm']):.1f} / "
        f"{float(result['candidate_app']):.3f} / {float(result['candidate_pps']):.3f}  vs  "
        f"{float(result['champion_apm']):.1f} / {float(result['champion_app']):.3f} / "
        f"{float(result['champion_pps']):.3f}"
    )
    print(
        f"  Avg pieces/sec : {float(result['candidate_avg_pieces']):.1f} / "
        f"{float(result['candidate_avg_seconds']):.2f}s  vs  "
        f"{float(result['champion_avg_pieces']):.1f} / "
        f"{float(result['champion_avg_seconds']):.2f}s"
    )
    print(
        f"  Survival       : {float(result['candidate_survival_rate']) * 100.0:.1f}% / "
        f"{float(result['champion_survival_rate']) * 100.0:.1f}%"
    )
    print(
        f"  Sent/recv/GC   : {float(result['candidate_sent_per_game']):.2f} / "
        f"{float(result['candidate_received_per_game']):.2f} / "
        f"{float(result['candidate_garbage_cleared_per_game']):.2f}  vs  "
        f"{float(result['champion_sent_per_game']):.2f} / "
        f"{float(result['champion_received_per_game']):.2f} / "
        f"{float(result['champion_garbage_cleared_per_game']):.2f} per game"
    )
    print(
        f"  Topout B/L/G   : {float(result['candidate_blockout_rate']) * 100.0:.1f}% / "
        f"{float(result['candidate_lockout_rate']) * 100.0:.1f}% / "
        f"{float(result['candidate_garbageout_rate']) * 100.0:.1f}%  vs  "
        f"{float(result['champion_blockout_rate']) * 100.0:.1f}% / "
        f"{float(result['champion_lockout_rate']) * 100.0:.1f}% / "
        f"{float(result['champion_garbageout_rate']) * 100.0:.1f}%"
    )
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
    search_mode = ap.add_mutually_exclusive_group()
    search_mode.add_argument("--gumbel", dest="gumbel", action="store_true",
                             help="use Gumbel sequential halving (default)")
    search_mode.add_argument("--puct", dest="gumbel", action="store_false",
                             help="use batched PUCT instead of Gumbel")
    ap.set_defaults(gumbel=True)
    ap.add_argument("--candidate-gumbel", action="store_true",
                    help="diagnostic override: enable Gumbel only for the candidate")
    ap.add_argument("--champion-gumbel", action="store_true",
                    help="diagnostic override: enable Gumbel only for the champion")
    ap.add_argument("--gumbel-c-scale", type=float, default=0.01,
                    help="diagnostic scale on Gumbel root Q ranking")
    ap.add_argument("--gumbel-noise-scale", type=float, default=0.05,
                    help="shared Gumbel root perturbation scale")
    ap.add_argument("--candidate-gumbel-noise-scale", type=float, default=-1.0,
                    help="candidate-only Gumbel noise override; negative inherits shared value")
    ap.add_argument("--champion-gumbel-noise-scale", type=float, default=-1.0,
                    help="champion-only Gumbel noise override; negative inherits shared value")
    ap.add_argument("--candidate-timing-actions", action="store_true",
                    help="enable FASTEST/WAIT_FOR_EVENT timing branches for the candidate")
    ap.add_argument("--champion-timing-actions", action="store_true",
                    help="enable FASTEST/WAIT_FOR_EVENT timing branches for the champion")
    ap.add_argument("--candidate-factor-timing-policy", action="store_true",
                    help="factor candidate FASTEST/WAIT logits without changing checkpoint weights")
    ap.add_argument("--champion-factor-timing-policy", action="store_true",
                    help="factor champion FASTEST/WAIT logits without changing checkpoint weights")
    ap.add_argument("--candidate-timing-wait-bias", type=float, default=0.0,
                    help="diagnostic WAIT logit bias inside candidate factorized timing pairs")
    ap.add_argument("--champion-timing-wait-bias", type=float, default=0.0,
                    help="diagnostic WAIT logit bias inside champion factorized timing pairs")
    ap.add_argument("--candidate-tactical-value-weight", type=float, default=0.0,
                    help="blend high-confidence predicted 8s top-out balance into candidate WDL")
    ap.add_argument("--champion-tactical-value-weight", type=float, default=0.0,
                    help="blend high-confidence predicted 8s top-out balance into champion WDL")
    ap.add_argument("--candidate-tactical-value-threshold", type=float, default=0.7,
                    help="candidate top-out confidence below this value leaves WDL unchanged")
    ap.add_argument("--champion-tactical-value-threshold", type=float, default=0.7,
                    help="champion top-out confidence below this value leaves WDL unchanged")
    ap.add_argument("--candidate-policy-temperature", type=float, default=1.0,
                    help="candidate policy-logit temperature; below 1 sharpens the search prior")
    ap.add_argument("--champion-policy-temperature", type=float, default=1.0,
                    help="champion policy-logit temperature; below 1 sharpens the search prior")
    ap.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    args = ap.parse_args()

    if args.gumbel_c_scale < 0.0:
        raise SystemExit("--gumbel-c-scale must be non-negative")
    if args.gumbel_noise_scale < 0.0:
        raise SystemExit("--gumbel-noise-scale must be non-negative")
    for side, noise in (
        ("candidate", args.candidate_gumbel_noise_scale),
        ("champion", args.champion_gumbel_noise_scale),
    ):
        if noise < -1.0:
            raise SystemExit(f"--{side}-gumbel-noise-scale must be -1 or non-negative")
    for side, weight, threshold in (
        ("candidate", args.candidate_tactical_value_weight,
         args.candidate_tactical_value_threshold),
        ("champion", args.champion_tactical_value_weight,
         args.champion_tactical_value_threshold),
    ):
        if not 0.0 <= weight <= 1.0:
            raise SystemExit(f"--{side}-tactical-value-weight must be in [0, 1]")
        if not 0.0 <= threshold < 1.0:
            raise SystemExit(f"--{side}-tactical-value-threshold must be in [0, 1)")
    for side, temperature in (
        ("candidate", args.candidate_policy_temperature),
        ("champion", args.champion_policy_temperature),
    ):
        if not 0.0 < temperature <= 4.0:
            raise SystemExit(f"--{side}-policy-temperature must be in (0, 4]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("GPU requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    candidate = load_model(args.candidate, device)
    champion = load_model(args.champion, device)
    for model, enabled, wait_bias, side in (
        (candidate, args.candidate_factor_timing_policy,
         args.candidate_timing_wait_bias, "candidate"),
        (champion, args.champion_factor_timing_policy,
         args.champion_timing_wait_bias, "champion"),
    ):
        if not enabled and abs(wait_bias) > 1e-12:
            raise SystemExit(f"--{side}-timing-wait-bias requires --{side}-factor-timing-policy")
        if not enabled:
            continue
        cfg = getattr(model, "cfg", None)
        if (cfg is None or not hasattr(cfg, "factor_timing_policy") or
                not hasattr(cfg, "timing_wait_logit_bias")):
            raise SystemExit(f"{side} checkpoint architecture does not support timing factorization")
        cfg.factor_timing_policy = True
        cfg.timing_wait_logit_bias = float(wait_bias)

    candidate = wrap_policy_temperature(candidate, args.candidate_policy_temperature)
    champion = wrap_policy_temperature(champion, args.champion_policy_temperature)
    candidate = wrap_tactical_value(
        candidate, args.candidate_tactical_value_weight,
        args.candidate_tactical_value_threshold,
    )
    champion = wrap_tactical_value(
        champion, args.champion_tactical_value_weight,
        args.champion_tactical_value_threshold,
    )

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
        args.gumbel_c_scale, args.gumbel_noise_scale,
        1 if args.candidate_timing_actions else -1,
        1 if args.champion_timing_actions else -1,
        args.candidate_gumbel_noise_scale,
        args.champion_gumbel_noise_scale,
    )
    report(
        result, args.candidate, args.champion, inference_seconds, max(1, args.pairs),
        max(1, args.sims), max(1, args.pieces), max(1, args.determinizations), args.gumbel,
        args.precision, max(0, args.seed), args.candidate_sims, args.champion_sims,
        1 if args.candidate_gumbel else -1, 1 if args.champion_gumbel else -1,
        args.gumbel_c_scale, args.gumbel_noise_scale,
        1 if args.candidate_timing_actions else -1,
        1 if args.champion_timing_actions else -1,
        args.candidate_tactical_value_weight,
        args.champion_tactical_value_weight,
        args.candidate_tactical_value_threshold,
        args.champion_tactical_value_threshold,
        args.candidate_policy_temperature,
        args.champion_policy_temperature,
        args.candidate_gumbel_noise_scale,
        args.champion_gumbel_noise_scale,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
