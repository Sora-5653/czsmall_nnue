# SPDX-License-Identifier: MIT
"""Run one guarded AlphaZero-style training generation.

The driver deliberately keeps promotion separate from training:

    champion checkpoint -> GPU self-play -> optional selective Reanalyse
        -> replay/source-aware mix -> GPU train -> Arena (GPU by default)
        -> promote only when the threshold passes

It is local-only.  The Arena remains the authority for promotion, so a lower
validation loss or a high APM/APP number cannot replace a stronger champion.

Example::

    python trainer/iterate.py --champion models/champion.pt \
        --replay data/gpu_train_20260805.tetradat \
        --generation 4 --champion-output models/champion \
        --device cuda --games 16 --pieces 300 --sims 64 \
        --train-steps 5000 --arena-pairs 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path


def run_checked(command: list[str], cwd: Path, capture: bool = False) -> str:
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=str(cwd),
        check=True,
        text=True if capture else None,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return result.stdout or "" if capture else ""


def parse_arena(output: str) -> dict[str, float | int | bool]:
    def number(pattern: str, default: float = 0.0) -> float:
        match = re.search(pattern, output)
        return float(match.group(1)) if match else default

    def integer(pattern: str, default: int = 0) -> int:
        match = re.search(pattern, output)
        return int(match.group(1)) if match else default

    status = re.search(r"Status\s+:\s+(.+)", output)
    promoted = bool(status and "PROMOTED" in status.group(1) and "RETAINED" not in status.group(1))
    return {
        "candidate_wins": integer(r"Candidate wins\s+:\s+(\d+)"),
        "champion_wins": integer(r"Champion wins\s+:\s+(\d+)"),
        "draws": integer(r"Draws\s+:\s+(\d+)"),
        "win_rate": number(r"Win Rate\s+:\s+([0-9.]+)%") / 100.0,
        "ci_lower": number(r"Win Rate\s+:\s+[0-9.]+%\s+\(95% CI:\s+([0-9.]+)%") / 100.0,
        "ci_upper": number(r"95% CI:\s+[0-9.]+%\s+-\s+([0-9.]+)%") / 100.0,
        "promoted": promoted,
    }


def checkpoint_aux_targets(path: Path) -> int:
    code = (
        "import torch,sys; "
        "x=torch.load(sys.argv[1],map_location='cpu',weights_only=False); "
        "cfg=(x.get('config',{}) if isinstance(x,dict) else getattr(x,'cfg',None)); "
        "print(int(cfg.get('aux_targets',-1)) if isinstance(cfg,dict) else "
        "int(getattr(cfg,'aux_targets',-1)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def dataset_aux_targets(path: Path) -> int:
    header = struct.Struct("<8s7IQI")
    with path.open("rb") as stream:
        raw = stream.read(header.size)
    if len(raw) != header.size:
        raise RuntimeError(f"dataset header is truncated: {path}")
    values = header.unpack(raw)
    if values[0] != b"TETRADAT":
        raise RuntimeError(f"invalid dataset magic: {path}")
    return int(values[7])


def parallel_shard_paths(base: Path, count: int) -> list[Path]:
    if count <= 1:
        return [base]
    suffix = base.suffix or ".tetradat"
    stem = base.name[:-len(base.suffix)] if base.suffix else base.name
    return [base.with_name(f"{stem}.part{i:02d}{suffix}") for i in range(count)]


def checkpoint_step(path: Path) -> int:
    # Avoid importing torch in the orchestration process until all subprocesses
    # have completed; this keeps the driver lightweight and launchable from a
    # normal Python environment.
    code = (
        "import torch,sys; "
        "x=torch.load(sys.argv[1],map_location='cpu',weights_only=False); "
        "print(int(x.get('step',0)) if isinstance(x,dict) else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", required=True, help="current champion PyTorch checkpoint")
    ap.add_argument("--champion-output", default="",
                    help="base path receiving promoted .pt and .tetrawts files")
    ap.add_argument("--replay", nargs="*", default=[],
                    help="older .tetradat generations to mix before the new data")
    ap.add_argument("--generation", type=int, required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--engine", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--pieces", type=int, default=300)
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--inference-batch", type=int, default=16)
    ap.add_argument("--selfplay-workers", type=int, default=0,
                    help="parallel C++ self-play exporters sharing one GPU queue; 0 selects up to 64")
    ap.add_argument("--selfplay-batch-window-ms", type=float, default=20.0,
                    help="self-play GPU micro-batch wait window in milliseconds")
    ap.add_argument("--selfplay-target-positions", type=int, default=256)
    ap.add_argument("--selfplay-inflight-batches", type=int, default=2)
    ap.add_argument("--selfplay-gpu-workers", type=int, default=2)
    ap.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16",
                    help="GPU inference arithmetic for self-play and GPU Arena")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--determinizations", type=int, default=2,
                    help="root futures used during GPU self-play")
    ap.add_argument("--no-gumbel", action="store_true",
                    help="use batched PUCT for self-play instead of Gumbel search")
    ap.add_argument("--train-steps", type=int, default=5000)
    ap.add_argument("--train-batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5,
                    help="production fine-tuning learning rate; passed explicitly to train.py")
    ap.add_argument("--new-data-repeat", type=int, default=4)
    ap.add_argument("--reanalyze", action="store_true",
                    help="refresh a selected fraction of the fresh self-play policy targets before training")
    ap.add_argument("--reanalyze-select-fraction", type=float, default=0.05,
                    help="fraction of fresh self-play rows selected by policy KL for deeper re-search")
    ap.add_argument("--reanalyze-sims", type=int, default=0,
                    help="deeper Reanalyse simulations; 0 selects 2x the self-play budget")
    ap.add_argument("--reanalyze-batch", type=int, default=0,
                    help="Reanalyse inference batch; 0 inherits --inference-batch")
    ap.add_argument("--reanalyze-determinizations", type=int, default=0,
                    help="Reanalyse root determinizations; 0 inherits --determinizations")
    ap.add_argument("--reanalyze-batch-window-ms", type=float, default=20.0,
                    help="Reanalyse GPU micro-batch wait window in milliseconds")
    ap.add_argument("--reanalyze-target-positions", type=int, default=256)
    ap.add_argument("--reanalyze-inflight-batches", type=int, default=2)
    ap.add_argument("--reanalyze-gpu-workers", type=int, default=2)
    ap.add_argument("--reanalyze-secondary-fraction", type=float, default=0.10,
                    help="fraction of each training batch drawn from refreshed rows")
    ap.add_argument("--reset-optimizer", action="store_true",
                    help="start a fresh optimizer when the champion came from a different training objective")
    ap.add_argument("--reset-sampling", action="store_true",
                    help="restart training-sample RNG from --seed instead of restoring checkpoint sampling state")
    ap.add_argument("--model", choices=("dev", "s"), default="s")
    # A promoted checkpoint may have been produced by an ablation with unusual
    # stored loss weights (for example hard-rank-only training).  The guarded
    # generation loop must therefore define its training objective explicitly
    # instead of silently inheriting experiment-specific weights from --resume.
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--value-weight", type=float, default=0.05)
    ap.add_argument("--aux-weight", type=float, default=1.0)
    ap.add_argument("--chosen-action-weight", type=float, default=0.0)
    ap.add_argument("--chosen-disagreement-weight", type=float, default=0.0)
    ap.add_argument("--policy-rank-weight", type=float, default=1.0,
                    help="production ranking-loss weight; Gen14 rank=1.0 preserved Gen12 safety and promoted")
    ap.add_argument("--policy-target-temperature", type=float, default=1.0)
    ap.add_argument("--policy-pair-rank-weight", type=float, default=0.0)
    ap.add_argument("--vs-aux-weight", type=float, default=0.0)
    ap.add_argument("--cancellation-aux-weight", type=float, default=0.0)
    ap.add_argument("--timing-pair-weight", type=float, default=0.0,
                    help="experimental FASTEST/WAIT soft pair loss; production default stays off")
    ap.add_argument("--timing-rank-weight", type=float, default=0.0,
                    help="experimental FASTEST/WAIT pair ranking loss; production default stays off")
    ap.add_argument("--factor-timing-policy", action="store_true",
                    help="experimental timing factorization; production default stays off")
    ap.add_argument("--timing-wait-bias", type=float, default=0.0,
                    help="experimental WAIT bias; requires --factor-timing-policy")
    ap.add_argument("--topout-aux-weight", type=float, default=0.0)
    ap.add_argument("--arena-pairs", type=int, default=10)
    ap.add_argument("--arena-sims", type=int, default=32)
    ap.add_argument("--arena-pieces", type=int, default=300)
    ap.add_argument("--arena-workers", type=int, default=0,
                    help="parallel GPU Arena pairs; 0 selects up to 32")
    ap.add_argument("--arena-batch-window-ms", type=float, default=12.0,
                    help="GPU Arena micro-batch wait window in milliseconds")
    ap.add_argument("--arena-target-positions", type=int, default=192)
    ap.add_argument("--arena-inflight-batches", type=int, default=2)
    ap.add_argument("--arena-gpu-workers", type=int, default=2)
    ap.add_argument("--arena-determinizations", type=int, default=0,
                    help="root futures used by Arena; 0 inherits --determinizations")
    ap.add_argument("--arena-seed", type=int, default=42,
                    help="base seed for the GPU Arena trial")
    arena_search = ap.add_mutually_exclusive_group()
    arena_search.add_argument("--arena-gumbel", dest="arena_gumbel", action="store_true",
                              help="force Gumbel search in the GPU Arena")
    arena_search.add_argument("--arena-no-gumbel", dest="arena_gumbel", action="store_false",
                              help="force PUCT search in the GPU Arena")
    ap.set_defaults(arena_gumbel=None)
    ap.add_argument("--cpu-arena", action="store_true",
                    help="use the legacy C++ CPU Arena instead of GPU Arena")
    ap.add_argument("--model-version", type=int, default=-1)
    ap.add_argument("--arena-candidate", choices=("final", "best"), default="final",
                    help="checkpoint evaluated/promoted by Arena after training")
    args = ap.parse_args()

    if args.reanalyze:
        if not 0.0 < args.reanalyze_select_fraction < 1.0:
            raise SystemExit("--reanalyze-select-fraction must be in (0, 1)")
        if not 0.0 < args.reanalyze_secondary_fraction < 1.0:
            raise SystemExit("--reanalyze-secondary-fraction must be in (0, 1)")
        if args.reanalyze_sims > 0 and args.reanalyze_sims <= max(1, args.sims):
            raise SystemExit("--reanalyze-sims must exceed the self-play simulation budget")

    root = Path(__file__).resolve().parents[1]
    champion = Path(args.champion)
    if not champion.is_absolute():
        champion = root / champion
    if not champion.exists():
        raise SystemExit(f"champion checkpoint not found: {champion}")

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    models_dir = Path(args.models_dir)
    if not models_dir.is_absolute():
        models_dir = root / models_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    tag = f"gen{args.generation}"
    new_data = data_dir / f"{tag}.tetradat"
    reanalyzed_data = data_dir / "reanalyze" / f"{tag}.reanalyzed.tetradat"
    candidate = models_dir / f"{tag}.pt"
    candidate_best = models_dir / f"{tag}.best.pt"
    candidate_weights = models_dir / f"{tag}.tetrawts"
    version = args.generation if args.model_version < 0 else args.model_version
    engine_path = Path(args.engine) if args.engine else Path("build/tetra_cli.exe" if os.name == "nt" else "build/tetra_cli")
    if not engine_path.is_absolute():
        engine_path = root / engine_path
    engine = str(engine_path)
    if not engine_path.exists():
        raise SystemExit(f"engine not found: {engine}; run make tools first")

    py = sys.executable
    started = time.time()
    arena_determinizations = (
        args.determinizations if args.arena_determinizations <= 0
        else args.arena_determinizations
    )
    arena_use_gumbel = (
        not args.no_gumbel if args.arena_gumbel is None
        else args.arena_gumbel
    )
    reanalyze_sims = (
        max(max(1, args.sims) + 1, max(1, args.sims) * 2)
        if args.reanalyze_sims <= 0 else args.reanalyze_sims
    )
    reanalyze_batch = (
        max(1, args.inference_batch) if args.reanalyze_batch <= 0
        else max(1, args.reanalyze_batch)
    )
    reanalyze_determinizations = (
        max(1, args.determinizations) if args.reanalyze_determinizations <= 0
        else max(1, args.reanalyze_determinizations)
    )

    total_games = max(1, args.games)
    selfplay_workers = (
        min(total_games, 64) if args.selfplay_workers <= 0
        else min(total_games, max(1, args.selfplay_workers))
    )
    if selfplay_workers > 1:
        selfplay_cmd = [
            py, str(root / "trainer/gpu_selfplay_parallel.py"),
            str(champion), str(new_data),
            "--device", args.device, "--engine", engine,
            "--games", str(total_games), "--pieces", str(max(1, args.pieces)),
            "--sims", str(max(1, args.sims)), "--batch", str(max(1, args.inference_batch)),
            "--seed", str(args.seed), "--model-version", str(max(0, version)),
            "--determinizations", str(max(1, args.determinizations)),
            "--precision", args.precision,
            "--workers", str(selfplay_workers),
            "--batch-window-ms", str(max(0.0, args.selfplay_batch_window_ms)),
            "--target-positions", str(max(1, args.selfplay_target_positions)),
            "--inflight-batches", str(max(1, args.selfplay_inflight_batches)),
            "--gpu-workers", str(max(1, args.selfplay_gpu_workers)),
        ]
    else:
        selfplay_cmd = [
            py, str(root / "trainer/gpu_selfplay.py"), str(champion), str(new_data),
            "--device", args.device, "--engine", engine,
            "--games", str(total_games), "--pieces", str(max(1, args.pieces)),
            "--sims", str(max(1, args.sims)), "--batch", str(max(1, args.inference_batch)),
            "--seed", str(args.seed), "--model-version", str(max(0, version)),
            "--determinizations", str(max(1, args.determinizations)),
            "--precision", args.precision,
        ]
    if args.no_gumbel:
        selfplay_cmd.append("--no-gumbel")
    run_checked(selfplay_cmd, root)

    new_data_shards = parallel_shard_paths(new_data, selfplay_workers)
    missing_fresh = [path for path in new_data_shards if not path.is_file()]
    if missing_fresh:
        raise RuntimeError(f"self-play did not produce expected shard: {missing_fresh[0]}")

    champion_aux_targets = checkpoint_aux_targets(champion)
    aux_widths = {dataset_aux_targets(path) for path in new_data_shards}
    if len(aux_widths) != 1:
        raise RuntimeError(f"fresh self-play shards disagree on auxiliary width: {sorted(aux_widths)}")
    dataset_aux_count = next(iter(aux_widths))
    if champion_aux_targets < 0:
        raise RuntimeError(f"cannot determine Champion auxiliary width: {champion}")
    if champion_aux_targets > dataset_aux_count:
        raise RuntimeError(
            "automatic auxiliary migration cannot narrow the Champion head: "
            f"checkpoint {champion_aux_targets} vs dataset {dataset_aux_count}"
        )
    auto_upgrade_aux_schema = champion_aux_targets < dataset_aux_count
    if auto_upgrade_aux_schema:
        print(
            f"migration     generation requires aux head widening "
            f"{champion_aux_targets} -> {dataset_aux_count}; optimizer will reset",
            flush=True,
        )

    reanalyzed_shards: list[Path] = []
    reanalyze_manifests: list[Path] = []
    if args.reanalyze:
        reanalyzed_data.parent.mkdir(parents=True, exist_ok=True)
        reanalyze_cmd = [
            py, str(root / "trainer/reanalyze_parallel.py"),
            str(champion), str(reanalyzed_data), *[str(path) for path in new_data_shards],
            "--device", args.device, "--engine", engine,
            "--precision", args.precision,
            "--select-fraction", str(args.reanalyze_select_fraction),
            "--original-sims", str(max(1, args.sims)),
            "--sims", str(reanalyze_sims),
            "--batch", str(reanalyze_batch),
            "--determinizations", str(reanalyze_determinizations),
            "--workers", str(len(new_data_shards)),
            "--batch-window-ms", str(max(0.0, args.reanalyze_batch_window_ms)),
            "--target-positions", str(max(1, args.reanalyze_target_positions)),
            "--inflight-batches", str(max(1, args.reanalyze_inflight_batches)),
            "--gpu-workers", str(max(1, args.reanalyze_gpu_workers)),
        ]
        if args.no_gumbel:
            reanalyze_cmd.append("--no-gumbel")
        run_checked(reanalyze_cmd, root)
        reanalyzed_shards = parallel_shard_paths(reanalyzed_data, len(new_data_shards))
        missing_reanalyzed = [path for path in reanalyzed_shards if not path.is_file()]
        if missing_reanalyzed:
            raise RuntimeError(
                f"parallel Reanalyse did not produce expected shard: {missing_reanalyzed[0]}"
            )
        reanalyze_manifests = [
            path.with_suffix(path.suffix + ".reanalyze.json") for path in reanalyzed_shards
        ]

    replay_paths = [str(Path(p) if Path(p).is_absolute() else root / p) for p in args.replay]
    train_inputs = replay_paths + [str(path) for path in new_data_shards]
    if args.reanalyze:
        train_inputs.extend(str(path) for path in reanalyzed_shards)
    train_new_data_repeat = max(1, args.new_data_repeat)
    train_cmd = [
        py, str(root / "trainer/train.py"), *train_inputs,
        "--resume", str(champion), "--steps", str(max(0, args.train_steps)),
        "--batch", str(max(1, args.train_batch)), "--model", args.model,
        "--device", args.device, "--require-gpu", "--lr", str(args.lr),
        "--policy-weight", str(args.policy_weight),
        "--value-weight", str(args.value_weight),
        "--aux-weight", str(args.aux_weight),
        "--chosen-action-weight", str(args.chosen_action_weight),
        "--chosen-disagreement-weight", str(args.chosen_disagreement_weight),
        "--policy-rank-weight", str(args.policy_rank_weight),
        "--policy-target-temperature", str(args.policy_target_temperature),
        "--policy-pair-rank-weight", str(args.policy_pair_rank_weight),
        "--vs-aux-weight", str(args.vs_aux_weight),
        "--cancellation-aux-weight", str(args.cancellation_aux_weight),
        "--timing-pair-weight", str(args.timing_pair_weight),
        "--timing-rank-weight", str(args.timing_rank_weight),
        "--timing-wait-bias", str(args.timing_wait_bias),
        "--factor-timing-policy" if args.factor_timing_policy else "--no-factor-timing-policy",
        "--topout-aux-weight", str(args.topout_aux_weight),
        "--new-data-repeat", str(train_new_data_repeat),
        "--checkpoint-every", str(max(0, args.train_steps // 5)),
        "--best-save", str(candidate_best), "--save", str(candidate),
    ]
    if args.reanalyze:
        train_cmd.extend([
            "--secondary-source-count", "1",
            "--secondary-source-fraction", str(args.reanalyze_secondary_fraction),
        ])
    if auto_upgrade_aux_schema:
        train_cmd.append("--upgrade-aux-schema")
    if args.reset_optimizer or auto_upgrade_aux_schema:
        train_cmd.append("--reset-optimizer")
    if args.reset_sampling:
        train_cmd.append("--reset-sampling")
    run_checked(train_cmd, root)

    arena_candidate = candidate_best if args.arena_candidate == "best" else candidate
    if not arena_candidate.exists():
        raise RuntimeError(f"Arena candidate checkpoint was not produced: {arena_candidate}")
    export_cmd = [
        py, str(root / "trainer/export_weights.py"),
        str(arena_candidate), str(candidate_weights),
    ]
    run_checked(export_cmd, root)

    arena_backend = "gpu"
    if args.cpu_arena:
        arena_backend = "cpu"
        champion_weights = champion.with_suffix(".tetrawts")
        if not champion_weights.exists():
            run_checked([py, str(root / "trainer/export_weights.py"), str(champion), str(champion_weights)], root)
        arena_cmd = [
            engine, "arena", str(candidate_weights), str(champion_weights),
            str(max(1, args.arena_pairs)), str(max(1, args.arena_sims)),
            str(max(1, args.arena_pieces)),
        ]
    else:
        arena_cmd = [
            py, str(root / "trainer/gpu_arena.py"), str(arena_candidate), str(champion),
            "--device", args.device, "--engine", engine,
            "--pairs", str(max(1, args.arena_pairs)),
            "--sims", str(max(1, args.arena_sims)),
            "--pieces", str(max(1, args.arena_pieces)),
            "--batch", str(max(1, args.inference_batch)),
            "--workers", str(args.arena_workers),
            "--batch-window-ms", str(max(0.0, args.arena_batch_window_ms)),
            "--target-positions", str(max(1, args.arena_target_positions)),
            "--inflight-batches", str(max(1, args.arena_inflight_batches)),
            "--gpu-workers", str(max(1, args.arena_gpu_workers)),
            "--determinizations", str(max(1, arena_determinizations)),
            "--seed", str(max(0, args.arena_seed)),
            "--precision", args.precision,
        ]
        if arena_use_gumbel:
            arena_cmd.append("--gumbel")
    arena_output = run_checked(arena_cmd, root, capture=True)
    print(arena_output, end="", flush=True)
    arena = parse_arena(arena_output)

    promoted = bool(arena["promoted"] and args.champion_output)
    promoted_to = ""
    if promoted:
        champion_output = Path(args.champion_output)
        if not champion_output.is_absolute():
            champion_output = root / champion_output
        promoted_to = str(champion_output)
        champion_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(arena_candidate, champion_output.with_suffix(".pt"))
        shutil.copy2(candidate_weights, champion_output.with_suffix(".tetrawts"))
        print(f"promoted      {champion_output.with_suffix('.pt')}", flush=True)
    elif arena["promoted"]:
        print("promotion     Arena passed; no --champion-output supplied, so no files replaced", flush=True)
    else:
        print("promotion     retained current champion", flush=True)

    summary = {
        "generation": args.generation,
        "champion": str(champion),
        "new_data": str(new_data),
        "new_data_shards": [str(path) for path in new_data_shards],
        "candidate": str(candidate),
        "candidate_best": str(candidate_best),
        "arena_candidate": str(arena_candidate),
        "candidate_weights": str(candidate_weights),
        "selfplay": {
            "games": max(1, args.games),
            "pieces": max(1, args.pieces),
            "simulations": max(1, args.sims),
            "inference_batch": max(1, args.inference_batch),
            "determinizations": max(1, args.determinizations),
            "use_gumbel": not args.no_gumbel,
            "precision": args.precision,
            "seed": args.seed,
        },
        "reanalyze": {
            "enabled": args.reanalyze,
            "dataset": str(reanalyzed_data) if args.reanalyze else "",
            "shards": [str(path) for path in reanalyzed_shards],
            "manifest": str(reanalyze_manifests[0]) if reanalyze_manifests else "",
            "manifests": [str(path) for path in reanalyze_manifests],
            "selection_fraction": args.reanalyze_select_fraction if args.reanalyze else 0.0,
            "simulations": reanalyze_sims if args.reanalyze else 0,
            "batch": reanalyze_batch if args.reanalyze else 0,
            "determinizations": reanalyze_determinizations if args.reanalyze else 0,
            "secondary_batch_fraction": (
                args.reanalyze_secondary_fraction if args.reanalyze else 0.0
            ),
        },
        "training": {
            "steps": max(0, args.train_steps),
            "batch": max(1, args.train_batch),
            "lr": args.lr,
            "new_data_repeat": max(1, args.new_data_repeat),
            "reset_optimizer": args.reset_optimizer or auto_upgrade_aux_schema,
            "auto_upgrade_aux_schema": auto_upgrade_aux_schema,
            "champion_aux_targets": champion_aux_targets,
            "dataset_aux_targets": dataset_aux_count,
            "reset_sampling": args.reset_sampling,
            "policy_weight": args.policy_weight,
            "value_weight": args.value_weight,
            "aux_weight": args.aux_weight,
            "chosen_action_weight": args.chosen_action_weight,
            "chosen_disagreement_weight": args.chosen_disagreement_weight,
            "policy_rank_weight": args.policy_rank_weight,
            "policy_target_temperature": args.policy_target_temperature,
            "policy_pair_rank_weight": args.policy_pair_rank_weight,
            "vs_aux_weight": args.vs_aux_weight,
            "cancellation_aux_weight": args.cancellation_aux_weight,
            "topout_aux_weight": args.topout_aux_weight,
        },
        "arena_backend": arena_backend,
        "arena_runtime": {
            "workers": (min(max(1, args.arena_pairs), 32)
                        if args.arena_workers <= 0
                        else min(max(1, args.arena_pairs), max(1, args.arena_workers))),
            "batch_window_ms": max(0.0, args.arena_batch_window_ms),
            "target_positions": max(1, args.arena_target_positions),
            "inflight_batches": max(1, args.arena_inflight_batches),
            "gpu_workers": max(1, args.arena_gpu_workers),
        },
        "candidate_step": checkpoint_step(candidate),
        "arena_candidate_step": checkpoint_step(arena_candidate),
        "arena": arena,
        "promoted_to": promoted_to,
        "elapsed_seconds": time.time() - started,
    }
    summary_path = models_dir / f"{tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary       {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
