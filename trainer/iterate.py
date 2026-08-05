# SPDX-License-Identifier: MIT
"""Run one guarded AlphaZero-style training generation.

The driver deliberately keeps promotion separate from training:

    champion checkpoint -> GPU self-play -> replay mix -> GPU train
        -> Arena (GPU by default) -> promote only when the threshold passes

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
    ap.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16",
                    help="GPU inference arithmetic for self-play and GPU Arena")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--determinizations", type=int, default=2,
                    help="root futures used during GPU self-play")
    ap.add_argument("--no-gumbel", action="store_true",
                    help="use batched PUCT for self-play instead of Gumbel search")
    ap.add_argument("--train-steps", type=int, default=5000)
    ap.add_argument("--train-batch", type=int, default=256)
    ap.add_argument("--new-data-repeat", type=int, default=4)
    ap.add_argument("--model", choices=("dev", "s"), default="s")
    ap.add_argument("--arena-pairs", type=int, default=10)
    ap.add_argument("--arena-sims", type=int, default=32)
    ap.add_argument("--arena-pieces", type=int, default=300)
    ap.add_argument("--arena-determinizations", type=int, default=1)
    ap.add_argument("--arena-seed", type=int, default=42,
                    help="base seed for the GPU Arena trial")
    ap.add_argument("--arena-gumbel", action="store_true",
                    help="use Gumbel search in the GPU Arena")
    ap.add_argument("--cpu-arena", action="store_true",
                    help="use the legacy C++ CPU Arena instead of GPU Arena")
    ap.add_argument("--model-version", type=int, default=-1)
    args = ap.parse_args()

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

    selfplay_cmd = [
        py, str(root / "trainer/gpu_selfplay.py"), str(champion), str(new_data),
        "--device", args.device, "--engine", engine,
        "--games", str(max(1, args.games)), "--pieces", str(max(1, args.pieces)),
        "--sims", str(max(1, args.sims)), "--batch", str(max(1, args.inference_batch)),
        "--seed", str(args.seed), "--model-version", str(max(0, version)),
        "--determinizations", str(max(1, args.determinizations)),
        "--precision", args.precision,
    ]
    if args.no_gumbel:
        selfplay_cmd.append("--no-gumbel")
    run_checked(selfplay_cmd, root)

    replay_paths = [str(Path(p) if Path(p).is_absolute() else root / p) for p in args.replay]
    train_inputs = replay_paths + [str(new_data)]
    train_cmd = [
        py, str(root / "trainer/train.py"), *train_inputs,
        "--resume", str(champion), "--steps", str(max(0, args.train_steps)),
        "--batch", str(max(1, args.train_batch)), "--model", args.model,
        "--device", args.device, "--require-gpu", "--value-weight", "1.0",
        "--new-data-repeat", str(max(1, args.new_data_repeat)),
        "--checkpoint-every", str(max(0, args.train_steps // 5)),
        "--best-save", str(candidate_best), "--save", str(candidate),
    ]
    run_checked(train_cmd, root)

    export_cmd = [py, str(root / "trainer/export_weights.py"), str(candidate), str(candidate_weights)]
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
            py, str(root / "trainer/gpu_arena.py"), str(candidate), str(champion),
            "--device", args.device, "--engine", engine,
            "--pairs", str(max(1, args.arena_pairs)),
            "--sims", str(max(1, args.arena_sims)),
            "--pieces", str(max(1, args.arena_pieces)),
            "--batch", str(max(1, args.inference_batch)),
            "--determinizations", str(max(1, args.arena_determinizations)),
            "--seed", str(max(0, args.arena_seed)),
            "--precision", args.precision,
        ]
        if args.arena_gumbel:
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
        shutil.copy2(candidate, champion_output.with_suffix(".pt"))
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
        "candidate": str(candidate),
        "candidate_weights": str(candidate_weights),
        "arena_backend": arena_backend,
        "candidate_step": checkpoint_step(candidate),
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
