# SPDX-License-Identifier: MIT
"""Run unattended AlphaZero-style self-improvement generations.

The loop keeps one frozen/current Champion authority and repeatedly invokes
``trainer/iterate.py`` for:

    Champion -> fresh GPU self-play -> selective Reanalyse -> GPU training
        -> paired parallel Arena -> conditional promotion -> next generation

Only Arena promotion changes the Champion.  Rejected generations still add
fresh Champion self-play to the rolling replay window, so the next Candidate
can learn from more data without inheriting rejected weights.

The driver is intentionally stateful and restartable.  It never overwrites a
partially-created generation: if artifacts already occupy a generation number,
it advances to the next free number and records the skip in the state file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_FORMAT = "tetraformer-auto-improve-v1"
DEFAULT_GEN14 = "models/gen14_rank100_100_20260814.best.pt"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as stream:
        state = json.load(stream)
    if state.get("format") != STATE_FORMAT:
        raise SystemExit(f"unsupported auto-improve state format in {path}")
    return state


def generation_artifacts(data_dir: Path, models_dir: Path, generation: int) -> list[Path]:
    tag = f"gen{generation}"
    return [
        data_dir / f"{tag}.tetradat",
        data_dir / "reanalyze" / f"{tag}.reanalyzed.tetradat",
        models_dir / f"{tag}.pt",
        models_dir / f"{tag}.best.pt",
        models_dir / f"{tag}.tetrawts",
        models_dir / f"{tag}.arena.json",
        models_dir / f"{tag}.json",
    ]


def next_free_generation(data_dir: Path, models_dir: Path, start: int) -> tuple[int, list[int]]:
    generation = max(1, start)
    skipped: list[int] = []
    while any(path.exists() for path in generation_artifacts(data_dir, models_dir, generation)):
        skipped.append(generation)
        generation += 1
    return generation, skipped


def run_checked(command: list[str], root: Path) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(root), check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", default=DEFAULT_GEN14,
                    help="initial Champion; ignored after a state file has established a newer Champion")
    ap.add_argument("--champion-output", default="models/auto_improve/champion",
                    help="stable base path replaced only after Arena promotion")
    ap.add_argument("--state", default="models/auto_improve/state.json")
    ap.add_argument("--data-dir", default="data/auto_improve")
    ap.add_argument("--models-dir", default="models/auto_improve")
    ap.add_argument("--engine", default="")
    ap.add_argument("--device", default="cuda:1",
                    help="accelerator device; this workstation's RX 9070 XT is cuda:1")
    ap.add_argument("--start-generation", type=int, default=40)
    ap.add_argument("--generations", type=int, default=10,
                    help="successful generations to run; 0 runs until interrupted")
    ap.add_argument("--bootstrap-replay", nargs="*", default=[],
                    help="fixed historical replay inputs retained outside the rolling window")
    ap.add_argument("--bootstrap-replay-dir", default="",
                    help="add every .tetradat shard under this directory to the fixed bootstrap replay")
    ap.add_argument("--replay-window", type=int, default=8,
                    help="number of completed fresh self-play generations kept in rolling replay")

    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--pieces", type=int, default=300)
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--inference-batch", type=int, default=16)
    ap.add_argument("--selfplay-workers", type=int, default=64)
    ap.add_argument("--selfplay-batch-window-ms", type=float, default=20.0)
    ap.add_argument("--selfplay-target-positions", type=int, default=256)
    ap.add_argument("--selfplay-inflight-batches", type=int, default=2)
    ap.add_argument("--selfplay-gpu-workers", type=int, default=2)
    ap.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    ap.add_argument("--determinizations", type=int, default=2)
    ap.add_argument("--seed", type=int, default=41000000,
                    help="base self-play seed; each attempt receives a disjoint deterministic offset")
    ap.add_argument("--seed-stride", type=int, default=1000003)
    ap.add_argument("--no-gumbel", action="store_true")

    ap.add_argument("--train-steps", type=int, default=5000)
    ap.add_argument("--train-batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--new-data-repeat", type=int, default=4)
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--value-weight", type=float, default=0.05)
    ap.add_argument("--aux-weight", type=float, default=1.0)
    ap.add_argument("--policy-rank-weight", type=float, default=1.0)

    reanalyse = ap.add_mutually_exclusive_group()
    reanalyse.add_argument("--reanalyze", dest="reanalyze", action="store_true",
                           help="enable selective deeper search on each fresh self-play generation")
    reanalyse.add_argument("--no-reanalyze", dest="reanalyze", action="store_false")
    ap.set_defaults(reanalyze=True)
    ap.add_argument("--reanalyze-select-fraction", type=float, default=0.05)
    ap.add_argument("--reanalyze-sims", type=int, default=0,
                    help="0 selects 2x --sims")
    ap.add_argument("--reanalyze-secondary-fraction", type=float, default=0.10)
    ap.add_argument("--reanalyze-batch-window-ms", type=float, default=20.0)
    ap.add_argument("--reanalyze-target-positions", type=int, default=256)
    ap.add_argument("--reanalyze-inflight-batches", type=int, default=2)
    ap.add_argument("--reanalyze-gpu-workers", type=int, default=2)

    ap.add_argument("--arena-pairs", type=int, default=20)
    ap.add_argument("--arena-sims", type=int, default=32)
    ap.add_argument("--arena-pieces", type=int, default=300)
    ap.add_argument("--arena-workers", type=int, default=0,
                    help="0 lets gpu_arena.py select all pairs up to 32 parallel workers")
    ap.add_argument("--arena-batch-window-ms", type=float, default=12.0)
    ap.add_argument("--arena-target-positions", type=int, default=192)
    ap.add_argument("--arena-inflight-batches", type=int, default=2)
    ap.add_argument("--arena-gpu-workers", type=int, default=2)
    ap.add_argument("--arena-seed", type=int, default=42,
                    help="kept fixed across generations for a stable promotion gate")
    ap.add_argument("--arena-candidate", choices=("final", "best"), default="best")

    ap.add_argument("--cooldown-seconds", type=float, default=10.0)
    ap.add_argument("--failure-cooldown-seconds", type=float, default=30.0)
    ap.add_argument("--max-consecutive-failures", type=int, default=3,
                    help="stop safely after this many failed generation attempts; 0 never stops for failures")
    args = ap.parse_args()

    if args.generations < 0:
        raise SystemExit("--generations must be non-negative")
    if args.replay_window < 0:
        raise SystemExit("--replay-window must be non-negative")
    if args.seed_stride <= 0:
        raise SystemExit("--seed-stride must be positive")
    if args.max_consecutive_failures < 0:
        raise SystemExit("--max-consecutive-failures must be non-negative")
    if args.reanalyze and not 0.0 < args.reanalyze_select_fraction < 1.0:
        raise SystemExit("--reanalyze-select-fraction must be in (0, 1)")
    if args.reanalyze and not 0.0 < args.reanalyze_secondary_fraction < 1.0:
        raise SystemExit("--reanalyze-secondary-fraction must be in (0, 1)")

    root = Path(__file__).resolve().parents[1]
    state_path = resolve(root, args.state)
    data_dir = resolve(root, args.data_dir)
    models_dir = resolve(root, args.models_dir)
    champion_output = resolve(root, args.champion_output)
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    champion_output.parent.mkdir(parents=True, exist_ok=True)

    initial_champion = resolve(root, args.champion)
    if not initial_champion.is_file():
        raise SystemExit(f"initial Champion not found: {initial_champion}")
    engine = resolve(
        root,
        args.engine or ("build/tetra_cli.exe" if os.name == "nt" else "build/tetra_cli"),
    )
    if not engine.is_file():
        raise SystemExit(f"engine not found: {engine}; run make tools first")

    fixed_replay = [str(resolve(root, path)) for path in args.bootstrap_replay]
    if args.bootstrap_replay_dir:
        bootstrap_dir = resolve(root, args.bootstrap_replay_dir)
        if not bootstrap_dir.is_dir():
            raise SystemExit(f"bootstrap replay directory not found: {bootstrap_dir}")
        fixed_replay.extend(str(path) for path in sorted(bootstrap_dir.rglob("*.tetradat")))
    # Avoid double weighting when a shard is named both explicitly and through
    # the directory convenience flag. Preserve the user's ordering otherwise.
    fixed_replay = list(dict.fromkeys(fixed_replay))
    missing_replay = [path for path in fixed_replay if not Path(path).is_file()]
    if missing_replay:
        raise SystemExit(f"bootstrap replay not found: {missing_replay[0]}")

    state = load_state(state_path)
    if state is None:
        state = {
            "format": STATE_FORMAT,
            "created_utc": utc_now(),
            "updated_utc": utc_now(),
            "initial_champion": str(initial_champion),
            "current_champion": str(initial_champion),
            "champion_output": str(champion_output),
            "next_generation": max(1, args.start_generation),
            "fixed_replay": fixed_replay,
            "rolling_replay": [],
            "rolling_replay_groups": [],
            "completed": [],
            "failures": [],
            "skipped_generation_numbers": [],
            "attempt_counter": 0,
            "consecutive_failures": 0,
        }
        atomic_json(state_path, state)
    else:
        current = Path(state["current_champion"])
        if not current.is_file():
            raise SystemExit(f"state Champion is missing: {current}")
        # Preserve the replay contract that created the state.  New CLI replay
        # paths are accepted only on a fresh state file, avoiding silent data
        # distribution changes after a restart.
        fixed_replay = list(state.get("fixed_replay", []))
        if "rolling_replay_groups" not in state:
            legacy_replay = [str(Path(path)) for path in state.get("rolling_replay", [])]
            state["rolling_replay_groups"] = [legacy_replay] if legacy_replay else []
            state["updated_utc"] = utc_now()
            atomic_json(state_path, state)

    target_successes = args.generations
    successes_this_run = 0
    py = sys.executable

    try:
        while target_successes == 0 or successes_this_run < target_successes:
            generation, skipped = next_free_generation(
                data_dir, models_dir, int(state["next_generation"])
            )
            if skipped:
                state.setdefault("skipped_generation_numbers", []).extend(skipped)
                state["next_generation"] = generation
                state["updated_utc"] = utc_now()
                atomic_json(state_path, state)
                print(
                    "resume       preserved occupied generation artifacts; "
                    f"advancing to gen{generation}", flush=True,
                )

            attempt_index = int(state.get("attempt_counter", 0))
            selfplay_seed = args.seed + attempt_index * args.seed_stride
            current_champion = Path(state["current_champion"])
            rolling_groups = [
                [str(Path(path)) for path in group]
                for group in state.get("rolling_replay_groups", [])
            ]
            rolling_replay = [path for group in rolling_groups for path in group]
            replay = fixed_replay + rolling_replay

            command = [
                py, str(root / "trainer/iterate.py"),
                "--champion", str(current_champion),
                "--champion-output", str(champion_output),
                "--generation", str(generation),
                "--data-dir", str(data_dir),
                "--models-dir", str(models_dir),
                "--engine", str(engine),
                "--device", args.device,
                "--games", str(max(1, args.games)),
                "--pieces", str(max(1, args.pieces)),
                "--sims", str(max(1, args.sims)),
                "--inference-batch", str(max(1, args.inference_batch)),
                "--selfplay-workers", str(max(1, args.selfplay_workers)),
                "--selfplay-batch-window-ms", str(max(0.0, args.selfplay_batch_window_ms)),
                "--selfplay-target-positions", str(max(1, args.selfplay_target_positions)),
                "--selfplay-inflight-batches", str(max(1, args.selfplay_inflight_batches)),
                "--selfplay-gpu-workers", str(max(1, args.selfplay_gpu_workers)),
                "--precision", args.precision,
                "--seed", str(selfplay_seed),
                "--determinizations", str(max(1, args.determinizations)),
                "--train-steps", str(max(0, args.train_steps)),
                "--train-batch", str(max(1, args.train_batch)),
                "--lr", str(args.lr),
                "--new-data-repeat", str(max(1, args.new_data_repeat)),
                "--policy-weight", str(args.policy_weight),
                "--value-weight", str(args.value_weight),
                "--aux-weight", str(args.aux_weight),
                "--policy-rank-weight", str(args.policy_rank_weight),
                "--arena-pairs", str(max(1, args.arena_pairs)),
                "--arena-sims", str(max(1, args.arena_sims)),
                "--arena-pieces", str(max(1, args.arena_pieces)),
                "--arena-workers", str(args.arena_workers),
                "--arena-batch-window-ms", str(max(0.0, args.arena_batch_window_ms)),
                "--arena-target-positions", str(max(1, args.arena_target_positions)),
                "--arena-inflight-batches", str(max(1, args.arena_inflight_batches)),
                "--arena-gpu-workers", str(max(1, args.arena_gpu_workers)),
                "--arena-seed", str(max(0, args.arena_seed)),
                "--arena-candidate", args.arena_candidate,
            ]
            if replay:
                command.extend(["--replay", *replay])
            if args.no_gumbel:
                command.append("--no-gumbel")
            if args.reanalyze:
                command.extend([
                    "--reanalyze",
                    "--reanalyze-select-fraction", str(args.reanalyze_select_fraction),
                    "--reanalyze-sims", str(args.reanalyze_sims),
                    "--reanalyze-secondary-fraction", str(args.reanalyze_secondary_fraction),
                    "--reanalyze-batch-window-ms", str(max(0.0, args.reanalyze_batch_window_ms)),
                    "--reanalyze-target-positions", str(max(1, args.reanalyze_target_positions)),
                    "--reanalyze-inflight-batches", str(max(1, args.reanalyze_inflight_batches)),
                    "--reanalyze-gpu-workers", str(max(1, args.reanalyze_gpu_workers)),
                ])

            state["attempt_counter"] = attempt_index + 1
            state["active_attempt"] = {
                "generation": generation,
                "started_utc": utc_now(),
                "champion": str(current_champion),
                "selfplay_seed": selfplay_seed,
                "replay": replay,
                "command": command,
            }
            state["updated_utc"] = utc_now()
            atomic_json(state_path, state)

            print(
                f"\n=== auto improve gen{generation} | Champion {current_champion.name} "
                f"| replay {len(replay)} | seed {selfplay_seed} ===",
                flush=True,
            )
            try:
                run_checked(command, root)
                summary_path = models_dir / f"gen{generation}.json"
                if not summary_path.is_file():
                    raise RuntimeError(f"generation completed without summary: {summary_path}")
                with summary_path.open("r", encoding="utf-8") as stream:
                    summary = json.load(stream)
            except (subprocess.CalledProcessError, RuntimeError) as exc:
                failure = {
                    "generation": generation,
                    "failed_utc": utc_now(),
                    "error": repr(exc),
                    "champion": str(current_champion),
                    "selfplay_seed": selfplay_seed,
                }
                state.setdefault("failures", []).append(failure)
                state["consecutive_failures"] = int(state.get("consecutive_failures", 0)) + 1
                state["next_generation"] = generation + 1
                state.pop("active_attempt", None)
                state["updated_utc"] = utc_now()
                atomic_json(state_path, state)
                print(f"generation failed safely: {exc}", file=sys.stderr, flush=True)
                limit = args.max_consecutive_failures
                if limit > 0 and int(state["consecutive_failures"]) >= limit:
                    print(
                        f"stopping after {state['consecutive_failures']} consecutive failures; "
                        f"state preserved at {state_path}",
                        file=sys.stderr, flush=True,
                    )
                    return 2
                time.sleep(max(0.0, args.failure_cooldown_seconds))
                continue

            promoted = bool(summary.get("arena", {}).get("promoted", False))
            promoted_to = str(summary.get("promoted_to", ""))
            if promoted:
                if not promoted_to:
                    raise RuntimeError("Arena reported promotion but iterate.py did not publish a Champion")
                promoted_checkpoint = Path(promoted_to).with_suffix(".pt")
                if not promoted_checkpoint.is_file():
                    raise RuntimeError(f"promoted Champion missing: {promoted_checkpoint}")
                state["current_champion"] = str(promoted_checkpoint)

            new_data = Path(summary["new_data"])
            fresh_group = [
                str(Path(path)) for path in summary.get("new_data_shards", [])
            ] or [str(new_data)]
            missing_fresh_group = [path for path in fresh_group if not Path(path).is_file()]
            if missing_fresh_group:
                raise RuntimeError(
                    f"completed generation references missing replay shard: {missing_fresh_group[0]}"
                )
            rolling_groups = [
                [str(Path(path)) for path in group]
                for group in state.get("rolling_replay_groups", [])
            ]
            rolling_groups.append(fresh_group)
            if args.replay_window == 0:
                rolling_groups = []
            elif len(rolling_groups) > args.replay_window:
                rolling_groups = rolling_groups[-args.replay_window:]
            state["rolling_replay_groups"] = rolling_groups
            state["rolling_replay"] = [
                path for group in rolling_groups for path in group
            ]
            state["next_generation"] = generation + 1
            state["consecutive_failures"] = 0
            state.setdefault("completed", []).append({
                "generation": generation,
                "completed_utc": utc_now(),
                "champion_before": str(current_champion),
                "champion_after": str(state["current_champion"]),
                "promoted": promoted,
                "summary": str(summary_path),
                "new_data": str(new_data),
                "new_data_shards": fresh_group,
                "reanalyzed_data": summary.get("reanalyze", {}).get("dataset", ""),
                "reanalyzed_shards": summary.get("reanalyze", {}).get("shards", []),
                "arena": summary.get("arena", {}),
            })
            state.pop("active_attempt", None)
            state["updated_utc"] = utc_now()
            atomic_json(state_path, state)
            successes_this_run += 1

            arena = summary.get("arena", {})
            print(
                f"completed    gen{generation}: "
                f"win={100.0 * float(arena.get('win_rate', 0.0)):.1f}% "
                f"CI=[{100.0 * float(arena.get('ci_lower', 0.0)):.1f}, "
                f"{100.0 * float(arena.get('ci_upper', 0.0)):.1f}] "
                f"promoted={'yes' if promoted else 'no'}",
                flush=True,
            )
            print(f"Champion     {state['current_champion']}", flush=True)
            print(f"state        {state_path}", flush=True)
            if target_successes == 0 or successes_this_run < target_successes:
                time.sleep(max(0.0, args.cooldown_seconds))
    except KeyboardInterrupt:
        state.pop("active_attempt", None)
        state["updated_utc"] = utc_now()
        state["stopped_utc"] = utc_now()
        state["stop_reason"] = "keyboard_interrupt"
        atomic_json(state_path, state)
        print(f"\nstopped cleanly; state preserved at {state_path}", flush=True)
        return 130

    print(
        f"finished      {successes_this_run} successful generations; "
        f"Champion {state['current_champion']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
