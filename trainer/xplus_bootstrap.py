#!/usr/bin/env python3
"""Exact X+ replay bootstrap for the local evaluator.

Default pipeline:
  TETRA CHANNEL X+ cohort
    -> current-v19 fail-closed replay reconstruction
    -> production C++ legal-action/tokenization validation
    -> ~0.95M teacher1m supervised on exact human placement policy
    -> frozen-teacher T=3 policy distillation
    -> 0.13M XS local evaluator

The pipeline deliberately requires a 100% exact import.  A source/round that
cannot be reconstructed or an exact turn that cannot be represented by the
production action space stops training instead of silently reducing the corpus.

Human replay WDL is intentionally disabled for the local evaluator:
in the current exact X+ corpus, local state -> eventual match outcome did not
produce a useful held-out value signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path


MANIFEST_FORMAT = "tetra-human-replay-manifest-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def run(command: list[str], *, root: Path, dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=str(root), check=True)


def rooted(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def runtime_path(root: Path, path: Path) -> str:
    """Prefer a cwd-relative path so WSL can launch a Windows ROCm Python."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def load_manifest(output_dir: Path) -> dict:
    path = output_dir / "manifest.json"
    if not path.exists():
        raise SystemExit(f"missing exact replay manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid exact replay manifest: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_shards(output_dir: Path, items: object) -> list[Path]:
    if not isinstance(items, list) or not items:
        raise SystemExit("exact replay quality gate failed: invalid shard list")
    shards: list[Path] = []
    seen: set[Path] = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise SystemExit("exact replay quality gate failed: invalid shard entry")
        expected_hash = item.get("dataset_sha256")
        if not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(expected_hash):
            raise SystemExit("exact replay quality gate failed: invalid shard sha256")
        # Manifest paths may have been produced by WSL while the learner is a
        # Windows ROCm Python.  The basename plus the selected output dir is the
        # stable cross-runtime identity.
        candidate = output_dir / Path(item["path"].replace("\\", "/")).name
        candidate = candidate.resolve()
        if candidate in seen:
            raise SystemExit("exact replay quality gate failed: duplicate shard path")
        if not candidate.exists():
            raise SystemExit(f"manifest references missing exact shard: {candidate}")
        if sha256_file(candidate) != expected_hash:
            raise SystemExit(f"exact replay quality gate failed: shard hash mismatch: {candidate}")
        seen.add(candidate)
        shards.append(candidate)
    return shards


def nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        result = value
    elif isinstance(value, str) and value.isdecimal():
        result = int(value)
    else:
        raise SystemExit(f"exact replay quality gate failed: invalid {label}")
    if result < 0:
        raise SystemExit(f"exact replay quality gate failed: invalid {label}")
    return result


def verify_exact_gate(output_dir: Path) -> list[Path]:
    manifest = load_manifest(output_dir)
    if manifest.get("format") != MANIFEST_FORMAT:
        raise SystemExit("exact replay quality gate failed: invalid manifest format")
    if manifest.get("normalization_mode") != "exact-v19":
        raise SystemExit(
            "exact replay quality gate failed: manifest was not produced by "
            "the exact-v19 normalizer"
        )
    totals = manifest.get("totals", {}) if isinstance(manifest, dict) else {}
    if not isinstance(totals, dict):
        raise SystemExit("exact replay quality gate failed: invalid totals")
    normalized = nonnegative_int(totals.get("normalized_turns"), "normalized_turns")
    normalized_games = nonnegative_int(totals.get("normalized_games"), "normalized_games")
    imported = nonnegative_int(totals.get("imported_samples"), "imported_samples")
    skipped = nonnegative_int(
        totals.get("skipped_during_cpp_validation"),
        "skipped_during_cpp_validation",
    )
    source_files = nonnegative_int(totals.get("source_files"), "source_files")
    try:
        fraction = float(totals.get("import_fraction", 0.0))
    except (TypeError, ValueError) as exc:
        raise SystemExit("exact replay quality gate failed: invalid totals") from exc
    if not math.isfinite(fraction):
        raise SystemExit("exact replay quality gate failed: non-finite import_fraction")

    sources = manifest.get("sources", [])
    if not isinstance(sources, list) or len(sources) != source_files:
        raise SystemExit("exact replay quality gate failed: inconsistent source list")
    source_errors: list[str] = []
    source_turns = 0
    source_games = 0
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("errors"), list):
            raise SystemExit("exact replay quality gate failed: invalid source entry")
        source_turns += nonnegative_int(source.get("turns"), "source turns")
        source_games += nonnegative_int(source.get("games"), "source games")
        source_errors.extend(str(error) for error in source["errors"])

    shard_items = manifest.get("shards")
    shards = verified_shards(output_dir, shard_items)
    requested_sum = 0
    games_sum = 0
    imported_sum = 0
    skipped_sum = 0
    assert isinstance(shard_items, list)
    for shard in shard_items:
        assert isinstance(shard, dict)
        requested_sum += nonnegative_int(shard.get("requested_turns"), "shard requested_turns")
        games_sum += nonnegative_int(shard.get("games"), "shard games")
        imported_sum += nonnegative_int(shard.get("imported"), "shard imported")
        skipped_sum += sum(
            nonnegative_int(shard.get(name), f"shard {name}")
            for name in ("invalid", "execution", "unmatched")
        )

    if source_turns != normalized or requested_sum != normalized:
        raise SystemExit("exact replay quality gate failed: inconsistent normalized turn counts")
    if source_games != normalized_games or games_sum != normalized_games:
        raise SystemExit("exact replay quality gate failed: inconsistent normalized game counts")
    if imported_sum != imported or skipped_sum != skipped:
        raise SystemExit("exact replay quality gate failed: inconsistent C++ validation counts")

    if normalized <= 0 or imported != normalized or skipped != 0:
        raise SystemExit(
            "exact replay quality gate failed: expected every normalized turn "
            f"to import (normalized={normalized}, imported={imported}, skipped={skipped})"
        )
    expected_fraction = imported / normalized if normalized else 0.0
    if abs(fraction - expected_fraction) > 1e-12 or abs(fraction - 1.0) > 1e-12:
        raise SystemExit(
            f"exact replay quality gate failed: import_fraction={fraction:.6f}"
        )
    if source_errors:
        raise SystemExit(
            "exact replay quality gate failed: a source/round was rejected; "
            f"first error: {source_errors[0]}"
        )
    print(
        f"exact replay gate: import_fraction={fraction:.1%}, "
        f"source_errors={len(source_errors)}, "
        "mode=fail-closed",
        flush=True,
    )
    return shards


def add_common_train_flags(command: list[str], args: argparse.Namespace) -> None:
    command.extend(
        [
            "--device",
            args.device,
            "--threads",
            str(max(1, args.threads)),
        ]
    )
    if args.require_gpu:
        command.append("--require-gpu")
    if args.checkpoint_every > 0:
        command.extend(["--checkpoint-every", str(args.checkpoint_every)])
    if args.eval_every > 0:
        command.extend(["--eval-every", str(args.eval_every)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)

    # Collection. Full-corpus runs are append/resume friendly.
    ap.add_argument("--collection-dir", default="data/xplus_replays")
    ap.add_argument("--skip-collect", action="store_true")
    ap.add_argument("--refresh-cohort", action="store_true")
    ap.add_argument("--refresh-records", action="store_true")
    ap.add_argument("--request-interval", type=float, default=1.05)
    ap.add_argument("--max-players", type=int, default=0)
    ap.add_argument("--max-leaderboard-pages", type=int, default=200)
    ap.add_argument("--max-record-pages", type=int, default=50)
    ap.add_argument("--max-replays", type=int, default=0)
    ap.add_argument("--replay-url-template", default="")

    # Exact import / sharding.
    ap.add_argument("--shard-dir", default="data/xplus_replay_shards")
    ap.add_argument("--cache-dir", default="data/xplus_replay_cache")
    ap.add_argument("--engine", default="")
    ap.add_argument("--samples-per-shard", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--model-version", type=int, default=0)
    ap.add_argument("--skip-import", action="store_true")
    ap.add_argument("--force-import", action="store_true")

    ap.add_argument(
        "--train-python",
        default="",
        help=(
            "learner Python; use .venv-rocm714/Scripts/python.exe on the "
            "RX 9070 XT host"
        ),
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)

    # XS student and distillation optimizer.
    ap.add_argument("--save", default="models/xplus_xs_distilled.pt")
    ap.add_argument("--best-save", default="models/xplus_xs_distilled_best.pt")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)

    # Teacher.
    ap.add_argument("--teacher-save", default="models/xplus_teacher1m.pt")
    ap.add_argument("--teacher-best-save", default="models/xplus_teacher1m_best.pt")
    ap.add_argument("--teacher-resume", default="")
    ap.add_argument("--teacher-steps", type=int, default=5000)
    ap.add_argument("--teacher-batch", type=int, default=256)
    ap.add_argument("--teacher-lr", type=float, default=3e-4)

    # Frozen-teacher categorical policy distillation.
    ap.add_argument("--temperature", type=float, default=3.0)

    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if min(args.steps, args.teacher_steps) < 0:
        ap.error("training steps must be non-negative")
    if min(args.batch, args.teacher_batch, args.samples_per_shard) <= 0:
        ap.error("batch and samples-per-shard values must be positive")
    if (
        args.request_interval < 0
        or args.max_record_pages <= 0
        or args.max_leaderboard_pages <= 0
    ):
        ap.error("collection interval must be non-negative and page limits must be positive")
    if args.temperature <= 0:
        ap.error("--temperature must be positive")

    root = Path(__file__).resolve().parents[1]
    py = sys.executable
    train_py = args.train_python or py
    collection_dir = rooted(root, args.collection_dir)
    shard_dir = rooted(root, args.shard_dir)
    cache_dir = rooted(root, args.cache_dir)

    if not args.skip_collect:
        collect_cmd = [
            py,
            str(root / "trainer/collect_xplus_replays.py"),
            "--output-dir", str(collection_dir),
            "--ranks", "x+",
            "--request-interval", str(args.request_interval),
            "--max-leaderboard-pages", str(args.max_leaderboard_pages),
            "--max-record-pages", str(args.max_record_pages),
            "--max-players", str(max(0, args.max_players)),
            "--max-replays", str(max(0, args.max_replays)),
        ]
        if args.replay_url_template:
            collect_cmd.extend(["--replay-url-template", args.replay_url_template])
        if args.refresh_cohort:
            collect_cmd.append("--refresh-cohort")
        if args.refresh_records:
            collect_cmd.append("--refresh-records")
        collect_cmd.append("--strict")
        run(collect_cmd, root=root, dry_run=args.dry_run)

    if not args.skip_import:
        import_cmd = [
            py,
            str(root / "trainer/import_human_replays.py"),
            str(collection_dir),
            "--output-dir", str(shard_dir),
            "--cache-dir", str(cache_dir),
            "--samples-per-shard", str(args.samples_per_shard),
            "--workers", str(max(1, args.workers)),
            "--ruleset", "league",
            "--model-version", str(max(0, args.model_version)),
            "--exact",
            "--strict-source",
        ]
        if args.engine:
            import_cmd.extend(["--engine", args.engine])
        if args.force_import:
            import_cmd.append("--force")
        run(import_cmd, root=root, dry_run=args.dry_run)

    if args.dry_run:
        shards = [runtime_path(root, shard_dir / "human_*.tetradat")]
    else:
        shards = [runtime_path(root, path) for path in verify_exact_gate(shard_dir)]

    teacher_cmd = [
        train_py,
        "trainer/train.py",
        *shards,
        "--model", "teacher1m",
        "--steps", str(args.teacher_steps),
        "--batch", str(args.teacher_batch),
        "--lr", str(args.teacher_lr),
        "--policy-weight", "1.0",
        "--value-weight", "0.0",
        "--aux-weight", "0.0",
        "--save", args.teacher_save,
    ]
    if args.teacher_resume:
        teacher_cmd.extend(["--resume", args.teacher_resume])
    if args.teacher_best_save:
        teacher_cmd.extend(["--best-save", args.teacher_best_save])
    add_common_train_flags(teacher_cmd, args)
    run(teacher_cmd, root=root, dry_run=args.dry_run)

    teacher_for_distill = args.teacher_best_save or args.teacher_save
    distill_cmd = [
        train_py,
        "trainer/distill.py",
        *shards,
        "--teacher", teacher_for_distill,
        "--student", "xs",
        "--steps", str(args.steps),
        "--batch", str(args.batch),
        "--lr", str(args.lr),
        "--temperature", str(args.temperature),
        "--policy-weight", "1.0",
        "--value-kl-weight", "0.0",
        "--value-mse-weight", "0.0",
        "--save", args.save,
    ]
    add_common_train_flags(distill_cmd, args)
    if args.best_save:
        distill_cmd.extend(["--best-save", args.best_save])
    run(distill_cmd, root=root, dry_run=args.dry_run)

    print(
        f"X+ exact distillation complete: teacher={teacher_for_distill} student={args.save}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
