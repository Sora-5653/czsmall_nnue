#!/usr/bin/env python3
"""Exact X+ replay bootstrap for the local evaluator.

Default pipeline:
  TETRA CHANNEL X+ cohort
    -> current-v19 fail-closed replay reconstruction
    -> production C++ legal-action/tokenization validation
    -> ~0.95M teacher1m supervised on exact human placement policy
    -> frozen-teacher T=3 policy distillation
    -> 0.13M XS local evaluator

The default path deliberately requires a 100% exact import.  A source/round that
cannot be reconstructed or an exact turn that cannot be represented by the
production action space stops training instead of silently reducing the corpus.
`--allow-partial-exact` exists only for explicit diagnostics/ablations.

Human replay WDL is intentionally disabled by default for the local evaluator:
in the current exact X+ corpus, local state -> eventual match outcome did not
produce a useful held-out value signal.  Value distillation remains configurable
for a future local value target/teacher.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


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


def manifest_shards(output_dir: Path, root: Path) -> list[str]:
    manifest = load_manifest(output_dir)
    items = manifest.get("shards", [])
    if not isinstance(items, list):
        raise SystemExit("invalid shard list in exact replay manifest")
    shards: list[str] = []
    seen: set[Path] = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        # Manifest paths may have been produced by WSL while the learner is a
        # Windows ROCm Python.  The basename plus the selected output dir is the
        # stable cross-runtime identity.
        candidate = output_dir / Path(item["path"].replace("\\", "/")).name
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        if not candidate.exists():
            raise SystemExit(f"manifest references missing exact shard: {candidate}")
        seen.add(candidate)
        shards.append(runtime_path(root, candidate))
    if not shards:
        raise SystemExit(f"no exact .tetradat shards in {output_dir}")
    return shards


def verify_exact_gate(
    output_dir: Path,
    *,
    min_fraction: float,
    allow_partial: bool,
) -> None:
    manifest = load_manifest(output_dir)
    totals = manifest.get("totals", {}) if isinstance(manifest, dict) else {}
    fraction = float(totals.get("import_fraction", 0.0)) if isinstance(totals, dict) else 0.0
    sources = manifest.get("sources", [])
    source_errors: list[str] = []
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            errors = source.get("errors", [])
            if isinstance(errors, list) and errors:
                source_errors.extend(str(error) for error in errors)

    required = min_fraction if allow_partial else 1.0
    if fraction + 1e-12 < required:
        raise SystemExit(
            f"exact replay quality gate failed: import_fraction={fraction:.6f} < {required:.6f}"
        )
    if source_errors and not allow_partial:
        raise SystemExit(
            "exact replay quality gate failed: at least one source/round was rejected; "
            f"first error: {source_errors[0]}"
        )
    print(
        f"exact replay gate: import_fraction={fraction:.1%}, source_errors={len(source_errors)}, "
        f"mode={'partial-opt-in' if allow_partial else 'fail-closed'}",
        flush=True,
    )


def add_common_train_flags(command: list[str], args: argparse.Namespace) -> None:
    command.extend([
        "--device", args.device,
        "--threads", str(max(1, args.threads)),
    ])
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
    ap.add_argument("--strict-collect", action="store_true")

    # Exact import / sharding.
    ap.add_argument("--shard-dir", default="data/xplus_replay_shards")
    ap.add_argument("--cache-dir", default="data/xplus_replay_cache")
    ap.add_argument("--engine", default="")
    ap.add_argument("--samples-per-shard", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--ruleset", choices=("league",), default="league")
    ap.add_argument("--model-version", type=int, default=0)
    ap.add_argument("--skip-import", action="store_true")
    ap.add_argument("--force-import", action="store_true")
    ap.add_argument(
        "--allow-partial-exact",
        action="store_true",
        help="diagnostic opt-in: permit rejected rounds/turns instead of the default 100%% exact gate",
    )
    ap.add_argument(
        "--min-import-fraction",
        type=float,
        default=1.0,
        help="minimum fraction only when --allow-partial-exact is set; default exact path requires 1.0",
    )

    # Training mode. Distillation is the production bootstrap; direct remains a
    # paired XS baseline.
    ap.add_argument("--training-mode", choices=("distill", "direct"), default="distill")
    ap.add_argument("--train-python", default="",
                    help="learner Python; use .venv-rocm714/Scripts/python.exe on the RX 9070 XT host")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)

    # Student / direct baseline options. In distill mode these configure the XS
    # student and distillation optimizer.
    ap.add_argument("--model", choices=("xs", "dev", "s"), default="xs")
    ap.add_argument("--save", default="models/xplus_xs_distilled.pt")
    ap.add_argument("--best-save", default="models/xplus_xs_distilled_best.pt")
    ap.add_argument("--resume", default="", help="direct-baseline resume checkpoint")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4,
                    help="student/distillation LR; direct mode may prefer 3e-4")
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--value-weight", type=float, default=0.0,
                    help="direct baseline WDL weight; local bootstrap default is policy-only")

    # Teacher.
    ap.add_argument("--teacher-save", default="models/xplus_teacher1m.pt")
    ap.add_argument("--teacher-best-save", default="models/xplus_teacher1m_best.pt")
    ap.add_argument("--teacher-resume", default="")
    ap.add_argument("--teacher-steps", type=int, default=5000)
    ap.add_argument("--teacher-batch", type=int, default=256)
    ap.add_argument("--teacher-lr", type=float, default=3e-4)
    ap.add_argument("--teacher-policy-weight", type=float, default=1.0)
    ap.add_argument("--teacher-value-weight", type=float, default=0.0)

    # MochBot-style frozen-teacher categorical distillation.
    ap.add_argument("--temperature", type=float, default=3.0)
    ap.add_argument("--distill-policy-weight", type=float, default=1.0)
    ap.add_argument("--distill-value-kl-weight", type=float, default=0.0)
    ap.add_argument("--distill-value-mse-weight", type=float, default=0.0)

    ap.add_argument("--collection-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if min(args.steps, args.teacher_steps) < 0:
        ap.error("training steps must be non-negative")
    if min(args.batch, args.teacher_batch, args.samples_per_shard) <= 0:
        ap.error("batch and samples-per-shard values must be positive")
    if args.request_interval < 0 or args.max_record_pages <= 0 or args.max_leaderboard_pages <= 0:
        ap.error("collection interval must be non-negative and page limits must be positive")
    if not 0.0 <= args.min_import_fraction <= 1.0:
        ap.error("--min-import-fraction must be in [0, 1]")
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
        if args.strict_collect:
            collect_cmd.append("--strict")
        run(collect_cmd, root=root, dry_run=args.dry_run)

    if args.collection_only:
        return 0

    if not args.skip_import:
        import_cmd = [
            py,
            str(root / "trainer/import_human_replays.py"),
            str(collection_dir),
            "--output-dir", str(shard_dir),
            "--cache-dir", str(cache_dir),
            "--samples-per-shard", str(args.samples_per_shard),
            "--workers", str(max(1, args.workers)),
            "--ruleset", args.ruleset,
            "--model-version", str(max(0, args.model_version)),
        ]
        if args.engine:
            import_cmd.extend(["--engine", args.engine])
        if args.force_import:
            import_cmd.append("--force")
        if not args.allow_partial_exact:
            import_cmd.append("--strict-source")
        run(import_cmd, root=root, dry_run=args.dry_run)

    if args.dry_run:
        shards = [runtime_path(root, shard_dir / "human_*.tetradat")]
    else:
        verify_exact_gate(
            shard_dir,
            min_fraction=args.min_import_fraction,
            allow_partial=args.allow_partial_exact,
        )
        shards = manifest_shards(shard_dir, root)

    if args.training_mode == "direct":
        direct_cmd = [
            train_py,
            "trainer/train.py",
            *shards,
            "--model", args.model,
            "--steps", str(args.steps),
            "--batch", str(args.batch),
            "--lr", str(args.lr),
            "--policy-weight", str(args.policy_weight),
            "--value-weight", str(args.value_weight),
            "--aux-weight", "0.0",
            "--save", args.save,
        ]
        if args.resume:
            direct_cmd.extend(["--resume", args.resume])
        if args.best_save:
            direct_cmd.extend(["--best-save", args.best_save])
        add_common_train_flags(direct_cmd, args)
        run(direct_cmd, root=root, dry_run=args.dry_run)
        print(f"X+ direct baseline complete: model={args.model} checkpoint={args.save}", flush=True)
        return 0

    teacher_cmd = [
        train_py,
        "trainer/train.py",
        *shards,
        "--model", "teacher1m",
        "--steps", str(args.teacher_steps),
        "--batch", str(args.teacher_batch),
        "--lr", str(args.teacher_lr),
        "--policy-weight", str(args.teacher_policy_weight),
        "--value-weight", str(args.teacher_value_weight),
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
        "--policy-weight", str(args.distill_policy_weight),
        "--value-kl-weight", str(args.distill_value_kl_weight),
        "--value-mse-weight", str(args.distill_value_mse_weight),
        "--device", args.device,
        "--threads", str(max(1, args.threads)),
        "--save", args.save,
    ]
    if args.require_gpu:
        distill_cmd.append("--require-gpu")
    if args.checkpoint_every > 0:
        distill_cmd.extend(["--checkpoint-every", str(args.checkpoint_every)])
    if args.eval_every > 0:
        distill_cmd.extend(["--eval-every", str(args.eval_every)])
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
