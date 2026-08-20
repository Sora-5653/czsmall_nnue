#!/usr/bin/env python3
"""One-command human replay bootstrap: .ttrm -> .tetradat -> policy/value checkpoint.

Human replay samples intentionally carry no local timing/auxiliary supervision.
They teach placement/hold policy and game outcome value; strategic timing and
auxiliary heads remain the job of self-play/Reanalyze after bootstrap.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str], root: Path, dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=str(root), check=True)


def manifest_shards(output_dir: Path) -> list[Path]:
    """Return exactly the shards selected by the latest import manifest.

    Old shard files are intentionally not deleted: a changed sharding target can
    leave earlier generated files in the directory, and globbing all of them
    would duplicate examples.  Manifest paths may have been written by WSL or
    Windows, so only the filename is carried across the runtime boundary.
    """
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return sorted(output_dir.glob("*.tetradat")) if output_dir.exists() else []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("shards", []) if isinstance(manifest, dict) else []
    if not isinstance(items, list):
        raise SystemExit(f"invalid shard list in {manifest_path}")
    shards: list[Path] = []
    seen: set[Path] = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        name = Path(item["path"].replace("\\", "/")).name
        candidate = output_dir / name
        if candidate not in seen:
            shards.append(candidate)
            seen.add(candidate)
    missing = [path for path in shards if not path.exists()]
    if missing:
        raise SystemExit(f"manifest references missing human replay shard: {missing[0]}")
    return shards


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help=".ttrm files/directories")
    ap.add_argument("--output-dir", default="data/human_replay_shards")
    ap.add_argument("--cache-dir", default="data/human_replay_cache")
    ap.add_argument("--engine", default="")
    ap.add_argument("--samples-per-shard", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--ruleset", choices=("league", "quickplay", "guideline"), default="league")
    ap.add_argument("--model-version", type=int, default=0)
    ap.add_argument("--force-import", action="store_true")
    ap.add_argument("--strict-source", action="store_true")
    ap.add_argument(
        "--min-import-fraction",
        type=float,
        default=0.0,
        help="refuse training when validated samples / normalized hard-drop turns falls below this fraction",
    )
    ap.add_argument("--skip-import", action="store_true", help="train from existing shards only")

    ap.add_argument("--save", default="models/human_pretrain.pt")
    ap.add_argument("--resume", default="")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument(
        "--model",
        choices=("dev", "xs", "s"),
        default="xs",
        help="model preset; xs is the named 64x2 lightweight transformer used by the size-ablation baseline",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--best-save", default="")
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--value-weight", type=float, default=0.25)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.steps < 0 or args.batch <= 0 or args.samples_per_shard <= 0:
        raise SystemExit("steps must be non-negative; batch and samples-per-shard must be positive")
    if not 0.0 <= args.min_import_fraction <= 1.0:
        raise SystemExit("--min-import-fraction must be in [0, 1]")

    root = Path(__file__).resolve().parents[1]
    py = sys.executable
    output_dir = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    cache_dir = (root / args.cache_dir).resolve() if not Path(args.cache_dir).is_absolute() else Path(args.cache_dir)
    save = (root / args.save).resolve() if not Path(args.save).is_absolute() else Path(args.save)
    save.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_import:
        import_cmd = [
            py,
            str(root / "trainer/import_human_replays.py"),
            *args.inputs,
            "--output-dir", str(output_dir),
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
        if args.strict_source:
            import_cmd.append("--strict-source")
        run(import_cmd, root, args.dry_run)

    if not args.dry_run and args.min_import_fraction > 0.0:
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"missing human replay manifest for quality gate: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        totals = manifest.get("totals", {}) if isinstance(manifest, dict) else {}
        fraction = float(totals.get("import_fraction", 0.0))
        if fraction < args.min_import_fraction:
            raise SystemExit(
                f"human replay quality gate failed: import_fraction={fraction:.3f} "
                f"< required {args.min_import_fraction:.3f}; inspect reconstruction before training"
            )
        print(f"human replay quality gate: import_fraction={fraction:.1%}", flush=True)

    shards = manifest_shards(output_dir)
    if args.dry_run and not shards:
        shards = [output_dir / "human_*.tetradat"]
    if not shards:
        raise SystemExit(f"no .tetradat human shards found in {output_dir}")

    train_cmd = [
        py,
        str(root / "trainer/train.py"),
        *[str(path) for path in shards],
        "--steps", str(args.steps),
        "--batch", str(args.batch),
        "--lr", str(args.lr),
        "--model", args.model,
        "--device", args.device,
        "--threads", str(max(1, args.threads)),
        "--policy-weight", str(args.policy_weight),
        "--value-weight", str(args.value_weight),
        "--aux-weight", "0.0",
        "--vs-aux-weight", "0.0",
        "--cancellation-aux-weight", "0.0",
        "--timing-pair-weight", "0.0",
        "--timing-rank-weight", "0.0",
        "--save", str(save),
    ]
    if args.resume:
        train_cmd.extend(["--resume", args.resume])
    if args.require_gpu:
        train_cmd.append("--require-gpu")
    if args.checkpoint_every > 0:
        train_cmd.extend(["--checkpoint-every", str(args.checkpoint_every)])
    if args.eval_every > 0:
        train_cmd.extend(["--eval-every", str(args.eval_every)])
    if args.best_save:
        train_cmd.extend(["--best-save", args.best_save])
    run(train_cmd, root, args.dry_run)

    print(f"human bootstrap checkpoint: {save}", flush=True)
    print(
        "next: use the checkpoint as --champion and the shard directory as "
        "--bootstrap-replay-dir in trainer/auto_improve.py",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
