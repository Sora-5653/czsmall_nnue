#!/usr/bin/env python3
"""Bulk-import TETR.IO .ttrm files into cached .tetradat human shards.

The expensive/stable boundary is content-addressed:
  .ttrm -> normalized replay cache -> C++ legality/tokenization -> .tetradat
Changing Python normalization invalidates the protocol cache version.  Changing
the C++ schema naturally invalidates downstream training via the dataset
contract embedded in every shard.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

try:
    from .ttrm_ingest import normalize_file, render_game
except ImportError:  # direct `python trainer/import_human_replays.py`
    from ttrm_ingest import normalize_file, render_game

CACHE_VERSION = 5
IMPORT_RE = re.compile(
    r"games=(?P<games>\d+) turns=(?P<turns>\d+) imported=(?P<imported>\d+) "
    r"invalid=(?P<invalid>\d+) execution=(?P<execution>\d+) unmatched=(?P<unmatched>\d+)"
)


@dataclass(frozen=True)
class CachedGame:
    source: str
    source_hash: str
    round_index: int
    samples: int
    text: str


@dataclass
class SourceReport:
    path: str
    source_hash: str = ""
    games: int = 0
    turns: int = 0
    errors: list[str] | None = None
    cache_hit: bool = False


@dataclass
class ShardReport:
    path: str
    protocol_hash: str
    requested_turns: int
    games: int
    imported: int = 0
    invalid: int = 0
    execution: int = 0
    unmatched: int = 0
    cache_hit: bool = False


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(inputs: Sequence[Path]) -> list[Path]:
    found: set[Path] = set()
    for entry in inputs:
        if entry.is_file() and entry.suffix.lower() == ".ttrm":
            found.add(entry.resolve())
        elif entry.is_dir():
            found.update(path.resolve() for path in entry.rglob("*.ttrm") if path.is_file())
    return sorted(found)


def _normalize_one(path_text: str, cache_dir_text: str, force: bool) -> tuple[list[CachedGame], SourceReport]:
    path = Path(path_text)
    cache_dir = Path(cache_dir_text)
    raw = path.read_bytes()
    source_hash = sha256_bytes(raw)
    cache_base = cache_dir / "normalized" / f"v{CACHE_VERSION}_{source_hash}"
    replay_cache = cache_base.with_suffix(".replay.json")
    meta_cache = cache_base.with_suffix(".meta.json")
    report = SourceReport(str(path), source_hash=source_hash, errors=[])

    if not force and replay_cache.exists() and meta_cache.exists():
        try:
            payload = json.loads(replay_cache.read_text(encoding="utf-8"))
            meta = json.loads(meta_cache.read_text(encoding="utf-8"))
            games = [
                CachedGame(str(path), source_hash, int(item["round_index"]), int(item["samples"]), str(item["text"]))
                for item in payload
            ]
            report.games = len(games)
            report.turns = sum(game.samples for game in games)
            report.errors = list(meta.get("errors", []))
            report.cache_hit = True
            return games, report
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass

    games, errors, _source_id = normalize_file(path, require_exact=True)
    cached = [CachedGame(str(path), source_hash, game.round_index, game.samples, render_game(game)) for game in games]
    report.games = len(cached)
    report.turns = sum(game.samples for game in cached)
    report.errors = errors

    replay_cache.parent.mkdir(parents=True, exist_ok=True)
    replay_cache.write_text(
        json.dumps(
            [{"round_index": game.round_index, "samples": game.samples, "text": game.text} for game in cached],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    meta_cache.write_text(
        json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "source": str(path),
                "source_hash": source_hash,
                "exact_replay_required": True,
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return cached, report


def normalize_sources(paths: Sequence[Path], cache_dir: Path, workers: int, force: bool) -> tuple[list[CachedGame], list[SourceReport]]:
    games: list[CachedGame] = []
    reports: list[SourceReport] = []
    worker_count = max(1, min(workers, len(paths) or 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(_normalize_one, str(path), str(cache_dir), force) for path in paths]
        for future in concurrent.futures.as_completed(futures):
            cached, report = future.result()
            games.extend(cached)
            reports.append(report)
    games.sort(key=lambda game: (game.source, game.round_index))
    reports.sort(key=lambda report: report.path)
    return games, reports


def pack_shards(games: Sequence[CachedGame], target_turns: int) -> list[list[CachedGame]]:
    shards: list[list[CachedGame]] = []
    current: list[CachedGame] = []
    turns = 0
    for game in games:
        if current and turns >= target_turns:
            shards.append(current)
            current = []
            turns = 0
        current.append(game)
        turns += game.samples
    if current:
        shards.append(current)
    return shards


def resolve_engine(explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit.resolve()
        raise FileNotFoundError(f"engine not found: {explicit}")
    root = Path(__file__).resolve().parent.parent
    candidates = (
        root / "build" / "tetra_cli",
        root / "build" / "tetra_cli.exe",
        root / "build-human" / "tetra_cli",
        root / "build-human" / "tetra_cli.exe",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("tetra_cli not found; run `make tools` (or an isolated BUILD=... tools) first")


def _parse_import_stats(output: str) -> dict[str, int]:
    match = IMPORT_RE.search(output)
    if not match:
        return {}
    return {name: int(value) for name, value in match.groupdict().items()}


def build_shards(
    shards: Sequence[Sequence[CachedGame]],
    output_dir: Path,
    cache_dir: Path,
    engine: Path,
    model_version: int,
    ruleset: str,
    force: bool,
) -> list[ShardReport]:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_dir = cache_dir / "shards"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    reports: list[ShardReport] = []

    for index, shard in enumerate(shards):
        protocol = "".join(game.text for game in shard)
        protocol_hash = sha256_bytes(protocol.encode("utf-8"))
        protocol_path = protocol_dir / f"{protocol_hash}.replay"
        protocol_path.write_text(protocol, encoding="utf-8", newline="\n")
        output_path = output_dir / f"human_{index:05d}_{protocol_hash[:12]}.tetradat"
        sidecar = output_path.with_suffix(".tetradat.json")
        requested = sum(game.samples for game in shard)
        report = ShardReport(str(output_path), protocol_hash, requested, len(shard))

        if not force and output_path.exists() and sidecar.exists():
            try:
                old = json.loads(sidecar.read_text(encoding="utf-8"))
                if (
                    old.get("protocol_hash") == protocol_hash
                    and int(old.get("model_version", -1)) == model_version
                    and old.get("ruleset") == ruleset
                ):
                    report.imported = int(old.get("imported", 0))
                    report.invalid = int(old.get("invalid", 0))
                    report.execution = int(old.get("execution", 0))
                    report.unmatched = int(old.get("unmatched", 0))
                    report.cache_hit = True
                    reports.append(report)
                    continue
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

        proc = subprocess.run(
            [str(engine), "import-human-replay", str(protocol_path), str(output_path), str(model_version), ruleset],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(proc.stdout, end="")
        if proc.returncode != 0:
            raise RuntimeError(f"C++ human replay import failed for shard {index}: exit {proc.returncode}")
        stats = _parse_import_stats(proc.stdout)
        report.imported = stats.get("imported", 0)
        report.invalid = stats.get("invalid", 0)
        report.execution = stats.get("execution", 0)
        report.unmatched = stats.get("unmatched", 0)
        sidecar.write_text(
            json.dumps(
                {
                    "protocol_hash": protocol_hash,
                    "model_version": model_version,
                    "ruleset": ruleset,
                    "requested_turns": requested,
                    "games": len(shard),
                    "imported": report.imported,
                    "invalid": report.invalid,
                    "execution": report.execution,
                    "unmatched": report.unmatched,
                    "dataset_sha256": sha256_file(output_path),
                    "sources": sorted({game.source for game in shard}),
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        reports.append(report)
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help=".ttrm files or directories containing them")
    parser.add_argument("--output-dir", type=Path, default=Path("data/human_replay_shards"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/human_replay_cache"))
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--samples-per-shard", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--model-version", type=int, default=0)
    parser.add_argument("--ruleset", choices=("league", "quickplay", "guideline"), default="league")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict-source", action="store_true", help="fail if any source/round cannot be normalized")
    args = parser.parse_args()

    if args.samples_per_shard <= 0:
        parser.error("--samples-per-shard must be positive")
    paths = discover(args.inputs)
    if not paths:
        print("no .ttrm files found", file=sys.stderr)
        return 2
    engine = resolve_engine(args.engine)
    print(f"discovered {len(paths)} .ttrm files; engine={engine}")

    games, source_reports = normalize_sources(paths, args.cache_dir, args.workers, args.force)
    source_errors = sum(len(report.errors or []) for report in source_reports)
    total_turns = sum(game.samples for game in games)
    cache_hits = sum(report.cache_hit for report in source_reports)
    print(
        f"normalized {len(games)} games / {total_turns} hard-drop turns; "
        f"source cache hits={cache_hits}/{len(source_reports)}; warnings={source_errors}"
    )
    if args.strict_source and source_errors:
        return 1
    if not games:
        print("no usable replay games after normalization", file=sys.stderr)
        return 1

    packed = pack_shards(games, args.samples_per_shard)
    shard_reports = build_shards(
        packed, args.output_dir, args.cache_dir, engine,
        args.model_version, args.ruleset, args.force,
    )
    imported = sum(report.imported for report in shard_reports)
    skipped = sum(report.invalid + report.execution + report.unmatched for report in shard_reports)
    import_fraction = (imported / total_turns) if total_turns else 0.0

    manifest = {
        "format": "tetra-human-replay-manifest-v1",
        "normalizer_cache_version": CACHE_VERSION,
        "ruleset": args.ruleset,
        "model_version": args.model_version,
        "sources": [asdict(report) for report in source_reports],
        "shards": [asdict(report) for report in shard_reports],
        "totals": {
            "source_files": len(paths),
            "normalized_games": len(games),
            "normalized_turns": total_turns,
            "imported_samples": imported,
            "skipped_during_cpp_validation": skipped,
            "import_fraction": import_fraction,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {len(shard_reports)} shards / {imported} validated samples "
        f"({import_fraction:.1%} of normalized turns); manifest={manifest_path}"
    )
    if imported == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
