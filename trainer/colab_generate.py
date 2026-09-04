# SPDX-License-Identifier: MIT
"""Generate and validate reproducible Colab self-play shards.

Colab is only a GPU worker in this project: the C++ engine still owns the
rules, Cobra move generation, search and dataset serialization.  This wrapper
adds the bookkeeping needed when several independent instances generate data
for the same training generation.

Generate one shard::

    python trainer/colab_generate.py generate models/champion.pt \
        data/colab/shard-0.tetradat --base-seed 100000 \
        --shard-id 0 --shard-count 4 --games 32 --pieces 300 --sims 64 \
        --model-version 4 --device cuda --build-engine

Validate downloaded shards before training::

    python trainer/colab_generate.py validate \
        data/colab/shard-0.tetradat.manifest.json \
        data/colab/shard-1.tetradat.manifest.json --require-complete

The manifest is deliberately JSON and the dataset is the rectangular v3
`.tetradat` format with explicit schema and termination metadata, so the normal
trainer can consume each validated file as a separate input. Legacy v1 files
remain readable for migration and comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
from typing import Any, Iterable


MANIFEST_FORMAT = "czsmall_nnue.colab-shard"
MANIFEST_VERSION = 2
DATASET_MAGIC = b"TETRADAT"
LEGACY_DATASET_VERSION = 1
DATASET_VERSION = 3
UINT64_LIMIT = 1 << 64

# magic, version, samples, max_tokens, max_actions, token_features,
# action_features, aux_targets, ruleset_hash, model_version
DATASET_HEADER = struct.Struct("<8s7IQI")
DATASET_CONTRACT = struct.Struct("<IIQQIIIIQQ")

TOKEN_KIND_ORDER = [
    "row", "col", "board", "active", "hold", "next", "garbage",
    "counters", "event", "rule", "time", "opp_row", "opp_col", "opp_board",
    "missing", "bag", "opp_counters",
]
AUX_TARGET_NAMES = [
    "legacy_future_attack_1s", "legacy_future_garbage_received",
    "legacy_time_to_terminal", "legacy_topped_out_within_8",
    "real_0_1s_attack", "real_0_1s_garbage_received", "real_0_1s_self_topout",
    "real_0_1s_opponent_topout", "real_1_2s_attack", "real_1_2s_garbage_received",
    "real_1_2s_self_topout", "real_1_2s_opponent_topout", "real_2_4s_attack",
    "real_2_4s_garbage_received", "real_2_4s_self_topout",
    "real_2_4s_opponent_topout", "real_4_8s_attack", "real_4_8s_garbage_received",
    "real_4_8s_self_topout", "real_4_8s_opponent_topout", "placements_0_1_attack",
    "placements_0_1_garbage_received", "placements_0_1_self_topout",
    "placements_0_1_opponent_topout", "placements_1_2_attack",
    "placements_1_2_garbage_received", "placements_1_2_self_topout",
    "placements_1_2_opponent_topout", "placements_2_4_attack",
    "placements_2_4_garbage_received", "placements_2_4_self_topout",
    "placements_2_4_opponent_topout", "placements_4_8_attack",
    "placements_4_8_garbage_received", "placements_4_8_self_topout",
    "placements_4_8_opponent_topout",
    "real_0_1s_garbage_cleared", "real_1_2s_garbage_cleared",
    "real_2_4s_garbage_cleared", "real_4_8s_garbage_cleared",
    "placements_0_1_garbage_cleared", "placements_1_2_garbage_cleared",
    "placements_2_4_garbage_cleared", "placements_4_8_garbage_cleared",
    "real_0_1s_garbage_cancelled", "real_1_2s_garbage_cancelled",
    "real_2_4s_garbage_cancelled", "real_4_8s_garbage_cancelled",
    "placements_0_1_garbage_cancelled", "placements_1_2_garbage_cancelled",
    "placements_2_4_garbage_cancelled", "placements_4_8_garbage_cancelled",
]
TOKENIZER_SCHEMA_VERSION = 2
TOKENIZER_SCHEMA_HASH = 0x5F1E2C9A7B43D816
OBSERVATION_SCHEMA_HASH = 0x8C74B1E2D6093A5F
ACTION_SCHEMA_VERSION = 1
INTERVAL_AUX_TARGET_SCHEMA_VERSION = 2
GARBAGE_CLEAR_AUX_TARGET_SCHEMA_VERSION = 3
AUX_TARGET_SCHEMA_VERSION = 4


class ManifestError(ValueError):
    """Raised when a shard or its provenance cannot be trusted."""


@dataclass(frozen=True)
class DatasetHeader:
    version: int
    samples: int
    max_tokens: int
    max_actions: int
    token_features: int
    action_features: int
    aux_targets: int
    ruleset_hash: int
    model_version: int
    contract_version: int = 0
    tokenizer_schema_version: int = TOKENIZER_SCHEMA_VERSION
    tokenizer_schema_hash: int = TOKENIZER_SCHEMA_HASH
    observation_schema_hash: int = OBSERVATION_SCHEMA_HASH
    action_schema_version: int = ACTION_SCHEMA_VERSION
    aux_target_schema_version: int = 1
    randomizer_type: int = 0
    termination_reason: int = 0
    self_play_seed: int = 0
    token_kind_order_hash: int = TOKENIZER_SCHEMA_HASH


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_dataset_size(header: DatasetHeader) -> int:
    n = header.samples
    floats = n * (
        header.max_tokens * header.token_features
        + header.max_tokens
        + header.max_actions * header.action_features
        + header.max_actions
        + header.max_actions
        + 1
        + header.aux_targets
    )
    metadata = 0
    header_bytes = DATASET_HEADER.size
    if header.version == DATASET_VERSION:
        floats += n * header.aux_targets  # aux_valid_mask
        metadata = n * (4 + 4 + 8 + 4)
        header_bytes += DATASET_CONTRACT.size
    return header_bytes + floats * 4 + metadata


def read_dataset_header(path: Path) -> DatasetHeader:
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"dataset not found: {path}")
    with path.open("rb") as fh:
        raw = fh.read(DATASET_HEADER.size)
    if len(raw) != DATASET_HEADER.size:
        raise ManifestError(f"dataset header is truncated: {path}")

    values = DATASET_HEADER.unpack(raw)
    if values[0] != DATASET_MAGIC:
        raise ManifestError(f"not a .tetradat file: {path}")
    version = values[1]
    if version not in (LEGACY_DATASET_VERSION, DATASET_VERSION):
        raise ManifestError(
            f"Colab shards must be rectangular dataset v1 or v3, got v{version}: {path}"
        )
    extra: dict[str, int] = {}
    if version == DATASET_VERSION:
        with path.open("rb") as fh:
            fh.seek(DATASET_HEADER.size)
            extension = fh.read(DATASET_CONTRACT.size)
        if len(extension) != DATASET_CONTRACT.size:
            raise ManifestError(f"dataset contract header is truncated: {path}")
        values_ext = DATASET_CONTRACT.unpack(extension)
        extra = {
            "contract_version": values_ext[0],
            "tokenizer_schema_version": values_ext[1],
            "tokenizer_schema_hash": values_ext[2],
            "observation_schema_hash": values_ext[3],
            "action_schema_version": values_ext[4],
            "aux_target_schema_version": values_ext[5],
            "randomizer_type": values_ext[6],
            "termination_reason": values_ext[7],
            "self_play_seed": values_ext[8],
            "token_kind_order_hash": values_ext[9],
        }
        if extra["contract_version"] != 1:
            raise ManifestError(f"unsupported dataset contract version: {path}")
    header = DatasetHeader(
        version=version,
        samples=values[2],
        max_tokens=values[3],
        max_actions=values[4],
        token_features=values[5],
        action_features=values[6],
        aux_targets=values[7],
        ruleset_hash=values[8],
        model_version=values[9],
        **extra,
    )
    if header.samples <= 0 or header.max_tokens <= 0 or header.max_actions <= 0:
        raise ManifestError(f"dataset has no usable samples or dimensions: {path}")
    if header.token_features <= 0 or header.action_features <= 0 or header.aux_targets <= 0:
        raise ManifestError(f"dataset has invalid feature widths: {path}")
    if header.version == DATASET_VERSION:
        if (header.tokenizer_schema_version != TOKENIZER_SCHEMA_VERSION or
                header.tokenizer_schema_hash != TOKENIZER_SCHEMA_HASH or
                header.token_kind_order_hash != TOKENIZER_SCHEMA_HASH or
                header.observation_schema_hash != OBSERVATION_SCHEMA_HASH or
                header.action_schema_version != ACTION_SCHEMA_VERSION):
            raise ManifestError(f"tokenizer/observation/action schema mismatch: {path}")
        if header.aux_target_schema_version == AUX_TARGET_SCHEMA_VERSION and header.aux_targets != 52:
            raise ManifestError(f"aux target width does not match schema v4: {path}")
        if (header.aux_target_schema_version == GARBAGE_CLEAR_AUX_TARGET_SCHEMA_VERSION and
                header.aux_targets != 44):
            raise ManifestError(f"aux target width does not match schema v3: {path}")
        if (header.aux_target_schema_version == INTERVAL_AUX_TARGET_SCHEMA_VERSION and
                header.aux_targets != 36):
            raise ManifestError(f"aux target width does not match schema v2: {path}")
        if header.aux_target_schema_version == 1 and header.aux_targets != 4:
            raise ManifestError(f"legacy aux target width does not match schema v1: {path}")
        if header.aux_target_schema_version not in (
                1, INTERVAL_AUX_TARGET_SCHEMA_VERSION,
                GARBAGE_CLEAR_AUX_TARGET_SCHEMA_VERSION, AUX_TARGET_SCHEMA_VERSION):
            raise ManifestError(f"unknown aux target schema: {path}")
    actual_size = path.stat().st_size
    expected_size = _expected_dataset_size(header)
    if actual_size != expected_size:
        raise ManifestError(
            f"dataset size mismatch for {path}: {actual_size} bytes, expected {expected_size}"
        )
    return header


def compute_seed_interval(
    base_seed: int, shard_id: int, shard_count: int, games_per_shard: int
) -> tuple[int, int]:
    if base_seed < 0 or base_seed >= UINT64_LIMIT:
        raise ManifestError("base_seed must fit in an unsigned 64-bit integer")
    if shard_count <= 0:
        raise ManifestError("shard_count must be positive")
    if shard_id < 0 or shard_id >= shard_count:
        raise ManifestError("shard_id must be in [0, shard_count)")
    if games_per_shard <= 0:
        raise ManifestError("games_per_shard must be positive")
    start = base_seed + shard_id * games_per_shard
    end = start + games_per_shard
    if end > UINT64_LIMIT:
        raise ManifestError("the shard seed interval exceeds uint64")
    return start, end


def _git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    commit = result.stdout.strip()
    return commit or "unknown"


def _relative_path(path: Path, parent: Path) -> str:
    return os.path.relpath(path.resolve(), parent.resolve()).replace(os.sep, "/")


def create_manifest(
    dataset_path: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    *,
    repo_root: Path,
    base_seed: int,
    shard_id: int,
    shard_count: int,
    games_per_shard: int,
    pieces: int,
    sims: int,
    inference_batch: int,
    determinizations: int,
    use_gumbel: bool,
    precision: str,
    device: str,
    model_version: int,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise ManifestError(f"checkpoint not found: {checkpoint_path}")
    if pieces <= 0 or sims <= 0 or inference_batch <= 0 or determinizations <= 0:
        raise ManifestError("pieces, sims, inference_batch and determinizations must be positive")
    if model_version < 0 or model_version >= (1 << 32):
        raise ManifestError("model_version must fit in an unsigned 32-bit integer")

    seed_start, seed_end = compute_seed_interval(
        base_seed, shard_id, shard_count, games_per_shard
    )
    header = read_dataset_header(dataset_path)
    if header.model_version != model_version:
        raise ManifestError(
            f"dataset model_version {header.model_version} does not match requested {model_version}"
        )

    return {
        "format": MANIFEST_FORMAT,
        "manifest_version": MANIFEST_VERSION,
        "repository": {"commit": _git_commit(Path(repo_root).resolve())},
        "checkpoint": {
            "path": _relative_path(checkpoint_path, manifest_path.parent),
            "sha256": sha256_file(checkpoint_path),
        },
        "dataset": {
            "path": _relative_path(dataset_path, manifest_path.parent),
            "sha256": sha256_file(dataset_path),
            "bytes": dataset_path.stat().st_size,
            "version": header.version,
            "dataset_version": header.version,
            "tokenizer_schema_version": header.tokenizer_schema_version,
            "tokenizer_schema_hash": f"{header.tokenizer_schema_hash:016x}",
            "observation_schema_hash": f"{header.observation_schema_hash:016x}",
            "action_schema_version": header.action_schema_version,
            "aux_target_schema_version": header.aux_target_schema_version,
            "max_token_count": header.max_tokens,
            "samples": header.samples,
            "max_tokens": header.max_tokens,
            "max_actions": header.max_actions,
            "token_features": header.token_features,
            "action_features": header.action_features,
            "aux_targets": header.aux_targets,
            "aux_target_names": AUX_TARGET_NAMES[:header.aux_targets],
            "ruleset_hash": f"{header.ruleset_hash:016x}",
            "model_version": header.model_version,
            "randomizer_type": header.randomizer_type,
            "termination_reason": header.termination_reason,
            "self_play_seed": header.self_play_seed,
            "token_kind_order_hash": f"{header.token_kind_order_hash:016x}",
        },
        "schema": {
            "token_kind_order": TOKEN_KIND_ORDER,
            "aux_targets": AUX_TARGET_NAMES[:header.aux_targets],
            "tokenizer_schema_version": header.tokenizer_schema_version,
            "tokenizer_schema_hash": f"{header.tokenizer_schema_hash:016x}",
            "observation_schema_hash": f"{header.observation_schema_hash:016x}",
            "action_schema_version": header.action_schema_version,
            "aux_target_schema_version": header.aux_target_schema_version,
        },
        "run": {
            "base_seed": base_seed,
            "shard_id": shard_id,
            "shard_count": shard_count,
            "games_per_shard": games_per_shard,
            "seed_start": seed_start,
            "seed_end_exclusive": seed_end,
            "pieces": pieces,
            "sims": sims,
            "inference_batch": inference_batch,
            "determinizations": determinizations,
            "use_gumbel": bool(use_gumbel),
            "precision": precision,
            "device": device,
            "model_version": model_version,
        },
    }


def write_manifest(path: Path, manifest: dict[str, Any], overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise ManifestError(f"manifest already exists: {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"manifest is not an object: {path}")
    if value.get("format") != MANIFEST_FORMAT:
        raise ManifestError(f"unsupported manifest format: {path}")
    if value.get("manifest_version") != MANIFEST_VERSION:
        raise ManifestError(f"unsupported manifest version: {path}")
    for key in ("repository", "checkpoint", "dataset", "schema", "run"):
        if not isinstance(value.get(key), dict):
            raise ManifestError(f"manifest is missing object '{key}': {path}")
    return value


def _manifest_dataset_path(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("path"), str):
        raise ManifestError(f"manifest has no dataset path: {manifest_path}")
    path = Path(dataset["path"])
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _ruleset_hash(value: Any) -> int:
    try:
        if isinstance(value, str):
            return int(value, 16)
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError("manifest has an invalid ruleset_hash") from exc


def _int_field(mapping: dict[str, Any], key: str, manifest_path: Path) -> int:
    try:
        return int(mapping[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"manifest field {key} is missing or invalid: {manifest_path}") from exc


def validate_manifests(
    manifest_paths: Iterable[Path],
    *,
    checkpoint_path: Path | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    paths = [Path(path).resolve() for path in manifest_paths]
    if not paths:
        raise ManifestError("at least one manifest is required")

    records: list[tuple[Path, dict[str, Any], DatasetHeader, Path, int, int]] = []
    for manifest_path in paths:
        manifest = _load_manifest(manifest_path)
        dataset_path = _manifest_dataset_path(manifest_path, manifest)
        dataset_info = manifest["dataset"]
        header = read_dataset_header(dataset_path)
        actual_hash = sha256_file(dataset_path)
        if actual_hash != dataset_info.get("sha256"):
            raise ManifestError(f"dataset sha256 mismatch: {dataset_path}")
        if dataset_info.get("bytes") != dataset_path.stat().st_size:
            raise ManifestError(f"dataset byte count mismatch: {dataset_path}")
        for key, actual in (
            ("version", header.version),
            ("dataset_version", header.version),
            ("samples", header.samples),
            ("max_tokens", header.max_tokens),
            ("max_token_count", header.max_tokens),
            ("max_actions", header.max_actions),
            ("token_features", header.token_features),
            ("action_features", header.action_features),
            ("aux_targets", header.aux_targets),
            ("tokenizer_schema_version", header.tokenizer_schema_version),
            ("action_schema_version", header.action_schema_version),
            ("aux_target_schema_version", header.aux_target_schema_version),
            ("model_version", header.model_version),
        ):
            if dataset_info.get(key) != actual:
                raise ManifestError(f"dataset header field {key} mismatch: {manifest_path}")
        if _ruleset_hash(dataset_info.get("ruleset_hash")) != header.ruleset_hash:
            raise ManifestError(f"dataset ruleset_hash mismatch: {manifest_path}")
        for key, actual in (
            ("tokenizer_schema_hash", header.tokenizer_schema_hash),
            ("observation_schema_hash", header.observation_schema_hash),
            ("token_kind_order_hash", header.token_kind_order_hash),
        ):
            if _ruleset_hash(dataset_info.get(key)) != actual:
                raise ManifestError(f"dataset schema field {key} mismatch: {manifest_path}")
        schema_info = manifest.get("schema")
        if schema_info.get("token_kind_order") != TOKEN_KIND_ORDER:
            raise ManifestError(f"token_kind_order mismatch: {manifest_path}")
        expected_aux_names = AUX_TARGET_NAMES[:header.aux_targets]
        if (schema_info.get("aux_targets") != expected_aux_names or
                dataset_info.get("aux_target_names") != expected_aux_names):
            raise ManifestError(f"aux target order mismatch: {manifest_path}")

        run = manifest.get("run")
        if not isinstance(run, dict):
            raise ManifestError(f"manifest has no run description: {manifest_path}")
        base_seed = _int_field(run, "base_seed", manifest_path)
        shard_id = _int_field(run, "shard_id", manifest_path)
        shard_count = _int_field(run, "shard_count", manifest_path)
        games = _int_field(run, "games_per_shard", manifest_path)
        seed_start, seed_end = compute_seed_interval(base_seed, shard_id, shard_count, games)
        if run.get("seed_start") != seed_start or run.get("seed_end_exclusive") != seed_end:
            raise ManifestError(f"seed interval does not match its shard formula: {manifest_path}")
        if run.get("model_version") != header.model_version:
            raise ManifestError(f"run model_version mismatch: {manifest_path}")

        checkpoint = manifest.get("checkpoint")
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("sha256"), str):
            raise ManifestError(f"manifest has no checkpoint hash: {manifest_path}")
        stored_checkpoint = checkpoint.get("path")
        if isinstance(stored_checkpoint, str):
            local_checkpoint = (manifest_path.parent / stored_checkpoint).resolve()
            if local_checkpoint.is_file() and sha256_file(local_checkpoint) != checkpoint["sha256"]:
                raise ManifestError(f"checkpoint sha256 mismatch: {local_checkpoint}")

        records.append((manifest_path, manifest, header, dataset_path, seed_start, seed_end))

    common_paths = [
        ("repository.commit", lambda m: m["repository"]["commit"]),
        ("checkpoint.sha256", lambda m: m["checkpoint"]["sha256"]),
        ("dataset.ruleset_hash", lambda m: m["dataset"]["ruleset_hash"]),
        ("dataset.tokenizer_schema_version", lambda m: m["dataset"]["tokenizer_schema_version"]),
        ("dataset.tokenizer_schema_hash", lambda m: m["dataset"]["tokenizer_schema_hash"]),
        ("dataset.observation_schema_hash", lambda m: m["dataset"]["observation_schema_hash"]),
        ("dataset.action_schema_version", lambda m: m["dataset"]["action_schema_version"]),
        ("dataset.aux_target_schema_version", lambda m: m["dataset"]["aux_target_schema_version"]),
        ("dataset.aux_target_names", lambda m: m["dataset"]["aux_target_names"]),
        ("schema.token_kind_order", lambda m: m["schema"]["token_kind_order"]),
        ("schema.aux_targets", lambda m: m["schema"]["aux_targets"]),
        ("run.base_seed", lambda m: m["run"]["base_seed"]),
        ("run.shard_count", lambda m: m["run"]["shard_count"]),
        ("run.games_per_shard", lambda m: m["run"]["games_per_shard"]),
        ("run.pieces", lambda m: m["run"]["pieces"]),
        ("run.sims", lambda m: m["run"]["sims"]),
        ("run.inference_batch", lambda m: m["run"]["inference_batch"]),
        ("run.determinizations", lambda m: m["run"]["determinizations"]),
        ("run.use_gumbel", lambda m: m["run"]["use_gumbel"]),
        ("run.precision", lambda m: m["run"]["precision"]),
        ("run.model_version", lambda m: m["run"]["model_version"]),
    ]
    first_manifest = records[0][1]
    for label, getter in common_paths:
        expected = getter(first_manifest)
        for manifest_path, manifest, *_ in records[1:]:
            if getter(manifest) != expected:
                raise ManifestError(f"incompatible {label}: {manifest_path}")

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            raise ManifestError(f"checkpoint not found: {checkpoint_path}")
        checkpoint_hash = sha256_file(checkpoint_path)
        if checkpoint_hash != first_manifest["checkpoint"]["sha256"]:
            raise ManifestError(f"checkpoint does not match the shard manifests: {checkpoint_path}")

    intervals = sorted((start, end, path) for path, _, _, _, start, end in records)
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise ManifestError(f"overlapping seed intervals: {previous[2]} and {current[2]}")

    shard_count = int(first_manifest["run"]["shard_count"])
    shard_ids = {int(record[1]["run"]["shard_id"]) for record in records}
    if len(shard_ids) != len(records):
        raise ManifestError("duplicate shard_id in manifests")
    if require_complete and shard_ids != set(range(shard_count)):
        missing = sorted(set(range(shard_count)) - shard_ids)
        raise ManifestError(f"incomplete shard set; missing shard ids: {missing}")

    return {
        "manifests": len(records),
        "datasets": [str(record[3]) for record in records],
        "samples": sum(record[2].samples for record in records),
        "shard_ids": sorted(shard_ids),
        "shard_count": shard_count,
        "seed_start": intervals[0][0],
        "seed_end_exclusive": intervals[-1][1],
        "ruleset_hash": first_manifest["dataset"]["ruleset_hash"],
        "checkpoint_sha256": first_manifest["checkpoint"]["sha256"],
        "complete": shard_ids == set(range(shard_count)),
    }


def _engine_path(root: Path, requested: str) -> Path:
    if requested:
        return Path(requested).expanduser().resolve()
    name = "tetra_cli.exe" if os.name == "nt" else "tetra_cli"
    return (root / "build" / name).resolve()


def ensure_engine(root: Path, requested: str, build_engine: bool) -> Path:
    engine = _engine_path(root, requested)
    if engine.is_file():
        return engine
    if not build_engine:
        raise ManifestError(f"engine not found: {engine}; pass --build-engine or --engine")
    if shutil.which("make") is None:
        raise ManifestError("make is required to build the engine in Colab")
    subprocess.run(["make", "tools"], cwd=str(root), check=True)
    if not engine.is_file():
        raise ManifestError(f"make tools did not produce the engine: {engine}")
    return engine


def _default_manifest_path(output: Path) -> Path:
    return Path(str(output) + ".manifest.json")


def _run_generation(args: argparse.Namespace) -> int:
    root = Path(args.repo_root).expanduser().resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else _default_manifest_path(output)
    )
    if not args.overwrite:
        for path in (output, manifest_path):
            if path.exists():
                raise ManifestError(f"output already exists: {path}; pass --overwrite to replace it")

    engine = ensure_engine(root, args.engine, args.build_engine)
    seed_start, _ = compute_seed_interval(
        args.base_seed, args.shard_id, args.shard_count, args.games
    )

    # Keep torch and the GPU protocol lazy so manifest validation remains
    # available on a CPU-only local machine.
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ManifestError("GPU requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    from gpu_match import load_model
    from gpu_selfplay import generate, report

    model = load_model(str(checkpoint), device)
    results, summary, inference_seconds = generate(
        model,
        device,
        str(engine),
        str(output),
        args.games,
        args.pieces,
        args.sims,
        args.inference_batch,
        seed_start,
        args.model_version,
        args.determinizations,
        args.use_gumbel,
        args.precision,
    )
    report(results, summary, inference_seconds, device, str(checkpoint), str(output))
    if summary.get("games") != args.games or len(results) != args.games:
        raise ManifestError(
            f"engine returned {summary.get('games', 0)} games, expected {args.games}"
        )

    manifest = create_manifest(
        output,
        manifest_path,
        checkpoint,
        repo_root=root,
        base_seed=args.base_seed,
        shard_id=args.shard_id,
        shard_count=args.shard_count,
        games_per_shard=args.games,
        pieces=args.pieces,
        sims=args.sims,
        inference_batch=args.inference_batch,
        determinizations=args.determinizations,
        use_gumbel=args.use_gumbel,
        precision=args.precision,
        device=args.device,
        model_version=args.model_version,
    )
    write_manifest(manifest_path, manifest, overwrite=args.overwrite)
    print(f"manifest     {manifest_path}")
    print(f"seed range   [{seed_start}, {manifest['run']['seed_end_exclusive']})")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate one deterministic GPU shard")
    generate.add_argument("checkpoint")
    generate.add_argument("output")
    generate.add_argument("--manifest", default="")
    generate.add_argument("--repo-root", default="")
    generate.add_argument("--engine", default="")
    generate.add_argument("--build-engine", action="store_true")
    generate.add_argument("--device", default="cuda")
    generate.add_argument("--games", type=int, default=32)
    generate.add_argument("--pieces", type=int, default=300)
    generate.add_argument("--sims", type=int, default=64)
    generate.add_argument("--inference-batch", type=int, default=16)
    generate.add_argument("--base-seed", type=int, default=1)
    generate.add_argument("--shard-id", type=int, required=True)
    generate.add_argument("--shard-count", type=int, required=True)
    generate.add_argument("--model-version", type=int, default=1)
    generate.add_argument("--determinizations", type=int, default=2)
    generate.add_argument("--no-gumbel", dest="use_gumbel", action="store_false")
    generate.set_defaults(use_gumbel=True)
    generate.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    generate.add_argument("--overwrite", action="store_true")

    validate = subparsers.add_parser("validate", help="validate shard files and seed intervals")
    validate.add_argument("manifests", nargs="+")
    validate.add_argument("--checkpoint", default="")
    validate.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            return _run_generation(args)
        summary = validate_manifests(
            [Path(path) for path in args.manifests],
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            require_complete=args.require_complete,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except ManifestError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
