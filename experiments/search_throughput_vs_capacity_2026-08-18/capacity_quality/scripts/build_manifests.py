#!/usr/bin/env python3
"""Create non-destructive Phase 0 checkpoint and seed manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch


ROOT = Path(__file__).resolve().parents[4]
CAPACITY = ROOT / "experiments" / "search_throughput_vs_capacity_2026-08-18" / "capacity_quality"
CHECKPOINTS = {
    "A": ROOT / "models" / "gen14_rank100_100_20260814.best.pt",
    "S400": ROOT / "models" / "size_search_ablation_20260816" / "seed42" / "transformer_s.final.pt",
    "M400": ROOT / "models" / "size_search_ablation_20260816" / "seed42" / "transformer_m.final.pt",
    "XS400": ROOT / "models" / "size_search_ablation_20260816" / "seed42" / "transformer_xs.final.pt",
}
TRAIN_RESULTS = ROOT / "models" / "size_search_ablation_20260816" / "seed42" / "train_results.json"
SEED_MULTIPLIER = 0x9E3779B97F4A7C15
UINT64_MASK = (1 << 64) - 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def parameter_count(state_dict: dict[str, object]) -> int:
    return sum(int(value.numel()) for value in state_dict.values() if hasattr(value, "numel"))


def jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def checkpoint_manifest(label: str, path: Path, train_results: dict[str, object]) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = jsonable(checkpoint.get("config", {})) if isinstance(checkpoint, dict) else {}
    state = checkpoint.get("state_dict", {}) if isinstance(checkpoint, dict) else {}
    entry: dict[str, object] = {
        "label": label,
        "path": path.relative_to(ROOT).as_posix(),
        "absolute_path": str(path.resolve()),
        "sha256": sha256(path),
        "file_size_bytes": path.stat().st_size,
        "architecture": checkpoint.get("architecture", "transformer") if isinstance(checkpoint, dict) else "unknown",
        "parameters": parameter_count(state),
        "config": config,
        "checkpoint_format_version": checkpoint.get("format_version", "unknown") if isinstance(checkpoint, dict) else "unknown",
        "checkpoint_step": checkpoint.get("step", "unknown") if isinstance(checkpoint, dict) else "unknown",
        "seed": checkpoint.get("seed", "unknown") if isinstance(checkpoint, dict) else "unknown",
        "training_commit": "unknown",
        "training_dataset_manifest_hash": "unknown",
        "training_dataset_provenance": "unknown",
        "training_metadata": {},
        "validation_metrics": {},
        "metadata_source": path.relative_to(ROOT).as_posix(),
        "evaluated_at_commit": git("rev-parse", "HEAD"),
    }
    if label in {"S400", "M400", "XS400"}:
        size = label[:-3].lower()
        size_info = train_results.get("sizes", {}).get(size, {})
        entry["training_metadata"] = {
            "dataset": jsonable(train_results.get("dataset", {})),
            "training": jsonable(train_results.get("training", {})),
            "size_ablation": size,
            "metadata_file": TRAIN_RESULTS.relative_to(ROOT).as_posix(),
            "metadata_file_sha256": sha256(TRAIN_RESULTS),
        }
        entry["validation_metrics"] = jsonable(
            checkpoint.get("metrics", size_info.get("final", {}))
            if isinstance(checkpoint, dict)
            else size_info.get("final", {})
        )
        entry["training_dataset_provenance"] = "train_results.json dataset.paths; no separate dataset manifest hash was supplied"
        entry["training_commit"] = "unknown"
    else:
        entry["validation_metrics"] = "unknown (not stored in checkpoint)"
        entry["training_metadata"] = {
            "checkpoint_step": entry["checkpoint_step"],
            "loss_weights": jsonable(checkpoint.get("loss_weights", "unknown")) if isinstance(checkpoint, dict) else "unknown",
            "source": "checkpoint only; dataset and convergence metadata not embedded",
        }
    return entry


def main() -> int:
    train_results = json.loads(TRAIN_RESULTS.read_text(encoding="utf-8"))
    manifests = {
        "manifest_version": 1,
        "created_at_unix": time.time(),
        "evaluated_at_commit": git("rev-parse", "HEAD"),
        "worktree_status": git("status", "--short", "--branch"),
        "checkpoints": {
            label: checkpoint_manifest(label, path, train_results)
            for label, path in CHECKPOINTS.items()
        },
        "frozen_paths": [
            path.relative_to(ROOT).as_posix() for path in CHECKPOINTS.values()
        ],
    }
    manifest_path = CAPACITY / "manifests" / "checkpoints.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifests, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    pair_seeds = [((42 + i * SEED_MULTIPLIER) & UINT64_MASK) for i in range(50)]
    seed_manifest = {
        "manifest_version": 1,
        "primary_seed": 42,
        "pairs": 50,
        "games_per_pair": 4,
        "expected_games": 200,
        "pair_seed_formula": "uint64(base_seed + pair_index * 0x9E3779B97F4A7C15)",
        "pair_seed_multiplier_hex": "0x9E3779B97F4A7C15",
        "pair_seed_multiplier_decimal": SEED_MULTIPLIER,
        "pair_seed_modulus": 2**64,
        "pair_indices": list(range(50)),
        "pair_seeds": pair_seeds,
        "games_in_each_pair_in_order": [
            {"game_offset": 0, "mirror": False, "roles_swapped": False},
            {"game_offset": 1, "mirror": False, "roles_swapped": True},
            {"game_offset": 2, "mirror": True, "roles_swapped": False},
            {"game_offset": 3, "mirror": True, "roles_swapped": True},
        ],
        "source": {
            "header": "include/tetra/arena.hpp",
            "function": "Arena::evaluate",
            "engine_command": "build/tetra_cli.exe gpu-arena-protocol 50 100000 300 16 1 1 42 ...",
            "note": "The C++ Arena expands one pair into the four factorial games above. No 2026081800-style seed list is used for primary Phase 0.",
        },
    }
    seed_path = CAPACITY / "manifests" / "seed_design.json"
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(json.dumps(seed_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoints": str(manifest_path), "seed_design": str(seed_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
