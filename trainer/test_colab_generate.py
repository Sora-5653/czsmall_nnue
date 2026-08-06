# SPDX-License-Identifier: MIT
"""Unit tests for the dependency-free Colab shard bookkeeping."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

try:
    from .colab_generate import (
        DATASET_HEADER,
        DATASET_MAGIC,
        ManifestError,
        compute_seed_interval,
        create_manifest,
        validate_manifests,
        write_manifest,
    )
except ImportError:  # pragma: no cover - supports direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from colab_generate import (  # type: ignore
        DATASET_HEADER,
        DATASET_MAGIC,
        ManifestError,
        compute_seed_interval,
        create_manifest,
        validate_manifests,
        write_manifest,
    )


class ColabShardTests(unittest.TestCase):
    def _dataset(self, path: Path, model_version: int = 4) -> None:
        # One sample, one token/action slot and the project's 24/24/4 widths.
        header = DATASET_HEADER.pack(
            DATASET_MAGIC,
            1,
            1,
            1,
            1,
            24,
            24,
            4,
            0x1234,
            model_version,
        )
        float_count = 24 + 1 + 24 + 1 + 1 + 1 + 4
        path.write_bytes(header + b"\0" * (float_count * 4))

    def _manifest(
        self,
        root: Path,
        shard_id: int,
        checkpoint: Path,
        games: int = 2,
    ) -> Path:
        dataset = root / f"shard-{shard_id}.tetradat"
        manifest_path = root / f"shard-{shard_id}.manifest.json"
        self._dataset(dataset)
        manifest = create_manifest(
            dataset,
            manifest_path,
            checkpoint,
            repo_root=root,
            base_seed=100,
            shard_id=shard_id,
            shard_count=2,
            games_per_shard=games,
            pieces=300,
            sims=64,
            inference_batch=16,
            determinizations=2,
            use_gumbel=True,
            precision="fp16",
            device="cuda",
            model_version=4,
        )
        write_manifest(manifest_path, manifest)
        return manifest_path

    def test_seed_interval_is_disjoint_and_bounded(self) -> None:
        self.assertEqual(compute_seed_interval(100, 0, 4, 32), (100, 132))
        self.assertEqual(compute_seed_interval(100, 3, 4, 32), (196, 228))
        with self.assertRaises(ManifestError):
            compute_seed_interval(0, 4, 4, 1)
        with self.assertRaises(ManifestError):
            compute_seed_interval((1 << 64) - 1, 0, 1, 2)

    def test_manifests_validate_and_require_complete_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "champion.pt"
            checkpoint.write_bytes(b"checkpoint")
            first = self._manifest(root, 0, checkpoint)
            second = self._manifest(root, 1, checkpoint)

            summary = validate_manifests([first, second], checkpoint_path=checkpoint,
                                         require_complete=True)
            self.assertEqual(summary["samples"], 2)
            self.assertEqual(summary["shard_ids"], [0, 1])
            self.assertTrue(summary["complete"])

    def test_overlapping_or_duplicate_seed_shards_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "champion.pt"
            checkpoint.write_bytes(b"checkpoint")
            first = self._manifest(root, 0, checkpoint)
            duplicate = root / "duplicate.manifest.json"
            # Point a second dataset at the same seed interval to test the
            # overlap guard rather than merely testing malformed JSON.
            second_dataset = root / "shard-1.tetradat"
            self._dataset(second_dataset)
            second_manifest = self._manifest(root, 1, checkpoint)
            second_payload = json.loads(second_manifest.read_text(encoding="utf-8"))
            second_payload["run"]["shard_id"] = 0
            second_payload["run"]["seed_start"] = 100
            second_payload["run"]["seed_end_exclusive"] = 102
            duplicate.write_text(json.dumps(second_payload), encoding="utf-8")

            with self.assertRaises(ManifestError):
                validate_manifests([first, duplicate])

    def test_different_checkpoint_hashes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            champion = root / "champion.pt"
            other = root / "other.pt"
            champion.write_bytes(b"checkpoint-a")
            other.write_bytes(b"checkpoint-b")
            first = self._manifest(root, 0, champion)
            second = self._manifest(root, 1, other)

            with self.assertRaises(ManifestError):
                validate_manifests([first, second])


if __name__ == "__main__":
    unittest.main()
