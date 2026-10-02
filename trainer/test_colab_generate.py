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
        AUX_TARGET_SCHEMA_VERSION,
        DATASET_CONTRACT,
        DATASET_HEADER,
        DATASET_MAGIC,
        OBSERVATION_SCHEMA_HASH,
        TOKENIZER_SCHEMA_HASH,
        TOKENIZER_SCHEMA_VERSION,
        ManifestError,
        compute_seed_interval,
        create_manifest,
        read_dataset_header,
        validate_manifests,
        write_manifest,
    )
except ImportError:  # pragma: no cover - supports direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from colab_generate import (  # type: ignore
        AUX_TARGET_SCHEMA_VERSION,
        DATASET_CONTRACT,
        DATASET_HEADER,
        DATASET_MAGIC,
        OBSERVATION_SCHEMA_HASH,
        TOKENIZER_SCHEMA_HASH,
        TOKENIZER_SCHEMA_VERSION,
        ManifestError,
        compute_seed_interval,
        create_manifest,
        read_dataset_header,
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

    def _contract_dataset(self, path: Path, version: int, *, trailing: int = 0) -> None:
        # One sample with the contract extension and aux schema v4 (52 targets),
        # laid out exactly as include/tetra/dataset.hpp serialize_dataset writes it.
        aux = 52
        header = DATASET_HEADER.pack(DATASET_MAGIC, version, 1, 1, 1, 24, 24, aux, 0x1234, 4)
        contract = DATASET_CONTRACT.pack(
            1,
            TOKENIZER_SCHEMA_VERSION,
            TOKENIZER_SCHEMA_HASH,
            OBSERVATION_SCHEMA_HASH,
            1,
            AUX_TARGET_SCHEMA_VERSION,
            0,
            0,
            100,
            TOKENIZER_SCHEMA_HASH,
        )
        floats = 24 + 1 + 24 + 1 + 1 + 1 + aux + aux  # ... aux_target, aux_valid_mask
        payload = b"\0" * (floats * 4)
        payload += struct.pack("<iiQI", 1, 0, 100, 0)
        if version >= 4:
            payload += struct.pack("<i", 7)  # chosen_action
        path.write_bytes(header + contract + payload + b"\0" * trailing)

    def test_v4_header_with_chosen_action_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "v4.tetradat"
            self._contract_dataset(dataset, 4)
            header = read_dataset_header(dataset)
            self.assertEqual(header.version, 4)
            self.assertEqual(header.contract_version, 1)
            self.assertEqual(header.aux_targets, 52)
            self.assertEqual(header.aux_target_schema_version, AUX_TARGET_SCHEMA_VERSION)
            self.assertEqual(header.self_play_seed, 100)

    def test_v3_header_remains_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "v3.tetradat"
            self._contract_dataset(dataset, 3)
            self.assertEqual(read_dataset_header(dataset).version, 3)

    def test_v4_size_must_include_chosen_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # A v4 header over a v3-sized payload lacks the chosen_action block.
            v3_bytes = root / "v3.tetradat"
            self._contract_dataset(v3_bytes, 3)
            truncated = root / "truncated.tetradat"
            raw = bytearray(v3_bytes.read_bytes())
            struct.pack_into("<I", raw, 8, 4)
            truncated.write_bytes(bytes(raw))
            with self.assertRaises(ManifestError):
                read_dataset_header(truncated)
            padded = root / "padded.tetradat"
            self._contract_dataset(padded, 4, trailing=4)
            with self.assertRaises(ManifestError):
                read_dataset_header(padded)

    def test_unknown_dataset_versions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "v5.tetradat"
            self._contract_dataset(dataset, 5)
            with self.assertRaises(ManifestError):
                read_dataset_header(dataset)

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
