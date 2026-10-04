#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from trainer.xplus_bootstrap import verify_exact_gate


class XplusBootstrapTest(unittest.TestCase):
    def write_manifest(
        self,
        root: Path,
        *,
        mode: str | None = "exact-v19",
        normalized: int = 10,
        imported: int = 10,
        skipped: int = 0,
        fraction: float = 1.0,
        errors: tuple[str, ...] = (),
    ) -> None:
        root.mkdir(parents=True, exist_ok=True)
        shard = root / "human_00000_test.tetradat"
        shard.write_bytes(b"dataset")
        digest = hashlib.sha256(shard.read_bytes()).hexdigest()
        payload: dict[str, object] = {
            "format": "tetra-human-replay-manifest-v1",
            "totals": {
                "source_files": 1,
                "normalized_games": 1,
                "normalized_turns": normalized,
                "imported_samples": imported,
                "skipped_during_cpp_validation": skipped,
                "import_fraction": fraction,
            },
            "sources": [
                {"games": 1, "turns": normalized, "errors": list(errors)}
            ],
            "shards": [
                {
                    "path": str(shard),
                    "requested_turns": normalized,
                    "games": 1,
                    "imported": imported,
                    "invalid": skipped,
                    "execution": 0,
                    "unmatched": 0,
                    "dataset_sha256": digest,
                }
            ],
        }
        if mode is not None:
            payload["normalization_mode"] = mode
        (root / "manifest.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    def test_gate_rejects_manifest_without_exact_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root, mode=None)
            with self.assertRaisesRegex(SystemExit, "exact-v19 normalizer"):
                verify_exact_gate(root)

    def test_gate_accepts_complete_exact_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            verify_exact_gate(root)

    def test_gate_rejects_cpp_skips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(
                root,
                imported=9,
                skipped=1,
                fraction=0.9,
            )
            with self.assertRaisesRegex(SystemExit, "every normalized turn"):
                verify_exact_gate(root)

    def test_gate_rejects_source_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root, errors=("round mismatch",))
            with self.assertRaisesRegex(SystemExit, "source/round was rejected"):
                verify_exact_gate(root)

    def test_gate_rejects_replaced_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            (root / "human_00000_test.tetradat").write_bytes(b"replaced")
            with self.assertRaisesRegex(SystemExit, "shard hash mismatch"):
                verify_exact_gate(root)

    def test_gate_rejects_nonfinite_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root, fraction=float("nan"))
            with self.assertRaisesRegex(SystemExit, "non-finite"):
                verify_exact_gate(root)

    def test_gate_rejects_inconsistent_shard_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["shards"][0]["requested_turns"] = 9
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "normalized turn counts"):
                verify_exact_gate(root)

    def test_dry_run_has_single_exact_distillation_path(self) -> None:
        script = Path(__file__).with_name("xplus_bootstrap.py")
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--skip-collect",
                "--dry-run",
                "--steps",
                "0",
                "--teacher-steps",
                "0",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        self.assertIn("--exact", completed.stdout)
        self.assertIn("--strict-source", completed.stdout)
        self.assertIn("--model teacher1m", completed.stdout)
        self.assertIn("--student xs", completed.stdout)


if __name__ == "__main__":
    unittest.main()
