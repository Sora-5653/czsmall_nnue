#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from trainer.human_pretrain import manifest_shards


class HumanPretrainTest(unittest.TestCase):
    def test_exact_mode_is_forwarded_to_importer(self) -> None:
        script = Path(__file__).with_name("human_pretrain.py")
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "fixture.ttrm",
                "--exact",
                "--dry-run",
                "--steps",
                "0",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        self.assertIn("import_human_replays.py", completed.stdout)
        self.assertIn("--exact", completed.stdout)

    def test_skip_import_exact_rejects_legacy_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(
                json.dumps({"normalization_mode": "legacy-keydown"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "exact-v19"):
                manifest_shards(root, "exact-v19")

    def test_skip_import_exact_rejects_replaced_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "human_00000_test.tetradat"
            shard.write_bytes(b"original")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "normalization_mode": "exact-v19",
                        "shards": [
                            {
                                "path": str(shard),
                                "dataset_sha256": hashlib.sha256(b"original").hexdigest(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            shard.write_bytes(b"replaced")
            with self.assertRaisesRegex(SystemExit, "hash mismatch"):
                manifest_shards(root, "exact-v19")


if __name__ == "__main__":
    unittest.main()
