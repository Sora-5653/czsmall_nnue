#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from trainer.import_human_replays import CachedGame, build_shards


class ImportHumanReplaysTest(unittest.TestCase):
    def test_replaced_cached_shard_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            cache_dir = root / "cache"
            output_dir.mkdir()
            engine = root / "engine"
            engine.write_bytes(b"test engine")
            game = CachedGame("source.ttrm", "source-hash", 0, 2, "GAME fixture\nEND\n")
            protocol_hash = hashlib.sha256(game.text.encode("utf-8")).hexdigest()
            output_path = output_dir / f"human_00000_{protocol_hash[:12]}.tetradat"
            output_path.write_bytes(b"original")
            sidecar = output_path.with_suffix(".tetradat.json")
            sidecar.write_text(
                json.dumps(
                    {
                        "protocol_hash": protocol_hash,
                        "engine_sha256": hashlib.sha256(b"test engine").hexdigest(),
                        "model_version": 3,
                        "ruleset": "league",
                        "requested_turns": 2,
                        "games": 1,
                        "imported": 2,
                        "invalid": 0,
                        "execution": 0,
                        "unmatched": 0,
                        "dataset_sha256": hashlib.sha256(b"original").hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            output_path.write_bytes(b"replaced")

            def rebuild(command: list[str], **_kwargs: object) -> SimpleNamespace:
                Path(command[3]).write_bytes(b"rebuilt")
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        "games=1 turns=2 imported=2 invalid=0 "
                        "execution=0 unmatched=0\n"
                    ),
                )

            with mock.patch("trainer.import_human_replays.subprocess.run", side_effect=rebuild) as run:
                reports = build_shards(
                    [[game]], output_dir, cache_dir, engine, 3, "league", False
                )

            run.assert_called_once()
            self.assertFalse(reports[0].cache_hit)
            self.assertEqual(
                reports[0].dataset_sha256,
                hashlib.sha256(b"rebuilt").hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
