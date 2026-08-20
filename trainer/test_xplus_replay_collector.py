#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trainer.collect_xplus_replays import (
    PlayerRef,
    ReplayRef,
    cursor_from_entry,
    load_completed_player_ids,
    load_downloaded_ids,
    load_replay_refs,
    output_path_for,
    parse_rank_targets,
    player_from_entry,
    replay_ref_from_record,
    validate_replay_bytes,
)


class XplusReplayCollectorTest(unittest.TestCase):
    def test_rank_filter_is_case_insensitive_and_deduplicated(self) -> None:
        self.assertEqual(parse_rank_targets(["X+", "x+", " X "]), ("x+", "x"))

    def test_player_and_cursor_extract_from_leaderboard_entry(self) -> None:
        entry = {
            "_id": "abc123",
            "username": "PLAYER",
            "country": "JP",
            "league": {"rank": "x+", "tr": 25000.5},
            "p": {"pri": 1, "sec": 2, "ter": 3},
        }
        player = player_from_entry(entry, {"x+"})
        self.assertEqual(player, PlayerRef("abc123", "PLAYER", "x+", 25000.5, "JP"))
        self.assertEqual(cursor_from_entry(entry), "1:2:3")
        self.assertIsNone(player_from_entry(entry, {"x"}))

    def test_record_ref_ignores_stubs_and_keeps_opponents(self) -> None:
        player = PlayerRef("abc", "P", "x+", 1.0, None)
        self.assertIsNone(replay_ref_from_record({"stub": True, "replayid": "bad"}, player))
        ref = replay_ref_from_record(
            {
                "replayid": "r1",
                "_id": "record1",
                "ts": "2026-08-18T00:00:00Z",
                "gamemode": "league",
                "otherusers": [{"id": "opp"}],
            },
            player,
        )
        self.assertEqual(ref, ReplayRef("r1", "abc", "x+", "2026-08-18T00:00:00Z", "record1", "league", ("opp",)))

    def test_output_path_cannot_escape_corpus_root(self) -> None:
        root = Path("corpus")
        ref = ReplayRef("../r", "p\\evil", "x+/bad", None, None, None, ())
        self.assertEqual(output_path_for(root, ref), root / "x+_bad" / "p_evil" / ".._r.ttrm")

    def test_ledgers_deduplicate_replay_ids_and_download_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / "replay_index.jsonl"
            index.write_text(
                "\n".join(
                    [
                        json.dumps({"replay_id": "a", "player_id": "p1", "rank": "x+", "opponents": []}),
                        json.dumps({"replay_id": "a", "player_id": "p2", "rank": "x+", "opponents": []}),
                        json.dumps({"replay_id": "b", "player_id": "p1", "rank": "x+", "opponents": []}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual([ref.replay_id for ref in load_replay_refs(index)], ["a", "b"])

            downloads = root / "downloads.jsonl"
            downloads.write_text(
                "\n".join(
                    [
                        json.dumps({"replay_id": "a", "status": "error"}),
                        json.dumps({"replay_id": "a", "status": "ok"}),
                        json.dumps({"replay_id": "b", "status": "exists"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_downloaded_ids(downloads), {"a", "b"})

            records_status = root / "records_status.jsonl"
            records_status.write_text(
                "\n".join(
                    [
                        json.dumps({"player_id": "p1", "status": "ok", "records_seen": 0}),
                        json.dumps({"player_id": "p2", "status": "error"}),
                        json.dumps({"player_id": "p3", "status": "ok", "records_seen": 5}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_completed_player_ids(records_status), {"p1", "p3"})

    def test_replay_validation_accepts_supported_root_shapes(self) -> None:
        validate_replay_bytes(json.dumps({"data": []}).encode())
        validate_replay_bytes(json.dumps({"replay": {"rounds": []}}).encode())
        with self.assertRaises(ValueError):
            validate_replay_bytes(json.dumps({"hello": "world"}).encode())


if __name__ == "__main__":
    unittest.main()
