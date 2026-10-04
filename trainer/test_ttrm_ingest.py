#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trainer.ttrm_ingest import normalize_document, normalize_file, render_games


def empty_board() -> list[list[None]]:
    return [[None for _ in range(10)] for _ in range(40)]


def stream(current: str, queue: list[str], *, winner: bool, frame: int) -> dict:
    events = [
        {
            "type": "full",
            "frame": 0,
            "data": {
                "game": {
                    "board": empty_board(),
                    "falling": {"type": current},
                    "hold": {"locked": False, "piece": None},
                    "bag": [current, *queue],
                    "stats": {"combo": -1, "btb": 0},
                }
            },
        },
        {"type": "keydown", "frame": frame - 1, "data": {"key": "moveLeft"}},
        {"type": "keydown", "frame": frame, "data": {"key_event": {"key": "hardDrop"}}},
        {"type": "end", "frame": frame + 20, "data": {"gameoverreason": "winner" if winner else "topout"}},
    ]
    return {"events": events}


def document() -> dict:
    return {
        "replay": {
            "rounds": [
                {
                    "replays": [
                        stream("t", ["o", "s", "z", "j", "l", "i"], winner=True, frame=10),
                        stream("l", ["j", "o", "s", "z", "i", "t"], winner=False, frame=12),
                    ]
                }
            ]
        }
    }


class TtrmIngestTest(unittest.TestCase):
    def test_normalizes_two_player_round_and_outcome(self) -> None:
        games, errors = normalize_document(document(), "fixture")
        self.assertEqual(errors, [])
        self.assertEqual(len(games), 1)
        game = games[0]
        self.assertEqual(game.outcome, (1.0, -1.0))
        self.assertEqual(game.initial[0].current, "T")
        self.assertEqual(game.initial[0].queue[0], "O")
        self.assertEqual([turn.player for turn in game.turns], [0, 1])
        self.assertEqual(game.turns[0].keys, ("moveLeft", "hardDrop"))
        self.assertEqual(game.samples, 2)

    def test_rendered_protocol_has_complete_game_boundary(self) -> None:
        games, errors = normalize_document(document(), "fixture")
        self.assertFalse(errors)
        text = render_games(games)
        self.assertTrue(text.startswith("GAME\tfixture\t0\t1.0\t-1.0\n"))
        self.assertEqual(text.count("\nINIT\t"), 2)
        self.assertEqual(text.count("\nTURN\t"), 2)
        self.assertTrue(text.endswith("END\n"))

    def test_file_source_id_is_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ttrm"
            path.write_text(json.dumps(document()), encoding="utf-8")
            games_a, errors_a, source_a = normalize_file(path)
            games_b, errors_b, source_b = normalize_file(path)
            self.assertFalse(errors_a)
            self.assertFalse(errors_b)
            self.assertEqual(source_a, source_b)
            self.assertEqual(len(games_a), len(games_b))
            self.assertTrue(source_a.startswith("sample:"))

    def test_canonical_ttrm_data_replay_set_is_accepted(self) -> None:
        root = {
            "_id": "fixture",
            "ismulti": True,
            "data": [
                {
                    "board": [],
                    "replays": [
                        stream("t", ["o", "s", "z", "j", "l", "i"], winner=True, frame=10),
                        stream("l", ["j", "o", "s", "z", "i", "t"], winner=False, frame=12),
                    ],
                }
            ],
        }
        games, errors = normalize_document(root, "fixture")
        self.assertEqual(errors, [])
        self.assertEqual(len(games), 1)
        self.assertEqual(games[0].samples, 2)

    def test_collector_style_round_list_is_accepted(self) -> None:
        left = stream("t", ["o", "s", "z", "j", "l", "i"], winner=True, frame=10)
        right = stream("l", ["j", "o", "s", "z", "i", "t"], winner=False, frame=12)
        root = {"replay": {"rounds": [[{"replay": left}, {"replay": right}]]}}
        games, errors = normalize_document(root, "fixture")
        self.assertEqual(errors, [])
        self.assertEqual(len(games), 1)
        self.assertEqual(games[0].samples, 2)

    def test_bad_round_is_reported_without_poisoning_good_round(self) -> None:
        root = document()
        root["replay"]["rounds"].append({"replays": []})
        games, errors = normalize_document(root, "fixture")
        self.assertEqual(len(games), 1)
        self.assertEqual(len(errors), 1)
        self.assertIn("expected exactly two", errors[0])


if __name__ == "__main__":
    unittest.main()
