#!/usr/bin/env python3
"""Normalize TETR.IO multiplayer replays into the C++ human-replay protocol.

This module deliberately stops at JSON normalization.  It never defines game
rules, move legality, rotations, or model features; those stay authoritative in
the C++ engine (`include/tetra/human_replay.hpp`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

PIECES = frozenset("IOTSZJL")
SUPPORTED_KEYS = frozenset(
    {"moveLeft", "moveRight", "softDrop", "hardDrop", "rotateCW", "rotateCCW", "rotate180", "hold"}
)


@dataclass(frozen=True)
class NormalizedState:
    rows: tuple[int, ...]
    current: str
    hold: str
    queue: str
    combo: int
    b2b: int


@dataclass(frozen=True)
class NormalizedTurn:
    frame: int
    player: int
    keys: tuple[str, ...]


@dataclass(frozen=True)
class NormalizedGame:
    source_id: str
    round_index: int
    outcome: tuple[float, float]
    initial: tuple[NormalizedState, NormalizedState]
    turns: tuple[NormalizedTurn, ...]

    @property
    def samples(self) -> int:
        return len(self.turns)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _first_mapping(container: dict[str, Any], names: Sequence[str]) -> dict[str, Any]:
    for name in names:
        value = container.get(name)
        if isinstance(value, dict):
            return value
    return {}


def _piece(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("type", "piece", "name", "id"):
            if key in value:
                return _piece(value[key])
        return "-"
    if not isinstance(value, str):
        return "-"
    value = value.strip().upper()
    return value if value in PIECES else "-"


def _queue(value: Any) -> list[str]:
    if isinstance(value, str):
        return [c for c in value.upper() if c in PIECES]
    out: list[str] = []
    for item in _list(value):
        piece = _piece(item)
        if piece != "-":
            out.append(piece)
    return out


def _occupied(cell: Any) -> bool:
    if cell is None or cell is False or cell == 0 or cell == "":
        return False
    if isinstance(cell, str):
        return cell.lower() not in {"empty", "none", "null", "0"}
    if isinstance(cell, dict):
        if "occupied" in cell:
            return bool(cell["occupied"])
        if "type" in cell:
            return _occupied(cell["type"])
    return True


def _row_mask(row: Any, width: int = 10) -> int:
    if isinstance(row, int):
        return row & ((1 << width) - 1)
    if isinstance(row, str):
        text = row.strip()
        if text and set(text) <= {"0", "1"}:
            bits = text[-width:].rjust(width, "0")
            mask = 0
            for x, bit in enumerate(bits):
                if bit == "1":
                    mask |= 1 << x
            return mask
    cells = _list(row)
    mask = 0
    for x, cell in enumerate(cells[:width]):
        if _occupied(cell):
            mask |= 1 << x
    return mask


def _board_rows(value: Any, *, width: int = 10, height: int = 40) -> tuple[int, ...]:
    if isinstance(value, dict):
        for key in ("board", "rows", "matrix", "field"):
            if key in value:
                return _board_rows(value[key], width=width, height=height)
    raw = _list(value)
    if not raw:
        return tuple(0 for _ in range(height))
    # TETR.IO JSON boards are displayed top-to-bottom.  C++ row zero is the
    # floor, so normalize to bottom-to-top and pad the hidden field.
    masks = [_row_mask(row, width) for row in reversed(raw[-height:])]
    masks.extend([0] * (height - len(masks)))
    return tuple(masks[:height])


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _game_from_full(event: dict[str, Any]) -> dict[str, Any]:
    data = _dict(event.get("data"))
    game = _dict(data.get("game"))
    if game:
        return game
    # Older/exported variants may place the snapshot one level deeper.
    return _first_mapping(data, ("full", "state", "snapshot"))


def _extract_state(full_event: dict[str, Any]) -> NormalizedState:
    game = _game_from_full(full_event)
    if not game:
        raise ValueError("full event has no game snapshot")

    rows = _board_rows(game.get("board"))
    falling = _first_mapping(game, ("falling", "active", "piece"))
    current = _piece(falling) if falling else _piece(game.get("current"))

    hold_raw = game.get("hold")
    hold = _piece(hold_raw)

    q: list[str] = []
    for key in ("queue", "bag", "next", "pieces"):
        candidate = _queue(game.get(key))
        if candidate:
            q = candidate
            break
    # Some replay snapshots expose the current piece as the head of bag/queue.
    if q and current != "-" and q[0] == current:
        q = q[1:]

    stats = _dict(game.get("stats"))
    if not stats:
        stats = _dict(_dict(full_event.get("data")).get("stats"))
    combo = _int(stats.get("combo", game.get("combo", -1)), -1)
    b2b = _int(stats.get("btb", stats.get("b2b", game.get("b2b", 0))), 0)
    if current == "-":
        raise ValueError("full event has no active/current tetromino")
    return NormalizedState(rows, current, hold, "".join(q), combo, b2b)


def _event_type(event: dict[str, Any]) -> str:
    return str(event.get("type", event.get("kind", ""))).strip().lower()


def _event_frame(event: dict[str, Any]) -> int:
    return _int(event.get("frame", event.get("tick", 0)), 0)


def _event_key(event: dict[str, Any]) -> str | None:
    data = event.get("data")
    if isinstance(data, str):
        return data if data in SUPPORTED_KEYS else None
    data = _dict(data)
    candidates: list[Any] = [data.get("key"), data.get("name"), data.get("input")]
    for nested_name in ("key_event", "keyEvent", "event"):
        nested = _dict(data.get(nested_name))
        candidates.extend((nested.get("key"), nested.get("name"), nested.get("input")))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate in SUPPORTED_KEYS:
            return candidate
    return None


def _events(replay: Any) -> list[dict[str, Any]]:
    obj = _dict(replay)
    for value in (obj.get("events"), _dict(obj.get("replay")).get("events")):
        if isinstance(value, list):
            return [event for event in value if isinstance(event, dict)]
    if isinstance(replay, list):
        return [event for event in replay if isinstance(event, dict)]
    return []


def _round_replays(round_data: Any) -> list[Any]:
    # Canonical TTRM stores each ReplaySet as a mapping with `replays`, while
    # older collector exports commonly use a two-element list of player
    # records, each containing its own `replay` object.
    if isinstance(round_data, list):
        return round_data
    obj = _dict(round_data)
    for key in ("replays", "replay"):
        value = obj.get(key)
        if isinstance(value, list):
            return value
    # Typed TTRM mirrors commonly store a replay-set object.
    for key in ("data", "set"):
        nested = _dict(obj.get(key))
        value = nested.get("replays")
        if isinstance(value, list):
            return value
    return []


def _rounds(root: Any) -> list[Any]:
    obj = _dict(root)
    candidates: list[Any] = [obj.get("rounds")]
    # Canonical .ttrm: top-level `data` is Vec<ReplaySet>.
    if isinstance(obj.get("data"), list):
        candidates.append(obj.get("data"))
    replay = _dict(obj.get("replay"))
    candidates.extend((replay.get("rounds"), replay.get("sets")))
    data = _dict(obj.get("data"))
    candidates.extend((_dict(data.get("replay")).get("rounds"), data.get("rounds")))
    for value in candidates:
        if isinstance(value, list):
            return value
    # A single replay-set is accepted for fixtures and historical exports.
    if len(_round_replays(obj)) >= 2:
        return [obj]
    return []


def _first_full(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    for event in events:
        if _event_type(event) == "full":
            return event
    raise ValueError("replay stream has no full event")


def _winner(events: Sequence[dict[str, Any]]) -> bool:
    for event in events:
        data = _dict(event.get("data"))
        reasons = (
            data.get("gameoverreason"),
            data.get("game_over_reason"),
            data.get("reason"),
            _dict(data.get("game")).get("gameoverreason"),
        )
        if any(isinstance(reason, str) and reason.lower() == "winner" for reason in reasons):
            return True
        if _event_type(event) == "end" and data.get("winner") is True:
            return True
    return False


def _outcome(streams: Sequence[Sequence[dict[str, Any]]]) -> tuple[float, float]:
    winners = [_winner(stream) for stream in streams[:2]]
    if winners == [True, False]:
        return 1.0, -1.0
    if winners == [False, True]:
        return -1.0, 1.0
    return 0.0, 0.0


def _turns_for_player(events: Sequence[dict[str, Any]], player: int) -> Iterator[NormalizedTurn]:
    pending: list[str] = []
    for event in events:
        if _event_type(event) not in {"keydown", "key_down"}:
            continue
        key = _event_key(event)
        if key is None:
            continue
        pending.append(key)
        if key == "hardDrop":
            yield NormalizedTurn(_event_frame(event), player, tuple(pending))
            pending.clear()


def normalize_round(round_data: Any, source_id: str, round_index: int) -> NormalizedGame:
    replays = _round_replays(round_data)
    if len(replays) != 2:
        raise ValueError(f"round {round_index}: expected exactly two replay streams, got {len(replays)}")
    streams = [_events(replay) for replay in replays]
    if any(not stream for stream in streams):
        raise ValueError(f"round {round_index}: empty replay stream")
    initial = tuple(_extract_state(_first_full(stream)) for stream in streams)
    turns = [*list(_turns_for_player(streams[0], 0)), *list(_turns_for_player(streams[1], 1))]
    turns.sort(key=lambda turn: (turn.frame, turn.player))
    if not turns:
        raise ValueError(f"round {round_index}: no hard-drop turns")
    return NormalizedGame(source_id, round_index, _outcome(streams), initial, tuple(turns))  # type: ignore[arg-type]


def normalize_document(root: Any, source_id: str) -> tuple[list[NormalizedGame], list[str]]:
    games: list[NormalizedGame] = []
    errors: list[str] = []
    rounds = _rounds(root)
    if not rounds:
        return [], ["document contains no multiplayer replay rounds"]
    for index, round_data in enumerate(rounds):
        try:
            games.append(normalize_round(round_data, source_id, index))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    return games, errors


def source_id_for(path: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()[:20]
    return f"{path.stem}:{digest}"


def normalize_file(path: str | Path) -> tuple[list[NormalizedGame], list[str], str]:
    path = Path(path)
    data = path.read_bytes()
    try:
        root = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [], [f"{path}: invalid JSON: {exc}"], source_id_for(path, data)
    source_id = source_id_for(path, data)
    games, errors = normalize_document(root, source_id)
    return games, errors, source_id


def _rows_text(rows: Iterable[int]) -> str:
    return ",".join(str(int(row)) for row in rows)


def _state_line(player: int, state: NormalizedState) -> str:
    queue = state.queue or "-"
    return "\t".join(
        ("INIT", str(player), _rows_text(state.rows), state.current, state.hold, queue, str(state.combo), str(state.b2b))
    )


def render_game(game: NormalizedGame) -> str:
    lines = [
        "\t".join(("GAME", game.source_id, str(game.round_index), str(game.outcome[0]), str(game.outcome[1]))),
        _state_line(0, game.initial[0]),
        _state_line(1, game.initial[1]),
    ]
    for turn in game.turns:
        lines.append("\t".join(("TURN", str(turn.frame), str(turn.player), ",".join(turn.keys))))
    lines.append("END")
    return "\n".join(lines) + "\n"


def render_games(games: Sequence[NormalizedGame]) -> str:
    return "".join(render_game(game) for game in games)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input .ttrm JSON")
    parser.add_argument("output", type=Path, nargs="?", help="normalized line-protocol output")
    parser.add_argument("--strict", action="store_true", help="fail if any round cannot be normalized")
    args = parser.parse_args()

    games, errors, source_id = normalize_file(args.input)
    for error in errors:
        print(f"warning: {error}")
    if not games or (args.strict and errors):
        print(f"normalization failed for {source_id}: games={len(games)} errors={len(errors)}")
        return 1
    text = render_games(games)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8", newline="\n")
    else:
        print(text, end="")
    print(f"normalized {len(games)} games / {sum(game.samples for game in games)} hard-drop turns from {source_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
