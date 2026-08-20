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
import math
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
    keys: tuple[str, ...] = ()
    time10: int = -1
    exact: bool = False
    pre_rows: tuple[int, ...] = ()
    pre_garbage_rows: tuple[int, ...] = ()
    current: str = "-"
    hold: str = "-"
    queue: str = ""
    combo: int = 0
    b2b: int = 0
    used_hold: bool = False
    final_piece: str = "-"
    final_x: int = 0
    final_y: int = 0
    final_rotation: int = 0
    placement_rows: tuple[int, ...] = ()
    placement_garbage_rows: tuple[int, ...] = ()
    post_rows: tuple[int, ...] = ()
    post_garbage_rows: tuple[int, ...] = ()
    combo_after: int = 0
    b2b_after: int = 0


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


def _tetrio_7bag_sequence(seed: int, count: int) -> list[str]:
    """Generate TETR.IO's seeded 7-bag sequence for replay reconstruction.

    TETR.IO replay seeds use MINSTD (16807 mod 2147483647) and Fisher-Yates
    over the canonical ZLOSIJT input order.  This implementation is local and
    deliberately tiny; move legality and board mechanics still remain in C++.
    """
    if count <= 0:
        return []
    modulus = 2147483647
    state = seed % modulus
    if state <= 0:
        state += modulus - 1

    def next_float() -> float:
        nonlocal state
        state = (16807 * state) % modulus
        return (state - 1) / 2147483646

    out: list[str] = []
    while len(out) < count:
        bag = list("ZLOSIJT")
        for index in range(len(bag) - 1, 0, -1):
            chosen = int(next_float() * (index + 1))
            bag[index], bag[chosen] = bag[chosen], bag[index]
        out.extend(bag)
    return out[:count]


def _seed_from_events(events: Sequence[dict[str, Any]]) -> int | None:
    for event in reversed(events):
        data = _dict(event.get("data"))
        candidates = (
            _dict(data.get("options")).get("seed"),
            _dict(_dict(data.get("game")).get("options")).get("seed"),
            data.get("seed"),
        )
        for value in candidates:
            try:
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _extract_state(
    full_event: dict[str, Any],
    *,
    seeded_sequence: Sequence[str] | None = None,
) -> NormalizedState:
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
    pieces_placed = _int(stats.get("piecesplaced", stats.get("pieces_placed", 0)), 0)

    # A fresh TETR.IO multiplayer `full` snapshot contains a placeholder
    # `falling` object, while `bag` is the actual seeded piece stream beginning
    # with the first playable piece.  For league replays, prefer the verified
    # seed stream and require its visible prefix to agree with the snapshot.
    if seeded_sequence is not None and pieces_placed == 0:
        seeded = [piece for piece in seeded_sequence if piece in PIECES]
        if not seeded:
            raise ValueError("seeded replay produced an empty piece sequence")
        if q and seeded[: len(q)] != q:
            raise ValueError(
                "replay seed/bag mismatch: "
                f"snapshot={''.join(q)} generated={''.join(seeded[:len(q)])}"
            )
        current = seeded[0]
        q = seeded[1:]

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
    player_turns = [list(_turns_for_player(streams[0], 0)), list(_turns_for_player(streams[1], 1))]
    seeds = [_seed_from_events(stream) for stream in streams]
    common_seed: int | None = None
    present_seeds = [seed for seed in seeds if seed is not None]
    if present_seeds:
        if len(present_seeds) != 2 or present_seeds[0] != present_seeds[1]:
            raise ValueError(f"round {round_index}: player replay seeds disagree: {seeds}")
        common_seed = present_seeds[0]

    # Holds can consume one extra queue item, so reserve a generous tail beyond
    # the observed hard-drop count.  The generated prefix is cross-checked
    # against each player's snapshot before it is trusted.
    sequence_count = max(len(player_turns[0]), len(player_turns[1])) + 32
    sequence = _tetrio_7bag_sequence(common_seed, sequence_count) if common_seed is not None else None
    initial = tuple(
        _extract_state(_first_full(stream), seeded_sequence=sequence)
        for stream in streams
    )
    turns = [*player_turns[0], *player_turns[1]]
    turns.sort(key=lambda turn: (turn.frame, turn.player))
    if not turns:
        raise ValueError(f"round {round_index}: no hard-drop turns")
    return NormalizedGame(source_id, round_index, _outcome(streams), initial, tuple(turns))  # type: ignore[arg-type]


def normalize_round_exact(round_data: Any, source_id: str, round_index: int) -> NormalizedGame:
    """Normalize only after both player streams pass exact v19 reconstruction."""
    try:
        from .ttrm_exact_replay import reconstruct_round
    except ImportError:
        from ttrm_exact_replay import reconstruct_round

    replays = _round_replays(round_data)
    if len(replays) != 2:
        raise ValueError(f"round {round_index}: expected exactly two replay streams, got {len(replays)}")
    streams = [_events(replay) for replay in replays]
    if any(not stream for stream in streams):
        raise ValueError(f"round {round_index}: empty replay stream")

    run0, run1 = reconstruct_round(round_data, garbage_column_mode="seed")
    runs = (run0, run1)
    turns: list[NormalizedTurn] = []
    for player, run in enumerate(runs):
        for placement in run.placements:
            turns.append(
                NormalizedTurn(
                    frame=placement.frame,
                    player=player,
                    time10=placement.subframe,
                    exact=True,
                    pre_rows=placement.pre_rows,
                    pre_garbage_rows=placement.pre_garbage_rows,
                    current=placement.current_before,
                    hold=placement.hold_before or "-",
                    queue="".join(placement.queue_before),
                    combo=placement.combo_before,
                    b2b=placement.b2b_before,
                    used_hold=placement.used_hold,
                    final_piece=placement.piece,
                    final_x=placement.center_x,
                    final_y=int(math.floor(placement.center_y + 1e-9)),
                    final_rotation=placement.rotation,
                    placement_rows=placement.placement_rows,
                    placement_garbage_rows=placement.placement_garbage_rows,
                    post_rows=placement.post_rows,
                    post_garbage_rows=placement.post_garbage_rows,
                    combo_after=placement.combo_after,
                    b2b_after=placement.b2b_after,
                )
            )
    turns.sort(key=lambda turn: (turn.time10, turn.player))
    if not turns:
        raise ValueError(f"round {round_index}: exact replay produced no hard-drop turns")

    # INIT is retained for the opponent-state cache before either player acts.
    first_by_player = []
    for player, run in enumerate(runs):
        if not run.placements:
            raise ValueError(f"round {round_index}: player {player} has no exact placements")
        first = run.placements[0]
        first_by_player.append(
            NormalizedState(
                rows=first.pre_rows,
                current=first.current_before,
                hold=first.hold_before or "-",
                queue="".join(first.queue_before),
                combo=first.combo_before,
                b2b=first.b2b_before,
            )
        )
    initial = (first_by_player[0], first_by_player[1])
    return NormalizedGame(source_id, round_index, _outcome(streams), initial, tuple(turns))


def normalize_document(
    root: Any,
    source_id: str,
    *,
    require_exact: bool = False,
) -> tuple[list[NormalizedGame], list[str]]:
    games: list[NormalizedGame] = []
    errors: list[str] = []
    rounds = _rounds(root)
    if not rounds:
        return [], ["document contains no multiplayer replay rounds"]
    for index, round_data in enumerate(rounds):
        try:
            games.append(
                normalize_round_exact(round_data, source_id, index)
                if require_exact
                else normalize_round(round_data, source_id, index)
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    return games, errors


def source_id_for(path: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()[:20]
    return f"{path.stem}:{digest}"


def normalize_file(
    path: str | Path,
    *,
    require_exact: bool = False,
) -> tuple[list[NormalizedGame], list[str], str]:
    path = Path(path)
    data = path.read_bytes()
    try:
        root = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [], [f"{path}: invalid JSON: {exc}"], source_id_for(path, data)
    source_id = source_id_for(path, data)
    games, errors = normalize_document(root, source_id, require_exact=require_exact)
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
        if turn.exact:
            lines.append(
                "\t".join(
                    (
                        "XTURN",
                        str(turn.time10),
                        str(turn.frame),
                        str(turn.player),
                        _rows_text(turn.pre_rows),
                        _rows_text(turn.pre_garbage_rows),
                        turn.current,
                        turn.hold,
                        turn.queue or "-",
                        str(turn.combo),
                        str(turn.b2b),
                        "1" if turn.used_hold else "0",
                        turn.final_piece,
                        str(turn.final_x),
                        str(turn.final_y),
                        str(turn.final_rotation),
                        _rows_text(turn.placement_rows),
                        _rows_text(turn.placement_garbage_rows),
                        _rows_text(turn.post_rows),
                        _rows_text(turn.post_garbage_rows),
                        str(turn.combo_after),
                        str(turn.b2b_after),
                    )
                )
            )
        else:
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
    parser.add_argument("--exact", action="store_true", help="require fail-closed exact replay reconstruction")
    args = parser.parse_args()

    games, errors, source_id = normalize_file(args.input, require_exact=args.exact)
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
