#!/usr/bin/env python3
"""Frame/subframe TETR.IO multiplayer replay reconstruction.

This module is intentionally separate from the model-facing C++ simulator.  It
owns only the unstable TETR.IO replay timeline: key-down/key-up handling,
DAS/ARR/DCD/SDF, gravity, hold, SRS+, garbage acknowledgement/activation and
piece locking.  A round is eligible for training only if its reconstructed end
state matches the authoritative ``end.game`` snapshot.

The structure is informed by the Apache-2.0 ``tetrio-replay`` reconstruction
work in zbrachinara/viewtris, but this is an independent implementation for the
current replay schema (observed replay version 19).  In particular, modern
``interaction_confirm`` garbage records no longer expose the historical
``column`` field, so garbage-column generation is kept behind an explicit
strategy and is never silently guessed into accepted training data.

The exactness contract is deliberately fail-closed: unsupported/unknown
mechanics produce a rejected round, not approximate samples.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

WIDTH = 10
HEIGHT = 40
PIECES = frozenset("IJLOSTZ")
PIECE_ORDER = tuple("ZLOSIJT")

# Centre-relative SRS cells, +x right, +y up.  These are the same coordinate
# convention used by the public replay reconstruction work and are equivalent
# to Tetra's authoritative C++ SRS+ shapes after translating the origin.
_BASE_CELLS: dict[str, tuple[tuple[int, int], ...]] = {
    "T": ((-1, 0), (0, 0), (1, 0), (0, 1)),
    "L": ((1, 1), (-1, 0), (0, 0), (1, 0)),
    "J": ((-1, 1), (-1, 0), (0, 0), (1, 0)),
    "O": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "S": ((0, 0), (-1, 0), (0, 1), (1, 1)),
    "Z": ((0, 0), (1, 0), (0, 1), (-1, 1)),
}

_I_CELLS: dict[int, tuple[tuple[int, int], ...]] = {
    0: ((2, 0), (-1, 0), (0, 0), (1, 0)),
    1: ((1, 1), (1, 0), (1, -1), (1, -2)),
    2: ((2, -1), (-1, -1), (0, -1), (1, -1)),
    3: ((0, 1), (0, 0), (0, -1), (0, -2)),
}

# Kicks include the identity first.  +y is up.  Values match the current C++
# SRS+ table used for legal-action validation.
_JLSTZ_90: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    (0, 1): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    (1, 0): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
    (1, 2): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
    (2, 1): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    (2, 3): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
    (3, 2): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
    (3, 0): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
    (0, 3): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
}

_I_90: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    (0, 1): ((0, 0), (1, 0), (-2, 0), (-2, -1), (1, 2)),
    (0, 3): ((0, 0), (-1, 0), (2, 0), (2, -1), (-1, 2)),
    (1, 0): ((0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)),
    (1, 2): ((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)),
    (2, 1): ((0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)),
    (2, 3): ((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)),
    (3, 0): ((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)),
    (3, 2): ((0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)),
}

_KICKS_180: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    (0, 2): ((0, 0), (0, 1), (1, 1), (-1, 1), (1, 0), (-1, 0)),
    (2, 0): ((0, 0), (0, -1), (-1, -1), (1, -1), (-1, 0), (1, 0)),
    (3, 1): ((0, 0), (-1, 0), (-1, 2), (-1, 1), (0, 2), (0, 1)),
    (1, 3): ((0, 0), (1, 0), (1, 2), (1, 1), (0, 2), (0, 1)),
}

SUPPORTED_KEYS = frozenset(
    {"moveLeft", "moveRight", "softDrop", "hardDrop", "rotateCW", "rotateCCW", "rotate180", "hold"}
)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def event_subframe(event: Mapping[str, Any]) -> int:
    """Replay time in tenths of a frame, preserving recorded subframe order."""
    frame = _integer(event.get("frame"), 0)
    sub = _number(_dict(event.get("data")).get("subframe"), 0.0)
    return frame * 10 + int(round(sub * 10.0))


class MinStd:
    """TETR.IO's documented MINSTD random source."""

    MODULUS = 2147483647
    MULTIPLIER = 16807

    def __init__(self, seed: int):
        state = int(seed) % self.MODULUS
        if state <= 0:
            state += self.MODULUS - 1
        self.state = state

    def next_int(self) -> int:
        self.state = (self.MULTIPLIER * self.state) % self.MODULUS
        return self.state

    def next_float(self) -> float:
        return (self.next_int() - 1) / (self.MODULUS - 1)

    def below(self, n: int) -> int:
        if n <= 1:
            return 0
        return int(self.next_float() * n)

    def shuffle(self, values: Sequence[str]) -> list[str]:
        out = list(values)
        for i in range(len(out) - 1, 0, -1):
            j = int(self.next_float() * (i + 1))
            out[i], out[j] = out[j], out[i]
        return out


class PieceStream:
    def __init__(self, seed: int):
        self.rng = MinStd(seed)
        self.queue: deque[str] = deque()

    def _refill(self) -> None:
        self.queue.extend(self.rng.shuffle(PIECE_ORDER))

    def pop(self) -> str:
        if not self.queue:
            self._refill()
        return self.queue.popleft()

    def preview(self, count: int) -> tuple[str, ...]:
        while len(self.queue) < count:
            self._refill()
        return tuple(list(self.queue)[:count])


@dataclass
class ShiftState:
    direction: int
    held: bool = False
    das: float = 0.0
    arr: float = 0.0


@dataclass
class ActivePiece:
    kind: str
    x: int = 4
    # Bottom-up continuous centre coordinate. TETR.IO spawns at client
    # y=B-2.04=17.96 on a 40-row internal board, hence 39-17.96=21.04 here.
    y: float = 21.04
    rotation: int = 0
    last_action: str = "spawn"
    last_kick: int = 0
    rotation_active: bool = False
    spin: str = "none"

    def clone(self) -> "ActivePiece":
        return copy.copy(self)


@dataclass
class GarbagePacket:
    cid: int
    iid: int
    ackiid: int
    amount: int
    source_frame: int
    observed_subframe: int
    confirmed_subframe: int | None = None
    active_subframe: int | None = None
    active: bool = False
    source_x: int | None = None
    source_y: int | None = None
    explicit_column: int | None = None
    hole_size: int = 1


@dataclass
class OutgoingAckPacket:
    iid: int
    amount: int


@dataclass(frozen=True)
class PlacementSnapshot:
    frame: int
    subframe: int
    piece: str
    rotation: int
    center_x: int
    center_y: float
    used_hold: bool
    current_before: str
    hold_before: str | None
    queue_before: tuple[str, ...]
    pre_rows: tuple[int, ...]
    placement_rows: tuple[int, ...]
    post_rows: tuple[int, ...]
    pre_garbage_rows: tuple[int, ...]
    placement_garbage_rows: tuple[int, ...]
    post_garbage_rows: tuple[int, ...]
    hold: str | None
    queue: tuple[str, ...]
    combo_before: int
    b2b_before: int
    combo_after: int
    b2b_after: int
    spin: str
    kick_index: int
    lines_cleared: int
    garbage_lines_cleared: int
    attack_before_blocking: int
    garbage_cancelled: int
    garbage_received: int


@dataclass
class EndStateMismatch:
    field: str
    expected: Any
    actual: Any


@dataclass
class ReplayVerification:
    exact: bool
    mismatches: list[EndStateMismatch] = field(default_factory=list)


@dataclass
class ReplayRun:
    placements: list[PlacementSnapshot]
    verification: ReplayVerification
    warnings: list[str]


class Board:
    """40x10 bottom-up board; cells are '.', 'G', or piece letters."""

    def __init__(self, rows: Sequence[Sequence[str | None]] | None = None):
        self.rows: list[list[str]] = [["."] * WIDTH for _ in range(HEIGHT)]
        if rows:
            source = list(rows)[-HEIGHT:]
            for y, row in enumerate(reversed(source)):
                for x, cell in enumerate(list(row)[:WIDTH]):
                    if cell is None or cell is False or cell == 0 or cell == "":
                        continue
                    text = str(cell).upper()
                    self.rows[y][x] = "G" if text in {"GB", "G", "GARBAGE"} else text[:1]

    def clone(self) -> "Board":
        out = Board()
        out.rows = [list(row) for row in self.rows]
        return out

    def occupied(self, x: int, y: int) -> bool:
        if x < 0 or x >= WIDTH or y < 0:
            return True
        if y >= HEIGHT:
            return False
        return self.rows[y][x] != "."

    def lock(self, piece: ActivePiece) -> None:
        for x, y in piece_cells(piece):
            if not 0 <= x < WIDTH or y < 0:
                raise ValueError(f"piece locks outside board at ({x},{y})")
            if y >= HEIGHT:
                raise ValueError(f"piece locks above internal board at ({x},{y})")
            if self.rows[y][x] != ".":
                raise ValueError(f"piece overlaps occupied cell at ({x},{y})")
            self.rows[y][x] = piece.kind

    def clear_full_rows(self) -> tuple[int, int]:
        kept: list[list[str]] = []
        cleared = 0
        garbage_cleared = 0
        for row in self.rows:
            if all(cell != "." for cell in row):
                cleared += 1
                if any(cell == "G" for cell in row):
                    garbage_cleared += 1
            else:
                kept.append(row)
        while len(kept) < HEIGHT:
            kept.append(["."] * WIDTH)
        self.rows = kept[:HEIGHT]
        return cleared, garbage_cleared

    def add_garbage(self, holes: Sequence[int]) -> None:
        for hole in holes:
            if not 0 <= int(hole) < WIDTH:
                raise ValueError(f"invalid garbage hole {hole}")
            if any(cell != "." for cell in self.rows[-1]):
                raise ValueError("garbage pushes occupied cells past internal ceiling")
            self.rows.pop()
            row = ["G"] * WIDTH
            row[int(hole)] = "."
            self.rows.insert(0, row)

    def occupancy_masks(self) -> tuple[int, ...]:
        out: list[int] = []
        for row in self.rows:
            mask = 0
            for x, cell in enumerate(row):
                if cell != ".":
                    mask |= 1 << x
            out.append(mask)
        return tuple(out)

    def garbage_masks(self) -> tuple[int, ...]:
        out: list[int] = []
        for row in self.rows:
            mask = 0
            for x, cell in enumerate(row):
                if cell == "G":
                    mask |= 1 << x
            out.append(mask)
        return tuple(out)

    def top_down_json(self) -> list[list[str | None]]:
        return [
            [None if cell == "." else ("gb" if cell == "G" else cell.lower()) for cell in row]
            for row in reversed(self.rows)
        ]


def rotate_offset(x: int, y: int, rotation: int) -> tuple[int, int]:
    r = rotation % 4
    if r == 0:
        return x, y
    if r == 1:
        return y, -x
    if r == 2:
        return -x, -y
    return -y, x


def piece_offsets(kind: str, rotation: int) -> tuple[tuple[int, int], ...]:
    kind = kind.upper()
    if kind == "I":
        return _I_CELLS[rotation % 4]
    if kind == "O":
        return _BASE_CELLS["O"]
    return tuple(rotate_offset(x, y, rotation) for x, y in _BASE_CELLS[kind])


def piece_cells(piece: ActivePiece) -> tuple[tuple[int, int], ...]:
    # Client IsOccupied() applies ceil() to its top-down Y coordinate. Under
    # y_bottom = 39 - y_client this is exactly floor(y_bottom + offset).
    base_y = math.floor(piece.y + 1e-9)
    return tuple((piece.x + dx, base_y + dy) for dx, dy in piece_offsets(piece.kind, piece.rotation))


def collides(board: Board, piece: ActivePiece) -> bool:
    return any(board.occupied(x, y) for x, y in piece_cells(piece))


def _internal_fall(
    board: Board,
    piece: ActivePiece,
    distance: float,
    *,
    clear_rotation_on_row_cross: bool = False,
) -> bool:
    """TETR.IO v19 `_InternalFall`, transformed to bottom-up coordinates."""
    if distance <= 0.0:
        return True
    client_y = 39.0 - piece.y
    target_client = round(client_y + distance, 6)
    if math.isclose(target_client % 1.0, 0.0, abs_tol=1e-12):
        target_client += 1e-6
    one_below_client = client_y + 1.0
    if math.isclose(one_below_client % 1.0, 0.0, abs_tol=1e-12):
        one_below_client -= 2e-6

    target = piece.clone()
    target.y = 39.0 - target_client
    one_below = piece.clone()
    one_below.y = 39.0 - one_below_client
    if collides(board, target) or collides(board, one_below):
        return False
    old_cell_y = math.floor(piece.y + 1e-9)
    piece.y = target.y
    piece.last_action = "drop"
    if clear_rotation_on_row_cross and math.floor(piece.y + 1e-9) != old_cell_y:
        piece.rotation_active = False
        piece.spin = "none"
        piece.last_kick = 0
    return True


def grounded(board: Board, piece: ActivePiece) -> bool:
    trial = piece.clone()
    return not _internal_fall(board, trial, 1.0)


def hard_drop(board: Board, piece: ActivePiece) -> int:
    distance = 0
    while _internal_fall(board, piece, 1.0):
        distance += 1
    return distance


def kick_list(kind: str, from_rotation: int, to_rotation: int) -> tuple[tuple[int, int], ...]:
    key = (from_rotation % 4, to_rotation % 4)
    if kind == "O":
        return ((0, 0),)
    if (to_rotation - from_rotation) % 4 == 2:
        return _KICKS_180.get(key, ((0, 0),))
    return (_I_90 if kind == "I" else _JLSTZ_90).get(key, ((0, 0),))


def try_rotate(board: Board, piece: ActivePiece, delta: int) -> bool:
    target = (piece.rotation + delta) % 4

    # ComputeKick first tests the rotated piece at the unchanged continuous
    # position. Only if that fails does it enter the kick table.
    direct = piece.clone()
    direct.rotation = target
    if not collides(board, direct):
        piece.rotation = target
        piece.last_action = "rotate"
        piece.last_kick = 0
        piece.rotation_active = True
        piece.spin = spin_type(board, piece, True)
        return True

    # For an actual kick, v19 snaps client Y to floor(y)+0.1 before adding the
    # kick. Transforming to bottom-up coordinates gives the baseline below.
    client_y = 39.0 - piece.y
    snapped_bottom = 39.0 - (math.floor(client_y) + 0.1)
    for index, (dx, dy) in enumerate(kick_list(piece.kind, piece.rotation, target)):
        if dx == 0 and dy == 0:
            continue
        trial = piece.clone()
        trial.rotation = target
        trial.x += dx
        trial.y = snapped_bottom + dy
        if not collides(board, trial):
            piece.rotation = target
            piece.x = trial.x
            piece.y = trial.y
            piece.last_action = "rotate"
            piece.last_kick = index
            piece.rotation_active = True
            piece.spin = spin_type(board, piece, True)
            return True
    return False


def move_horizontal(board: Board, piece: ActivePiece, amount: int) -> int:
    moved = 0
    direction = 1 if amount > 0 else -1
    for _ in range(abs(amount)):
        trial = piece.clone()
        trial.x += direction
        if collides(board, trial):
            break
        piece.x += direction
        piece.last_action = "move"
        piece.last_kick = 0
        moved += direction
    return moved


def _corner_occupied(board: Board, x: int, y: int) -> bool:
    return board.occupied(x, y)


def immobile(board: Board, piece: ActivePiece) -> bool:
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        trial = piece.clone()
        trial.x += dx
        trial.y += dy
        if not collides(board, trial):
            return False
    return True


def spin_type(board: Board, piece: ActivePiece, all_mini_plus: bool) -> str:
    if not piece.rotation_active:
        return "none"
    if piece.kind != "T":
        return "mini" if all_mini_plus and immobile(board, piece) else "none"
    base_y = math.floor(piece.y + 1e-9)
    corners = {
        "bl": _corner_occupied(board, piece.x - 1, base_y - 1),
        "br": _corner_occupied(board, piece.x + 1, base_y - 1),
        "tl": _corner_occupied(board, piece.x - 1, base_y + 1),
        "tr": _corner_occupied(board, piece.x + 1, base_y + 1),
    }
    r = piece.rotation % 4
    fronts = (("tl", "tr"), ("tr", "br"), ("bl", "br"), ("tl", "bl"))[r]
    occupied = sum(bool(value) for value in corners.values())
    front_count = sum(bool(corners[name]) for name in fronts)
    if occupied >= 3:
        # ComputeKick checks the no-kick rotation before enumerating the kick
        # table. Our table includes identity at index 0, so client kick==3 is
        # represented here as index 4.
        if front_count == 2 or piece.last_kick == 4:
            return "full"
        return "mini"
    if all_mini_plus and immobile(board, piece):
        return "mini"
    return "none"


@dataclass
class AttackState:
    # TETR.IO's stats.combo is 0 off-combo, 1 after the first consecutive clear.
    combo: int = 0
    b2b: int = 0
    pieces_placed: int = 0


def base_attack(spin: str, lines: int) -> float:
    """Current client garbage constants before B2B/combo modifiers."""
    if lines <= 0:
        return 0.0
    if spin == "mini":
        return float({1: 0, 2: 1, 3: 2, 4: 4, 5: 5}.get(lines, 5 + max(0, lines - 5)))
    if spin == "full":
        return float({1: 2, 2: 4, 3: 6, 4: 10, 5: 12}.get(lines, 12 + 2 * max(0, lines - 5)))
    return float({1: 0, 2: 1, 3: 2, 4: 4, 5: 5}.get(lines, 5 + max(0, lines - 5)))


def _split_charge(amount: int) -> list[int]:
    if amount <= 0:
        return []
    third = int(round(amount / 3.0))
    return [value for value in (third, third, amount - 2 * third) if value > 0]


def compute_attack_chunks(
    state: AttackState,
    *,
    spin: str,
    lines: int,
    garbage_cleared: int,
    all_clear: bool,
    options: Mapping[str, Any],
) -> list[int]:
    """Return the ordered FightLines calls made by current League for a lock.

    B2B-charge release, normal clear attack, and all-clear attack are distinct
    calls in the client. Keeping those boundaries matters because each call can
    exhaust a pending packet and therefore advance `rngex`.
    """
    chunks: list[int] = []
    # Current v19 computes a single B2B increment `a` *before* deciding
    # whether this clear breaks the chain.  A difficult line clear contributes
    # one.  All Clear contributes allclear_b2b, but with dupes disabled the AC
    # contribution and the difficult-clear contribution do not stack.
    difficult = lines >= 4 or (lines > 0 and spin != "none")
    b2b_increment = 1 if difficult else 0
    all_clear_b2b = _integer(options.get("allclear_b2b"), 0) if all_clear else 0
    if all_clear_b2b > 0:
        if bool(options.get("allclear_b2b_dupes", False)):
            b2b_increment += all_clear_b2b
        else:
            b2b_increment = max(all_clear_b2b, b2b_increment)
    if all_clear and bool(options.get("allclear_charges", False)):
        b2b_increment = max(
            b2b_increment,
            _integer(options.get("b2bcharge_at"), 4) + 1 - state.b2b,
        )

    if lines > 0:
        state.combo += 1
    else:
        state.combo = 0

    if b2b_increment > 0:
        state.b2b += b2b_increment
    elif lines > 0:
        if bool(options.get("b2bcharging", True)) and state.b2b > _integer(options.get("b2bcharge_at"), 4):
            charge = int(
                math.floor(
                    (state.b2b - _integer(options.get("b2bcharge_at"), 4)
                     + _integer(options.get("b2bcharge_base"), 3))
                    * _number(options.get("garbagemultiplier"), 1.0)
                )
            )
            chunks.extend(_split_charge(charge))
        state.b2b = 0

    if lines > 0:
        value = base_attack(spin, lines)
        # The current clear receives B2B attack whenever the resulting B2B
        # counter is above one and this placement contributed a B2B increment.
        # This includes an All Clear on an otherwise non-difficult clear.
        if state.b2b > 1 and b2b_increment > 0 and not bool(options.get("b2bchaining", False)):
            bonus = 1.0
            if bool(options.get("b2bextras", False)) and (lines == 4 or (spin == "full" and lines >= 2)):
                bonus *= 2.0
            value += bonus
        if state.combo > 1:
            if str(options.get("combotable", "multiplier")) == "multiplier":
                value *= 1.0 + 0.25 * (state.combo - 1)
                if state.combo > 2:
                    value = max(math.log1p((state.combo - 1) * 1.25), value)
        value *= _number(options.get("garbagemultiplier"), 1.0)
        attack = int(math.floor(value + 1e-12))  # roundmode=down in League
        if (
            bool(options.get("garbagespecialbonus", True))
            and garbage_cleared > 0
            and difficult
        ):
            attack += 1
        if attack > 0:
            chunks.append(attack)

    if all_clear and bool(options.get("allclears", True)):
        all_clear_attack = int(
            math.floor(
                _number(options.get("allclear_garbage"), 5.0)
                * _number(options.get("garbagemultiplier"), 1.0)
                + 1e-12
            )
        )
        if all_clear_attack > 0:
            chunks.append(all_clear_attack)

    state.pieces_placed += 1
    return chunks


class GarbageColumnSource:
    """TETR.IO's auxiliary RNG (`rngex`) used for garbage holes.

    The current client seeds `rng` and `rngex` as separate MINSTD instances with
    the same replay seed. Piece-bag generation advances only `rng`; garbage
    hole/messiness decisions advance `rngex`. Diagnostic alternate seed modes
    are retained only to make regressions obvious on older replay schemas.
    """

    def __init__(self, mode: str, seed: int, game_id: int):
        self.mode = mode
        if mode == "seed":
            self.rng = MinStd(seed)
        elif mode.startswith("seed-skip-"):
            self.rng = MinStd(seed)
            try:
                skip = max(0, int(mode.rsplit("-", 1)[1]))
            except ValueError as exc:
                raise ValueError(f"invalid garbage seed skip mode: {mode}") from exc
            for _ in range(skip):
                self.rng.next_int()
        elif mode == "gameid":
            self.rng = MinStd(game_id)
        elif mode == "seed-xor-gameid":
            self.rng = MinStd(seed ^ game_id)
        elif mode == "seed-plus-gameid":
            self.rng = MinStd(seed + game_id)
        else:
            self.rng = None
        self.last: int | None = None

    def next_float(self) -> float:
        if self.rng is None:
            raise ValueError("garbage auxiliary RNG is unavailable")
        return self.rng.next_float()

    def reroll(self, *, no_same: bool = False) -> int:
        if self.rng is None:
            raise ValueError("garbage auxiliary RNG is unavailable")
        for _ in range(100):
            value = self.rng.below(WIDTH)
            if not no_same or value != self.last:
                self.last = value
                return value
        if self.last is None:
            self.last = self.rng.below(WIDTH)
        return self.last


class ReplayMachine:
    def __init__(
        self,
        full_event: Mapping[str, Any],
        end_event: Mapping[str, Any],
        *,
        garbage_column_mode: str,
        recorded_outgoing_by_frame: Mapping[int, Sequence[OutgoingAckPacket]] | None = None,
        opponent_end_frame: int | None = None,
        opponent_min_interaction_delay: int | None = None,
        garbage_travel_override_frames: int | None = None,
    ):
        full_data = _dict(full_event.get("data"))
        end_data = _dict(end_event.get("data"))
        # The initial full snapshot has the replay options in data.options on
        # canonical current files; end.options is a robust fallback.
        self.options = _dict(full_data.get("options")) or _dict(end_data.get("options"))
        game = _dict(full_data.get("game"))
        if not game:
            raise ValueError("full event lacks game snapshot")
        self.board = Board(_list(game.get("board")))
        seed = _integer(self.options.get("seed"), 0)
        if seed == 0:
            raise ValueError("replay has no deterministic seed")
        self.seed = seed
        self.game_id = _integer(self.options.get("gameid"), 0)
        self.pieces = PieceStream(seed)
        snapshot_bag = [str(piece).upper() for piece in _list(game.get("bag")) if str(piece).upper() in PIECES]
        if snapshot_bag:
            generated = [self.pieces.pop() for _ in range(len(snapshot_bag))]
            if generated != snapshot_bag:
                raise ValueError(f"piece seed mismatch: snapshot={snapshot_bag} generated={generated}")
            # Re-create the stream and consume the playable first piece only;
            # the bag snapshot begins with that piece on fresh multiplayer full.
            self.pieces = PieceStream(seed)
        first = self.pieces.pop()
        self.active = ActivePiece(first)
        self.hold: str | None = None
        hold_obj = _dict(game.get("hold"))
        hold_piece = hold_obj.get("piece")
        if isinstance(hold_piece, str) and hold_piece.upper() in PIECES:
            self.hold = hold_piece.upper()
        self.hold_locked = bool(hold_obj.get("locked", False))
        self.used_hold_this_piece = False
        self.last_was_clear = False

        handling = _dict(game.get("handling")) or _dict(self.options.get("handling"))
        self.arr = max(0.0, _number(handling.get("arr"), 0.0))
        self.das = max(0.0, _number(handling.get("das"), 0.0))
        self.dcd = max(0.0, _number(handling.get("dcd"), 0.0))
        self.handling_cancel = bool(handling.get("cancel", False))
        self.handling_safelock = bool(handling.get("safelock", False))
        self.sdf = max(0.0, _number(handling.get("sdf"), 41.0))
        self.handling_may20g = bool(handling.get("may20g", True))
        self.gravity_may20g = bool(self.options.get("gravitymay20g", True))
        self.gravity_base = max(0.0, _number(self.options.get("g"), _number(game.get("g"), 0.02)))
        self.gravity_increase = max(0.0, _number(self.options.get("gincrease"), 0.0))
        self.gravity_margin = max(0.0, _number(self.options.get("gmargin"), 0.0))
        self.gravity = self.gravity_base
        self.lock_delay = float(max(0, _integer(self.options.get("locktime"), 30)))
        self.lock_reset_limit = max(0, _integer(self.options.get("lockresets"), 15))
        replay_garbage_speed = max(0, _integer(self.options.get("garbagespeed"), 20))
        if garbage_travel_override_frames is not None:
            replay_garbage_speed = max(0, int(garbage_travel_override_frames))
        self.garbage_speed = replay_garbage_speed * 10
        self.garbage_cap = max(0, _integer(self.options.get("garbagecap"), 8))
        self.opener_phase = max(0, _integer(self.options.get("openerphase"), 14))
        self.messiness_change = max(0.0, _number(self.options.get("messiness_change"), 1.0))
        self.messiness_inner = max(0.0, _number(self.options.get("messiness_inner"), 0.0))
        self.messiness_nosame = bool(self.options.get("messiness_nosame", False))
        self.passthrough = str(self.options.get("passthrough", "zero")).lower()
        self.all_mini_plus = str(self.options.get("spinbonuses", "all-mini+")).lower() in {"all-mini", "all-mini+"}

        self.left_shift = ShiftState(-1)
        self.right_shift = ShiftState(1)
        self.held_soft = False
        self.last_shift_dir = -1
        self.hit_wall = False
        self.gravity_accumulator = 0.0
        self.lock_elapsed = 0.0
        self.lock_resets = 0
        self.rot_resets = 0
        self.hy_client = 18.0
        self.safelock_remaining = 0
        self.last_time = 0

        self.attack = AttackState()
        self.garbage: deque[GarbagePacket] = deque()
        self.garbage_columns = GarbageColumnSource(garbage_column_mode, seed, self.game_id)
        self.garbage_column_mode = garbage_column_mode
        self.recorded_outgoing_by_frame = {
            int(frame): [OutgoingAckPacket(int(packet.iid), int(packet.amount)) for packet in packets]
            for frame, packets in (recorded_outgoing_by_frame or {}).items()
        }
        self.opponent_end_frame = opponent_end_frame
        self.opponent_min_interaction_delay = opponent_min_interaction_delay
        self.outgoing_ack: list[OutgoingAckPacket] = []
        self.latest_incoming_iid = 0
        self.placements: list[PlacementSnapshot] = []
        self.warnings: list[str] = []
        self.garbage_received_stat = 0
        self.garbage_raised = 0
        self.lines_cleared = 0
        self.garbage_cleared = 0
        self.lines_sent_net = 0
        self.fight_sent_stat = 0
        self.last_confirm_iid = 0
        self.garbage_attack_stat = 0
        self.turn_start_current = self.active.kind
        self.turn_start_hold = self.hold
        self.turn_start_queue = self.pieces.preview(8)

    def _set_turn_start(self) -> None:
        self.turn_start_current = self.active.kind
        self.turn_start_hold = self.hold
        self.turn_start_queue = self.pieces.preview(8)

    def _reset_spawn(self) -> None:
        self.active.x = 4
        self.active.y = 21.04
        self.active.rotation = 0
        self.active.last_action = "spawn"
        self.active.last_kick = 0
        self.active.rotation_active = False
        self.active.spin = "none"
        self.gravity_accumulator = 0.0
        self.lock_elapsed = 0.0
        self.lock_resets = 0
        self.rot_resets = 0
        self.hy_client = 18.0
        self.hold_locked = False
        self.used_hold_this_piece = False
        if (
            collides(self.board, self.active)
            and self.last_was_clear
            and bool(self.options.get("clutch", True))
        ):
            # Current ConsiderBlockout walks the just-spawned piece upward until
            # it reaches the first legal position (only after the previous lock
            # cleared a line). Client coordinates grow downward; our bottom-up
            # board therefore uses +y for the same clutch motion.
            while collides(self.board, self.active) and self.active.y < HEIGHT + 4:
                self.active.y += 1
        # `Next()` applies 20G after blockout/clutch handling.
        if not collides(self.board, self.active) and self._is_20g():
            self._slam_to_floor()

    def _apply_dcd(self) -> None:
        # Current `_InternalDCD`: DCD is not a generic post-input pause. It only
        # rewinds both DAS charges when the piece has previously hit a wall.
        if not self.hit_wall or self.dcd <= 0.0:
            return
        target = self.das - self.dcd
        for shift in (self.left_shift, self.right_shift):
            shift.das = min(shift.das, target)
            shift.arr = self.arr

    def _spawn_next(self) -> None:
        # `Next()` applies DCD to the previous piece's wall state before
        # resetting the falling-piece flags.
        self._apply_dcd()
        self.active = ActivePiece(self.pieces.pop())
        self.hit_wall = False
        self._reset_spawn()
        self._set_turn_start()

    def _hold(self) -> None:
        if self.hold_locked:
            return
        self._apply_dcd()
        old = self.active.kind
        if self.hold is None:
            self.hold = old
            self.active = ActivePiece(self.pieces.pop())
        else:
            swap = self.hold
            self.hold = old
            self.active = ActivePiece(swap)
        self.hit_wall = False
        self._reset_spawn()
        self.hold_locked = True
        self.used_hold_this_piece = True

    def _record_manipulation(self, *, rotation: bool = False) -> None:
        # v19 increments movement resets regardless of whether the piece is
        # currently grounded. A later fall below the historical `hy` resets
        # them to zero again.
        if self.lock_resets < 31:
            self.lock_resets += 1
        if rotation and self.rot_resets < 63:
            self.rot_resets += 1
        if self.lock_resets < self.lock_reset_limit:
            self.lock_elapsed = 0.0

    def _fall_active(
        self,
        distance: float,
        *,
        clear_rotation_on_row_cross: bool = False,
    ) -> bool:
        old_hy = self.hy_client
        if not _internal_fall(
            self.board,
            self.active,
            distance,
            clear_rotation_on_row_cross=clear_rotation_on_row_cross,
        ):
            return False
        client_y = 39.0 - self.active.y
        if client_y > old_hy:
            self.hy_client = float(math.ceil(client_y))
            self.lock_resets = 0
            self.rot_resets = 0
        return True

    def _is_20g(self) -> bool:
        gravity_20g = self.gravity >= 20.0
        if self.held_soft:
            allowed = self.handling_may20g or (gravity_20g and self.gravity_may20g)
            return (self.sdf == 41.0 or self.gravity * self.sdf >= 20.0) and allowed
        return gravity_20g and self.gravity_may20g

    def _slam_to_floor(self) -> None:
        while self._fall_active(1.0):
            pass

    def _internal_shift(self, direction: int) -> bool:
        trial = self.active.clone()
        trial.x += direction
        if collides(self.board, trial):
            self.hit_wall = True
            return False
        self.active.x += direction
        self.active.last_action = "move"
        self.active.last_kick = 0
        self.active.rotation_active = False
        self.active.spin = "none"
        self.hit_wall = False
        self._record_manipulation(rotation=False)
        # v19 `_InternalShift`: at effective 20G, every successful horizontal
        # step immediately slams again before the next ARR=0 repeat. This is
        # essential for floor/cavity traversal under SDF=41.
        if self._is_20g():
            self._slam_to_floor()
        if self.lock_resets < self.lock_reset_limit:
            self.lock_elapsed = 0.0
        return True

    def _process_shift(self, shift: ShiftState, delta_frames: float) -> None:
        if not shift.held or self.last_shift_dir != shift.direction:
            return
        # Version >=15 only credits the fraction of this interval that occurs
        # after DAS reaches its threshold toward ARR.
        arr_delta = max(0.0, delta_frames - max(0.0, self.das - shift.das))
        shift.das = min(shift.das + delta_frames, self.das)
        if shift.das < self.das:
            return
        shift.arr += arr_delta
        if shift.arr < self.arr:
            return
        repeats = WIDTH if self.arr == 0.0 else int(math.floor(shift.arr / self.arr))
        if self.arr != 0.0:
            shift.arr -= self.arr * repeats
        for _ in range(repeats):
            self._internal_shift(shift.direction)

    def _update_gravity_for_frame(self, frame: int) -> None:
        """Apply the v19 per-frame gravity ramp used by current League.

        ``gmargin`` is measured in replay frames and ``gincrease`` is expressed
        per second at 60 FPS. Gravity is constant inside one replay frame, so
        all subframe physics for frame ``f`` use the same value.
        """
        elapsed = max(0.0, float(frame) - self.gravity_margin)
        self.gravity = self.gravity_base + self.gravity_increase * elapsed / 60.0

    def _activate_due_garbage(self) -> None:
        for packet in self.garbage:
            if (
                not packet.active
                and packet.active_subframe is not None
                and self.last_time >= packet.active_subframe
            ):
                packet.active = True

    def _advance_physics(self, delta_frames: float, end_time10: int) -> None:
        """Process one client `_ProcessSubframe`/frame-tail interval.

        TETR.IO does not integrate movement in fixed tenths.  It calls
        ProcessAllShift(delta) and Fall(delta) once for each recorded subframe
        interval, then once more for the remainder of the frame.  This matters
        near high stacks because Fall's legality probe is not additive.
        """
        if delta_frames <= 0.0:
            return
        # Fall() decrements safelock once per invocation, not once per frame of
        # delta.  This matches the client even for subframe intervals.
        if self.safelock_remaining > 0:
            self.safelock_remaining -= 1

        self._process_shift(self.left_shift, delta_frames)
        self._process_shift(self.right_shift, delta_frames)

        fall_distance = self.gravity * delta_frames
        if self.held_soft:
            if self.sdf == 41.0:
                fall_distance = 400.0 * delta_frames
            else:
                fall_distance *= self.sdf
                fall_distance = max(fall_distance, 0.05 * self.sdf)
        if self.rot_resets > self.lock_reset_limit + 15:
            fall_distance += (
                0.5
                * delta_frames
                * (self.rot_resets - (self.lock_reset_limit + 15))
            )

        remaining = fall_distance
        failed_fall = False
        while remaining > 1e-12:
            step = min(1.0, remaining)
            if not self._fall_active(
                step,
                clear_rotation_on_row_cross=True,
            ):
                failed_fall = True
                break
            remaining -= step

        if failed_fall:
            self.lock_elapsed += delta_frames
            forced = self.lock_resets >= self.lock_reset_limit
            if forced or self.lock_elapsed > self.lock_delay:
                if self.handling_safelock:
                    self.safelock_remaining = 7
                self._lock(end_time10, automatic=True)
        elif not grounded(self.board, self.active):
            self.lock_elapsed = 0.0

    def advance_to(self, target_subframe: int) -> None:
        if target_subframe < self.last_time:
            raise ValueError("replay events are not monotonic")

        # A waiting-frame callback that fired at the previous frame tail is
        # visible to an input at subframe 0 of this frame. Gravity is keyed by
        # the replay frame, including zero-length advances to a frame boundary.
        self._activate_due_garbage()
        self._update_gravity_for_frame(self.last_time // 10)
        while self.last_time < target_subframe:
            current_frame = self.last_time // 10
            self._update_gravity_for_frame(current_frame)
            current_tick = self.last_time % 10
            target_frame = target_subframe // 10
            target_tick = target_subframe % 10

            if current_frame == target_frame:
                delta_ticks = target_tick - current_tick
                self._advance_physics(delta_ticks / 10.0, target_subframe)
                self.last_time = target_subframe
                self._activate_due_garbage()
                break

            # Finish the current frame exactly once, matching the main loop's
            # ProcessAllShift()/Fall() call with e = 1 - current subframe.
            remaining_ticks = 10 - current_tick
            frame_end = (current_frame + 1) * 10
            self._advance_physics(remaining_ticks / 10.0, frame_end)
            self.last_time = frame_end
            self._activate_due_garbage()

    def receive_interaction(self, event: Mapping[str, Any]) -> None:
        """`AddPendingGarbage`: apply zero/consistent passthrough, then queue.

        The current client reconciles the sender's `ackiid` against our still
        unacknowledged outgoing packets *before* constructing impendingdamage.
        This exchange cancellation is network provenance, not FightLines, so it
        does not consume the garbage messiness RNG.
        """
        data = _dict(event.get("data"))
        if data.get("type") != "interaction":
            return
        inner = _dict(data.get("data"))
        if inner.get("type") != "garbage":
            return
        iid = _integer(inner.get("iid"), 0)
        cid = _integer(inner.get("cid"), 0)
        ackiid = _integer(inner.get("ackiid"), 0)
        amount = max(0, _integer(inner.get("amt"), 0))
        if iid <= 0 or cid <= 0 or amount <= 0:
            return

        self.latest_incoming_iid = max(self.latest_incoming_iid, iid)
        if self.passthrough in {"zero", "consistent"}:
            remaining_outgoing: list[OutgoingAckPacket] = []
            incoming = amount
            for outgoing in self.outgoing_ack:
                if outgoing.iid <= ackiid:
                    # The remote side has already acknowledged this attack.
                    continue
                cancelled = min(outgoing.amount, incoming)
                outgoing.amount -= cancelled
                incoming -= cancelled
                if outgoing.amount > 0:
                    remaining_outgoing.append(outgoing)
            self.outgoing_ack = remaining_outgoing
            amount = incoming
        if amount <= 0:
            return

        observed = _integer(event.get("frame"), 0) * 10
        source_frame = _integer(inner.get("frame"), _integer(data.get("frame"), 0))
        explicit_column = inner.get("column")
        packet = GarbagePacket(
            cid=cid,
            iid=iid,
            ackiid=ackiid,
            amount=amount,
            source_frame=source_frame,
            observed_subframe=observed,
            source_x=_integer(inner.get("x"), -1),
            source_y=_integer(inner.get("y"), -1),
            explicit_column=(
                _integer(explicit_column, -1)
                if explicit_column is not None
                else None
            ),
            hole_size=max(1, _integer(inner.get("size"), 1)),
        )
        self.garbage.append(packet)

    def confirm_interaction(self, event: Mapping[str, Any]) -> None:
        """`IncomingAttack`: count post-passthrough damage and schedule hit."""
        data = _dict(event.get("data"))
        if data.get("type") != "interaction_confirm":
            return
        inner = _dict(data.get("data"))
        if inner.get("type") != "garbage":
            return
        cid = _integer(inner.get("cid"), 0)
        if cid <= 0:
            return
        packet = next((entry for entry in self.garbage if entry.cid == cid), None)
        if packet is None:
            # Entire packet may already have disappeared through passthrough.
            return
        confirmed = _integer(event.get("frame"), 0) * 10
        packet.confirmed_subframe = confirmed
        # Replay event ordering already preserves whether same-frame inputs ran
        # before this confirmation. `garbagespeed` is therefore the exact frame
        # delta used by WaitFrames; adding another frame here double-counts the
        # boundary and delays tanking by one frame.
        packet.active_subframe = confirmed + self.garbage_speed
        self.garbage_received_stat += packet.amount
        self.last_confirm_iid = max(self.last_confirm_iid, packet.iid)

    def _packet_finished(self) -> None:
        # Current `FightLines` and `TakeAllDamage` share this exact side effect:
        # when a packet reaches zero, test messiness_change and possibly reroll.
        if self.garbage_columns.next_float() < self.messiness_change:
            self.garbage_columns.reroll(no_same=self.messiness_nosame)

    def _cancel_garbage(self, amount: int) -> int:
        cancelled = 0
        budget = max(0, amount)
        # FightLines offsets impendingdamage whether or not it has completed its
        # incoming-attack-hit delay. Cancellation is FIFO.
        while self.garbage and budget > 0:
            packet = self.garbage[0]
            take = min(budget, packet.amount)
            packet.amount -= take
            budget -= take
            cancelled += take
            if packet.amount == 0:
                self.garbage.popleft()
                self._packet_finished()
        return cancelled

    def _hole_for_line(self, packet: GarbagePacket) -> int:
        if packet.explicit_column is not None and 0 <= packet.explicit_column < WIDTH:
            return packet.explicit_column
        if self.garbage_columns.last is None:
            # JS short-circuit: no inner-probability draw when lastcolumn is null.
            return self.garbage_columns.reroll(no_same=self.messiness_nosame)
        # Even at messiness_inner==0 the comparison consumes one nextFloat.
        if self.garbage_columns.next_float() < self.messiness_inner:
            return self.garbage_columns.reroll(no_same=self.messiness_nosame)
        return self.garbage_columns.last

    def _fight_lines(self, amount: int) -> tuple[int, int]:
        """Current League `FightLines`: return (cancelled, emitted)."""
        attack = max(0, int(amount))
        self.garbage_attack_stat += attack
        cancellation_only = 0
        pending_count = sum(packet.amount for packet in self.garbage)
        if (
            self.attack.pieces_placed <= self.opener_phase
            and pending_count >= self.fight_sent_stat
        ):
            cancellation_only += attack

        original_attack = attack
        cancelled = 0
        while self.garbage and (attack > 0 or cancellation_only > 0):
            packet = self.garbage[0]
            packet.amount -= 1
            cancelled += 1
            if packet.amount <= 0:
                self.garbage.popleft()
                self._packet_finished()
            if attack > 0:
                attack -= 1
            else:
                cancellation_only -= 1
        emitted = attack
        self.fight_sent_stat += emitted
        if cancelled > original_attack + max(0, original_attack):
            raise AssertionError("FightLines cancelled beyond attack+opener budget")
        return cancelled, emitted

    def _receive_active_garbage(self) -> int:
        """Current League `TakeAllDamage` for instant garbage entry."""
        budget = self.garbage_cap if self.garbage_cap > 0 else 1 << 30
        holes: list[int] = []
        # The client temporarily removes inactive/non-spawn packets, tanks active
        # packets up to cap, then restores the deferred packets. Rebuild the
        # deque to preserve that behavior without mutating while iterating.
        original = list(self.garbage)
        active = [packet for packet in original if packet.active]
        deferred = [packet for packet in original if not packet.active]
        survivors: list[GarbagePacket] = []
        for packet in active:
            if budget <= 0:
                survivors.append(packet)
                continue
            take = min(budget, packet.amount)
            for _ in range(take):
                holes.append(self._hole_for_line(packet))
            packet.amount -= take
            budget -= take
            if packet.amount == 0:
                self._packet_finished()
            else:
                survivors.append(packet)
        # Active packets precede the temporarily deferred packets, matching the
        # client after TakeAllDamage pushes the deferred list back.
        self.garbage = deque(survivors + deferred)
        if holes:
            self.board.add_garbage(holes)
            self.garbage_raised += len(holes)
        return len(holes)

    def _lock(self, event_time: int, *, automatic: bool) -> None:
        if automatic and not grounded(self.board, self.active):
            return
        pre_rows = self.board.occupancy_masks()
        pre_garbage = self.board.garbage_masks()
        combo_before = self.attack.combo
        b2b_before = self.attack.b2b
        piece = self.active.clone()
        if not automatic:
            hard_drop(self.board, piece)
            self.active = piece
        spin = piece.spin if piece.rotation_active else "none"
        self.board.lock(piece)
        cleared, garbage_cleared = self.board.clear_full_rows()
        self.lines_cleared += cleared
        self.garbage_cleared += garbage_cleared
        self.last_was_clear = cleared > 0
        all_clear = cleared > 0 and not any(any(cell != "." for cell in row) for row in self.board.rows)
        attack_chunks = compute_attack_chunks(
            self.attack,
            spin=spin,
            lines=cleared,
            garbage_cleared=garbage_cleared,
            all_clear=all_clear,
            options=self.options,
        )
        before_blocking = sum(attack_chunks)
        cancelled = 0
        computed_emitted = 0
        for chunk in attack_chunks:
            chunk_cancelled, chunk_emitted = self._fight_lines(chunk)
            cancelled += chunk_cancelled
            computed_emitted += chunk_emitted

        # Frame-tail Fall() runs under the replay frame that was just pulled;
        # our advance cursor is already at the next boundary.  Automatic locks
        # at an exact boundary therefore carry the previous replay/source frame.
        replay_time = event_time - 10 if automatic and event_time % 10 == 0 else event_time
        lock_frame = replay_time // 10
        recorded_packets = self.recorded_outgoing_by_frame.get(lock_frame, [])
        recorded_emitted = sum(packet.amount for packet in recorded_packets)
        # Remote interaction packets are asynchronous provenance. Their
        # presence is strong evidence and preserves iid boundaries; absence is
        # not evidence that local Offence did not run (especially at round end).
        if recorded_packets:
            if computed_emitted != recorded_emitted:
                self.warnings.append(
                    f"FightLines emitted {computed_emitted} but replay emitted {recorded_emitted} "
                    f"at frame {lock_frame} (chunks={attack_chunks})"
                )
            # `interaction` packets are the actual Offence output and preserve
            # iid boundaries before receiver-side passthrough cancellation.
            self.outgoing_ack.extend(
                OutgoingAckPacket(packet.iid, packet.amount) for packet in recorded_packets
            )
            self.lines_sent_net += recorded_emitted
        else:
            self.lines_sent_net += computed_emitted
        placement_rows = self.board.occupancy_masks()
        placement_garbage = self.board.garbage_masks()
        received = 0
        if cleared == 0:
            received = self._receive_active_garbage()
        post_rows = self.board.occupancy_masks()
        post_garbage = self.board.garbage_masks()
        self.placements.append(
            PlacementSnapshot(
                frame=lock_frame,
                subframe=replay_time,
                piece=piece.kind,
                rotation=piece.rotation,
                center_x=piece.x,
                center_y=piece.y,
                used_hold=self.used_hold_this_piece,
                current_before=self.turn_start_current,
                hold_before=self.turn_start_hold,
                queue_before=self.turn_start_queue,
                pre_rows=pre_rows,
                placement_rows=placement_rows,
                post_rows=post_rows,
                pre_garbage_rows=pre_garbage,
                placement_garbage_rows=placement_garbage,
                post_garbage_rows=post_garbage,
                hold=self.hold,
                queue=self.pieces.preview(5),
                combo_before=combo_before,
                b2b_before=b2b_before,
                combo_after=self.attack.combo,
                b2b_after=self.attack.b2b,
                spin=spin,
                kick_index=piece.last_kick,
                lines_cleared=cleared,
                garbage_lines_cleared=garbage_cleared,
                attack_before_blocking=before_blocking,
                garbage_cancelled=cancelled,
                garbage_received=received,
            )
        )
        self._spawn_next()

    def _activate_shift(self, shift: ShiftState, *, hoisted: bool) -> None:
        shift.held = True
        shift.das = self.das - self.dcd if hoisted else 0.0
        # Engine version >=12 primes ARR on key activation. The initial one-cell
        # tap is still performed separately below.
        shift.arr = self.arr
        self.last_shift_dir = shift.direction

    def key_down(self, key: str, at: int, *, hoisted: bool = False) -> None:
        if key == "moveLeft":
            self._activate_shift(self.left_shift, hoisted=hoisted)
            self._internal_shift(-1)
        elif key == "moveRight":
            self._activate_shift(self.right_shift, hoisted=hoisted)
            self._internal_shift(1)
        elif key == "softDrop":
            # The current client only flips inputSoftdrop here. Actual motion is
            # performed by Fall() for the remaining subframe interval.
            self.held_soft = True
        elif key == "rotateCW":
            if try_rotate(self.board, self.active, 1):
                self._record_manipulation(rotation=True)
                self._apply_dcd()
                if self.lock_resets < self.lock_reset_limit:
                    self.lock_elapsed = 0.0
        elif key == "rotateCCW":
            if try_rotate(self.board, self.active, -1):
                self._record_manipulation(rotation=True)
                self._apply_dcd()
                if self.lock_resets < self.lock_reset_limit:
                    self.lock_elapsed = 0.0
        elif key == "rotate180":
            if try_rotate(self.board, self.active, 2):
                self._record_manipulation(rotation=True)
                self._apply_dcd()
                if self.lock_resets < self.lock_reset_limit:
                    self.lock_elapsed = 0.0
        elif key == "hold":
            self._hold()
        elif key == "hardDrop":
            if self.safelock_remaining == 0:
                self._lock(at, automatic=False)

    def key_up(self, key: str) -> None:
        if key == "moveLeft":
            self.left_shift.held = False
            self.left_shift.das = 0.0
            if self.right_shift.held:
                self.last_shift_dir = 1
            if self.handling_cancel:
                self.right_shift.arr = self.arr
                self.right_shift.das = 0.0
        elif key == "moveRight":
            self.right_shift.held = False
            self.right_shift.das = 0.0
            if self.left_shift.held:
                self.last_shift_dir = -1
            if self.handling_cancel:
                self.left_shift.arr = self.arr
                self.left_shift.das = 0.0
        elif key == "softDrop":
            self.held_soft = False

    def verify_end(self, end_event: Mapping[str, Any]) -> ReplayVerification:
        data = _dict(end_event.get("data"))
        game = _dict(data.get("game"))
        stats = _dict(data.get("stats"))
        mismatches: list[EndStateMismatch] = []
        expected_board = Board(_list(game.get("board")))
        if self.board.occupancy_masks() != expected_board.occupancy_masks():
            mismatches.append(EndStateMismatch("board.occupancy", expected_board.occupancy_masks(), self.board.occupancy_masks()))
        if self.board.garbage_masks() != expected_board.garbage_masks():
            mismatches.append(EndStateMismatch("board.garbage", expected_board.garbage_masks(), self.board.garbage_masks()))
        expected_hold = _dict(game.get("hold")).get("piece")
        expected_hold = str(expected_hold).upper() if isinstance(expected_hold, str) else None
        if self.hold != expected_hold:
            mismatches.append(EndStateMismatch("hold", expected_hold, self.hold))
        pieces_placed = _integer(stats.get("piecesplaced"), len(self.placements))
        if len(self.placements) != pieces_placed:
            mismatches.append(EndStateMismatch("stats.piecesplaced", pieces_placed, len(self.placements)))
        expected_lines = _integer(stats.get("lines"), self.lines_cleared)
        if self.lines_cleared != expected_lines:
            mismatches.append(EndStateMismatch("stats.lines", expected_lines, self.lines_cleared))
        garbage_stats = _dict(stats.get("garbage"))
        if garbage_stats:
            expected_received = _integer(garbage_stats.get("received"), self.garbage_received_stat)
            if self.garbage_received_stat != expected_received:
                mismatches.append(
                    EndStateMismatch(
                        "stats.garbage.received", expected_received, self.garbage_received_stat
                    )
                )
            expected_cleared = _integer(garbage_stats.get("cleared"), self.garbage_cleared)
            if self.garbage_cleared != expected_cleared:
                mismatches.append(EndStateMismatch("stats.garbage.cleared", expected_cleared, self.garbage_cleared))
            expected_attack = _integer(garbage_stats.get("attack"), self.garbage_attack_stat)
            if self.garbage_attack_stat != expected_attack:
                mismatches.append(EndStateMismatch("stats.garbage.attack", expected_attack, self.garbage_attack_stat))
            # stats.sent is incremented by Offence even if the opponent has
            # already topped out and therefore never records a matching IGE.
            # The locally reconstructed FightLines total is the authoritative
            # quantity for this stat; recorded opponent packets remain the
            # authoritative iid/ack provenance while both players are alive.
            expected_sent = _integer(garbage_stats.get("sent"), self.fight_sent_stat)
            if self.fight_sent_stat != expected_sent:
                mismatches.append(
                    EndStateMismatch("stats.garbage.sent", expected_sent, self.fight_sent_stat)
                )
        return ReplayVerification(exact=not mismatches, mismatches=mismatches)


def _stream_events(replay: Any) -> list[dict[str, Any]]:
    obj = _dict(replay)
    replay_obj = _dict(obj.get("replay"))
    raw = replay_obj.get("events") if isinstance(replay_obj.get("events"), list) else obj.get("events")
    return [event for event in _list(raw) if isinstance(event, dict)]


def first_event(events: Sequence[Mapping[str, Any]], kind: str) -> Mapping[str, Any]:
    for event in events:
        if str(event.get("type", "")).lower() == kind:
            return event
    raise ValueError(f"replay stream has no {kind} event")


def last_event(events: Sequence[Mapping[str, Any]], kind: str) -> Mapping[str, Any]:
    for event in reversed(events):
        if str(event.get("type", "")).lower() == kind:
            return event
    raise ValueError(f"replay stream has no {kind} event")


def recorded_outgoing_by_source_frame(opponent_replay: Any) -> dict[int, list[OutgoingAckPacket]]:
    """Packets emitted by us, reconstructed from opponent `interaction` events.

    These are the Offence packets before receiver-side passthrough. Keeping the
    original iid/packet boundaries is essential because later `ackiid` values
    refer to these exact packets.
    """
    out: dict[int, list[OutgoingAckPacket]] = {}
    for event in _stream_events(opponent_replay):
        if str(event.get("type", "")).lower() != "ige":
            continue
        data = _dict(event.get("data"))
        if data.get("type") != "interaction":
            continue
        inner = _dict(data.get("data"))
        if inner.get("type") != "garbage":
            continue
        frame = _integer(inner.get("frame"), -1)
        iid = _integer(inner.get("iid"), 0)
        amount = max(0, _integer(inner.get("amt"), 0))
        if frame >= 0 and iid > 0 and amount > 0:
            out.setdefault(frame, []).append(OutgoingAckPacket(iid=iid, amount=amount))
    return out


def minimum_interaction_delay(replay: Any) -> int | None:
    """Smallest observed server/replay delay from source attack to interaction."""
    delays: list[int] = []
    for event in _stream_events(replay):
        if str(event.get("type", "")).lower() != "ige":
            continue
        data = _dict(event.get("data"))
        if data.get("type") != "interaction":
            continue
        inner = _dict(data.get("data"))
        if inner.get("type") != "garbage":
            continue
        source = _integer(inner.get("frame"), -1)
        observed = _integer(event.get("frame"), -1)
        if source >= 0 and observed >= source:
            delays.append(observed - source)
    return min(delays) if delays else None


def reconstruct_stream(
    replay: Any,
    *,
    garbage_column_mode: str = "unknown",
    recorded_outgoing: Mapping[int, Sequence[OutgoingAckPacket]] | None = None,
    opponent_end_frame: int | None = None,
    opponent_min_interaction_delay: int | None = None,
    garbage_travel_override_frames: int | None = None,
) -> ReplayRun:
    events = _stream_events(replay)
    if not events:
        raise ValueError("empty replay event stream")
    full = first_event(events, "full")
    end = last_event(events, "end")
    machine = ReplayMachine(
        full,
        end,
        garbage_column_mode=garbage_column_mode,
        recorded_outgoing_by_frame=recorded_outgoing,
        opponent_end_frame=opponent_end_frame,
        opponent_min_interaction_delay=opponent_min_interaction_delay,
        garbage_travel_override_frames=garbage_travel_override_frames,
    )

    # Replay order is semantically significant. Key events advance the current
    # frame to their recorded subframe via `_ProcessSubframe`; an IGE without a
    # subframe is handled at whatever subframe has already been reached in that
    # frame. Sorting IGE back to subframe 0 changes FightLines/passthrough order.
    last_gameplay_event: Mapping[str, Any] | None = None
    for event in events:
        kind = str(event.get("type", "")).lower()
        if kind in {"start", "full", "end", "target", "targets", "allow_targeting", "sizzle"}:
            continue
        frame_start = _integer(event.get("frame"), 0) * 10
        data_for_time = _dict(event.get("data"))
        if "subframe" in data_for_time:
            at = frame_start + int(round(_number(data_for_time.get("subframe"), 0.0) * 10.0))
            if at < machine.last_time:
                raise ValueError(
                    f"replay input subframe moved backwards at frame {event.get('frame')}: "
                    f"target={at} current={machine.last_time}"
                )
        else:
            at = max(frame_start, machine.last_time)
        machine.advance_to(at)
        last_gameplay_event = event
        if kind == "keydown":
            key = _dict(event.get("data")).get("key")
            if isinstance(key, str) and key in SUPPORTED_KEYS:
                machine.key_down(
                    key,
                    at,
                    hoisted=bool(_dict(event.get("data")).get("hoisted", False)),
                )
        elif kind == "keyup":
            key = _dict(event.get("data")).get("key")
            if isinstance(key, str) and key in SUPPORTED_KEYS:
                machine.key_up(key)
        elif kind == "ige":
            data = _dict(event.get("data"))
            if data.get("type") == "interaction":
                machine.receive_interaction(event)
            elif data.get("type") == "interaction_confirm":
                machine.confirm_interaction(event)
        else:
            # Unknown cosmetic events are common; gameplay-bearing unknowns
            # must be surfaced so exact certification cannot ignore them.
            machine.warnings.append(f"unhandled event type {kind} at {event.get('frame')}")

    # `end.frame` identifies the replay frame whose normal frame-tail physics
    # has completed in the serialized end snapshot. Processing only the start
    # of that frame misses late DAS/ARR, gravity and automatic-lock effects.
    end_frame = _integer(end.get("frame"), 0)
    end_tail = (end_frame + 1) * 10
    if end_tail >= machine.last_time:
        machine.advance_to(end_tail)

    # If Hold itself causes blockout/topout, v19 serializes the failed held
    # piece as both `falling.type` and `hold.piece`. Normal Hold semantics have
    # already swapped the previous active piece into hold, so reconcile this
    # terminal-only snapshot quirk without changing any placement sample.
    end_data = _dict(end.get("data"))
    if end_data.get("gameoverreason") == "topout" and last_gameplay_event is not None:
        last_kind = str(last_gameplay_event.get("type", "")).lower()
        last_data = _dict(last_gameplay_event.get("data"))
        end_game = _dict(end_data.get("game"))
        expected_hold_raw = _dict(end_game.get("hold")).get("piece")
        expected_falling_raw = _dict(end_game.get("falling")).get("type")
        expected_hold = str(expected_hold_raw).upper() if isinstance(expected_hold_raw, str) else None
        expected_falling = (
            str(expected_falling_raw).upper() if isinstance(expected_falling_raw, str) else None
        )
        if (
            last_kind == "keydown"
            and last_data.get("key") == "hold"
            and _integer(last_gameplay_event.get("frame"), -1) == end_frame
            and expected_hold == expected_falling == machine.active.kind
        ):
            machine.hold = machine.active.kind

    verification = machine.verify_end(end)
    if garbage_column_mode == "unknown" and any(packet.amount > 0 for packet in machine.garbage):
        verification.exact = False
        verification.mismatches.append(
            EndStateMismatch("garbage.column_rng", "proven deterministic derivation", "unknown")
        )
    return ReplayRun(machine.placements, verification, machine.warnings)


def reconstruct_round(
    round_data: Any,
    *,
    garbage_column_mode: str = "seed",
) -> tuple[ReplayRun, ReplayRun]:
    """Reconstruct both players and require fail-closed exact certification."""
    streams = player_streams(round_data)
    if len(streams) != 2:
        raise ValueError(f"expected exactly two replay streams, got {len(streams)}")
    runs: list[ReplayRun] = []
    for player_index, stream in enumerate(streams):
        opponent = streams[1 - player_index]
        opponent_end = last_event(_stream_events(opponent), "end")
        run = reconstruct_stream(
            stream,
            garbage_column_mode=garbage_column_mode,
            recorded_outgoing=recorded_outgoing_by_source_frame(opponent),
            opponent_end_frame=_integer(opponent_end.get("frame"), 0),
            opponent_min_interaction_delay=minimum_interaction_delay(opponent),
        )
        if run.warnings:
            raise ValueError(
                f"player {player_index}: exact replay emitted warnings: {run.warnings[:3]}"
            )
        if not run.verification.exact:
            detail = ", ".join(
                f"{m.field}: expected={m.expected!r} actual={m.actual!r}"
                for m in run.verification.mismatches[:4]
            )
            raise ValueError(f"player {player_index}: exact replay mismatch: {detail}")
        runs.append(run)
    return runs[0], runs[1]


def rounds_from_document(root: Any) -> list[Any]:
    obj = _dict(root)
    replay = _dict(obj.get("replay"))
    for candidate in (replay.get("rounds"), obj.get("rounds"), obj.get("data")):
        if isinstance(candidate, list):
            return candidate
    return []


def player_streams(round_data: Any) -> list[Any]:
    if isinstance(round_data, list):
        return round_data
    obj = _dict(round_data)
    for key in ("replays", "replay"):
        if isinstance(obj.get(key), list):
            return obj[key]
    return []


def diagnose_file(path: Path, modes: Sequence[str]) -> dict[str, Any]:
    root = json.loads(path.read_text(encoding="utf-8"))
    report: dict[str, Any] = {"path": str(path), "modes": {}}
    rounds = rounds_from_document(root)
    for mode in modes:
        mode_rows: list[dict[str, Any]] = []
        for round_index, round_data in enumerate(rounds):
            streams = player_streams(round_data)
            for player_index, stream in enumerate(streams[:2]):
                try:
                    opponent = streams[1 - player_index]
                    opponent_events = _stream_events(opponent)
                    opponent_end = last_event(opponent_events, "end")
                    run = reconstruct_stream(
                        stream,
                        garbage_column_mode=mode,
                        recorded_outgoing=recorded_outgoing_by_source_frame(opponent),
                        opponent_end_frame=_integer(opponent_end.get("frame"), 0),
                        opponent_min_interaction_delay=minimum_interaction_delay(opponent),
                    )
                    mode_rows.append(
                        {
                            "round": round_index,
                            "player": player_index,
                            "exact": run.verification.exact,
                            "placements": len(run.placements),
                            "mismatches": [
                                {"field": item.field, "expected": item.expected, "actual": item.actual}
                                for item in run.verification.mismatches
                            ],
                            "warnings": run.warnings,
                        }
                    )
                except Exception as exc:
                    mode_rows.append(
                        {
                            "round": round_index,
                            "player": player_index,
                            "exact": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        report["modes"][mode] = mode_rows
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("replay", type=Path)
    ap.add_argument(
        "--garbage-column-mode",
        nargs="+",
        default=("seed", "gameid", "seed-xor-gameid", "seed-plus-gameid"),
        help="diagnostic garbage RNG modes; seed-skip-N is also accepted",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    report = diagnose_file(args.replay, args.garbage_column_mode)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for mode, rows in report["modes"].items():
            exact = sum(bool(row.get("exact")) for row in rows)
            print(f"mode={mode}: exact {exact}/{len(rows)} streams")
            for row in rows:
                if row.get("exact"):
                    continue
                fields = [item.get("field") for item in row.get("mismatches", [])]
                print(
                    f"  round {row.get('round')} player {row.get('player')}: "
                    f"placements={row.get('placements', '?')} error={row.get('error', '')} "
                    f"mismatch={','.join(str(field) for field in fields[:8])}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
