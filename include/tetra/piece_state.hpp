// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- active piece state, collision, rotation, spins.
#pragma once

#include "tetra/bitboard.hpp"
#include "tetra/pieces.hpp"
#include "tetra/ruleset.hpp"
#include "tetra/types.hpp"

#include <cstdint>

namespace tetra {

// How the piece arrived at its current position. Needed for spin detection:
// a spin only counts if the LAST successful action was a rotation.
enum class LastAction : std::uint8_t { Spawn = 0, Move = 1, Rotate = 2, Drop = 3 };

struct ActivePiece {
    Piece type = Piece::None;
    int x = 0;   // bounding-box origin, column of its left edge
    int y = 0;   // bounding-box origin, row of its bottom edge (row 0 = floor)
    Rot rot = Rot::N;
    LastAction last_action = LastAction::Spawn;
    int last_kick = 0;  // index into the kick list that succeeded (0 = no kick)

    bool valid() const { return type != Piece::None; }

    bool operator==(const ActivePiece& o) const {
        return type == o.type && x == o.x && y == o.y && rot == o.rot;
    }
};

// Absolute cell coordinates occupied by a piece.
inline void piece_cells(const ActivePiece& p, Offset out[4]) {
    const PieceShape& s = shape_of(p.type, p.rot);
    for (int i = 0; i < 4; ++i) {
        out[i].x = p.x + s.cells[static_cast<size_t>(i)].x;
        out[i].y = p.y + s.cells[static_cast<size_t>(i)].y;
    }
}

// Collision test using the precomputed row bitmasks. This is the single
// hottest routine in the engine (the movegen BFS and the search both hammer
// it), so it works on whole rows instead of individual cells.
inline bool collides(const Board& b, const ActivePiece& p) {
    const PieceShape& s = shape_of(p.type, p.rot);

    // Horizontal bounds: anything outside the walls is solid. Checking this
    // first also guarantees that p.x + min_dx >= 0, so the row masks below can
    // be shifted left by p.x without ever shifting by a negative amount.
    if (p.x + s.min_dx < 0) return true;
    if (p.x + s.max_dx >= b.width()) return true;
    // Below the floor is solid.
    if (p.y + s.min_dy < 0) return true;

    for (int dy = s.min_dy; dy <= s.max_dy; ++dy) {
        const std::uint32_t m = s.rows[static_cast<size_t>(dy)];
        if (!m) continue;
        const int y = p.y + dy;
        if (y >= b.height()) continue;  // above the ceiling is empty
        // p.x may still be negative (the bounding box can hang off the left
        // edge as long as no filled cell does), so shift the mask, not by a
        // negative count: min_dx >= -p.x is guaranteed above.
        const std::uint32_t placed =
            (p.x >= 0) ? (m << p.x) : (m >> static_cast<unsigned>(-p.x));
        if (b.row(y) & placed) return true;
    }
    return false;
}

inline bool collides_at(const Board& b, Piece type, Rot rot, int x, int y) {
    ActivePiece p;
    p.type = type;
    p.rot = rot;
    p.x = x;
    p.y = y;
    return collides(b, p);
}

// Is the piece resting on something (i.e. would moving down collide)?
inline bool grounded(const Board& b, const ActivePiece& p) {
    ActivePiece q = p;
    q.y -= 1;
    return collides(b, q);
}

// Drop the piece as far as it will go; returns the number of cells fallen.
inline int hard_drop_distance(const Board& b, const ActivePiece& p) {
    int d = 0;
    ActivePiece q = p;
    while (true) {
        q.y -= 1;
        if (collides(b, q)) break;
        ++d;
        if (d > Board::MAX_HEIGHT * 2) break;  // safety
    }
    return d;
}

// Spawn position for a piece (guideline: horizontally centred, sitting just
// above the visible field so that it is fully inside the internal field).
inline ActivePiece spawn_piece(Piece type, const RulesetConfig& cfg) {
    ActivePiece p;
    p.type = type;
    p.rot = Rot::N;
    p.last_action = LastAction::Spawn;
    p.last_kick = 0;
    const PieceShape& s = shape_of(type, Rot::N);
    // Guideline: pieces spawn in the two columns left of centre for 3-wide
    // boxes, and columns 3..6 for the I piece on a 10-wide field.
    p.x = (cfg.geometry.width - s.box) / 2;
    // Bottom of the piece sits on the first row above the visible playfield.
    p.y = cfg.geometry.visible_height;
    // Pull the bounding box down so the lowest *filled* row of the piece rests
    // on the spawn row (empty rows in the box must not add height).
    int lowest = s.box;
    for (int i = 0; i < 4; ++i) lowest = std::min(lowest, s.cells[static_cast<size_t>(i)].y);
    p.y -= lowest;
    return p;
}

// --- rotation --------------------------------------------------------------
struct RotationResult {
    bool success = false;
    int kick_index = 0;
};

// Attempt a rotation with the configured kick table. On success the piece is
// updated in place and `last_action` becomes Rotate (which is what makes spin
// detection possible).
inline RotationResult try_rotate(const Board& b, ActivePiece& p, Rot to, const RulesetConfig& cfg) {
    RotationResult res;
    if (p.rot == to) return res;
    if (!cfg.movement.allow_180 && to == rot_180(p.rot)) return res;

    const KickTables& tables = kick_tables_for(cfg.movement.kick_table);
    const KickList& list = kicks_for(tables, p.type, p.rot, to);

    for (size_t i = 0; i < list.size(); ++i) {
        ActivePiece q = p;
        q.rot = to;
        q.x = p.x + list[i].x;
        q.y = p.y + list[i].y;
        if (!collides(b, q)) {
            q.last_action = LastAction::Rotate;
            q.last_kick = static_cast<int>(i);
            p = q;
            res.success = true;
            res.kick_index = static_cast<int>(i);
            return res;
        }
    }
    return res;
}

// --- spin detection --------------------------------------------------------
// The T piece's four "corners" relative to its 3x3 bounding box, split into the
// two that sit next to the stem (front) and the two behind it (back).
inline void t_corners(const ActivePiece& p, Offset front[2], Offset back[2]) {
    // Corner cells of the 3x3 box in absolute coordinates.
    const Offset bl{p.x + 0, p.y + 0};
    const Offset br{p.x + 2, p.y + 0};
    const Offset tl{p.x + 0, p.y + 2};
    const Offset tr{p.x + 2, p.y + 2};
    switch (p.rot) {
        case Rot::N:  front[0] = tl; front[1] = tr; back[0] = bl; back[1] = br; break;
        case Rot::R:  front[0] = tr; front[1] = br; back[0] = tl; back[1] = bl; break;
        case Rot::S2: front[0] = bl; front[1] = br; back[0] = tl; back[1] = tr; break;
        case Rot::L:  front[0] = tl; front[1] = bl; back[0] = tr; back[1] = br; break;
    }
}

// A piece is "immobile" if it cannot move left, right, UP or down after the
// rotation. The up-check ("overhang") is the decisive one in practice: without
// it, any piece resting in a flat notch would score a spin. TETR.IO's own
// documentation calls the up-check the key to performing All-Mini spins.
inline bool is_immobile(const Board& b, const ActivePiece& p) {
    ActivePiece q = p;
    q.y -= 1;
    if (!collides(b, q)) return false;  // can drop
    q = p; q.y += 1;
    if (!collides(b, q)) return false;  // can rise: no overhang above it
    q = p; q.x -= 1;
    if (!collides(b, q)) return false;  // can slide left
    q = p; q.x += 1;
    if (!collides(b, q)) return false;  // can slide right
    return true;
}

// Classify the spin state of a piece at the moment it locks.
// `kick_index` is the kick that was used by the final rotation.
inline SpinType detect_spin(const Board& b, const ActivePiece& p, const RulesetConfig& cfg) {
    const SpinDetection mode = cfg.clear_rules.spin_detection;
    if (mode == SpinDetection::None) return SpinType::None;
    // Only a rotation immediately before the lock can produce a spin.
    if (p.last_action != LastAction::Rotate) return SpinType::None;

    if (p.type == Piece::T) {
        Offset front[2], back[2];
        t_corners(p, front, back);
        int nfront = 0, nback = 0;
        for (int i = 0; i < 2; ++i) {
            if (b.is_solid(front[i].x, front[i].y)) ++nfront;
            if (b.is_solid(back[i].x, back[i].y)) ++nback;
        }
        if (nfront + nback >= 3) {
            if (nfront == 2) return SpinType::Full;
            // Two back corners + one front corner is a mini, unless the piece
            // arrived via the "TST/fin" kick, which upgrades it to a full spin.
            if (p.last_kick >= 4) return SpinType::Full;
            return SpinType::Mini;
        }
        // All-Mini+ additionally lets an immobile T count as a spin.
        if (mode == SpinDetection::AllMiniPlus && is_immobile(b, p)) return SpinType::Mini;
        return SpinType::None;
    }

    // Non-T pieces only spin under All-Mini / All-Mini+.
    if (mode == SpinDetection::AllMini || mode == SpinDetection::AllMiniPlus) {
        if (is_immobile(b, p)) return SpinType::Mini;
    }
    return SpinType::None;
}

}  // namespace tetra