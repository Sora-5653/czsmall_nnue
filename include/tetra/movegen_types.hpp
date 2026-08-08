// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- movegen action and timing types.
//
// The legal placement search itself lives in src/movegen.cpp and is provided
// by the vendored Cobra backend.  This header contains the project-facing
// action representation and the small, shared replay/evaluation helpers.
#pragma once

#include "tetra/bitboard.hpp"
#include "tetra/piece_state.hpp"
#include "tetra/pieces.hpp"
#include "tetra/ruleset.hpp"
#include "tetra/timing.hpp"
#include "tetra/types.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace tetra {

// Canonical input alphabet (spec 8.3 canonical_input_sequence).
enum class Input : std::uint8_t {
    Left = 0,
    Right,
    DasLeft,   // hold left until the wall / obstruction
    DasRight,
    Cw,
    Ccw,
    Flip,      // 180
    SoftDrop,  // move down one cell without locking
    HardDrop,
    Hold,
};

inline const char* input_name(Input i) {
    switch (i) {
        case Input::Left: return "L";
        case Input::Right: return "R";
        case Input::DasLeft: return "DL";
        case Input::DasRight: return "DR";
        case Input::Cw: return "CW";
        case Input::Ccw: return "CCW";
        case Input::Flip: return "180";
        case Input::SoftDrop: return "SD";
        case Input::HardDrop: return "HD";
        case Input::Hold: return "HOLD";
    }
    return "?";
}

// A fully specified placement (spec 8.1 PlacementAction).
struct PlacementAction {
    bool use_hold = false;
    Piece final_piece = Piece::None;
    int final_x = 0;
    int final_y = 0;
    Rot final_rotation = Rot::N;

    SpinType spin = SpinType::None;
    int last_kick = 0;
    std::vector<Input> canonical_input_sequence;

    int cleared_lines = 0;
    bool cleared_garbage = false;
    bool all_clear = false;
    std::uint64_t resulting_board_hash = 0;

    Tick base_duration = 0;
    Tick base_duration_adjust = 0;
    DelayBin delay_bin = DelayBin::Fastest;
    Tick delay_ticks = 0;

    Tick total_duration() const { return base_duration + delay_ticks; }

    ActivePiece piece_state() const {
        ActivePiece p;
        p.type = final_piece;
        p.x = final_x;
        p.y = final_y;
        p.rot = final_rotation;
        p.last_action = (spin != SpinType::None) ? LastAction::Rotate : LastAction::Drop;
        p.last_kick = last_kick;
        return p;
    }
};

namespace detail {

inline std::uint64_t board_hash(const Board& b) {
    std::uint64_t h = 1469598103934665603ull;
    for (int y = 0; y < b.height(); ++y) {
        const std::uint32_t row = b.row(y);
        for (int i = 0; i < 4; ++i) {
            h ^= static_cast<std::uint8_t>((row >> (i * 8)) & 0xFF);
            h *= 1099511628211ull;
        }
    }
    return h;
}

}  // namespace detail

// Replay a canonical input sequence and return the resulting piece and exact
// tick cost under the ruleset's handling settings.
struct ExecutionResult {
    ActivePiece piece;
    Tick cost = 0;
    bool ok = false;
};

inline ExecutionResult execute_inputs(const Board& board, const ActivePiece& start,
                                      const std::vector<Input>& seq,
                                      const RulesetConfig& cfg) {
    ExecutionResult r;
    const HandlingModel h = HandlingModel::from(cfg);
    ActivePiece p = start;
    if (collides(board, p)) return r;

    for (const Input in : seq) {
        switch (in) {
            case Input::Hold:
                r.cost += h.hold;
                break;
            case Input::Left:
            case Input::Right: {
                ActivePiece q = p;
                q.x += (in == Input::Left) ? -1 : 1;
                if (!collides(board, q)) {
                    q.last_action = LastAction::Move;
                    q.last_kick = 0;
                    p = q;
                }
                r.cost += h.tap;
                break;
            }
            case Input::DasLeft:
            case Input::DasRight: {
                const int dx = (in == Input::DasLeft) ? -1 : 1;
                int moved = 0;
                while (true) {
                    ActivePiece q = p;
                    q.x += dx;
                    if (collides(board, q)) break;
                    q.last_action = LastAction::Move;
                    q.last_kick = 0;
                    p = q;
                    if (++moved > Board::MAX_WIDTH) break;
                }
                r.cost += h.das_shift_cost(moved);
                break;
            }
            case Input::Cw:
                try_rotate(board, p, rot_cw(p.rot), cfg);
                r.cost += h.rotate;
                break;
            case Input::Ccw:
                try_rotate(board, p, rot_ccw(p.rot), cfg);
                r.cost += h.rotate;
                break;
            case Input::Flip:
                try_rotate(board, p, rot_180(p.rot), cfg);
                r.cost += h.rotate;
                break;
            case Input::SoftDrop: {
                ActivePiece q = p;
                q.y -= 1;
                if (!collides(board, q)) {
                    q.last_action = LastAction::Drop;
                    q.last_kick = 0;
                    p = q;
                }
                r.cost += h.soft_drop_cost(1);
                break;
            }
            case Input::HardDrop: {
                const int d = hard_drop_distance(board, p);
                if (d > 0) {
                    p.y -= d;
                    p.last_action = LastAction::Drop;
                    p.last_kick = 0;
                }
                r.cost += h.hard_drop;
                break;
            }
        }
    }

    r.cost += h.are;
    r.piece = p;
    r.ok = true;
    return r;
}

struct PlacementOutcome {
    int cleared_lines = 0;
    bool cleared_garbage = false;
    bool all_clear = false;
    SpinType spin = SpinType::None;
    std::uint64_t board_hash = 0;
    Board board;
};

inline PlacementOutcome evaluate_placement(const Board& board, const ActivePiece& piece,
                                           const RulesetConfig& cfg) {
    PlacementOutcome out;
    out.spin = detect_spin(board, piece, cfg);
    out.board = board;

    Offset cells[4];
    piece_cells(piece, cells);
    for (const auto& c : cells) out.board.fill_cell(c.x, c.y, false);

    for (int y = 0; y < out.board.height(); ++y) {
        if (out.board.row_full(y)) {
            ++out.cleared_lines;
            if (out.board.garbage_row(y) != 0u) out.cleared_garbage = true;
        }
    }
    if (out.cleared_lines > 0) out.board.clear_full_rows();
    out.all_clear = out.cleared_lines > 0 && out.board.empty();
    out.board_hash = detail::board_hash(out.board);
    return out;
}

}  // namespace tetra
