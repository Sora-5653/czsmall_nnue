// SPDX-License-Identifier: Apache-2.0
// Adaptation boundary for Kixenon/cobra-movegen. The vendored implementation
// itself is retained under include/cobra/src with its upstream source text.
//
// Copyright 2026 Kixenon
// Modifications: this translation unit connects the upstream row backend to
// Tetra's runtime ruleset and board types.
#include "tetra/movegen.hpp"

#include "cobra/src/row/board.hpp"
#include "cobra/src/row/movegen.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <unordered_set>

namespace tetra {
namespace cobra_movegen_detail {

using CobraBoard = Cobra::Board<>;

struct BoardView {
    CobraBoard board{};
    int horizontal_offset = 0;
};

inline bool supported(const Board& board, const RulesetConfig& cfg) {
    // Cobra's row backend is deliberately a fixed 10-column, 48-row engine.
    // Narrow boards are represented inside that field; wider or unusually
    // tall custom rooms stay on the compatibility path.
    if (cfg.geometry.width <= 0 || cfg.geometry.width > Cobra::COL_NB) return false;
    if (board.width() != cfg.geometry.width || board.height() > Cobra::ROW_NB) return false;
    if (cfg.geometry.internal_height > Cobra::ROW_NB) return false;
    if (cfg.geometry.visible_height != 20) return false;
    if (cfg.movement.kick_table == KickTableId::None) return false;
    if (board.stack_height() >= Cobra::ROW_NB - 4) return false;
    return true;
}

inline BoardView to_cobra_board(const Board& source) {
    BoardView view;
    view.horizontal_offset = (Cobra::COL_NB - source.width()) / 2;
    for (int y = 0; y < source.height(); ++y) {
        const std::uint32_t row = source.row(y);
        for (int x = 0; x < source.width(); ++x) {
            if ((row & (std::uint32_t{1} << x)) != 0)
                view.board.set(x + view.horizontal_offset, y);
        }
    }
    return view;
}

inline Cobra::Piece to_cobra_piece(Piece p) {
    switch (p) {
        case Piece::I: return Cobra::Piece::I;
        case Piece::J: return Cobra::Piece::J;
        case Piece::L: return Cobra::Piece::L;
        case Piece::O: return Cobra::Piece::O;
        case Piece::S: return Cobra::Piece::S;
        case Piece::T: return Cobra::Piece::T;
        case Piece::Z: return Cobra::Piece::Z;
        case Piece::None: break;
    }
    return Cobra::Piece::NO_PIECE;
}

inline Rot to_tetra_rotation(Cobra::Rotation r) {
    switch (r.value) {
        case Cobra::Rotation::NORTH: return Rot::N;
        case Cobra::Rotation::EAST: return Rot::R;
        case Cobra::Rotation::SOUTH: return Rot::S2;
        case Cobra::Rotation::WEST: return Rot::L;
    }
    return Rot::N;
}

inline std::array<Offset, 4> cobra_offsets(Cobra::Piece p, Cobra::Rotation r) {
    return p.route([&]<Cobra::Piece P> {
        return r.route([&]<Cobra::Rotation R> {
            constexpr auto cells = Cobra::piece_table<P, R>();
            return std::array<Offset, 4>{
                Offset{0, 0},
                Offset{static_cast<int>(cells[0].x), static_cast<int>(cells[0].y)},
                Offset{static_cast<int>(cells[1].x), static_cast<int>(cells[1].y)},
                Offset{static_cast<int>(cells[2].x), static_cast<int>(cells[2].y)},
            };
        });
    });
}

struct OriginDelta {
    int x = 0;
    int y = 0;
};

// Cobra rotates around a compact anchor, while Tetra stores the lower-left
// corner of the SRS bounding box.  Find the constant translation for each
// piece/rotation pair instead of duplicating a second hand-written table.
inline OriginDelta origin_delta(Piece p, Rot r) {
    const auto cobra = cobra_offsets(to_cobra_piece(p), [&] {
        switch (r) {
            case Rot::N: return Cobra::Rotation{Cobra::Rotation::NORTH};
            case Rot::R: return Cobra::Rotation{Cobra::Rotation::EAST};
            case Rot::S2: return Cobra::Rotation{Cobra::Rotation::SOUTH};
            case Rot::L: return Cobra::Rotation{Cobra::Rotation::WEST};
        }
        return Cobra::Rotation{Cobra::Rotation::NORTH};
    }());

    const PieceShape& shape = shape_of(p, r);
    for (int dx = -4; dx <= 4; ++dx) {
        for (int dy = -4; dy <= 4; ++dy) {
            std::array<bool, 4> used{};
            bool match = true;
            for (const Offset& c : cobra) {
                bool found = false;
                for (size_t i = 0; i < shape.cells.size(); ++i) {
                    const Offset& t = shape.cells[i];
                    if (!used[i] && c.x + dx == t.x && c.y + dy == t.y) {
                        used[i] = true;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    match = false;
                    break;
                }
            }
            if (match) return OriginDelta{dx, dy};
        }
    }
    assert(false && "Cobra and Tetra piece tables are not aligned");
    return {};
}

std::uint64_t placement_key(const ActivePiece& piece) {
    Offset cells[4];
    piece_cells(piece, cells);

    std::array<std::uint32_t, 4> sorted{};
    for (size_t i = 0; i < sorted.size(); ++i) {
        const int x = cells[i].x + 32;
        const int y = cells[i].y + 32;
        sorted[i] = (static_cast<std::uint32_t>(x) << 8) |
                    static_cast<std::uint32_t>(y & 0xFF);
    }
    std::sort(sorted.begin(), sorted.end());

    std::uint64_t h = 1469598103934665603ull;
    h ^= static_cast<std::uint64_t>(piece.type);
    h *= 1099511628211ull;
    for (const std::uint32_t cell : sorted) {
        h ^= cell;
        h *= 1099511628211ull;
    }
    return h;
}

template <Cobra::Policy::KickRule Kick, bool Enable180, bool TrackTSpin>
struct CobraRules : Cobra::RulesetBase {
    static constexpr Cobra::Policy::KickRule KICKS = Kick;
    static constexpr Cobra::Policy::SpinRule SPINS =
        TrackTSpin ? Cobra::Policy::SpinRule::TSPIN : Cobra::Policy::SpinRule::NONE;
    static constexpr int SPAWN_Y = 20;
    static constexpr bool ENABLE_180 = Enable180;
};

template <typename Rules, Cobra::Piece CP>
void collect_piece(const BoardView& view, Piece tetra_piece, std::unordered_set<std::uint64_t>& out) {
    const Cobra::MoveList<Rules, CP, CobraBoard> moves(view.board, view.board.max_y());
    moves.for_each_move([&]<Cobra::Rotation R>(const int x, const int y,
                                                [[maybe_unused]] const Cobra::SpinType spin) {
        const Rot rotation = to_tetra_rotation(R);
        const OriginDelta delta = origin_delta(tetra_piece, rotation);

        ActivePiece piece;
        piece.type = tetra_piece;
        piece.rot = rotation;
        piece.x = x - delta.x - view.horizontal_offset;
        piece.y = y - delta.y;
        out.insert(placement_key(piece));
    });
}

template <typename Rules>
void collect_piece(const BoardView& view, Piece piece, std::unordered_set<std::uint64_t>& out) {
    switch (piece) {
        case Piece::I: collect_piece<Rules, Cobra::Piece::I>(view, piece, out); break;
        case Piece::J: collect_piece<Rules, Cobra::Piece::J>(view, piece, out); break;
        case Piece::L: collect_piece<Rules, Cobra::Piece::L>(view, piece, out); break;
        case Piece::O: collect_piece<Rules, Cobra::Piece::O>(view, piece, out); break;
        case Piece::S: collect_piece<Rules, Cobra::Piece::S>(view, piece, out); break;
        case Piece::T: collect_piece<Rules, Cobra::Piece::T>(view, piece, out); break;
        case Piece::Z: collect_piece<Rules, Cobra::Piece::Z>(view, piece, out); break;
        case Piece::None: break;
    }
}

template <Cobra::Policy::KickRule Kick, bool Enable180>
void collect_with_kick(const BoardView& view, Piece piece, const RulesetConfig& cfg,
                       std::unordered_set<std::uint64_t>& out) {
    if (piece == Piece::T && cfg.clear_rules.spin_detection != SpinDetection::None)
        collect_piece<CobraRules<Kick, Enable180, true>>(view, piece, out);
    else
        collect_piece<CobraRules<Kick, Enable180, false>>(view, piece, out);
}

bool collect_targets(const Board& board, Piece piece, const RulesetConfig& cfg,
                     std::unordered_set<std::uint64_t>& out) {
    if (!supported(board, cfg)) return false;
    const BoardView view = to_cobra_board(board);

    switch (cfg.movement.kick_table) {
        case KickTableId::SRS:
            if (cfg.movement.allow_180)
                collect_with_kick<Cobra::Policy::KickRule::SRS, true>(view, piece, cfg, out);
            else
                collect_with_kick<Cobra::Policy::KickRule::SRS, false>(view, piece, cfg, out);
            return true;
        case KickTableId::SRS_PLUS:
        case KickTableId::SRS_X:
            if (cfg.movement.allow_180)
                collect_with_kick<Cobra::Policy::KickRule::SRS_PLUS, true>(view, piece, cfg, out);
            else
                collect_with_kick<Cobra::Policy::KickRule::SRS_PLUS, false>(view, piece, cfg, out);
            return true;
        case KickTableId::None:
            return false;
    }
    return false;
}

}  // namespace cobra_movegen_detail
}  // namespace tetra
