// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- M1 legal placement generation (spec 8).
//
// A macro action is "one piece, fully locked". The generator does a BFS over
// reachable (x, y, rot) states using the *actual* movement primitives, so that
// kicks, spins and tucks are discovered rather than enumerated by hand, and
// every action carries a canonical input sequence that reproduces it.
#pragma once

#include "tetra/bitboard.hpp"
#include "tetra/piece_state.hpp"
#include "tetra/pieces.hpp"
#include "tetra/ruleset.hpp"
#include "tetra/types.hpp"

#include <algorithm>
#include <cstdint>
#include <queue>
#include <unordered_map>
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
    SoftDrop,  // fall to the floor without locking
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

    SpinType spin = SpinType::None;   // classification at the moment of locking
    int last_kick = 0;
    std::vector<Input> canonical_input_sequence;

    // Filled in by evaluate_placement(): the outcome used for action equality.
    int cleared_lines = 0;
    bool cleared_garbage = false;
    bool all_clear = false;
    std::uint64_t resulting_board_hash = 0;

    // Spec 8.4 timing action. The base generator emits FASTEST; the delay bins
    // are layered on by the search, which knows the opponent's clock.
    int delay_bin = 0;

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

// Key for the BFS visited set.
//
// Spec 8.2 requires the search state to carry more than the coordinates: two
// paths that reach the same (x, y, rot) are NOT interchangeable if one arrived
// by rotating and the other by moving, because only the former can register a
// spin, and the kick index that was used further distinguishes mini from full
// T-spins. Collapsing them on coordinates alone makes the result depend on BFS
// visit order (and, observably, breaks left/right mirror invariance).
//
// Packed layout: x (6 bits) | y (7 bits) | rot (2 bits) | rotated (1 bit) |
//                kick (4 bits)
inline std::uint32_t pack_state(int x, int y, Rot r, bool arrived_by_rotation, int kick) {
    const std::uint32_t ux = static_cast<std::uint32_t>(x + 8) & 0x3F;
    const std::uint32_t uy = static_cast<std::uint32_t>(y + 8) & 0x7F;
    const std::uint32_t rr = arrived_by_rotation ? 1u : 0u;
    const std::uint32_t kk = static_cast<std::uint32_t>(arrived_by_rotation ? kick : 0) & 0xF;
    return (ux << 14) | (uy << 7) | (static_cast<std::uint32_t>(r) << 5) | (rr << 4) | kk;
}

inline std::uint32_t pack_state(const ActivePiece& p) {
    return pack_state(p.x, p.y, p.rot, p.last_action == LastAction::Rotate, p.last_kick);
}

struct BfsNode {
    ActivePiece piece;
    std::vector<Input> path;
};

// Hash of the locked-in board, used to merge input paths that produce the same
// final position (spec 8.3).
inline std::uint64_t board_hash(const Board& b) {
    std::uint64_t h = 1469598103934665603ull;
    for (int y = 0; y < b.height(); ++y) {
        std::uint32_t row = b.row(y);
        for (int i = 0; i < 4; ++i) {
            h ^= static_cast<std::uint8_t>((row >> (i * 8)) & 0xFF);
            h *= 1099511628211ull;
        }
    }
    return h;
}

}  // namespace detail

// Simulated outcome of locking a piece, without mutating the player.
struct PlacementOutcome {
    int cleared_lines = 0;
    bool cleared_garbage = false;
    bool all_clear = false;
    SpinType spin = SpinType::None;
    std::uint64_t board_hash = 0;
    Board board;  // post-lock, post-clear
};

// Apply a placement to a copy of the board and describe the result.
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

// ---------------------------------------------------------------------------
// The generator
// ---------------------------------------------------------------------------
class MoveGenerator {
public:
    struct Options {
        bool include_hold = true;
        bool allow_180 = true;
        bool merge_equivalent = true;  // spec 8.3 path equivalence merging
        int max_states = 20000;        // BFS safety valve
    };

    MoveGenerator() = default;
    explicit MoveGenerator(const Options& o) : opt_(o) {}

    // Enumerate every reachable locked placement for `piece` on `board`.
    // `use_hold` is recorded on each action but does not change the search.
    std::vector<PlacementAction> generate_for_piece(const Board& board, Piece piece,
                                                    const RulesetConfig& cfg,
                                                    bool use_hold) const {
        std::vector<PlacementAction> out;
        ActivePiece start = spawn_piece(piece, cfg);
        if (collides(board, start)) return out;  // block out: no legal moves

        // BFS over movement states.
        std::unordered_map<std::uint32_t, std::vector<Input>> visited;
        std::queue<detail::BfsNode> q;

        visited.emplace(detail::pack_state(start), std::vector<Input>{});
        q.push(detail::BfsNode{start, {}});

        std::vector<detail::BfsNode> landings;
        int explored = 0;

        while (!q.empty() && explored < opt_.max_states) {
            const detail::BfsNode node = q.front();
            q.pop();
            ++explored;

            // If the piece can lock here, record it as a candidate landing.
            if (grounded(board, node.piece)) landings.push_back(node);

            // --- successors ---
            tryMove(board, node, -1, Input::Left, visited, q);
            tryMove(board, node, +1, Input::Right, visited, q);
            tryDas(board, node, -1, Input::DasLeft, visited, q);
            tryDas(board, node, +1, Input::DasRight, visited, q);
            trySoftDrop(board, node, visited, q);
            tryRotate(board, node, rot_cw(node.piece.rot), Input::Cw, cfg, visited, q);
            tryRotate(board, node, rot_ccw(node.piece.rot), Input::Ccw, cfg, visited, q);
            if (opt_.allow_180 && cfg.movement.allow_180)
                tryRotate(board, node, rot_180(node.piece.rot), Input::Flip, cfg, visited, q);
        }

        // Turn landings into actions, merging paths that are truly equivalent.
        std::unordered_map<std::uint64_t, size_t> by_outcome;
        for (const auto& node : landings) {
            PlacementAction a;
            a.use_hold = use_hold;
            a.final_piece = node.piece.type;
            a.final_x = node.piece.x;
            a.final_y = node.piece.y;
            a.final_rotation = node.piece.rot;
            a.last_kick = node.piece.last_kick;
            a.canonical_input_sequence = node.path;
            a.canonical_input_sequence.push_back(Input::HardDrop);
            if (use_hold)
                a.canonical_input_sequence.insert(a.canonical_input_sequence.begin(), Input::Hold);

            const PlacementOutcome oc = evaluate_placement(board, node.piece, cfg);
            a.spin = oc.spin;
            a.cleared_lines = oc.cleared_lines;
            a.cleared_garbage = oc.cleared_garbage;
            a.all_clear = oc.all_clear;
            a.resulting_board_hash = oc.board_hash;

            if (!opt_.merge_equivalent) {
                out.push_back(std::move(a));
                continue;
            }

            // Spec 8.3: two paths are the same action only if the post-lock
            // board, the cleared rows, the spin class and the attack all match.
            const std::uint64_t key = equivalence_key(a);
            auto it = by_outcome.find(key);
            if (it == by_outcome.end()) {
                by_outcome.emplace(key, out.size());
                out.push_back(std::move(a));
            } else {
                // Keep the shortest (most stable) input sequence.
                PlacementAction& kept = out[it->second];
                if (a.canonical_input_sequence.size() < kept.canonical_input_sequence.size())
                    kept.canonical_input_sequence = a.canonical_input_sequence;
            }
        }
        return out;
    }

    // Full action list including the hold branch (spec 8.1 use_hold).
    std::vector<PlacementAction> generate(const Board& board, Piece current, Piece hold_piece,
                                          Piece next_after_hold, const RulesetConfig& cfg) const {
        std::vector<PlacementAction> out = generate_for_piece(board, current, cfg, false);

        if (!opt_.include_hold || !cfg.randomizer.hold_enabled) return out;

        // Holding swaps in the held piece, or the next piece if hold is empty.
        const Piece swapped = (hold_piece != Piece::None) ? hold_piece : next_after_hold;
        if (swapped == Piece::None || swapped == current) return out;

        std::vector<PlacementAction> held = generate_for_piece(board, swapped, cfg, true);
        out.insert(out.end(), held.begin(), held.end());
        return out;
    }

    const Options& options() const { return opt_; }

private:
    static std::uint64_t equivalence_key(const PlacementAction& a) {
        std::uint64_t k = a.resulting_board_hash;
        k = k * 1099511628211ull + static_cast<std::uint64_t>(a.cleared_lines);
        k = k * 1099511628211ull + static_cast<std::uint64_t>(a.spin);
        k = k * 1099511628211ull + static_cast<std::uint64_t>(a.cleared_garbage ? 1 : 0);
        k = k * 1099511628211ull + static_cast<std::uint64_t>(a.all_clear ? 1 : 0);
        k = k * 1099511628211ull + static_cast<std::uint64_t>(a.use_hold ? 1 : 0);
        k = k * 1099511628211ull + static_cast<std::uint64_t>(a.final_piece);
        return k;
    }

    void push_if_new(const Board& board, const ActivePiece& p, const std::vector<Input>& path,
                     Input step, std::unordered_map<std::uint32_t, std::vector<Input>>& visited,
                     std::queue<detail::BfsNode>& q) const {
        const std::uint32_t key = detail::pack_state(p);
        auto it = visited.find(key);
        std::vector<Input> next_path = path;
        next_path.push_back(step);
        if (it != visited.end()) {
            // Already reachable; keep the shorter path but do not re-expand.
            if (next_path.size() < it->second.size()) it->second = next_path;
            return;
        }
        visited.emplace(key, next_path);
        q.push(detail::BfsNode{p, std::move(next_path)});
    }

    void tryMove(const Board& board, const detail::BfsNode& node, int dx, Input step,
                 std::unordered_map<std::uint32_t, std::vector<Input>>& visited,
                 std::queue<detail::BfsNode>& q) const {
        ActivePiece p = node.piece;
        p.x += dx;
        if (collides(board, p)) return;
        p.last_action = LastAction::Move;
        p.last_kick = 0;
        push_if_new(board, p, node.path, step, visited, q);
    }

    // DAS: slide as far as possible in one direction. Modelled as a single
    // input so that the canonical sequences stay short and human-playable.
    void tryDas(const Board& board, const detail::BfsNode& node, int dx, Input step,
                std::unordered_map<std::uint32_t, std::vector<Input>>& visited,
                std::queue<detail::BfsNode>& q) const {
        ActivePiece p = node.piece;
        int moved = 0;
        while (true) {
            ActivePiece n = p;
            n.x += dx;
            if (collides(board, n)) break;
            p = n;
            if (++moved > Board::MAX_WIDTH) break;
        }
        if (moved == 0) return;
        p.last_action = LastAction::Move;
        p.last_kick = 0;
        push_if_new(board, p, node.path, step, visited, q);
    }

    void trySoftDrop(const Board& board, const detail::BfsNode& node,
                     std::unordered_map<std::uint32_t, std::vector<Input>>& visited,
                     std::queue<detail::BfsNode>& q) const {
        const int d = hard_drop_distance(board, node.piece);
        if (d <= 0) return;
        ActivePiece p = node.piece;
        p.y -= d;
        p.last_action = LastAction::Drop;
        p.last_kick = 0;
        push_if_new(board, p, node.path, Input::SoftDrop, visited, q);
    }

    void tryRotate(const Board& board, const detail::BfsNode& node, Rot to, Input step,
                   const RulesetConfig& cfg,
                   std::unordered_map<std::uint32_t, std::vector<Input>>& visited,
                   std::queue<detail::BfsNode>& q) const {
        ActivePiece p = node.piece;
        const RotationResult rr = try_rotate(board, p, to, cfg);
        if (!rr.success) return;
        push_if_new(board, p, node.path, step, visited, q);
    }

    Options opt_{};
};

}  // namespace tetra
