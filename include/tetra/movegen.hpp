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
#include "tetra/timing.hpp"
#include "tetra/types.hpp"

#include <algorithm>
#include <cstdint>
#include <algorithm>
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

    // Spec 8.4 timing. `base_duration` is the cost of actually executing the
    // canonical input sequence under the ruleset's handling settings, measured
    // in integer ticks. `delay_bin` is the deliberate extra wait layered on top
    // by the search, and `delay_ticks` is that wait already resolved against
    // the world (for WAIT_FOR_EVENT it depends on garbage and the opponent).
    Tick base_duration = 0;
    Tick base_duration_adjust = 0;  // internal: trailing-soft-drop elision
    DelayBin delay_bin = DelayBin::Fastest;
    Tick delay_ticks = 0;

    // Total ticks this action occupies, which is what the simulator is given.
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

// Flat, generation-stamped visited table.
//
// The packed movement state is 20 bits, so a direct-mapped array of ~1M slots
// replaces the hash map entirely. Clearing it between calls would cost more
// than the search itself, so each call bumps a generation counter and a slot
// only counts as occupied when its stamp matches the current generation. The
// buffer is reused across calls via a thread_local scratch object, which keeps
// the generator allocation-free in the MCTS inner loop.
struct VisitedTable {
    static constexpr std::uint32_t SIZE = 1u << 20;
    std::vector<std::uint32_t> stamp;
    std::vector<std::int32_t> value;
    std::uint32_t generation = 0;

    VisitedTable() : stamp(SIZE, 0), value(SIZE, -1) {}

    void begin() {
        if (++generation == 0) {  // wrapped: clear once every 4 billion calls
            std::fill(stamp.begin(), stamp.end(), 0);
            generation = 1;
        }
    }
    std::int32_t* find(std::uint32_t key) {
        return (stamp[key] == generation) ? &value[key] : nullptr;
    }
    void insert(std::uint32_t key, std::int32_t v) {
        stamp[key] = generation;
        value[key] = v;
    }
};

inline VisitedTable& scratch_visited() {
    static thread_local VisitedTable t;
    return t;
}

inline VisitedTable& scratch_landing_table() {
    static thread_local VisitedTable t;
    return t;
}

// BFS node with a parent link instead of an owned path. Copying a path per
// expansion dominated the profile (the generator is the MCTS inner loop), so
// the path is reconstructed only for the placements that are actually emitted.
struct BfsNode {
    ActivePiece piece;
    std::int32_t parent = -1;  // index into the node arena, -1 for the root
    Input step = Input::HardDrop;
    std::uint16_t depth = 0;
    // Accumulated execution cost from the spawn to this node, in ticks. Kept
    // incrementally so that pricing an action is free; re-executing the input
    // sequence to price it would double the cost of generation.
    Tick cost = 0;
};

// Walk the parent chain back to the root and return the inputs in order.
inline std::vector<Input> reconstruct_path(const std::vector<BfsNode>& arena, std::int32_t idx) {
    std::vector<Input> out;
    if (idx < 0) return out;
    out.reserve(arena[static_cast<size_t>(idx)].depth);
    while (idx >= 0) {
        const BfsNode& n = arena[static_cast<size_t>(idx)];
        if (n.parent < 0) break;
        out.push_back(n.step);
        idx = n.parent;
    }
    std::reverse(out.begin(), out.end());
    return out;
}

// Reusable BFS working buffers, so a generate() call performs no heap
// allocation once the buffers have grown to their steady-state size.
struct BfsScratch {
    std::vector<BfsNode> arena;
    std::vector<std::int32_t> landings;
    std::vector<std::int32_t> unique_landings;
};

// Monotone bucket queue (Dial's algorithm) over the node cost.
//
// A binary heap is the textbook choice for Dijkstra, but here every edge weight
// is a small integer number of ticks and the settled cost never decreases, so
// bucketing by cost is both simpler and measurably faster: pops and pushes are
// O(1) and there is no comparison overhead. Buckets are reused between calls
// via the scratch object, so steady-state generation stays allocation-free.
class NodeQueue {
public:
    void reset() {
        for (auto& b : buckets_) b.clear();
        head_ = 0;
        size_ = 0;
    }

    void push(Tick cost, std::int32_t idx) {
        const size_t c = static_cast<size_t>(cost);
        if (c >= buckets_.size()) buckets_.resize(c + 16);
        buckets_[c].push_back(idx);
        ++size_;
    }

    bool empty() const { return size_ == 0; }

    // Returns the index of a cheapest pending node.
    std::int32_t pop() {
        while (head_ < buckets_.size() && buckets_[head_].empty()) ++head_;
        if (head_ >= buckets_.size()) return -1;
        const std::int32_t idx = buckets_[head_].back();
        buckets_[head_].pop_back();
        --size_;
        return idx;
    }

    Tick current_cost() const { return static_cast<Tick>(head_); }

private:
    std::vector<std::vector<std::int32_t>> buckets_;
    size_t head_ = 0;
    size_t size_ = 0;
};

inline NodeQueue& scratch_queue() {
    static thread_local NodeQueue q;
    return q;
}

inline BfsScratch& scratch_bfs() {
    static thread_local BfsScratch s;
    return s;
}

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

// ---------------------------------------------------------------------------
// Input execution
// ---------------------------------------------------------------------------
// Replay a canonical input sequence against a board, returning the resulting
// piece and the exact tick cost under the ruleset's handling settings.
//
// This is the single source of truth for "what does this input sequence do and
// how long does it take". The generator uses it to price every action it
// emits, and the tests use it to verify that a sequence really lands where the
// generator claims -- so the cost model can never drift away from the movement
// model.
struct ExecutionResult {
    ActivePiece piece;
    Tick cost = 0;
    bool ok = false;  // false if the sequence was not executable from `start`
};

inline ExecutionResult execute_inputs(const Board& board, const ActivePiece& start,
                                      const std::vector<Input>& seq, const RulesetConfig& cfg) {
    ExecutionResult r;
    const HandlingModel h = HandlingModel::from(cfg);
    ActivePiece p = start;
    if (collides(board, p)) return r;

    for (const Input in : seq) {
        switch (in) {
            case Input::Hold:
                // The caller performs the swap; only the input cost is charged.
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
                const int d = hard_drop_distance(board, p);
                if (d > 0) {
                    p.y -= d;
                    p.last_action = LastAction::Drop;
                    p.last_kick = 0;
                }
                r.cost += h.soft_drop_cost(d);
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
    // Spawning the next piece costs ARE regardless of what the player did.
    r.cost += h.are;
    r.piece = p;
    r.ok = true;
    return r;
}

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
        int max_states = 20000;        // search safety valve

        // Reject placements that gravity would make unreachable in time.
        // Enabled by default; `gravity_check_threshold` is the slowest gravity
        // (in ticks per cell) at which the check is worth running, so the
        // common 1/60 G case pays nothing.
        bool enforce_gravity = true;
        Tick gravity_check_threshold = 8;
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

        // Spec 8.2 calls for BFS *or Dijkstra*; because inputs have different
        // costs (a DAS slide is not a tap) this has to be Dijkstra, otherwise a
        // node reached first by an expensive route would keep that route's
        // price. Settling nodes in cost order also makes "cheapest execution"
        // literally true rather than approximately true.
        const HandlingModel h = HandlingModel::from(cfg);
        detail::BfsScratch& scratch = detail::scratch_bfs();
        std::vector<detail::BfsNode>& arena = scratch.arena;
        std::vector<std::int32_t>& landings = scratch.landings;
        arena.clear();
        landings.clear();

        detail::VisitedTable& visited = detail::scratch_visited();
        visited.begin();

        detail::NodeQueue& pq = detail::scratch_queue();
        pq.reset();

        arena.push_back(detail::BfsNode{start, -1, Input::HardDrop, 0, use_hold ? h.hold : 0});
        visited.insert(detail::pack_state(start), 0);
        pq.push(arena[0].cost, 0);

        int explored = 0;
        const bool use_180 = opt_.allow_180 && cfg.movement.allow_180;
        // Gravity only constrains anything when it is fast enough to move the
        // piece within the handful of ticks a placement takes.
        const bool check_gravity = opt_.enforce_gravity && h.gravity_num > 0 &&
                                   h.ticks_per_cell() <= opt_.gravity_check_threshold;
        const int spawn_y = start.y;

        while (!pq.empty() && explored < opt_.max_states) {
            const Tick popped_cost = pq.current_cost();
            const std::int32_t idx = pq.pop();
            if (idx < 0) break;
            // Stale entry left behind by a later improvement of the same node.
            if (popped_cost > arena[static_cast<size_t>(idx)].cost) continue;
            ++explored;

            const ActivePiece node_piece = arena[static_cast<size_t>(idx)].piece;
            const Input node_step = arena[static_cast<size_t>(idx)].step;

            // Landing candidates.
            //
            // A grounded node can obviously lock. An airborne node can also
            // lock, via the implicit hard drop.
            //
            // Any node reached by a move OR a rotation can undercut the
            // soft-drop-first route ("L HD" beats "SD L HD" by a tick), so all
            // of them are candidates. Restricting this to horizontal moves
            // alone is tempting but wrong: a mid-air rotation can also reach a
            // landing more cheaply than rotating after the drop, and
            // `action_cost_is_the_true_shortest_path` catches the difference.
            const bool airborne_shortcut =
                node_step != Input::SoftDrop && node_step != Input::HardDrop;
            if (airborne_shortcut || grounded(board, node_piece)) landings.push_back(idx);

            // Gravity reachability (spec 6, movement.gravity).
            //
            // By the time this node is reached, `cost` ticks have elapsed and
            // gravity has pulled the piece down by floor(cost * num / den)
            // cells. A node that sits higher than that is one the player could
            // never actually occupy, so it is not expanded and cannot become a
            // placement. Under the default 1/60 G nothing is ever rejected;
            // the check only bites at high gravity, which is precisely where a
            // gravity-blind generator starts inventing illegal placements.
            if (check_gravity) {
                const Tick fallen = h.gravity_fall(arena[static_cast<size_t>(idx)].cost);
                const Tick required_y = static_cast<Tick>(spawn_y) - fallen;
                if (static_cast<Tick>(node_piece.y) > required_y) continue;
            }

            // --- successors ---
            tryMove(board, node_piece, idx, -1, Input::Left, h, arena, visited, pq);
            tryMove(board, node_piece, idx, +1, Input::Right, h, arena, visited, pq);
            tryDas(board, node_piece, idx, -1, Input::DasLeft, h, arena, visited, pq);
            tryDas(board, node_piece, idx, +1, Input::DasRight, h, arena, visited, pq);
            trySoftDrop(board, node_piece, idx, h, arena, visited, pq);
            tryRotate(board, node_piece, idx, rot_cw(node_piece.rot), Input::Cw, cfg, h, arena,
                      visited, pq);
            tryRotate(board, node_piece, idx, rot_ccw(node_piece.rot), Input::Ccw, cfg, h, arena,
                      visited, pq);
            if (use_180)
                tryRotate(board, node_piece, idx, rot_180(node_piece.rot), Input::Flip, cfg, h,
                          arena, visited, pq);
        }

        // Turn landings into actions, merging paths that are truly equivalent.
        std::unordered_map<std::uint64_t, size_t> by_outcome;

        // Collapse landings that hard-drop to the same final placement before
        // doing any expensive work. Airborne shortcut nodes mean several BFS
        // nodes can share one landing spot, and evaluate_placement() is by far
        // the costliest step per candidate, so the cheapest route to each
        // distinct landing is chosen here rather than after.
        detail::VisitedTable& landing_seen = detail::scratch_landing_table();
        landing_seen.begin();
        std::vector<std::int32_t>& unique_landings = scratch.unique_landings;
        unique_landings.clear();

        for (const std::int32_t idx : landings) {
            ActivePiece l = arena[static_cast<size_t>(idx)].piece;
            const int fall = hard_drop_distance(board, l);
            if (fall > 0) {
                l.y -= fall;
                l.last_action = LastAction::Drop;
                l.last_kick = 0;
            }
            const std::uint32_t key = detail::pack_state(l);
            const Tick cost = arena[static_cast<size_t>(idx)].cost;
            if (std::int32_t* slot = landing_seen.find(key)) {
                if (cost < arena[static_cast<size_t>(*slot)].cost) *slot = idx;
                continue;
            }
            landing_seen.insert(key, idx);
            unique_landings.push_back(idx);
        }

        for (const std::int32_t slot_idx : unique_landings) {
            // `unique_landings` holds the first node seen for each landing; the
            // table holds the cheapest, which is the one to emit.
            ActivePiece probe = arena[static_cast<size_t>(slot_idx)].piece;
            const int probe_fall = hard_drop_distance(board, probe);
            if (probe_fall > 0) {
                probe.y -= probe_fall;
                probe.last_action = LastAction::Drop;
                probe.last_kick = 0;
            }
            const std::int32_t idx = *landing_seen.find(detail::pack_state(probe));

            // The action locks wherever a hard drop from this node lands.
            // Nodes that are already grounded drop zero cells.
            const detail::BfsNode& node = arena[static_cast<size_t>(idx)];
            ActivePiece landed = node.piece;
            const int fall = hard_drop_distance(board, landed);
            if (fall > 0) {
                landed.y -= fall;
                landed.last_action = LastAction::Drop;
                landed.last_kick = 0;
            }
            PlacementAction a;
            a.use_hold = use_hold;
            a.final_piece = landed.type;
            a.final_x = landed.x;
            a.final_y = landed.y;
            a.final_rotation = landed.rot;
            a.last_kick = landed.last_kick;
            a.canonical_input_sequence = detail::reconstruct_path(arena, idx);
            // A hard drop already falls all the way to the floor, so a soft
            // drop immediately before it is redundant.
            if (!a.canonical_input_sequence.empty() &&
                a.canonical_input_sequence.back() == Input::SoftDrop) {
                a.canonical_input_sequence.pop_back();
                a.base_duration_adjust = -h.soft_drop_cost(fall > 0 ? fall : 1);
            }
            a.canonical_input_sequence.push_back(Input::HardDrop);
            if (use_hold)
                a.canonical_input_sequence.insert(a.canonical_input_sequence.begin(), Input::Hold);

            // Price the action from the cost accumulated along its BFS path,
            // plus the hard drop that locks it and the spawn delay for the next
            // piece. `duration_matches_replaying_the_canonical_sequence` checks
            // this against a full re-execution of the input sequence, so the
            // incremental sum can never silently drift from the movement model.
            a.base_duration = arena[static_cast<size_t>(idx)].cost + h.hard_drop + h.are +
                              a.base_duration_adjust;

            const PlacementOutcome oc = evaluate_placement(board, landed, cfg);
            a.spin = oc.spin;
            if (oc.cleared_lines > 0) a.base_duration += cfg.clear_rules.line_clear_delay;
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
                // Keep the cheapest execution of this outcome, breaking ties on
                // the shorter input sequence. Now that actions are priced, "the
                // fewest inputs" and "the fastest" are not always the same
                // thing: a single DAS costs DAS+ARR while two taps cost two
                // frames.
                PlacementAction& kept = out[it->second];
                const bool cheaper = a.base_duration < kept.base_duration;
                const bool same_cost_shorter =
                    a.base_duration == kept.base_duration &&
                    a.canonical_input_sequence.size() < kept.canonical_input_sequence.size();
                if (cheaper || same_cost_shorter) {
                    // Replace the WHOLE execution, not just the inputs. Two
                    // different (x, y, rot) triples can place the identical set
                    // of cells -- an I piece is the obvious case, since rot N at
                    // y and rot 2 at y+1 fill the same row -- so they hash to
                    // the same outcome and get merged here. Copying only the
                    // input sequence would leave the action describing one
                    // representation while its inputs produce the other, and
                    // `canonical_input_sequence` would no longer reproduce
                    // `final_x/final_y/final_rotation`.
                    kept.canonical_input_sequence = a.canonical_input_sequence;
                    kept.base_duration = a.base_duration;
                    kept.last_kick = a.last_kick;
                    kept.final_x = a.final_x;
                    kept.final_y = a.final_y;
                    kept.final_rotation = a.final_rotation;
                    kept.final_piece = a.final_piece;
                }
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

    // -----------------------------------------------------------------------
    // Timing actions (spec 8.4)
    // -----------------------------------------------------------------------
    // Expand each placement into one action per delay bin. This is what turns
    // "where do I put the piece" into "where, and when" -- the decision the
    // spec insists must be learned rather than hardcoded (spec 12).
    //
    // `now` is the current tick. `next_garbage_activation` and
    // `opponent_next_lock` bound WAIT_FOR_EVENT; pass TICK_NEVER when unknown.
    // Bins that resolve to the same wait as an earlier bin are dropped, so a
    // state with nothing to wait for does not get six copies of one action.
    static std::vector<PlacementAction> expand_delay_bins(
        const std::vector<PlacementAction>& actions, const RulesetConfig& cfg, Tick now,
        Tick next_garbage_activation = TICK_NEVER, Tick opponent_next_lock = TICK_NEVER,
        const std::vector<DelayBin>& bins = default_delay_bins()) {
        std::vector<PlacementAction> out;
        out.reserve(actions.size() * bins.size());

        const HandlingModel h = HandlingModel::from(cfg);
        for (const auto& base : actions) {
            std::vector<Tick> seen_waits;
            for (const DelayBin bin : bins) {
                Tick wait = delay_bin_ticks(bin);
                if (bin == DelayBin::WaitForEvent) {
                    wait = resolve_wait_for_event(now, next_garbage_activation,
                                                  opponent_next_lock, h);
                    // A wait of zero is just FASTEST, and a wait that coincides
                    // with a fixed bin is that bin: no point duplicating it.
                    if (wait <= 0) continue;
                }
                if (std::find(seen_waits.begin(), seen_waits.end(), wait) != seen_waits.end())
                    continue;
                seen_waits.push_back(wait);

                PlacementAction a = base;
                a.delay_bin = bin;
                a.delay_ticks = wait;
                out.push_back(std::move(a));
            }
        }
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

    // Insert a successor if its movement state has not been reached before, or
    // if this path reaches it in fewer inputs.
    static void push_if_new(const ActivePiece& p, std::int32_t parent, Input step, Tick step_cost,
                            std::vector<detail::BfsNode>& arena, detail::VisitedTable& visited,
                            detail::NodeQueue& pq) {
        const std::uint32_t key = detail::pack_state(p);
        const detail::BfsNode& from = arena[static_cast<size_t>(parent)];
        const std::uint16_t depth = static_cast<std::uint16_t>(from.depth + 1);
        const Tick cost = from.cost + step_cost;
        if (std::int32_t* slot = visited.find(key)) {
            // Already reachable; keep the cheapest route (breaking ties on the
            // shorter one) but do not re-expand.
            detail::BfsNode& existing = arena[static_cast<size_t>(*slot)];
            if (cost < existing.cost || (cost == existing.cost && depth < existing.depth)) {
                existing.parent = parent;
                existing.step = step;
                existing.depth = depth;
                existing.cost = cost;
                pq.push(cost, *slot);  // decrease-key by re-insertion
            }
            return;
        }
        const std::int32_t idx = static_cast<std::int32_t>(arena.size());
        arena.push_back(detail::BfsNode{p, parent, step, depth, cost});
        visited.insert(key, idx);
        pq.push(cost, idx);
    }

    static void tryMove(const Board& board, const ActivePiece& from, std::int32_t idx, int dx,
                        Input step, const HandlingModel& h, std::vector<detail::BfsNode>& arena,
                        detail::VisitedTable& visited,
                        detail::NodeQueue& pq) {
        ActivePiece p = from;
        p.x += dx;
        if (collides(board, p)) return;
        p.last_action = LastAction::Move;
        p.last_kick = 0;
        push_if_new(p, idx, step, h.tap, arena, visited, pq);
    }

    // DAS: slide as far as possible in one direction. Modelled as a single
    // input so that the canonical sequences stay short and human-playable.
    static void tryDas(const Board& board, const ActivePiece& from, std::int32_t idx, int dx,
                       Input step, const HandlingModel& h, std::vector<detail::BfsNode>& arena,
                       detail::VisitedTable& visited,
                       detail::NodeQueue& pq) {
        ActivePiece p = from;
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
        push_if_new(p, idx, step, h.das_shift_cost(moved), arena, visited, pq);
    }

    static void trySoftDrop(const Board& board, const ActivePiece& from, std::int32_t idx,
                            const HandlingModel& h, std::vector<detail::BfsNode>& arena,
                            detail::VisitedTable& visited,
                            detail::NodeQueue& pq) {
        const int d = hard_drop_distance(board, from);
        if (d <= 0) return;
        ActivePiece p = from;
        p.y -= d;
        p.last_action = LastAction::Drop;
        p.last_kick = 0;
        push_if_new(p, idx, Input::SoftDrop, h.soft_drop_cost(d), arena, visited, pq);
    }

    static void tryRotate(const Board& board, const ActivePiece& from, std::int32_t idx, Rot to,
                          Input step, const RulesetConfig& cfg, const HandlingModel& h,
                          std::vector<detail::BfsNode>& arena,
                          detail::VisitedTable& visited,
                          detail::NodeQueue& pq) {
        ActivePiece p = from;
        if (!try_rotate(board, p, to, cfg).success) return;
        push_if_new(p, idx, step, h.rotate, arena, visited, pq);
    }

    Options opt_{};
};

}  // namespace tetra