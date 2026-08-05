// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- observation masking (spec 5.1, 7, 18.3).
//
// This is the ONLY place a policy is allowed to read game state from. The
// simulator's full GameState contains hidden information (the RNG state, the
// unshuffled remainder of the bag, the hole column of garbage that has not
// risen yet). The Observation is what survives the mask `M`, and the
// information-leak tests assert that nothing else can get through.
#pragma once

#include "tetra/bitboard.hpp"
#include "tetra/events.hpp"
#include "tetra/player.hpp"
#include "tetra/ruleset.hpp"

#include <vector>

namespace tetra {

// A garbage group as the receiving player is allowed to see it: how many lines
// and when they land -- never which column the hole is in.
struct ObservedGarbage {
    int lines = 0;
    Tick ticks_until_arrival = 0;    // <= 0 once it has arrived in the queue
    Tick ticks_until_activation = 0; // <= 0 once it may rise
    bool cancellable = true;
};

// Everything one player may legally condition on.
struct Observation {
    std::uint64_t ruleset_hash = 0;
    // The rules this observation was taken under. Carried by value so that an
    // evaluator or a search node is self-contained: nothing downstream has to
    // be handed a matching RulesetConfig out of band and risk pairing an
    // observation with the wrong rules (spec 6 requires the two to travel
    // together, which is also why `ruleset_hash` is recorded alongside).
    RulesetConfig ruleset;
    Tick timestamp = 0;

    // --- self ---
    Board board;                     // occupancy + garbage planes
    ActivePiece active;
    Piece hold = Piece::None;
    bool hold_used = false;
    std::vector<Piece> next;         // exactly preview_count entries
    std::vector<Piece> bag_remaining; // public in a 7-bag game

    int combo = -1;
    int b2b_streak = 0;
    int surge = 0;
    int pieces_placed = 0;

    std::vector<ObservedGarbage> pending_garbage;
    int pending_lines = 0;
    int active_garbage_lines = 0;

    bool alive = true;

    // --- opponent (M4; present in two-board search/self-play) ---
    bool has_opponent = false;
    Board opponent_board;
    int opponent_pending_lines = 0;
    int opponent_combo = -1;
    int opponent_b2b = 0;
    bool opponent_alive = true;

    // --- history ---
    std::vector<Event> recent_events;
};

// Build the observation for `p`. `opponent` may be null (single-board modes).
inline Observation observe(const Player& p, const Player* opponent = nullptr) {
    Observation o;
    const RulesetConfig& cfg = p.ruleset();

    o.ruleset_hash = cfg.hash();
    o.ruleset = cfg;
    o.timestamp = p.now();
    o.board = p.board();
    o.active = p.active();
    o.hold = p.hold();
    o.hold_used = p.hold_used();
    o.next = p.visible_next();  // already truncated to preview_count
    o.bag_remaining = p.queue().bag_remaining();

    o.combo = p.attack_state().combo;
    o.b2b_streak = p.attack_state().b2b_streak;
    o.surge = p.attack_state().surge;
    o.pieces_placed = p.attack_state().pieces_placed;
    o.alive = p.alive();

    const Tick now = p.now();
    for (const auto& e : p.garbage().entries()) {
        if (e.lines <= 0) continue;
        ObservedGarbage g;
        g.lines = e.lines;
        g.ticks_until_arrival = e.arrival_at - now;
        g.ticks_until_activation = e.activation_at - now;
        g.cancellable = e.cancellable;
        // NOTE: e.hole_column is deliberately NOT copied.
        o.pending_garbage.push_back(g);
        o.pending_lines += e.lines;
    }
    o.active_garbage_lines = p.garbage().active_lines(now);

    for (const auto& e : p.events().events()) o.recent_events.push_back(e);

    if (opponent) {
        o.has_opponent = true;
        o.opponent_board = opponent->board();
        o.opponent_pending_lines = opponent->garbage().total_lines();
        o.opponent_combo = opponent->attack_state().combo;
        o.opponent_b2b = opponent->attack_state().b2b_streak;
        o.opponent_alive = opponent->alive();
    }
    return o;
}

}  // namespace tetra
