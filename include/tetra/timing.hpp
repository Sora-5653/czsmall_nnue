// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- action duration and timing actions (spec 8.4).
//
// Every macro action costs time, and that time is what makes "cancel now",
// "eat the garbage", and "hold the attack back" different decisions rather
// than the same one. This header turns a canonical input sequence into an
// exact tick count under the ruleset's handling settings, and defines the
// delay bins the search layers on top.
//
// All arithmetic is integer ticks (spec 5.2): no floating-point clocks.
#pragma once

#include "tetra/piece_state.hpp"
#include "tetra/ruleset.hpp"
#include "tetra/types.hpp"

#include <algorithm>
#include <vector>

namespace tetra {

// Input alphabet is declared in movegen.hpp; forward-declare the enum's
// underlying contract here by including it from the user side. To keep this
// header standalone the cost model takes the already-classified counts.
enum class Input : std::uint8_t;  // defined in movegen.hpp

// ---------------------------------------------------------------------------
// Handling model
// ---------------------------------------------------------------------------
// The cost of executing inputs, derived from RulesetConfig::movement so that a
// custom room's handling settings flow straight through to the search.
//
// The model is intentionally simple and explicit rather than a frame-perfect
// emulation of a human's finger: one discrete input occupies one tick unless
// DAS/ARR/SDF say otherwise. What matters for the bot is that the ordering and
// the relative magnitudes are right, that it is deterministic, and that it
// responds to the ruleset.
struct HandlingModel {
    Tick tap = 1;              // a single-cell tap shift
    Tick das = 6;              // frames to charge DAS before auto-repeat
    Tick arr = 0;              // frames per cell during auto-repeat (0 = instant)
    Tick rotate = 1;           // one rotation input
    Tick sdf = 0;              // soft drop factor (0 = instant)
    Tick hard_drop = 1;        // the drop-and-lock input itself
    Tick are = 0;              // spawn delay after a lock
    Tick line_clear_delay = 0; // extra delay when the placement clears lines
    Tick hold = 1;             // the hold input
    Tick lock_delay = 30;      // upper bound on a deliberate wait

    // Gravity as an exact rational, cells per tick (spec 6: gravity_num/den).
    // Kept rational rather than floating point so reachability is decided by
    // integer arithmetic and stays bit-reproducible (spec 5.2, 18.1).
    int gravity_num = 1;
    int gravity_den = 60;

    static HandlingModel from(const RulesetConfig& cfg) {
        HandlingModel h;
        h.das = cfg.movement.das;
        h.arr = cfg.movement.arr;
        h.sdf = cfg.movement.sdf;
        h.are = cfg.movement.are;
        h.line_clear_delay = cfg.clear_rules.line_clear_delay;
        h.lock_delay = cfg.movement.lock_delay;
        h.gravity_num = cfg.movement.gravity_num;
        h.gravity_den = cfg.movement.gravity_den;
        return h;
    }

    // Cost of shifting `cells` columns by holding the direction (DAS + ARR).
    Tick das_shift_cost(int cells) const {
        if (cells <= 0) return 0;
        // The first cell moves on the initial press, then DAS charges, then
        // auto-repeat delivers the rest at ARR each. With ARR = 0 the whole
        // remaining slide happens on one frame.
        if (cells == 1) return tap;
        const Tick repeat = (arr > 0) ? arr * static_cast<Tick>(cells - 1) : 1;
        return tap + das + repeat;
    }

    // How many cells gravity pulls the piece down over `ticks`.
    // Integer-exact: floor(ticks * num / den).
    Tick gravity_fall(Tick ticks) const {
        if (gravity_num <= 0 || gravity_den <= 0 || ticks <= 0) return 0;
        return (ticks * static_cast<Tick>(gravity_num)) / static_cast<Tick>(gravity_den);
    }

    // Ticks of free manoeuvring available before gravity drags the piece down
    // by one cell. At 20G this is 0: the piece is on the floor immediately.
    Tick ticks_per_cell() const {
        if (gravity_num <= 0) return TICK_NEVER;  // no gravity
        return static_cast<Tick>(gravity_den) / static_cast<Tick>(gravity_num);
    }

    bool is_high_gravity() const { return ticks_per_cell() <= 1; }

    // Cost of soft dropping `cells`.
    Tick soft_drop_cost(int cells) const {
        if (cells <= 0) return 0;
        if (sdf <= 0) return 1;  // infinite SDF: one frame
        return sdf * static_cast<Tick>(cells);
    }
};

// ---------------------------------------------------------------------------
// Delay bins (spec 8.4)
// ---------------------------------------------------------------------------
enum class DelayBin : std::uint8_t {
    Fastest = 0,
    Plus1F = 1,
    Plus2F = 2,
    Plus4F = 3,
    Plus8F = 4,
    WaitForEvent = 5,
};

inline constexpr int DELAY_BIN_COUNT = 6;

inline const char* delay_bin_name(DelayBin b) {
    switch (b) {
        case DelayBin::Fastest: return "FASTEST";
        case DelayBin::Plus1F: return "+1F";
        case DelayBin::Plus2F: return "+2F";
        case DelayBin::Plus4F: return "+4F";
        case DelayBin::Plus8F: return "+8F";
        case DelayBin::WaitForEvent: return "WAIT_FOR_EVENT";
    }
    return "?";
}

// Fixed extra ticks contributed by a bin. WAIT_FOR_EVENT is resolved
// separately because its length depends on the state of the world.
inline Tick delay_bin_ticks(DelayBin b) {
    switch (b) {
        case DelayBin::Fastest: return 0;
        case DelayBin::Plus1F: return 1;
        case DelayBin::Plus2F: return 2;
        case DelayBin::Plus4F: return 4;
        case DelayBin::Plus8F: return 8;
        case DelayBin::WaitForEvent: return 0;  // see resolve_wait_for_event
    }
    return 0;
}

// Bound on a WAIT_FOR_EVENT action (spec 8.4): a wait is never open-ended. It
// ends at the earliest of
//   * the next garbage activation,
//   * the opponent's next lock,
//   * the lock delay ceiling,
//   * an explicit maximum.
//
// `now` is the tick at which the wait starts. Times that have already passed
// or are unknown are passed as TICK_NEVER. The result is always in
// [0, max_wait].
inline Tick resolve_wait_for_event(Tick now, Tick next_garbage_activation,
                                   Tick opponent_next_lock, const HandlingModel& h,
                                   Tick max_wait = 60) {
    Tick best = now + std::min(h.lock_delay, max_wait);
    if (next_garbage_activation != TICK_NEVER && next_garbage_activation > now)
        best = std::min(best, next_garbage_activation);
    if (opponent_next_lock != TICK_NEVER && opponent_next_lock > now)
        best = std::min(best, opponent_next_lock);
    const Tick wait = best - now;
    return std::max<Tick>(0, std::min(wait, max_wait));
}

// The bins a search should actually branch on. Emitting all six for every
// placement multiplies the action space by six for little gain, so the
// default set is the ones that change outcomes: act now, nudge slightly, or
// wait for the next event.
inline std::vector<DelayBin> default_delay_bins() {
    return {DelayBin::Fastest, DelayBin::Plus2F, DelayBin::Plus8F, DelayBin::WaitForEvent};
}

}  // namespace tetra