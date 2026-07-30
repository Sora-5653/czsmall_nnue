// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- attack computation (spec 6, attack subtree).
#pragma once

#include "tetra/ruleset.hpp"
#include "tetra/rng.hpp"
#include "tetra/types.hpp"

#include <cmath>

namespace tetra {

// Mutable per-player attack state (spec 7.1: combo_state, b2b_or_charge_state).
struct AttackState {
    // -1 == no combo running. The first clear of a chain is combo 0 (which
    // carries a 1.0x multiplier), the second is combo 1 (1.25x), and so on.
    int combo = -1;
    int b2b_streak = 0;   // number of consecutive difficult clears - 1
    int surge = 0;        // charged Surge lines (B2B Charging)
    int pieces_placed = 0;

    void reset() { *this = AttackState{}; }
};

struct AttackResult {
    int lines = 0;            // total lines to send
    int surge_released = 0;   // portion of `lines` that came from a Surge break
    int base = 0;             // base weight used (diagnostics)
    int b2b_bonus = 0;
    bool b2b_continued = false;
    bool b2b_broken = false;
};

// Base attack weight for a clear, before B2B / combo / bonuses.
inline int base_attack_for(const ClearDescriptor& c, const AttackCfg& a) {
    if (c.lines <= 0) return 0;
    if (c.piece == Piece::T && c.spin != SpinType::None) {
        const bool mini = (c.spin == SpinType::Mini);
        switch (c.lines) {
            case 1: return mini ? a.tspin_mini_single : a.tspin_single;
            case 2: return mini ? a.tspin_mini_double : a.tspin_double;
            case 3: return a.tspin_triple;
            default: return a.tspin_triple;
        }
    }
    if (c.spin != SpinType::None) {  // non-T spin (All-Mini)
        switch (c.lines) {
            case 1: return a.spin_mini_single;
            case 2: return a.spin_mini_double;
            case 3: return a.spin_mini_triple;
            default: return a.quad;
        }
    }
    switch (c.lines) {
        case 1: return a.single;
        case 2: return a.doubl;
        case 3: return a.triple;
        default: return a.quad;
    }
}

// B2B bonus lines under the configured mode.
// Chaining uses osk's published step table; Charging is a flat +1.
inline int b2b_bonus_for(int streak, const AttackCfg& a) {
    if (a.b2b_mode == B2BMode::Off || streak <= 0) return 0;
    if (a.b2b_mode == B2BMode::Charging) return a.b2b_charging_bonus;
    // Chaining: level boundaries 1,3,8,24,67,185,504,1370 (osk, Discord).
    static const int bounds[] = {1, 3, 8, 24, 67, 185, 504, 1370};
    int level = 0;
    for (int i = 0; i < 8; ++i)
        if (streak >= bounds[i]) level = i + 1;
    return level;
}

// Round a rational attack value according to the ruleset's rounding mode.
// `rng` may be null when the mode is Down (which is the deterministic default).
inline int round_attack(double value, RoundingMode mode, Rng* rng) {
    if (value <= 0.0) return 0;
    const double floored = std::floor(value);
    if (mode == RoundingMode::Down || rng == nullptr) return static_cast<int>(floored);
    const double frac = value - floored;
    // RNG mode: the fractional part is the probability of rounding up.
    const int num = static_cast<int>(frac * 10000.0 + 0.5);
    return static_cast<int>(floored) + (rng->chance(num, 10000) ? 1 : 0);
}

// Full TETR.IO attack computation for one placement.
//
//   attack = round( (base + b2b_bonus) * (1 + 0.25*combo) ) + garbage_bonus
//            + all_clear + surge_release
//
// with the special case that a zero-base clear inside a combo sends
// floor(ln(1 + 1.25*combo)) instead (osk's singles formula).
inline AttackResult compute_attack(const ClearDescriptor& clear, AttackState& st,
                                   const RulesetConfig& cfg, Rng* rng) {
    const AttackCfg& a = cfg.attack;
    AttackResult r;

    if (clear.lines <= 0) {
        // No clear: the combo ends. The B2B streak is untouched (only a
        // non-difficult *clear* breaks it).
        st.combo = -1;
        return r;
    }

    const bool difficult = is_difficult_clear(clear);

    // --- back to back ---
    int prev_streak = st.b2b_streak;
    if (difficult) {
        st.b2b_streak = prev_streak + 1;
        r.b2b_continued = prev_streak > 0;
    } else {
        if (a.b2b_mode == B2BMode::Charging && prev_streak >= a.surge_start_streak) {
            // Breaking a charged streak releases the stored Surge.
            r.surge_released = st.surge;
            r.b2b_broken = true;
        } else if (prev_streak > 0) {
            r.b2b_broken = true;
        }
        st.b2b_streak = 0;
        st.surge = 0;
    }

    // The bonus applied to *this* attack uses the streak before it was
    // incremented: the first difficult clear of a chain gets no bonus.
    r.b2b_bonus = b2b_bonus_for(prev_streak, a);

    // --- combo ---
    st.combo = (st.combo < 0) ? 0 : st.combo + 1;
    int combo = st.combo;
    if (a.combo_cap >= 0 && combo > a.combo_cap) combo = a.combo_cap;

    // --- base ---
    r.base = base_attack_for(clear, a);
    const int weighted = r.base + r.b2b_bonus;

    double value;
    if (weighted == 0 && a.combo_log_for_zero_base) {
        // Zero-weight clears (singles without B2B) use the logarithmic curve.
        value = (combo >= 1) ? std::log1p(static_cast<double>(combo) * 1.25) : 0.0;
    } else {
        const double mult = 1.0 + (static_cast<double>(a.combo_multiplier_num) /
                                   static_cast<double>(a.combo_multiplier_den)) *
                                      static_cast<double>(combo);
        value = static_cast<double>(weighted) * mult;
    }

    int lines = round_attack(value, a.rounding_mode, rng);

    // --- flat bonuses (outside the combo multiplier) ---
    if (clear.cleared_garbage && (clear.lines >= 4 || clear.spin != SpinType::None))
        lines += a.garbage_clear_bonus;

    if (clear.all_clear && cfg.clear_rules.all_clear_enabled) {
        lines += a.all_clear;
        if (a.all_clear_b2b_bonus > 0) st.b2b_streak += a.all_clear_b2b_bonus;
    }

    // --- surge charging ---
    if (a.b2b_mode == B2BMode::Charging && difficult) {
        if (st.b2b_streak >= a.surge_start_streak) {
            if (st.surge == 0)
                st.surge = a.surge_base;
            else
                st.surge += 1;
        }
    }

    lines += r.surge_released;
    r.lines = lines;
    return r;
}

// Split a released Surge into its segments (spec: three segments, remainder
// carried by the first and then the second).
inline std::vector<int> split_surge(int lines, int segments) {
    std::vector<int> out;
    if (lines <= 0 || segments <= 0) return out;
    const int base = lines / segments;
    int rem = lines % segments;
    for (int i = 0; i < segments; ++i) {
        int v = base;
        if (rem > 0) { ++v; --rem; }
        if (v > 0) out.push_back(v);
    }
    return out;
}

}  // namespace tetra
