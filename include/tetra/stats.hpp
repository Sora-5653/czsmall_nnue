// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- compact match statistics (Tetr.io-style).
#pragma once

#include "tetra/ruleset.hpp"

#include <algorithm>
#include <cstdint>

namespace tetra {

// These are intentionally derived from the simulator's authoritative counters
// rather than from wall-clock time.  That keeps runs deterministic and makes
// APM/APP comparable across CPU and GPU evaluators.
inline double match_seconds(Tick duration, const RulesetConfig& rules) {
    if (duration <= 0 || rules.tick_rate <= 0) return 0.0;
    return static_cast<double>(duration) / static_cast<double>(rules.tick_rate);
}

inline double pieces_per_second(int pieces, Tick duration, const RulesetConfig& rules) {
    const double seconds = match_seconds(duration, rules);
    return seconds > 0.0 ? static_cast<double>(pieces) / seconds : 0.0;
}

// Tetr.io-style APM: outgoing attack lines per minute.
inline double attacks_per_minute(std::int64_t lines_sent, Tick duration,
                                 const RulesetConfig& rules) {
    const double seconds = match_seconds(duration, rules);
    return seconds > 0.0 ? static_cast<double>(lines_sent) * 60.0 / seconds : 0.0;
}

// Tetr.io-style APP: outgoing attack lines per placed piece.
inline double attacks_per_piece(std::int64_t lines_sent, int pieces) {
    return pieces > 0 ? static_cast<double>(lines_sent) / static_cast<double>(pieces) : 0.0;
}

// TETR.IO VS score: ((lines sent + garbage lines cleared) / pieces) * PPS * 100.
// Algebraically this is 100 * (sent + garbage-cleared) / seconds; keeping the
// original factors here makes the metric's game meaning explicit.
inline double versus_score(std::int64_t lines_sent, std::int64_t garbage_lines_cleared,
                           int pieces, Tick duration, const RulesetConfig& rules) {
    const double pps = pieces_per_second(pieces, duration, rules);
    if (pieces <= 0 || pps <= 0.0) return 0.0;
    const double pressure_per_piece =
        static_cast<double>(lines_sent + garbage_lines_cleared) /
        static_cast<double>(pieces);
    return pressure_per_piece * pps * 100.0;
}

}  // namespace tetra
