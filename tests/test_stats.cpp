// SPDX-License-Identifier: MIT
// Tetr.io-style display metrics derived from deterministic simulator counters.
#include "test_util.hpp"
#include "tetra/stats.hpp"

using namespace tetra;

TEST(match_stats_use_simulator_time) {
    const RulesetConfig rules = RulesetConfig::tetra_league();
    const Tick duration = static_cast<Tick>(rules.tick_rate * 2);
    CHECK_EQ(match_seconds(duration, rules), 2.0);
    CHECK_EQ(pieces_per_second(10, duration, rules), 5.0);
    CHECK_EQ(attacks_per_minute(6, duration, rules), 180.0);
    CHECK_EQ(attacks_per_piece(6, 20), 0.3);
    // ((6 sent + 2 garbage cleared) / 10 pieces) * 5 PPS * 100 = 400.
    CHECK_EQ(versus_score(6, 2, 10, duration, rules), 400.0);
}

TEST(match_stats_are_zero_without_time_or_pieces) {
    const RulesetConfig rules = RulesetConfig::tetra_league();
    CHECK_EQ(match_seconds(0, rules), 0.0);
    CHECK_EQ(pieces_per_second(10, 0, rules), 0.0);
    CHECK_EQ(attacks_per_minute(6, 0, rules), 0.0);
    CHECK_EQ(attacks_per_piece(6, 0), 0.0);
    CHECK_EQ(versus_score(6, 2, 10, 0, rules), 0.0);
    CHECK_EQ(versus_score(6, 2, 0, rules.tick_rate, rules), 0.0);
}
