// SPDX-License-Identifier: MIT
// Candidate gating and the Arena (spec 20).
#include "test_util.hpp"
#include "tetra/arena.hpp"

using namespace tetra;

namespace {

RulesetConfig league() { return RulesetConfig::tetra_league(); }

ArenaConfig quick_arena(int pairs = 2, int sims = 4, int pieces = 20) {
    ArenaConfig cfg;
    cfg.pairs = pairs;
    cfg.search.simulations = sims;
    cfg.search.max_depth = 3;
    cfg.max_pieces = pieces;
    cfg.garbage_style = GarbageStyle::None;
    return cfg;
}

}  // namespace

TEST(arena_paired_games_run_normal_and_mirrored) {
    HeuristicEvaluator ev1;
    HeuristicEvaluator ev2;
    Arena arena(ev1, ev2, quick_arena(/*pairs=*/3, /*sims=*/4, /*pieces=*/15));
    const ArenaResult r = arena.evaluate(league(), 123);

    CHECK_EQ(r.games_played, 6);
    CHECK_EQ(static_cast<int>(r.games.size()), 6);
    for (int i = 0; i < 3; ++i) {
        const auto& norm = r.games[static_cast<size_t>(2 * i)];
        const auto& mirr = r.games[static_cast<size_t>(2 * i + 1)];
        CHECK_EQ(norm.pair_index, i);
        CHECK_EQ(mirr.pair_index, i);
        CHECK(!norm.is_mirrored);
        CHECK(mirr.is_mirrored);
        CHECK_EQ(norm.seed, mirr.seed);
    }
}

TEST(arena_identifies_win_draw_loss) {
    HeuristicEvaluator ev1;
    HeuristicEvaluator ev2;
    Arena arena(ev1, ev2, quick_arena(2, 4, 15));
    const ArenaResult r = arena.evaluate(league(), 42);

    CHECK(r.win_rate >= 0.0f);
    CHECK(r.win_rate <= 1.0f);
    CHECK_EQ(r.candidate_wins + r.champion_wins + r.draws, r.games_played);
}

TEST(arena_wilson_ci_bounds) {
    HeuristicEvaluator ev1;
    HeuristicEvaluator ev2;
    Arena arena(ev1, ev2, quick_arena(3, 4, 15));
    const ArenaResult r = arena.evaluate(league(), 99);

    CHECK(r.ci_lower >= 0.0f);
    CHECK(r.ci_lower <= r.win_rate + 1e-5f);
    CHECK(r.win_rate <= r.ci_upper + 1e-5f);
    CHECK(r.ci_upper <= 1.0f);
}

TEST(arena_promotion_threshold_check) {
    HeuristicEvaluator ev1;
    HeuristicEvaluator ev2;
    ArenaConfig cfg = quick_arena(1, 2, 10);
    cfg.promotion_threshold = 0.55f;
    Arena arena(ev1, ev2, cfg);
    const ArenaResult r = arena.evaluate(league(), 1);
    CHECK(r.promoted == (r.win_rate >= cfg.promotion_threshold && r.ci_lower > 0.5f));
}

TEST(arena_is_deterministic) {
    HeuristicEvaluator ev1;
    HeuristicEvaluator ev2;
    Arena arena(ev1, ev2, quick_arena(2, 4, 15));
    const ArenaResult a = arena.evaluate(league(), 777);
    const ArenaResult b = arena.evaluate(league(), 777);

    CHECK_EQ(a.games_played, b.games_played);
    CHECK_EQ(a.candidate_wins, b.candidate_wins);
    CHECK_EQ(a.champion_wins, b.champion_wins);
    CHECK_EQ(a.draws, b.draws);
    CHECK_EQ(a.win_rate, b.win_rate);
    for (size_t i = 0; i < a.games.size(); ++i) {
        CHECK_EQ(a.games[i].candidate_score, b.games[i].candidate_score);
        CHECK_EQ(a.games[i].candidate_pieces, b.games[i].candidate_pieces);
        CHECK_EQ(a.games[i].champion_pieces, b.games[i].champion_pieces);
    }
}
