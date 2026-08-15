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

    CHECK_EQ(r.games_played, 12);
    CHECK_EQ(static_cast<int>(r.games.size()), 12);
    for (int i = 0; i < 3; ++i) {
        const size_t base = static_cast<size_t>(4 * i);
        const auto& norm0 = r.games[base];
        const auto& norm1 = r.games[base + 1];
        const auto& mirr0 = r.games[base + 2];
        const auto& mirr1 = r.games[base + 3];
        for (const ArenaGameResult* game : {&norm0, &norm1, &mirr0, &mirr1}) {
            CHECK_EQ(game->pair_index, i);
            CHECK_EQ(game->seed, norm0.seed);
        }
        CHECK(!norm0.is_mirrored);
        CHECK(!norm1.is_mirrored);
        CHECK(mirr0.is_mirrored);
        CHECK(mirr1.is_mirrored);
        CHECK(!norm0.roles_swapped);
        CHECK(norm1.roles_swapped);
        CHECK(!mirr0.roles_swapped);
        CHECK(mirr1.roles_swapped);
    }
    CHECK_EQ(r.candidate_wins, r.champion_wins);
}

TEST(arena_reports_aggregate_diagnostics_from_game_counters) {
    HeuristicEvaluator ev1;
    HeuristicEvaluator ev2;
    const RulesetConfig rules = league();
    Arena arena(ev1, ev2, quick_arena(2, 4, 15));
    const ArenaResult r = arena.evaluate(rules, 4242);

    std::int64_t candidate_sent = 0;
    std::int64_t champion_sent = 0;
    std::int64_t candidate_gc = 0;
    std::int64_t champion_gc = 0;
    std::int64_t candidate_received = 0;
    std::int64_t champion_received = 0;
    std::int64_t candidate_pieces = 0;
    std::int64_t champion_pieces = 0;
    Tick candidate_duration = 0;
    Tick champion_duration = 0;
    int candidate_survived = 0;
    int champion_survived = 0;
    for (const auto& game : r.games) {
        candidate_sent += game.candidate_sent;
        champion_sent += game.champion_sent;
        candidate_gc += game.candidate_garbage_cleared;
        champion_gc += game.champion_garbage_cleared;
        candidate_received += game.candidate_received;
        champion_received += game.champion_received;
        candidate_pieces += game.candidate_pieces;
        champion_pieces += game.champion_pieces;
        candidate_duration += game.candidate_duration;
        champion_duration += game.champion_duration;
        candidate_survived += game.candidate_survived ? 1 : 0;
        champion_survived += game.champion_survived ? 1 : 0;
    }
    const float games = static_cast<float>(r.games_played);
    CHECK(std::abs(r.candidate_avg_pieces - static_cast<float>(candidate_pieces) / games) < 1e-5f);
    CHECK(std::abs(r.champion_avg_pieces - static_cast<float>(champion_pieces) / games) < 1e-5f);
    CHECK(std::abs(r.candidate_survival_rate - static_cast<float>(candidate_survived) / games) < 1e-5f);
    CHECK(std::abs(r.champion_survival_rate - static_cast<float>(champion_survived) / games) < 1e-5f);
    CHECK(std::abs(r.candidate_sent_per_game - static_cast<float>(candidate_sent) / games) < 1e-5f);
    CHECK(std::abs(r.champion_sent_per_game - static_cast<float>(champion_sent) / games) < 1e-5f);
    CHECK(std::abs(r.candidate_garbage_cleared_per_game - static_cast<float>(candidate_gc) / games) < 1e-5f);
    CHECK(std::abs(r.champion_garbage_cleared_per_game - static_cast<float>(champion_gc) / games) < 1e-5f);
    CHECK(std::abs(r.candidate_received_per_game - static_cast<float>(candidate_received) / games) < 1e-5f);
    CHECK(std::abs(r.champion_received_per_game - static_cast<float>(champion_received) / games) < 1e-5f);
    CHECK(std::abs(
        r.candidate_survival_rate + r.candidate_blockout_rate + r.candidate_lockout_rate +
        r.candidate_garbageout_rate - 1.0f) < 1e-5f);
    CHECK(std::abs(
        r.champion_survival_rate + r.champion_blockout_rate + r.champion_lockout_rate +
        r.champion_garbageout_rate - 1.0f) < 1e-5f);
    CHECK(std::abs(r.candidate_vs - static_cast<float>(versus_score(
        candidate_sent, candidate_gc, static_cast<int>(candidate_pieces), candidate_duration, rules))) < 1e-4f);
    CHECK(std::abs(r.champion_vs - static_cast<float>(versus_score(
        champion_sent, champion_gc, static_cast<int>(champion_pieces), champion_duration, rules))) < 1e-4f);
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
