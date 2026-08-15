// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- Candidate Gating and the Arena (spec 20).
//
// Evaluates a Candidate network against a Champion network using paired,
// two-board games with mirrored boards and identical seeds. To eliminate
// left-right bias and queue luck, every trial is evaluated twice:
//   1. Normal board, normal piece sequence (seed S)
//   2. Mirrored board, mirrored piece sequence (seed S, J <-> L, S <-> Z)
//
// Promotion criteria follow spec 20:
//   * Direct match win rate against Champion >= promotion_threshold (default 55%)
//   * Wilson score 95% confidence interval reported for reliability
#pragma once

#include "tetra/evaluator.hpp"
#include "tetra/movegen.hpp"
#include "tetra/player.hpp"
#include "tetra/rng.hpp"
#include "tetra/ruleset.hpp"
#include "tetra/search.hpp"
#include "tetra/selfplay.hpp"
#include "tetra/stats.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace tetra {

struct ArenaConfig {
    int pairs = 10;                     // Factorial blocks (4 * pairs total games played)
    int max_pieces = 300;               // Placements before truncation
    SearchConfig search{};              // Shared search settings and default budget.
    // Diagnostic overrides. Negative values keep the shared budget; zero is
    // policy-only (Searcher's zero-simulation fallback), which lets Arena
    // measure whether search actually improves a fixed network.
    int candidate_simulations = -1;
    int champion_simulations = -1;
    // -1 inherits search.use_gumbel; 0/1 override per side for diagnostics.
    int candidate_gumbel = -1;
    int champion_gumbel = -1;
    // Negative values inherit search.gumbel_noise_scale.  Per-side overrides
    // let Arena calibrate exploration on one fixed checkpoint without changing
    // any network weights or the opponent's search distribution.
    float candidate_gumbel_noise_scale = -1.0f;
    float champion_gumbel_noise_scale = -1.0f;
    // -1 inherits search.enable_timing_actions; 0/1 override per side. This
    // permits causal timing experiments without changing the default Arena
    // action contract used by historical checkpoints.
    int candidate_timing_actions = -1;
    int champion_timing_actions = -1;
    GarbageStyle garbage_style = GarbageStyle::Steady;
    int garbage_period = 8;
    int garbage_lines = 2;
    float promotion_threshold = 0.55f;  // Spec 20: 55% direct match threshold
};

struct ArenaGameResult {
    int pair_index = 0;
    bool is_mirrored = false;
    bool roles_swapped = false;
    std::uint64_t seed = 0;
    int candidate_pieces = 0;
    int champion_pieces = 0;
    std::int64_t candidate_cleared = 0;
    std::int64_t champion_cleared = 0;
    std::int64_t candidate_garbage_cleared = 0;
    std::int64_t champion_garbage_cleared = 0;
    std::int64_t candidate_sent = 0;
    std::int64_t champion_sent = 0;
    std::int64_t candidate_received = 0;
    std::int64_t champion_received = 0;
    Tick candidate_duration = 0;
    Tick champion_duration = 0;
    bool candidate_survived = false;
    bool champion_survived = false;
    TopoutReason candidate_topout = TopoutReason::None;
    TopoutReason champion_topout = TopoutReason::None;
    float candidate_score = 0.0f;       // 1.0 = win, 0.5 = draw, 0.0 = loss
};

struct ArenaResult {
    int games_played = 0;
    int candidate_wins = 0;
    int champion_wins = 0;
    int draws = 0;
    float win_rate = 0.0f;              // (candidate_wins + 0.5 * draws) / games_played
    float ci_lower = 0.0f;              // Wilson score 95% CI lower bound
    float ci_upper = 0.0f;              // Wilson score 95% CI upper bound
    float candidate_vs = 0.0f;          // aggregate TETR.IO VS score
    float champion_vs = 0.0f;           // aggregate TETR.IO VS score
    float candidate_apm = 0.0f;
    float champion_apm = 0.0f;
    float candidate_app = 0.0f;
    float champion_app = 0.0f;
    float candidate_pps = 0.0f;
    float champion_pps = 0.0f;
    float candidate_avg_pieces = 0.0f;
    float champion_avg_pieces = 0.0f;
    float candidate_avg_seconds = 0.0f;
    float champion_avg_seconds = 0.0f;
    float candidate_survival_rate = 0.0f;
    float champion_survival_rate = 0.0f;
    float candidate_sent_per_game = 0.0f;
    float champion_sent_per_game = 0.0f;
    float candidate_garbage_cleared_per_game = 0.0f;
    float champion_garbage_cleared_per_game = 0.0f;
    float candidate_received_per_game = 0.0f;
    float champion_received_per_game = 0.0f;
    float candidate_blockout_rate = 0.0f;
    float champion_blockout_rate = 0.0f;
    float candidate_lockout_rate = 0.0f;
    float champion_lockout_rate = 0.0f;
    float candidate_garbageout_rate = 0.0f;
    float champion_garbageout_rate = 0.0f;
    bool promoted = false;              // win_rate >= promotion_threshold
    std::vector<ArenaGameResult> games;
};

class Arena {
public:
    Arena(Evaluator& candidate, Evaluator& champion, const ArenaConfig& cfg)
        : candidate_(candidate), champion_(champion), cfg_(cfg) {}

    ArenaResult evaluate(const RulesetConfig& rules, std::uint64_t base_seed = 42) {
        ArenaResult res;
        for (int i = 0; i < cfg_.pairs; ++i) {
            const std::uint64_t pair_seed =
                base_seed + static_cast<std::uint64_t>(i) * 0x9E3779B97F4A7C15ull;

            // Four-game factorial block: normal/mirrored geometry crossed with
            // player-role assignment.  Role swapping alone cancels queue/index
            // and tie-break advantages; the mirrored pair separately averages
            // geometry asymmetry (notably TETR.IO 180-kick asymmetry).
            res.games.push_back(play_game(rules, pair_seed, i,
                                          /*mirror=*/false, /*swap_roles=*/false));
            res.games.push_back(play_game(rules, pair_seed, i,
                                          /*mirror=*/false, /*swap_roles=*/true));
            res.games.push_back(play_game(rules, pair_seed, i,
                                          /*mirror=*/true, /*swap_roles=*/false));
            res.games.push_back(play_game(rules, pair_seed, i,
                                          /*mirror=*/true, /*swap_roles=*/true));
        }

        res.games_played = static_cast<int>(res.games.size());
        for (const auto& g : res.games) {
            if (g.candidate_score > 0.75f) {
                res.candidate_wins++;
            } else if (g.candidate_score < 0.25f) {
                res.champion_wins++;
            } else {
                res.draws++;
            }
        }

        if (res.games_played > 0) {
            std::int64_t candidate_pressure = 0;
            std::int64_t champion_pressure = 0;
            std::int64_t candidate_sent = 0;
            std::int64_t champion_sent = 0;
            std::int64_t candidate_garbage_cleared = 0;
            std::int64_t champion_garbage_cleared = 0;
            std::int64_t candidate_received = 0;
            std::int64_t champion_received = 0;
            std::int64_t candidate_pieces = 0;
            std::int64_t champion_pieces = 0;
            int candidate_survived = 0;
            int champion_survived = 0;
            int candidate_blockout = 0;
            int champion_blockout = 0;
            int candidate_lockout = 0;
            int champion_lockout = 0;
            int candidate_garbageout = 0;
            int champion_garbageout = 0;
            Tick candidate_duration = 0;
            Tick champion_duration = 0;
            for (const auto& g : res.games) {
                candidate_sent += g.candidate_sent;
                champion_sent += g.champion_sent;
                candidate_garbage_cleared += g.candidate_garbage_cleared;
                champion_garbage_cleared += g.champion_garbage_cleared;
                candidate_pressure += g.candidate_sent + g.candidate_garbage_cleared;
                champion_pressure += g.champion_sent + g.champion_garbage_cleared;
                candidate_received += g.candidate_received;
                champion_received += g.champion_received;
                candidate_pieces += g.candidate_pieces;
                champion_pieces += g.champion_pieces;
                candidate_survived += g.candidate_survived ? 1 : 0;
                champion_survived += g.champion_survived ? 1 : 0;
                candidate_blockout += g.candidate_topout == TopoutReason::BlockOut ? 1 : 0;
                champion_blockout += g.champion_topout == TopoutReason::BlockOut ? 1 : 0;
                candidate_lockout += g.candidate_topout == TopoutReason::LockOut ? 1 : 0;
                champion_lockout += g.champion_topout == TopoutReason::LockOut ? 1 : 0;
                candidate_garbageout += g.candidate_topout == TopoutReason::GarbageOut ? 1 : 0;
                champion_garbageout += g.champion_topout == TopoutReason::GarbageOut ? 1 : 0;
                candidate_duration += g.candidate_duration;
                champion_duration += g.champion_duration;
            }
            const double tick_rate = static_cast<double>(std::max(1, rules.tick_rate));
            const double candidate_seconds =
                candidate_duration > 0 ? static_cast<double>(candidate_duration) / tick_rate : 0.0;
            const double champion_seconds =
                champion_duration > 0 ? static_cast<double>(champion_duration) / tick_rate : 0.0;
            res.candidate_vs = candidate_seconds > 0.0
                ? static_cast<float>(100.0 * static_cast<double>(candidate_pressure) /
                                     candidate_seconds)
                : 0.0f;
            res.champion_vs = champion_seconds > 0.0
                ? static_cast<float>(100.0 * static_cast<double>(champion_pressure) /
                                     champion_seconds)
                : 0.0f;
            res.candidate_apm = candidate_seconds > 0.0
                ? static_cast<float>(60.0 * static_cast<double>(candidate_sent) /
                                     candidate_seconds)
                : 0.0f;
            res.champion_apm = champion_seconds > 0.0
                ? static_cast<float>(60.0 * static_cast<double>(champion_sent) /
                                     champion_seconds)
                : 0.0f;
            res.candidate_app = candidate_pieces > 0
                ? static_cast<float>(static_cast<double>(candidate_sent) /
                                     static_cast<double>(candidate_pieces))
                : 0.0f;
            res.champion_app = champion_pieces > 0
                ? static_cast<float>(static_cast<double>(champion_sent) /
                                     static_cast<double>(champion_pieces))
                : 0.0f;
            res.candidate_pps = candidate_seconds > 0.0
                ? static_cast<float>(static_cast<double>(candidate_pieces) /
                                     candidate_seconds)
                : 0.0f;
            res.champion_pps = champion_seconds > 0.0
                ? static_cast<float>(static_cast<double>(champion_pieces) /
                                     champion_seconds)
                : 0.0f;
            const float games = static_cast<float>(res.games_played);
            res.candidate_avg_pieces = static_cast<float>(candidate_pieces) / games;
            res.champion_avg_pieces = static_cast<float>(champion_pieces) / games;
            res.candidate_avg_seconds = static_cast<float>(candidate_seconds / games);
            res.champion_avg_seconds = static_cast<float>(champion_seconds / games);
            res.candidate_survival_rate = static_cast<float>(candidate_survived) / games;
            res.champion_survival_rate = static_cast<float>(champion_survived) / games;
            res.candidate_sent_per_game = static_cast<float>(candidate_sent) / games;
            res.champion_sent_per_game = static_cast<float>(champion_sent) / games;
            res.candidate_garbage_cleared_per_game =
                static_cast<float>(candidate_garbage_cleared) / games;
            res.champion_garbage_cleared_per_game =
                static_cast<float>(champion_garbage_cleared) / games;
            res.candidate_received_per_game = static_cast<float>(candidate_received) / games;
            res.champion_received_per_game = static_cast<float>(champion_received) / games;
            res.candidate_blockout_rate = static_cast<float>(candidate_blockout) / games;
            res.champion_blockout_rate = static_cast<float>(champion_blockout) / games;
            res.candidate_lockout_rate = static_cast<float>(candidate_lockout) / games;
            res.champion_lockout_rate = static_cast<float>(champion_lockout) / games;
            res.candidate_garbageout_rate = static_cast<float>(candidate_garbageout) / games;
            res.champion_garbageout_rate = static_cast<float>(champion_garbageout) / games;

            const float cw = static_cast<float>(res.candidate_wins);
            const float dw = static_cast<float>(res.draws);
            const float gp = static_cast<float>(res.games_played);
            res.win_rate = (cw + 0.5f * dw) / gp;
            compute_wilson_ci(res.win_rate, res.games_played, &res.ci_lower, &res.ci_upper);
            // A point estimate above 55% is not enough to call a noisy trial
            // an improvement.  Require the Wilson lower bound to clear 50%
            // as well, so promotion means the candidate is supported as
            // stronger than the champion rather than merely lucky.
            res.promoted = (res.win_rate >= cfg_.promotion_threshold &&
                            res.ci_lower > 0.5f);
        }
        return res;
    }

    const ArenaConfig& config() const { return cfg_; }

private:
    ArenaGameResult play_game(const RulesetConfig& rules, std::uint64_t seed,
                              int pair_idx, bool mirror, bool swap_roles) {
        ArenaGameResult g;
        g.pair_index = pair_idx;
        g.is_mirrored = mirror;
        g.roles_swapped = swap_roles;
        g.seed = seed;

        // SelfPlayWorker already defines the game's transition semantics:
        // both boards advance in timestamp order, attacks are delivered to
        // the other board, and the same model/search contract is used on each
        // turn.  Arena must use that same interaction model; running the two
        // evaluators independently would train a two-board outcome while
        // evaluating a single-board survival proxy.
        const int candidate_index = swap_roles ? 1 : 0;
        const int champion_index = swap_roles ? 0 : 1;
        Player candidate_player;
        candidate_player.reset(rules, seed, candidate_index);
        candidate_player.set_mirror(mirror);
        Player champion_player;
        champion_player.reset(rules, seed, champion_index);
        champion_player.set_mirror(mirror);

        MoveGenerator gen;
        SearchConfig candidate_sc = cfg_.search;
        SearchConfig champion_sc = cfg_.search;
        if (cfg_.candidate_simulations >= 0)
            candidate_sc.simulations = cfg_.candidate_simulations;
        if (cfg_.champion_simulations >= 0)
            champion_sc.simulations = cfg_.champion_simulations;
        if (cfg_.candidate_gumbel >= 0)
            candidate_sc.use_gumbel = cfg_.candidate_gumbel != 0;
        if (cfg_.champion_gumbel >= 0)
            champion_sc.use_gumbel = cfg_.champion_gumbel != 0;
        if (cfg_.candidate_gumbel_noise_scale >= 0.0f)
            candidate_sc.gumbel_noise_scale = cfg_.candidate_gumbel_noise_scale;
        if (cfg_.champion_gumbel_noise_scale >= 0.0f)
            champion_sc.gumbel_noise_scale = cfg_.champion_gumbel_noise_scale;
        if (cfg_.candidate_timing_actions >= 0)
            candidate_sc.enable_timing_actions = cfg_.candidate_timing_actions != 0;
        if (cfg_.champion_timing_actions >= 0)
            champion_sc.enable_timing_actions = cfg_.champion_timing_actions != 0;
        Searcher candidate_search(candidate_, candidate_sc);
        Searcher champion_search(champion_, champion_sc);
        int candidate_pieces = 0;
        int champion_pieces = 0;

        while (candidate_player.alive() && champion_player.alive() &&
               candidate_pieces < cfg_.max_pieces && champion_pieces < cfg_.max_pieces) {
            const bool candidate_turn =
                candidate_player.now() < champion_player.now() ||
                (candidate_player.now() == champion_player.now() &&
                 candidate_player.index() < champion_player.index());
            Player& active = candidate_turn ? candidate_player : champion_player;
            Player& inactive = candidate_turn ? champion_player : candidate_player;
            int& active_pieces = candidate_turn ? candidate_pieces : champion_pieces;
            Searcher& searcher = candidate_turn ? candidate_search : champion_search;
            SearchConfig& sc = candidate_turn ? candidate_sc : champion_sc;

            auto actions = gen.generate(
                active.board(), active.active().type, active.hold(),
                active.visible_next().empty() ? Piece::None : active.visible_next()[0],
                rules);
            const Tick next_activation = active.garbage().next_activation(active.now());
            if (sc.enable_timing_actions && next_activation != TICK_NEVER &&
                !actions.empty()) {
                const std::vector<DelayBin> timing_bins{
                    DelayBin::Fastest, DelayBin::WaitForEvent
                };
                actions = MoveGenerator::expand_delay_bins(
                    actions, rules, active.now(), next_activation, TICK_NEVER,
                    timing_bins);
            }
            if (actions.empty()) {
                active.die(TopoutReason::BlockOut);
                break;
            }

            sc.seed = seed * 0x9E3779B97F4A7C15ull +
                      static_cast<std::uint64_t>(candidate_pieces + champion_pieces);
            searcher.set_config(sc);
            Evaluator& opponent_evaluator = candidate_turn ? champion_ : candidate_;
            const SearchResult r = searcher.search(
                active, &inactive, &opponent_evaluator,
                /*deliver_attacks=*/cfg_.garbage_style != GarbageStyle::None);
            if (r.best_action < 0 || r.best_action >= static_cast<int>(actions.size())) {
                active.die(TopoutReason::BlockOut);
                break;
            }

            const PlacementAction& chosen = actions[static_cast<size_t>(r.best_action)];
            if (chosen.use_hold && !active.do_hold()) {
                active.die(TopoutReason::BlockOut);
                break;
            }
            active.set_active(chosen.piece_state());
            int sent = 0;
            const LockResult lr = active.lock_piece(chosen.total_duration(), &sent);
            if (!lr.ok && !lr.topped_out) {
                active.die(TopoutReason::BlockOut);
                break;
            }
            if (lr.topped_out) break;

            if (sent > 0 && cfg_.garbage_style != GarbageStyle::None)
                inactive.receive_attack(sent, active.now(), active.index());
            ++active_pieces;
        }

        g.candidate_pieces = candidate_pieces;
        g.champion_pieces = champion_pieces;
        g.candidate_cleared = candidate_player.lines_cleared();
        g.champion_cleared = champion_player.lines_cleared();
        g.candidate_garbage_cleared = candidate_player.garbage_lines_cleared();
        g.champion_garbage_cleared = champion_player.garbage_lines_cleared();
        g.candidate_sent = candidate_player.lines_sent();
        g.champion_sent = champion_player.lines_sent();
        g.candidate_received = candidate_player.lines_received();
        g.champion_received = champion_player.lines_received();
        g.candidate_duration = candidate_player.now();
        g.champion_duration = champion_player.now();
        g.candidate_survived = candidate_player.alive();
        g.champion_survived = champion_player.alive();
        g.candidate_topout = candidate_player.topout_reason();
        g.champion_topout = champion_player.topout_reason();

        float score = 0.5f;
        if (g.candidate_survived && !g.champion_survived) {
            score = 1.0f;
        } else if (!g.candidate_survived && g.champion_survived) {
            score = 0.0f;
        } else if (!g.candidate_survived && !g.champion_survived) {
            if (g.candidate_pieces > g.champion_pieces) score = 1.0f;
            else if (g.candidate_pieces < g.champion_pieces) score = 0.0f;
            else {
                if (g.candidate_sent > g.champion_sent) score = 1.0f;
                else if (g.candidate_sent < g.champion_sent) score = 0.0f;
            }
        } else {
            // Both survived to the truncation boundary: compare attack lines
            // first, then cleared lines as a deterministic tie-break.
            if (g.candidate_sent > g.champion_sent) score = 1.0f;
            else if (g.candidate_sent < g.champion_sent) score = 0.0f;
            else if (g.candidate_cleared > g.champion_cleared) score = 1.0f;
            else if (g.candidate_cleared < g.champion_cleared) score = 0.0f;
        }
        g.candidate_score = score;
        return g;
    }

    // Wilson score interval for binomial proportion (95% CI)
    static void compute_wilson_ci(float p, int n, float* lower, float* upper) {
        if (n <= 0) {
            *lower = 0.0f;
            *upper = 1.0f;
            return;
        }
        const float z = 1.95996f; // 95% confidence
        const float z2 = z * z;
        const float fn = static_cast<float>(n);
        const float denom = 1.0f + z2 / fn;
        const float center = (p + z2 / (2.0f * fn)) / denom;
        const float spread =
            z * std::sqrt((p * (1.0f - p) + z2 / (4.0f * fn)) / fn) / denom;
        *lower = std::max(0.0f, center - spread);
        *upper = std::min(1.0f, center + spread);
    }

    Evaluator& candidate_;
    Evaluator& champion_;
    ArenaConfig cfg_;
};

}  // namespace tetra
