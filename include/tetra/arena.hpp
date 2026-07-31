// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- Candidate Gating and the Arena (spec 20).
//
// Evaluates a Candidate network against a Champion network using paired games
// with mirrored boards and identical piece sequences. To eliminate left-right
// bias and queue luck, every trial is evaluated twice:
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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace tetra {

struct ArenaConfig {
    int pairs = 10;                     // Number of paired games (2 * pairs total games played)
    int max_pieces = 300;               // Placements before truncation
    SearchConfig search{};              // Equal search budget for Candidate and Champion
    GarbageStyle garbage_style = GarbageStyle::Steady;
    int garbage_period = 8;
    int garbage_lines = 2;
    float promotion_threshold = 0.55f;  // Spec 20: 55% direct match threshold
};

struct ArenaGameResult {
    int pair_index = 0;
    bool is_mirrored = false;
    std::uint64_t seed = 0;
    int candidate_pieces = 0;
    int champion_pieces = 0;
    std::int64_t candidate_cleared = 0;
    std::int64_t champion_cleared = 0;
    std::int64_t candidate_sent = 0;
    std::int64_t champion_sent = 0;
    bool candidate_survived = false;
    bool champion_survived = false;
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

            // Game A: normal pair
            res.games.push_back(play_game(rules, pair_seed, i, /*mirror=*/false));
            // Game B: mirrored pair (same seed, mirrored pieces and hole positions)
            res.games.push_back(play_game(rules, pair_seed, i, /*mirror=*/true));
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
            const float cw = static_cast<float>(res.candidate_wins);
            const float dw = static_cast<float>(res.draws);
            const float gp = static_cast<float>(res.games_played);
            res.win_rate = (cw + 0.5f * dw) / gp;
            compute_wilson_ci(res.win_rate, res.games_played, &res.ci_lower, &res.ci_upper);
            res.promoted = (res.win_rate >= cfg_.promotion_threshold);
        }
        return res;
    }

    const ArenaConfig& config() const { return cfg_; }

private:
    struct SideStats {
        int pieces = 0;
        std::int64_t cleared = 0;
        std::int64_t sent = 0;
        bool survived = false;
    };

    void maybe_send_garbage(Player& p, int move, Rng& rng) {
        if (cfg_.garbage_style == GarbageStyle::None || move == 0) return;
        switch (cfg_.garbage_style) {
            case GarbageStyle::Steady:
                if (move % cfg_.garbage_period == 0)
                    p.receive_attack(cfg_.garbage_lines, p.now(), 1);
                break;
            case GarbageStyle::FastSmall:
                if (move % std::max(1, cfg_.garbage_period / 3) == 0)
                    p.receive_attack(1, p.now(), 1);
                break;
            case GarbageStyle::SlowLarge:
                if (move % (cfg_.garbage_period * 3) == 0)
                    p.receive_attack(cfg_.garbage_lines * 3, p.now(), 1);
                break;
            case GarbageStyle::Burst:
                if (rng.chance(1, std::max(2, cfg_.garbage_period)))
                    p.receive_attack(
                        cfg_.garbage_lines + static_cast<int>(rng.below(4)), p.now(), 1);
                break;
            case GarbageStyle::None:
                break;
        }
    }

    SideStats run_one_side(Evaluator& ev, const RulesetConfig& rules,
                           std::uint64_t seed, bool mirror) {
        Player p;
        p.reset(rules, seed, 0);
        p.set_mirror(mirror);

        MoveGenerator gen;
        SearchConfig sc = cfg_.search;
        Searcher searcher(ev, sc);
        Rng garbage_rng(seed ^ 0x5EEDFACEull);

        int i = 0;
        for (; i < cfg_.max_pieces && p.alive(); ++i) {
            maybe_send_garbage(p, i, garbage_rng);

            const auto actions = gen.generate(
                p.board(), p.active().type, p.hold(),
                p.visible_next().empty() ? Piece::None : p.visible_next()[0],
                rules);
            if (actions.empty()) break;

            sc.seed = seed * 0x9E3779B97F4A7C15ull + static_cast<std::uint64_t>(i);
            searcher.set_config(sc);

            const SearchResult r = searcher.search(p);
            if (r.best_action < 0 || r.best_action >= static_cast<int>(actions.size()))
                break;

            const PlacementAction& chosen = actions[static_cast<size_t>(r.best_action)];
            if (chosen.use_hold && !p.do_hold()) break;
            p.set_active(chosen.piece_state());

            int sent = 0;
            const LockResult lr = p.lock_piece(chosen.total_duration(), &sent);
            if (!lr.ok && !lr.topped_out) break;
            if (lr.topped_out) break;
        }

        SideStats st;
        st.pieces = i;
        st.cleared = p.lines_cleared();
        st.sent = p.lines_sent();
        st.survived = p.alive();
        return st;
    }

    ArenaGameResult play_game(const RulesetConfig& rules, std::uint64_t seed,
                              int pair_idx, bool mirror) {
        ArenaGameResult g;
        g.pair_index = pair_idx;
        g.is_mirrored = mirror;
        g.seed = seed;

        const SideStats c = run_one_side(candidate_, rules, seed, mirror);
        const SideStats h = run_one_side(champion_, rules, seed, mirror);

        g.candidate_pieces = c.pieces;
        g.champion_pieces = h.pieces;
        g.candidate_cleared = c.cleared;
        g.champion_cleared = h.cleared;
        g.candidate_sent = c.sent;
        g.champion_sent = h.sent;
        g.candidate_survived = c.survived;
        g.champion_survived = h.survived;

        float score = 0.5f;
        if (c.survived && !h.survived) {
            score = 1.0f;
        } else if (!c.survived && h.survived) {
            score = 0.0f;
        } else if (!c.survived && !h.survived) {
            if (c.pieces > h.pieces) score = 1.0f;
            else if (c.pieces < h.pieces) score = 0.0f;
            else {
                if (c.sent > h.sent) score = 1.0f;
                else if (c.sent < h.sent) score = 0.0f;
            }
        } else {
            // Both survived: compare attack lines sent
            if (c.sent > h.sent) score = 1.0f;
            else if (c.sent < h.sent) score = 0.0f;
            else if (c.cleared > h.cleared) score = 1.0f;
            else if (c.cleared < h.cleared) score = 0.0f;
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
