// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- self-play worker (spec 13.3, 13.5, 17 /selfplay).
//
// Plays games with the search and emits training samples. This is the loop the
// trainer consumes, and the last piece of scaffolding needed before a network
// can be attached: with it, "swap in a TetraFormer" is an Evaluator change and
// nothing else has to move.
//
// Garbage handling follows spec 13.3: rather than requiring a second player,
// the opponent is modelled as a stochastic attack process. That is deliberate
// -- it lets the single-board curriculum run before M4 exists, and it exercises
// the pending-garbage and timing state the bot must learn to reason about.
#pragma once

#include "tetra/replay_buffer.hpp"
#include "tetra/search.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace tetra {

struct SelfPlayConfig {
    int max_pieces = 300;
    SearchConfig search{};
    GarbageStyle garbage_style = GarbageStyle::Steady;
    int garbage_period = 8;   // placements between attacks
    int garbage_lines = 2;
    std::uint32_t model_version = 0;

    // A game that survives to `max_pieces` is scored as a draw rather than a
    // win: the bot did not actually beat anything, and labelling survival as a
    // win would teach it that stalling is optimal.
    bool truncation_is_draw = true;
};

struct SelfPlayStats {
    int pieces = 0;
    int lines_cleared = 0;
    int lines_sent = 0;
    int lines_received = 0;
    bool survived = true;
    TopoutReason topout = TopoutReason::None;
    float outcome = 0.0f;
    Tick duration = 0;
};

// Plays one game and returns its samples.
class SelfPlayWorker {
public:
    SelfPlayWorker(Evaluator& evaluator, const SelfPlayConfig& cfg)
        : eval_(evaluator), cfg_(cfg) {}

    std::vector<TrainingSample> play(const RulesetConfig& rules, std::uint64_t seed,
                                     SelfPlayStats* stats_out = nullptr) {
        Player p;
        p.reset(rules, seed, 0);

        SearchConfig sc = cfg_.search;
        Searcher searcher(eval_, sc);
        GameRecorder recorder(
            cfg_.model_version, seed, static_cast<std::uint8_t>(cfg_.garbage_style),
            static_cast<std::uint8_t>(cfg_.garbage_period),
            static_cast<std::uint8_t>(cfg_.garbage_lines));
        Tokenizer tokenizer;
        Rng garbage_rng(seed ^ 0x5EEDFACEull);

        SelfPlayStats stats;
        int i = 0;
        for (; i < cfg_.max_pieces && p.alive(); ++i) {
            maybe_send_garbage(p, i, garbage_rng);

            const auto actions =
                movegen_.generate(p.board(), p.active().type, p.hold(),
                                  p.visible_next().empty() ? Piece::None : p.visible_next()[0],
                                  rules);
            if (actions.empty()) break;

            // Vary the search seed per move so determinizations differ, while
            // staying a deterministic function of (game seed, move number).
            sc.seed = seed * 0x9E3779B97F4A7C15ull + static_cast<std::uint64_t>(i);
            searcher.set_config(sc);

            const Observation obs = observe(p);
            const SearchResult r = searcher.search(p);
            if (r.best_action < 0 ||
                r.best_action >= static_cast<int>(actions.size()))
                break;

            recorder.add(obs, actions, r, tokenizer);

            const PlacementAction& chosen = actions[static_cast<size_t>(r.best_action)];
            if (chosen.use_hold && !p.do_hold()) {
                // Hold was refused; the sample is still valid but the move is
                // not playable, so stop rather than substitute a different one.
                break;
            }
            p.set_active(chosen.piece_state());
            int sent = 0;
            const LockResult lr = p.lock_piece(chosen.total_duration(), &sent);
            recorder.note_outcome_of_last(sent, lr.garbage_received);
            if (!lr.ok && !lr.topped_out) break;
            if (lr.topped_out) break;
        }

        stats.pieces = i;
        stats.lines_cleared = static_cast<int>(p.lines_cleared());
        stats.lines_sent = static_cast<int>(p.lines_sent());
        stats.lines_received = static_cast<int>(p.lines_received());
        stats.survived = p.alive();
        stats.topout = p.topout_reason();
        stats.duration = p.now();

        // Spec 12.3: the reward is the game result and nothing else. Garbage
        // and attack are auxiliary targets, never added to it.
        float z;
        if (!p.alive()) z = -1.0f;
        else if (cfg_.truncation_is_draw) z = 0.0f;
        else z = 1.0f;
        stats.outcome = z;

        if (stats_out) *stats_out = stats;
        return recorder.finalize(z);
    }

    const SelfPlayConfig& config() const { return cfg_; }
    void set_config(const SelfPlayConfig& c) { cfg_ = c; }

private:
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
                // Poisson-ish: a burst with small probability each placement.
                if (rng.chance(1, std::max(2, cfg_.garbage_period)))
                    p.receive_attack(
                        cfg_.garbage_lines + static_cast<int>(rng.below(4)), p.now(), 1);
                break;
            case GarbageStyle::None:
                break;
        }
    }

    Evaluator& eval_;
    SelfPlayConfig cfg_;
    MoveGenerator movegen_;
};

}  // namespace tetra