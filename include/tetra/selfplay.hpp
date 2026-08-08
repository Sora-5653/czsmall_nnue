// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- self-play worker (spec 13.3, 13.5, 17 /selfplay).
//
// Plays games with the search and emits training samples. This is the loop the
// trainer consumes, and the last piece of scaffolding needed before a network
// can be attached: with it, "swap in a TetraFormer" is an Evaluator change and
// nothing else has to move.
//
// Self-play is a real two-board event-driven game. The board whose clock is
// earlier acts, outgoing attacks are delivered to the other board, and the
// same pair of states is handed to the search and the observation layer. The
// garbage style flag remains as the explicit switch for no-attack curriculum
// runs and as replay provenance; it is no longer a hidden third opponent.
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
    TerminationReason termination_reason = TerminationReason::Unknown;
};

// Plays one game and returns its samples.
class SelfPlayWorker {
public:
    SelfPlayWorker(Evaluator& evaluator, const SelfPlayConfig& cfg)
        : eval_(evaluator), cfg_(cfg) {}

    std::vector<TrainingSample> play(const RulesetConfig& rules, std::uint64_t seed,
                                     SelfPlayStats* stats_out = nullptr) {
        Player p0;
        p0.reset(rules, seed, 0);
        Player p1;
        p1.reset(rules, seed ^ 0xFEEDFACEull, 1);

        SearchConfig sc = cfg_.search;
        Searcher searcher(eval_, sc);
        GameRecorder recorder(
            cfg_.model_version, seed, static_cast<std::uint8_t>(cfg_.garbage_style),
            static_cast<std::uint8_t>(cfg_.garbage_period),
            static_cast<std::uint8_t>(cfg_.garbage_lines));
        Tokenizer tokenizer;
        SelfPlayStats stats;
        int p0_pieces = 0;
        int p1_pieces = 0;

        while (p0.alive() && p1.alive() && p0_pieces < cfg_.max_pieces && p1_pieces < cfg_.max_pieces) {
            bool p0_turn = (p0.now() <= p1.now());
            Player& p_active = p0_turn ? p0 : p1;
            Player& p_inactive = p0_turn ? p1 : p0;
            int& active_pieces = p0_turn ? p0_pieces : p1_pieces;

            const auto actions =
                movegen_.generate(p_active.board(), p_active.active().type, p_active.hold(),
                                  p_active.visible_next().empty() ? Piece::None : p_active.visible_next()[0],
                                  rules);
            if (actions.empty()) {
                p_active.die(TopoutReason::BlockOut);
                recorder.note_topout(p_active.index(), p_active.now(),
                                     recorder.size() == 0 ? 0 : recorder.size() - 1);
                break;
            }

            // Vary the search seed per move so determinizations differ, while
            // staying a deterministic function of (game seed, move number).
            sc.seed = seed * 0x9E3779B97F4A7C15ull + static_cast<std::uint64_t>(p0_pieces + p1_pieces);
            searcher.set_config(sc);

            const Observation obs = observe(p_active, &p_inactive);
            const SearchResult r = searcher.search(
                p_active, &p_inactive, &eval_,
                /*deliver_attacks=*/cfg_.garbage_style != GarbageStyle::None);
            if (r.best_action < 0 ||
                r.best_action >= static_cast<int>(actions.size())) {
                p_active.die(TopoutReason::BlockOut);
                recorder.note_topout(p_active.index(), p_active.now(),
                                     recorder.size() == 0 ? 0 : recorder.size() - 1);
                break;
            }

            recorder.add(obs, actions, r, tokenizer, p0_turn ? 1 : -1,
                         p_active.index(), p_active.now());

            const PlacementAction& chosen = actions[static_cast<size_t>(r.best_action)];
            if (chosen.use_hold && !p_active.do_hold()) {
                p_active.die(TopoutReason::BlockOut);
                recorder.note_topout(p_active.index(), p_active.now(), recorder.size() - 1);
                break;
            }
            p_active.set_active(chosen.piece_state());
            int sent = 0;
            const LockResult lr = p_active.lock_piece(chosen.total_duration(), &sent);
            
            recorder.note_outcome_of_last(sent, lr.garbage_received, p_active.now(),
                                          lr.topped_out);

            if (!lr.ok && !lr.topped_out) {
                p_active.die(TopoutReason::BlockOut);
                recorder.note_topout(p_active.index(), p_active.now(), recorder.size() - 1);
                break;
            }
            if (lr.topped_out) break;

            if (sent > 0 && cfg_.garbage_style != GarbageStyle::None) {
                p_inactive.receive_attack(sent, p_active.now(), p_active.index());
            }

            ++active_pieces;
        }

        stats.pieces = p0_pieces;
        stats.lines_cleared = static_cast<int>(p0.lines_cleared());
        stats.lines_sent = static_cast<int>(p0.lines_sent());
        stats.lines_received = static_cast<int>(p0.lines_received());
        stats.survived = p0.alive();
        stats.topout = p0.topout_reason();
        stats.duration = p0.now();

        float z = 0.0f;
        if (p0.alive() && !p1.alive()) {
            z = 1.0f;
        } else if (!p0.alive() && p1.alive()) {
            z = -1.0f;
        } else if (!p0.alive() && !p1.alive()) {
            if (p0_pieces > p1_pieces) {
                z = 1.0f;
            } else if (p0_pieces < p1_pieces) {
                z = -1.0f;
            } else {
                if (p0.lines_sent() > p1.lines_sent()) {
                    z = 1.0f;
                } else if (p0.lines_sent() < p1.lines_sent()) {
                    z = -1.0f;
                } else {
                    z = 0.0f;
                }
            }
        } else {
            if (cfg_.truncation_is_draw) {
                z = 0.0f;
            } else {
                if (p0.lines_sent() > p1.lines_sent()) {
                    z = 1.0f;
                } else if (p0.lines_sent() < p1.lines_sent()) {
                    z = -1.0f;
                } else if (p0.lines_cleared() > p1.lines_cleared()) {
                    z = 1.0f;
                } else if (p0.lines_cleared() < p1.lines_cleared()) {
                    z = -1.0f;
                } else {
                    z = 0.0f;
                }
            }
        }
        stats.outcome = z;

        stats.termination_reason =
            (p0.alive() && p1.alive()) ? TerminationReason::Truncated
                                       : TerminationReason::Terminated;
        recorder.set_termination(stats.termination_reason,
                                 std::max(p0.now(), p1.now()));

        if (stats_out) *stats_out = stats;
        return recorder.finalize(z);
    }

    const SelfPlayConfig& config() const { return cfg_; }
    void set_config(const SelfPlayConfig& c) { cfg_ = c; }

    Evaluator& eval_;
    SelfPlayConfig cfg_;
    MoveGenerator movegen_;
};

}  // namespace tetra
