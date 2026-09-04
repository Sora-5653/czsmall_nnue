// SPDX-License-Identifier: MIT
// Historical-state reconstruction and target refresh support.
#pragma once

#include "tetra/batch.hpp"
#include "tetra/dataset.hpp"
#include "tetra/player.hpp"
#include "tetra/search.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tetra {

struct ReanalyseRoot {
    std::size_t source_row = 0;
    Player active;
    Player inactive;
    Observation observation;
    std::vector<PlacementAction> actions;
};

struct ReconstructionReport {
    std::vector<ReanalyseRoot> roots;
    std::size_t token_rows_verified = 0;
    std::size_t action_rows_verified = 0;
    std::string error;

    bool ok() const { return error.empty(); }
};

inline bool reanalyse_float_equal(float a, float b) {
    return std::fabs(a - b) <= 1e-6f;
}

// Rebuild every historical root by replaying the recorded behavioural action.
// The stored tensors are used only as a parity oracle; they are never used to
// fabricate simulator state.
inline ReconstructionReport reconstruct_historical_roots(
    const TensorBatch& source, const RulesetConfig& rules,
    bool deliver_attacks, bool enable_timing_actions = false) {
    ReconstructionReport report;
    if (source.batch <= 0) {
        report.error = "source dataset is empty";
        return report;
    }
    if (source.game_seed.size() != static_cast<std::size_t>(source.batch) ||
        source.move_number.size() != static_cast<std::size_t>(source.batch) ||
        source.chosen_action.size() != static_cast<std::size_t>(source.batch) ||
        source.player_perspective.size() != static_cast<std::size_t>(source.batch)) {
        report.error = "dataset lacks v4 replay provenance";
        return report;
    }

    Player p0;
    Player p1;
    std::uint64_t current_seed = 0;
    bool have_game = false;
    std::uint32_t expected_move = 0;
    MoveGenerator movegen;
    Tokenizer tokenizer;

    for (int row = 0; row < source.batch; ++row) {
        const std::size_t r = static_cast<std::size_t>(row);
        const std::uint64_t seed = source.game_seed[r];
        if (!have_game || seed != current_seed) {
            if (seed == 0) {
                report.error = "row " + std::to_string(row) + " has no game seed";
                return report;
            }
            p0.reset(rules, seed, 0);
            p1.reset(rules, seed ^ 0xFEEDFACEull, 1);
            current_seed = seed;
            expected_move = 0;
            have_game = true;
        }
        if (source.move_number[r] != expected_move) {
            report.error = "non-contiguous move number at row " + std::to_string(row);
            return report;
        }

        const bool p0_turn = p0.now() <= p1.now();
        Player& active = p0_turn ? p0 : p1;
        Player& inactive = p0_turn ? p1 : p0;
        const int expected_perspective = p0_turn ? 1 : -1;
        if (source.player_perspective[r] != expected_perspective) {
            report.error = "active-player mismatch at row " + std::to_string(row);
            return report;
        }

        std::vector<PlacementAction> actions = movegen.generate(
            active.board(), active.active().type, active.hold(),
            active.visible_next().empty() ? Piece::None : active.visible_next()[0], rules);
        if (enable_timing_actions) {
            const Tick activation = active.garbage().next_activation(active.now());
            if (activation != TICK_NEVER && !actions.empty()) {
                actions = MoveGenerator::expand_delay_bins(
                    actions, rules, active.now(), activation, TICK_NEVER,
                    {DelayBin::Fastest, DelayBin::WaitForEvent});
            }
        }
        const int chosen = source.chosen_action[r];
        if (chosen < 0 || chosen >= static_cast<int>(actions.size())) {
            report.error = "invalid chosen_action at row " + std::to_string(row);
            return report;
        }

        const Observation obs = observe(active, &inactive);
        const TokenizedObservation encoded = tokenizer.encode(obs, rules);
        int stored_tokens = 0;
        while (stored_tokens < source.max_tokens &&
               source.token_mask[r * static_cast<std::size_t>(source.max_tokens) +
                                 static_cast<std::size_t>(stored_tokens)] > 0.5f)
            ++stored_tokens;
        int stored_actions = 0;
        while (stored_actions < source.max_actions &&
               source.action_mask[r * static_cast<std::size_t>(source.max_actions) +
                                  static_cast<std::size_t>(stored_actions)] > 0.5f)
            ++stored_actions;
        if (stored_tokens != static_cast<int>(encoded.tokens.size()) ||
            stored_actions != static_cast<int>(actions.size())) {
            report.error = "token/action count mismatch at row " + std::to_string(row);
            return report;
        }
        for (int i = 0; i < stored_tokens; ++i) {
            const std::size_t base = r * source.token_stride() +
                                     static_cast<std::size_t>(i) * TOKEN_FEATURES;
            for (int k = 0; k < TOKEN_FEATURES; ++k) {
                if (!reanalyse_float_equal(
                        source.tokens[base + static_cast<std::size_t>(k)],
                        encoded.tokens[static_cast<std::size_t>(i)].f[static_cast<std::size_t>(k)])) {
                    report.error = "token parity mismatch at row " + std::to_string(row);
                    return report;
                }
            }
        }
        for (int i = 0; i < stored_actions; ++i) {
            const ActionEmbedding embedded = embed_action(
                actions[static_cast<std::size_t>(i)], obs.board, rules);
            const std::size_t base = r * source.action_stride() +
                                     static_cast<std::size_t>(i) * ACTION_FEATURES;
            for (int k = 0; k < ACTION_FEATURES; ++k) {
                if (!reanalyse_float_equal(
                        source.actions[base + static_cast<std::size_t>(k)],
                        embedded.f[static_cast<std::size_t>(k)])) {
                    report.error = "action parity mismatch at row " + std::to_string(row);
                    return report;
                }
            }
        }
        ++report.token_rows_verified;
        ++report.action_rows_verified;
        report.roots.push_back(ReanalyseRoot{r, active, inactive, obs, actions});

        const PlacementAction& action = actions[static_cast<std::size_t>(chosen)];
        if (action.use_hold && !active.do_hold()) {
            report.error = "historical hold failed at row " + std::to_string(row);
            return report;
        }
        active.set_active(action.piece_state());
        int sent = 0;
        const LockResult lr = active.lock_piece(action.total_duration(), &sent);
        if (!lr.ok && !lr.topped_out) {
            report.error = "historical lock failed at row " + std::to_string(row);
            return report;
        }
        if (sent > 0 && deliver_attacks)
            inactive.receive_attack(sent, active.now(), active.index());
        ++expected_move;
    }
    return report;
}

inline double policy_kl(const float* historical, const std::vector<float>& current,
                        int action_count) {
    constexpr double eps = 1e-8;
    double score = 0.0;
    for (int i = 0; i < action_count; ++i) {
        const double p = std::max(eps, static_cast<double>(historical[i]));
        const double q = std::max(eps, static_cast<double>(current[static_cast<std::size_t>(i)]));
        score += p * std::log(p / q);
    }
    return score;
}

inline TensorBatch select_reanalysed_rows(
    const TensorBatch& source, const std::vector<std::size_t>& rows,
    const std::vector<std::vector<float>>& refreshed_policies) {
    if (rows.size() != refreshed_policies.size())
        throw std::invalid_argument("row/policy count mismatch");
    TensorBatch out;
    detail::allocate_batch(out, static_cast<int>(rows.size()), source.max_tokens,
                           source.max_actions, true);
    const std::size_t token_stride = source.token_stride();
    const std::size_t action_stride = source.action_stride();
    const std::size_t token_mask_stride = static_cast<std::size_t>(source.max_tokens);
    const std::size_t action_mask_stride = static_cast<std::size_t>(source.max_actions);
    const std::size_t aux_stride = source.batch > 0
        ? source.aux_target.size() / static_cast<std::size_t>(source.batch) : 0;
    out.aux_target.assign(rows.size() * aux_stride, 0.0f);
    out.aux_valid_mask.assign(rows.size() * aux_stride, 0.0f);

    for (std::size_t dst = 0; dst < rows.size(); ++dst) {
        const std::size_t src = rows[dst];
        std::copy_n(source.tokens.data() + src * token_stride, token_stride,
                    out.tokens.data() + dst * token_stride);
        std::copy_n(source.token_mask.data() + src * token_mask_stride, token_mask_stride,
                    out.token_mask.data() + dst * token_mask_stride);
        std::copy_n(source.actions.data() + src * action_stride, action_stride,
                    out.actions.data() + dst * action_stride);
        std::copy_n(source.action_mask.data() + src * action_mask_stride, action_mask_stride,
                    out.action_mask.data() + dst * action_mask_stride);
        const int count = static_cast<int>(std::count_if(
            out.action_mask.data() + dst * action_mask_stride,
            out.action_mask.data() + (dst + 1) * action_mask_stride,
            [](float x) { return x > 0.5f; }));
        out.action_count[dst] = count;
        if (static_cast<int>(refreshed_policies[dst].size()) != count)
            throw std::invalid_argument("refreshed policy/action mismatch");
        std::copy(refreshed_policies[dst].begin(), refreshed_policies[dst].end(),
                  out.policy_target.data() + dst * action_mask_stride);
        out.value_target[dst] = source.value_target[src];
        if (aux_stride > 0) {
            std::copy_n(source.aux_target.data() + src * aux_stride, aux_stride,
                        out.aux_target.data() + dst * aux_stride);
            std::copy_n(source.aux_valid_mask.data() + src * aux_stride, aux_stride,
                        out.aux_valid_mask.data() + dst * aux_stride);
        }
        out.player_perspective[dst] = source.player_perspective[src];
        out.termination_reason[dst] = source.termination_reason[src];
        out.game_seed[dst] = source.game_seed[src];
        out.move_number[dst] = source.move_number[src];
        out.chosen_action[dst] = source.chosen_action[src];
    }
    return out;
}

}  // namespace tetra
