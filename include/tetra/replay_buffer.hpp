// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- training samples and the replay buffer
// (spec 13.5, 17 /replay-buffer).
//
// This is the last piece between the search and a trainable network: self-play
// produces (observation, legal actions, search policy, outcome) tuples, and the
// buffer stores them until a learner consumes them.
//
// Two decisions worth stating:
//
//   * A sample stores the *tokenized* observation and action embeddings, not
//     the raw Player. Tokenization is deterministic and the network only ever
//     sees tokens, so storing them keeps samples compact, makes the on-disk
//     format independent of the simulator's internals, and guarantees the
//     learner cannot accidentally read state the observation mask removed.
//   * Every sample carries `ruleset_hash` and `model_version`, so samples from
//     different rules or different generations are never silently mixed
//     (spec 6, 14).
#pragma once

#include "tetra/evaluator.hpp"
#include "tetra/search.hpp"
#include "tetra/schema.hpp"
#include "tetra/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <vector>

namespace tetra {

// One training example (spec 13.5 TrainingSample).
struct TrainingSample {
    // Inputs, already tokenized.
    std::vector<Token> tokens;
    std::vector<ActionEmbedding> action_embeddings;

    // Targets.
    std::vector<float> search_policy;  // pi from the search visit counts
    float outcome = 0.0f;              // z in [-1, 1], filled in at game end
    float n_step_return = 0.0f;
    float search_value = 0.0f;         // the search's own value estimate

    // Auxiliary targets (spec 10.2, 13.5). Filled in once the future is known.
    int time_to_terminal = 0;          // placements remaining in the game
    float future_attack_1s = 0.0f;
    float future_garbage_received = 0.0f;
    bool topped_out_within_4 = false;
    bool topped_out_within_8 = false;

    // Phase 2a interval targets.  The first four entries retain the legacy
    // target order; the remaining entries are real-time and placement
    // interval targets in the order documented by schema.hpp.  Raw counts
    // stay in engine units here and are normalised only when a TensorBatch is
    // built, so statistics can be reported before the loss sees them.
    std::array<float, schema::AUX_TARGET_COUNT> aux_targets{};
    std::array<std::uint8_t, schema::AUX_TARGET_COUNT> aux_valid{};
    std::uint32_t aux_target_schema_version =
        schema::LEGACY_AUX_TARGET_SCHEMA_VERSION;
    std::uint32_t tokenizer_schema_version = schema::TOKENIZER_SCHEMA_VERSION;

    // Per-sample contract metadata.  It is intentionally stored beside the
    // target rather than inferred from array width, because two schemas may
    // have the same feature dimensions.
    TerminationReason termination_reason = TerminationReason::Unknown;
    std::int32_t player_index = 0;
    std::int32_t value_perspective = 1;
    Tick timestamp = 0;
    std::uint32_t trajectory_index = 0;
    std::uint8_t randomizer_type = 0;

    // Provenance.
    std::uint64_t ruleset_hash = 0;
    std::uint64_t game_seed = 0;
    std::uint32_t model_version = 0;
    std::uint32_t move_number = 0;
    int chosen_action = -1;
    std::uint8_t garbage_style = 1;
    std::uint8_t garbage_period = 8;
    std::uint8_t garbage_lines = 2;

    size_t action_count() const { return action_embeddings.size(); }
};

// Build a sample from a completed search. The outcome is left at zero and
// filled in by `finalize_game` once the result is known.
inline TrainingSample make_sample(const Observation& obs,
                                  const std::vector<PlacementAction>& actions,
                                  const SearchResult& result, const Tokenizer& tok,
                                  std::uint32_t model_version = 0,
                                  std::uint32_t move_number = 0,
                                  std::uint64_t game_seed = 0,
                                  std::uint8_t garbage_style = 1,
                                  std::uint8_t garbage_period = 8,
                                  std::uint8_t garbage_lines = 2) {
    TrainingSample s;
    const TokenizedObservation t = tok.encode(obs, obs.ruleset);
    s.tokens = t.tokens;
    s.action_embeddings.reserve(actions.size());
    for (const auto& a : actions) s.action_embeddings.push_back(embed_action(a, obs.board, obs.ruleset));
    s.search_policy = result.search_policy;
    s.search_value = result.value.scalar();
    s.chosen_action = result.best_action;
    s.ruleset_hash = obs.ruleset_hash;
    s.randomizer_type = static_cast<std::uint8_t>(obs.ruleset.randomizer.type);
    s.game_seed = game_seed;
    s.model_version = model_version;
    s.move_number = move_number;
    s.garbage_style = garbage_style;
    s.garbage_period = garbage_period;
    s.garbage_lines = garbage_lines;
    return s;
}

// Records one game's samples and stamps the outcome onto all of them.
//
// Spec 12.3 is emphatic that the primary reward is the game result and that
// garbage or attack must NOT be added to it; those are auxiliary prediction
// targets only. This class enforces that split structurally: `outcome` comes
// solely from win/draw/loss, and the attack/garbage fields are written to the
// auxiliary members.
class GameRecorder {
public:
    explicit GameRecorder(std::uint32_t model_version = 0, std::uint64_t game_seed = 0,
                          std::uint8_t garbage_style = 1, std::uint8_t garbage_period = 8,
                          std::uint8_t garbage_lines = 2)
        : model_version_(model_version),
          game_seed_(game_seed),
          garbage_style_(garbage_style),
          garbage_period_(garbage_period),
          garbage_lines_(garbage_lines) {}

    void add(const Observation& obs, const std::vector<PlacementAction>& actions,
             const SearchResult& result, const Tokenizer& tok) {
        add(obs, actions, result, tok, /*value_perspective=*/1);
    }

    // `value_perspective` identifies the player for whom the sample's value
    // target must be interpreted. Self-play records both players in one
    // chronological stream, so the final game result is converted to each
    // sample's own perspective instead of being copied blindly to all samples.
    void add(const Observation& obs, const std::vector<PlacementAction>& actions,
             const SearchResult& result, const Tokenizer& tok,
             int value_perspective) {
        add(obs, actions, result, tok, value_perspective,
            value_perspective < 0 ? 1 : 0, obs.timestamp);
    }

    void add(const Observation& obs, const std::vector<PlacementAction>& actions,
             const SearchResult& result, const Tokenizer& tok,
             int value_perspective, int player_index, Tick timestamp) {
        samples_.push_back(make_sample(obs, actions, result, tok, model_version_,
                                       static_cast<std::uint32_t>(samples_.size()),
                                       game_seed_, garbage_style_, garbage_period_,
                                       garbage_lines_));
        TrainingSample& sample = samples_.back();
        sample.value_perspective = value_perspective < 0 ? -1 : 1;
        sample.player_index = player_index;
        sample.timestamp = timestamp;
        sample.trajectory_index = static_cast<std::uint32_t>(samples_.size() - 1);
        sample.aux_target_schema_version = schema::AUX_TARGET_SCHEMA_VERSION;
        if (samples_.size() == 1)
            tick_rate_ = std::max(1, static_cast<int>(obs.ruleset.tick_rate));
        attack_at_.push_back(0.0f);
        garbage_at_.push_back(0.0f);
        perspectives_.push_back(value_perspective < 0 ? -1 : 1);
        player_indices_.push_back(player_index);
        timestamps_.push_back(timestamp);
        trajectory_indices_.push_back(samples_.size() - 1);
    }

    // Record what actually happened after the most recent placement, so the
    // auxiliary targets are ground truth rather than predictions.
    void note_outcome_of_last(int attack_sent, int garbage_received) {
        const Tick timestamp = timestamps_.empty() ? 0 : timestamps_.back();
        note_outcome_of_last(attack_sent, garbage_received, timestamp, false);
    }

    void note_outcome_of_last(int attack_sent, int garbage_received,
                              Tick timestamp, bool topped_out) {
        if (attack_at_.empty()) return;
        const size_t index = attack_at_.size() - 1;
        attack_at_[index] = static_cast<float>(attack_sent);
        garbage_at_[index] = static_cast<float>(garbage_received);
        const int actor = player_indices_[index];
        if (attack_sent > 0)
            events_.push_back(TraceEvent{timestamp, index, actor, TraceKind::Attack,
                                         attack_sent});
        if (garbage_received > 0)
            events_.push_back(TraceEvent{timestamp, index, actor, TraceKind::Garbage,
                                         garbage_received});
        if (topped_out) note_topout(actor, timestamp, index);
    }

    // Record a terminal top-out that happened before a sample could be added
    // (for example, a blocked spawn).  Duplicate notifications are harmless;
    // target generation treats top-out as a boolean per interval.
    void note_topout(int player_index, Tick timestamp,
                     size_t trajectory_index = std::numeric_limits<size_t>::max()) {
        if (trajectory_index == std::numeric_limits<size_t>::max() && !samples_.empty())
            trajectory_index = trajectory_indices_.back();
        if (trajectory_index == std::numeric_limits<size_t>::max()) trajectory_index = 0;
        events_.push_back(TraceEvent{timestamp, trajectory_index, player_index,
                                     TraceKind::Topout, 1});
    }

    void set_termination(TerminationReason reason, Tick end_timestamp) {
        termination_reason_ = reason;
        end_timestamp_ = end_timestamp;
    }

    // Stamp the final result onto every sample and return them.
    //
    // `z` is +1 for a win, 0 for a draw, -1 for a loss (spec 12.3). Discounting
    // is deliberately absent from the primary target: AlphaZero-style training
    // uses the plain final outcome.
    std::vector<TrainingSample> finalize(float z) {
        const int n = static_cast<int>(samples_.size());
        const TerminationReason reason =
            termination_reason_ == TerminationReason::Unknown
                ? TerminationReason::Terminated
                : termination_reason_;
        Tick observed_end = end_timestamp_;
        for (Tick timestamp : timestamps_) observed_end = std::max(observed_end, timestamp);
        const std::array<int, schema::HORIZON_COUNT> horizon_seconds = {1, 2, 4, 8};

        auto in_real_interval = [](Tick delta, int horizon, int previous) {
            return delta >= static_cast<Tick>(previous) &&
                   delta < static_cast<Tick>(horizon);
        };
        auto in_placement_interval = [](std::size_t distance, int horizon, int previous) {
            return distance >= static_cast<size_t>(previous) &&
                   distance < static_cast<size_t>(horizon);
        };
        auto interval_valid = [&](Tick sample_time, int horizon) {
            if (reason != TerminationReason::Truncated) return true;
            const Tick end = sample_time + static_cast<Tick>(horizon) * tick_rate_;
            return end <= observed_end;
        };
        auto placement_valid = [&](size_t sample_index, int horizon) {
            if (reason != TerminationReason::Truncated) return true;
            return sample_index + static_cast<size_t>(horizon) <= samples_.size();
        };

        for (int i = 0; i < n; ++i) {
            TrainingSample& s = samples_[static_cast<size_t>(i)];
            const int perspective = perspectives_[static_cast<size_t>(i)];
            const Tick sample_time = timestamps_[static_cast<size_t>(i)];
            const float local_z = z * static_cast<float>(perspective);
            s.outcome = local_z;
            s.n_step_return = local_z;
            s.time_to_terminal = n - i;
            s.topped_out_within_4 = (local_z < 0.0f) && (n - i <= 4);
            s.topped_out_within_8 = (local_z < 0.0f) && (n - i <= 8);

            // Attack and garbage over a short horizon: auxiliary targets, never
            // part of the reward. Only placements by the same player are
            // included; otherwise recording both players would mix their
            // future attack and garbage streams.
            float attack = 0.0f, garbage = 0.0f;
            for (int k = i; k < std::min(n, i + 4); ++k) {
                if (perspectives_[static_cast<size_t>(k)] == perspective) {
                    attack += attack_at_[static_cast<size_t>(k)];
                    garbage += garbage_at_[static_cast<size_t>(k)];
                }
            }
            s.future_attack_1s = attack;
            s.future_garbage_received = garbage;

            s.termination_reason = reason;
            s.aux_target_schema_version = schema::AUX_TARGET_SCHEMA_VERSION;
            s.aux_targets.fill(0.0f);
            s.aux_valid.fill(0);
            s.aux_targets[0] = s.future_attack_1s;
            s.aux_targets[1] = s.future_garbage_received;
            s.aux_targets[2] = static_cast<float>(s.time_to_terminal);
            s.aux_targets[3] = s.topped_out_within_8 ? 1.0f : 0.0f;
            const bool short_horizon_known =
                reason != TerminationReason::Truncated ||
                sample_time + static_cast<Tick>(tick_rate_) <= observed_end;
            s.aux_valid[0] = short_horizon_known ? 1 : 0;
            s.aux_valid[1] = short_horizon_known ? 1 : 0;
            s.aux_valid[2] = reason == TerminationReason::Terminated ? 1 : 0;
            s.aux_valid[3] = reason == TerminationReason::Terminated ? 1 : 0;

            const int actor = player_indices_[static_cast<size_t>(i)];
            const size_t sample_trajectory = trajectory_indices_[static_cast<size_t>(i)];
            for (int h = 0; h < schema::HORIZON_COUNT; ++h) {
                const int previous_seconds = h == 0 ? 0 : horizon_seconds[h - 1];
                const int end_seconds = horizon_seconds[h];
                float attack_value = 0.0f;
                float garbage_value = 0.0f;
                bool self_topout = false;
                bool opponent_topout = false;
                for (const TraceEvent& event : events_) {
                    const Tick delta = event.timestamp - sample_time;
                    if (!in_real_interval(delta, end_seconds * tick_rate_,
                                          previous_seconds * tick_rate_))
                        continue;
                    const bool self = event.actor == actor;
                    if (event.kind == TraceKind::Attack && self)
                        attack_value += static_cast<float>(event.amount);
                    else if (event.kind == TraceKind::Garbage && self)
                        garbage_value += static_cast<float>(event.amount);
                    else if (event.kind == TraceKind::Topout) {
                        if (self) self_topout = true;
                        else opponent_topout = true;
                    }
                }
                const bool valid = interval_valid(sample_time, end_seconds * tick_rate_);
                const int attack_index = schema::real_aux_index(h, schema::attack_channel);
                const int garbage_index = schema::real_aux_index(h, schema::garbage_channel);
                const int self_topout_index =
                    schema::real_aux_index(h, schema::self_topout_channel);
                const int opponent_topout_index =
                    schema::real_aux_index(h, schema::opponent_topout_channel);
                s.aux_targets[static_cast<size_t>(attack_index)] = attack_value;
                s.aux_targets[static_cast<size_t>(garbage_index)] = garbage_value;
                s.aux_targets[static_cast<size_t>(self_topout_index)] = self_topout ? 1.0f : 0.0f;
                s.aux_targets[static_cast<size_t>(opponent_topout_index)] =
                    opponent_topout ? 1.0f : 0.0f;
                s.aux_valid[static_cast<size_t>(attack_index)] = valid ? 1 : 0;
                s.aux_valid[static_cast<size_t>(garbage_index)] = valid ? 1 : 0;
                s.aux_valid[static_cast<size_t>(self_topout_index)] = valid ? 1 : 0;
                s.aux_valid[static_cast<size_t>(opponent_topout_index)] = valid ? 1 : 0;

                float placement_attack = 0.0f;
                float placement_garbage = 0.0f;
                bool placement_self_topout = false;
                bool placement_opponent_topout = false;
                for (const TraceEvent& event : events_) {
                    if (event.trajectory_index < sample_trajectory) continue;
                    const size_t distance = event.trajectory_index - sample_trajectory;
                    if (!in_placement_interval(distance, end_seconds, previous_seconds))
                        continue;
                    const bool self = event.actor == actor;
                    if (event.kind == TraceKind::Attack && self)
                        placement_attack += static_cast<float>(event.amount);
                    else if (event.kind == TraceKind::Garbage && self)
                        placement_garbage += static_cast<float>(event.amount);
                    else if (event.kind == TraceKind::Topout) {
                        if (self) placement_self_topout = true;
                        else placement_opponent_topout = true;
                    }
                }
                const bool pvalid = placement_valid(sample_trajectory, end_seconds);
                const int p_attack_index =
                    schema::placement_aux_index(h, schema::attack_channel);
                const int p_garbage_index =
                    schema::placement_aux_index(h, schema::garbage_channel);
                const int p_self_topout_index =
                    schema::placement_aux_index(h, schema::self_topout_channel);
                const int p_opponent_topout_index =
                    schema::placement_aux_index(h, schema::opponent_topout_channel);
                s.aux_targets[static_cast<size_t>(p_attack_index)] = placement_attack;
                s.aux_targets[static_cast<size_t>(p_garbage_index)] = placement_garbage;
                s.aux_targets[static_cast<size_t>(p_self_topout_index)] =
                    placement_self_topout ? 1.0f : 0.0f;
                s.aux_targets[static_cast<size_t>(p_opponent_topout_index)] =
                    placement_opponent_topout ? 1.0f : 0.0f;
                s.aux_valid[static_cast<size_t>(p_attack_index)] = pvalid ? 1 : 0;
                s.aux_valid[static_cast<size_t>(p_garbage_index)] = pvalid ? 1 : 0;
                s.aux_valid[static_cast<size_t>(p_self_topout_index)] = pvalid ? 1 : 0;
                s.aux_valid[static_cast<size_t>(p_opponent_topout_index)] = pvalid ? 1 : 0;
            }
        }
        std::vector<TrainingSample> out;
        out.swap(samples_);
        attack_at_.clear();
        garbage_at_.clear();
        perspectives_.clear();
        player_indices_.clear();
        timestamps_.clear();
        trajectory_indices_.clear();
        events_.clear();
        termination_reason_ = TerminationReason::Unknown;
        end_timestamp_ = 0;
        tick_rate_ = 60;
        return out;
    }

    size_t size() const { return samples_.size(); }
    void clear() {
        samples_.clear();
        attack_at_.clear();
        garbage_at_.clear();
        perspectives_.clear();
        player_indices_.clear();
        timestamps_.clear();
        trajectory_indices_.clear();
        events_.clear();
        termination_reason_ = TerminationReason::Unknown;
        end_timestamp_ = 0;
        tick_rate_ = 60;
    }

private:
    enum class TraceKind : std::uint8_t { Attack, Garbage, Topout };
    struct TraceEvent {
        Tick timestamp = 0;
        size_t trajectory_index = 0;
        int actor = 0;
        TraceKind kind = TraceKind::Attack;
        int amount = 0;
    };

    std::vector<TrainingSample> samples_;
    std::vector<float> attack_at_;
    std::vector<float> garbage_at_;
    std::vector<std::int8_t> perspectives_;
    std::vector<int> player_indices_;
    std::vector<Tick> timestamps_;
    std::vector<size_t> trajectory_indices_;
    std::vector<TraceEvent> events_;
    TerminationReason termination_reason_ = TerminationReason::Unknown;
    Tick end_timestamp_ = 0;
    int tick_rate_ = 60;
    std::uint32_t model_version_ = 0;
    std::uint64_t game_seed_ = 0;
    std::uint8_t garbage_style_ = 1;
    std::uint8_t garbage_period_ = 8;
    std::uint8_t garbage_lines_ = 2;
};

// ---------------------------------------------------------------------------
// Replay buffer
// ---------------------------------------------------------------------------
// A bounded FIFO of samples with reproducible sampling. Prioritisation is
// deliberately left out for now: uniform sampling is the correct default for
// AlphaZero-style training, and adding priorities before there is a learner to
// measure them against would be guesswork.
class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t capacity = 100000) : capacity_(capacity) {}

    void push(TrainingSample s) {
        samples_.push_back(std::move(s));
        ++total_added_;
        while (samples_.size() > capacity_) samples_.pop_front();
    }

    void push_game(std::vector<TrainingSample> game) {
        for (auto& s : game) push(std::move(s));
    }

    // Uniform sample without replacement within one batch.
    std::vector<const TrainingSample*> sample(size_t n, Rng& rng) const {
        std::vector<const TrainingSample*> out;
        if (samples_.empty() || n == 0) return out;
        n = std::min(n, samples_.size());
        // Partial Fisher-Yates over indices, so a batch never repeats a sample.
        std::vector<size_t> idx(samples_.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        for (size_t i = 0; i < n; ++i) {
            const size_t j =
                i + static_cast<size_t>(rng.below(static_cast<std::uint32_t>(idx.size() - i)));
            std::swap(idx[i], idx[j]);
            out.push_back(&samples_[idx[i]]);
        }
        return out;
    }

    // Reject samples recorded under different rules (spec 6, 14): mixing them
    // without an identifier is explicitly listed as a forbidden augmentation.
    size_t drop_other_rulesets(std::uint64_t keep_hash) {
        const size_t before = samples_.size();
        std::deque<TrainingSample> kept;
        for (auto& s : samples_)
            if (s.ruleset_hash == keep_hash) kept.push_back(std::move(s));
        samples_.swap(kept);
        return before - samples_.size();
    }

    size_t size() const { return samples_.size(); }
    size_t capacity() const { return capacity_; }
    std::uint64_t total_added() const { return total_added_; }
    bool empty() const { return samples_.empty(); }
    void clear() {
        samples_.clear();
        total_added_ = 0;
    }

    const TrainingSample& at(size_t i) const { return samples_[i]; }

    // Distinct rulesets currently present, for a mixed-buffer sanity check.
    std::vector<std::uint64_t> ruleset_hashes() const {
        std::vector<std::uint64_t> v;
        for (const auto& s : samples_)
            if (std::find(v.begin(), v.end(), s.ruleset_hash) == v.end())
                v.push_back(s.ruleset_hash);
        return v;
    }

private:
    std::deque<TrainingSample> samples_;
    size_t capacity_;
    std::uint64_t total_added_ = 0;
};

}  // namespace tetra
