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
#include "tetra/tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <deque>
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

    // Provenance.
    std::uint64_t ruleset_hash = 0;
    std::uint32_t model_version = 0;
    std::uint32_t move_number = 0;
    int chosen_action = -1;

    size_t action_count() const { return action_embeddings.size(); }
};

// Build a sample from a completed search. The outcome is left at zero and
// filled in by `finalize_game` once the result is known.
inline TrainingSample make_sample(const Observation& obs,
                                  const std::vector<PlacementAction>& actions,
                                  const SearchResult& result, const Tokenizer& tok,
                                  std::uint32_t model_version = 0,
                                  std::uint32_t move_number = 0) {
    TrainingSample s;
    const TokenizedObservation t = tok.encode(obs, obs.ruleset);
    s.tokens = t.tokens;
    s.action_embeddings.reserve(actions.size());
    for (const auto& a : actions) s.action_embeddings.push_back(embed_action(a, obs.board, obs.ruleset));
    s.search_policy = result.search_policy;
    s.search_value = result.value.scalar();
    s.chosen_action = result.best_action;
    s.ruleset_hash = obs.ruleset_hash;
    s.model_version = model_version;
    s.move_number = move_number;
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
    explicit GameRecorder(std::uint32_t model_version = 0) : model_version_(model_version) {}

    void add(const Observation& obs, const std::vector<PlacementAction>& actions,
             const SearchResult& result, const Tokenizer& tok) {
        samples_.push_back(make_sample(obs, actions, result, tok, model_version_,
                                       static_cast<std::uint32_t>(samples_.size())));
        attack_at_.push_back(0.0f);
        garbage_at_.push_back(0.0f);
    }

    // Record what actually happened after the most recent placement, so the
    // auxiliary targets are ground truth rather than predictions.
    void note_outcome_of_last(int attack_sent, int garbage_received) {
        if (attack_at_.empty()) return;
        attack_at_.back() = static_cast<float>(attack_sent);
        garbage_at_.back() = static_cast<float>(garbage_received);
    }

    // Stamp the final result onto every sample and return them.
    //
    // `z` is +1 for a win, 0 for a draw, -1 for a loss (spec 12.3). Discounting
    // is deliberately absent from the primary target: AlphaZero-style training
    // uses the plain final outcome.
    std::vector<TrainingSample> finalize(float z) {
        const int n = static_cast<int>(samples_.size());
        for (int i = 0; i < n; ++i) {
            TrainingSample& s = samples_[static_cast<size_t>(i)];
            s.outcome = z;
            s.n_step_return = z;
            s.time_to_terminal = n - i;
            s.topped_out_within_4 = (z < 0.0f) && (n - i <= 4);
            s.topped_out_within_8 = (z < 0.0f) && (n - i <= 8);

            // Attack and garbage over a short horizon: auxiliary targets, never
            // part of the reward.
            float attack = 0.0f, garbage = 0.0f;
            for (int k = i; k < std::min(n, i + 4); ++k) {
                attack += attack_at_[static_cast<size_t>(k)];
                garbage += garbage_at_[static_cast<size_t>(k)];
            }
            s.future_attack_1s = attack;
            s.future_garbage_received = garbage;
        }
        std::vector<TrainingSample> out;
        out.swap(samples_);
        attack_at_.clear();
        garbage_at_.clear();
        return out;
    }

    size_t size() const { return samples_.size(); }
    void clear() {
        samples_.clear();
        attack_at_.clear();
        garbage_at_.clear();
    }

private:
    std::vector<TrainingSample> samples_;
    std::vector<float> attack_at_;
    std::vector<float> garbage_at_;
    std::uint32_t model_version_ = 0;
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