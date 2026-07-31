// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- fixed-shape tensor batching (spec 9, 10.1, 17).
//
// The last structural gap between the engine and a network.
//
// Everything upstream is variable length by design: a position yields 48-65
// state tokens depending on the preview and event history, and 9-70 legal
// actions depending on the board. A Transformer needs rectangular tensors, so
// somewhere the ragged data has to be padded and masked. Doing that here rather
// than inside each backend means:
//
//   * the padding convention is defined once and tested once,
//   * masks are produced alongside the data, so a backend cannot forget them
//     and silently attend to padding, and
//   * the same flat buffers serve C++ inference, an ONNX session and a PyTorch
//     trainer without reshaping.
//
// Layout is row-major and contiguous, which is what every tensor library
// expects to wrap without a copy:
//
//   tokens  [B, T, TOKEN_FEATURES]
//   actions [B, A, ACTION_FEATURES]
//   masks   [B, T] and [B, A], 1 for real entries and 0 for padding
#pragma once

#include "tetra/replay_buffer.hpp"
#include "tetra/tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace tetra {

// Padded, rectangular view of a set of positions.
struct TensorBatch {
    int batch = 0;
    int max_tokens = 0;
    int max_actions = 0;

    std::vector<float> tokens;         // [B, T, TOKEN_FEATURES]
    std::vector<float> token_mask;     // [B, T]
    std::vector<float> actions;        // [B, A, ACTION_FEATURES]
    std::vector<float> action_mask;    // [B, A]

    // Per-position metadata the model conditions on or the loss needs.
    std::vector<std::int32_t> token_kind;    // [B, T]
    std::vector<std::int32_t> token_player;  // [B, T]
    std::vector<std::int32_t> token_index;   // [B, T]
    std::vector<std::int32_t> action_count;  // [B]

    // Targets. Empty for an inference batch.
    std::vector<float> policy_target;  // [B, A], zero-padded, sums to 1 per row
    std::vector<float> value_target;   // [B]
    std::vector<float> aux_target;     // [B, AUX_TARGETS]

    static constexpr int AUX_TARGETS = 4;  // attack, garbage, ttk, topout risk

    size_t token_stride() const { return static_cast<size_t>(max_tokens) * TOKEN_FEATURES; }
    size_t action_stride() const { return static_cast<size_t>(max_actions) * ACTION_FEATURES; }

    const float* token_data(int b) const {
        return tokens.data() + static_cast<size_t>(b) * token_stride();
    }
    const float* action_data(int b) const {
        return actions.data() + static_cast<size_t>(b) * action_stride();
    }

    void clear() {
        batch = max_tokens = max_actions = 0;
        tokens.clear();
        token_mask.clear();
        actions.clear();
        action_mask.clear();
        token_kind.clear();
        token_player.clear();
        token_index.clear();
        action_count.clear();
        policy_target.clear();
        value_target.clear();
        aux_target.clear();
    }
};

namespace detail {

// Allocate and zero the buffers for a [B, T, A] batch.
inline void allocate_batch(TensorBatch& out, int b, int t, int a, bool with_targets) {
    out.batch = b;
    out.max_tokens = t;
    out.max_actions = a;
    const size_t bt = static_cast<size_t>(b) * static_cast<size_t>(t);
    const size_t ba = static_cast<size_t>(b) * static_cast<size_t>(a);

    out.tokens.assign(bt * TOKEN_FEATURES, 0.0f);
    out.token_mask.assign(bt, 0.0f);
    out.actions.assign(ba * ACTION_FEATURES, 0.0f);
    out.action_mask.assign(ba, 0.0f);
    out.token_kind.assign(bt, 0);
    out.token_player.assign(bt, 0);
    out.token_index.assign(bt, 0);
    out.action_count.assign(static_cast<size_t>(b), 0);

    if (with_targets) {
        out.policy_target.assign(ba, 0.0f);
        out.value_target.assign(static_cast<size_t>(b), 0.0f);
        out.aux_target.assign(static_cast<size_t>(b) * TensorBatch::AUX_TARGETS, 0.0f);
    }
}

// Copy one position's tokens and actions into row `b`.
inline void fill_row(TensorBatch& out, int b, const std::vector<Token>& tokens,
                     const std::vector<ActionEmbedding>& actions) {
    const int t = std::min(static_cast<int>(tokens.size()), out.max_tokens);
    const int a = std::min(static_cast<int>(actions.size()), out.max_actions);

    for (int i = 0; i < t; ++i) {
        const size_t base =
            static_cast<size_t>(b) * out.token_stride() + static_cast<size_t>(i) * TOKEN_FEATURES;
        for (int k = 0; k < TOKEN_FEATURES; ++k)
            out.tokens[base + static_cast<size_t>(k)] = tokens[static_cast<size_t>(i)].f[static_cast<size_t>(k)];
        const size_t m = static_cast<size_t>(b) * static_cast<size_t>(out.max_tokens) +
                         static_cast<size_t>(i);
        out.token_mask[m] = 1.0f;
        out.token_kind[m] = static_cast<std::int32_t>(tokens[static_cast<size_t>(i)].kind);
        out.token_player[m] = tokens[static_cast<size_t>(i)].player;
        out.token_index[m] = tokens[static_cast<size_t>(i)].index;
    }

    for (int i = 0; i < a; ++i) {
        const size_t base = static_cast<size_t>(b) * out.action_stride() +
                            static_cast<size_t>(i) * ACTION_FEATURES;
        for (int k = 0; k < ACTION_FEATURES; ++k)
            out.actions[base + static_cast<size_t>(k)] =
                actions[static_cast<size_t>(i)].f[static_cast<size_t>(k)];
        out.action_mask[static_cast<size_t>(b) * static_cast<size_t>(out.max_actions) +
                        static_cast<size_t>(i)] = 1.0f;
    }
    out.action_count[static_cast<size_t>(b)] = a;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Inference batching
// ---------------------------------------------------------------------------
// Build a batch from live positions. `pad_to_*` lets a backend with a static
// input shape (ONNX, TensorRT) request fixed dimensions; leave them at zero to
// size the batch to its contents, which is cheaper for a CPU backend.
inline TensorBatch make_inference_batch(const std::vector<EvalRequest>& batch,
                                        const Tokenizer& tok, int pad_tokens = 0,
                                        int pad_actions = 0) {
    TensorBatch out;
    if (batch.empty()) return out;

    std::vector<TokenizedObservation> encoded;
    encoded.reserve(batch.size());
    int max_t = 0, max_a = 0;
    for (const auto& req : batch) {
        if (req.observation)
            encoded.push_back(tok.encode(*req.observation, req.observation->ruleset));
        else
            encoded.emplace_back();
        max_t = std::max(max_t, static_cast<int>(encoded.back().tokens.size()));
        max_a = std::max(max_a,
                         static_cast<int>(req.actions ? req.actions->size() : 0));
    }
    if (pad_tokens > 0) max_t = pad_tokens;
    if (pad_actions > 0) max_a = pad_actions;
    max_t = std::max(max_t, 1);
    max_a = std::max(max_a, 1);

    detail::allocate_batch(out, static_cast<int>(batch.size()), max_t, max_a,
                           /*with_targets=*/false);

    for (size_t b = 0; b < batch.size(); ++b) {
        std::vector<ActionEmbedding> embeddings;
        if (batch[b].actions && batch[b].observation) {
            embeddings.reserve(batch[b].actions->size());
            for (const auto& a : *batch[b].actions)
                embeddings.push_back(
                    embed_action(a, batch[b].observation->board, batch[b].observation->ruleset));
        }
        detail::fill_row(out, static_cast<int>(b), encoded[b].tokens, embeddings);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Training batching
// ---------------------------------------------------------------------------
// Samples already carry tokens and action embeddings, so this is pure padding
// plus target assembly. The policy target is renormalised over the actions that
// actually fit, so a truncated row is still a valid distribution.
inline TensorBatch make_training_batch(const std::vector<const TrainingSample*>& samples,
                                       int pad_tokens = 0, int pad_actions = 0) {
    TensorBatch out;
    if (samples.empty()) return out;

    int max_t = 0, max_a = 0;
    for (const auto* s : samples) {
        max_t = std::max(max_t, static_cast<int>(s->tokens.size()));
        max_a = std::max(max_a, static_cast<int>(s->action_embeddings.size()));
    }
    if (pad_tokens > 0) max_t = pad_tokens;
    if (pad_actions > 0) max_a = pad_actions;
    max_t = std::max(max_t, 1);
    max_a = std::max(max_a, 1);

    detail::allocate_batch(out, static_cast<int>(samples.size()), max_t, max_a,
                           /*with_targets=*/true);

    for (size_t b = 0; b < samples.size(); ++b) {
        const TrainingSample& s = *samples[b];
        detail::fill_row(out, static_cast<int>(b), s.tokens, s.action_embeddings);

        // Policy target, renormalised over the retained actions.
        const int a = out.action_count[b];
        float sum = 0.0f;
        for (int i = 0; i < a && i < static_cast<int>(s.search_policy.size()); ++i)
            sum += s.search_policy[static_cast<size_t>(i)];
        const size_t row = static_cast<size_t>(b) * static_cast<size_t>(out.max_actions);
        if (sum > 1e-9f) {
            for (int i = 0; i < a && i < static_cast<int>(s.search_policy.size()); ++i)
                out.policy_target[row + static_cast<size_t>(i)] =
                    s.search_policy[static_cast<size_t>(i)] / sum;
        } else if (a > 0) {
            // A sample with no policy mass (possible only if the search found
            // nothing) becomes a uniform target rather than an all-zero row,
            // which would produce an undefined cross-entropy.
            for (int i = 0; i < a; ++i)
                out.policy_target[row + static_cast<size_t>(i)] = 1.0f / static_cast<float>(a);
        }

        out.value_target[b] = s.outcome;

        // Auxiliary targets are squashed into roughly [0, 1] before they leave
        // the engine. They are trained with MSE alongside the policy and value
        // losses, so a raw count would dominate the total purely by scale:
        // `time_to_terminal` reaches a few hundred placements, which produced
        // an auxiliary loss around 150 against a policy loss around 3 and made
        // the auxiliary head the only thing the model was really fitting.
        // Normalising here keeps the loss weights in the trainer meaningful.
        const size_t aux = static_cast<size_t>(b) * TensorBatch::AUX_TARGETS;
        auto squash = [](float v, float scale) { return v / (v + scale); };
        out.aux_target[aux + 0] = squash(std::max(0.0f, s.future_attack_1s), 8.0f);
        out.aux_target[aux + 1] = squash(std::max(0.0f, s.future_garbage_received), 8.0f);
        out.aux_target[aux + 2] =
            squash(static_cast<float>(std::max(0, s.time_to_terminal)), 64.0f);
        out.aux_target[aux + 3] = s.topped_out_within_8 ? 1.0f : 0.0f;
    }
    return out;
}

}  // namespace tetra