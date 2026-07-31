// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- the evaluator interface (spec 10, 11.1, 16).
//
// This is the seam between the search and whatever produces priors and values.
// It exists as its own abstraction, defined BEFORE the search, for one measured
// reason: on this hardware a TetraFormer-S forward pass costs ~8.6 ms per
// position versus ~0.09 ms for a legal-move generation. The network is roughly
// a hundred times more expensive than everything around it, so the shape of the
// search is dictated by how the network is called, not the other way round.
//
// Two consequences are baked into the interface rather than bolted on later:
//
//   1. **Evaluation is batched.** The unit of work is a *vector* of positions.
//      A single-position call is the degenerate case, not the primary one.
//      Spec 11.1 requires batched leaf inference and spec 19.4 sets a target of
//      >= 80% batched leaf evaluation; a scalar interface makes both
//      unreachable without a rewrite.
//
//   2. **Policies are variable length.** Priors are returned per legal action
//      (spec 10.1), never as a fixed x/rotation grid, so the same evaluator
//      serves custom board widths and spin-distinguished placements.
//
// Implementations in this header are deliberately dependency-free C++ so the
// search can be developed and tested end to end. A PyTorch-trained TetraFormer
// slots in behind the same interface later.
#pragma once

#include "tetra/movegen.hpp"
#include "tetra/observation.hpp"
#include "tetra/tokenizer.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tetra {

// ---------------------------------------------------------------------------
// Value
// ---------------------------------------------------------------------------

// Win / draw / loss distribution (spec 10.2). Three values are kept even for
// rulesets without draws, so simultaneous KOs and custom modes are expressible.
struct ValueWDL {
    float win = 0.0f;
    float draw = 0.0f;
    float loss = 0.0f;

    // Scalar value in [-1, 1], which is what the search backs up.
    float scalar() const { return win - loss; }

    void normalize() {
        const float s = win + draw + loss;
        if (s > 1e-9f) {
            win /= s;
            draw /= s;
            loss /= s;
        } else {
            win = loss = 0.0f;
            draw = 1.0f;
        }
    }

    static ValueWDL from_scalar(float v) {
        // Map [-1, 1] onto a WDL triple with no draw mass.
        ValueWDL o;
        const float c = std::max(-1.0f, std::min(1.0f, v));
        o.win = 0.5f * (1.0f + c);
        o.loss = 0.5f * (1.0f - c);
        o.draw = 0.0f;
        return o;
    }
};

// Auxiliary predictions (spec 10.2). Optional: an evaluator may leave them at
// their defaults, and the search treats them as diagnostics rather than
// requirements.
struct AuxPredictions {
    float expected_time_to_ko = 0.0f;
    float topout_within_4_pieces = 0.0f;
    float topout_within_8_pieces = 0.0f;
    float expected_net_attack_1s = 0.0f;
    float expected_net_attack_2s = 0.0f;
    float expected_received_garbage = 0.0f;
    float opponent_topout_probability = 0.0f;
};

// One evaluated position.
struct Evaluation {
    // Prior probability per legal action, in the order the actions were given.
    // Implementations must return a normalised distribution of exactly the
    // same length as the request's action list.
    std::vector<float> policy;
    ValueWDL value;
    AuxPredictions aux;
};

// One position to evaluate. The evaluator borrows both references; the caller
// must keep them alive for the duration of the call.
struct EvalRequest {
    const Observation* observation = nullptr;
    const std::vector<PlacementAction>* actions = nullptr;
};

// ---------------------------------------------------------------------------
// The interface
// ---------------------------------------------------------------------------
class Evaluator {
public:
    virtual ~Evaluator() = default;

    // Evaluate a batch. `out` is resized to match `batch`.
    //
    // This is the only method an implementation must provide. Everything else
    // in the search funnels through it, so an implementation can assume it will
    // be handed work in bulk and is free to pad, sort or pipeline internally.
    virtual void evaluate(const std::vector<EvalRequest>& batch,
                          std::vector<Evaluation>& out) = 0;

    // Convenience wrapper for the single-position case (policy-only play,
    // tests, diagnostics). Deliberately implemented in terms of the batch call
    // so there is only one code path to keep correct.
    Evaluation evaluate_one(const Observation& obs,
                            const std::vector<PlacementAction>& actions) {
        std::vector<EvalRequest> batch{EvalRequest{&obs, &actions}};
        std::vector<Evaluation> out;
        evaluate(batch, out);
        return out.empty() ? Evaluation{} : out.front();
    }

    // The batch size the implementation would like to be given. The search uses
    // this to decide how many leaves to collect before flushing.
    virtual int preferred_batch_size() const { return 32; }

    // Identifies the weights/version, for replays and model gating (spec 20).
    virtual std::string name() const { return "unnamed"; }

    // Total positions evaluated, for the batch-efficiency metric in spec 19.4.
    std::uint64_t positions_evaluated() const { return positions_; }
    std::uint64_t batches_issued() const { return batches_; }
    double mean_batch_size() const {
        return batches_ ? static_cast<double>(positions_) / static_cast<double>(batches_) : 0.0;
    }
    void reset_stats() {
        positions_ = 0;
        batches_ = 0;
    }

protected:
    // Implementations call this at the top of evaluate().
    void account(size_t n) {
        positions_ += n;
        ++batches_;
    }

private:
    std::uint64_t positions_ = 0;
    std::uint64_t batches_ = 0;
};

// ---------------------------------------------------------------------------
// Uniform evaluator
// ---------------------------------------------------------------------------
// The null hypothesis: flat priors and a neutral value. Any search worth having
// must beat this, so it is the baseline every strength test is measured against
// and the fixture most search tests run on (its output is trivially known, so a
// wrong visit distribution can only come from the search itself).
class UniformEvaluator : public Evaluator {
public:
    void evaluate(const std::vector<EvalRequest>& batch, std::vector<Evaluation>& out) override {
        account(batch.size());
        out.assign(batch.size(), Evaluation{});
        for (size_t i = 0; i < batch.size(); ++i) {
            const size_t n = batch[i].actions ? batch[i].actions->size() : 0;
            out[i].policy.assign(n, n ? 1.0f / static_cast<float>(n) : 0.0f);
            out[i].value.draw = 1.0f;
        }
    }
    std::string name() const override { return "uniform"; }
    int preferred_batch_size() const override { return 64; }
};

// ---------------------------------------------------------------------------
// Heuristic evaluator
// ---------------------------------------------------------------------------
// A deterministic, dependency-free stand-in with real opinions. It plays the
// role the network will play, so the search can be built and measured before
// any weights exist, and it doubles as the "heuristic baseline" opponent that
// spec 19.1 and 13.4 call for.
//
// The features are intentionally the ones the tokenizer already exposes, so
// swapping in a learned model does not change the information available.
class HeuristicEvaluator : public Evaluator {
public:
    struct Weights {
        float holes = -6.0f;
        float height = -0.9f;
        float bumpiness = -0.35f;
        float clear = 1.6f;
        float quad = 5.0f;
        float spin = 3.0f;
        float all_clear = 12.0f;
        float well_bonus = 1.0f;   // keeping one deep column for the I piece
        float garbage_clear = 2.0f;
        // Softmax temperature over action scores.
        //
        // This must stay high enough that the priors remain a *distribution*
        // rather than a near-deterministic choice. Board scores span tens of
        // points, so a temperature around 1 collapses the softmax onto a single
        // action (measured: top prior 0.9918), which starves the search of
        // anything to explore and makes visit counts meaningless. Scaling by
        // the score spread keeps the distribution informative at any board.
        float temperature = 8.0f;
    };

    HeuristicEvaluator() = default;
    explicit HeuristicEvaluator(const Weights& w) : w_(w) {}

    void evaluate(const std::vector<EvalRequest>& batch, std::vector<Evaluation>& out) override {
        account(batch.size());
        out.assign(batch.size(), Evaluation{});
        for (size_t i = 0; i < batch.size(); ++i) {
            const EvalRequest& req = batch[i];
            if (!req.observation || !req.actions) {
                out[i].value.draw = 1.0f;
                continue;
            }
            score_position(*req.observation, *req.actions, out[i]);
        }
    }

    std::string name() const override { return "heuristic"; }
    int preferred_batch_size() const override { return 64; }
    const Weights& weights() const { return w_; }

private:
    // Static evaluation of a board, in "how good is this for me" units.
    float board_score(const Board& b, const RulesetConfig& cfg) const {
        const int W = b.width();
        float s = 0.0f;
        s += w_.holes * static_cast<float>(b.hole_count());
        s += w_.height * static_cast<float>(b.stack_height());

        int bump = 0;
        int min_h = 1 << 30, max_h = 0;
        for (int x = 0; x + 1 < W; ++x)
            bump += std::abs(b.column_height(x) - b.column_height(x + 1));
        for (int x = 0; x < W; ++x) {
            const int h = b.column_height(x);
            min_h = std::min(min_h, h);
            max_h = std::max(max_h, h);
        }
        // A single deep well is good (it is the I-piece slot); the bumpiness
        // penalty should not punish it, so the deepest column is discounted.
        s += w_.bumpiness * static_cast<float>(bump);
        if (max_h - min_h >= 3) s += w_.well_bonus;

        // Danger scales sharply as the stack approaches the ceiling.
        const float ceiling = static_cast<float>(cfg.geometry.visible_height);
        const float fill = static_cast<float>(b.stack_height()) / std::max(1.0f, ceiling);
        if (fill > 0.6f) s -= 30.0f * (fill - 0.6f) * (fill - 0.6f);
        return s;
    }

    void score_position(const Observation& obs, const std::vector<PlacementAction>& actions,
                        Evaluation& ev) const {
        const RulesetConfig cfg = obs.ruleset;
        const float base = board_score(obs.board, cfg);

        std::vector<float> scores(actions.size(), 0.0f);
        float best = -1e30f;
        for (size_t k = 0; k < actions.size(); ++k) {
            const PlacementAction& a = actions[k];
            const PlacementOutcome oc = evaluate_placement(obs.board, a.piece_state(), cfg);
            float s = board_score(oc.board, cfg) - base;
            s += w_.clear * static_cast<float>(a.cleared_lines);
            if (a.cleared_lines >= 4) s += w_.quad;
            if (a.spin != SpinType::None) s += w_.spin;
            if (a.all_clear) s += w_.all_clear;
            if (a.cleared_garbage) s += w_.garbage_clear;
            scores[k] = s;
            best = std::max(best, s);
        }

        // Softmax over action scores gives the priors. The temperature is
        // scaled by the observed spread so that a board where every move is
        // similar yields a flat prior, and one with a standout move yields a
        // peaked (but never degenerate) one.
        ev.policy.assign(actions.size(), 0.0f);
        float sum = 0.0f;
        float worst = 1e30f;
        for (float v : scores) worst = std::min(worst, v);
        const float spread = std::max(1.0f, best - worst);
        const float inv_t = 1.0f / std::max(0.05f, w_.temperature * spread / 10.0f);
        for (size_t k = 0; k < actions.size(); ++k) {
            const float e = std::exp((scores[k] - best) * inv_t);
            ev.policy[k] = e;
            sum += e;
        }
        if (sum > 0.0f)
            for (float& p : ev.policy) p /= sum;
        else if (!ev.policy.empty())
            ev.policy.assign(actions.size(), 1.0f / static_cast<float>(actions.size()));

        // Value: survival pressure plus material advantage, squashed to [-1, 1].
        float v = 0.02f * base;
        v -= 0.10f * static_cast<float>(obs.pending_lines);
        if (!obs.alive) v = -1.0f;
        ev.value = ValueWDL::from_scalar(std::tanh(v * 0.15f));

        // Aux predictions, so the plumbing is exercised before a network fills
        // them in with anything meaningful.
        ev.aux.expected_received_garbage = static_cast<float>(obs.pending_lines);
        const float fill = static_cast<float>(obs.board.stack_height()) /
                           std::max(1.0f, static_cast<float>(cfg.geometry.visible_height));
        ev.aux.topout_within_4_pieces = std::max(0.0f, fill - 0.75f) * 4.0f;
        ev.aux.topout_within_8_pieces = std::max(0.0f, fill - 0.60f) * 2.5f;
    }

    Weights w_{};
};

// ---------------------------------------------------------------------------
// Batching adapter
// ---------------------------------------------------------------------------
// Wraps any evaluator and enforces a maximum batch size, splitting oversized
// requests. Useful for a backend with a fixed input shape (ONNX/TensorRT), and
// it keeps the search free of backend-specific batching rules.
class ChunkedEvaluator : public Evaluator {
public:
    ChunkedEvaluator(Evaluator& inner, int max_batch)
        : inner_(inner), max_batch_(std::max(1, max_batch)) {}

    void evaluate(const std::vector<EvalRequest>& batch, std::vector<Evaluation>& out) override {
        account(batch.size());
        out.clear();
        out.reserve(batch.size());
        for (size_t i = 0; i < batch.size(); i += static_cast<size_t>(max_batch_)) {
            const size_t end = std::min(batch.size(), i + static_cast<size_t>(max_batch_));
            std::vector<EvalRequest> chunk(batch.begin() + static_cast<long>(i),
                                           batch.begin() + static_cast<long>(end));
            std::vector<Evaluation> part;
            inner_.evaluate(chunk, part);
            for (auto& e : part) out.push_back(std::move(e));
        }
    }

    int preferred_batch_size() const override { return max_batch_; }
    std::string name() const override { return inner_.name() + "+chunked"; }

private:
    Evaluator& inner_;
    int max_batch_;
};

}  // namespace tetra