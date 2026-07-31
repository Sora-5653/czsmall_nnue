// SPDX-License-Identifier: MIT
// The evaluator interface (spec 10, 11.1, 19.4).
//
// These tests pin the contract the search is about to be built on top of. The
// contract matters more than any particular implementation: the network is
// ~100x more expensive than the rest of the engine, so if the interface does
// not make batching natural, the search will have to be rewritten when real
// weights arrive.
#include "test_util.hpp"
#include "tetra/evaluator.hpp"

#include <cmath>
#include <numeric>
#include <set>

using namespace tetra;

namespace {

RulesetConfig league() { return RulesetConfig::tetra_league(); }

// A position with a non-trivial board, so heuristics have something to say.
struct Fixture {
    Player player;
    std::vector<PlacementAction> actions;
    Observation obs;

    explicit Fixture(std::uint64_t seed = 42, int warmup = 20) {
        const RulesetConfig cfg = league();
        player.reset(cfg, seed, 0);
        MoveGenerator gen;
        for (int i = 0; i < warmup && player.alive(); ++i) {
            const auto a = gen.generate_for_piece(player.board(), player.active().type, cfg, false);
            if (a.empty()) break;
            size_t best = 0;
            for (size_t k = 1; k < a.size(); ++k)
                if (a[k].final_y < a[best].final_y) best = k;
            player.set_active(a[best].piece_state());
            int out = 0;
            if (!player.lock_piece(a[best].total_duration(), &out).ok) break;
        }
        actions = gen.generate_for_piece(player.board(), player.active().type, cfg, false);
        obs = observe(player);
    }
};

void check_valid_distribution(const std::vector<float>& p, size_t expected_n,
                              const std::string& who) {
    CHECK_MSG(p.size() == expected_n,
              who + ": policy length must match the action count (" +
                  std::to_string(p.size()) + " vs " + std::to_string(expected_n) + ")");
    float sum = 0.0f;
    for (float v : p) {
        CHECK_MSG(std::isfinite(v), who + ": prior must be finite");
        CHECK_MSG(v >= 0.0f, who + ": prior must be non-negative");
        sum += v;
    }
    if (expected_n > 0)
        CHECK_MSG(std::fabs(sum - 1.0f) < 1e-4f,
                  who + ": priors must sum to 1, got " + std::to_string(sum));
}

}  // namespace

TEST(uniform_evaluator_returns_a_flat_distribution) {
    Fixture f;
    UniformEvaluator e;
    const Evaluation ev = e.evaluate_one(f.obs, f.actions);
    check_valid_distribution(ev.policy, f.actions.size(), "uniform");
    for (float p : ev.policy)
        CHECK(std::fabs(p - 1.0f / static_cast<float>(f.actions.size())) < 1e-6f);
    CHECK_EQ(ev.value.scalar(), 0.0f);
}

TEST(heuristic_evaluator_returns_a_valid_distribution) {
    HeuristicEvaluator e;
    for (std::uint64_t seed : {1ull, 7ull, 42ull, 999ull}) {
        Fixture f(seed);
        if (f.actions.empty()) continue;
        const Evaluation ev = e.evaluate_one(f.obs, f.actions);
        check_valid_distribution(ev.policy, f.actions.size(), "heuristic");
        CHECK(std::isfinite(ev.value.scalar()));
        CHECK(ev.value.scalar() >= -1.0f && ev.value.scalar() <= 1.0f);
    }
}

TEST(wdl_normalises_and_maps_to_scalar) {
    ValueWDL v;
    v.win = 2.0f;
    v.draw = 1.0f;
    v.loss = 1.0f;
    v.normalize();
    CHECK(std::fabs(v.win + v.draw + v.loss - 1.0f) < 1e-6f);
    CHECK(std::fabs(v.win - 0.5f) < 1e-6f);
    CHECK(std::fabs(v.scalar() - 0.25f) < 1e-6f);

    // Degenerate input becomes a pure draw rather than NaN.
    ValueWDL z;
    z.normalize();
    CHECK(std::fabs(z.draw - 1.0f) < 1e-6f);
    CHECK(std::isfinite(z.scalar()));

    // from_scalar round-trips and clamps.
    for (float s : {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f}) {
        const ValueWDL w = ValueWDL::from_scalar(s);
        CHECK(std::fabs(w.scalar() - s) < 1e-6f);
        CHECK(std::fabs(w.win + w.draw + w.loss - 1.0f) < 1e-6f);
    }
    CHECK(std::fabs(ValueWDL::from_scalar(5.0f).scalar() - 1.0f) < 1e-6f);
    CHECK(std::fabs(ValueWDL::from_scalar(-5.0f).scalar() + 1.0f) < 1e-6f);
}

TEST(batched_and_single_evaluation_agree) {
    // The single-position helper is implemented in terms of the batch call, so
    // there is exactly one code path. This test guards that invariant: if a
    // future backend special-cases batch size 1, the results must not drift.
    HeuristicEvaluator e;
    std::vector<Fixture> fixtures;
    for (std::uint64_t seed = 1; seed <= 6; ++seed) fixtures.emplace_back(seed);

    std::vector<EvalRequest> batch;
    for (const auto& f : fixtures)
        if (!f.actions.empty()) batch.push_back(EvalRequest{&f.obs, &f.actions});
    CHECK(batch.size() >= 4);

    std::vector<Evaluation> batched;
    e.evaluate(batch, batched);
    CHECK_EQ(batched.size(), batch.size());

    for (size_t i = 0; i < batch.size(); ++i) {
        const Evaluation single =
            e.evaluate_one(*batch[i].observation, *batch[i].actions);
        CHECK_EQ(single.policy.size(), batched[i].policy.size());
        for (size_t k = 0; k < single.policy.size(); ++k)
            CHECK_MSG(std::fabs(single.policy[k] - batched[i].policy[k]) < 1e-6f,
                      "batched and single evaluation must agree");
        CHECK(std::fabs(single.value.scalar() - batched[i].value.scalar()) < 1e-6f);
    }
}

TEST(evaluation_is_deterministic) {
    // Spec 19.4: the same seed must reproduce exactly. A stochastic evaluator
    // would make the whole search non-reproducible.
    HeuristicEvaluator e;
    Fixture f(31337);
    const Evaluation a = e.evaluate_one(f.obs, f.actions);
    for (int i = 0; i < 5; ++i) {
        const Evaluation b = e.evaluate_one(f.obs, f.actions);
        CHECK_EQ(a.policy.size(), b.policy.size());
        for (size_t k = 0; k < a.policy.size(); ++k) CHECK(a.policy[k] == b.policy[k]);
        CHECK(a.value.scalar() == b.value.scalar());
    }
}

TEST(evaluator_handles_an_empty_action_list) {
    // A blocked-out position has no legal moves. The evaluator must return an
    // empty policy rather than crash or produce NaN.
    Fixture f;
    const std::vector<PlacementAction> none;
    for (Evaluator* e : std::initializer_list<Evaluator*>{
             new UniformEvaluator(), new HeuristicEvaluator()}) {
        const Evaluation ev = e->evaluate_one(f.obs, none);
        CHECK_EQ(static_cast<int>(ev.policy.size()), 0);
        CHECK(std::isfinite(ev.value.scalar()));
        delete e;
    }
}

TEST(evaluator_handles_an_empty_batch) {
    HeuristicEvaluator e;
    std::vector<EvalRequest> empty;
    std::vector<Evaluation> out;
    e.evaluate(empty, out);
    CHECK_EQ(static_cast<int>(out.size()), 0);
}

TEST(evaluator_tolerates_null_requests) {
    // Defensive: a search bug that leaves a slot unfilled must not corrupt
    // memory, it must produce a neutral evaluation.
    HeuristicEvaluator e;
    std::vector<EvalRequest> batch{EvalRequest{nullptr, nullptr}};
    std::vector<Evaluation> out;
    e.evaluate(batch, out);
    CHECK_EQ(static_cast<int>(out.size()), 1);
    CHECK(std::isfinite(out[0].value.scalar()));
}

TEST(policy_length_always_matches_the_action_list) {
    // The variable-length policy head (spec 10.1) is the whole reason the
    // interface takes actions rather than a fixed grid. Vary the board width
    // and the action count and confirm the invariant holds.
    HeuristicEvaluator e;
    for (int width : {6, 8, 10, 12}) {
        RulesetConfig cfg = league();
        cfg.geometry.width = width;
        Player p;
        p.reset(cfg, 5, 0);
        MoveGenerator gen;
        const auto actions = gen.generate_for_piece(p.board(), p.active().type, cfg, false);
        const Observation obs = observe(p);
        const Evaluation ev = e.evaluate_one(obs, actions);
        check_valid_distribution(ev.policy, actions.size(),
                                 "width " + std::to_string(width));
    }
}

TEST(heuristic_prefers_clearing_lines_over_stacking) {
    // The stand-in must have genuine opinions, otherwise it is not a useful
    // baseline for measuring whether search helps.
    const RulesetConfig cfg = league();
    Player p;
    p.reset(cfg, 1, 0);
    // Leave row 0 one cell short so some placements clear it.
    Board& b = p.mutable_board();
    for (int x = 0; x < 9; ++x) b.fill_cell(x, 0, false);

    MoveGenerator gen;
    const auto actions = gen.generate_for_piece(b, p.active().type, cfg, false);
    const Observation obs = observe(p);
    HeuristicEvaluator e;
    const Evaluation ev = e.evaluate_one(obs, actions);

    float clearing = 0.0f, other = 0.0f;
    int n_clear = 0;
    for (size_t k = 0; k < actions.size(); ++k) {
        if (actions[k].cleared_lines > 0) {
            clearing += ev.policy[k];
            ++n_clear;
        } else {
            other += ev.policy[k];
        }
    }
    if (n_clear > 0) {
        const float mean_clear = clearing / static_cast<float>(n_clear);
        const float mean_other =
            other / static_cast<float>(std::max<size_t>(1, actions.size() - n_clear));
        CHECK_MSG(mean_clear > mean_other,
                  "a line clear should attract more prior mass than an average move");
    }
}

TEST(heuristic_dislikes_creating_holes) {
    const RulesetConfig cfg = league();
    Player p;
    p.reset(cfg, 3, 0);
    MoveGenerator gen;
    const auto actions = gen.generate_for_piece(p.board(), p.active().type, cfg, false);
    const Observation obs = observe(p);
    HeuristicEvaluator e;
    const Evaluation ev = e.evaluate_one(obs, actions);

    // Compare the prior mass on hole-free placements against hole-making ones.
    float clean_max = 0.0f, holey_max = 0.0f;
    for (size_t k = 0; k < actions.size(); ++k) {
        const PlacementOutcome oc = evaluate_placement(obs.board, actions[k].piece_state(), cfg);
        if (oc.board.hole_count() > obs.board.hole_count())
            holey_max = std::max(holey_max, ev.policy[k]);
        else
            clean_max = std::max(clean_max, ev.policy[k]);
    }
    if (holey_max > 0.0f && clean_max > 0.0f)
        CHECK_MSG(clean_max > holey_max,
                  "the best hole-free move should outrank the best hole-making one");
}

TEST(heuristic_is_a_credible_baseline) {
    // Spec 19.1 measures the bot against a heuristic baseline, so the baseline
    // has to actually play. Greedy over the priors must survive a long game.
    const RulesetConfig cfg = league();
    HeuristicEvaluator e;
    MoveGenerator gen;

    int survived = 0, total_pieces = 0;
    const int games = 5, limit = 300;
    for (int seed = 0; seed < games; ++seed) {
        Player p;
        p.reset(cfg, static_cast<std::uint64_t>(seed), 0);
        int i = 0;
        for (; i < limit && p.alive(); ++i) {
            const auto acts = gen.generate_for_piece(p.board(), p.active().type, cfg, false);
            if (acts.empty()) break;
            const Evaluation ev = e.evaluate_one(observe(p), acts);
            size_t best = 0;
            for (size_t k = 1; k < ev.policy.size(); ++k)
                if (ev.policy[k] > ev.policy[best]) best = k;
            p.set_active(acts[best].piece_state());
            int out = 0;
            if (!p.lock_piece(acts[best].total_duration(), &out).ok) break;
        }
        total_pieces += i;
        if (p.alive()) ++survived;
    }
    const double mean = static_cast<double>(total_pieces) / games;
    CHECK_MSG(mean > 200.0,
              "the heuristic baseline should survive a long game; mean " +
                  std::to_string(mean) + " pieces");
    CHECK_MSG(survived >= 3, "most baseline games should reach the piece limit");
}

TEST(evaluator_counts_batch_statistics) {
    // Spec 19.4 tracks the batched-leaf-evaluation rate, so the interface has
    // to make that measurable.
    HeuristicEvaluator e;
    e.reset_stats();
    CHECK_EQ(static_cast<int>(e.positions_evaluated()), 0);

    Fixture f;
    e.evaluate_one(f.obs, f.actions);
    CHECK_EQ(static_cast<int>(e.positions_evaluated()), 1);
    CHECK_EQ(static_cast<int>(e.batches_issued()), 1);

    std::vector<EvalRequest> batch(8, EvalRequest{&f.obs, &f.actions});
    std::vector<Evaluation> out;
    e.evaluate(batch, out);
    CHECK_EQ(static_cast<int>(e.positions_evaluated()), 9);
    CHECK_EQ(static_cast<int>(e.batches_issued()), 2);
    CHECK(e.mean_batch_size() > 4.0);
}

TEST(chunked_evaluator_splits_without_changing_results) {
    // A fixed-shape backend (ONNX/TensorRT) needs bounded batches. Splitting
    // must be invisible to the caller.
    HeuristicEvaluator inner;
    ChunkedEvaluator chunked(inner, /*max_batch=*/3);

    std::vector<Fixture> fixtures;
    for (std::uint64_t seed = 1; seed <= 8; ++seed) fixtures.emplace_back(seed);
    std::vector<EvalRequest> batch;
    for (const auto& f : fixtures)
        if (!f.actions.empty()) batch.push_back(EvalRequest{&f.obs, &f.actions});

    std::vector<Evaluation> direct, split;
    inner.evaluate(batch, direct);
    chunked.evaluate(batch, split);

    CHECK_EQ(direct.size(), split.size());
    for (size_t i = 0; i < direct.size() && i < split.size(); ++i) {
        CHECK_EQ(direct[i].policy.size(), split[i].policy.size());
        for (size_t k = 0; k < direct[i].policy.size(); ++k)
            CHECK(std::fabs(direct[i].policy[k] - split[i].policy[k]) < 1e-6f);
        CHECK(std::fabs(direct[i].value.scalar() - split[i].value.scalar()) < 1e-6f);
    }
    CHECK_EQ(chunked.preferred_batch_size(), 3);
}

TEST(evaluators_report_a_usable_batch_preference) {
    UniformEvaluator u;
    HeuristicEvaluator h;
    CHECK(u.preferred_batch_size() > 0);
    CHECK(h.preferred_batch_size() > 0);
    CHECK(!u.name().empty());
    CHECK(!h.name().empty());
    CHECK(u.name() != h.name());
}

TEST(observation_carries_its_ruleset) {
    // The evaluator must be self-contained: pairing an observation with the
    // wrong RulesetConfig would silently mis-score every position.
    for (const RulesetConfig& cfg :
         {RulesetConfig::tetra_league(), RulesetConfig::quick_play(),
          RulesetConfig::guideline()}) {
        Player p;
        p.reset(cfg, 1, 0);
        const Observation obs = observe(p);
        CHECK_EQ(obs.ruleset_hash, cfg.hash());
        CHECK_EQ(obs.ruleset.hash(), cfg.hash());
        CHECK_EQ(obs.ruleset.id, cfg.id);
    }
}

TEST(aux_predictions_are_finite) {
    HeuristicEvaluator e;
    for (std::uint64_t seed : {1ull, 5ull, 50ull}) {
        Fixture f(seed, 40);
        if (f.actions.empty()) continue;
        const Evaluation ev = e.evaluate_one(f.obs, f.actions);
        const AuxPredictions& a = ev.aux;
        CHECK(std::isfinite(a.expected_time_to_ko));
        CHECK(std::isfinite(a.topout_within_4_pieces));
        CHECK(std::isfinite(a.topout_within_8_pieces));
        CHECK(std::isfinite(a.expected_received_garbage));
        CHECK(a.expected_received_garbage >= 0.0f);
    }
}

TEST(danger_lowers_the_value) {
    // A near-death board must evaluate worse than a clean one. Without this
    // the search has no reason to survive.
    const RulesetConfig cfg = league();
    HeuristicEvaluator e;
    MoveGenerator gen;

    Player safe;
    safe.reset(cfg, 1, 0);
    const auto safe_actions = gen.generate_for_piece(safe.board(), safe.active().type, cfg, false);
    const float safe_v = e.evaluate_one(observe(safe), safe_actions).value.scalar();

    Player danger;
    danger.reset(cfg, 1, 0);
    Board& b = danger.mutable_board();
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 9; ++x) b.fill_cell(x, y, false);
    const auto danger_actions =
        gen.generate_for_piece(danger.board(), danger.active().type, cfg, false);
    const float danger_v = e.evaluate_one(observe(danger), danger_actions).value.scalar();

    CHECK_MSG(danger_v < safe_v,
              "a tall stack must evaluate worse: " + std::to_string(danger_v) + " vs " +
                  std::to_string(safe_v));
}

TEST(pending_garbage_lowers_the_value) {
    const RulesetConfig cfg = league();
    HeuristicEvaluator e;
    MoveGenerator gen;

    Player p;
    p.reset(cfg, 9, 0);
    const auto actions = gen.generate_for_piece(p.board(), p.active().type, cfg, false);
    const float clean = e.evaluate_one(observe(p), actions).value.scalar();

    p.receive_attack(6, 0, 1);
    const float threatened = e.evaluate_one(observe(p), actions).value.scalar();
    CHECK_MSG(threatened < clean, "pending garbage must reduce the value");
}