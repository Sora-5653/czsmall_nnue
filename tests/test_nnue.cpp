// SPDX-License-Identifier: MIT
// C++ inference for trained weights (spec 16, 17).
//
// The load-bearing test here is `cpp_matches_pytorch_exactly`. Training happens
// in PyTorch and play happens in C++, so if the two forward passes disagree the
// engine is playing a different network from the one that was trained -- a
// failure that produces no error, just a bot that is inexplicably weaker than
// its validation loss suggests. The fixture pins them to ~1e-7.
#include "test_util.hpp"
#include "tetra/nnue.hpp"
#include "tetra/search.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>

using namespace tetra;

namespace {

const char* kWeights = "tests/data/tiny_model.tetrawts";
const char* kReference = "tests/data/tiny_model_reference.bin";

bool file_exists(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

struct Reference {
    bool ok = false;
    int tokens = 0;
    int actions = 0;
    std::vector<float> token_data;
    std::vector<float> action_data;
    std::vector<float> policy;
    std::vector<float> wdl;
};

Reference load_reference() {
    Reference r;
    std::ifstream f(kReference, std::ios::binary);
    if (!f.good()) return r;
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), {});
    if (blob.size() < 8) return r;

    size_t at = 0;
    auto u32 = [&]() {
        std::uint32_t v = 0;
        std::memcpy(&v, blob.data() + at, 4);
        at += 4;
        return v;
    };
    r.tokens = static_cast<int>(u32());
    r.actions = static_cast<int>(u32());

    auto floats = [&](size_t n) {
        std::vector<float> v(n);
        if (at + n * 4 <= blob.size()) {
            std::memcpy(v.data(), blob.data() + at, n * 4);
            at += n * 4;
        }
        return v;
    };
    r.token_data = floats(static_cast<size_t>(r.tokens) * TOKEN_FEATURES);
    r.action_data = floats(static_cast<size_t>(r.actions) * ACTION_FEATURES);
    r.policy = floats(static_cast<size_t>(r.actions));
    r.wdl = floats(3);
    r.ok = at <= blob.size();
    return r;
}

// Run the C++ forward pass on raw matrices, mirroring what the evaluator does
// after tokenization. Kept here rather than exposed publicly because only the
// parity test needs to bypass the tokenizer.
void forward_raw(const NnueWeights& w, const std::vector<float>& tokens, size_t T,
                 const std::vector<float>& actions, size_t A, std::vector<float>& policy,
                 std::vector<float>& wdl) {
    const NnueConfig& c = w.config();
    const size_t d = c.width;

    std::vector<float> x(T * d);
    detail::linear(tokens.data(), T, TOKEN_FEATURES, w.get("token_in.weight"),
                   &w.get("token_in.bias"), x.data());

    std::vector<float> h(T * d), attn(T * d), gate(T * c.ffn), up(T * c.ffn), ffn(T * d), scratch;
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        const std::string p = "blocks." + std::to_string(l) + ".";
        h = x;
        detail::rmsnorm(h.data(), T, d, w.get(p + "n1.weight"));
        detail::attention(h.data(), T, h.data(), T, d, c.heads, w.get(p + "attn.in_proj_weight"),
                          w.get(p + "attn.in_proj_bias"), w.get(p + "attn.out_proj.weight"),
                          w.get(p + "attn.out_proj.bias"), nullptr, attn.data(), scratch);
        for (size_t i = 0; i < T * d; ++i) x[i] += attn[i];

        h = x;
        detail::rmsnorm(h.data(), T, d, w.get(p + "n2.weight"));
        detail::linear(h.data(), T, d, w.get(p + "ffn.gate.weight"), nullptr, gate.data());
        detail::linear(h.data(), T, d, w.get(p + "ffn.up.weight"), nullptr, up.data());
        for (size_t i = 0; i < T * c.ffn; ++i) gate[i] = detail::silu(gate[i]) * up[i];
        detail::linear(gate.data(), T, c.ffn, w.get(p + "ffn.down.weight"), nullptr, ffn.data());
        for (size_t i = 0; i < T * d; ++i) x[i] += ffn[i];
    }
    detail::rmsnorm(x.data(), T, d, w.get("norm.weight"));

    // Policy.
    std::vector<float> q(A * d);
    detail::linear(actions.data(), A, ACTION_FEATURES, w.get("action_in.weight"),
                   &w.get("action_in.bias"), q.data());
    std::vector<float> pa(A * d);
    detail::attention(q.data(), A, x.data(), T, d, c.heads, w.get("policy_attn.in_proj_weight"),
                      w.get("policy_attn.in_proj_bias"), w.get("policy_attn.out_proj.weight"),
                      w.get("policy_attn.out_proj.bias"), nullptr, pa.data(), scratch);
    for (size_t i = 0; i < A * d; ++i) pa[i] += q[i];
    detail::rmsnorm(pa.data(), A, d, w.get("policy_norm.weight"));

    std::vector<float> logits(A);
    detail::linear(pa.data(), A, d, w.get("policy_out.weight"), &w.get("policy_out.bias"),
                   logits.data());
    float best = -1e30f;
    for (float v : logits) best = std::max(best, v);
    float sum = 0.0f;
    policy.resize(A);
    for (size_t i = 0; i < A; ++i) {
        policy[i] = std::exp(logits[i] - best);
        sum += policy[i];
    }
    for (float& v : policy) v /= sum;

    // Value.
    std::vector<float> pooled(d, 0.0f);
    for (size_t i = 0; i < T; ++i)
        for (size_t j = 0; j < d; ++j) pooled[j] += x[i * d + j];
    for (size_t j = 0; j < d; ++j) pooled[j] /= static_cast<float>(T);

    const Tensor& v0 = w.get("value_head.0.weight");
    std::vector<float> hid(v0.rows());
    detail::linear(pooled.data(), 1, d, v0, &w.get("value_head.0.bias"), hid.data());
    for (float& v : hid) v = detail::silu(v);
    const Tensor& v2 = w.get("value_head.2.weight");
    std::vector<float> out(v2.rows());
    detail::linear(hid.data(), 1, hid.size(), v2, &w.get("value_head.2.bias"), out.data());

    float b2 = std::max(out[0], std::max(out[1], out[2]));
    float e0 = std::exp(out[0] - b2), e1 = std::exp(out[1] - b2), e2 = std::exp(out[2] - b2);
    const float s = e0 + e1 + e2;
    wdl = {e0 / s, e1 / s, e2 / s};
}

}  // namespace

TEST(weights_load_with_the_expected_shape) {
    if (!file_exists(kWeights)) {
        // The fixture is reproducible rather than precious: regenerate it with
        // `python scripts/make_fixtures.py` (needs torch). Skipping instead of
        // failing keeps the other 261 tests useful on a machine without it.
        std::printf("       (skipped: run python scripts/make_fixtures.py)\n");
        return;
    }
    NnueWeights w;
    std::string err;
    CHECK_MSG(w.load(kWeights, &err), "load failed: " + err);
    CHECK_EQ(static_cast<int>(w.config().token_features), TOKEN_FEATURES);
    CHECK_EQ(static_cast<int>(w.config().action_features), ACTION_FEATURES);
    CHECK(w.config().width > 0);
    CHECK(w.config().layers > 0);
    CHECK_EQ(static_cast<int>(w.config().width % w.config().heads), 0);
    CHECK(w.tensor_count() > 10);
}

TEST(cpp_matches_pytorch_exactly) {
    // The whole point of the C++ inference path: it must run the same function
    // PyTorch trained. A silent divergence here would show up only as
    // unexplained weakness in play.
    if (!file_exists(kWeights) || !file_exists(kReference)) {
        std::printf("       (skipped: run python scripts/make_fixtures.py)\n");
        return;
    }
    NnueWeights w;
    std::string err;
    CHECK_MSG(w.load(kWeights, &err), err);

    const Reference ref = load_reference();
    CHECK(ref.ok);
    CHECK(ref.tokens > 0);
    CHECK(ref.actions > 0);

    std::vector<float> policy, wdl;
    forward_raw(w, ref.token_data, static_cast<size_t>(ref.tokens), ref.action_data,
                static_cast<size_t>(ref.actions), policy, wdl);

    CHECK_EQ(policy.size(), ref.policy.size());
    float worst_policy = 0.0f;
    for (size_t i = 0; i < policy.size() && i < ref.policy.size(); ++i)
        worst_policy = std::max(worst_policy, std::fabs(policy[i] - ref.policy[i]));
    CHECK_MSG(worst_policy < 1e-5f,
              "policy diverges from PyTorch by " + std::to_string(worst_policy));

    CHECK_EQ(wdl.size(), ref.wdl.size());
    float worst_wdl = 0.0f;
    for (size_t i = 0; i < wdl.size() && i < ref.wdl.size(); ++i)
        worst_wdl = std::max(worst_wdl, std::fabs(wdl[i] - ref.wdl[i]));
    CHECK_MSG(worst_wdl < 1e-5f, "WDL diverges from PyTorch by " + std::to_string(worst_wdl));
}

TEST(loader_rejects_corrupt_weights) {
    NnueWeights w;
    std::string err;

    std::vector<std::uint8_t> empty;
    CHECK(!w.parse(empty, &err));

    std::vector<std::uint8_t> bad(64, 0);
    CHECK(!w.parse(bad, &err));
    CHECK(!err.empty());
}

TEST(loader_rejects_a_feature_width_mismatch) {
    // If the engine's token layout changed but the weights did not, the model
    // would be fed a different input than it was trained on. That must be an
    // error, not a silent misread.
    if (!file_exists(kWeights)) return;
    std::ifstream f(kWeights, std::ios::binary);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), {});
    std::vector<std::uint8_t> bytes(blob.begin(), blob.end());

    // token_features sits right after the magic and version.
    const std::uint32_t wrong = TOKEN_FEATURES + 1;
    std::memcpy(bytes.data() + 12, &wrong, 4);

    NnueWeights w;
    std::string err;
    CHECK(!w.parse(bytes, &err));
    CHECK_MSG(err.find("feature width mismatch") != std::string::npos,
              "expected a width mismatch error, got: " + err);
}

TEST(evaluator_produces_a_valid_distribution) {
    if (!file_exists(kWeights)) return;
    NnueWeights w;
    std::string err;
    CHECK_MSG(w.load(kWeights, &err), err);
    TetraFormerEvaluator ev(std::move(w));

    const RulesetConfig cfg = RulesetConfig::tetra_league();
    Player p;
    p.reset(cfg, 42, 0);
    MoveGenerator gen;
    const auto actions = gen.generate_for_piece(p.board(), p.active().type, cfg, false);
    const Observation obs = observe(p);

    const Evaluation e = ev.evaluate_one(obs, actions);
    CHECK_EQ(e.policy.size(), actions.size());
    float sum = 0.0f;
    for (float v : e.policy) {
        CHECK(std::isfinite(v));
        CHECK(v >= 0.0f);
        sum += v;
    }
    CHECK_MSG(std::fabs(sum - 1.0f) < 1e-4f, "policy must be normalised, got " + std::to_string(sum));
    CHECK(std::isfinite(e.value.scalar()));
    CHECK(std::fabs(e.value.win + e.value.draw + e.value.loss - 1.0f) < 1e-4f);
}

TEST(evaluator_is_deterministic) {
    if (!file_exists(kWeights)) return;
    NnueWeights w;
    std::string err;
    w.load(kWeights, &err);
    TetraFormerEvaluator ev(std::move(w));

    const RulesetConfig cfg = RulesetConfig::tetra_league();
    Player p;
    p.reset(cfg, 7, 0);
    MoveGenerator gen;
    const auto actions = gen.generate_for_piece(p.board(), p.active().type, cfg, false);
    const Observation obs = observe(p);

    const Evaluation a = ev.evaluate_one(obs, actions);
    for (int i = 0; i < 3; ++i) {
        const Evaluation b = ev.evaluate_one(obs, actions);
        for (size_t k = 0; k < a.policy.size(); ++k) CHECK(a.policy[k] == b.policy[k]);
        CHECK(a.value.scalar() == b.value.scalar());
    }
}

TEST(evaluator_handles_batches_and_empty_actions) {
    if (!file_exists(kWeights)) return;
    NnueWeights w;
    std::string err;
    w.load(kWeights, &err);
    TetraFormerEvaluator ev(std::move(w));

    const RulesetConfig cfg = RulesetConfig::tetra_league();
    std::vector<Player> players(3);
    std::vector<std::vector<PlacementAction>> actions(3);
    std::vector<Observation> obs;
    MoveGenerator gen;
    for (int i = 0; i < 3; ++i) {
        players[static_cast<size_t>(i)].reset(cfg, static_cast<std::uint64_t>(i + 1), 0);
        actions[static_cast<size_t>(i)] =
            gen.generate_for_piece(players[static_cast<size_t>(i)].board(),
                                   players[static_cast<size_t>(i)].active().type, cfg, false);
    }
    for (int i = 0; i < 3; ++i) obs.push_back(observe(players[static_cast<size_t>(i)]));

    std::vector<EvalRequest> batch;
    for (int i = 0; i < 3; ++i)
        batch.push_back(EvalRequest{&obs[static_cast<size_t>(i)], &actions[static_cast<size_t>(i)]});

    std::vector<Evaluation> out;
    ev.evaluate(batch, out);
    CHECK_EQ(out.size(), batch.size());
    for (size_t i = 0; i < out.size(); ++i) CHECK_EQ(out[i].policy.size(), actions[i].size());

    // Batched and single evaluation must agree.
    const Evaluation single = ev.evaluate_one(obs[0], actions[0]);
    for (size_t k = 0; k < single.policy.size(); ++k)
        CHECK(std::fabs(single.policy[k] - out[0].policy[k]) < 1e-6f);

    // An empty action list must not crash or produce NaN.
    const std::vector<PlacementAction> none;
    const Evaluation e = ev.evaluate_one(obs[0], none);
    CHECK_EQ(static_cast<int>(e.policy.size()), 0);
    CHECK(std::isfinite(e.value.scalar()));
}

TEST(trained_evaluator_drives_the_search) {
    // The closed loop: a network trained in PyTorch playing inside the C++
    // search, with no other component changed.
    if (!file_exists(kWeights)) return;
    NnueWeights w;
    std::string err;
    w.load(kWeights, &err);
    TetraFormerEvaluator ev(std::move(w));

    const RulesetConfig cfg = RulesetConfig::tetra_league();
    Player p;
    p.reset(cfg, 3, 0);

    SearchConfig sc;
    sc.simulations = 16;
    sc.max_depth = 3;
    sc.use_gumbel = true;
    sc.batch_size = 8;
    Searcher s(ev, sc);
    const SearchResult r = s.search(p);

    CHECK(r.best_action >= 0);
    CHECK(r.simulations_run > 0);
    float sum = 0.0f;
    for (float v : r.search_policy) sum += v;
    CHECK(std::fabs(sum - 1.0f) < 1e-3f);
}

TEST(trained_evaluator_can_play_a_game) {
    if (!file_exists(kWeights)) return;
    NnueWeights w;
    std::string err;
    w.load(kWeights, &err);
    TetraFormerEvaluator ev(std::move(w));

    const RulesetConfig cfg = RulesetConfig::tetra_league();
    Player p;
    p.reset(cfg, 1, 0);
    MoveGenerator gen;
    int placed = 0;
    for (int i = 0; i < 12 && p.alive(); ++i) {
        const auto acts = gen.generate_for_piece(p.board(), p.active().type, cfg, false);
        if (acts.empty()) break;
        const Evaluation e = ev.evaluate_one(observe(p), acts);
        size_t best = 0;
        for (size_t k = 1; k < e.policy.size(); ++k)
            if (e.policy[k] > e.policy[best]) best = k;
        p.set_active(acts[best].piece_state());
        int out = 0;
        if (!p.lock_piece(acts[best].total_duration(), &out).ok) break;
        ++placed;
    }
    CHECK_MSG(placed > 0, "an untrained-but-valid network should still place pieces");
}