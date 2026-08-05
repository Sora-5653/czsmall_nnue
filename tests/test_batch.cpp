// SPDX-License-Identifier: MIT
// Fixed-shape batching and dataset export (spec 9, 10.1, 13.5, 17).
//
// This layer is where ragged engine data becomes rectangular tensors. The
// failure modes it guards against are quiet ones: padding that is not zeroed,
// masks that disagree with the data, and policy mass landing on a padded slot.
// None of those crash; they just train a worse model.
#include "test_util.hpp"
#include "tetra/dataset.hpp"
#include "tetra/selfplay.hpp"

#include <cmath>
#include <cstdio>
#include <set>

using namespace tetra;

namespace {

RulesetConfig league() { return RulesetConfig::tetra_league(); }

std::vector<TrainingSample> make_samples(int games = 2, int pieces = 25) {
    HeuristicEvaluator ev;
    SelfPlayConfig cfg;
    cfg.max_pieces = pieces;
    cfg.search.simulations = 8;
    cfg.search.max_depth = 3;
    cfg.garbage_style = GarbageStyle::Steady;
    SelfPlayWorker w(ev, cfg);
    std::vector<TrainingSample> all;
    for (int g = 0; g < games; ++g) {
        auto s = w.play(league(), static_cast<std::uint64_t>(g));
        for (auto& x : s) all.push_back(std::move(x));
    }
    return all;
}

std::vector<const TrainingSample*> pointers(const std::vector<TrainingSample>& v) {
    std::vector<const TrainingSample*> p;
    p.reserve(v.size());
    for (const auto& s : v) p.push_back(&s);
    return p;
}

}  // namespace

TEST(training_batch_is_rectangular) {
    const auto samples = make_samples();
    CHECK(!samples.empty());
    const auto ptrs = pointers(samples);
    const TensorBatch b = make_training_batch(ptrs);

    CHECK_EQ(b.batch, static_cast<int>(samples.size()));
    CHECK(b.max_tokens > 0);
    CHECK(b.max_actions > 0);

    const size_t bt = static_cast<size_t>(b.batch) * static_cast<size_t>(b.max_tokens);
    const size_t ba = static_cast<size_t>(b.batch) * static_cast<size_t>(b.max_actions);
    CHECK_EQ(b.tokens.size(), bt * TOKEN_FEATURES);
    CHECK_EQ(b.token_mask.size(), bt);
    CHECK_EQ(b.actions.size(), ba * ACTION_FEATURES);
    CHECK_EQ(b.action_mask.size(), ba);
    CHECK_EQ(b.policy_target.size(), ba);
    CHECK_EQ(b.value_target.size(), static_cast<size_t>(b.batch));
    CHECK_EQ(b.aux_target.size(), static_cast<size_t>(b.batch) * TensorBatch::AUX_TARGETS);
}

TEST(batch_dimensions_cover_the_widest_sample) {
    const auto samples = make_samples(3, 30);
    const auto ptrs = pointers(samples);
    const TensorBatch b = make_training_batch(ptrs);

    size_t widest_tokens = 0, widest_actions = 0;
    for (const auto* s : ptrs) {
        widest_tokens = std::max(widest_tokens, s->tokens.size());
        widest_actions = std::max(widest_actions, s->action_embeddings.size());
    }
    CHECK_EQ(static_cast<size_t>(b.max_tokens), widest_tokens);
    CHECK_EQ(static_cast<size_t>(b.max_actions), widest_actions);
}

TEST(masks_match_the_real_content) {
    const auto samples = make_samples();
    const auto ptrs = pointers(samples);
    const TensorBatch b = make_training_batch(ptrs);

    for (int i = 0; i < b.batch; ++i) {
        const size_t n_tokens = ptrs[static_cast<size_t>(i)]->tokens.size();
        const size_t n_actions = ptrs[static_cast<size_t>(i)]->action_embeddings.size();
        for (int t = 0; t < b.max_tokens; ++t) {
            const float m = b.token_mask[static_cast<size_t>(i) * static_cast<size_t>(b.max_tokens) +
                                         static_cast<size_t>(t)];
            CHECK_EQ(m, static_cast<size_t>(t) < n_tokens ? 1.0f : 0.0f);
        }
        for (int a = 0; a < b.max_actions; ++a) {
            const float m = b.action_mask[static_cast<size_t>(i) * static_cast<size_t>(b.max_actions) +
                                          static_cast<size_t>(a)];
            CHECK_EQ(m, static_cast<size_t>(a) < n_actions ? 1.0f : 0.0f);
        }
        CHECK_EQ(static_cast<size_t>(b.action_count[static_cast<size_t>(i)]), n_actions);
    }
}

TEST(padding_is_exactly_zero) {
    // A model that forgot its mask would still appear to work if padding held
    // stale data, then fail on differently shaped batches.
    const auto samples = make_samples();
    const auto ptrs = pointers(samples);
    const TensorBatch b = make_training_batch(ptrs);

    for (int i = 0; i < b.batch; ++i) {
        const size_t n_tokens = ptrs[static_cast<size_t>(i)]->tokens.size();
        for (int t = static_cast<int>(n_tokens); t < b.max_tokens; ++t)
            for (int k = 0; k < TOKEN_FEATURES; ++k) {
                const size_t idx = static_cast<size_t>(i) * b.token_stride() +
                                   static_cast<size_t>(t) * TOKEN_FEATURES + static_cast<size_t>(k);
                CHECK_MSG(b.tokens[idx] == 0.0f, "padded token slots must be zero");
            }

        const size_t n_actions = ptrs[static_cast<size_t>(i)]->action_embeddings.size();
        for (int a = static_cast<int>(n_actions); a < b.max_actions; ++a)
            for (int k = 0; k < ACTION_FEATURES; ++k) {
                const size_t idx = static_cast<size_t>(i) * b.action_stride() +
                                   static_cast<size_t>(a) * ACTION_FEATURES + static_cast<size_t>(k);
                CHECK_MSG(b.actions[idx] == 0.0f, "padded action slots must be zero");
            }
    }
}

TEST(token_data_survives_padding_intact) {
    const auto samples = make_samples(1, 15);
    const auto ptrs = pointers(samples);
    const TensorBatch b = make_training_batch(ptrs);

    for (int i = 0; i < b.batch; ++i) {
        const auto& src = ptrs[static_cast<size_t>(i)]->tokens;
        for (size_t t = 0; t < src.size(); ++t)
            for (int k = 0; k < TOKEN_FEATURES; ++k) {
                const size_t idx = static_cast<size_t>(i) * b.token_stride() +
                                   t * TOKEN_FEATURES + static_cast<size_t>(k);
                CHECK(b.tokens[idx] == src[t].f[static_cast<size_t>(k)]);
            }
    }
}

TEST(policy_rows_are_distributions_over_real_actions) {
    const auto samples = make_samples();
    const auto ptrs = pointers(samples);
    const TensorBatch b = make_training_batch(ptrs);

    for (int i = 0; i < b.batch; ++i) {
        const size_t row = static_cast<size_t>(i) * static_cast<size_t>(b.max_actions);
        float sum = 0.0f;
        for (int a = 0; a < b.max_actions; ++a) {
            const float p = b.policy_target[row + static_cast<size_t>(a)];
            CHECK(std::isfinite(p));
            CHECK(p >= 0.0f);
            sum += p;
            // Padding must never carry policy mass.
            if (b.action_mask[row + static_cast<size_t>(a)] == 0.0f)
                CHECK_MSG(p == 0.0f, "policy mass assigned to a padded action");
        }
        CHECK_MSG(std::fabs(sum - 1.0f) < 1e-4f,
                  "policy row must sum to 1, got " + std::to_string(sum));
    }
}

TEST(value_targets_are_the_game_result) {
    const auto samples = make_samples();
    const auto ptrs = pointers(samples);
    const TensorBatch b = make_training_batch(ptrs);
    for (int i = 0; i < b.batch; ++i) {
        const float v = b.value_target[static_cast<size_t>(i)];
        CHECK_MSG(v == -1.0f || v == 0.0f || v == 1.0f,
                  "value target must be a game result, got " + std::to_string(v));
        CHECK_EQ(v, ptrs[static_cast<size_t>(i)]->outcome);
    }
}

TEST(aux_targets_are_normalised) {
    // Raw counts would dominate an MSE loss purely by scale: an unsquashed
    // `time_to_terminal` produced an auxiliary loss of ~150 against a policy
    // loss of ~3, so the model fitted almost nothing else.
    const auto samples = make_samples(3, 40);
    const auto ptrs = pointers(samples);
    const TensorBatch b = make_training_batch(ptrs);

    for (size_t i = 0; i < b.aux_target.size(); ++i) {
        const float v = b.aux_target[i];
        CHECK(std::isfinite(v));
        CHECK_MSG(v >= 0.0f && v <= 1.0f,
                  "auxiliary target out of [0,1]: " + std::to_string(v));
    }
}

TEST(explicit_padding_produces_a_static_shape) {
    // A fixed-shape backend (ONNX/TensorRT) needs the same dimensions every
    // call regardless of what the position happens to contain.
    const auto samples = make_samples(2, 20);
    const auto ptrs = pointers(samples);
    const TensorBatch b = make_training_batch(ptrs, /*pad_tokens=*/128, /*pad_actions=*/96);
    CHECK_EQ(b.max_tokens, 128);
    CHECK_EQ(b.max_actions, 96);
    CHECK_EQ(b.tokens.size(),
             static_cast<size_t>(b.batch) * 128 * TOKEN_FEATURES);
    // Content is unchanged; only the trailing padding grew.
    for (int i = 0; i < b.batch; ++i) {
        const size_t n = ptrs[static_cast<size_t>(i)]->tokens.size();
        CHECK(b.token_mask[static_cast<size_t>(i) * 128] == (n > 0 ? 1.0f : 0.0f));
        CHECK_EQ(b.token_mask[static_cast<size_t>(i) * 128 + 127], 0.0f);
    }
}

TEST(inference_batch_matches_the_live_position) {
    // The inference path must produce the same tensors the trainer sees,
    // otherwise the network is fed one thing and trained on another.
    const RulesetConfig cfg = league();
    Player p;
    p.reset(cfg, 5, 0);
    MoveGenerator gen;
    const auto actions = gen.generate_for_piece(p.board(), p.active().type, cfg, false);
    const Observation obs = observe(p);

    std::vector<EvalRequest> reqs{EvalRequest{&obs, &actions}};
    Tokenizer tok;
    const TensorBatch b = make_inference_batch(reqs, tok);

    CHECK_EQ(b.batch, 1);
    CHECK_EQ(static_cast<size_t>(b.action_count[0]), actions.size());

    // The tokens must equal a direct encoding of the same observation.
    const TokenizedObservation direct = tok.encode(obs, cfg);
    CHECK_EQ(static_cast<size_t>(b.max_tokens), direct.tokens.size());
    for (size_t t = 0; t < direct.tokens.size(); ++t)
        for (int k = 0; k < TOKEN_FEATURES; ++k)
            CHECK(b.tokens[t * TOKEN_FEATURES + static_cast<size_t>(k)] ==
                  direct.tokens[t].f[static_cast<size_t>(k)]);
}

TEST(inference_batch_handles_mixed_sizes) {
    // Positions in one batch have different token and action counts; the
    // shorter ones must be padded, not truncated.
    const RulesetConfig cfg = league();
    MoveGenerator gen;
    std::vector<Player> players;
    std::vector<std::vector<PlacementAction>> actions;
    std::vector<Observation> observations;

    for (std::uint64_t seed = 1; seed <= 4; ++seed) {
        Player p;
        p.reset(cfg, seed, 0);
        for (int i = 0; i < static_cast<int>(seed) * 5 && p.alive(); ++i) {
            const auto a = gen.generate_for_piece(p.board(), p.active().type, cfg, false);
            if (a.empty()) break;
            p.set_active(a[0].piece_state());
            int out = 0;
            if (!p.lock_piece(a[0].total_duration(), &out).ok) break;
        }
        players.push_back(std::move(p));
    }
    for (auto& p : players) {
        actions.push_back(gen.generate_for_piece(p.board(), p.active().type, cfg, false));
        observations.push_back(observe(p));
    }

    std::vector<EvalRequest> reqs;
    for (size_t i = 0; i < players.size(); ++i)
        reqs.push_back(EvalRequest{&observations[i], &actions[i]});

    Tokenizer tok;
    const TensorBatch b = make_inference_batch(reqs, tok);
    CHECK_EQ(b.batch, static_cast<int>(players.size()));
    for (size_t i = 0; i < actions.size(); ++i)
        CHECK_EQ(static_cast<size_t>(b.action_count[i]), actions[i].size());
}

TEST(empty_batches_are_safe) {
    Tokenizer tok;
    const TensorBatch a = make_inference_batch({}, tok);
    CHECK_EQ(a.batch, 0);
    const TensorBatch b = make_training_batch({});
    CHECK_EQ(b.batch, 0);
}

// ---------------------------------------------------------------------------
// Dataset export
// ---------------------------------------------------------------------------

TEST(dataset_round_trips) {
    const auto samples = make_samples(2, 25);
    const auto ptrs = pointers(samples);
    const TensorBatch batch = make_training_batch(ptrs);

    const auto bytes = serialize_dataset(batch, 0xABCDEF12ull, 7);
    const DatasetReadResult r = deserialize_dataset(bytes);
    CHECK_MSG(r.ok, "deserialize failed: " + r.error);

    CHECK_EQ(static_cast<int>(r.header.samples), batch.batch);
    CHECK_EQ(static_cast<int>(r.header.max_tokens), batch.max_tokens);
    CHECK_EQ(static_cast<int>(r.header.max_actions), batch.max_actions);
    CHECK_EQ(static_cast<int>(r.header.token_features), TOKEN_FEATURES);
    CHECK_EQ(static_cast<int>(r.header.action_features), ACTION_FEATURES);
    CHECK_EQ(r.header.ruleset_hash, 0xABCDEF12ull);
    CHECK_EQ(static_cast<int>(r.header.model_version), 7);

    CHECK(r.batch.tokens == batch.tokens);
    CHECK(r.batch.token_mask == batch.token_mask);
    CHECK(r.batch.actions == batch.actions);
    CHECK(r.batch.action_mask == batch.action_mask);
    CHECK(r.batch.policy_target == batch.policy_target);
    CHECK(r.batch.value_target == batch.value_target);
    CHECK(r.batch.aux_target == batch.aux_target);
}

TEST(dataset_rejects_corruption) {
    const auto samples = make_samples(1, 15);
    const auto batch = make_training_batch(pointers(samples));
    const auto good = serialize_dataset(batch, 1, 0);

    std::vector<std::uint8_t> bad_magic = good;
    bad_magic[0] = 'X';
    CHECK(!deserialize_dataset(bad_magic).ok);

    std::vector<std::uint8_t> truncated(good.begin(), good.begin() + static_cast<long>(good.size() / 2));
    const auto r = deserialize_dataset(truncated);
    CHECK(!r.ok);
    CHECK(r.error == "truncated payload");

    std::vector<std::uint8_t> tiny{'T', 'E', 'T', 'R'};
    CHECK(!deserialize_dataset(tiny).ok);
}

TEST(dataset_file_round_trips) {
    const auto samples = make_samples(2, 20);
    const auto batch = make_training_batch(pointers(samples));
    const std::string path = "build/test_dataset.tetradat";
    CHECK(write_dataset_file(path, batch, 42, 3));

    const DatasetReadResult r = read_dataset_file(path);
    CHECK_MSG(r.ok, "read failed: " + r.error);
    CHECK_EQ(static_cast<int>(r.header.samples), batch.batch);
    CHECK_EQ(r.header.ruleset_hash, 42ull);
    CHECK(r.batch.policy_target == batch.policy_target);
    std::remove(path.c_str());
}

TEST(missing_dataset_file_is_reported) {
    const auto r = read_dataset_file("build/no_such_dataset.tetradat");
    CHECK(!r.ok);
    CHECK(!r.error.empty());
}

TEST(buffer_exports_end_to_end) {
    // The complete handover: self-play -> buffer -> file the trainer reads.
    HeuristicEvaluator ev;
    SelfPlayConfig cfg;
    cfg.max_pieces = 25;
    cfg.search.simulations = 8;
    cfg.search.max_depth = 3;
    SelfPlayWorker w(ev, cfg);
    ReplayBuffer buf(10000);
    for (int g = 0; g < 2; ++g) buf.push_game(w.play(league(), static_cast<std::uint64_t>(g)));
    CHECK(buf.size() > 10);

    const std::string path = "build/test_export.tetradat";
    CHECK(export_buffer(path, buf, /*model_version=*/2));

    const DatasetReadResult r = read_dataset_file(path);
    CHECK_MSG(r.ok, "export could not be read back: " + r.error);
    CHECK_EQ(static_cast<size_t>(r.header.samples), buf.size());
    CHECK_EQ(r.header.ruleset_hash, league().hash());
    CHECK_EQ(static_cast<int>(r.header.model_version), 2);

    // Everything a learner needs is present and well formed.
    for (size_t i = 0; i < r.header.samples; ++i) {
        float sum = 0.0f;
        for (std::uint32_t a = 0; a < r.header.max_actions; ++a)
            sum += r.batch.policy_target[i * r.header.max_actions + a];
        CHECK(std::fabs(sum - 1.0f) < 1e-4f);
    }
    std::remove(path.c_str());
}

TEST(exporting_an_empty_buffer_fails_cleanly) {
    ReplayBuffer empty(10);
    CHECK(!export_buffer("build/should_not_exist.tetradat", empty));
}

TEST(batch_construction_is_deterministic) {
    const auto samples = make_samples(2, 20);
    const auto ptrs = pointers(samples);
    const TensorBatch a = make_training_batch(ptrs);
    const TensorBatch b = make_training_batch(ptrs);
    CHECK(a.tokens == b.tokens);
    CHECK(a.actions == b.actions);
    CHECK(a.policy_target == b.policy_target);
    CHECK(a.value_target == b.value_target);
}

TEST(two_player_dataset_preserves_opponent_tokens_in_rectangular_format) {
    HeuristicEvaluator ev;
    SelfPlayConfig cfg;
    cfg.max_pieces = 30;
    cfg.search.simulations = 8;
    cfg.search.max_depth = 3;
    SelfPlayWorker w(ev, cfg);
    ReplayBuffer buf(1000);
    buf.push_game(w.play(league(), 42));
    CHECK(buf.size() > 5);

    const std::string v1_path = "build/test_v1.tetradat";
    const std::string v2_path = "build/test_v2.tetradat";
    CHECK(export_buffer(v1_path, buf, 3, 0, 0, /*compact=*/false));
    CHECK(export_buffer(v2_path, buf, 3, 0, 0, /*compact=*/true));

    const DatasetReadResult r1 = read_dataset_file(v1_path);
    const DatasetReadResult r2 = read_dataset_file(v2_path);
    CHECK_MSG(r1.ok, "read v1 failed: " + r1.error);
    CHECK_MSG(r2.ok, "read v2 failed: " + r2.error);

    CHECK_EQ(static_cast<int>(r2.header.samples), static_cast<int>(r1.header.samples));
    // Compact Replay+ cannot reconstruct the opponent's board/event stream;
    // requesting compact therefore falls back to rectangular v1 so the
    // already-masked observation is kept exactly.
    CHECK_EQ(static_cast<int>(r2.header.version), static_cast<int>(DatasetHeader::VERSION));
    CHECK_EQ(r2.header.ruleset_hash, r1.header.ruleset_hash);

    CHECK(r2.batch.tokens == r1.batch.tokens);
    CHECK(r2.batch.token_mask == r1.batch.token_mask);
    CHECK(r2.batch.actions == r1.batch.actions);
    CHECK(r2.batch.action_mask == r1.batch.action_mask);
    CHECK(r2.batch.policy_target == r1.batch.policy_target);
    CHECK(r2.batch.value_target == r1.batch.value_target);
    CHECK(r2.batch.aux_target == r1.batch.aux_target);

    std::FILE* f1 = std::fopen(v1_path.c_str(), "rb");
    std::fseek(f1, 0, SEEK_END);
    const long sz1 = std::ftell(f1);
    std::fclose(f1);

    std::FILE* f2 = std::fopen(v2_path.c_str(), "rb");
    std::fseek(f2, 0, SEEK_END);
    const long sz2 = std::ftell(f2);
    std::fclose(f2);

    CHECK(sz2 == sz1);

    std::remove(v1_path.c_str());
    std::remove(v2_path.c_str());
}
