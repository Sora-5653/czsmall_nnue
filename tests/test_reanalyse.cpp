// SPDX-License-Identifier: MIT
#include "test_util.hpp"
#include "tetra/reanalyse.hpp"
#include "tetra/selfplay.hpp"

using namespace tetra;

TEST(reanalyse_reconstructs_exact_dual_player_historical_roots) {
    UniformEvaluator evaluator;
    SelfPlayConfig cfg;
    cfg.max_pieces = 12;
    cfg.search.simulations = 2;
    cfg.search.batch_size = 2;
    cfg.search.root_noise_fraction = 0.0f;
    cfg.garbage_style = GarbageStyle::None;
    SelfPlayWorker worker(evaluator, cfg);
    const RulesetConfig rules = RulesetConfig::tetra_league();
    std::vector<TrainingSample> samples = worker.play(rules, 424242);
    CHECK(samples.size() >= 4);
    std::vector<const TrainingSample*> ptrs;
    for (const TrainingSample& sample : samples) ptrs.push_back(&sample);
    const TensorBatch batch = make_training_batch(ptrs);
    const ReconstructionReport report = reconstruct_historical_roots(batch, rules, false);
    CHECK(report.ok());
    CHECK_EQ(report.roots.size(), samples.size());
    CHECK_EQ(report.token_rows_verified, samples.size());
    CHECK_EQ(report.action_rows_verified, samples.size());
}

TEST(reanalyse_output_refreshes_policy_while_preserving_trajectory_labels) {
    UniformEvaluator evaluator;
    SelfPlayConfig cfg;
    cfg.max_pieces = 8;
    cfg.search.simulations = 2;
    cfg.garbage_style = GarbageStyle::None;
    const RulesetConfig rules = RulesetConfig::tetra_league();
    SelfPlayWorker worker(evaluator, cfg);
    std::vector<TrainingSample> samples = worker.play(rules, 98765);
    CHECK(samples.size() >= 3);
    std::vector<const TrainingSample*> ptrs;
    for (const TrainingSample& sample : samples) ptrs.push_back(&sample);
    const TensorBatch source = make_training_batch(ptrs);
    const std::vector<std::size_t> rows{1, 2};
    std::vector<std::vector<float>> policies;
    for (std::size_t row : rows) {
        const int count = source.action_count[row];
        std::vector<float> policy(static_cast<std::size_t>(count), 0.0f);
        policy.back() = 1.0f;
        policies.push_back(std::move(policy));
    }
    const TensorBatch refreshed = select_reanalysed_rows(source, rows, policies);
    CHECK_EQ(refreshed.batch, 2);
    for (std::size_t dst = 0; dst < rows.size(); ++dst) {
        const std::size_t src = rows[dst];
        CHECK_EQ(refreshed.game_seed[dst], source.game_seed[src]);
        CHECK_EQ(refreshed.move_number[dst], source.move_number[src]);
        CHECK_EQ(refreshed.chosen_action[dst], source.chosen_action[src]);
        CHECK_EQ(refreshed.value_target[dst], source.value_target[src]);
    }
}

TEST(reanalyse_rejects_missing_chosen_action_provenance) {
    TensorBatch source;
    detail::allocate_batch(source, 1, 1, 1, true);
    source.game_seed[0] = 1;
    source.move_number[0] = 0;
    source.chosen_action.clear();
    const ReconstructionReport report = reconstruct_historical_roots(
        source, RulesetConfig::tetra_league(), false);
    CHECK(!report.ok());
    CHECK(report.error.find("v4 replay provenance") != std::string::npos);
}
