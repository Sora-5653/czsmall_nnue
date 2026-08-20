// SPDX-License-Identifier: MIT
#include "test_util.hpp"

#include "tetra/human_replay.hpp"

#include <numeric>

using namespace tetra;

namespace {

HumanReplayState empty_state(const RulesetConfig& rules, Piece current,
                             Piece hold, std::vector<Piece> queue) {
    HumanReplayState state;
    state.board = Board(rules.geometry.width, rules.geometry.internal_height);
    state.current = current;
    state.hold = hold;
    state.queue = std::move(queue);
    state.combo = -1;
    state.b2b = 0;
    state.valid = true;
    return state;
}

HumanReplayGame one_turn_game(const RulesetConfig& rules,
                              std::vector<std::string> keys) {
    HumanReplayGame game;
    game.source_id = "unit-test";
    game.round_index = 3;
    game.outcome = {1.0f, -1.0f};
    game.initial[0] = empty_state(rules, Piece::T, Piece::I,
                                  {Piece::O, Piece::S, Piece::Z, Piece::J, Piece::L});
    game.initial[1] = empty_state(rules, Piece::L, Piece::None,
                                  {Piece::J, Piece::S, Piece::Z, Piece::O, Piece::I});
    game.turns.push_back(HumanReplayTurn{12, 0, std::move(keys)});
    return game;
}

}  // namespace

TEST(human_replay_hard_drop_becomes_one_hot_policy) {
    const RulesetConfig rules = RulesetConfig::tetra_league();
    std::vector<TrainingSample> samples;
    const HumanReplayImportStats stats = build_human_replay_samples(
        {one_turn_game(rules, {"hardDrop"})}, rules, 7, samples);

    CHECK_EQ(stats.games, static_cast<std::size_t>(1));
    CHECK_EQ(stats.turns, static_cast<std::size_t>(1));
    CHECK_EQ(stats.imported, static_cast<std::size_t>(1));
    CHECK_EQ(stats.skipped_execution, static_cast<std::size_t>(0));
    CHECK_EQ(stats.skipped_no_legal_match, static_cast<std::size_t>(0));
    CHECK_EQ(samples.size(), static_cast<std::size_t>(1));

    const TrainingSample& sample = samples.front();
    CHECK(sample.chosen_action >= 0);
    CHECK(sample.chosen_action < static_cast<int>(sample.search_policy.size()));
    CHECK_EQ(sample.search_policy[static_cast<std::size_t>(sample.chosen_action)], 1.0f);
    CHECK_EQ(std::accumulate(sample.search_policy.begin(), sample.search_policy.end(), 0.0f), 1.0f);
    CHECK_EQ(sample.outcome, 1.0f);
    CHECK_EQ(sample.model_version, static_cast<std::uint32_t>(7));
    CHECK_EQ(sample.ruleset_hash, rules.hash());
    CHECK_EQ(sample.aux_target_schema_version, schema::AUX_TARGET_SCHEMA_VERSION);
    CHECK(std::all_of(sample.aux_valid.begin(), sample.aux_valid.end(),
                      [](std::uint8_t value) { return value == 0; }));
}

TEST(human_replay_hold_is_matched_against_hold_action) {
    const RulesetConfig rules = RulesetConfig::tetra_league();
    std::vector<TrainingSample> samples;
    const HumanReplayImportStats stats = build_human_replay_samples(
        {one_turn_game(rules, {"hold", "hardDrop"})}, rules, 0, samples);

    CHECK_EQ(stats.imported, static_cast<std::size_t>(1));
    CHECK_EQ(samples.size(), static_cast<std::size_t>(1));
    const TrainingSample& sample = samples.front();
    CHECK(sample.chosen_action >= 0);

    const HumanReplayGame game = one_turn_game(rules, {"hold", "hardDrop"});
    const HumanReplayState& state = game.initial[0];
    MoveGenerator movegen;
    const auto actions = movegen.generate(state.board, state.current, state.hold,
                                          state.queue.front(), rules);
    CHECK(sample.chosen_action < static_cast<int>(actions.size()));
    CHECK(actions[static_cast<std::size_t>(sample.chosen_action)].use_hold);
    CHECK_EQ(actions[static_cast<std::size_t>(sample.chosen_action)].final_piece, Piece::I);
}

TEST(exact_same_piece_hold_canonicalizes_to_no_hold_action) {
    const RulesetConfig rules = RulesetConfig::tetra_league();
    HumanReplayGame game = one_turn_game(rules, {});
    game.turns.clear();
    game.initial[0] = empty_state(
        rules, Piece::T, Piece::T,
        {Piece::O, Piece::S, Piece::Z, Piece::J, Piece::L});

    MoveGenerator movegen;
    const HumanReplayState& before = game.initial[0];
    const auto actions = movegen.generate(
        before.board, before.current, before.hold, before.queue.front(), rules);
    CHECK(!actions.empty());
    CHECK(std::none_of(actions.begin(), actions.end(),
                       [](const PlacementAction& action) { return action.use_hold; }));

    const PlacementAction& target = actions.front();
    const PlacementOutcome target_outcome =
        evaluate_placement(before.board, target.piece_state(), rules);

    HumanReplayTurn turn;
    turn.frame = 12;
    turn.player = 0;
    turn.exact = true;
    turn.exact_before = before;
    turn.exact_after = before;
    turn.exact_placement_board = target_outcome.board;
    turn.exact_used_hold = true;
    turn.exact_final_piece = Piece::T;
    turn.exact_final_x = target.final_x + 1;
    turn.exact_final_y = target.final_y + 1;
    turn.exact_final_rotation = target.final_rotation;
    game.turns.push_back(std::move(turn));

    std::vector<TrainingSample> samples;
    const HumanReplayImportStats stats =
        build_human_replay_samples({game}, rules, 0, samples);
    CHECK_EQ(stats.imported, static_cast<std::size_t>(1));
    CHECK_EQ(stats.skipped_no_legal_match, static_cast<std::size_t>(0));
    CHECK_EQ(samples.size(), static_cast<std::size_t>(1));
    CHECK(samples.front().chosen_action >= 0);
    CHECK(!actions[static_cast<std::size_t>(samples.front().chosen_action)].use_hold);
}

TEST(human_replay_missing_hard_drop_is_rejected_not_fabricated) {
    const RulesetConfig rules = RulesetConfig::tetra_league();
    std::vector<TrainingSample> samples;
    const HumanReplayImportStats stats = build_human_replay_samples(
        {one_turn_game(rules, {"moveLeft", "rotateCW"})}, rules, 0, samples);

    CHECK_EQ(stats.imported, static_cast<std::size_t>(0));
    CHECK_EQ(stats.skipped_execution, static_cast<std::size_t>(1));
    CHECK(samples.empty());
}
