// SPDX-License-Identifier: MIT
// Training samples, replay buffer and the self-play loop (spec 12.3, 13.3, 13.5).
#include "test_util.hpp"
#include "tetra/selfplay.hpp"

#include <cmath>
#include <set>

using namespace tetra;

namespace {

RulesetConfig league() { return RulesetConfig::tetra_league(); }

SelfPlayConfig quick_config(int pieces = 40, int sims = 12) {
    SelfPlayConfig cfg;
    cfg.max_pieces = pieces;
    cfg.search.simulations = sims;
    cfg.search.max_depth = 3;
    cfg.search.use_gumbel = true;
    cfg.search.batch_size = 8;
    cfg.garbage_style = GarbageStyle::None;
    return cfg;
}

}  // namespace

TEST(selfplay_records_each_player_placement) {
    HeuristicEvaluator ev;
    SelfPlayWorker w(ev, quick_config());
    SelfPlayStats st;
    const auto samples = w.play(league(), 1, &st);
    CHECK_MSG(samples.size() >= static_cast<size_t>(st.pieces),
              "the recorder must include the active player's placements");
    CHECK(st.pieces > 0);
}

TEST(samples_carry_tokens_actions_and_a_policy) {
    HeuristicEvaluator ev;
    SelfPlayWorker w(ev, quick_config());
    const auto samples = w.play(league(), 7);
    CHECK(!samples.empty());
    for (const auto& s : samples) {
        CHECK(!s.tokens.empty());
        CHECK(!s.action_embeddings.empty());
        CHECK_MSG(s.search_policy.size() == s.action_embeddings.size(),
                  "the policy target must have one entry per legal action");
        CHECK(s.chosen_action >= 0);
        CHECK(s.chosen_action < static_cast<int>(s.action_embeddings.size()));

        float sum = 0.0f;
        for (float v : s.search_policy) {
            CHECK(std::isfinite(v));
            CHECK(v >= 0.0f);
            sum += v;
        }
        CHECK_MSG(std::fabs(sum - 1.0f) < 1e-3f, "policy target must be normalised");

        for (const auto& t : s.tokens)
            for (int k = 0; k < TOKEN_FEATURES; ++k)
                CHECK(std::isfinite(t.f[static_cast<size_t>(k)]));
        for (const auto& a : s.action_embeddings)
            for (int k = 0; k < ACTION_FEATURES; ++k)
                CHECK(std::isfinite(a.f[static_cast<size_t>(k)]));
    }
}

TEST(samples_record_provenance) {
    // Spec 6 / 14: a sample without its ruleset cannot be safely reused.
    HeuristicEvaluator ev;
    SelfPlayConfig cfg = quick_config(20);
    cfg.model_version = 42;
    SelfPlayWorker w(ev, cfg);

    for (const RulesetConfig& rules :
         {RulesetConfig::tetra_league(), RulesetConfig::guideline()}) {
        const auto samples = w.play(rules, 3);
        CHECK(!samples.empty());
        for (const auto& s : samples) {
            CHECK_EQ(s.ruleset_hash, rules.hash());
            CHECK_EQ(static_cast<int>(s.model_version), 42);
        }
        // Move numbers are sequential.
        for (size_t i = 0; i < samples.size(); ++i)
            CHECK_EQ(static_cast<int>(samples[i].move_number), static_cast<int>(i));
    }
}

TEST(outcome_is_the_game_result_only) {
    // Spec 12.3 is explicit: making garbage a negative reward prevents the bot
    // from ever learning a sensible non-cancel. The primary target must depend
    // on win/draw/loss and nothing else.
    HeuristicEvaluator ev;

    // A game under heavy pressure that ends in a top-out.
    SelfPlayConfig hard = quick_config(200, 8);
    hard.garbage_style = GarbageStyle::SlowLarge;
    hard.garbage_period = 2;
    hard.garbage_lines = 4;
    SelfPlayWorker w(ev, hard);
    SelfPlayStats st;
    const auto samples = w.play(league(), 5, &st);

    CHECK(!samples.empty());
    const float z = st.outcome;
    for (const auto& s : samples)
        CHECK_MSG(s.outcome == z || s.outcome == -z,
                  "every sample must use either player's game-result perspective");

    if (!st.survived) {
        CHECK_EQ(z, -1.0f);
        // Garbage received must NOT have leaked into the reward.
        CHECK(st.lines_received > 0);
        for (const auto& s : samples)
            CHECK(s.outcome == -1.0f || s.outcome == 1.0f);
    }
}

TEST(truncated_games_are_draws_not_wins) {
    // Surviving the piece limit is not a victory; scoring it as one would teach
    // the bot that stalling is optimal.
    HeuristicEvaluator ev;
    SelfPlayWorker w(ev, quick_config(30));
    SelfPlayStats st;
    const auto samples = w.play(league(), 11, &st);
    if (st.survived) {
        CHECK_EQ(st.outcome, 0.0f);
        for (const auto& s : samples) CHECK(s.outcome == 0.0f);
    }
}

TEST(auxiliary_targets_are_separate_from_the_reward) {
    // Attack and garbage are auxiliary predictions (spec 10.2), stored on their
    // own fields rather than folded into `outcome`.
    HeuristicEvaluator ev;
    SelfPlayConfig cfg = quick_config(60, 8);
    cfg.garbage_style = GarbageStyle::Steady;
    cfg.garbage_period = 6;
    SelfPlayWorker w(ev, cfg);
    const auto samples = w.play(league(), 2);

    CHECK(!samples.empty());
    bool saw_garbage = false;
    for (const auto& s : samples) {
        CHECK(std::isfinite(s.future_attack_1s));
        CHECK(std::isfinite(s.future_garbage_received));
        CHECK(s.future_attack_1s >= 0.0f);
        CHECK(s.future_garbage_received >= 0.0f);
        if (s.future_garbage_received > 0.0f) saw_garbage = true;
        // The reward stays in {-1, 0, +1}.
        CHECK(s.outcome == -1.0f || s.outcome == 0.0f || s.outcome == 1.0f);
    }
    CHECK_MSG(saw_garbage, "a garbage stream should show up in the auxiliary target");
}

TEST(time_to_terminal_counts_down) {
    HeuristicEvaluator ev;
    SelfPlayWorker w(ev, quick_config(30));
    const auto samples = w.play(league(), 4);
    CHECK(!samples.empty());
    for (size_t i = 0; i + 1 < samples.size(); ++i)
        CHECK_MSG(samples[i].time_to_terminal > samples[i + 1].time_to_terminal,
                  "time to terminal must decrease along a game");
    CHECK_EQ(samples.back().time_to_terminal, 1);
}

TEST(topout_flags_mark_only_the_final_placements) {
    HeuristicEvaluator ev;
    SelfPlayConfig cfg = quick_config(200, 8);
    cfg.garbage_style = GarbageStyle::SlowLarge;
    cfg.garbage_period = 2;
    cfg.garbage_lines = 4;
    SelfPlayWorker w(ev, cfg);
    SelfPlayStats st;
    const auto samples = w.play(league(), 5, &st);

    if (!st.survived && samples.size() > 10) {
        CHECK(samples.back().topped_out_within_4);
        CHECK(samples.back().topped_out_within_8);
        CHECK(!samples.front().topped_out_within_4);
        CHECK(!samples.front().topped_out_within_8);
    }
}

TEST(selfplay_is_reproducible) {
    // Spec 19.4: the same seed must produce the same game and the same samples.
    HeuristicEvaluator ev;
    SelfPlayWorker w(ev, quick_config(30));
    SelfPlayStats sa, sb;
    const auto a = w.play(league(), 1234, &sa);
    const auto b = w.play(league(), 1234, &sb);

    CHECK_EQ(a.size(), b.size());
    CHECK_EQ(sa.pieces, sb.pieces);
    CHECK_EQ(sa.lines_cleared, sb.lines_cleared);
    CHECK_EQ(sa.outcome, sb.outcome);
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        CHECK_EQ(a[i].chosen_action, b[i].chosen_action);
        CHECK_EQ(a[i].search_policy.size(), b[i].search_policy.size());
        for (size_t k = 0; k < a[i].search_policy.size(); ++k)
            CHECK(a[i].search_policy[k] == b[i].search_policy[k]);
    }
}

TEST(different_seeds_give_different_games) {
    HeuristicEvaluator ev;
    SelfPlayWorker w(ev, quick_config(30));
    const auto a = w.play(league(), 1);
    const auto b = w.play(league(), 2);
    bool differs = a.size() != b.size();
    for (size_t i = 0; i < a.size() && i < b.size() && !differs; ++i)
        if (a[i].chosen_action != b[i].chosen_action) differs = true;
    CHECK(differs);
}

TEST(garbage_styles_apply_pressure) {
    // Spec 13.3 lists the attack patterns the curriculum needs.
    HeuristicEvaluator ev;
    for (GarbageStyle style : {GarbageStyle::Steady, GarbageStyle::FastSmall,
                               GarbageStyle::SlowLarge, GarbageStyle::Burst}) {
        SelfPlayConfig cfg = quick_config(60, 8);
        cfg.garbage_style = style;
        cfg.garbage_period = 6;
        SelfPlayWorker w(ev, cfg);
        SelfPlayStats st;
        w.play(league(), 3, &st);
        CHECK_MSG(st.lines_received > 0,
                  "style " + std::to_string(static_cast<int>(style)) +
                      " should deliver garbage");
    }

    SelfPlayConfig none = quick_config(60, 8);
    none.garbage_style = GarbageStyle::None;
    SelfPlayWorker w(ev, none);
    SelfPlayStats st;
    w.play(league(), 3, &st);
    CHECK_EQ(st.lines_received, 0);
}

// ---------------------------------------------------------------------------
// Replay buffer
// ---------------------------------------------------------------------------

TEST(buffer_stores_and_samples) {
    ReplayBuffer buf(1000);
    CHECK(buf.empty());

    HeuristicEvaluator ev;
    SelfPlayWorker w(ev, quick_config(25));
    buf.push_game(w.play(league(), 1));
    CHECK(!buf.empty());
    const size_t n = buf.size();

    Rng rng(1);
    const auto batch = buf.sample(8, rng);
    CHECK_EQ(static_cast<int>(batch.size()), 8);
    for (const auto* s : batch) {
        CHECK(s != nullptr);
        CHECK(!s->tokens.empty());
    }
    CHECK_EQ(buf.size(), n);  // sampling must not consume
}

TEST(buffer_respects_its_capacity) {
    ReplayBuffer buf(50);
    HeuristicEvaluator ev;
    SelfPlayWorker w(ev, quick_config(40));
    for (int i = 0; i < 4; ++i) buf.push_game(w.play(league(), static_cast<std::uint64_t>(i)));
    CHECK_MSG(buf.size() <= 50, "the buffer must not exceed its capacity");
    CHECK(buf.total_added() > 50);
}

TEST(buffer_evicts_oldest_first) {
    ReplayBuffer buf(3);
    for (std::uint32_t i = 0; i < 6; ++i) {
        TrainingSample s;
        s.move_number = i;
        buf.push(std::move(s));
    }
    CHECK_EQ(static_cast<int>(buf.size()), 3);
    CHECK_EQ(static_cast<int>(buf.at(0).move_number), 3);
    CHECK_EQ(static_cast<int>(buf.at(2).move_number), 5);
}

TEST(buffer_sampling_is_reproducible_and_without_replacement) {
    ReplayBuffer buf(200);
    for (std::uint32_t i = 0; i < 100; ++i) {
        TrainingSample s;
        s.move_number = i;
        buf.push(std::move(s));
    }
    Rng a(7), b(7);
    const auto x = buf.sample(20, a);
    const auto y = buf.sample(20, b);
    CHECK_EQ(x.size(), y.size());
    for (size_t i = 0; i < x.size(); ++i) CHECK(x[i]->move_number == y[i]->move_number);

    std::set<std::uint32_t> seen;
    for (const auto* s : x)
        CHECK_MSG(seen.insert(s->move_number).second,
                  "a batch must not contain the same sample twice");
}

TEST(buffer_sampling_handles_edge_cases) {
    ReplayBuffer buf(10);
    Rng rng(1);
    CHECK(buf.sample(5, rng).empty());  // empty buffer

    TrainingSample s;
    buf.push(s);
    CHECK_EQ(static_cast<int>(buf.sample(0, rng).size()), 0);
    // Asking for more than exists returns everything, not garbage.
    CHECK_EQ(static_cast<int>(buf.sample(100, rng).size()), 1);
}

TEST(buffer_can_drop_foreign_rulesets) {
    // Spec 14 forbids mixing samples from different rulesets without an
    // identifier; the buffer must be able to enforce that.
    ReplayBuffer buf(1000);
    const auto league_hash = RulesetConfig::tetra_league().hash();
    const auto guideline_hash = RulesetConfig::guideline().hash();

    for (int i = 0; i < 10; ++i) {
        TrainingSample a;
        a.ruleset_hash = league_hash;
        buf.push(std::move(a));
        TrainingSample b;
        b.ruleset_hash = guideline_hash;
        buf.push(std::move(b));
    }
    CHECK_EQ(static_cast<int>(buf.size()), 20);
    CHECK_EQ(static_cast<int>(buf.ruleset_hashes().size()), 2);

    const size_t dropped = buf.drop_other_rulesets(league_hash);
    CHECK_EQ(static_cast<int>(dropped), 10);
    CHECK_EQ(static_cast<int>(buf.size()), 10);
    CHECK_EQ(static_cast<int>(buf.ruleset_hashes().size()), 1);
    for (size_t i = 0; i < buf.size(); ++i) CHECK_EQ(buf.at(i).ruleset_hash, league_hash);
}

TEST(game_recorder_stamps_the_outcome_on_every_sample) {
    GameRecorder rec(3);
    Tokenizer tok;
    const RulesetConfig cfg = league();
    Player p;
    p.reset(cfg, 1, 0);
    MoveGenerator gen;
    HeuristicEvaluator ev;
    SearchConfig sc;
    sc.simulations = 8;
    sc.max_depth = 2;
    Searcher s(ev, sc);

    for (int i = 0; i < 5 && p.alive(); ++i) {
        const auto acts = gen.generate(p.board(), p.active().type, p.hold(),
                                       p.visible_next().empty() ? Piece::None
                                                                : p.visible_next()[0],
                                       cfg);
        if (acts.empty()) break;
        const Observation obs = observe(p);
        const SearchResult r = s.search(p);
        if (r.best_action < 0) break;
        rec.add(obs, acts, r, tok);
        p.set_active(acts[static_cast<size_t>(r.best_action)].piece_state());
        int sent = 0;
        const LockResult lr = p.lock_piece(20, &sent);
        rec.note_outcome_of_last(sent, lr.garbage_received);
        if (!lr.ok) break;
    }

    const size_t n = rec.size();
    CHECK(n > 0);
    const auto samples = rec.finalize(-1.0f);
    CHECK_EQ(samples.size(), n);
    for (const auto& s2 : samples) {
        CHECK_EQ(s2.outcome, -1.0f);
        CHECK_EQ(static_cast<int>(s2.model_version), 3);
    }
    CHECK_EQ(static_cast<int>(rec.size()), 0);  // finalize drains the recorder
}

TEST(game_recorder_converts_outcome_to_each_sample_perspective) {
    GameRecorder rec;
    Tokenizer tok;
    const RulesetConfig cfg = league();
    Player p;
    p.reset(cfg, 17, 0);
    MoveGenerator gen;
    HeuristicEvaluator ev;
    SearchConfig sc;
    sc.simulations = 8;
    sc.max_depth = 2;
    Searcher s(ev, sc);

    const auto acts = gen.generate(p.board(), p.active().type, p.hold(),
                                   p.visible_next().empty() ? Piece::None
                                                            : p.visible_next()[0],
                                   cfg);
    CHECK(!acts.empty());
    const Observation obs = observe(p);
    const SearchResult r = s.search(p);
    CHECK(r.best_action >= 0);

    rec.add(obs, acts, r, tok, /*value_perspective=*/1);
    rec.note_outcome_of_last(/*attack_sent=*/3, /*garbage_received=*/0);
    rec.add(obs, acts, r, tok, /*value_perspective=*/-1);
    rec.note_outcome_of_last(/*attack_sent=*/5, /*garbage_received=*/0);
    const auto samples = rec.finalize(1.0f);

    CHECK_EQ(samples.size(), static_cast<size_t>(2));
    CHECK_EQ(samples[0].outcome, 1.0f);
    CHECK_EQ(samples[0].n_step_return, 1.0f);
    CHECK_EQ(samples[0].future_attack_1s, 3.0f);
    CHECK_EQ(samples[1].outcome, -1.0f);
    CHECK_EQ(samples[1].n_step_return, -1.0f);
    CHECK_EQ(samples[1].future_attack_1s, 5.0f);
}

TEST(selfplay_feeds_a_buffer_end_to_end) {
    // The whole pipeline: search -> samples -> buffer -> training batch.
    // With this in place, attaching a network is an Evaluator change alone.
    HeuristicEvaluator ev;
    SelfPlayWorker w(ev, quick_config(30, 8));
    ReplayBuffer buf(5000);

    for (int seed = 0; seed < 3; ++seed) buf.push_game(w.play(league(), static_cast<std::uint64_t>(seed)));

    CHECK(buf.size() > 50);
    CHECK_EQ(static_cast<int>(buf.ruleset_hashes().size()), 1);

    Rng rng(5);
    const auto batch = buf.sample(32, rng);
    CHECK_EQ(static_cast<int>(batch.size()), 32);

    // Everything a learner needs must be present and well formed.
    for (const auto* s : batch) {
        CHECK(!s->tokens.empty());
        CHECK_EQ(s->search_policy.size(), s->action_embeddings.size());
        CHECK(s->outcome >= -1.0f && s->outcome <= 1.0f);
        CHECK(std::isfinite(s->search_value));
        CHECK(s->time_to_terminal > 0);
        CHECK(s->ruleset_hash != 0);
    }
}

TEST(selfplay_works_with_a_uniform_evaluator) {
    // An untrained network behaves like the uniform evaluator, so this is the
    // first iteration of the real training loop.
    UniformEvaluator ev;
    SelfPlayWorker w(ev, quick_config(25, 8));
    SelfPlayStats st;
    const auto samples = w.play(league(), 1, &st);
    CHECK(!samples.empty());
    CHECK_MSG(samples.size() >= static_cast<size_t>(st.pieces),
              "the recorder must include the active player's placements");
}

TEST(selfplay_runs_under_every_preset) {
    HeuristicEvaluator ev;
    SelfPlayWorker w(ev, quick_config(20, 8));
    for (const RulesetConfig& rules :
         {RulesetConfig::tetra_league(), RulesetConfig::quick_play(),
          RulesetConfig::guideline()}) {
        SelfPlayStats st;
        const auto samples = w.play(rules, 2, &st);
        CHECK_MSG(!samples.empty(), std::string("no samples under ") + rules.id);
        for (const auto& s : samples) CHECK_EQ(s.ruleset_hash, rules.hash());
    }
}
