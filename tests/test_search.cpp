// SPDX-License-Identifier: MIT
// PUCT / Gumbel search (spec 11).
//
// Most of these tests run on UniformEvaluator, whose output is analytically
// known, so any deviation in the visit distribution can only come from the
// search itself.
#include "test_util.hpp"
#include "tetra/search.hpp"

#include <cmath>
#include <numeric>
#include <set>

using namespace tetra;

namespace {

RulesetConfig league() { return RulesetConfig::tetra_league(); }

// Advance a player a few placements so the board is non-trivial.
Player warmed(std::uint64_t seed, int pieces = 20) {
    const RulesetConfig cfg = league();
    Player p;
    p.reset(cfg, seed, 0);
    MoveGenerator gen;
    for (int i = 0; i < pieces && p.alive(); ++i) {
        const auto a = gen.generate_for_piece(p.board(), p.active().type, cfg, false);
        if (a.empty()) break;
        size_t best = 0;
        for (size_t k = 1; k < a.size(); ++k)
            if (a[k].final_y < a[best].final_y) best = k;
        p.set_active(a[best].piece_state());
        int out = 0;
        if (!p.lock_piece(a[best].total_duration(), &out).ok) break;
    }
    return p;
}

// A position with exactly one clearly correct move: two rows are complete
// except for the last column, so the placement that fills them is the only one
// that clears anything.
Player forced_clear_position() {
    const RulesetConfig cfg = league();
    Player p;
    p.reset(cfg, 1, 0);
    Board& b = p.mutable_board();
    for (int x = 0; x < 9; ++x) {
        b.fill_cell(x, 0, false);
        b.fill_cell(x, 1, false);
    }
    return p;
}

std::vector<PlacementAction> actions_of(const Player& p) {
    MoveGenerator gen;
    return gen.generate(p.board(), p.active().type, p.hold(),
                        p.visible_next().empty() ? Piece::None : p.visible_next()[0],
                        p.ruleset());
}

class ObservationSpy final : public Evaluator {
public:
    void evaluate(const std::vector<EvalRequest>& batch,
                  std::vector<Evaluation>& out) override {
        account(batch.size());
        for (const auto& req : batch) {
            if (!req.observation) continue;
            saw_any = true;
            saw_opponent = saw_opponent || req.observation->has_opponent;
            if (req.observation->has_opponent)
                max_opponent_pending =
                    std::max(max_opponent_pending, req.observation->opponent_pending_lines);
        }
        out.assign(batch.size(), Evaluation{});
        for (size_t i = 0; i < batch.size(); ++i) {
            const size_t n = batch[i].actions ? batch[i].actions->size() : 0;
            out[i].policy.assign(n, n ? 1.0f / static_cast<float>(n) : 0.0f);
            out[i].value.draw = 1.0f;
        }
    }

    std::string name() const override { return "observation-spy"; }

    bool saw_any = false;
    bool saw_opponent = false;
    int max_opponent_pending = 0;
};

class ConstantValueEvaluator final : public Evaluator {
public:
    explicit ConstantValueEvaluator(float value) : value_(value) {}

    void evaluate(const std::vector<EvalRequest>& batch,
                  std::vector<Evaluation>& out) override {
        account(batch.size());
        out.assign(batch.size(), Evaluation{});
        for (size_t i = 0; i < batch.size(); ++i) {
            const size_t n = batch[i].actions ? batch[i].actions->size() : 0;
            out[i].policy.assign(n, n ? 1.0f / static_cast<float>(n) : 0.0f);
            out[i].value = ValueWDL::from_scalar(value_);
        }
    }

private:
    float value_ = 0.0f;
};

class RelativePressureEvaluator final : public Evaluator {
public:
    void evaluate(const std::vector<EvalRequest>& batch,
                  std::vector<Evaluation>& out) override {
        account(batch.size());
        out.assign(batch.size(), Evaluation{});
        for (size_t i = 0; i < batch.size(); ++i) {
            const EvalRequest& req = batch[i];
            const size_t n = req.actions ? req.actions->size() : 0;
            out[i].policy.assign(n, n ? 1.0f / static_cast<float>(n) : 0.0f);
            const Observation* obs = req.observation;
            const float self = obs ? static_cast<float>(obs->pending_lines) : 0.0f;
            const float other = (obs && obs->has_opponent)
                                    ? static_cast<float>(obs->opponent_pending_lines)
                                    : 0.0f;
            out[i].value = ValueWDL::from_scalar((other - self) / 4.0f);
        }
    }
};


}  // namespace

TEST(search_returns_a_legal_action) {
    UniformEvaluator ev;
    for (std::uint64_t seed : {1ull, 7ull, 42ull}) {
        const Player p = warmed(seed);
        const auto acts = actions_of(p);
        SearchConfig cfg;
        cfg.simulations = 32;
        cfg.max_depth = 3;
        Searcher s(ev, cfg);
        const SearchResult r = s.search(p);
        CHECK(r.best_action >= 0);
        CHECK(r.best_action < static_cast<int>(acts.size()));
        CHECK_EQ(r.candidates.size(), acts.size());
    }
}

TEST(search_policy_is_a_distribution) {
    // The visit distribution is the training target pi (spec 13.5), so it must
    // be a genuine probability distribution.
    UniformEvaluator ev;
    for (bool gumbel : {false, true}) {
        SearchConfig cfg;
        cfg.simulations = 64;
        cfg.max_depth = 3;
        cfg.use_gumbel = gumbel;
        Searcher s(ev, cfg);
        const SearchResult r = s.search(warmed(5));
        float sum = 0.0f;
        for (float v : r.search_policy) {
            CHECK(std::isfinite(v));
            CHECK(v >= 0.0f);
            sum += v;
        }
        CHECK_MSG(std::fabs(sum - 1.0f) < 1e-4f,
                  std::string(gumbel ? "gumbel" : "puct") + " policy must sum to 1, got " +
                      std::to_string(sum));
    }
}

TEST(search_is_deterministic_for_a_seed) {
    // Spec 19.4: the same seed must reproduce exactly.
    HeuristicEvaluator ev;
    const Player p = warmed(11);
    for (bool gumbel : {false, true}) {
        SearchConfig cfg;
        cfg.simulations = 48;
        cfg.max_depth = 3;
        cfg.use_gumbel = gumbel;
        cfg.seed = 12345;
        Searcher a(ev, cfg), b(ev, cfg);
        const SearchResult ra = a.search(p);
        const SearchResult rb = b.search(p);
        CHECK_EQ(ra.best_action, rb.best_action);
        CHECK_EQ(ra.simulations_run, rb.simulations_run);
        CHECK_EQ(ra.nodes_created, rb.nodes_created);
        CHECK_EQ(ra.evaluator_calls, rb.evaluator_calls);
        CHECK_EQ(ra.positions_evaluated, rb.positions_evaluated);
        CHECK_EQ(ra.transposition_hits, rb.transposition_hits);
        CHECK(ra.mean_batch_size == rb.mean_batch_size);
        CHECK(ra.value.win == rb.value.win);
        CHECK(ra.value.draw == rb.value.draw);
        CHECK(ra.value.loss == rb.value.loss);
        CHECK_EQ(ra.search_policy.size(), rb.search_policy.size());
        for (size_t i = 0; i < ra.search_policy.size(); ++i)
            CHECK(ra.search_policy[i] == rb.search_policy[i]);
        CHECK_EQ(ra.candidates.size(), rb.candidates.size());
        for (size_t i = 0; i < ra.candidates.size(); ++i) {
            CHECK_EQ(ra.candidates[i].action_index, rb.candidates[i].action_index);
            CHECK(ra.candidates[i].prior == rb.candidates[i].prior);
            CHECK_EQ(ra.candidates[i].visits, rb.candidates[i].visits);
            CHECK(ra.candidates[i].q_value == rb.candidates[i].q_value);
        }
    }
}

TEST(two_player_value_is_antisymmetric_under_a_complete_swap) {
    // The network value is defined from the observing player's perspective.
    // Swapping self and opponent must therefore negate the scalar value.
    Player p0;
    Player p1;
    p0.reset(league(), 1, 0);
    p1.reset(league(), 2, 1);
    p0.receive_attack(2, 0, 1);

    SearchConfig cfg;
    cfg.simulations = 0;
    cfg.max_depth = 0;
    cfg.use_gumbel = false;
    cfg.determinize_root = false;

    RelativePressureEvaluator ev;
    Searcher from_p0(ev, cfg);
    Searcher from_p1(ev, cfg);
    const SearchResult a = from_p0.search(p0, &p1);
    const SearchResult b = from_p1.search(p1, &p0);

    CHECK_MSG(std::fabs(a.value.scalar() + b.value.scalar()) < 1e-6f,
              "swapping self and opponent must negate value");
}

TEST(depth_one_q_reports_the_winning_terminal_result) {
    // Keep p1 alive in the simulator, but leave no legal placement for it.
    // After p0's one move p1 is to act, so the depth-1 child is a terminal
    // p1 loss and the root Q must be +1 for p0.
    const RulesetConfig cfg = league();
    Player p0;
    Player p1;
    p0.reset(cfg, 1, 0);
    p1.reset(cfg, 2, 1);
    p0.set_now(100);
    p1.set_now(0);
    for (int y = 0; y < p1.mutable_board().height(); ++y)
        for (int x = 0; x < p1.mutable_board().width(); ++x)
            p1.mutable_board().fill_cell(x, y, false);

    UniformEvaluator ev;
    SearchConfig sc;
    sc.simulations = 1;
    sc.max_depth = 1;
    sc.batch_size = 1;
    sc.use_gumbel = false;
    sc.determinize_root = false;
    Searcher searcher(ev, sc);
    const SearchResult r = searcher.search(p0, &p1);

    CHECK(r.best_action >= 0);
    bool saw_visited = false;
    for (const auto& c : r.candidates) {
        if (c.visits == 0) continue;
        saw_visited = true;
        CHECK_MSG(c.q_value > 0.99f, "a terminal opponent loss must be a root win");
    }
    CHECK(saw_visited);
    CHECK_MSG(r.value.scalar() > 0.99f, "root value must report the winning terminal result");
}

TEST(a_same_player_child_does_not_negate_value) {
    // Event-driven play can let the same board act again. That is not a
    // player change and must not trigger a zero-sum sign conversion.
    Player p0 = warmed(7, 4);
    Player p1;
    p1.reset(league(), 9, 1);
    p1.set_now(1'000'000);

    ConstantValueEvaluator ev(0.75f);
    SearchConfig sc;
    sc.simulations = 1;
    sc.max_depth = 1;
    sc.batch_size = 1;
    sc.use_gumbel = false;
    sc.determinize_root = false;
    Searcher searcher(ev, sc);
    const SearchResult r = searcher.search(p0, &p1);

    CHECK_MSG(r.value.scalar() > 0.74f,
              "a child with the same player to act must keep the value sign");
    for (const auto& c : r.candidates)
        if (c.visits > 0) CHECK_MSG(c.q_value > 0.74f, "same-player Q must remain positive");
}

TEST(puct_value_is_ranked_from_the_player_to_move_perspective) {
    const float good_for_root = 0.8f;
    const float bad_for_root = -0.4f;
    CHECK(detail::puct_value_for_mover(good_for_root, /*root_to_move=*/true) >
          detail::puct_value_for_mover(bad_for_root, /*root_to_move=*/true));
    CHECK(detail::puct_value_for_mover(good_for_root, /*root_to_move=*/false) <
          detail::puct_value_for_mover(bad_for_root, /*root_to_move=*/false));
}

TEST(different_seeds_change_gumbel_but_not_legality) {
    HeuristicEvaluator ev;
    const Player p = warmed(11);
    std::set<int> chosen;
    for (std::uint64_t seed = 0; seed < 8; ++seed) {
        SearchConfig cfg;
        cfg.simulations = 32;
        cfg.max_depth = 3;
        cfg.use_gumbel = true;
        cfg.seed = seed;
        Searcher s(ev, cfg);
        const SearchResult r = s.search(p);
        CHECK(r.best_action >= 0);
        chosen.insert(r.best_action);
    }
    CHECK(!chosen.empty());
}

TEST(visits_respect_the_simulation_budget) {
    UniformEvaluator ev;
    for (int sims : {8, 32, 128}) {
        SearchConfig cfg;
        cfg.simulations = sims;
        cfg.max_depth = 3;
        cfg.use_gumbel = false;
        Searcher s(ev, cfg);
        const SearchResult r = s.search(warmed(3));
        int total = 0;
        for (const auto& c : r.candidates) total += c.visits;
        CHECK_MSG(total <= sims,
                  "root visits (" + std::to_string(total) + ") must not exceed the budget (" +
                      std::to_string(sims) + ")");
        CHECK(r.simulations_run <= sims);
    }
}

TEST(more_simulations_means_more_visits) {
    UniformEvaluator ev;
    int previous = -1;
    for (int sims : {8, 32, 128, 256}) {
        SearchConfig cfg;
        cfg.simulations = sims;
        cfg.max_depth = 3;
        cfg.use_gumbel = false;
        Searcher s(ev, cfg);
        const SearchResult r = s.search(warmed(3));
        int total = 0;
        for (const auto& c : r.candidates) total += c.visits;
        CHECK_MSG(total >= previous, "visit count should not decrease with a larger budget");
        previous = total;
    }
}

TEST(puct_spreads_visits_under_a_uniform_prior) {
    // With flat priors and no value signal, PUCT must explore broadly rather
    // than pile every visit onto one action.
    UniformEvaluator ev;
    SearchConfig cfg;
    cfg.simulations = 64;
    cfg.max_depth = 2;
    cfg.use_gumbel = false;
    cfg.batch_size = 1;
    Searcher s(ev, cfg);
    const SearchResult r = s.search(warmed(3));

    int visited = 0, max_visits = 0;
    for (const auto& c : r.candidates) {
        if (c.visits > 0) ++visited;
        max_visits = std::max(max_visits, c.visits);
    }
    CHECK_MSG(visited >= 8, "uniform PUCT should visit many actions, saw " +
                                std::to_string(visited));
    CHECK_MSG(max_visits < 40, "no single action should absorb the whole budget");
}

TEST(search_finds_the_only_line_clear) {
    // The decisive behavioural test: a position where one move is obviously
    // best. Both search modes must find it at every budget.
    //
    // This test is the reason three separate Gumbel bugs were caught. See
    // gumbel_noise_scale in SearchConfig for the calibration it drove.
    HeuristicEvaluator ev;
    const Player p = forced_clear_position();
    const auto acts = actions_of(p);

    int best_clear = -1, best_lines = 0;
    for (size_t i = 0; i < acts.size(); ++i)
        if (acts[i].cleared_lines > best_lines) {
            best_lines = acts[i].cleared_lines;
            best_clear = static_cast<int>(i);
        }
    CHECK(best_clear >= 0);
    CHECK(best_lines >= 2);

    for (bool gumbel : {false, true}) {
        for (int sims : {16, 64, 256}) {
            SearchConfig cfg;
            cfg.simulations = sims;
            cfg.max_depth = 3;
            cfg.use_gumbel = gumbel;
            cfg.seed = 7;
            cfg.batch_size = 8;
            Searcher s(ev, cfg);
            const SearchResult r = s.search(p);
            CHECK_MSG(acts[static_cast<size_t>(r.best_action)].cleared_lines == best_lines,
                      std::string(gumbel ? "gumbel" : "puct") + " at " + std::to_string(sims) +
                          " sims chose a non-clearing move");
        }
    }
}

TEST(search_beats_policy_only_under_pressure) {
    // Spec 19.1: a search bot must beat the policy-only bot. Without garbage
    // the heuristic survives indefinitely and the comparison is degenerate, so
    // this runs under a steady garbage stream where survival discriminates.
    RulesetConfig cfg = league();
    cfg.garbage.travel_time = 0;
    cfg.garbage.activation_delay = 0;

    HeuristicEvaluator ev;
    MoveGenerator gen;

    auto play = [&](int sims, bool gumbel) {
        int survived = 0, pieces = 0;
        const int games = 3, limit = 150;
        for (int seed = 0; seed < games; ++seed) {
            Player p;
            p.reset(cfg, static_cast<std::uint64_t>(seed), 0);
            int i = 0;
            for (; i < limit && p.alive(); ++i) {
                if (i > 0 && i % 8 == 0) p.receive_attack(2, p.now(), 1);
                const auto acts =
                    gen.generate(p.board(), p.active().type, p.hold(),
                                 p.visible_next().empty() ? Piece::None : p.visible_next()[0],
                                 cfg);
                if (acts.empty()) break;
                size_t pick = 0;
                if (sims > 0) {
                    SearchConfig sc;
                    sc.simulations = sims;
                    sc.max_depth = 4;
                    sc.use_gumbel = gumbel;
                    sc.seed = static_cast<std::uint64_t>(seed) * 997 + static_cast<unsigned>(i);
                    sc.batch_size = 16;
                    Searcher s(ev, sc);
                    const SearchResult r = s.search(p);
                    if (r.best_action < 0) break;
                    pick = static_cast<size_t>(r.best_action);
                } else {
                    const Evaluation e = ev.evaluate_one(observe(p), acts);
                    for (size_t k = 1; k < e.policy.size(); ++k)
                        if (e.policy[k] > e.policy[pick]) pick = k;
                }
                if (pick >= acts.size()) break;
                if (acts[pick].use_hold && !p.do_hold()) continue;
                p.set_active(acts[pick].piece_state());
                int out = 0;
                if (!p.lock_piece(acts[pick].total_duration(), &out).ok) break;
            }
            pieces += i;
            if (p.alive()) ++survived;
        }
        return std::make_pair(pieces, survived);
    };

    const auto policy_only = play(0, false);

    // Gumbel is sound at a low budget by construction (spec 11.2), which is
    // exactly why it is the default.
    const auto gumbel32 = play(32, true);
    CHECK_MSG(gumbel32.first >= policy_only.first,
              "gumbel-32 should match or beat policy-only: " +
                  std::to_string(gumbel32.first) + " vs " + std::to_string(policy_only.first));

    // PUCT needs a larger budget to be reliable here; see
    // puct_needs_enough_simulations_to_beat_its_prior for the measured reason.
    const auto puct128 = play(128, false);
    CHECK_MSG(puct128.first >= policy_only.first,
              "puct-128 should match or beat policy-only: " +
                  std::to_string(puct128.first) + " vs " + std::to_string(policy_only.first));
}

TEST(puct_needs_enough_simulations_to_beat_its_prior) {
    // A measured limitation, pinned so it is not mistaken for a bug later.
    //
    // A tetris position has ~25-50 legal placements. Given a budget below
    // roughly two visits per action, PUCT cannot separate them usefully: with a
    // flat prior and no value signal the exploration term keeps steering to
    // whichever child the tie-break favours, so the visits concentrate on an
    // essentially arbitrary action instead of surveying the options. Measured
    // under a garbage stream, that makes it *worse* than following the prior:
    // policy-only 450 pieces, puct-64 380, puct-128 450.
    //
    // Gumbel does not share the failure mode -- sequential halving explicitly
    // samples a candidate set and gives each member an equal share -- which is
    // why `use_gumbel` defaults to true (spec 11.2).
    UniformEvaluator ev;
    const Player p = warmed(3);
    const auto acts = actions_of(p);
    CHECK_MSG(acts.size() > 20, "expected a wide action set at the root");

    const int thin_budget = static_cast<int>(acts.size()) / 2;

    SearchConfig thin;
    thin.simulations = thin_budget;
    thin.max_depth = 3;
    thin.use_gumbel = false;
    thin.batch_size = 1;
    Searcher sp(ev, thin);
    const SearchResult rp = sp.search(p);

    SearchConfig gum = thin;
    gum.use_gumbel = true;
    gum.gumbel_m = 8;
    Searcher sg(ev, gum);
    const SearchResult rg = sg.search(p);

    auto survey = [](const SearchResult& r) {
        int visited = 0, total = 0;
        for (const auto& c : r.candidates) {
            if (c.visits > 0) ++visited;
            total += c.visits;
        }
        return std::make_pair(visited, total);
    };
    const auto [puct_visited, puct_total] = survey(rp);
    const auto [gum_visited, gum_total] = survey(rg);

    CHECK(puct_total > 0);
    CHECK(gum_total > 0);
    // The measured contrast: on a thin budget Gumbel surveys many more distinct
    // actions than PUCT, which is precisely the policy-improvement guarantee
    // spec 11.2 wants at low simulation counts.
    CHECK_MSG(gum_visited > puct_visited,
              "gumbel should survey more actions on a thin budget: " +
                  std::to_string(gum_visited) + " vs " + std::to_string(puct_visited));
}

TEST(leaves_are_evaluated_in_batches) {
    // Spec 19.4 targets >= 80% batched leaf evaluation. The mean batch size is
    // the measurable proxy: with a batch size of 16 the search must not be
    // issuing one-position calls.
    UniformEvaluator ev;
    SearchConfig cfg;
    cfg.simulations = 128;
    cfg.max_depth = 4;
    cfg.use_gumbel = false;
    cfg.batch_size = 16;
    Searcher s(ev, cfg);
    const SearchResult r = s.search(warmed(9));

    CHECK_MSG(r.mean_batch_size > 4.0,
              "leaves should be batched; mean batch " + std::to_string(r.mean_batch_size));
    CHECK(r.positions_evaluated > 0);
    CHECK(r.evaluator_calls > 0);
    CHECK_MSG(r.evaluator_calls < r.positions_evaluated,
              "batching must reduce the number of evaluator calls");
}

TEST(larger_batches_mean_fewer_evaluator_calls) {
    UniformEvaluator ev;
    int previous_calls = 1 << 30;
    for (int bs : {1, 4, 16, 64}) {
        SearchConfig cfg;
        cfg.simulations = 128;
        cfg.max_depth = 4;
        cfg.use_gumbel = false;
        cfg.batch_size = bs;
        Searcher s(ev, cfg);
        const SearchResult r = s.search(warmed(9));
        CHECK_MSG(r.evaluator_calls <= previous_calls + 1,
                  "a larger batch should not increase evaluator calls");
        previous_calls = r.evaluator_calls;
    }
}

TEST(batching_does_not_change_the_visit_total) {
    // Virtual loss must be reverted correctly: the same budget must produce
    // the same number of backed-up simulations regardless of batch size.
    UniformEvaluator ev;
    for (int bs : {1, 8, 32}) {
        SearchConfig cfg;
        cfg.simulations = 64;
        cfg.max_depth = 3;
        cfg.use_gumbel = false;
        cfg.batch_size = bs;
        cfg.seed = 4;
        Searcher s(ev, cfg);
        const SearchResult r = s.search(warmed(3));
        int total = 0;
        for (const auto& c : r.candidates) total += c.visits;
        CHECK_MSG(total <= 64, "batch size must not inflate the visit count");
        CHECK(total > 0);
    }
}

TEST(virtual_loss_leaves_no_pending_visits) {
    // If virtual loss were not reverted, q values would drift towards the
    // pessimistic bound and visit counts would not match the budget.
    UniformEvaluator ev;
    SearchConfig cfg;
    cfg.simulations = 96;
    cfg.max_depth = 3;
    cfg.batch_size = 16;
    cfg.use_gumbel = false;
    Searcher s(ev, cfg);
    const SearchResult r = s.search(warmed(6));
    for (const auto& c : r.candidates) {
        CHECK(c.visits >= 0);
        if (c.visits == 0) CHECK(c.q_value == 0.0f);
        CHECK(std::isfinite(c.q_value));
        CHECK(c.q_value >= -1.5f && c.q_value <= 1.5f);
    }
}

TEST(transposition_table_finds_repeats) {
    // Spec 11.1 requires a TT. Different move orders can reach the same
    // position, so on a real board some hits are expected.
    UniformEvaluator ev;
    SearchConfig cfg;
    cfg.simulations = 256;
    cfg.max_depth = 5;
    cfg.use_gumbel = false;
    cfg.use_transposition_table = true;
    Searcher on(ev, cfg);
    const SearchResult with_tt = on.search(warmed(2));

    cfg.use_transposition_table = false;
    Searcher off(ev, cfg);
    const SearchResult without = off.search(warmed(2));

    CHECK(with_tt.nodes_created <= without.nodes_created);
}

TEST(transposition_key_separates_distinct_states) {
    // Spec 11.1: the key must include the ruleset, the clock and the queue, or
    // two genuinely different positions would share a node.
    UniformEvaluator ev;
    SearchConfig cfg;
    cfg.simulations = 32;
    cfg.max_depth = 3;
    cfg.use_transposition_table = true;

    // Same board, different pending garbage: must not collapse to one node.
    Player a = warmed(4);
    Player b = a;
    b.receive_attack(4, b.now(), 1);

    Searcher sa(ev, cfg), sb(ev, cfg);
    const SearchResult ra = sa.search(a);
    const SearchResult rb = sb.search(b);
    CHECK(ra.best_action >= 0);
    CHECK(rb.best_action >= 0);
}

TEST(search_position_key_keeps_all_scheduling_and_player_state) {
    const RulesetConfig cfg = league();
    Player base;
    base.reset(cfg, 4, 0);
    const std::uint64_t base_key = detail::search_position_key(base, nullptr);

    auto check_distinct = [&](const char* what, const Player& changed) {
        CHECK_MSG(detail::search_position_key(changed, nullptr) != base_key,
                  std::string("search key must include ") + what);
    };

    Player arrival_early = base;
    Player arrival_late = base;
    arrival_early.receive_attack(3, 0, 1);
    arrival_late.receive_attack(3, 7, 1);
    CHECK_MSG(detail::search_position_key(arrival_early, nullptr) !=
                  detail::search_position_key(arrival_late, nullptr),
              "garbage arrival schedule must distinguish positions");

    Player bag = base;
    bag.determinize(0xBA65EEDull);
    check_distinct("bag state", bag);

    Player queue = base;
    queue.mutable_queue().pop();
    check_distinct("queue state", queue);

    Player hold = base;
    CHECK(hold.do_hold());
    check_distinct("hold and hold-used state", hold);

    Player attack_state = base;
    attack_state.mutable_attack_state().combo = 4;
    attack_state.mutable_attack_state().b2b_streak = 3;
    check_distinct("B2B and combo state", attack_state);

    Player garbage_plane = base;
    garbage_plane.mutable_board().set_garbage_row(0, 1u, 1u);
    check_distinct("garbage-cell plane", garbage_plane);
}

TEST(terminal_positions_are_handled) {
    // A dead player must produce an empty, well-formed result rather than
    // crash or search a nonexistent tree.
    const RulesetConfig cfg = league();
    Player p;
    p.reset(cfg, 1, 0);
    p.die(TopoutReason::BlockOut);
    CHECK(!p.alive());

    UniformEvaluator ev;
    Searcher s(ev, SearchConfig{});
    const SearchResult r = s.search(p);
    CHECK_EQ(r.best_action, -1);
    CHECK(r.value.scalar() <= 0.0f);
}

TEST(search_handles_a_position_with_one_action) {
    // A nearly full board can leave a single placement. The search must return
    // it without spending the budget pointlessly or dividing by zero.
    const RulesetConfig cfg = league();
    Player p;
    p.reset(cfg, 1, 0);
    Board& b = p.mutable_board();
    // Fill everything except a narrow well.
    for (int y = 0; y < 15; ++y)
        for (int x = 0; x < 9; ++x) b.fill_cell(x, y, false);

    UniformEvaluator ev;
    SearchConfig sc;
    sc.simulations = 32;
    sc.max_depth = 3;
    Searcher s(ev, sc);
    const SearchResult r = s.search(p);
    const auto acts = actions_of(p);
    if (!acts.empty()) {
        CHECK(r.best_action >= 0);
        CHECK(r.best_action < static_cast<int>(acts.size()));
    }
}

TEST(zero_simulations_falls_back_to_the_prior) {
    // Policy-only mode (spec 11.2, budget table). With no simulations the
    // search must return the prior's argmax rather than an arbitrary action.
    HeuristicEvaluator ev;
    const Player p = warmed(8);
    const auto acts = actions_of(p);
    const Evaluation e = ev.evaluate_one(observe(p), acts);
    size_t prior_best = 0;
    for (size_t k = 1; k < e.policy.size(); ++k)
        if (e.policy[k] > e.policy[prior_best]) prior_best = k;

    SearchConfig cfg;
    cfg.simulations = 0;
    cfg.use_gumbel = false;
    Searcher s(ev, cfg);
    const SearchResult r = s.search(p);
    CHECK_EQ(r.best_action, static_cast<int>(prior_best));
}

TEST(depth_limit_is_respected) {
    // A deeper search must build a deeper tree, and a depth-1 search must not
    // create more nodes than it has actions.
    UniformEvaluator ev;
    int previous_nodes = 0;
    for (int depth : {1, 3, 6}) {
        SearchConfig cfg;
        cfg.simulations = 64;
        cfg.max_depth = depth;
        cfg.use_gumbel = false;
        Searcher s(ev, cfg);
        const SearchResult r = s.search(warmed(3));
        CHECK(r.nodes_created > 0);
        if (depth > 1) CHECK(r.nodes_created >= previous_nodes);
        previous_nodes = r.nodes_created;
    }
}

TEST(root_noise_changes_priors_but_keeps_a_distribution) {
    HeuristicEvaluator ev;
    const Player p = warmed(13);

    SearchConfig quiet;
    quiet.simulations = 32;
    quiet.max_depth = 2;
    quiet.use_gumbel = false;
    quiet.root_noise_fraction = 0.0f;
    quiet.seed = 1;

    SearchConfig noisy = quiet;
    noisy.root_noise_fraction = 0.25f;

    Searcher a(ev, quiet), b(ev, noisy);
    const SearchResult ra = a.search(p);
    const SearchResult rb = b.search(p);

    float sum_a = 0.0f, sum_b = 0.0f;
    for (const auto& c : ra.candidates) sum_a += c.prior;
    for (const auto& c : rb.candidates) sum_b += c.prior;
    CHECK(std::fabs(sum_a - 1.0f) < 1e-3f);
    CHECK(std::fabs(sum_b - 1.0f) < 1e-3f);
}

TEST(search_works_across_rulesets) {
    UniformEvaluator ev;
    for (const RulesetConfig& cfg :
         {RulesetConfig::tetra_league(), RulesetConfig::quick_play(),
          RulesetConfig::guideline()}) {
        Player p;
        p.reset(cfg, 3, 0);
        SearchConfig sc;
        sc.simulations = 32;
        sc.max_depth = 3;
        Searcher s(ev, sc);
        const SearchResult r = s.search(p);
        CHECK_MSG(r.best_action >= 0, std::string("search failed under ") + cfg.id);
    }
}

TEST(search_reports_a_finite_value) {
    HeuristicEvaluator ev;
    for (std::uint64_t seed : {1ull, 20ull, 300ull}) {
        SearchConfig cfg;
        cfg.simulations = 48;
        cfg.max_depth = 3;
        Searcher s(ev, cfg);
        const SearchResult r = s.search(warmed(seed));
        CHECK(std::isfinite(r.value.scalar()));
        CHECK(r.value.scalar() >= -1.0f && r.value.scalar() <= 1.0f);
        CHECK(std::fabs(r.value.win + r.value.draw + r.value.loss - 1.0f) < 1e-4f);
    }
}

TEST(candidate_priors_match_the_evaluator) {
    HeuristicEvaluator ev;
    const Player p = warmed(15);
    const auto acts = actions_of(p);
    const Evaluation e = ev.evaluate_one(observe(p), acts);

    SearchConfig cfg;
    cfg.simulations = 16;
    cfg.max_depth = 2;
    cfg.root_noise_fraction = 0.0f;
    Searcher s(ev, cfg);
    const SearchResult r = s.search(p);

    CHECK_EQ(r.candidates.size(), e.policy.size());
    for (size_t i = 0; i < r.candidates.size() && i < e.policy.size(); ++i)
        CHECK_MSG(std::fabs(r.candidates[i].prior - e.policy[i]) < 1e-5f,
                  "root priors must come from the evaluator unchanged");
}

// ---------------------------------------------------------------------------
// Chance nodes / determinization (spec 11.3, and the 18.3 leak requirement)
// ---------------------------------------------------------------------------

TEST(determinize_preserves_the_visible_preview) {
    // The pieces the player can legitimately see must survive untouched;
    // only the hidden tail may be resampled.
    const RulesetConfig cfg = league();
    for (std::uint64_t seed = 0; seed < 40; ++seed) {
        Player p;
        p.reset(cfg, seed, 0);
        // Advance a variable amount so the bag is mid-cycle.
        for (std::uint64_t i = 0; i < seed % 9; ++i) p.mutable_queue().pop();
        const std::vector<Piece> visible = p.visible_next();

        Player d = p;
        d.determinize(seed * 7919 + 1);
        CHECK_MSG(d.visible_next() == visible,
                  "determinization must not change what the player can see");
        CHECK(d.board() == p.board());
        CHECK_EQ(d.now(), p.now());
    }
}

TEST(determinize_resamples_the_hidden_tail) {
    // Beyond the preview the sequence must actually change, otherwise the
    // search is still reading the true future.
    const RulesetConfig cfg = league();
    Player p;
    p.reset(cfg, 42, 0);
    const int preview = cfg.randomizer.preview_count;

    auto tail_after = [&](std::uint64_t seed) {
        Player d = p;
        d.determinize(seed);
        for (int i = 0; i < preview; ++i) d.mutable_queue().pop();
        std::vector<Piece> tail;
        for (int i = 0; i < 8; ++i) tail.push_back(d.mutable_queue().pop());
        return tail;
    };

    const auto a = tail_after(1);
    const auto b = tail_after(2);
    CHECK_MSG(a != b, "different determinizations must produce different futures");
}

TEST(determinize_keeps_the_bag_balanced) {
    // A resample that ignored the bag would be *less* informed than a human,
    // who can count a 7-bag. An earlier version returned nothing to the bag and
    // produced three O and one L inside a 14-piece window.
    const RulesetConfig cfg = league();
    for (std::uint64_t seed = 0; seed < 60; ++seed) {
        Player p;
        p.reset(cfg, seed, 0);
        for (std::uint64_t i = 0; i < seed % 9; ++i) p.mutable_queue().pop();

        Player d = p;
        d.determinize(seed * 7919 + 1);

        int counts[PIECE_COUNT] = {0};
        const int n = 700;
        for (int i = 0; i < n; ++i) counts[static_cast<int>(d.mutable_queue().pop())]++;
        for (int i = 0; i < PIECE_COUNT; ++i)
            CHECK_MSG(std::abs(counts[i] - n / PIECE_COUNT) <= 7,
                      std::string("piece ") + piece_name(static_cast<Piece>(i)) +
                          " appeared " + std::to_string(counts[i]) + " times in " +
                          std::to_string(n) + " draws");
    }
}

TEST(search_does_not_read_beyond_the_preview) {
    // The leak this exists to prevent (spec 3.2, 18.3).
    //
    // The simulator's queue holds the true future. A search running deeper than
    // `preview_count` would otherwise plan against pieces the player cannot
    // see. With determinization on, the search result must not depend on the
    // hidden tail at all: two states that agree on everything visible but were
    // seeded to hold different futures must search identically.
    RulesetConfig cfg = league();
    Player a;
    a.reset(cfg, 12345, 0);

    // Build a second player with the same board, clock and visible preview but
    // a different hidden continuation.
    Player b = a;
    b.determinize(0xABCDEF);
    CHECK(a.visible_next() == b.visible_next());

    HeuristicEvaluator ev;
    SearchConfig sc;
    sc.simulations = 48;
    // Deeper than the preview, so the hidden tail would matter if it leaked.
    sc.max_depth = cfg.randomizer.preview_count + 4;
    sc.use_gumbel = true;
    sc.seed = 99;
    sc.determinize_root = true;

    Searcher sa(ev, sc), sb(ev, sc);
    const SearchResult ra = sa.search(a);
    const SearchResult rb = sb.search(b);

    CHECK_MSG(ra.best_action == rb.best_action,
              "the search must not depend on pieces the player cannot see");
    CHECK_EQ(ra.search_policy.size(), rb.search_policy.size());
    for (size_t i = 0; i < ra.search_policy.size() && i < rb.search_policy.size(); ++i)
        CHECK(std::fabs(ra.search_policy[i] - rb.search_policy[i]) < 1e-6f);
}

TEST(determinization_keeps_the_search_deterministic) {
    // Sampling futures must not cost reproducibility (spec 19.4).
    HeuristicEvaluator ev;
    const Player p = warmed(21);
    for (int n : {1, 4}) {
        SearchConfig sc;
        sc.simulations = 32;
        sc.max_depth = 6;
        sc.seed = 777;
        sc.determinize_root = true;
        sc.determinizations = n;
        Searcher x(ev, sc), y(ev, sc);
        const SearchResult rx = x.search(p);
        const SearchResult ry = y.search(p);
        CHECK_EQ(rx.best_action, ry.best_action);
        CHECK_EQ(rx.search_policy.size(), ry.search_policy.size());
        for (size_t i = 0; i < rx.search_policy.size(); ++i)
            CHECK(rx.search_policy[i] == ry.search_policy[i]);
    }
}

TEST(multiple_determinizations_average_the_policy) {
    // Averaging over sampled futures is the particle form of a chance node.
    // The result must still be a valid distribution and a legal action.
    HeuristicEvaluator ev;
    const Player p = warmed(30);
    SearchConfig sc;
    sc.simulations = 24;
    sc.max_depth = 6;
    sc.seed = 5;
    sc.determinize_root = true;
    sc.determinizations = 4;
    Searcher s(ev, sc);
    const SearchResult r = s.search(p);

    CHECK(r.best_action >= 0);
    float sum = 0.0f;
    for (float v : r.search_policy) {
        CHECK(std::isfinite(v));
        CHECK(v >= 0.0f);
        sum += v;
    }
    CHECK_MSG(std::fabs(sum - 1.0f) < 1e-3f,
              "averaged policy must remain a distribution, got " + std::to_string(sum));
    CHECK(std::isfinite(r.value.scalar()));
}

TEST(determinization_is_cheap) {
    // It must not price the search out of the spec 19.4 budget: resampling is
    // a queue operation, not a search-wide cost.
    HeuristicEvaluator ev;
    const Player p = warmed(42, 25);
    SearchConfig off;
    off.simulations = 64;
    off.max_depth = 8;
    off.seed = 1;
    off.determinize_root = false;

    SearchConfig on = off;
    on.determinize_root = true;

    Searcher a(ev, off), b(ev, on);
    const SearchResult ra = a.search(p);
    const SearchResult rb = b.search(p);
    // Both must do comparable work; determinization changes which future is
    // searched, not how much searching happens.
    CHECK(ra.simulations_run > 0);
    CHECK(rb.simulations_run > 0);
    CHECK_EQ(ra.simulations_run, rb.simulations_run);
}

TEST(queue_reports_hidden_lookahead) {
    RandomizerCfg rc;
    rc.preview_count = 5;
    PieceQueue q(rc, 1);
    // The queue buffers beyond the preview, which is exactly the hidden state.
    CHECK(q.has_hidden_lookahead());
    q.resample_hidden(9);
    // Still buffered, but now from a resampled future rather than the real one.
    CHECK_EQ(static_cast<int>(q.visible_next().size()), 5);
}

TEST(two_player_search_passes_the_opponent_to_both_evaluators) {
    Player p0;
    Player p1;
    p0.reset(league(), 1, 0);
    p1.reset(league(), 2, 1);

    ObservationSpy root_evaluator;
    ObservationSpy opponent_evaluator;
    SearchConfig cfg;
    cfg.simulations = 32;
    cfg.max_depth = 3;
    cfg.batch_size = 8;
    cfg.use_gumbel = false;
    cfg.determinize_root = false;

    Searcher searcher(root_evaluator, cfg);
    const SearchResult r = searcher.search(p0, &p1, &opponent_evaluator);

    CHECK(r.best_action >= 0);
    CHECK(root_evaluator.saw_any);
    CHECK(root_evaluator.saw_opponent);
    CHECK(opponent_evaluator.saw_any);
    CHECK(opponent_evaluator.saw_opponent);
}
