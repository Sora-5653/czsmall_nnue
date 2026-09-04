// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- PUCT / Gumbel search (spec 11).
//
// The search is written against `Evaluator` and collects leaves into batches
// from the outset. That is not an optimisation: measurement showed a
// spec-sized network costs ~8.6 ms per position against ~0.09 ms for move
// generation, so how leaves are gathered determines whether the whole thing
// fits the budget in spec 19.4. See ADR 0007.
//
// Two root policies are provided:
//
//   * PUCT (spec 11.1), the classic AlphaZero rule.
//   * Gumbel sequential halving (spec 11.2), which produces a sound policy
//     improvement at low simulation counts -- the regime this bot runs in.
#pragma once

#include "tetra/evaluator.hpp"
#include "tetra/movegen.hpp"
#include "tetra/observation.hpp"
#include "tetra/player.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <unordered_map>
#include <vector>

namespace tetra {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct SearchConfig {
    int simulations = 128;
    int max_depth = 8;          // in placements (spec 11.1 suggests 8-20)
    int batch_size = 16;        // leaves collected before an evaluator flush
    // Disabled by default. When positive, the search stops starting new leaves
    // after this per-decision wall-clock budget. An already-running evaluator
    // batch is allowed to finish and its overshoot is reported.
    double time_budget_ms = 0.0;
    // Disabled by default. Diagnostic equal-node control; the root itself is
    // counted, and a path may overshoot only by terminal/transposition work
    // that was already started before the cap was observed.
    int node_budget = 0;

    // PUCT
    float c_puct = 1.5f;
    float fpu_reduction = 0.2f; // first-play-urgency for unvisited children

    // Gumbel sequential halving (spec 11.2)
    bool use_gumbel = true;
    int gumbel_m = 16;          // candidate actions considered at the root
    float gumbel_c_visit = 50.0f;
    // The paper-style unit scale assumes a well-calibrated root Q.  With this
    // network it overwhelms log-priors: same-checkpoint Arena measured the
    // 32-sim Gumbel side at 1-15 versus policy-only.  A 0.01 scale restored
    // parity across matched trials while still allowing Q to break close ties.
    float gumbel_c_scale = 0.01f;

    // Scale applied to the Gumbel perturbation when picking the root
    // candidates.
    //
    // Gumbel top-k assumes the logits carry most of the signal. That holds for
    // a trained network but NOT for a weak or freshly initialised one: with an
    // untrained policy the logits of ~34 tetris placements span ~0.6 nats while
    // the Gumbel noise over that many draws spans ~5, so selection becomes ~90%
    // random and the search plays far worse than its own priors (measured: 27
    // pieces per game against 200 for policy-only).
    //
    // Damping the noise keeps the exploration Gumbel provides while letting the
    // priors decide which actions are worth considering at all. A trained,
    // confident policy can raise this back towards 1.
    //
    // Calibrated on a position with one obviously correct move (the only line
    // clear available). At 1.0 the search picks the wrong action at every
    // budget; at 0.2 it recovers only by 256 simulations; at 0.05 it is correct
    // at 16, 64 and 256. Self-play should raise it once the policy is trained,
    // and root Dirichlet noise remains the intended exploration knob.
    float gumbel_noise_scale = 0.05f;

    // Dirichlet-style root exploration noise. Off by default; self-play turns
    // it on. Implemented with the engine RNG so runs stay reproducible.
    float root_noise_fraction = 0.0f;
    float root_noise_alpha = 0.3f;

    std::uint64_t seed = 0;
    bool use_transposition_table = true;

    // Experimental timing branches. Off by default so existing checkpoints,
    // datasets and Arena results keep the historical action contract. When
    // enabled, positions with pending garbage branch each placement into
    // FASTEST and WAIT_FOR_EVENT variants, which is enough to represent the
    // core "cancel now vs deliberately receive first" decision without
    // multiplying every ordinary position's action count.
    bool enable_timing_actions = false;

    // Chance nodes / determinization (spec 11.3).
    //
    // The simulator's queue holds the true future, but the player may only see
    // `preview_count` pieces. Searching the real sequence is an information
    // leak (spec 3.2, 18.3): it rewards setups that only work because the
    // engine peeked. With this enabled the root state is copied and its hidden
    // tail resampled, so each determinization is one future consistent with the
    // observation, and averaging over several approximates the chance node.
    bool determinize_root = true;
    int determinizations = 1;  // >1 averages several sampled futures
};

// What the search returns (spec 16 DecisionResponse).
struct SearchCandidate {
    int action_index = 0;
    float prior = 0.0f;
    int visits = 0;
    float q_value = 0.0f;
};

struct SearchTelemetry {
    // LC3-style pipeline counters. These are observability only: changing or
    // reading them must never affect search decisions.
    int gather_attempts = 0;
    int gathered_leaves = 0;
    int selection_steps = 0;
    int terminal_backups = 0;
    int depth_cutoffs = 0;
    int evaluation_flushes = 0;
    int max_edge_inflight = 0;
    double elapsed_ms = 0.0;
    double evaluator_elapsed_ms = 0.0;
    // Diagnostic wall-clock decomposition.  These fields are observational;
    // the search decisions must not depend on them.  The fine-grained
    // selection/state/legal/node fields are inclusive subcomponents of the
    // broader gather/CPU intervals and therefore are not expected to sum to
    // elapsed time.
    double root_setup_us = 0.0;
    double gather_us = 0.0;
    double backup_us = 0.0;
    double finalize_us = 0.0;
    double node_allocation_us = 0.0;
    double legal_action_generation_us = 0.0;
    double state_transition_us = 0.0;
    double selection_us = 0.0;
    std::int64_t depth_sum = 0;
    int depth_samples = 0;
    int max_depth = 0;
    int node_budget_cutoffs = 0;
    bool budget_exhausted = false;
};

struct SearchResult {
    int best_action = -1;
    ValueWDL value;
    std::vector<SearchCandidate> candidates;
    std::vector<float> search_policy;  // the training target pi (spec 13.5)
    int simulations_run = 0;
    int nodes_created = 0;
    int evaluator_calls = 0;
    int positions_evaluated = 0;
    int transposition_hits = 0;
    double mean_batch_size = 0.0;
    double elapsed_ms = 0.0;
    double evaluator_elapsed_ms = 0.0;
    double mean_depth = 0.0;
    int max_depth = 0;
    int raw_policy_action = -1;
    bool time_budget_exhausted = false;
    SearchTelemetry telemetry;
};

// ---------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------
namespace detail {

struct SearchEdge {
    // LC3 keeps the complete edge state together. P, Q, N and N_inflight are
    // therefore one cache-friendly unit instead of several parallel vectors.
    float prior = 0.0f;              // P
    float value_sum = 0.0f;          // Q numerator, root-player perspective
    int visits = 0;                  // N
    int inflight = 0;                // N_inflight / virtual loss
    std::int32_t child = -1;

    float q() const { return visits > 0 ? value_sum / static_cast<float>(visits) : 0.0f; }
    int effective_visits() const { return visits + inflight; }
};

struct SearchNode {
    // Game state at this node. Copying a Player costs ~0.24 us, which is
    // negligible beside evaluation, and it keeps the tree free of undo logic.
    Player state;
    // In a 1v1 search this is the other board at the same event time. The
    // active player is kept in `state`; children swap the two clocks when the
    // opponent is next to act.
    Player opponent;
    bool has_opponent = false;
    std::vector<PlacementAction> actions;
    std::vector<SearchEdge> edges;

    int visits = 0;
    float value_sum = 0.0f;
    float leaf_value = 0.0f;  // the evaluator's value for this node
    bool expanded = false;
    bool terminal = false;
    float terminal_value = 0.0f;

    size_t size() const { return actions.size(); }
};

inline float puct_value_for_mover(float root_value, bool root_to_move) {
    return root_to_move ? root_value : -root_value;
}

// Full simulator-state key used by the transposition table. This is kept in
// `detail` so the key contract can be tested without making it part of the
// public search API. In particular, the board hash used by replay records does
// not include every piece of scheduling state needed by search.
inline void mix_search_player_key(std::uint64_t& h, const Player& p) {
    auto mix = [&h](std::uint64_t v) {
        h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    };
    mix(static_cast<std::uint64_t>(p.index()));
    mix(static_cast<std::uint64_t>(p.board().width()));
    mix(static_cast<std::uint64_t>(p.board().height()));
    for (int y = 0; y < p.board().height(); ++y) {
        mix(static_cast<std::uint64_t>(p.board().row(y)));
        mix(static_cast<std::uint64_t>(p.board().garbage_row(y)));
    }
    mix(p.ruleset().hash());
    mix(static_cast<std::uint64_t>(p.now()));

    const ActivePiece& active = p.active();
    mix(static_cast<std::uint64_t>(active.type));
    mix(static_cast<std::uint64_t>(active.rot));
    mix(static_cast<std::uint64_t>(active.x));
    mix(static_cast<std::uint64_t>(active.y));
    mix(static_cast<std::uint64_t>(active.last_action));
    mix(static_cast<std::uint64_t>(active.last_kick));
    mix(static_cast<std::uint64_t>(p.hold()));
    mix(static_cast<std::uint64_t>(p.hold_used()));

    const AttackState& attack = p.attack_state();
    mix(static_cast<std::uint64_t>(attack.combo + 1));
    mix(static_cast<std::uint64_t>(attack.b2b_streak));
    mix(static_cast<std::uint64_t>(attack.surge));
    mix(static_cast<std::uint64_t>(attack.pieces_placed));

    const auto& garbage = p.garbage().entries();
    mix(static_cast<std::uint64_t>(garbage.size()));
    for (const auto& g : garbage) {
        mix(static_cast<std::uint64_t>(g.lines));
        mix(static_cast<std::uint64_t>(g.sent_at));
        mix(static_cast<std::uint64_t>(g.arrival_at));
        mix(static_cast<std::uint64_t>(g.activation_at));
        mix(static_cast<std::uint64_t>(g.cancellable));
        mix(static_cast<std::uint64_t>(g.tankable));
        mix(static_cast<std::uint64_t>(g.hole_column + 1));
        mix(static_cast<std::uint64_t>(g.source_player + 1));
        mix(static_cast<std::uint64_t>(g.rule_metadata));
    }

    const auto buffered = p.queue().buffered_pieces();
    mix(static_cast<std::uint64_t>(buffered.size()));
    for (Piece piece : buffered) mix(static_cast<std::uint64_t>(piece));
    const auto bag = p.queue().bag_remaining();
    mix(static_cast<std::uint64_t>(bag.size()));
    for (Piece piece : bag) mix(static_cast<std::uint64_t>(piece));
    mix(p.queue().pieces_generated());
    mix(static_cast<std::uint64_t>(p.queue().mirrored()));
    for (std::uint64_t word : p.queue().rng().state()) mix(word);
    for (std::uint64_t word : p.garbage_rng_state()) mix(word);
    for (std::uint64_t word : p.attack_rng_state()) mix(word);
    mix(static_cast<std::uint64_t>(p.mirrored()));

    mix(static_cast<std::uint64_t>(p.alive()));
    mix(static_cast<std::uint64_t>(p.topout_reason()));
    mix(static_cast<std::uint64_t>(p.lines_sent()));
    mix(static_cast<std::uint64_t>(p.lines_received()));
    mix(static_cast<std::uint64_t>(p.lines_cleared()));

    const auto& events = p.events().events();
    mix(static_cast<std::uint64_t>(events.size()));
    for (const auto& e : events) {
        mix(static_cast<std::uint64_t>(e.timestamp));
        mix(static_cast<std::uint64_t>(e.timestamp_delta));
        mix(static_cast<std::uint64_t>(e.actor));
        mix(static_cast<std::uint64_t>(e.type));
        mix(static_cast<std::uint64_t>(e.piece));
        mix(static_cast<std::uint64_t>(e.spin));
        mix(static_cast<std::uint64_t>(e.lines));
        mix(static_cast<std::uint64_t>(e.duration));
    }
}

inline std::uint64_t search_position_key(const Player& p, const Player* opponent) {
    std::uint64_t h = 0xCBF29CE484222325ull;
    mix_search_player_key(h, p);
    if (opponent) {
        h ^= 0xD6E8FEB86659FD93ull;
        mix_search_player_key(h, *opponent);
    }
    return h;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Search policy
// ---------------------------------------------------------------------------
// LC3 separates tree orchestration from edge scoring. Keeping this as a small
// policy object makes it possible to change selection/calibration without
// entangling node construction, inference batching or backup semantics.
class SearchPolicy {
public:
    explicit SearchPolicy(const SearchConfig& cfg) : cfg_(&cfg) {}

    int select_puct(const detail::SearchNode& n, int root_player_index) const {
        const float sqrt_total = std::sqrt(static_cast<float>(std::max(1, n.visits)));
        const float parent_q =
            n.visits > 0 ? n.value_sum / static_cast<float>(n.visits) : 0.0f;
        const bool root_to_move = !n.has_opponent || n.state.index() == root_player_index;

        int best = 0;
        float best_score = -1e30f;
        for (size_t a = 0; a < n.edges.size(); ++a) {
            const detail::SearchEdge& edge = n.edges[a];
            float mover_q;
            if (edge.visits > 0) {
                mover_q = detail::puct_value_for_mover(edge.q(), root_to_move);
            } else {
                const float mover_parent_q =
                    detail::puct_value_for_mover(parent_q, root_to_move);
                mover_q = mover_parent_q - cfg_->fpu_reduction;
            }
            const float u = cfg_->c_puct * edge.prior * sqrt_total /
                            (1.0f + static_cast<float>(edge.effective_visits()));
            const float score = mover_q + u;
            if (score > best_score) {
                best_score = score;
                best = static_cast<int>(a);
            }
        }
        return best;
    }

    float gumbel_score(const detail::SearchNode& root, int action,
                       const std::vector<float>& logits,
                       const std::vector<float>& gumbel,
                       int max_child_visits) const {
        const detail::SearchEdge& edge = root.edges[static_cast<size_t>(action)];
        const float parent_q = root.visits > 0
                                   ? root.value_sum / static_cast<float>(root.visits)
                                   : root.leaf_value;
        const float q = edge.visits > 0 ? edge.q() : parent_q;
        const float sigma =
            (cfg_->gumbel_c_visit + static_cast<float>(max_child_visits)) *
            cfg_->gumbel_c_scale * q;
        return gumbel[static_cast<size_t>(action)] + logits[static_cast<size_t>(action)] + sigma;
    }

private:
    const SearchConfig* cfg_;
};

// ---------------------------------------------------------------------------
// The searcher
// ---------------------------------------------------------------------------
class Searcher {
public:
    Searcher(Evaluator& evaluator, const SearchConfig& cfg = {})
        : eval_(evaluator), cfg_(cfg), policy_(cfg_), rng_(cfg.seed) {}

    // Run a search from `root_state` and return the improved policy.
    //
    // When `determinize_root` is set and more than one determinization is
    // requested, the search is repeated over independently sampled futures and
    // the visit distributions are averaged -- the particle form of the chance
    // node in spec 11.3.
    SearchResult search(const Player& root_state, const Player* opponent_state = nullptr,
                        Evaluator* opponent_evaluator = nullptr,
                        bool deliver_attacks = true) {
        const auto started = Clock::now();
        budget_active_ = cfg_.time_budget_ms > 0.0;
        deadline_ = budget_active_
            ? started + std::chrono::duration_cast<Clock::duration>(
                  std::chrono::duration<double, std::milli>(cfg_.time_budget_ms))
            : Clock::time_point{};
        auto finish = [&](SearchResult result) {
            result.elapsed_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - started).count();
            result.telemetry.elapsed_ms = result.elapsed_ms;
            result.time_budget_exhausted = result.time_budget_exhausted || budget_expired();
            budget_active_ = false;
            return result;
        };

        root_player_index_ = root_state.index();
        has_opponent_ = opponent_state != nullptr;
        opponent_eval_ = opponent_evaluator ? opponent_evaluator : &eval_;
        deliver_attacks_ = has_opponent_ && deliver_attacks;

        if (!cfg_.determinize_root || cfg_.determinizations <= 1)
            return finish(search_once(root_state, opponent_state, cfg_.seed));

        SearchResult combined;
        std::vector<double> policy_sum;
        std::vector<int> best_votes;
        double value_sum = 0.0;
        int runs = 0;
        int simulations_sum = 0;
        int nodes_sum = 0;
        int evaluator_calls_sum = 0;
        int positions_sum = 0;
        int transposition_hits_sum = 0;
        SearchTelemetry telemetry_sum;

        for (int d = 0; d < cfg_.determinizations; ++d) {
            const SearchResult r = search_once(
                root_state, opponent_state,
                cfg_.seed + static_cast<std::uint64_t>(d) * 0x9E3779B9ull);
            if (r.best_action < 0) continue;
            if (policy_sum.empty()) {
                policy_sum.assign(r.search_policy.size(), 0.0);
                best_votes.assign(r.search_policy.size(), 0);
                combined = r;
            }
            if (r.search_policy.size() != policy_sum.size()) continue;
            for (size_t i = 0; i < policy_sum.size(); ++i)
                policy_sum[i] += r.search_policy[i];
            if (r.best_action >= 0 &&
                static_cast<size_t>(r.best_action) < best_votes.size())
                ++best_votes[static_cast<size_t>(r.best_action)];
            value_sum += r.value.scalar();
            simulations_sum += r.simulations_run;
            nodes_sum += r.nodes_created;
            evaluator_calls_sum += r.evaluator_calls;
            positions_sum += r.positions_evaluated;
            transposition_hits_sum += r.transposition_hits;
            telemetry_sum.gather_attempts += r.telemetry.gather_attempts;
            telemetry_sum.gathered_leaves += r.telemetry.gathered_leaves;
            telemetry_sum.selection_steps += r.telemetry.selection_steps;
            telemetry_sum.terminal_backups += r.telemetry.terminal_backups;
            telemetry_sum.depth_cutoffs += r.telemetry.depth_cutoffs;
            telemetry_sum.evaluation_flushes += r.telemetry.evaluation_flushes;
            telemetry_sum.max_edge_inflight =
                std::max(telemetry_sum.max_edge_inflight, r.telemetry.max_edge_inflight);
            telemetry_sum.evaluator_elapsed_ms += r.telemetry.evaluator_elapsed_ms;
            telemetry_sum.root_setup_us += r.telemetry.root_setup_us;
            telemetry_sum.gather_us += r.telemetry.gather_us;
            telemetry_sum.backup_us += r.telemetry.backup_us;
            telemetry_sum.finalize_us += r.telemetry.finalize_us;
            telemetry_sum.node_allocation_us += r.telemetry.node_allocation_us;
            telemetry_sum.legal_action_generation_us +=
                r.telemetry.legal_action_generation_us;
            telemetry_sum.state_transition_us += r.telemetry.state_transition_us;
            telemetry_sum.selection_us += r.telemetry.selection_us;
            telemetry_sum.depth_sum += r.telemetry.depth_sum;
            telemetry_sum.depth_samples += r.telemetry.depth_samples;
            telemetry_sum.max_depth = std::max(telemetry_sum.max_depth, r.telemetry.max_depth);
            telemetry_sum.node_budget_cutoffs += r.telemetry.node_budget_cutoffs;
            telemetry_sum.budget_exhausted =
                telemetry_sum.budget_exhausted || r.telemetry.budget_exhausted;
            ++runs;
        }
        if (runs == 0) return finish(search_once(root_state, opponent_state, cfg_.seed));

        combined.search_policy.assign(policy_sum.size(), 0.0f);
        int best = 0;
        for (size_t i = 0; i < policy_sum.size(); ++i) {
            combined.search_policy[i] = static_cast<float>(policy_sum[i] / runs);
            if (combined.search_policy[i] > combined.search_policy[static_cast<size_t>(best)])
                best = static_cast<int>(i);
        }
        if (cfg_.use_gumbel && !best_votes.empty()) {
            // Each particle's Gumbel survivor is the action that search_once()
            // would actually play. Root visit counts can tie at sequential-
            // halving budgets, so averaging visits and taking argmax can throw
            // away a unanimous survivor and fall back to action-index order.
            // Preserve the averaged visit distribution as the training target,
            // but aggregate the executed action by survivor vote. Average visit
            // mass is the deterministic tie-break between equal vote counts.
            best = 0;
            for (size_t i = 1; i < best_votes.size(); ++i) {
                const size_t current = static_cast<size_t>(best);
                if (best_votes[i] > best_votes[current] ||
                    (best_votes[i] == best_votes[current] &&
                     policy_sum[i] > policy_sum[current]))
                    best = static_cast<int>(i);
            }
        }
        combined.best_action = best;
        combined.value = ValueWDL::from_scalar(static_cast<float>(value_sum / runs));
        combined.simulations_run = simulations_sum;
        combined.nodes_created = nodes_sum;
        combined.evaluator_calls = evaluator_calls_sum;
        combined.positions_evaluated = positions_sum;
        combined.transposition_hits = transposition_hits_sum;
        combined.mean_batch_size = evaluator_calls_sum
                                       ? static_cast<double>(positions_sum) / evaluator_calls_sum
                                       : 0.0;
        combined.evaluator_elapsed_ms = telemetry_sum.evaluator_elapsed_ms;
        combined.mean_depth = telemetry_sum.depth_samples
            ? static_cast<double>(telemetry_sum.depth_sum) / telemetry_sum.depth_samples
            : 0.0;
        combined.max_depth = telemetry_sum.max_depth;
        combined.time_budget_exhausted = telemetry_sum.budget_exhausted;
        combined.telemetry = telemetry_sum;
        return finish(combined);
    }

private:
    using Clock = std::chrono::steady_clock;

    bool budget_expired() const {
        return budget_active_ && Clock::now() >= deadline_;
    }

    bool node_budget_reached() const {
        return cfg_.node_budget > 0 &&
               static_cast<int>(nodes_.size()) >= cfg_.node_budget;
    }

    void note_depth(int depth) {
        telemetry_.depth_sum += depth;
        ++telemetry_.depth_samples;
        telemetry_.max_depth = std::max(telemetry_.max_depth, depth);
    }

    SearchResult search_once(const Player& root_state, const Player* opponent_state,
                             std::uint64_t seed) {
        nodes_.clear();
        tt_.clear();
        telemetry_ = SearchTelemetry{};
        rng_.reseed(seed);
        SearchResult result;

        // Determinize: never let the tree read pieces the player cannot see.
        const auto root_setup_started = Clock::now();
        Player rooted = root_state;
        Player rooted_opponent;
        const Player* opponent = nullptr;
        if (opponent_state) {
            rooted_opponent = *opponent_state;
            opponent = &rooted_opponent;
        }
        if (cfg_.determinize_root) {
            rooted.determinize(seed ^ 0xC0FFEEull);
            if (opponent) rooted_opponent.determinize(seed ^ 0xA11CEull);
        }

        const std::int32_t root = create_node(rooted, opponent);
        telemetry_.root_setup_us +=
            std::chrono::duration<double, std::micro>(Clock::now() - root_setup_started).count();
        if (nodes_[root].terminal || nodes_[root].actions.empty()) {
            result.value = ValueWDL::from_scalar(nodes_[root].terminal_value);
            result.telemetry = telemetry_;
            return result;
        }

        // The root is always evaluated on its own first: everything else
        // depends on its priors.
        expand(root);
        if (cfg_.root_noise_fraction > 0.0f) add_root_noise(root);

        if (cfg_.use_gumbel)
            run_gumbel(root, result);
        else
            run_puct(root, result);

        const auto finalize_started = Clock::now();
        fill_result(root, result);
        result.telemetry.finalize_us +=
            std::chrono::duration<double, std::micro>(Clock::now() - finalize_started).count();
        return result;
    }

public:
    const SearchConfig& config() const { return cfg_; }
    void set_config(const SearchConfig& c) { cfg_ = c; }

private:
    // --- node construction -------------------------------------------------
    float terminal_value_for(const Player& state, const Player* opponent,
                             bool current_player_lost = false) const {
        if (!opponent) return -1.0f;
        if (current_player_lost)
            return state.index() == root_player_index_ ? -1.0f : 1.0f;

        const bool root_alive = state.index() == root_player_index_ ? state.alive()
                                                                      : opponent->alive();
        const bool other_alive = state.index() == root_player_index_ ? opponent->alive()
                                                                       : state.alive();
        if (root_alive && !other_alive) return 1.0f;
        if (!root_alive && other_alive) return -1.0f;
        return 0.0f;
    }

    std::int32_t make_terminal(const Player& state, const Player* opponent,
                               bool current_player_lost) {
        const std::int32_t idx = static_cast<std::int32_t>(nodes_.size());
        const auto allocation_started = Clock::now();
        nodes_.emplace_back();
        telemetry_.node_allocation_us +=
            std::chrono::duration<double, std::micro>(Clock::now() - allocation_started).count();
        detail::SearchNode& n = nodes_.back();
        n.state = state;
        if (opponent) {
            n.has_opponent = true;
            n.opponent = *opponent;
        }
        n.terminal = true;
        n.terminal_value = terminal_value_for(state, opponent, current_player_lost);
        return idx;
    }

    std::int32_t create_node(const Player& state, const Player* opponent = nullptr) {
        const std::int32_t idx = static_cast<std::int32_t>(nodes_.size());
        const auto allocation_started = Clock::now();
        nodes_.emplace_back();
        telemetry_.node_allocation_us +=
            std::chrono::duration<double, std::micro>(Clock::now() - allocation_started).count();
        detail::SearchNode& n = nodes_.back();
        n.state = state;
        if (opponent) {
            n.has_opponent = true;
            n.opponent = *opponent;
        }

        const Player* other = n.has_opponent ? &n.opponent : nullptr;
        if (!state.alive() || (other && !other->alive())) {
            n.terminal = true;
            n.terminal_value = terminal_value_for(state, other);
            return idx;
        }

        const RulesetConfig& cfg = state.ruleset();
        const auto legal_started = Clock::now();
        n.actions = movegen_.generate(state.board(), state.active().type, state.hold(),
                                      state.visible_next().empty() ? Piece::None
                                                                   : state.visible_next()[0],
                                      cfg);
        const Tick next_activation = state.garbage().next_activation(state.now());
        if (cfg_.enable_timing_actions && next_activation != TICK_NEVER &&
            !n.actions.empty()) {
            const std::vector<DelayBin> timing_bins{
                DelayBin::Fastest, DelayBin::WaitForEvent
            };
            n.actions = MoveGenerator::expand_delay_bins(
                n.actions, cfg, state.now(), next_activation, TICK_NEVER,
                timing_bins);
        }
        telemetry_.legal_action_generation_us +=
            std::chrono::duration<double, std::micro>(Clock::now() - legal_started).count();
        if (n.actions.empty()) {
            n.terminal = true;
            n.terminal_value = terminal_value_for(state, other,
                                                  /*current_player_lost=*/true);
        }
        return idx;
    }

    void expand(std::int32_t idx) {
        detail::SearchNode& n = nodes_[static_cast<size_t>(idx)];
        if (n.expanded || n.terminal) return;

        const Observation obs = n.has_opponent ? observe(n.state, &n.opponent)
                                                : observe(n.state);
        const auto eval_started = Clock::now();
        const Evaluation ev = evaluator_for(n)->evaluate_one(obs, n.actions);
        telemetry_.evaluator_elapsed_ms +=
            std::chrono::duration<double, std::milli>(Clock::now() - eval_started).count();
        ++eval_calls_;
        ++telemetry_.evaluation_flushes;
        positions_evaluated_ += 1;
        apply_evaluation(idx, ev);
    }

    Evaluator* evaluator_for(const detail::SearchNode& n) {
        if (n.has_opponent && n.state.index() != root_player_index_)
            return opponent_eval_ ? opponent_eval_ : &eval_;
        return &eval_;
    }

    float root_perspective_value(const detail::SearchNode& n, float value) const {
        if (n.has_opponent && n.state.index() != root_player_index_) return -value;
        return value;
    }

    void apply_evaluation(std::int32_t idx, const Evaluation& ev) {
        detail::SearchNode& n = nodes_[static_cast<size_t>(idx)];
        std::vector<float> prior = ev.policy;
        if (prior.size() != n.actions.size())
            prior.assign(n.actions.size(),
                         n.actions.empty() ? 0.0f : 1.0f / static_cast<float>(n.actions.size()));
        n.edges.assign(n.actions.size(), detail::SearchEdge{});
        for (size_t a = 0; a < n.edges.size(); ++a) n.edges[a].prior = prior[a];
        n.expanded = true;
        n.leaf_value = ev.value.scalar();
    }

    // --- selection ---------------------------------------------------------
    int select_puct(const detail::SearchNode& n) const {
        return policy_.select_puct(n, root_player_index_);
    }

    // --- one simulation, stopping at a leaf that needs evaluation ----------
    struct Pending {
        std::int32_t leaf = -1;
        std::vector<std::pair<std::int32_t, int>> path;  // (node, action)
        Observation obs;
        Evaluator* evaluator = nullptr;
        float evaluated_value = 0.0f;
        bool has_evaluated_value = false;
    };

    // Walk from the root to a leaf, applying virtual loss along the way.
    // Returns false when the walk finished at a terminal node (already backed
    // up) rather than at a node needing evaluation.
    bool descend(std::int32_t root, Pending& out) {
        ++telemetry_.gather_attempts;
        std::int32_t idx = root;
        int depth = 0;

        while (true) {
            detail::SearchNode& n = nodes_[static_cast<size_t>(idx)];
            if (n.terminal) {
                ++telemetry_.terminal_backups;
                note_depth(depth);
                backup(out.path, n.terminal_value);
                return false;
            }
            if (!n.expanded) {
                ++telemetry_.gathered_leaves;
                note_depth(depth);
                out.leaf = idx;
                out.obs = n.has_opponent ? observe(n.state, &n.opponent)
                                         : observe(n.state);
                out.evaluator = evaluator_for(n);
                return true;
            }
            if (depth >= cfg_.max_depth) {
                ++telemetry_.depth_cutoffs;
                note_depth(depth);
                backup(out.path, root_perspective_value(n, n.leaf_value));
                return false;
            }

            const auto selection_started = Clock::now();
            const int a = select_puct(n);
            telemetry_.selection_us +=
                std::chrono::duration<double, std::micro>(Clock::now() - selection_started).count();
            ++telemetry_.selection_steps;
            out.path.emplace_back(idx, a);
            detail::SearchEdge& edge = n.edges[static_cast<size_t>(a)];
            edge.inflight += 1;
            telemetry_.max_edge_inflight = std::max(telemetry_.max_edge_inflight, edge.inflight);

            std::int32_t child = edge.child;
            if (child < 0) {
                if (node_budget_reached()) {
                    ++telemetry_.node_budget_cutoffs;
                    note_depth(depth);
                    backup(out.path, root_perspective_value(n, n.leaf_value));
                    return false;
                }
                child = apply_action(idx, a);
                nodes_[static_cast<size_t>(idx)].edges[static_cast<size_t>(a)].child = child;
            }
            idx = child;
            ++depth;
        }
    }

    std::int32_t intern_node(const Player& state, const Player* opponent) {
        if (!cfg_.use_transposition_table) return create_node(state, opponent);

        const std::uint64_t key = detail::search_position_key(state, opponent);
        auto it = tt_.find(key);
        if (it != tt_.end()) {
            ++tt_hits_;
            return it->second;
        }
        const std::int32_t idx = create_node(state, opponent);
        tt_.emplace(key, idx);
        return idx;
    }

    // Apply an action to a copy of the node's state, producing a child node.
    std::int32_t apply_action(std::int32_t parent, int action) {
        const auto transition_started = Clock::now();
        auto finish_transition = [&](std::int32_t child) {
            telemetry_.state_transition_us +=
                std::chrono::duration<double, std::micro>(Clock::now() - transition_started).count();
            return child;
        };
        const detail::SearchNode& parent_node = nodes_[static_cast<size_t>(parent)];
        Player next = parent_node.state;
        Player other;
        const bool has_opponent = parent_node.has_opponent;
        if (has_opponent) other = parent_node.opponent;
        const PlacementAction& a = parent_node.actions[
            static_cast<size_t>(action)];

        if (a.use_hold && !next.do_hold()) {
            // Hold was refused: treat as a dead end rather than silently
            // playing a different move.
            return finish_transition(make_terminal(next, has_opponent ? &other : nullptr,
                                                    /*current_player_lost=*/true));
        }
        next.set_active(a.piece_state());
        int outgoing = 0;
        const LockResult locked = next.lock_piece(a.total_duration(), &outgoing);
        if (!locked.ok && !locked.topped_out)
            return finish_transition(make_terminal(next, has_opponent ? &other : nullptr,
                                                    /*current_player_lost=*/true));

        // Search must use the same event protocol as Arena/self-play: an
        // attack is delivered to the other board, then the next actor is the
        // board whose clock is earlier. The caller can disable delivery for
        // the explicit single-board/no-garbage curriculum.
        if (has_opponent && deliver_attacks_ && outgoing > 0 && other.alive())
            other.receive_attack(outgoing, next.now(), next.index());

        if (has_opponent) {
            const bool other_next =
                other.now() < next.now() ||
                (other.now() == next.now() && other.index() < next.index());
            if (other.alive() && next.alive() && other_next)
                return finish_transition(intern_node(other, &next));
            return finish_transition(intern_node(next, &other));
        }
        return finish_transition(intern_node(next, nullptr));
    }

    void backup(const std::vector<std::pair<std::int32_t, int>>& path, float value) {
        // Values have already been converted to the root player's point of
        // view before reaching this function. In a two-board tree this is
        // what makes an opponent leaf a minimizing node without changing the
        // single-board path representation.
        const auto backup_started = Clock::now();
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            detail::SearchNode& n = nodes_[static_cast<size_t>(it->first)];
            detail::SearchEdge& edge = n.edges[static_cast<size_t>(it->second)];
            edge.visits += 1;
            edge.value_sum += value;
            edge.inflight = std::max(0, edge.inflight - 1);
            n.visits += 1;
            n.value_sum += value;
        }
        telemetry_.backup_us +=
            std::chrono::duration<double, std::micro>(Clock::now() - backup_started).count();
    }

    void revert_virtual_loss(const std::vector<std::pair<std::int32_t, int>>& path) {
        for (const auto& [node, action] : path) {
            detail::SearchEdge& edge =
                nodes_[static_cast<size_t>(node)].edges[static_cast<size_t>(action)];
            edge.inflight = std::max(0, edge.inflight - 1);
        }
    }

    // --- PUCT driver -------------------------------------------------------
    void run_puct(std::int32_t root, SearchResult& result) {
        int done = 0;
        std::vector<Pending> batch;
        batch.reserve(static_cast<size_t>(cfg_.batch_size));

        while (done < cfg_.simulations && !budget_expired() && !node_budget_reached()) {
            batch.clear();
            while (static_cast<int>(batch.size()) < cfg_.batch_size &&
                   done + static_cast<int>(batch.size()) < cfg_.simulations &&
                !budget_expired() && !node_budget_reached()) {
                Pending p;
                const auto gather_started = Clock::now();
                const bool gathered = descend(root, p);
                telemetry_.gather_us +=
                    std::chrono::duration<double, std::micro>(Clock::now() - gather_started).count();
                if (gathered) {
                    batch.push_back(std::move(p));
                } else {
                    ++done;
                    if (done >= cfg_.simulations) break;
                }
                // Guard against a tree that can only reach terminals.
                if (batch.empty() && done >= cfg_.simulations) break;
            }
            if (batch.empty()) break;

            flush(batch);
            done += static_cast<int>(batch.size());
        }
        telemetry_.budget_exhausted = budget_expired();
        result.simulations_run = done;
    }

    // LC3 pipeline stage 2: evaluate gathered leaves. No tree backup occurs in
    // this stage, which keeps inference batching independent of backup policy.
    void evaluate_batch(std::vector<Pending>& batch) {
        struct Group {
            Evaluator* evaluator = nullptr;
            std::vector<size_t> items;
        };
        std::vector<Group> groups;
        for (size_t i = 0; i < batch.size(); ++i) {
            Evaluator* evaluator = batch[i].evaluator ? batch[i].evaluator : &eval_;
            auto it = std::find_if(groups.begin(), groups.end(),
                                   [&](const Group& g) { return g.evaluator == evaluator; });
            if (it == groups.end()) {
                groups.push_back(Group{evaluator, {}});
                it = std::prev(groups.end());
            }
            it->items.push_back(i);
        }

        for (auto& group : groups) {
            std::vector<EvalRequest> requests;
            requests.reserve(group.items.size());
            for (size_t item : group.items) {
                const Pending& p = batch[item];
                requests.push_back(
                    EvalRequest{&p.obs, &nodes_[static_cast<size_t>(p.leaf)].actions});
            }

            std::vector<Evaluation> out;
            const auto eval_started = Clock::now();
            group.evaluator->evaluate(requests, out);
            telemetry_.evaluator_elapsed_ms +=
                std::chrono::duration<double, std::milli>(Clock::now() - eval_started).count();
            ++eval_calls_;
            ++telemetry_.evaluation_flushes;
            positions_evaluated_ += static_cast<int>(group.items.size());

            for (size_t j = 0; j < group.items.size(); ++j) {
                const size_t item = group.items[j];
                Pending& pending = batch[item];
                const std::int32_t leaf = pending.leaf;
                // The same leaf can appear twice in one batch via a
                // transposition; initialise its node once but preserve the
                // evaluator result associated with every gathered path.
                if (!nodes_[static_cast<size_t>(leaf)].expanded && j < out.size())
                    apply_evaluation(leaf, out[j]);
                pending.evaluated_value =
                    (j < out.size()) ? out[j].value.scalar()
                                     : nodes_[static_cast<size_t>(leaf)].leaf_value;
                pending.has_evaluated_value = true;
            }
        }
    }

    // LC3 pipeline stage 3: backpropagate completed evaluations. This is kept
    // separate from evaluate_batch so a future streaming scheduler can overlap
    // Gather(N+1) with Eval(N) without changing edge semantics.
    void backpropagate_batch(std::vector<Pending>& batch) {
        for (Pending& pending : batch) {
            if (!pending.has_evaluated_value) {
                revert_virtual_loss(pending.path);
                continue;
            }
            const detail::SearchNode& leaf = nodes_[static_cast<size_t>(pending.leaf)];
            backup(pending.path, root_perspective_value(leaf, pending.evaluated_value));
        }
    }

    void flush(std::vector<Pending>& batch) {
        evaluate_batch(batch);
        backpropagate_batch(batch);
    }

    // --- Gumbel sequential halving (spec 11.2) -----------------------------
    // At low simulation counts, sampling m actions without replacement using
    // Gumbel noise and then halving gives a policy improvement guarantee that
    // plain PUCT visit counts do not.
    void run_gumbel(std::int32_t root, SearchResult& result) {
        detail::SearchNode& r = nodes_[static_cast<size_t>(root)];
        const int n_actions = static_cast<int>(r.actions.size());
        if (n_actions <= 1) {
            result.simulations_run = 0;
            return;
        }

        // Gumbel-top-k: g(a) + logits(a) picks m actions without replacement.
        std::vector<float> logits(static_cast<size_t>(n_actions));
        std::vector<float> gumbel(static_cast<size_t>(n_actions));
        for (int a = 0; a < n_actions; ++a) {
            const float p = std::max(1e-9f, r.edges[static_cast<size_t>(a)].prior);
            logits[static_cast<size_t>(a)] = std::log(p);
            gumbel[static_cast<size_t>(a)] = cfg_.gumbel_noise_scale * sample_gumbel();
        }
        gumbel_logits_ = gumbel;

        std::vector<int> candidates(static_cast<size_t>(n_actions));
        for (int a = 0; a < n_actions; ++a) candidates[static_cast<size_t>(a)] = a;
        std::sort(candidates.begin(), candidates.end(), [&](int x, int y) {
            return gumbel[static_cast<size_t>(x)] + logits[static_cast<size_t>(x)] >
                   gumbel[static_cast<size_t>(y)] + logits[static_cast<size_t>(y)];
        });

        int m = std::min(cfg_.gumbel_m, n_actions);
        candidates.resize(static_cast<size_t>(m));

        int used = 0;
        const int budget = cfg_.simulations;

        // Sequential halving: repeatedly give every surviving candidate an
        // equal share of the remaining budget, then keep the best half.
        while (m > 1 && used < budget && !budget_expired() && !node_budget_reached()) {
            const int rounds = std::max(1, (budget - used) / (m * std::max(1, log2_ceil(m))));
            for (int i = 0; i < rounds && used < budget && !budget_expired() &&
                 !node_budget_reached(); ++i) {
                std::vector<Pending> batch;
                for (int a : candidates) {
                    if (used >= budget || budget_expired() || node_budget_reached()) break;
                    Pending p;
                    const auto gather_started = Clock::now();
                    const bool gathered = visit_root_action(root, a, p);
                    telemetry_.gather_us +=
                        std::chrono::duration<double, std::micro>(Clock::now() - gather_started).count();
                    if (gathered) {
                        batch.push_back(std::move(p));
                        if (static_cast<int>(batch.size()) >= cfg_.batch_size) {
                            flush(batch);
                            batch.clear();
                        }
                    }
                    ++used;
                }
                if (!batch.empty()) flush(batch);
            }

            // Keep the top half by the Gumbel-adjusted score.
            std::sort(candidates.begin(), candidates.end(), [&](int x, int y) {
                return gumbel_score(root, x, logits, gumbel) >
                       gumbel_score(root, y, logits, gumbel);
            });
            m = std::max(1, m / 2);
            candidates.resize(static_cast<size_t>(m));
        }

        // Spend anything left on the survivor.
        while (used < budget && !candidates.empty() && !budget_expired() &&
               !node_budget_reached()) {
            std::vector<Pending> batch;
            const int take = std::min(cfg_.batch_size, budget - used);
            for (int i = 0; i < take; ++i) {
                Pending p;
                const auto gather_started = Clock::now();
                const bool gathered = visit_root_action(root, candidates.front(), p);
                telemetry_.gather_us +=
                    std::chrono::duration<double, std::micro>(Clock::now() - gather_started).count();
                if (gathered) batch.push_back(std::move(p));
                ++used;
            }
            if (!batch.empty()) flush(batch);
            else break;
        }

        result.simulations_run = used;
        telemetry_.budget_exhausted = budget_expired();
        gumbel_selected_ = candidates.empty() ? -1 : candidates.front();
    }

    // Descend from the root having already committed to `action`.
    bool visit_root_action(std::int32_t root, int action, Pending& out) {
        ++telemetry_.gather_attempts;
        detail::SearchNode& r = nodes_[static_cast<size_t>(root)];
        out.path.emplace_back(root, action);
        detail::SearchEdge& root_edge = r.edges[static_cast<size_t>(action)];
        root_edge.inflight += 1;
        telemetry_.max_edge_inflight =
            std::max(telemetry_.max_edge_inflight, root_edge.inflight);
        ++telemetry_.selection_steps;

        std::int32_t child = root_edge.child;
        if (child < 0) {
            child = apply_action(root, action);
            nodes_[static_cast<size_t>(root)].edges[static_cast<size_t>(action)].child = child;
        }

        // Continue down normally from the child.
        std::int32_t idx = child;
        int depth = 1;
        while (true) {
            detail::SearchNode& n = nodes_[static_cast<size_t>(idx)];
            if (n.terminal) {
                ++telemetry_.terminal_backups;
                note_depth(depth);
                backup(out.path, n.terminal_value);
                return false;
            }
            if (!n.expanded) {
                ++telemetry_.gathered_leaves;
                note_depth(depth);
                out.leaf = idx;
                out.obs = n.has_opponent ? observe(n.state, &n.opponent)
                                         : observe(n.state);
                out.evaluator = evaluator_for(n);
                return true;
            }
            if (depth >= cfg_.max_depth) {
                ++telemetry_.depth_cutoffs;
                note_depth(depth);
                backup(out.path, root_perspective_value(n, n.leaf_value));
                return false;
            }
            const auto selection_started = Clock::now();
            const int a = select_puct(n);
            telemetry_.selection_us +=
                std::chrono::duration<double, std::micro>(Clock::now() - selection_started).count();
            ++telemetry_.selection_steps;
            out.path.emplace_back(idx, a);
            detail::SearchEdge& edge = n.edges[static_cast<size_t>(a)];
            edge.inflight += 1;
            telemetry_.max_edge_inflight = std::max(telemetry_.max_edge_inflight, edge.inflight);
            std::int32_t next = edge.child;
            if (next < 0) {
                if (node_budget_reached()) {
                    ++telemetry_.node_budget_cutoffs;
                    note_depth(depth);
                    backup(out.path, root_perspective_value(n, n.leaf_value));
                    return false;
                }
                next = apply_action(idx, a);
                nodes_[static_cast<size_t>(idx)].edges[static_cast<size_t>(a)].child = next;
            }
            idx = next;
            ++depth;
        }
    }

    // Gumbel AlphaZero's ranking function: g(a) + logit(a) + sigma(q(a)).
    //
    // Two details matter and both were got wrong first time round, with the
    // measured effect that Gumbel played far *worse* than its own priors
    // (47 pieces per game against 200 for policy-only):
    //
    //   1. Unvisited actions must not be scored as if q = 0. Values here are
    //      mostly negative, so a q of 0 made every unexplored action look
    //      better than every explored one, and the halving kept discarding the
    //      actions it had actually learned something about. Unvisited actions
    //      inherit the parent's value instead.
    //
    //   2. sigma must actually dominate once visits accumulate. It is
    //      deliberately scaled by (c_visit + max_visits) so that a well-explored
    //      action's measured value outweighs both its prior and the noise;
    //      normalising that factor away (an earlier attempt) left sigma at ~0.02
    //      against a noise range of ~1.0, so q was ignored roughly 50:1 and the
    //      halving discarded the best action at every simulation count.
    float gumbel_score(std::int32_t root, int a, const std::vector<float>& logits,
                       const std::vector<float>& gumbel) const {
        const detail::SearchNode& r = nodes_[static_cast<size_t>(root)];
        return policy_.gumbel_score(r, a, logits, gumbel, max_child_visits(root));
    }

    int max_child_visits(std::int32_t idx) const {
        const detail::SearchNode& n = nodes_[static_cast<size_t>(idx)];
        int m = 0;
        for (const detail::SearchEdge& edge : n.edges) m = std::max(m, edge.visits);
        return m;
    }

    static int log2_ceil(int n) {
        int k = 0;
        while ((1 << k) < n) ++k;
        return std::max(1, k);
    }

    float sample_gumbel() {
        // -log(-log(U)) with U in (0,1), from the engine RNG so the search is
        // reproducible from a seed (spec 19.4).
        const double u =
            (static_cast<double>(rng_.next_u64() >> 11) + 0.5) / 9007199254740992.0;
        return static_cast<float>(-std::log(-std::log(u)));
    }

    void add_root_noise(std::int32_t root) {
        detail::SearchNode& n = nodes_[static_cast<size_t>(root)];
        if (n.edges.empty()) return;
        // Gamma(alpha,1) samples normalised to a Dirichlet draw.
        std::vector<float> noise(n.edges.size());
        float sum = 0.0f;
        for (auto& v : noise) {
            const double u =
                (static_cast<double>(rng_.next_u64() >> 11) + 0.5) / 9007199254740992.0;
            v = static_cast<float>(std::pow(u, 1.0 / cfg_.root_noise_alpha));
            sum += v;
        }
        if (sum <= 0.0f) return;
        const float f = cfg_.root_noise_fraction;
        for (size_t i = 0; i < n.edges.size(); ++i)
            n.edges[i].prior = (1.0f - f) * n.edges[i].prior + f * (noise[i] / sum);
    }

    // --- results -----------------------------------------------------------
    void fill_result(std::int32_t root, SearchResult& result) {
        const detail::SearchNode& n = nodes_[static_cast<size_t>(root)];
        result.candidates.reserve(n.actions.size());

        int total_visits = 0;
        for (const detail::SearchEdge& edge : n.edges) total_visits += edge.visits;

        result.search_policy.assign(n.actions.size(), 0.0f);
        int best = -1;
        int best_visits = -1;
        int raw_policy_action = -1;
        float raw_policy_prior = -1.0f;
        for (size_t a = 0; a < n.actions.size(); ++a) {
            const detail::SearchEdge& edge = n.edges[a];
            SearchCandidate c;
            c.action_index = static_cast<int>(a);
            c.prior = edge.prior;
            c.visits = edge.visits;
            c.q_value = edge.q();
            result.candidates.push_back(c);
            if (c.prior > raw_policy_prior) {
                raw_policy_prior = c.prior;
                raw_policy_action = static_cast<int>(a);
            }
            if (total_visits > 0)
                result.search_policy[a] =
                    static_cast<float>(c.visits) / static_cast<float>(total_visits);
            if (c.visits > best_visits) {
                best_visits = c.visits;
                best = static_cast<int>(a);
            }
        }

        // Gumbel selects its surviving candidate rather than the most-visited
        // action; with one simulation those can differ.
        if (cfg_.use_gumbel && gumbel_selected_ >= 0) best = gumbel_selected_;

        // If nothing was searched at all, fall back to the prior.
        if (best_visits <= 0 && !n.edges.empty()) {
            best = 0;
            for (size_t a = 1; a < n.edges.size(); ++a)
                if (n.edges[a].prior > n.edges[static_cast<size_t>(best)].prior)
                    best = static_cast<int>(a);
            result.search_policy.resize(n.edges.size());
            for (size_t a = 0; a < n.edges.size(); ++a)
                result.search_policy[a] = n.edges[a].prior;
        }

        result.best_action = best;
        result.raw_policy_action = raw_policy_action;
        result.value = ValueWDL::from_scalar(
            n.visits > 0 ? n.value_sum / static_cast<float>(n.visits) : n.leaf_value);
        result.nodes_created = static_cast<int>(nodes_.size());
        result.evaluator_calls = eval_calls_;
        result.positions_evaluated = positions_evaluated_;
        result.transposition_hits = tt_hits_;
        result.mean_batch_size =
            eval_calls_ ? static_cast<double>(positions_evaluated_) / eval_calls_ : 0.0;
        result.evaluator_elapsed_ms = telemetry_.evaluator_elapsed_ms;
        result.mean_depth = telemetry_.depth_samples
            ? static_cast<double>(telemetry_.depth_sum) / telemetry_.depth_samples
            : 0.0;
        result.max_depth = telemetry_.max_depth;
        result.time_budget_exhausted = budget_expired();
        result.telemetry = telemetry_;

        eval_calls_ = 0;
        positions_evaluated_ = 0;
        tt_hits_ = 0;
        telemetry_ = SearchTelemetry{};
        gumbel_selected_ = -1;
    }

    Evaluator& eval_;
    SearchConfig cfg_;
    SearchPolicy policy_;
    Rng rng_;
    MoveGenerator movegen_;
    std::vector<detail::SearchNode> nodes_;
    std::unordered_map<std::uint64_t, std::int32_t> tt_;
    std::vector<float> gumbel_logits_;
    int gumbel_selected_ = -1;
    int eval_calls_ = 0;
    int positions_evaluated_ = 0;
    int tt_hits_ = 0;
    SearchTelemetry telemetry_;
    Clock::time_point deadline_{};
    bool budget_active_ = false;
    int root_player_index_ = 0;
    bool has_opponent_ = false;
    bool deliver_attacks_ = false;
    Evaluator* opponent_eval_ = nullptr;
};

}  // namespace tetra
