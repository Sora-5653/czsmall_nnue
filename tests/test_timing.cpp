// SPDX-License-Identifier: MIT
// Action duration and timing actions (spec 8.4).
#include "test_util.hpp"
#include "tetra/movegen.hpp"
#include "tetra/player.hpp"
#include "tetra/tokenizer.hpp"

#include <algorithm>
#include <map>
#include <set>

using namespace tetra;

namespace {

RulesetConfig league() { return RulesetConfig::tetra_league(); }

Board board_from(std::initializer_list<const char*> rows_top_first, int width = 10,
                 int height = 40) {
    Board b(width, height);
    const int n = static_cast<int>(rows_top_first.size());
    int y = n - 1;
    for (const char* row : rows_top_first) {
        for (int x = 0; x < width; ++x)
            if (row[x] == 'X' || row[x] == 'G') b.fill_cell(x, y, row[x] == 'G');
        --y;
    }
    return b;
}

}  // namespace

TEST(handling_model_reads_the_ruleset) {
    RulesetConfig cfg = league();
    cfg.movement.das = 10;
    cfg.movement.arr = 2;
    cfg.movement.sdf = 3;
    cfg.movement.are = 4;
    cfg.clear_rules.line_clear_delay = 5;
    const HandlingModel h = HandlingModel::from(cfg);
    CHECK_EQ(h.das, 10);
    CHECK_EQ(h.arr, 2);
    CHECK_EQ(h.sdf, 3);
    CHECK_EQ(h.are, 4);
    CHECK_EQ(h.line_clear_delay, 5);
}

TEST(das_shift_cost_is_monotonic_and_beats_taps_when_far) {
    RulesetConfig cfg = league();
    cfg.movement.das = 6;
    cfg.movement.arr = 1;
    const HandlingModel h = HandlingModel::from(cfg);

    CHECK_EQ(h.das_shift_cost(0), 0);
    CHECK_EQ(h.das_shift_cost(1), h.tap);  // a single cell is just a tap
    // Cost must never decrease as the distance grows.
    Tick prev = 0;
    for (int cells = 0; cells <= 12; ++cells) {
        const Tick c = h.das_shift_cost(cells);
        CHECK_MSG(c >= prev, "DAS cost must be monotonic in distance");
        prev = c;
    }
    // With ARR = 0 a long slide collapses to one repeat frame.
    RulesetConfig instant = league();
    instant.movement.arr = 0;
    const HandlingModel hi = HandlingModel::from(instant);
    CHECK_EQ(hi.das_shift_cost(9), hi.tap + hi.das + 1);
}

TEST(soft_drop_cost_respects_sdf) {
    RulesetConfig infinite = league();
    infinite.movement.sdf = 0;  // instant
    CHECK_EQ(HandlingModel::from(infinite).soft_drop_cost(20), 1);

    RulesetConfig slow = league();
    slow.movement.sdf = 2;
    const HandlingModel h = HandlingModel::from(slow);
    CHECK_EQ(h.soft_drop_cost(0), 0);
    CHECK_EQ(h.soft_drop_cost(5), 10);
}

TEST(every_generated_action_has_a_positive_duration) {
    const RulesetConfig cfg = league();
    MoveGenerator gen;
    Rng rng(31337);
    for (int trial = 0; trial < 25; ++trial) {
        Board b(10, 40);
        for (int y = 0; y < 7; ++y)
            for (int x = 0; x < 10; ++x)
                if (rng.chance(1, 3)) b.fill_cell(x, y, false);
        for (int pi = 0; pi < PIECE_COUNT; ++pi) {
            for (const auto& a : gen.generate_for_piece(b, static_cast<Piece>(pi), cfg, false)) {
                CHECK_MSG(a.base_duration > 0, "a placement must cost time");
                CHECK_EQ(a.delay_ticks, 0);  // the raw generator emits FASTEST
                CHECK(a.delay_bin == DelayBin::Fastest);
                CHECK_EQ(a.total_duration(), a.base_duration);
            }
        }
    }
}

TEST(duration_matches_replaying_the_canonical_sequence) {
    // The cost model and the movement model must never disagree: re-executing
    // the emitted sequence has to reproduce both the position and the price.
    const RulesetConfig cfg = league();
    MoveGenerator gen;
    Rng rng(918273);
    int verified = 0;

    for (int trial = 0; trial < 25; ++trial) {
        Board b(10, 40);
        for (int y = 0; y < 6; ++y)
            for (int x = 0; x < 10; ++x)
                if (rng.chance(1, 3)) b.fill_cell(x, y, false);

        for (int pi = 0; pi < PIECE_COUNT; ++pi) {
            const Piece p = static_cast<Piece>(pi);
            for (const auto& a : gen.generate_for_piece(b, p, cfg, false)) {
                const ActivePiece spawn = spawn_piece(p, cfg);
                const ExecutionResult r =
                    execute_inputs(b, spawn, a.canonical_input_sequence, cfg);
                CHECK(r.ok);
                // Position must match.
                CHECK_EQ(r.piece.x, a.final_x);
                CHECK_EQ(r.piece.y, a.final_y);
                CHECK(r.piece.rot == a.final_rotation);
                // Cost must match, once the line-clear delay is added back.
                const Tick expected =
                    r.cost + (a.cleared_lines > 0 ? cfg.clear_rules.line_clear_delay : 0);
                CHECK_MSG(expected == a.base_duration,
                          "cost model drifted from the movement model: " +
                              std::to_string(expected) + " vs " +
                              std::to_string(a.base_duration));
                ++verified;
            }
        }
    }
    CHECK(verified > 500);
}

TEST(harder_placements_cost_more_than_easy_ones) {
    // Sanity of the ordering: dropping straight down must be cheaper than
    // sliding to the far wall and rotating.
    const RulesetConfig cfg = league();
    MoveGenerator gen;
    Board b(10, 40);
    const auto acts = gen.generate_for_piece(b, Piece::T, cfg, false);
    CHECK(!acts.empty());

    Tick cheapest = TICK_NEVER, dearest = 0;
    for (const auto& a : acts) {
        cheapest = std::min(cheapest, a.base_duration);
        dearest = std::max(dearest, a.base_duration);
    }
    CHECK_MSG(dearest > cheapest, "placements must not all cost the same");

    // The spawn-column, spawn-rotation placement should be among the cheapest.
    const ActivePiece spawn = spawn_piece(Piece::T, cfg);
    for (const auto& a : acts) {
        if (a.final_x == spawn.x && a.final_rotation == Rot::N)
            CHECK_MSG(a.base_duration <= cheapest + 2,
                      "the straight-down drop should be near-cheapest");
    }
}

TEST(line_clear_delay_is_charged_to_clearing_placements) {
    RulesetConfig cfg = league();
    cfg.clear_rules.line_clear_delay = 20;
    MoveGenerator gen;

    // A well that a vertical I fills to clear four rows.
    Board b = board_from({
        "XXXX.XXXXX",
        "XXXX.XXXXX",
        "XXXX.XXXXX",
        "XXXX.XXXXX",
    });
    const auto acts = gen.generate_for_piece(b, Piece::I, cfg, false);
    bool checked = false;
    for (const auto& a : acts) {
        if (a.cleared_lines == 4) {
            // Re-price without the clear delay and confirm the difference.
            const ActivePiece spawn = spawn_piece(Piece::I, cfg);
            const ExecutionResult r = execute_inputs(b, spawn, a.canonical_input_sequence, cfg);
            CHECK_EQ(a.base_duration, r.cost + 20);
            checked = true;
        }
    }
    CHECK_MSG(checked, "expected a quad placement to price the clear delay");
}

TEST(are_is_charged_to_every_placement) {
    RulesetConfig fast = league();
    RulesetConfig slow = league();
    slow.movement.are = 12;

    MoveGenerator gen;
    Board b(10, 40);
    const auto a = gen.generate_for_piece(b, Piece::O, fast, false);
    const auto c = gen.generate_for_piece(b, Piece::O, slow, false);
    CHECK_EQ(a.size(), c.size());
    for (size_t i = 0; i < a.size() && i < c.size(); ++i)
        CHECK_EQ(c[i].base_duration, a[i].base_duration + 12);
}

TEST(handling_settings_change_action_costs) {
    // A custom room with sluggish handling must produce slower actions, which
    // is the whole reason handling lives in the ruleset.
    RulesetConfig quick = league();
    quick.movement.das = 1;
    quick.movement.arr = 0;

    RulesetConfig sluggish = league();
    sluggish.movement.das = 20;
    sluggish.movement.arr = 5;

    MoveGenerator gen;
    Board b(10, 40);
    Tick sum_quick = 0, sum_slow = 0;
    for (const auto& a : gen.generate_for_piece(b, Piece::T, quick, false))
        sum_quick += a.base_duration;
    for (const auto& a : gen.generate_for_piece(b, Piece::T, sluggish, false))
        sum_slow += a.base_duration;
    CHECK_MSG(sum_slow > sum_quick, "worse handling must cost more time overall");
}

TEST(merging_keeps_the_cheapest_execution) {
    const RulesetConfig cfg = league();
    MoveGenerator gen;
    Rng rng(4242);
    for (int trial = 0; trial < 20; ++trial) {
        Board b(10, 40);
        for (int y = 0; y < 6; ++y)
            for (int x = 0; x < 10; ++x)
                if (rng.chance(1, 3)) b.fill_cell(x, y, false);

        // Generate merged and unmerged, then confirm that for every merged
        // action there is no cheaper unmerged action with the same outcome.
        MoveGenerator raw{MoveGenerator::Options{true, true, false, 20000}};
        for (int pi = 0; pi < PIECE_COUNT; ++pi) {
            const Piece p = static_cast<Piece>(pi);
            const auto merged = gen.generate_for_piece(b, p, cfg, false);
            const auto all = raw.generate_for_piece(b, p, cfg, false);
            for (const auto& m : merged) {
                Tick best = m.base_duration;
                for (const auto& r : all) {
                    if (r.resulting_board_hash == m.resulting_board_hash &&
                        r.cleared_lines == m.cleared_lines && r.spin == m.spin &&
                        r.use_hold == m.use_hold)
                        best = std::min(best, r.base_duration);
                }
                CHECK_MSG(m.base_duration == best,
                          "merging must retain the cheapest execution: kept " +
                              std::to_string(m.base_duration) + ", best " +
                              std::to_string(best));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Delay bins
// ---------------------------------------------------------------------------

TEST(delay_bin_ticks_are_the_specified_values) {
    CHECK_EQ(delay_bin_ticks(DelayBin::Fastest), 0);
    CHECK_EQ(delay_bin_ticks(DelayBin::Plus1F), 1);
    CHECK_EQ(delay_bin_ticks(DelayBin::Plus2F), 2);
    CHECK_EQ(delay_bin_ticks(DelayBin::Plus4F), 4);
    CHECK_EQ(delay_bin_ticks(DelayBin::Plus8F), 8);
}

TEST(wait_for_event_is_always_bounded) {
    // Spec 8.4: WAIT_FOR_EVENT is never an open-ended wait.
    const RulesetConfig cfg = league();
    const HandlingModel h = HandlingModel::from(cfg);

    // Nothing to wait for: bounded by the lock delay / max wait.
    const Tick none = resolve_wait_for_event(100, TICK_NEVER, TICK_NEVER, h, 60);
    CHECK(none > 0);
    CHECK(none <= 60);
    CHECK(none <= h.lock_delay);

    // Garbage activation comes first.
    CHECK_EQ(resolve_wait_for_event(100, 110, TICK_NEVER, h, 60), 10);
    // The opponent's lock comes first.
    CHECK_EQ(resolve_wait_for_event(100, 140, 105, h, 60), 5);
    // Events already in the past do not produce negative waits.
    CHECK(resolve_wait_for_event(100, 50, 60, h, 60) >= 0);
    // The explicit maximum always wins.
    CHECK(resolve_wait_for_event(100, 10000, 10000, h, 3) <= 3);

    // Never negative, whatever the inputs.
    Rng rng(7);
    for (int i = 0; i < 2000; ++i) {
        const Tick now = static_cast<Tick>(rng.below(1000));
        const Tick g = static_cast<Tick>(rng.below(2000));
        const Tick o = static_cast<Tick>(rng.below(2000));
        const Tick w = resolve_wait_for_event(now, g, o, h, 60);
        CHECK(w >= 0);
        CHECK(w <= 60);
    }
}

TEST(expand_delay_bins_produces_distinct_waits) {
    const RulesetConfig cfg = league();
    MoveGenerator gen;
    Board b(10, 40);
    const auto base = gen.generate_for_piece(b, Piece::O, cfg, false);
    CHECK(!base.empty());

    const auto expanded = MoveGenerator::expand_delay_bins(base, cfg, /*now=*/0,
                                                           /*next_garbage=*/30,
                                                           /*opp_lock=*/TICK_NEVER);
    CHECK(expanded.size() > base.size());

    // Group by the underlying placement and check the waits are unique.
    for (const auto& b0 : base) {
        std::set<Tick> waits;
        for (const auto& e : expanded) {
            if (e.final_x == b0.final_x && e.final_rotation == b0.final_rotation &&
                e.final_piece == b0.final_piece && e.final_y == b0.final_y) {
                CHECK_MSG(waits.insert(e.delay_ticks).second,
                          "duplicate wait for the same placement");
                // The base cost is untouched; only the delay differs.
                CHECK_EQ(e.base_duration, b0.base_duration);
                CHECK_EQ(e.total_duration(), b0.base_duration + e.delay_ticks);
            }
        }
        CHECK(waits.size() >= 2);
        CHECK(waits.count(0) == 1);  // FASTEST is always available
    }
}

TEST(expand_delay_bins_drops_a_pointless_wait) {
    // With nothing to wait for, WAIT_FOR_EVENT must not duplicate a fixed bin.
    const RulesetConfig cfg = league();
    MoveGenerator gen;
    Board b(10, 40);
    const auto base = gen.generate_for_piece(b, Piece::O, cfg, false);

    // A garbage activation exactly 2 ticks away collides with the +2F bin.
    const auto expanded =
        MoveGenerator::expand_delay_bins(base, cfg, 0, /*next_garbage=*/2, TICK_NEVER,
                                         {DelayBin::Fastest, DelayBin::Plus2F,
                                          DelayBin::WaitForEvent});
    for (const auto& b0 : base) {
        int matches = 0;
        for (const auto& e : expanded)
            if (e.final_x == b0.final_x && e.final_rotation == b0.final_rotation &&
                e.final_y == b0.final_y && e.delay_ticks == 2)
                ++matches;
        CHECK_MSG(matches == 1, "a wait equal to a fixed bin must be deduplicated");
    }
}

TEST(delay_bins_are_deterministic) {
    const RulesetConfig cfg = league();
    MoveGenerator gen;
    Board b(10, 40);
    const auto base = gen.generate_for_piece(b, Piece::T, cfg, false);
    const auto a = MoveGenerator::expand_delay_bins(base, cfg, 10, 40, 25);
    const auto c = MoveGenerator::expand_delay_bins(base, cfg, 10, 40, 25);
    CHECK_EQ(a.size(), c.size());
    for (size_t i = 0; i < a.size() && i < c.size(); ++i) {
        CHECK(a[i].delay_bin == c[i].delay_bin);
        CHECK_EQ(a[i].delay_ticks, c[i].delay_ticks);
        CHECK_EQ(a[i].total_duration(), c[i].total_duration());
    }
}

// ---------------------------------------------------------------------------
// The behaviour this all exists to enable (spec 12)
// ---------------------------------------------------------------------------

TEST(waiting_lets_garbage_activate_before_the_next_placement) {
    // The mechanism behind "deliberately take the garbage": if the player acts
    // fast the garbage is not active yet and cannot rise; if the player waits
    // past the activation tick, it does. Nothing about this is hardcoded --
    // it falls out of the clock.
    RulesetConfig cfg = league();
    cfg.garbage.travel_time = 0;
    cfg.garbage.activation_delay = 30;

    auto run = [&](Tick extra_delay) {
        Player p;
        p.reset(cfg, 4242, 0);
        p.receive_attack(4, 0, 1);
        // A placement that clears nothing, executed either immediately or late.
        ActivePiece a = p.active();
        a.y -= hard_drop_distance(p.board(), a);
        p.set_active(a);
        int out = 0;
        return p.lock_piece(5 + extra_delay, &out).garbage_received;
    };

    CHECK_EQ(run(0), 0);    // acted before activation: nothing rises
    CHECK_EQ(run(40), 4);   // waited past activation: the garbage lands
}

TEST(wait_for_event_targets_the_activation_tick) {
    // End to end: the wait resolved for a real pending-garbage state must be
    // exactly long enough to reach activation.
    RulesetConfig cfg = league();
    cfg.garbage.travel_time = 10;
    cfg.garbage.activation_delay = 20;

    Player p;
    p.reset(cfg, 1, 0);
    p.receive_attack(3, 0, 1);
    const Tick activation = p.garbage().next_activation(p.now());
    CHECK_EQ(activation, 30);

    const HandlingModel h = HandlingModel::from(cfg);
    const Tick wait = resolve_wait_for_event(p.now(), activation, TICK_NEVER, h, /*max=*/60);
    CHECK_EQ(p.now() + wait, activation);
}

TEST(delay_bins_are_visible_to_the_model) {
    // The action embedding must distinguish the bins, otherwise the policy
    // cannot express a timing decision at all.
    const RulesetConfig cfg = league();
    MoveGenerator gen;
    Board b(10, 40);
    const auto base = gen.generate_for_piece(b, Piece::O, cfg, false);
    const auto expanded = MoveGenerator::expand_delay_bins(base, cfg, 0, 45, TICK_NEVER);

    std::set<std::string> seen;
    for (const auto& a : expanded) {
        const ActionEmbedding e = embed_action(a, b, cfg);
        std::string key;
        for (int k = 0; k < ACTION_FEATURES; ++k)
            key += std::to_string(static_cast<int>(e.f[static_cast<size_t>(k)] * 10000)) + ",";
        CHECK_MSG(seen.insert(key).second,
                  "delayed variants must have distinguishable embeddings");
    }
}

TEST(merged_action_coordinates_match_their_input_sequence) {
    // Regression test. Two different (x, y, rot) triples can place the exact
    // same cells -- an I piece in rotation N at row y fills the same cells as
    // rotation 2 at row y+1 -- so they hash to the same outcome and get merged.
    // An earlier version of the merge copied the cheaper input sequence but
    // kept the other candidate's coordinates, leaving 162 of 5191 actions whose
    // canonical_input_sequence did not reproduce their own final position.
    const RulesetConfig cfg = league();
    MoveGenerator gen;
    Rng rng(918273);
    int checked = 0;

    for (int trial = 0; trial < 25; ++trial) {
        Board b(10, 40);
        for (int y = 0; y < 6; ++y)
            for (int x = 0; x < 10; ++x)
                if (rng.chance(1, 3)) b.fill_cell(x, y, false);

        for (int pi = 0; pi < PIECE_COUNT; ++pi) {
            const Piece p = static_cast<Piece>(pi);
            for (const auto& a : gen.generate_for_piece(b, p, cfg, false)) {
                const ExecutionResult r =
                    execute_inputs(b, spawn_piece(p, cfg), a.canonical_input_sequence, cfg);
                CHECK(r.ok);
                CHECK_MSG(r.piece.x == a.final_x && r.piece.y == a.final_y &&
                              r.piece.rot == a.final_rotation,
                          "a merged action must describe the placement its own "
                          "inputs produce");
                ++checked;
            }
        }
    }
    CHECK(checked > 2000);
}

TEST(hold_actions_are_priced_and_reproducible) {
    // Hold placements start from the swapped-in piece's spawn, and the Hold
    // input itself has to be paid for.
    const RulesetConfig cfg = league();
    MoveGenerator gen;
    Board b(10, 40);

    const auto acts = gen.generate(b, Piece::S, Piece::I, Piece::T, cfg);
    int hold_actions = 0;
    for (const auto& a : acts) {
        if (!a.use_hold) continue;
        ++hold_actions;
        CHECK(a.final_piece == Piece::I);
        CHECK(a.canonical_input_sequence.front() == Input::Hold);

        // Replaying from the held piece's spawn must reproduce the placement.
        const ExecutionResult r =
            execute_inputs(b, spawn_piece(a.final_piece, cfg), a.canonical_input_sequence, cfg);
        CHECK(r.ok);
        CHECK_EQ(r.piece.x, a.final_x);
        CHECK_EQ(r.piece.y, a.final_y);
        CHECK(r.piece.rot == a.final_rotation);
        CHECK_EQ(r.cost, a.base_duration);

        // And it must cost strictly more than the same placement without hold.
        CHECK(a.base_duration > 0);
    }
    CHECK(hold_actions > 0);
}

TEST(hold_costs_more_than_the_same_placement_without_it) {
    const RulesetConfig cfg = league();
    MoveGenerator gen;
    Board b(10, 40);
    // Same piece reachable with and without hold: compare like for like.
    const auto plain = gen.generate_for_piece(b, Piece::I, cfg, false);
    const auto held = gen.generate_for_piece(b, Piece::I, cfg, true);
    CHECK_EQ(plain.size(), held.size());
    for (size_t i = 0; i < plain.size() && i < held.size(); ++i) {
        CHECK_MSG(held[i].base_duration > plain[i].base_duration,
                  "the hold input must be charged");
    }
}

TEST(action_cost_is_the_true_shortest_path) {
    // The generator settles nodes in cost order (Dial's bucket queue), so the
    // price it reports must be the genuine optimum. Verify against an
    // independent brute-force search over short input sequences.
    const RulesetConfig cfg = league();
    MoveGenerator gen;
    Rng rng(2468);

    for (int trial = 0; trial < 6; ++trial) {
        Board b(10, 40);
        for (int y = 0; y < 5; ++y)
            for (int x = 0; x < 10; ++x)
                if (rng.chance(1, 4)) b.fill_cell(x, y, false);

        for (int pi = 0; pi < PIECE_COUNT; ++pi) {
            const Piece p = static_cast<Piece>(pi);
            const auto acts = gen.generate_for_piece(b, p, cfg, false);

            // Brute force: enumerate every input sequence up to length 4 and
            // record the cheapest cost that reaches each final placement.
            const std::vector<Input> alphabet = {Input::Left,  Input::Right, Input::DasLeft,
                                                 Input::DasRight, Input::Cw, Input::Ccw,
                                                 Input::Flip, Input::SoftDrop};
            std::map<std::tuple<int, int, int>, Tick> best;
            std::vector<std::vector<Input>> frontier{{}};
            for (int depth = 0; depth <= 3; ++depth) {
                std::vector<std::vector<Input>> next;
                for (const auto& prefix : frontier) {
                    std::vector<Input> seq = prefix;
                    seq.push_back(Input::HardDrop);
                    const ExecutionResult r =
                        execute_inputs(b, spawn_piece(p, cfg), seq, cfg);
                    if (r.ok && grounded(b, r.piece)) {
                        const auto key = std::make_tuple(r.piece.x, r.piece.y,
                                                         static_cast<int>(r.piece.rot));
                        auto it = best.find(key);
                        if (it == best.end() || r.cost < it->second) best[key] = r.cost;
                    }
                    if (depth == 3) continue;
                    for (Input in : alphabet) {
                        std::vector<Input> ext = prefix;
                        ext.push_back(in);
                        next.push_back(std::move(ext));
                    }
                }
                frontier.swap(next);
            }

            // Anything brute force can reach, the generator must price at most
            // as cheaply (it may know a cheaper longer route, never a dearer one).
            for (const auto& [key, brute_cost] : best) {
                const auto [bx, by, brot] = key;
                for (const auto& a : acts) {
                    if (a.final_x == bx && a.final_y == by &&
                        static_cast<int>(a.final_rotation) == brot) {
                        const Tick generator_cost =
                            a.base_duration -
                            (a.cleared_lines > 0 ? cfg.clear_rules.line_clear_delay : 0);
                        CHECK_MSG(generator_cost <= brute_cost,
                                  "generator cost " + std::to_string(generator_cost) +
                                      " exceeds a known route at " + std::to_string(brute_cost));
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Gravity (spec 6 movement.gravity)
// ---------------------------------------------------------------------------

TEST(gravity_model_is_integer_exact) {
    RulesetConfig cfg = league();
    cfg.movement.gravity_num = 1;
    cfg.movement.gravity_den = 60;
    const HandlingModel h = HandlingModel::from(cfg);
    CHECK_EQ(h.gravity_fall(0), 0);
    CHECK_EQ(h.gravity_fall(59), 0);
    CHECK_EQ(h.gravity_fall(60), 1);
    CHECK_EQ(h.gravity_fall(119), 1);
    CHECK_EQ(h.gravity_fall(120), 2);
    CHECK_EQ(h.ticks_per_cell(), 60);
    CHECK(!h.is_high_gravity());

    cfg.movement.gravity_num = 20;
    cfg.movement.gravity_den = 1;
    const HandlingModel g20 = HandlingModel::from(cfg);
    CHECK_EQ(g20.gravity_fall(1), 20);
    CHECK_EQ(g20.ticks_per_cell(), 0);
    CHECK(g20.is_high_gravity());

    // Zero gravity never moves the piece.
    cfg.movement.gravity_num = 0;
    const HandlingModel none = HandlingModel::from(cfg);
    CHECK_EQ(none.gravity_fall(10000), 0);
}

TEST(normal_gravity_does_not_restrict_placements) {
    // The default 1/60 G must cost nothing: a placement takes a handful of
    // ticks, far less than the 60 needed to fall one cell.
    Board b(10, 40);
    MoveGenerator with_check;
    MoveGenerator without{MoveGenerator::Options{true, true, true, 20000, false, 8}};
    const RulesetConfig cfg = league();
    for (int pi = 0; pi < PIECE_COUNT; ++pi) {
        const Piece p = static_cast<Piece>(pi);
        CHECK_EQ(with_check.generate_for_piece(b, p, cfg, false).size(),
                 without.generate_for_piece(b, p, cfg, false).size());
    }
}

TEST(high_gravity_removes_unreachable_placements) {
    // At 20G a piece hits the floor the instant it spawns, so most of the
    // manoeuvres available at normal speed simply cannot be performed. A
    // generator that ignored gravity would happily emit them.
    RulesetConfig fast = league();
    fast.movement.gravity_num = 20;
    fast.movement.gravity_den = 1;

    Board b(10, 40);
    MoveGenerator gen;
    size_t restricted = 0, unrestricted = 0;
    MoveGenerator off{MoveGenerator::Options{true, true, true, 20000, false, 8}};
    for (int pi = 0; pi < PIECE_COUNT; ++pi) {
        const Piece p = static_cast<Piece>(pi);
        restricted += gen.generate_for_piece(b, p, fast, false).size();
        unrestricted += off.generate_for_piece(b, p, fast, false).size();
    }
    CHECK_MSG(restricted < unrestricted,
              "20G must remove placements: " + std::to_string(restricted) + " vs " +
                  std::to_string(unrestricted));
    CHECK_MSG(restricted > 0, "20G must still leave legal placements");
}

TEST(gravity_restriction_is_monotonic) {
    // Faster gravity can only ever remove options, never add them.
    Board b(10, 40);
    MoveGenerator gen;
    const std::vector<std::pair<int, int>> speeds = {{1, 60}, {1, 8}, {1, 2}, {1, 1}, {5, 1},
                                                     {20, 1}};
    size_t previous = SIZE_MAX;
    for (const auto& [num, den] : speeds) {
        RulesetConfig cfg = league();
        cfg.movement.gravity_num = num;
        cfg.movement.gravity_den = den;
        size_t total = 0;
        for (int pi = 0; pi < PIECE_COUNT; ++pi)
            total += gen.generate_for_piece(b, static_cast<Piece>(pi), cfg, false).size();
        CHECK_MSG(total <= previous,
                  "gravity " + std::to_string(num) + "/" + std::to_string(den) +
                      " produced MORE placements than a slower one");
        previous = total;
    }
}

TEST(placements_remain_legal_under_high_gravity) {
    // Whatever survives the gravity filter must still be a legal, grounded,
    // reproducible placement.
    RulesetConfig cfg = league();
    cfg.movement.gravity_num = 20;
    cfg.movement.gravity_den = 1;

    MoveGenerator gen;
    Rng rng(4242);
    for (int trial = 0; trial < 15; ++trial) {
        Board b(10, 40);
        for (int y = 0; y < 6; ++y)
            for (int x = 0; x < 10; ++x)
                if (rng.chance(1, 3)) b.fill_cell(x, y, false);
        for (int pi = 0; pi < PIECE_COUNT; ++pi) {
            const Piece p = static_cast<Piece>(pi);
            for (const auto& a : gen.generate_for_piece(b, p, cfg, false)) {
                const ActivePiece piece = a.piece_state();
                CHECK(!collides(b, piece));
                CHECK(grounded(b, piece));
                const ExecutionResult r =
                    execute_inputs(b, spawn_piece(p, cfg), a.canonical_input_sequence, cfg);
                CHECK(r.ok);
                CHECK_EQ(r.piece.x, a.final_x);
                CHECK_EQ(r.piece.y, a.final_y);
            }
        }
    }
}

TEST(gravity_is_part_of_the_ruleset_hash) {
    // Two engines disagreeing about gravity must not silently share replays.
    RulesetConfig a = league();
    RulesetConfig b = league();
    b.movement.gravity_num = 20;
    b.movement.gravity_den = 1;
    CHECK(a.hash() != b.hash());
}

TEST(gravity_filtering_is_deterministic) {
    RulesetConfig cfg = league();
    cfg.movement.gravity_num = 5;
    cfg.movement.gravity_den = 1;
    Board b(10, 40);
    MoveGenerator gen;
    const auto x = gen.generate_for_piece(b, Piece::T, cfg, false);
    const auto y = gen.generate_for_piece(b, Piece::T, cfg, false);
    CHECK_EQ(x.size(), y.size());
    for (size_t i = 0; i < x.size() && i < y.size(); ++i) {
        CHECK_EQ(x[i].final_x, y[i].final_x);
        CHECK_EQ(x[i].final_y, y[i].final_y);
        CHECK_EQ(x[i].base_duration, y[i].base_duration);
    }
}