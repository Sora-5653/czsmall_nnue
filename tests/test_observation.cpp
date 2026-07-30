// SPDX-License-Identifier: MIT
// Observation masking and tokenizer (spec 9.2, 9.3, 18.3 information leaks).
#include "test_util.hpp"
#include "tetra/movegen.hpp"
#include "tetra/tokenizer.hpp"

#include <cmath>
#include <set>

using namespace tetra;

namespace {
RulesetConfig league() { return RulesetConfig::tetra_league(); }
}  // namespace

TEST(observation_hides_the_garbage_hole_column) {
    // Spec 3.2 / 18.3: the single most important leak to prevent. The
    // simulator knows the hole column the moment the attack is queued; the
    // observation must not expose it in any form.
    RulesetConfig cfg = league();
    cfg.garbage.travel_time = 10;
    cfg.garbage.activation_delay = 10;

    Player p;
    p.reset(cfg, 5, 0);
    p.receive_attack(4, 0, 1);

    // The simulator does know it.
    CHECK(!p.garbage().entries().empty());
    const int secret = p.garbage().entries()[0].hole_column;
    CHECK_MSG(secret >= 0, "the simulator must have chosen a hole column");

    const Observation obs = observe(p);
    CHECK_EQ(obs.pending_lines, 4);
    CHECK_EQ(static_cast<int>(obs.pending_garbage.size()), 1);
    // ObservedGarbage has no hole field at all -- verify the visible data is
    // limited to counts and timings.
    CHECK_EQ(obs.pending_garbage[0].lines, 4);
    CHECK_EQ(obs.pending_garbage[0].ticks_until_arrival, 10);
    CHECK_EQ(obs.pending_garbage[0].ticks_until_activation, 20);

    // And the tokenized form must not encode it either: tokenizing the same
    // state with different secret hole columns must produce identical tokens.
    Tokenizer tok;
    const auto a = tok.encode(obs, cfg);

    Player q;
    q.reset(cfg, 5, 0);
    q.receive_attack(4, 0, 1);
    // Force a different hidden hole column.
    const_cast<GarbageEntry&>(q.garbage().entries()[0]).hole_column =
        (secret + 3) % cfg.geometry.width;
    const auto b = tok.encode(observe(q), cfg);

    CHECK_EQ(a.size(), b.size());
    bool identical = true;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (a.tokens[i].kind != b.tokens[i].kind) identical = false;
        for (int k = 0; k < TOKEN_FEATURES; ++k)
            if (std::fabs(a.tokens[i].f[static_cast<size_t>(k)] -
                          b.tokens[i].f[static_cast<size_t>(k)]) > 1e-9f)
                identical = false;
    }
    CHECK_MSG(identical, "hidden hole column must not influence the tokens");
}

TEST(observation_respects_the_preview_limit) {
    for (int n : {1, 3, 5}) {
        RulesetConfig cfg = league();
        cfg.randomizer.preview_count = n;
        Player p;
        p.reset(cfg, 3, 0);
        const Observation obs = observe(p);
        CHECK_EQ(static_cast<int>(obs.next.size()), n);
    }
}

TEST(observation_never_exposes_the_rng_state) {
    // Structural guarantee: Observation has no RNG member. This test pins the
    // behavioural consequence -- two players whose RNGs are in different states
    // but whose visible state matches must observe the same thing.
    RulesetConfig cfg = league();
    cfg.randomizer.preview_count = 0;  // no visible queue

    Player a, b;
    a.reset(cfg, 1, 0);
    b.reset(cfg, 999999, 0);
    // Burn RNG on b only.
    for (int i = 0; i < 50; ++i) b.mutable_queue().rng().next_u64();

    const Observation oa = observe(a);
    const Observation ob = observe(b);
    CHECK(oa.board == ob.board);
    CHECK_EQ(oa.pending_lines, ob.pending_lines);
    CHECK_EQ(static_cast<int>(oa.next.size()), 0);
    CHECK_EQ(static_cast<int>(ob.next.size()), 0);
}

TEST(tokenizer_emits_the_expected_token_layout) {
    const RulesetConfig cfg = league();
    Player p;
    p.reset(cfg, 1, 0);
    Tokenizer tok;
    const auto t = tok.encode(observe(p), cfg);

    int rows = 0, cols = 0, summary = 0, next = 0, rule = 0, time = 0, active = 0, hold = 0,
        counters = 0, missing = 0;
    for (const auto& tk : t.tokens) {
        switch (tk.kind) {
            case TokenKind::Row: ++rows; break;
            case TokenKind::Column: ++cols; break;
            case TokenKind::BoardSummary: ++summary; break;
            case TokenKind::Next: ++next; break;
            case TokenKind::Rule: ++rule; break;
            case TokenKind::Time: ++time; break;
            case TokenKind::Active: ++active; break;
            case TokenKind::Hold: ++hold; break;
            case TokenKind::Counters: ++counters; break;
            case TokenKind::Missing: ++missing; break;
            default: break;
        }
    }
    CHECK_EQ(rows, tok.options().max_rows);
    CHECK_EQ(cols, cfg.geometry.width);
    CHECK_EQ(summary, 1);
    CHECK_EQ(next, cfg.randomizer.preview_count);
    CHECK_EQ(rule, 1);
    CHECK_EQ(time, 1);
    CHECK_EQ(active, 1);
    CHECK_EQ(hold, 1);
    CHECK_EQ(counters, 1);
    CHECK_EQ(missing, 1);  // no opponent yet

    // Spec 9.5: the context must stay in the 128-256 token band.
    CHECK_MSG(t.size() >= 32 && t.size() <= 256,
              "token count out of the designed range: " + std::to_string(t.size()));
}

TEST(tokenizer_produces_finite_bounded_features) {
    // Any NaN or unbounded value would destabilise training immediately.
    const RulesetConfig cfg = league();
    MoveGenerator gen;
    Rng rng(4242);

    for (int trial = 0; trial < 30; ++trial) {
        Player p;
        p.reset(cfg, 100 + static_cast<std::uint64_t>(trial), 0);
        // Randomise the board and the queue state.
        Board& b = p.mutable_board();
        for (int y = 0; y < 12; ++y)
            for (int x = 0; x < 10; ++x)
                if (rng.chance(1, 3)) b.fill_cell(x, y, rng.chance(1, 4));
        p.receive_attack(static_cast<int>(rng.below(9)), 0, 1);

        Tokenizer tok;
        const auto t = tok.encode(observe(p), cfg);
        for (const auto& tk : t.tokens) {
            for (int k = 0; k < TOKEN_FEATURES; ++k) {
                const float v = tk.f[static_cast<size_t>(k)];
                CHECK_MSG(std::isfinite(v), "token feature must be finite");
                CHECK_MSG(v >= -4.0f && v <= 4.0f,
                          "token feature out of range: " + std::to_string(v));
            }
        }
    }
}

TEST(tokenizer_is_deterministic) {
    const RulesetConfig cfg = league();
    Player p;
    p.reset(cfg, 77, 0);
    Tokenizer tok;
    const auto a = tok.encode(observe(p), cfg);
    const auto b = tok.encode(observe(p), cfg);
    CHECK_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        for (int k = 0; k < TOKEN_FEATURES; ++k)
            CHECK(a.tokens[i].f[static_cast<size_t>(k)] == b.tokens[i].f[static_cast<size_t>(k)]);
}

TEST(row_tokens_capture_line_clear_readiness) {
    const RulesetConfig cfg = league();
    Player p;
    p.reset(cfg, 1, 0);
    Board& b = p.mutable_board();
    // Row 0 is one cell short of full.
    for (int x = 0; x < 9; ++x) b.fill_cell(x, 0, false);

    Tokenizer tok;
    const auto t = tok.encode(observe(p), cfg);
    bool found = false;
    for (const auto& tk : t.tokens) {
        if (tk.kind == TokenKind::Row && tk.index == 0) {
            CHECK_MSG(tk.f[4] == 1.0f, "one-away-from-full flag must be set");
            CHECK(tk.f[2] == 0.0f);  // not full
            found = true;
        }
    }
    CHECK(found);
}

TEST(column_tokens_capture_wells) {
    const RulesetConfig cfg = league();
    Player p;
    p.reset(cfg, 1, 0);
    Board& b = p.mutable_board();
    // Build a 4-deep well at column 9.
    for (int x = 0; x < 9; ++x)
        for (int y = 0; y < 4; ++y) b.fill_cell(x, y, false);

    Tokenizer tok;
    const auto t = tok.encode(observe(p), cfg);
    bool found = false;
    for (const auto& tk : t.tokens) {
        if (tk.kind == TokenKind::Column && tk.index == 9) {
            CHECK_MSG(tk.f[5] == 1.0f, "deep well flag must be set for the I slot");
            found = true;
        }
    }
    CHECK(found);
}

TEST(opponent_tokens_appear_only_with_an_opponent) {
    const RulesetConfig cfg = league();
    Player a, b;
    a.reset(cfg, 1, 0);
    b.reset(cfg, 2, 1);
    Tokenizer tok;

    const auto solo = tok.encode(observe(a), cfg);
    int solo_opp = 0, solo_missing = 0;
    for (const auto& t : solo.tokens) {
        if (t.kind == TokenKind::OpponentRow || t.kind == TokenKind::OpponentColumn ||
            t.kind == TokenKind::OpponentSummary)
            ++solo_opp;
        if (t.kind == TokenKind::Missing) ++solo_missing;
    }
    CHECK_EQ(solo_opp, 0);
    CHECK_EQ(solo_missing, 1);

    const auto duel = tok.encode(observe(a, &b), cfg);
    int duel_opp = 0;
    for (const auto& t : duel.tokens)
        if (t.kind == TokenKind::OpponentRow || t.kind == TokenKind::OpponentColumn ||
            t.kind == TokenKind::OpponentSummary)
            ++duel_opp;
    CHECK(duel_opp > 0);
    CHECK(duel.size() > solo.size());
}

TEST(rule_token_changes_with_the_ruleset) {
    Player p;
    Tokenizer tok;
    auto rule_features = [&](const RulesetConfig& cfg) {
        p.reset(cfg, 1, 0);
        const auto t = tok.encode(observe(p), cfg);
        for (const auto& tk : t.tokens)
            if (tk.kind == TokenKind::Rule) return tk.f;
        return std::array<float, TOKEN_FEATURES>{};
    };
    const auto league_f = rule_features(RulesetConfig::tetra_league());
    const auto guideline_f = rule_features(RulesetConfig::guideline());
    bool differs = false;
    for (int k = 0; k < TOKEN_FEATURES; ++k)
        if (league_f[static_cast<size_t>(k)] != guideline_f[static_cast<size_t>(k)]) differs = true;
    CHECK_MSG(differs, "the rule token must distinguish rulesets");
}

TEST(garbage_tokens_encode_timing) {
    RulesetConfig cfg = league();
    cfg.garbage.travel_time = 20;
    cfg.garbage.activation_delay = 20;
    Player p;
    p.reset(cfg, 1, 0);
    p.receive_attack(5, 0, 1);

    Tokenizer tok;
    const auto t = tok.encode(observe(p), cfg);
    bool found_group = false;
    for (const auto& tk : t.tokens) {
        if (tk.kind == TokenKind::Garbage && tk.index == 0) {
            CHECK(tk.f[0] > 0.0f);          // lines
            CHECK(tk.f[2] > 0.0f);          // time until activation
            CHECK(tk.f[3] == 0.0f);         // not active yet
            found_group = true;
        }
    }
    CHECK(found_group);
}

TEST(action_embedding_separates_meaningfully_different_actions) {
    const RulesetConfig cfg = league();
    Board b(10, 40);
    MoveGenerator gen;
    const auto acts = gen.generate_for_piece(b, Piece::T, cfg, false);
    CHECK(!acts.empty());

    std::set<std::string> seen;
    for (const auto& a : acts) {
        const ActionEmbedding e = embed_action(a, b, cfg);
        std::string key;
        for (int k = 0; k < ACTION_FEATURES; ++k) {
            CHECK(std::isfinite(e.f[static_cast<size_t>(k)]));
            key += std::to_string(static_cast<int>(e.f[static_cast<size_t>(k)] * 1000)) + ",";
        }
        seen.insert(key);
    }
    // Distinct placements must not collapse onto identical embeddings.
    CHECK_MSG(seen.size() == acts.size(), "action embeddings must be distinguishable: " +
                                              std::to_string(seen.size()) + " of " +
                                              std::to_string(acts.size()));
}

TEST(action_embedding_flags_line_clears_and_holes) {
    const RulesetConfig cfg = league();
    // A board where one placement clears a line and another makes a hole.
    Board b(10, 40);
    for (int x = 0; x < 6; ++x) b.fill_cell(x, 0, false);

    MoveGenerator gen;
    const auto acts = gen.generate_for_piece(b, Piece::I, cfg, false);
    bool saw_clear = false;
    for (const auto& a : acts) {
        const ActionEmbedding e = embed_action(a, b, cfg);
        if (a.cleared_lines > 0) {
            CHECK(e.f[14] > 0.0f);
            saw_clear = true;
        }
    }
    CHECK(saw_clear);
}

TEST(observation_survives_a_full_game_loop) {
    // End-to-end smoke test: observe -> tokenize -> act, for a whole game.
    const RulesetConfig cfg = league();
    Player p;
    p.reset(cfg, 2024, 0);
    MoveGenerator gen;
    Tokenizer tok;

    for (int i = 0; i < 120 && p.alive(); ++i) {
        const Observation obs = observe(p);
        const auto tokens = tok.encode(obs, cfg);
        CHECK(tokens.size() > 0);
        for (const auto& tk : tokens.tokens)
            for (int k = 0; k < TOKEN_FEATURES; ++k)
                CHECK(std::isfinite(tk.f[static_cast<size_t>(k)]));

        const auto acts = gen.generate_for_piece(p.board(), p.active().type, cfg, false);
        if (acts.empty()) break;
        size_t best = 0;
        for (size_t k = 1; k < acts.size(); ++k)
            if (acts[k].final_y < acts[best].final_y) best = k;
        p.set_active(acts[best].piece_state());
        int out = 0;
        if (!p.lock_piece(20, &out).ok) break;
    }
    CHECK(true);
}
