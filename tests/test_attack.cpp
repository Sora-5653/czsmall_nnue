// SPDX-License-Identifier: MIT
// Attack table, combo multiplier, B2B and Surge (spec 6 / 18.1).
#include "test_util.hpp"
#include "tetra/attack.hpp"
#include "tetra/ruleset.hpp"

using namespace tetra;

namespace {

RulesetConfig league() { return RulesetConfig::tetra_league(); }

ClearDescriptor clear_of(int lines, SpinType spin = SpinType::None, Piece p = Piece::I) {
    ClearDescriptor c;
    c.lines = lines;
    c.spin = spin;
    c.piece = p;
    return c;
}

// Send one clear at a given combo / b2b state and return the lines sent.
int send(const ClearDescriptor& c, int combo, int b2b, const RulesetConfig& cfg) {
    AttackState st;
    st.combo = combo - 1;  // compute_attack increments before use
    st.b2b_streak = b2b;
    return compute_attack(c, st, cfg, nullptr).lines;
}

}  // namespace

TEST(base_attack_values_match_tetrio) {
    const RulesetConfig cfg = league();
    // No combo, no B2B: the plain weights.
    CHECK_EQ(send(clear_of(1), 0, 0, cfg), 0);  // single
    CHECK_EQ(send(clear_of(2), 0, 0, cfg), 1);  // double
    CHECK_EQ(send(clear_of(3), 0, 0, cfg), 2);  // triple
    CHECK_EQ(send(clear_of(4), 0, 0, cfg), 4);  // quad
}

TEST(t_spin_attack_values_match_tetrio) {
    const RulesetConfig cfg = league();
    CHECK_EQ(send(clear_of(1, SpinType::Full, Piece::T), 0, 0, cfg), 2);  // TSS
    CHECK_EQ(send(clear_of(2, SpinType::Full, Piece::T), 0, 0, cfg), 4);  // TSD
    CHECK_EQ(send(clear_of(3, SpinType::Full, Piece::T), 0, 0, cfg), 6);  // TST
    CHECK_EQ(send(clear_of(1, SpinType::Mini, Piece::T), 0, 0, cfg), 0);  // TSMS
    CHECK_EQ(send(clear_of(2, SpinType::Mini, Piece::T), 0, 0, cfg), 1);  // TSMD
}

TEST(b2b_charging_adds_one_line) {
    const RulesetConfig cfg = league();
    // A quad with an existing B2B streak sends 4 + 1.
    CHECK_EQ(send(clear_of(4), 0, 1, cfg), 5);
    CHECK_EQ(send(clear_of(4), 0, 3, cfg), 5);  // Charging is flat, not scaled
    // A TSD with B2B: 4 + 1.
    CHECK_EQ(send(clear_of(2, SpinType::Full, Piece::T), 0, 1, cfg), 5);
}

TEST(first_difficult_clear_gets_no_b2b_bonus) {
    const RulesetConfig cfg = league();
    AttackState st;
    // Fresh state: streak 0 -> no bonus on this attack, but the streak starts.
    const AttackResult r1 = compute_attack(clear_of(4), st, cfg, nullptr);
    CHECK_EQ(r1.lines, 4);
    CHECK_EQ(r1.b2b_bonus, 0);
    CHECK_EQ(st.b2b_streak, 1);
    // Second quad in a row: now the bonus applies. Combo is also running, so
    // the multiplier is 1.25 -> floor((4+1) * 1.25) = 6.
    const AttackResult r2 = compute_attack(clear_of(4), st, cfg, nullptr);
    CHECK_EQ(r2.b2b_bonus, 1);
    CHECK_EQ(r2.lines, 6);
    CHECK_EQ(st.b2b_streak, 2);
}

TEST(combo_multiplier_matches_the_published_table) {
    const RulesetConfig cfg = league();
    // Quad at increasing combo, no B2B: floor(4 * (1 + 0.25c)).
    CHECK_EQ(send(clear_of(4), 0, 0, cfg), 4);   // 4 * 1.00
    CHECK_EQ(send(clear_of(4), 1, 0, cfg), 5);   // 4 * 1.25
    CHECK_EQ(send(clear_of(4), 2, 0, cfg), 6);   // 4 * 1.50
    CHECK_EQ(send(clear_of(4), 3, 0, cfg), 7);   // 4 * 1.75
    CHECK_EQ(send(clear_of(4), 4, 0, cfg), 8);   // 4 * 2.00
    CHECK_EQ(send(clear_of(4), 8, 0, cfg), 12);  // 4 * 3.00

    // Doubles: floor(1 * (1 + 0.25c)).
    CHECK_EQ(send(clear_of(2), 0, 0, cfg), 1);
    CHECK_EQ(send(clear_of(2), 4, 0, cfg), 2);   // 1 * 2.00
    CHECK_EQ(send(clear_of(2), 8, 0, cfg), 3);   // 1 * 3.00

    // A community-cited example: combo 6 T-spin triple = 6 * 2.50 = 15.
    CHECK_EQ(send(clear_of(3, SpinType::Full, Piece::T), 6, 0, cfg), 15);
}

TEST(zero_base_clears_use_the_logarithmic_combo_curve) {
    const RulesetConfig cfg = league();
    // osk's singles formula: floor(ln(1 + 1.25 * combo)).
    CHECK_EQ(send(clear_of(1), 0, 0, cfg), 0);  // ln(1) = 0
    CHECK_EQ(send(clear_of(1), 1, 0, cfg), 0);  // ln(2.25) = 0.81
    CHECK_EQ(send(clear_of(1), 2, 0, cfg), 1);  // ln(3.5)  = 1.25
    CHECK_EQ(send(clear_of(1), 4, 0, cfg), 1);  // ln(6)    = 1.79
    CHECK_EQ(send(clear_of(1), 5, 0, cfg), 1);  // ln(7.25) = 1.98
    CHECK_EQ(send(clear_of(1), 6, 0, cfg), 2);  // ln(8.5)  = 2.14
}

TEST(a_single_with_b2b_leaves_the_log_curve) {
    const RulesetConfig cfg = league();
    // A T-spin single has a non-zero base, so it uses the normal multiplier
    // even at combo 0.
    CHECK_EQ(send(clear_of(1, SpinType::Full, Piece::T), 0, 0, cfg), 2);
    CHECK_EQ(send(clear_of(1, SpinType::Full, Piece::T), 2, 0, cfg), 3);  // 2 * 1.5
}

TEST(combo_breaks_when_no_lines_are_cleared) {
    const RulesetConfig cfg = league();
    AttackState st;
    compute_attack(clear_of(2), st, cfg, nullptr);
    CHECK_EQ(st.combo, 0);
    compute_attack(clear_of(2), st, cfg, nullptr);
    CHECK_EQ(st.combo, 1);
    // A placement with no clear resets the combo but keeps B2B.
    const int b2b_before = st.b2b_streak;
    compute_attack(clear_of(0), st, cfg, nullptr);
    CHECK_EQ(st.combo, -1);
    CHECK_EQ(st.b2b_streak, b2b_before);
}

TEST(non_difficult_clear_breaks_b2b) {
    const RulesetConfig cfg = league();
    AttackState st;
    compute_attack(clear_of(4), st, cfg, nullptr);
    compute_attack(clear_of(4), st, cfg, nullptr);
    CHECK_EQ(st.b2b_streak, 2);
    const AttackResult r = compute_attack(clear_of(2), st, cfg, nullptr);
    CHECK(r.b2b_broken);
    CHECK_EQ(st.b2b_streak, 0);
}

TEST(surge_charges_from_streak_four_and_releases_on_break) {
    const RulesetConfig cfg = league();
    AttackState st;
    // Four difficult clears in a row: Surge starts charging at streak 4.
    for (int i = 0; i < 4; ++i) compute_attack(clear_of(4), st, cfg, nullptr);
    CHECK_EQ(st.b2b_streak, 4);
    CHECK_EQ(st.surge, cfg.attack.surge_base);  // 4 in non-QUICK-PLAY modes

    // Each further difficult clear adds one line.
    compute_attack(clear_of(4), st, cfg, nullptr);
    CHECK_EQ(st.b2b_streak, 5);
    CHECK_EQ(st.surge, cfg.attack.surge_base + 1);

    // Breaking the streak releases the whole charge on top of the attack.
    const int charged = st.surge;
    const AttackResult r = compute_attack(clear_of(2), st, cfg, nullptr);
    CHECK_EQ(r.surge_released, charged);
    CHECK(r.lines >= charged);
    CHECK_EQ(st.surge, 0);
    CHECK_EQ(st.b2b_streak, 0);
}

TEST(surge_does_not_charge_below_the_threshold) {
    const RulesetConfig cfg = league();
    AttackState st;
    for (int i = 0; i < 3; ++i) compute_attack(clear_of(4), st, cfg, nullptr);
    CHECK_EQ(st.b2b_streak, 3);
    CHECK_EQ(st.surge, 0);
    const AttackResult r = compute_attack(clear_of(2), st, cfg, nullptr);
    CHECK_EQ(r.surge_released, 0);
}

TEST(quick_play_surge_starts_at_one) {
    const RulesetConfig cfg = RulesetConfig::quick_play();
    AttackState st;
    for (int i = 0; i < 4; ++i) compute_attack(clear_of(4), st, cfg, nullptr);
    CHECK_EQ(st.surge, 1);
}

TEST(surge_split_carries_the_remainder_first) {
    // "splits into three segments, with the first and sometimes second segment
    // carrying the remainder".
    CHECK(split_surge(9, 3) == std::vector<int>({3, 3, 3}));
    CHECK(split_surge(10, 3) == std::vector<int>({4, 3, 3}));
    CHECK(split_surge(11, 3) == std::vector<int>({4, 4, 3}));
    CHECK(split_surge(4, 3) == std::vector<int>({2, 1, 1}));
    CHECK(split_surge(0, 3).empty());
    // Total is always preserved.
    for (int n = 1; n < 40; ++n) {
        const auto parts = split_surge(n, 3);
        int sum = 0;
        for (int v : parts) sum += v;
        CHECK_EQ(sum, n);
    }
}

TEST(garbage_clear_bonus_applies_to_quads_and_spins_only) {
    const RulesetConfig cfg = league();
    ClearDescriptor quad = clear_of(4);
    quad.cleared_garbage = true;
    CHECK_EQ(send(quad, 0, 0, cfg), 5);  // 4 + 1 garbage bonus

    ClearDescriptor tsd = clear_of(2, SpinType::Full, Piece::T);
    tsd.cleared_garbage = true;
    CHECK_EQ(send(tsd, 0, 0, cfg), 5);  // 4 + 1

    // A plain double that clears garbage gets no bonus.
    ClearDescriptor dbl = clear_of(2);
    dbl.cleared_garbage = true;
    CHECK_EQ(send(dbl, 0, 0, cfg), 1);
}

TEST(garbage_bonus_is_outside_the_combo_multiplier) {
    const RulesetConfig cfg = league();
    ClearDescriptor quad = clear_of(4);
    quad.cleared_garbage = true;
    // combo 4 -> floor(4 * 2.0) = 8, then +1 flat = 9 (not floor(5*2) = 10).
    CHECK_EQ(send(quad, 4, 0, cfg), 9);
}

TEST(all_clear_adds_a_flat_bonus) {
    const RulesetConfig cfg = league();
    ClearDescriptor ac = clear_of(4);
    ac.all_clear = true;
    CHECK_EQ(send(ac, 0, 0, cfg), 4 + cfg.attack.all_clear);
}

TEST(all_clear_can_be_disabled) {
    RulesetConfig cfg = league();
    cfg.clear_rules.all_clear_enabled = false;
    ClearDescriptor ac = clear_of(4);
    ac.all_clear = true;
    CHECK_EQ(send(ac, 0, 0, cfg), 4);
}

TEST(difficult_clear_classification) {
    CHECK(is_difficult_clear(clear_of(4)));
    CHECK(is_difficult_clear(clear_of(1, SpinType::Full, Piece::T)));
    CHECK(is_difficult_clear(clear_of(1, SpinType::Mini, Piece::S)));
    CHECK(!is_difficult_clear(clear_of(1)));
    CHECK(!is_difficult_clear(clear_of(2)));
    CHECK(!is_difficult_clear(clear_of(3)));
    CHECK(!is_difficult_clear(clear_of(0)));
}

TEST(b2b_chaining_uses_the_step_table) {
    RulesetConfig cfg = league();
    cfg.attack.b2b_mode = B2BMode::Chaining;
    // osk's level boundaries: 1-2 -> +1, 3-7 -> +2, 8-23 -> +3, 24-66 -> +4.
    CHECK_EQ(b2b_bonus_for(0, cfg.attack), 0);
    CHECK_EQ(b2b_bonus_for(1, cfg.attack), 1);
    CHECK_EQ(b2b_bonus_for(2, cfg.attack), 1);
    CHECK_EQ(b2b_bonus_for(3, cfg.attack), 2);
    CHECK_EQ(b2b_bonus_for(7, cfg.attack), 2);
    CHECK_EQ(b2b_bonus_for(8, cfg.attack), 3);
    CHECK_EQ(b2b_bonus_for(23, cfg.attack), 3);
    CHECK_EQ(b2b_bonus_for(24, cfg.attack), 4);
    CHECK_EQ(b2b_bonus_for(66, cfg.attack), 4);
    CHECK_EQ(b2b_bonus_for(67, cfg.attack), 5);
    CHECK_EQ(b2b_bonus_for(1370, cfg.attack), 8);
}

TEST(rounding_down_is_deterministic) {
    CHECK_EQ(round_attack(1.25, RoundingMode::Down, nullptr), 1);
    CHECK_EQ(round_attack(1.99, RoundingMode::Down, nullptr), 1);
    CHECK_EQ(round_attack(2.0, RoundingMode::Down, nullptr), 2);
    CHECK_EQ(round_attack(0.4, RoundingMode::Down, nullptr), 0);
}

TEST(rng_rounding_matches_the_fractional_part_on_average) {
    // QUICK PLAY: a 1.25 attack should round up ~25% of the time.
    Rng rng(4242);
    int ups = 0;
    const int N = 20000;
    for (int i = 0; i < N; ++i)
        if (round_attack(1.25, RoundingMode::Rng, &rng) == 2) ++ups;
    const double rate = static_cast<double>(ups) / N;
    CHECK_MSG(rate > 0.23 && rate < 0.27,
              "expected ~25% round-up rate, got " + std::to_string(rate));
    // The value is never below the floor or above the ceiling.
    for (int i = 0; i < 1000; ++i) {
        const int v = round_attack(1.25, RoundingMode::Rng, &rng);
        CHECK(v == 1 || v == 2);
    }
}

TEST(attack_is_never_negative) {
    const RulesetConfig cfg = league();
    for (int lines = 0; lines <= 4; ++lines)
        for (int combo = 0; combo < 20; ++combo)
            for (int b2b = 0; b2b < 10; ++b2b)
                CHECK(send(clear_of(lines), combo, b2b, cfg) >= 0);
}

TEST(ruleset_hash_is_stable_and_sensitive) {
    const RulesetConfig a = league();
    const RulesetConfig b = league();
    CHECK_EQ(a.hash(), b.hash());
    CHECK(a.hash_hex().size() == 16);

    RulesetConfig c = league();
    c.attack.quad = 5;
    CHECK(c.hash() != a.hash());

    RulesetConfig d = league();
    d.garbage.travel_time += 1;
    CHECK(d.hash() != a.hash());

    // Different presets must hash differently.
    CHECK(RulesetConfig::quick_play().hash() != a.hash());
    CHECK(RulesetConfig::guideline().hash() != a.hash());
}
