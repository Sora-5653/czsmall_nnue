// SPDX-License-Identifier: MIT
#include "tetra/ruleset.hpp"

#include <cstdio>

namespace tetra {
namespace {

// FNV-1a 64. Stable across runs and platforms, which is what a ruleset hash
// needs (it is written into replays and training samples).
struct Hasher {
    std::uint64_t h = 1469598103934665603ull;
    void byte(std::uint8_t b) {
        h ^= b;
        h *= 1099511628211ull;
    }
    void i64(std::int64_t v) {
        for (int i = 0; i < 8; ++i) byte(static_cast<std::uint8_t>((v >> (i * 8)) & 0xFF));
    }
    void i32(int v) { i64(v); }
    void b(bool v) { byte(v ? 1 : 0); }
    void str(const std::string& s) {
        for (char c : s) byte(static_cast<std::uint8_t>(c));
        byte(0);
    }
};

}  // namespace

std::uint64_t RulesetConfig::hash() const {
    Hasher h;
    h.str(id);
    h.i32(version);
    h.i32(tick_rate);

    h.i32(geometry.width);
    h.i32(geometry.internal_height);
    h.i32(geometry.visible_height);

    h.i32(static_cast<int>(randomizer.type));
    h.i32(randomizer.preview_count);
    h.b(randomizer.hold_enabled);

    h.i32(static_cast<int>(movement.kick_table));
    h.b(movement.allow_180);
    h.i32(movement.gravity_num);
    h.i32(movement.gravity_den);
    h.i64(movement.lock_delay);
    h.i32(movement.reset_limit);
    h.i64(movement.das);
    h.i64(movement.arr);
    h.i64(movement.sdf);
    h.i64(movement.are);
    h.b(movement.spawn_above_stack);

    h.i32(static_cast<int>(clear_rules.spin_detection));
    h.b(clear_rules.all_clear_enabled);
    h.i64(clear_rules.line_clear_delay);

    h.i32(attack.single);
    h.i32(attack.doubl);
    h.i32(attack.triple);
    h.i32(attack.quad);
    h.i32(attack.tspin_mini_single);
    h.i32(attack.tspin_single);
    h.i32(attack.tspin_mini_double);
    h.i32(attack.tspin_double);
    h.i32(attack.tspin_triple);
    h.i32(attack.spin_mini_single);
    h.i32(attack.spin_mini_double);
    h.i32(attack.spin_mini_triple);
    h.i32(attack.all_clear);
    h.i32(attack.all_clear_b2b_bonus);
    h.i32(attack.combo_multiplier_num);
    h.i32(attack.combo_multiplier_den);
    h.b(attack.combo_log_for_zero_base);
    h.i32(attack.combo_cap);
    h.i32(static_cast<int>(attack.b2b_mode));
    h.i32(attack.b2b_charging_bonus);
    h.i32(attack.surge_start_streak);
    h.i32(attack.surge_base);
    h.i32(attack.surge_segments);
    h.i32(static_cast<int>(attack.rounding_mode));
    h.i32(attack.garbage_clear_bonus);
    h.i32(attack.opener_phase_pieces);
    h.b(attack.opener_phase_enabled);

    h.i64(garbage.travel_time);
    h.i64(garbage.activation_delay);
    h.i32(static_cast<int>(garbage.blocking_mode));
    h.i32(static_cast<int>(garbage.hole_change_rule));
    h.i32(garbage.messiness_percent);
    h.i32(garbage.messiness_between_attacks_percent);
    h.i32(garbage.cap);
    h.b(garbage.passthrough);
    h.b(garbage.cancellable);

    return h.h;
}

std::string RulesetConfig::hash_hex() const {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(hash()));
    return std::string(buf);
}

RulesetConfig RulesetConfig::tetra_league() {
    RulesetConfig c;
    c.id = "tetrio.league";
    c.version = 1;
    return c;  // defaults already model the TETRA LEAGUE ruleset
}

RulesetConfig RulesetConfig::quick_play() {
    RulesetConfig c;
    c.id = "tetrio.quickplay";
    c.version = 1;
    c.attack.rounding_mode = RoundingMode::Rng;
    c.attack.surge_base = 1;
    c.attack.all_clear = 3;
    c.attack.all_clear_b2b_bonus = 2;
    c.garbage.messiness_percent = 5;
    return c;
}

RulesetConfig RulesetConfig::guideline() {
    RulesetConfig c;
    c.id = "guideline";
    c.version = 1;
    c.movement.kick_table = KickTableId::SRS;
    c.movement.allow_180 = false;
    c.clear_rules.spin_detection = SpinDetection::TSpin;
    c.attack.b2b_mode = B2BMode::Chaining;
    c.attack.single = 0;
    c.attack.doubl = 1;
    c.attack.triple = 2;
    c.attack.quad = 4;
    c.attack.combo_log_for_zero_base = false;
    c.attack.garbage_clear_bonus = 0;
    c.attack.opener_phase_enabled = false;
    return c;
}

}  // namespace tetra