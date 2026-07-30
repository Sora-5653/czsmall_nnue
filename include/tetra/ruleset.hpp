// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- M0 rule core: versioned ruleset configuration.
//
// Spec section 6: no rule may be hardcoded in the engine. Everything that
// TETR.IO can patch or a custom room can toggle lives here, and every replay,
// training sample and model records the resulting `ruleset_hash`.
#pragma once

#include "tetra/pieces.hpp"
#include "tetra/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace tetra {

// --- randomizer ------------------------------------------------------------
enum class RandomizerType : std::uint8_t {
    Bag7 = 0,      // classic 7-bag
    Bag14 = 1,     // double bag
    Uniform = 2,   // memoryless
    OnePiece = 3,  // debugging: always the same piece
};

// --- spin detection --------------------------------------------------------
enum class SpinDetection : std::uint8_t {
    None = 0,       // no spin bonuses at all
    TSpin = 1,      // guideline 3-corner T only
    AllMini = 2,    // T uses 3-corner; other pieces use immobile => Mini
    AllMiniPlus = 3 // TETR.IO default since Beta 1.5.0: T may also go immobile
};

// --- back to back ----------------------------------------------------------
enum class B2BMode : std::uint8_t {
    Off = 0,
    Chaining = 1,  // pre-Beta 1.0.0: bonus grows in steps with the streak
    Charging = 2,  // TETR.IO default: flat +1, streak charges a Surge attack
};

// --- rounding --------------------------------------------------------------
enum class RoundingMode : std::uint8_t {
    Down = 0,  // TETRA LEAGUE / custom default
    Rng = 1,   // QUICK PLAY: fractional part is the probability of rounding up
};

// --- garbage ---------------------------------------------------------------
enum class GarbageBlocking : std::uint8_t {
    Combined = 0,   // incoming garbage is cancelled by outgoing attacks
    LimitedBlocking = 1,
    None = 2,       // attacks never cancel; everything lands
};

enum class GarbageHoleRule : std::uint8_t {
    PerAttack = 0,  // "change on attack": one hole column per attack group
    PerLine = 1,    // every line rolls its own column
    Static = 2,     // hole column never changes during the round
};

struct Geometry {
    int width = 10;
    int internal_height = 40;
    int visible_height = 20;
};

struct RandomizerCfg {
    RandomizerType type = RandomizerType::Bag7;
    int preview_count = 5;
    bool hold_enabled = true;
};

struct MovementCfg {
    KickTableId kick_table = KickTableId::SRS_PLUS;
    bool allow_180 = true;
    // Gravity in cells per tick, expressed as a rational to stay integer-exact.
    int gravity_num = 1;
    int gravity_den = 60;
    Tick lock_delay = 30;     // 500 ms at 60 Hz
    int reset_limit = 15;     // lock delay resets before a forced lock
    Tick das = 6;
    Tick arr = 0;
    Tick sdf = 0;             // 0 == infinite soft drop factor
    Tick are = 0;             // spawn delay
    bool spawn_above_stack = false;  // clutch-style spawn push
};

struct ClearRulesCfg {
    SpinDetection spin_detection = SpinDetection::AllMiniPlus;
    bool all_clear_enabled = true;
    Tick line_clear_delay = 0;
};

struct AttackCfg {
    // Base attack ("weight") per clear type. TETR.IO applies:
    //     attack = floor((base + b2b_bonus + garbage_bonus) * combo_multiplier)
    //              + all_clear_bonus  (+ surge on B2B break)
    int single = 0;
    int doubl = 1;
    int triple = 2;
    int quad = 4;
    int tspin_mini_single = 0;
    int tspin_single = 2;
    int tspin_mini_double = 1;
    int tspin_double = 4;
    int tspin_triple = 6;
    int spin_mini_single = 0;   // non-T immobile spins (All-Mini)
    int spin_mini_double = 1;
    int spin_mini_triple = 2;
    int all_clear = 10;         // 3 in QUICK PLAY
    int all_clear_b2b_bonus = 0;

    // combo: multiplier = 1 + combo_multiplier_num/den * combo
    int combo_multiplier_num = 1;
    int combo_multiplier_den = 4;
    // Zero-base clears in a combo fall back to floor(ln(1 + 1.25 * combo)).
    bool combo_log_for_zero_base = true;
    int combo_cap = -1;  // -1 = uncapped

    B2BMode b2b_mode = B2BMode::Charging;
    int b2b_charging_bonus = 1;      // flat +1 per attack while the streak is up
    int surge_start_streak = 4;      // streak at which Surge starts charging
    int surge_base = 4;              // 1 in QUICK PLAY
    int surge_segments = 3;

    RoundingMode rounding_mode = RoundingMode::Down;

    // Season 2: quads and spins that clear garbage send +1, outside the combo
    // multiplier.
    int garbage_clear_bonus = 1;

    // Opener phase: for the first N placements, cancel at double rate.
    int opener_phase_pieces = 14;
    bool opener_phase_enabled = true;
};

struct GarbageCfg {
    Tick travel_time = 20;      // ticks from send to arrival in the queue
    Tick activation_delay = 20; // ticks in queue before the lines may rise
    GarbageBlocking blocking_mode = GarbageBlocking::Combined;
    GarbageHoleRule hole_change_rule = GarbageHoleRule::PerAttack;
    int messiness_percent = 0;  // chance per line to reroll the hole column
    int messiness_between_attacks_percent = 100;
    int cap = 8;                // max lines that may rise at once (<=0: no cap)
    bool passthrough = false;   // disabled by default since Alpha 6.1.2
    bool cancellable = true;
};

// Full, versioned rule configuration.
struct RulesetConfig {
    std::string id = "tetrio.league";
    int version = 1;

    Geometry geometry{};
    RandomizerCfg randomizer{};
    MovementCfg movement{};
    ClearRulesCfg clear_rules{};
    AttackCfg attack{};
    GarbageCfg garbage{};

    int tick_rate = 60;  // ticks per second

    // Deterministic content hash over every field that can change play.
    std::uint64_t hash() const;
    std::string hash_hex() const;

    // Named presets.
    static RulesetConfig tetra_league();
    static RulesetConfig quick_play();
    static RulesetConfig guideline();
};

}  // namespace tetra
