// SPDX-License-Identifier: MIT
// Stable data-contract identifiers shared by the engine and dataset tooling.
#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace tetra::schema {

// These values are deliberately independent of feature widths.  A model may
// still have TOKEN_FEATURES=24 while the meaning or order of the tokens has
// changed, so consumers must compare the schema identifiers as well.
inline constexpr std::uint32_t TOKENIZER_SCHEMA_VERSION = 2;
inline constexpr std::uint64_t TOKENIZER_SCHEMA_HASH = 0x5f1e2c9a7b43d816ull;
inline constexpr std::uint64_t OBSERVATION_SCHEMA_HASH = 0x8c74b1e2d6093a5full;
inline constexpr std::uint32_t ACTION_SCHEMA_VERSION = 1;

inline constexpr std::uint32_t LEGACY_AUX_TARGET_SCHEMA_VERSION = 1;
inline constexpr std::uint32_t AUX_TARGET_SCHEMA_VERSION = 2;
inline constexpr int LEGACY_AUX_TARGET_COUNT = 4;

inline constexpr int HORIZON_COUNT = 4;
inline constexpr int AUX_CHANNEL_COUNT = 4;
inline constexpr int INTERVAL_AUX_TARGET_COUNT = HORIZON_COUNT * AUX_CHANNEL_COUNT;
inline constexpr int AUX_TARGET_COUNT =
    LEGACY_AUX_TARGET_COUNT + 2 * INTERVAL_AUX_TARGET_COUNT;

inline constexpr std::array<std::string_view, 17> TOKEN_KIND_ORDER = {
    "row", "col", "board", "active", "hold", "next", "garbage",
    "counters", "event", "rule", "time", "opp_row", "opp_col", "opp_board",
    "missing", "bag", "opp_counters",
};

inline constexpr std::array<std::string_view, AUX_TARGET_COUNT> AUX_TARGET_NAMES = {
    "legacy_future_attack_1s",
    "legacy_future_garbage_received",
    "legacy_time_to_terminal",
    "legacy_topped_out_within_8",
    "real_0_1s_attack",
    "real_0_1s_garbage_received",
    "real_0_1s_self_topout",
    "real_0_1s_opponent_topout",
    "real_1_2s_attack",
    "real_1_2s_garbage_received",
    "real_1_2s_self_topout",
    "real_1_2s_opponent_topout",
    "real_2_4s_attack",
    "real_2_4s_garbage_received",
    "real_2_4s_self_topout",
    "real_2_4s_opponent_topout",
    "real_4_8s_attack",
    "real_4_8s_garbage_received",
    "real_4_8s_self_topout",
    "real_4_8s_opponent_topout",
    "placements_0_1_attack",
    "placements_0_1_garbage_received",
    "placements_0_1_self_topout",
    "placements_0_1_opponent_topout",
    "placements_1_2_attack",
    "placements_1_2_garbage_received",
    "placements_1_2_self_topout",
    "placements_1_2_opponent_topout",
    "placements_2_4_attack",
    "placements_2_4_garbage_received",
    "placements_2_4_self_topout",
    "placements_2_4_opponent_topout",
    "placements_4_8_attack",
    "placements_4_8_garbage_received",
    "placements_4_8_self_topout",
    "placements_4_8_opponent_topout",
};

inline constexpr int REAL_AUX_BASE = LEGACY_AUX_TARGET_COUNT;
inline constexpr int PLACEMENT_AUX_BASE =
    REAL_AUX_BASE + INTERVAL_AUX_TARGET_COUNT;

inline constexpr int attack_channel = 0;
inline constexpr int garbage_channel = 1;
inline constexpr int self_topout_channel = 2;
inline constexpr int opponent_topout_channel = 3;

inline constexpr int real_aux_index(int horizon, int channel) {
    return REAL_AUX_BASE + horizon * AUX_CHANNEL_COUNT + channel;
}

inline constexpr int placement_aux_index(int horizon, int channel) {
    return PLACEMENT_AUX_BASE + horizon * AUX_CHANNEL_COUNT + channel;
}

inline constexpr bool is_count_target(int index) {
    if (index == 0 || index == 1) return true;
    if (index >= REAL_AUX_BASE && index < AUX_TARGET_COUNT) {
        const int channel = (index - REAL_AUX_BASE) % AUX_CHANNEL_COUNT;
        return channel == attack_channel || channel == garbage_channel;
    }
    return false;
}

}  // namespace tetra::schema
