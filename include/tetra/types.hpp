// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- M0 rule core: fundamental types.
#pragma once

#include <cstdint>
#include <string>
#include <array>

namespace tetra {

// ---------------------------------------------------------------------------
// Time
// ---------------------------------------------------------------------------
// Spec 5.2: never use floating point seconds internally. All timestamps are
// integer ticks. The tick rate is defined by RulesetConfig::tick_rate
// (default 60 ticks/second, i.e. one tick == one frame).
using Tick = std::int64_t;

inline constexpr Tick TICK_NEVER = INT64_MAX;

// ---------------------------------------------------------------------------
// Pieces
// ---------------------------------------------------------------------------
enum class Piece : std::uint8_t { I = 0, J = 1, L = 2, O = 3, S = 4, T = 5, Z = 6, None = 7 };

inline constexpr int PIECE_COUNT = 7;

inline const char* piece_name(Piece p) {
    switch (p) {
        case Piece::I: return "I";
        case Piece::J: return "J";
        case Piece::L: return "L";
        case Piece::O: return "O";
        case Piece::S: return "S";
        case Piece::T: return "T";
        case Piece::Z: return "Z";
        default: return ".";
    }
}

// Return the left-right mirrored twin of a piece (spec 14, 20):
//   * J <-> L, S <-> Z
//   * I, O, T are self-symmetric
inline Piece mirror_piece(Piece p) {
    switch (p) {
        case Piece::J: return Piece::L;
        case Piece::L: return Piece::J;
        case Piece::S: return Piece::Z;
        case Piece::Z: return Piece::S;
        default: return p;
    }
}

inline bool piece_from_char(char c, Piece& out) {
    switch (c) {
        case 'I': out = Piece::I; return true;
        case 'J': out = Piece::J; return true;
        case 'L': out = Piece::L; return true;
        case 'O': out = Piece::O; return true;
        case 'S': out = Piece::S; return true;
        case 'T': out = Piece::T; return true;
        case 'Z': out = Piece::Z; return true;
        default: return false;
    }
}

// Rotation states. 0 = spawn, 1 = R (one CW turn), 2 = 180, 3 = L (one CCW turn).
enum class Rot : std::uint8_t { N = 0, R = 1, S2 = 2, L = 3 };

inline constexpr int ROT_COUNT = 4;

inline Rot rot_cw(Rot r)   { return static_cast<Rot>((static_cast<int>(r) + 1) & 3); }
inline Rot rot_ccw(Rot r)  { return static_cast<Rot>((static_cast<int>(r) + 3) & 3); }
inline Rot rot_180(Rot r)  { return static_cast<Rot>((static_cast<int>(r) + 2) & 3); }

inline const char* rot_name(Rot r) {
    switch (r) {
        case Rot::N: return "N";
        case Rot::R: return "R";
        case Rot::S2: return "2";
        case Rot::L: return "L";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// Line clear / spin classification
// ---------------------------------------------------------------------------
enum class SpinType : std::uint8_t {
    None = 0,
    Mini = 1,  // mini spin (T mini, or any non-T immobile spin under All-Mini)
    Full = 2,  // full spin (T-spin proper, or All-Mini+ immobile T)
};

inline const char* spin_name(SpinType s) {
    switch (s) {
        case SpinType::None: return "none";
        case SpinType::Mini: return "mini";
        case SpinType::Full: return "full";
    }
    return "?";
}

// The clear descriptor is what the attack table is keyed on.
struct ClearDescriptor {
    int      lines        = 0;      // number of rows cleared (0..4+)
    SpinType spin         = SpinType::None;
    Piece    piece        = Piece::None;
    bool     all_clear    = false;  // board empty after the clear
    bool     cleared_garbage = false; // at least one cleared row contained garbage

    bool operator==(const ClearDescriptor& o) const {
        return lines == o.lines && spin == o.spin && piece == o.piece &&
               all_clear == o.all_clear && cleared_garbage == o.cleared_garbage;
    }
};

// Is this clear "difficult" (i.e. does it continue a B2B streak)?
// Quads and any spin clear are difficult; plain single/double/triple are not.
inline bool is_difficult_clear(const ClearDescriptor& c) {
    if (c.lines <= 0) return false;
    if (c.spin != SpinType::None) return true;
    return c.lines >= 4;
}

// ---------------------------------------------------------------------------
// Player-visible outcome
// ---------------------------------------------------------------------------
enum class TopoutReason : std::uint8_t {
    None = 0,
    BlockOut,   // spawned piece overlaps existing blocks
    LockOut,    // piece locked entirely above the visible ceiling
    GarbageOut, // rising garbage pushed the stack past the internal ceiling
};

inline const char* topout_name(TopoutReason r) {
    switch (r) {
        case TopoutReason::None: return "none";
        case TopoutReason::BlockOut: return "block_out";
        case TopoutReason::LockOut: return "lock_out";
        case TopoutReason::GarbageOut: return "garbage_out";
    }
    return "?";
}

// Attack pattern used to pressure the bot (spec 13.3's list of garbage styles).
enum class GarbageStyle : std::uint8_t {
    None = 0,
    Steady,     // a small attack at a fixed cadence
    Burst,      // occasional large attacks
    FastSmall,  // frequent single lines
    SlowLarge,  // rare, heavy attacks
};

}  // namespace tetra