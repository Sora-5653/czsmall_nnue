// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- M0 rule core: piece shapes and kick tables.
#pragma once

#include "tetra/types.hpp"

#include <array>
#include <vector>

namespace tetra {

// Cell offset relative to the piece origin. +x right, +y UP (row 0 = bottom).
struct Offset {
    int x = 0;
    int y = 0;
};

// Piece shapes are expressed inside the classic SRS bounding box:
//   * I uses a 4x4 box, O uses a 4x4 box (guideline uses 2x2 but keeping the
//     same origin convention as I keeps the spawn maths uniform),
//   * every other piece uses a 3x3 box.
// The origin is the bottom-left corner of the bounding box; +y is up, which is
// the opposite vertical direction from the row-major tables usually published
// on the wikis, so these tables are written out explicitly and verified by the
// unit tests in tests/test_rotation.cpp.
struct PieceShape {
    std::array<Offset, 4> cells{};
    int box = 3;
};

using ShapeTable = std::array<std::array<PieceShape, ROT_COUNT>, PIECE_COUNT>;

namespace detail {

// Build a shape from a visual, top-row-first ASCII description so that the
// tables below can be read against the guideline diagrams directly.
inline PieceShape make_shape(int box, const char* rows) {
    PieceShape s;
    s.box = box;
    int n = 0;
    for (int r = 0; r < box; ++r) {
        for (int c = 0; c < box; ++c) {
            const char ch = rows[r * box + c];
            if (ch == '#') {
                // ASCII row 0 is the TOP of the box -> y = box-1-r.
                s.cells[static_cast<size_t>(n++)] = Offset{c, box - 1 - r};
            }
        }
    }
    return s;
}

inline ShapeTable build_shapes() {
    ShapeTable t{};

    // ---- I (4x4 box) ----
    t[static_cast<size_t>(Piece::I)][0] = make_shape(4, "...."
                                                       "####"
                                                       "...."
                                                       "....");
    t[static_cast<size_t>(Piece::I)][1] = make_shape(4, "..#."
                                                       "..#."
                                                       "..#."
                                                       "..#.");
    t[static_cast<size_t>(Piece::I)][2] = make_shape(4, "...."
                                                       "...."
                                                       "####"
                                                       "....");
    t[static_cast<size_t>(Piece::I)][3] = make_shape(4, ".#.."
                                                       ".#.."
                                                       ".#.."
                                                       ".#..");

    // ---- O (4x4 box, occupies the same two columns in every state) ----
    for (int r = 0; r < ROT_COUNT; ++r) {
        t[static_cast<size_t>(Piece::O)][static_cast<size_t>(r)] =
            make_shape(4, ".##."
                          ".##."
                          "...."
                          "....");
    }

    // ---- J ----
    t[static_cast<size_t>(Piece::J)][0] = make_shape(3, "#.."
                                                       "###"
                                                       "...");
    t[static_cast<size_t>(Piece::J)][1] = make_shape(3, ".##"
                                                       ".#."
                                                       ".#.");
    t[static_cast<size_t>(Piece::J)][2] = make_shape(3, "..."
                                                       "###"
                                                       "..#");
    t[static_cast<size_t>(Piece::J)][3] = make_shape(3, ".#."
                                                       ".#."
                                                       "##.");

    // ---- L ----
    t[static_cast<size_t>(Piece::L)][0] = make_shape(3, "..#"
                                                       "###"
                                                       "...");
    t[static_cast<size_t>(Piece::L)][1] = make_shape(3, ".#."
                                                       ".#."
                                                       ".##");
    t[static_cast<size_t>(Piece::L)][2] = make_shape(3, "..."
                                                       "###"
                                                       "#..");
    t[static_cast<size_t>(Piece::L)][3] = make_shape(3, "##."
                                                       ".#."
                                                       ".#.");

    // ---- S ----
    t[static_cast<size_t>(Piece::S)][0] = make_shape(3, ".##"
                                                       "##."
                                                       "...");
    t[static_cast<size_t>(Piece::S)][1] = make_shape(3, ".#."
                                                       ".##"
                                                       "..#");
    t[static_cast<size_t>(Piece::S)][2] = make_shape(3, "..."
                                                       ".##"
                                                       "##.");
    t[static_cast<size_t>(Piece::S)][3] = make_shape(3, "#.."
                                                       "##."
                                                       ".#.");

    // ---- Z ----
    t[static_cast<size_t>(Piece::Z)][0] = make_shape(3, "##."
                                                       ".##"
                                                       "...");
    t[static_cast<size_t>(Piece::Z)][1] = make_shape(3, "..#"
                                                       ".##"
                                                       ".#.");
    t[static_cast<size_t>(Piece::Z)][2] = make_shape(3, "..."
                                                       "##."
                                                       ".##");
    t[static_cast<size_t>(Piece::Z)][3] = make_shape(3, ".#."
                                                       "##."
                                                       "#..");

    // ---- T ----
    t[static_cast<size_t>(Piece::T)][0] = make_shape(3, ".#."
                                                       "###"
                                                       "...");
    t[static_cast<size_t>(Piece::T)][1] = make_shape(3, ".#."
                                                       ".##"
                                                       ".#.");
    t[static_cast<size_t>(Piece::T)][2] = make_shape(3, "..."
                                                       "###"
                                                       ".#.");
    t[static_cast<size_t>(Piece::T)][3] = make_shape(3, ".#."
                                                       "##."
                                                       ".#.");
    return t;
}

}  // namespace detail

inline const ShapeTable& shapes() {
    static const ShapeTable t = detail::build_shapes();
    return t;
}

inline const PieceShape& shape_of(Piece p, Rot r) {
    return shapes()[static_cast<size_t>(p)][static_cast<size_t>(r)];
}

// ---------------------------------------------------------------------------
// Kick tables
// ---------------------------------------------------------------------------
// A kick table maps (from_rot, to_rot) to the ordered list of translations to
// test. The first entry is always the pure rotation (0,0). +y is UP.
//
// Index a transition as `from * 4 + to`.
using KickList  = std::vector<Offset>;
using KickTable = std::array<KickList, 16>;

inline constexpr int kick_index(Rot from, Rot to) {
    return static_cast<int>(from) * 4 + static_cast<int>(to);
}

namespace detail {

// Guideline SRS, JLSTZ. Published tables use +y = up already.
inline KickTable build_jlstz_kicks() {
    KickTable k{};
    auto set = [&](Rot f, Rot t, std::initializer_list<Offset> l) {
        k[static_cast<size_t>(kick_index(f, t))] = KickList(l);
    };
    set(Rot::N,  Rot::R,  {{0,0},{-1,0},{-1,+1},{0,-2},{-1,-2}});
    set(Rot::R,  Rot::N,  {{0,0},{+1,0},{+1,-1},{0,+2},{+1,+2}});
    set(Rot::R,  Rot::S2, {{0,0},{+1,0},{+1,-1},{0,+2},{+1,+2}});
    set(Rot::S2, Rot::R,  {{0,0},{-1,0},{-1,+1},{0,-2},{-1,-2}});
    set(Rot::S2, Rot::L,  {{0,0},{+1,0},{+1,+1},{0,-2},{+1,-2}});
    set(Rot::L,  Rot::S2, {{0,0},{-1,0},{-1,-1},{0,+2},{-1,+2}});
    set(Rot::L,  Rot::N,  {{0,0},{-1,0},{-1,-1},{0,+2},{-1,+2}});
    set(Rot::N,  Rot::L,  {{0,0},{+1,0},{+1,+1},{0,-2},{+1,-2}});
    return k;
}

// Guideline SRS, I piece (the asymmetric original).
inline KickTable build_i_kicks_srs() {
    KickTable k{};
    auto set = [&](Rot f, Rot t, std::initializer_list<Offset> l) {
        k[static_cast<size_t>(kick_index(f, t))] = KickList(l);
    };
    set(Rot::N,  Rot::R,  {{0,0},{-2,0},{+1,0},{-2,-1},{+1,+2}});
    set(Rot::R,  Rot::N,  {{0,0},{+2,0},{-1,0},{+2,+1},{-1,-2}});
    set(Rot::R,  Rot::S2, {{0,0},{-1,0},{+2,0},{-1,+2},{+2,-1}});
    set(Rot::S2, Rot::R,  {{0,0},{+1,0},{-2,0},{+1,-2},{-2,+1}});
    set(Rot::S2, Rot::L,  {{0,0},{+2,0},{-1,0},{+2,+1},{-1,-2}});
    set(Rot::L,  Rot::S2, {{0,0},{-2,0},{+1,0},{-2,-1},{+1,+2}});
    set(Rot::L,  Rot::N,  {{0,0},{+1,0},{-2,0},{+1,-2},{-2,+1}});
    set(Rot::N,  Rot::L,  {{0,0},{-1,0},{+2,0},{-1,+2},{+2,-1}});
    return k;
}

// TETR.IO SRS+ : symmetric I kicks. Transcribed from the SRS+ table used by
// TETR.IO (as mirrored in python-tetris' TetrioSRS.srs_plus_i_kicks), converted
// from that library's (row-down, col) convention into (+x right, +y up).
inline KickTable build_i_kicks_srs_plus() {
    KickTable k{};
    auto set = [&](Rot f, Rot t, std::initializer_list<Offset> l) {
        k[static_cast<size_t>(kick_index(f, t))] = KickList(l);
    };
    set(Rot::N,  Rot::R,  {{0,0},{+1,0},{-2,0},{-2,-1},{+1,+2}});
    set(Rot::N,  Rot::L,  {{0,0},{-1,0},{+2,0},{+2,-1},{-1,+2}});
    set(Rot::R,  Rot::N,  {{0,0},{-1,0},{+2,0},{-1,-2},{+2,+1}});
    set(Rot::R,  Rot::S2, {{0,0},{-1,0},{+2,0},{-1,+2},{+2,-1}});
    set(Rot::S2, Rot::R,  {{0,0},{-2,0},{+1,0},{-2,+1},{+1,-2}});
    set(Rot::S2, Rot::L,  {{0,0},{+2,0},{-1,0},{+2,+1},{-1,-2}});
    set(Rot::L,  Rot::N,  {{0,0},{+1,0},{-2,0},{+1,-2},{-2,+1}});
    set(Rot::L,  Rot::S2, {{0,0},{+1,0},{-2,0},{+1,+2},{-2,-1}});
    return k;
}

// TETR.IO 180 kick table (osk's pinned table), shared by all pieces.
// Converted to (+x right, +y up).
inline void add_180_kicks(KickTable& k) {
    auto set = [&](Rot f, Rot t, std::initializer_list<Offset> l) {
        k[static_cast<size_t>(kick_index(f, t))] = KickList(l);
    };
    set(Rot::N,  Rot::S2, {{0,0},{0,+1},{+1,+1},{-1,+1},{+1,0},{-1,0}});
    set(Rot::S2, Rot::N,  {{0,0},{0,-1},{-1,-1},{+1,-1},{-1,0},{+1,0}});
    set(Rot::L,  Rot::R,  {{0,0},{+1,0},{+1,+2},{+1,+1},{0,+2},{0,+1}});
    set(Rot::R,  Rot::L,  {{0,0},{-1,0},{-1,+2},{-1,+1},{0,+2},{0,+1}});
}

}  // namespace detail

// Kick tables selected by RulesetConfig::movement.kick_table.
enum class KickTableId : std::uint8_t {
    SRS = 0,       // guideline SRS, no 180
    SRS_PLUS = 1,  // TETR.IO default: symmetric I kicks + 180 table
    SRS_X = 2,     // reserved (behaves as SRS+ for now)
    None = 3,      // no kicks at all
};

struct KickTables {
    KickTable jlstz;
    KickTable i;
    KickTable o;  // O piece: pure rotation only, unless a table says otherwise
};

inline const KickTables& kick_tables_for(KickTableId id) {
    static const KickTables srs = [] {
        KickTables t;
        t.jlstz = detail::build_jlstz_kicks();
        t.i = detail::build_i_kicks_srs();
        for (auto& l : t.o) l = KickList{{0, 0}};
        return t;
    }();
    static const KickTables srs_plus = [] {
        KickTables t;
        t.jlstz = detail::build_jlstz_kicks();
        detail::add_180_kicks(t.jlstz);
        t.i = detail::build_i_kicks_srs_plus();
        detail::add_180_kicks(t.i);
        for (auto& l : t.o) l = KickList{{0, 0}};
        return t;
    }();
    static const KickTables none = [] {
        KickTables t;
        for (auto& l : t.jlstz) l = KickList{{0, 0}};
        for (auto& l : t.i) l = KickList{{0, 0}};
        for (auto& l : t.o) l = KickList{{0, 0}};
        return t;
    }();

    switch (id) {
        case KickTableId::SRS: return srs;
        case KickTableId::None: return none;
        case KickTableId::SRS_PLUS:
        case KickTableId::SRS_X:
        default: return srs_plus;
    }
}

inline const KickList& kicks_for(const KickTables& t, Piece p, Rot from, Rot to) {
    static const KickList identity{{0, 0}};
    const KickTable& tab = (p == Piece::I) ? t.i : (p == Piece::O ? t.o : t.jlstz);
    const KickList& l = tab[static_cast<size_t>(kick_index(from, to))];
    return l.empty() ? identity : l;
}

}  // namespace tetra
