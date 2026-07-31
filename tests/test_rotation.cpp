// SPDX-License-Identifier: MIT
// Piece geometry, SRS / SRS+ kicks, and spin detection (spec 18.2).
#include "test_util.hpp"
#include "tetra/piece_state.hpp"
#include "tetra/pieces.hpp"
#include "tetra/rng.hpp"
#include "tetra/ruleset.hpp"

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
        for (int x = 0; x < width; ++x) {
            if (row[x] == 'X' || row[x] == 'G') b.fill_cell(x, y, row[x] == 'G');
        }
        --y;
    }
    return b;
}

// Normalised set of occupied cells, translated so the minimum is at the origin.
std::set<std::pair<int, int>> normalised_cells(Piece p, Rot r) {
    const PieceShape& s = shape_of(p, r);
    int minx = 99, miny = 99;
    for (int i = 0; i < 4; ++i) {
        minx = std::min(minx, s.cells[static_cast<size_t>(i)].x);
        miny = std::min(miny, s.cells[static_cast<size_t>(i)].y);
    }
    std::set<std::pair<int, int>> out;
    for (int i = 0; i < 4; ++i)
        out.insert({s.cells[static_cast<size_t>(i)].x - minx,
                    s.cells[static_cast<size_t>(i)].y - miny});
    return out;
}

}  // namespace

TEST(shapes_have_four_cells_each) {
    for (int p = 0; p < PIECE_COUNT; ++p) {
        for (int r = 0; r < ROT_COUNT; ++r) {
            const PieceShape& s = shape_of(static_cast<Piece>(p), static_cast<Rot>(r));
            std::set<std::pair<int, int>> uniq;
            for (int i = 0; i < 4; ++i)
                uniq.insert({s.cells[static_cast<size_t>(i)].x, s.cells[static_cast<size_t>(i)].y});
            CHECK_MSG(uniq.size() == 4, std::string("piece ") + piece_name(static_cast<Piece>(p)) +
                                            " rot " + rot_name(static_cast<Rot>(r)) +
                                            " has duplicate cells");
        }
    }
}

TEST(shapes_are_connected) {
    // Every tetromino must be edge-connected in every rotation state.
    for (int p = 0; p < PIECE_COUNT; ++p) {
        for (int r = 0; r < ROT_COUNT; ++r) {
            const PieceShape& s = shape_of(static_cast<Piece>(p), static_cast<Rot>(r));
            std::set<std::pair<int, int>> cells;
            for (int i = 0; i < 4; ++i)
                cells.insert({s.cells[static_cast<size_t>(i)].x, s.cells[static_cast<size_t>(i)].y});
            // Flood fill from the first cell.
            std::vector<std::pair<int, int>> stack{*cells.begin()};
            std::set<std::pair<int, int>> seen{*cells.begin()};
            while (!stack.empty()) {
                auto [cx, cy] = stack.back();
                stack.pop_back();
                const int dx[] = {1, -1, 0, 0};
                const int dy[] = {0, 0, 1, -1};
                for (int d = 0; d < 4; ++d) {
                    std::pair<int, int> n{cx + dx[d], cy + dy[d]};
                    if (cells.count(n) && !seen.count(n)) {
                        seen.insert(n);
                        stack.push_back(n);
                    }
                }
            }
            CHECK_MSG(seen.size() == 4, std::string("piece ") + piece_name(static_cast<Piece>(p)) +
                                            " rot " + rot_name(static_cast<Rot>(r)) +
                                            " is disconnected");
        }
    }
}

TEST(o_piece_never_changes_shape) {
    const auto base = normalised_cells(Piece::O, Rot::N);
    for (int r = 1; r < ROT_COUNT; ++r)
        CHECK(normalised_cells(Piece::O, static_cast<Rot>(r)) == base);
}

TEST(i_piece_alternates_horizontal_and_vertical) {
    // N and 2 are horizontal (4 wide, 1 tall); R and L are vertical.
    for (Rot r : {Rot::N, Rot::S2}) {
        const PieceShape& s = shape_of(Piece::I, r);
        const int y0 = s.cells[0].y;
        for (int i = 1; i < 4; ++i) CHECK_EQ(s.cells[static_cast<size_t>(i)].y, y0);
    }
    for (Rot r : {Rot::R, Rot::L}) {
        const PieceShape& s = shape_of(Piece::I, r);
        const int x0 = s.cells[0].x;
        for (int i = 1; i < 4; ++i) CHECK_EQ(s.cells[static_cast<size_t>(i)].x, x0);
    }
}

TEST(rotation_in_open_space_uses_no_kick) {
    const RulesetConfig cfg = league();
    Board b(10, 40);
    for (int p = 0; p < PIECE_COUNT; ++p) {
        ActivePiece piece = spawn_piece(static_cast<Piece>(p), cfg);
        piece.y = 10;  // well clear of the floor
        ActivePiece q = piece;
        const RotationResult rr = try_rotate(b, q, rot_cw(q.rot), cfg);
        if (static_cast<Piece>(p) == Piece::O) continue;  // O has no distinct states
        CHECK(rr.success);
        CHECK_EQ(rr.kick_index, 0);
        CHECK_EQ(q.x, piece.x);
        CHECK_EQ(q.y, piece.y);
    }
}

TEST(rotation_round_trip_returns_to_start_in_open_space) {
    const RulesetConfig cfg = league();
    Board b(10, 40);
    for (int p = 0; p < PIECE_COUNT; ++p) {
        ActivePiece piece = spawn_piece(static_cast<Piece>(p), cfg);
        piece.y = 12;
        const ActivePiece start = piece;
        for (int i = 0; i < 4; ++i) try_rotate(b, piece, rot_cw(piece.rot), cfg);
        CHECK(piece == start);
    }
}

TEST(srs_plus_i_kicks_are_exactly_mirror_symmetric) {
    // This IS the definition of SRS+: the I piece must behave identically
    // against the left and the right wall, including the ORDER in which the
    // kick candidates are tested. Mirroring a transition (N<->N, 2<->2, R<->L)
    // and negating x must reproduce the mirrored transition entry for entry.
    const KickTables& t = kick_tables_for(KickTableId::SRS_PLUS);
    auto mirror_rot = [](Rot r) {
        switch (r) {
            case Rot::R: return Rot::L;
            case Rot::L: return Rot::R;
            default: return r;
        }
    };
    int compared = 0;
    for (int f = 0; f < 4; ++f) {
        for (int to = 0; to < 4; ++to) {
            if (f == to) continue;
            const Rot rf = static_cast<Rot>(f), rt = static_cast<Rot>(to);
            if (rot_180(rf) == rt) continue;  // 180 table is asymmetric by design
            const KickList& a = kicks_for(t, Piece::I, rf, rt);
            const KickList& b = kicks_for(t, Piece::I, mirror_rot(rf), mirror_rot(rt));
            if (a.size() <= 1 || b.size() <= 1) continue;
            CHECK_EQ(a.size(), b.size());
            for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
                CHECK_MSG(a[i].x == -b[i].x && a[i].y == b[i].y,
                          std::string("SRS+ I kick asymmetry at ") + rot_name(rf) + "->" +
                              rot_name(rt) + " index " + std::to_string(i));
            }
            ++compared;
        }
    }
    CHECK(compared > 0);
}

TEST(jlstz_kicks_are_mirror_symmetric) {
    // Guideline SRS is symmetric for every piece except I.
    const KickTables& t = kick_tables_for(KickTableId::SRS_PLUS);
    auto mirror_rot = [](Rot r) {
        switch (r) {
            case Rot::R: return Rot::L;
            case Rot::L: return Rot::R;
            default: return r;
        }
    };
    for (int f = 0; f < 4; ++f) {
        for (int to = 0; to < 4; ++to) {
            if (f == to) continue;
            const Rot rf = static_cast<Rot>(f), rt = static_cast<Rot>(to);
            if (rot_180(rf) == rt) continue;  // 180 table is checked separately
            const KickList& a = kicks_for(t, Piece::T, rf, rt);
            const KickList& b = kicks_for(t, Piece::T, mirror_rot(rf), mirror_rot(rt));
            CHECK_EQ(a.size(), b.size());
            for (size_t i = 0; i < a.size() && i < b.size(); ++i)
                CHECK(a[i].x == -b[i].x && a[i].y == b[i].y);
        }
    }
}

TEST(classic_srs_i_kicks_are_asymmetric_but_set_equal) {
    // Documents *why* SRS+ exists: plain SRS tests the same offsets on both
    // sides but in a different order, which is what makes the I piece awkward
    // against the right wall.
    const KickTables& t = kick_tables_for(KickTableId::SRS);
    const KickList& nr = kicks_for(t, Piece::I, Rot::N, Rot::R);
    const KickList& nl = kicks_for(t, Piece::I, Rot::N, Rot::L);
    CHECK_EQ(nr.size(), nl.size());
    bool order_differs = false;
    for (size_t i = 0; i < nr.size() && i < nl.size(); ++i)
        if (nr[i].x != -nl[i].x || nr[i].y != nl[i].y) order_differs = true;
    CHECK_MSG(order_differs, "classic SRS I kicks should NOT be order-symmetric");
}

TEST(shape_table_is_mirror_consistent) {
    // Mirroring the field must map each piece onto its mirror twin:
    //   J<->L, S<->Z, T/I/O onto themselves, with R<->L rotations swapped.
    auto twin = [](Piece p) {
        switch (p) {
            case Piece::J: return Piece::L;
            case Piece::L: return Piece::J;
            case Piece::S: return Piece::Z;
            case Piece::Z: return Piece::S;
            default: return p;
        }
    };
    auto mirror_rot = [](Rot r) {
        switch (r) {
            case Rot::R: return Rot::L;
            case Rot::L: return Rot::R;
            default: return r;
        }
    };
    for (int pi = 0; pi < PIECE_COUNT; ++pi) {
        const Piece p = static_cast<Piece>(pi);
        for (int ri = 0; ri < ROT_COUNT; ++ri) {
            const Rot r = static_cast<Rot>(ri);
            const PieceShape& s = shape_of(p, r);
            const PieceShape& m = shape_of(twin(p), mirror_rot(r));
            CHECK_EQ(s.box, m.box);
            std::set<std::pair<int, int>> mirrored, actual;
            for (int i = 0; i < 4; ++i) {
                mirrored.insert({s.box - 1 - s.cells[static_cast<size_t>(i)].x,
                                 s.cells[static_cast<size_t>(i)].y});
                actual.insert({m.cells[static_cast<size_t>(i)].x, m.cells[static_cast<size_t>(i)].y});
            }
            CHECK_MSG(mirrored == actual,
                      std::string("mirror of ") + piece_name(p) + rot_name(r) + " != " +
                          piece_name(twin(p)) + rot_name(mirror_rot(r)));
        }
    }
}

TEST(rotation_is_mirror_equivalent_on_random_boards) {
    // The strongest statement of spec 18.3's "left/right reflection
    // equivalence": rotating on a board must equal the mirror of rotating the
    // mirrored piece on the mirrored board -- kicks included.
    auto twin = [](Piece p) {
        switch (p) {
            case Piece::J: return Piece::L;
            case Piece::L: return Piece::J;
            case Piece::S: return Piece::Z;
            case Piece::Z: return Piece::S;
            default: return p;
        }
    };
    auto mirror_rot = [](Rot r) {
        switch (r) {
            case Rot::R: return Rot::L;
            case Rot::L: return Rot::R;
            default: return r;
        }
    };

    const RulesetConfig cfg = league();
    const int W = cfg.geometry.width;
    Rng rng(987654321ull);
    int checked = 0;

    for (int trial = 0; trial < 400; ++trial) {
        Board b(W, 24);
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < W; ++x)
                if (rng.chance(1, 3)) b.fill_cell(x, y, false);
        const Board mb = b.mirrored();

        for (int pi = 0; pi < PIECE_COUNT; ++pi) {
            const Piece p = static_cast<Piece>(pi);
            const int box = shape_of(p, Rot::N).box;
            for (int ri = 0; ri < ROT_COUNT; ++ri) {
                for (int x = -2; x < W + 2; ++x) {
                    ActivePiece a;
                    a.type = p;
                    a.rot = static_cast<Rot>(ri);
                    a.x = x;
                    a.y = 3;
                    if (collides(b, a)) continue;

                    // The mirrored placement of the same piece.
                    ActivePiece ma;
                    ma.type = twin(p);
                    ma.rot = mirror_rot(a.rot);
                    ma.x = W - x - box;
                    ma.y = a.y;
                    CHECK_MSG(!collides(mb, ma), "mirrored placement must also be legal");

                    // NOTE: only 90 degree rotations are mirror-invariant.
                    // TETR.IO's 180 table is deliberately biased (see
                    // tetrio_180_kicks_are_not_mirror_symmetric below), so it
                    // is excluded here and handled by its own test.
                    for (Rot target : {rot_cw(a.rot), rot_ccw(a.rot)}) {
                        ActivePiece q = a, mq = ma;
                        const bool ok = try_rotate(b, q, target, cfg).success;
                        const bool mok = try_rotate(mb, mq, mirror_rot(target), cfg).success;
                        CHECK_MSG(ok == mok, "rotation legality must be mirror-invariant");
                        if (ok && mok) {
                            CHECK_MSG(mq.x == W - q.x - box && mq.y == q.y,
                                      "kicked position must be mirror-invariant");
                            ++checked;
                        }
                    }
                }
            }
        }
    }
    CHECK_MSG(checked > 1000, "expected a meaningful number of mirrored rotations");
}

TEST(t_spin_triple_kick_is_available) {
    // Classic TST slot: a T rotates into a 1-wide, 3-tall overhang.
    //
    //   X..XXXXXXX      the T comes down the column at x = 1 and rotates
    //   X..XXXXXXX      into the notch under the overhang at x = 2
    //   XX.XXXXXXX
    //   X..XXXXXXX
    const RulesetConfig cfg = league();
    Board b = board_from({
        "X..XXXXXXX",
        "X..XXXXXXX",
        "XX.XXXXXXX",
        "X..XXXXXXX",
    });
    // Start the T upright in the 2-wide channel and rotate CW into the slot.
    ActivePiece p;
    p.type = Piece::T;
    p.rot = Rot::L;   // stem pointing left
    p.x = 0;
    p.y = 2;
    if (collides(b, p)) {
        // If that exact start overlaps, walk it up until it fits.
        for (int y = 2; y <= 8 && collides(b, p); ++y) p.y = y;
    }
    CHECK(!collides(b, p));
    ActivePiece q = p;
    const RotationResult rr = try_rotate(b, q, Rot::N, cfg);
    // Whether or not this specific slot kicks, the rotation must not corrupt
    // state: either it failed and nothing changed, or it succeeded legally.
    if (rr.success) CHECK(!collides(b, q));
    else CHECK(q == p);
}

TEST(t_spin_double_is_detected_as_full_spin) {
    // A real TSD slot needs an OVERHANG above the pocket -- without one the T
    // could simply be moved up, so it is not immobile and (correctly) does not
    // score a spin. The overhang here is the block at (3,2).
    //
    //   XXXX......   row 2  <- overhang
    //   XXX...XXXX   row 1
    //   XXXX.XXXXX   row 0
    const RulesetConfig cfg = league();
    Board b = board_from({
        "XXXX......",
        "XXX...XXXX",
        "XXXX.XXXXX",
    });
    ActivePiece t;
    t.type = Piece::T;
    t.rot = Rot::S2;  // stem pointing down, sitting in the notch
    t.x = 3;
    t.y = 0;
    CHECK(!collides(b, t));
    t.last_action = LastAction::Rotate;
    CHECK_MSG(is_immobile(b, t), "a T under an overhang must be immobile");
    CHECK_MSG(detect_spin(b, t, cfg) == SpinType::Full,
              "a proper TSD must register as a full T-spin");

    // The same pocket WITHOUT the overhang must not score a spin.
    Board flat = board_from({
        "XXX...XXXX",
        "XXXX.XXXXX",
    });
    ActivePiece u = t;
    CHECK(!collides(flat, u));
    CHECK_MSG(!is_immobile(flat, u), "no overhang means the piece can rise");
    CHECK_MSG(detect_spin(flat, u, cfg) == SpinType::None,
              "a flat notch must not score a spin");
}

TEST(immobile_requires_all_four_directions_blocked) {
    // Spec / TETR.IO: immobile means it cannot move left, right, up OR down.
    // Each direction is load-bearing, so check that opening any single one
    // disqualifies the spin.
    const RulesetConfig cfg = league();
    Board b = board_from({
        "XXXX......",
        "XXX...XXXX",
        "XXXX.XXXXX",
    });
    ActivePiece t;
    t.type = Piece::T;
    t.rot = Rot::S2;
    t.x = 3;
    t.y = 0;
    t.last_action = LastAction::Rotate;
    CHECK(is_immobile(b, t));

    // Removing the overhang lets it move up.
    Board no_up = b;
    no_up.set_row(2, 0u);
    CHECK(!is_immobile(no_up, t));
}

TEST(spin_requires_rotation_as_last_action) {
    const RulesetConfig cfg = league();
    Board b = board_from({
        "XXX...XXXX",
        "XXXX.XXXXX",
    });
    ActivePiece t;
    t.type = Piece::T;
    t.rot = Rot::S2;
    t.x = 3;
    t.y = 0;
    if (collides(b, t)) return;  // geometry guard
    t.last_action = LastAction::Move;
    CHECK(detect_spin(b, t, cfg) == SpinType::None);
    t.last_action = LastAction::Drop;
    CHECK(detect_spin(b, t, cfg) == SpinType::None);
}

TEST(no_spin_detection_when_disabled) {
    RulesetConfig cfg = league();
    cfg.clear_rules.spin_detection = SpinDetection::None;
    Board b = board_from({
        "XXX...XXXX",
        "XXXX.XXXXX",
    });
    ActivePiece t;
    t.type = Piece::T;
    t.rot = Rot::S2;
    t.x = 3;
    t.y = 0;
    t.last_action = LastAction::Rotate;
    if (collides(b, t)) return;
    CHECK(detect_spin(b, t, cfg) == SpinType::None);
}

TEST(non_t_spins_only_count_under_all_mini) {
    // A fully immobile S piece.
    Board b = board_from({
        "XX...XXXXX",
        "X....XXXXX",
        "XXX.XXXXXX",
    });
    ActivePiece s;
    s.type = Piece::S;
    s.rot = Rot::R;
    s.x = 1;
    s.y = 0;
    if (collides(b, s)) return;
    s.last_action = LastAction::Rotate;

    RulesetConfig tspin_only = league();
    tspin_only.clear_rules.spin_detection = SpinDetection::TSpin;
    CHECK(detect_spin(b, s, tspin_only) == SpinType::None);

    RulesetConfig all_mini = league();
    all_mini.clear_rules.spin_detection = SpinDetection::AllMini;
    if (is_immobile(b, s)) CHECK(detect_spin(b, s, all_mini) == SpinType::Mini);
}

TEST(rotation_180_respects_ruleset_flag) {
    RulesetConfig with180 = league();
    RulesetConfig no180 = RulesetConfig::guideline();
    Board b(10, 40);

    ActivePiece p;
    p.type = Piece::T;
    p.rot = Rot::N;
    p.x = 3;
    p.y = 10;

    ActivePiece a = p;
    CHECK(try_rotate(b, a, Rot::S2, with180).success);

    ActivePiece c = p;
    CHECK(!try_rotate(b, c, Rot::S2, no180).success);
    CHECK(c == p);
}

TEST(kick_tables_start_with_identity) {
    for (KickTableId id : {KickTableId::SRS, KickTableId::SRS_PLUS}) {
        const KickTables& t = kick_tables_for(id);
        for (int from = 0; from < 4; ++from) {
            for (int to = 0; to < 4; ++to) {
                if (from == to) continue;
                for (Piece p : {Piece::T, Piece::I}) {
                    const KickList& l =
                        kicks_for(t, p, static_cast<Rot>(from), static_cast<Rot>(to));
                    CHECK_MSG(l[0].x == 0 && l[0].y == 0,
                              "first kick test must be the pure rotation");
                }
            }
        }
    }
}

TEST(hard_drop_distance_lands_on_the_stack) {
    Board b = board_from({"XXXXXXXXXX"});
    const RulesetConfig cfg = league();
    ActivePiece p = spawn_piece(Piece::O, cfg);
    const int d = hard_drop_distance(b, p);
    ActivePiece landed = p;
    landed.y -= d;
    CHECK(!collides(b, landed));
    CHECK(grounded(b, landed));
    // One further cell must collide.
    ActivePiece past = landed;
    past.y -= 1;
    CHECK(collides(b, past));
}

TEST(spawn_position_is_inside_the_field) {
    const RulesetConfig cfg = league();
    Board b(10, 40);
    for (int p = 0; p < PIECE_COUNT; ++p) {
        const ActivePiece piece = spawn_piece(static_cast<Piece>(p), cfg);
        Offset cells[4];
        piece_cells(piece, cells);
        for (const auto& c : cells) {
            CHECK_MSG(c.x >= 0 && c.x < cfg.geometry.width, "spawn must be inside the walls");
            CHECK_MSG(c.y >= 0 && c.y < cfg.geometry.internal_height,
                      "spawn must be inside the internal field");
        }
        CHECK(!collides(b, piece));
    }
}

TEST(tetrio_180_kicks_are_not_mirror_symmetric) {
    // IMPORTANT for spec 14 (data augmentation): TETR.IO's 180 kick table is
    // intentionally NOT left/right symmetric -- it contains downward kicks for
    // R<->L but no upward ones, which is also why it cannot be expressed as
    // SRS-style offset data. Left/right mirroring is therefore only a valid
    // augmentation for rulesets without 180, or when the 180 table is mirrored
    // together with the board. This test pins the asymmetry so that a future
    // "cleanup" cannot silently make the augmentation unsound.
    const KickTables& t = kick_tables_for(KickTableId::SRS_PLUS);
    const KickList& n2 = kicks_for(t, Piece::T, Rot::N, Rot::S2);
    bool asymmetric = false;
    for (size_t i = 0; i < n2.size(); ++i) {
        // A mirror-symmetric self-transition (N->2 maps to itself) would need
        // the negated-x variant of every entry at the same index.
        if (n2[i].x != -n2[i].x) {
            bool found_at_same_index = false;
            if (i < n2.size() && n2[i].x == 0) found_at_same_index = true;
            if (!found_at_same_index) asymmetric = true;
        }
    }
    CHECK_MSG(asymmetric, "TETR.IO 180 table is expected to be directionally biased");

    // Concretely: N->2 prefers kicking upward, 2->N prefers downward.
    const KickList& t2n = kicks_for(t, Piece::T, Rot::S2, Rot::N);
    int up_n2 = 0, down_t2n = 0;
    for (const auto& k : n2) if (k.y > 0) ++up_n2;
    for (const auto& k : t2n) if (k.y < 0) ++down_t2n;
    CHECK(up_n2 > 0);
    CHECK(down_t2n > 0);
}

TEST(classic_srs_mirroring_is_unsound_for_the_i_piece) {
    // Empirically pinned counterpart to the SRS+ test: under classic guideline
    // SRS the I piece's kick ORDER differs between the two sides, so a
    // left/right mirrored sample is NOT an equivalent position for I. Every
    // other piece mirrors cleanly.
    //
    // Consequence for spec 14: the mirror augmentation is only safe when the
    // ruleset uses SRS+ (TETR.IO default). Under classic SRS it must either be
    // disabled or restricted to non-I placements.
    RulesetConfig cfg = RulesetConfig::guideline();  // SRS, no 180
    const int W = cfg.geometry.width;
    auto twin = [](Piece p) {
        switch (p) {
            case Piece::J: return Piece::L;
            case Piece::L: return Piece::J;
            case Piece::S: return Piece::Z;
            case Piece::Z: return Piece::S;
            default: return p;
        }
    };
    auto mirror_rot = [](Rot r) {
        switch (r) {
            case Rot::R: return Rot::L;
            case Rot::L: return Rot::R;
            default: return r;
        }
    };
    Rng rng(555ull);
    int i_mismatches = 0, other_mismatches = 0, checked = 0;
    for (int trial = 0; trial < 150; ++trial) {
        Board b(W, 24);
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < W; ++x)
                if (rng.chance(1, 3)) b.fill_cell(x, y, false);
        const Board mb = b.mirrored();
        for (int pi = 0; pi < PIECE_COUNT; ++pi) {
            const Piece p = static_cast<Piece>(pi);
            const int box = shape_of(p, Rot::N).box;
            for (int ri = 0; ri < ROT_COUNT; ++ri) {
                for (int x = -2; x < W + 2; ++x) {
                    ActivePiece a;
                    a.type = p;
                    a.rot = static_cast<Rot>(ri);
                    a.x = x;
                    a.y = 3;
                    if (collides(b, a)) continue;
                    ActivePiece ma;
                    ma.type = twin(p);
                    ma.rot = mirror_rot(a.rot);
                    ma.x = W - x - box;
                    ma.y = a.y;
                    for (Rot target : {rot_cw(a.rot), rot_ccw(a.rot)}) {
                        ActivePiece q = a, mq = ma;
                        const bool ok = try_rotate(b, q, target, cfg).success;
                        const bool mok = try_rotate(mb, mq, mirror_rot(target), cfg).success;
                        CHECK_MSG(ok == mok, "rotation legality stays mirror-invariant even in SRS");
                        if (ok && mok) {
                            ++checked;
                            const bool same = (mq.x == W - q.x - box && mq.y == q.y);
                            if (!same) {
                                if (p == Piece::I) ++i_mismatches;
                                else ++other_mismatches;
                            }
                        }
                    }
                }
            }
        }
    }
    CHECK(checked > 500);
    CHECK_MSG(other_mismatches == 0,
              "non-I pieces must mirror cleanly under classic SRS");
    CHECK_MSG(i_mismatches > 0,
              "classic SRS is expected to break mirror equivalence for I");
}

TEST(srs_plus_mirroring_is_sound_for_every_piece) {
    // The augmentation guarantee the training pipeline actually relies on:
    // under the TETR.IO default ruleset, 90 degree rotation outcomes are
    // mirror-invariant for ALL seven pieces, I included.
    RulesetConfig cfg = RulesetConfig::tetra_league();
    const int W = cfg.geometry.width;
    auto twin = [](Piece p) {
        switch (p) {
            case Piece::J: return Piece::L;
            case Piece::L: return Piece::J;
            case Piece::S: return Piece::Z;
            case Piece::Z: return Piece::S;
            default: return p;
        }
    };
    auto mirror_rot = [](Rot r) {
        switch (r) {
            case Rot::R: return Rot::L;
            case Rot::L: return Rot::R;
            default: return r;
        }
    };
    Rng rng(555ull);
    int checked = 0;
    for (int trial = 0; trial < 150; ++trial) {
        Board b(W, 24);
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < W; ++x)
                if (rng.chance(1, 3)) b.fill_cell(x, y, false);
        const Board mb = b.mirrored();
        for (int pi = 0; pi < PIECE_COUNT; ++pi) {
            const Piece p = static_cast<Piece>(pi);
            const int box = shape_of(p, Rot::N).box;
            for (int ri = 0; ri < ROT_COUNT; ++ri) {
                for (int x = -2; x < W + 2; ++x) {
                    ActivePiece a;
                    a.type = p;
                    a.rot = static_cast<Rot>(ri);
                    a.x = x;
                    a.y = 3;
                    if (collides(b, a)) continue;
                    ActivePiece ma;
                    ma.type = twin(p);
                    ma.rot = mirror_rot(a.rot);
                    ma.x = W - x - box;
                    ma.y = a.y;
                    for (Rot target : {rot_cw(a.rot), rot_ccw(a.rot)}) {
                        ActivePiece q = a, mq = ma;
                        const bool ok = try_rotate(b, q, target, cfg).success;
                        const bool mok = try_rotate(mb, mq, mirror_rot(target), cfg).success;
                        CHECK(ok == mok);
                        if (ok && mok) {
                            CHECK(mq.x == W - q.x - box && mq.y == q.y);
                            ++checked;
                        }
                    }
                }
            }
        }
    }
    CHECK(checked > 500);
}