// SPDX-License-Identifier: MIT
// Legal placement generation (spec 8, 18.2).
#include "test_util.hpp"
#include "tetra/movegen.hpp"
#include "tetra/rng.hpp"

#include <map>
#include <set>
#include <algorithm>
#include <tuple>

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

// Replay a canonical input sequence and return the resulting piece.
// This is the ground truth check: the sequence must land exactly where the
// generator claimed (spec 8.3).
bool replay(const Board& b, Piece piece, const std::vector<Input>& seq, const RulesetConfig& cfg,
            ActivePiece& out) {
    ActivePiece p = spawn_piece(piece, cfg);
    if (collides(b, p)) return false;
    for (Input in : seq) {
        switch (in) {
            case Input::Hold:
                break;  // the caller has already swapped the piece
            case Input::Left: {
                ActivePiece q = p;
                q.x -= 1;
                if (!collides(b, q)) { q.last_action = LastAction::Move; q.last_kick = 0; p = q; }
                break;
            }
            case Input::Right: {
                ActivePiece q = p;
                q.x += 1;
                if (!collides(b, q)) { q.last_action = LastAction::Move; q.last_kick = 0; p = q; }
                break;
            }
            case Input::DasLeft: {
                while (true) {
                    ActivePiece q = p;
                    q.x -= 1;
                    if (collides(b, q)) break;
                    q.last_action = LastAction::Move;
                    q.last_kick = 0;
                    p = q;
                }
                break;
            }
            case Input::DasRight: {
                while (true) {
                    ActivePiece q = p;
                    q.x += 1;
                    if (collides(b, q)) break;
                    q.last_action = LastAction::Move;
                    q.last_kick = 0;
                    p = q;
                }
                break;
            }
            case Input::Cw: try_rotate(b, p, rot_cw(p.rot), cfg); break;
            case Input::Ccw: try_rotate(b, p, rot_ccw(p.rot), cfg); break;
            case Input::Flip: try_rotate(b, p, rot_180(p.rot), cfg); break;
            case Input::SoftDrop: {
                const int d = hard_drop_distance(b, p);
                p.y -= d;
                if (d > 0) { p.last_action = LastAction::Drop; p.last_kick = 0; }
                break;
            }
            case Input::HardDrop: {
                const int d = hard_drop_distance(b, p);
                p.y -= d;
                if (d > 0) { p.last_action = LastAction::Drop; p.last_kick = 0; }
                break;
            }
        }
    }
    out = p;
    return true;
}

}  // namespace

TEST(movegen_finds_all_columns_on_an_empty_board) {
    const RulesetConfig cfg = league();
    Board b(10, 40);
    MoveGenerator gen;

    // O piece: 9 distinct placements on an empty 10-wide board.
    const auto o = gen.generate_for_piece(b, Piece::O, cfg, false);
    std::set<int> o_cols;
    for (const auto& a : o) o_cols.insert(a.final_x);
    CHECK_EQ(static_cast<int>(o.size()), 9);

    // I piece: 7 horizontal + 10 vertical = 17 distinct resting positions.
    const auto i = gen.generate_for_piece(b, Piece::I, cfg, false);
    CHECK_EQ(static_cast<int>(i.size()), 17);

    // T piece: 4 rotations x their valid columns = 8 + 9 + 8 + 9 = 34.
    const auto t = gen.generate_for_piece(b, Piece::T, cfg, false);
    CHECK_EQ(static_cast<int>(t.size()), 34);

    // S and Z are symmetric to each other.
    const auto s = gen.generate_for_piece(b, Piece::S, cfg, false);
    const auto z = gen.generate_for_piece(b, Piece::Z, cfg, false);
    CHECK_EQ(s.size(), z.size());

    // J and L likewise.
    const auto j = gen.generate_for_piece(b, Piece::J, cfg, false);
    const auto l = gen.generate_for_piece(b, Piece::L, cfg, false);
    CHECK_EQ(j.size(), l.size());
}

TEST(every_generated_placement_is_legal_and_grounded) {
    const RulesetConfig cfg = league();
    MoveGenerator gen;
    Rng rng(31337);

    for (int trial = 0; trial < 60; ++trial) {
        Board b(10, 40);
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 10; ++x)
                if (rng.chance(2, 5)) b.fill_cell(x, y, rng.chance(1, 3));

        for (int pi = 0; pi < PIECE_COUNT; ++pi) {
            const Piece p = static_cast<Piece>(pi);
            for (const auto& a : gen.generate_for_piece(b, p, cfg, false)) {
                const ActivePiece piece = a.piece_state();
                CHECK_MSG(!collides(b, piece), "generated placement must not overlap");
                CHECK_MSG(grounded(b, piece), "generated placement must rest on support");
                // It must also be inside the walls.
                Offset cells[4];
                piece_cells(piece, cells);
                for (const auto& c : cells) {
                    CHECK(c.x >= 0 && c.x < 10);
                    CHECK(c.y >= 0);
                }
            }
        }
    }
}

TEST(canonical_input_sequence_reproduces_the_placement) {
    // The most important movegen guarantee (spec 8.3): replaying the emitted
    // input sequence must land the piece on exactly the promised cells.
    const RulesetConfig cfg = league();
    MoveGenerator gen;
    Rng rng(9182736);
    int verified = 0;

    for (int trial = 0; trial < 40; ++trial) {
        Board b(10, 40);
        for (int y = 0; y < 7; ++y)
            for (int x = 0; x < 10; ++x)
                if (rng.chance(1, 3)) b.fill_cell(x, y, false);

        for (int pi = 0; pi < PIECE_COUNT; ++pi) {
            const Piece p = static_cast<Piece>(pi);
            for (const auto& a : gen.generate_for_piece(b, p, cfg, false)) {
                ActivePiece landed;
                if (!replay(b, p, a.canonical_input_sequence, cfg, landed)) continue;

                // The replayed piece must occupy the same cells as promised.
                const ActivePiece promised = a.piece_state();
                Offset want[4], got[4];
                piece_cells(promised, want);
                piece_cells(landed, got);
                std::set<std::pair<int, int>> ws, gs;
                for (int k = 0; k < 4; ++k) {
                    ws.insert({want[k].x, want[k].y});
                    gs.insert({got[k].x, got[k].y});
                }
                CHECK_MSG(ws == gs, "canonical input sequence must reproduce the placement");
                ++verified;
            }
        }
    }
    CHECK_MSG(verified > 500, "expected many verified sequences");
}

TEST(movegen_merges_equivalent_paths) {
    const RulesetConfig cfg = league();
    Board b(10, 40);

    MoveGenerator merged{MoveGenerator::Options{true, true, true, 20000}};
    MoveGenerator raw{MoveGenerator::Options{true, true, false, 20000}};

    const auto m = merged.generate_for_piece(b, Piece::T, cfg, false);
    const auto r = raw.generate_for_piece(b, Piece::T, cfg, false);
    // Without merging there are strictly more (duplicate) landings.
    CHECK(r.size() >= m.size());

    // After merging, no two actions may share an equivalence class.
    std::set<std::tuple<std::uint64_t, int, int, int>> keys;
    for (const auto& a : m) {
        auto key = std::make_tuple(a.resulting_board_hash, a.cleared_lines,
                                   static_cast<int>(a.spin), a.use_hold ? 1 : 0);
        CHECK_MSG(keys.insert(key).second, "merged actions must be unique");
    }
}

TEST(movegen_distinguishes_spins_from_plain_placements) {
    // Same final cells reachable by rotation (spin) and by sliding (no spin)
    // must remain distinguishable, because they send different attacks.
    const RulesetConfig cfg = league();
    // The overhang on row 2 is what makes the pocket a real spin slot.
    Board b = board_from({
        "XXXX......",
        "XXX...XXXX",
        "XXXX.XXXXX",
    });
    MoveGenerator gen;
    const auto acts = gen.generate_for_piece(b, Piece::T, cfg, false);
    bool found_tsd = false;
    for (const auto& a : acts)
        if (a.spin == SpinType::Full && a.cleared_lines == 2) found_tsd = true;
    CHECK_MSG(found_tsd, "a TSD slot must yield a full-spin double placement");
}

TEST(movegen_finds_tuck_placements_under_overhangs) {
    // A genuine tuck: a shelf covers the left half of the board, so the only
    // way under it is to drop beside the shelf and then slide sideways. A
    // column-scan generator would miss these entirely.
    const RulesetConfig cfg = league();
    Board b = board_from({
        "XXXXX.....",  // shelf at row 2
        "..........",  // row 1 (open, under the shelf)
        "..........",  // row 0 (open, under the shelf)
    });
    MoveGenerator gen;

    // The O piece is 2x2, so it fits in the 2-tall pocket under the shelf.
    const auto acts = gen.generate_for_piece(b, Piece::O, cfg, false);
    bool tucked = false;
    for (const auto& a : acts) {
        Offset cells[4];
        piece_cells(a.piece_state(), cells);
        bool under_shelf = true, touches_floor = false;
        for (const auto& c : cells) {
            if (c.x > 4) under_shelf = false;   // not beneath the shelf
            if (c.y >= 2) under_shelf = false;  // not below it
            if (c.y == 0) touches_floor = true;
        }
        if (under_shelf && touches_floor) tucked = true;
    }
    CHECK_MSG(tucked, "movegen should discover tuck placements under an overhang");
}

TEST(movegen_finds_placements_requiring_rotation_after_drop) {
    // A 1-wide well two cells deep: the vertical I must be rotated and then
    // dropped in, which only a path search can find.
    const RulesetConfig cfg = league();
    Board b = board_from({
        "XXXX.XXXXX",
        "XXXX.XXXXX",
        "XXXX.XXXXX",
        "XXXX.XXXXX",
    });
    MoveGenerator gen;
    const auto acts = gen.generate_for_piece(b, Piece::I, cfg, false);
    bool filled_well = false;
    for (const auto& a : acts) {
        Offset cells[4];
        piece_cells(a.piece_state(), cells);
        int in_well = 0;
        for (const auto& c : cells)
            if (c.x == 4 && c.y < 4) ++in_well;
        if (in_well == 4) filled_well = true;
    }
    CHECK_MSG(filled_well, "vertical I must be able to fill a 1-wide 4-deep well");
    // And filling it must clear four lines.
    for (const auto& a : acts) {
        Offset cells[4];
        piece_cells(a.piece_state(), cells);
        int in_well = 0;
        for (const auto& c : cells)
            if (c.x == 4 && c.y < 4) ++in_well;
        if (in_well == 4) CHECK_EQ(a.cleared_lines, 4);
    }
}

TEST(movegen_returns_nothing_when_spawn_is_blocked) {
    const RulesetConfig cfg = league();
    Board b(10, 40);
    // Fill the whole spawn region.
    for (int y = cfg.geometry.visible_height - 1; y < cfg.geometry.visible_height + 4; ++y)
        for (int x = 0; x < 10; ++x) b.fill_cell(x, y, false);
    MoveGenerator gen;
    for (int pi = 0; pi < PIECE_COUNT; ++pi) {
        const auto acts = gen.generate_for_piece(b, static_cast<Piece>(pi), cfg, false);
        CHECK_MSG(acts.empty(), "blocked spawn must yield no legal placements");
    }
}

TEST(movegen_hold_branch_uses_the_held_piece) {
    const RulesetConfig cfg = league();
    Board b(10, 40);
    MoveGenerator gen;
    const auto acts = gen.generate(b, Piece::S, Piece::I, Piece::T, cfg);

    bool saw_plain_s = false, saw_hold_i = false;
    for (const auto& a : acts) {
        if (!a.use_hold) {
            CHECK(a.final_piece == Piece::S);
            saw_plain_s = true;
        } else {
            CHECK(a.final_piece == Piece::I);
            saw_hold_i = true;
            // The hold input must be first in the sequence.
            CHECK(!a.canonical_input_sequence.empty() &&
                  a.canonical_input_sequence.front() == Input::Hold);
        }
    }
    CHECK(saw_plain_s);
    CHECK(saw_hold_i);
}

TEST(movegen_hold_uses_next_piece_when_hold_is_empty) {
    const RulesetConfig cfg = league();
    Board b(10, 40);
    MoveGenerator gen;
    const auto acts = gen.generate(b, Piece::S, Piece::None, Piece::T, cfg);
    bool saw_t = false;
    for (const auto& a : acts)
        if (a.use_hold && a.final_piece == Piece::T) saw_t = true;
    CHECK(saw_t);
}

TEST(movegen_respects_disabled_hold) {
    RulesetConfig cfg = league();
    cfg.randomizer.hold_enabled = false;
    Board b(10, 40);
    MoveGenerator gen;
    for (const auto& a : gen.generate(b, Piece::S, Piece::I, Piece::T, cfg))
        CHECK(!a.use_hold);
}

TEST(movegen_is_deterministic) {
    const RulesetConfig cfg = league();
    Rng rng(5150);
    Board b(10, 40);
    for (int y = 0; y < 6; ++y)
        for (int x = 0; x < 10; ++x)
            if (rng.chance(1, 2)) b.fill_cell(x, y, false);
    MoveGenerator gen;
    const auto a = gen.generate_for_piece(b, Piece::T, cfg, false);
    const auto c = gen.generate_for_piece(b, Piece::T, cfg, false);
    CHECK_EQ(a.size(), c.size());
    for (size_t i = 0; i < a.size() && i < c.size(); ++i) {
        CHECK(a[i].final_x == c[i].final_x);
        CHECK(a[i].final_y == c[i].final_y);
        CHECK(a[i].final_rotation == c[i].final_rotation);
        CHECK(a[i].canonical_input_sequence == c[i].canonical_input_sequence);
    }
}

TEST(movegen_placement_count_is_mirror_invariant_without_180) {
    // Spec 18.3: mirroring the board and the piece must yield exactly the same
    // set of legal placements. This holds unconditionally for SRS+ 90 degree
    // rotations; the 180 table is excluded because TETR.IO defines it
    // asymmetrically (see movegen_mirror_asymmetry_comes_only_from_180).
    RulesetConfig cfg = league();
    cfg.movement.allow_180 = false;
    auto twin = [](Piece p) {
        switch (p) {
            case Piece::J: return Piece::L;
            case Piece::L: return Piece::J;
            case Piece::S: return Piece::Z;
            case Piece::Z: return Piece::S;
            default: return p;
        }
    };
    MoveGenerator gen;
    Rng rng(777);
    for (int trial = 0; trial < 25; ++trial) {
        Board b(10, 40);
        for (int y = 0; y < 7; ++y)
            for (int x = 0; x < 10; ++x)
                if (rng.chance(1, 3)) b.fill_cell(x, y, false);
        const Board mb = b.mirrored();
        for (int pi = 0; pi < PIECE_COUNT; ++pi) {
            const Piece p = static_cast<Piece>(pi);
            const auto a = gen.generate_for_piece(b, p, cfg, false);
            const auto m = gen.generate_for_piece(mb, twin(p), cfg, false);
            CHECK_MSG(a.size() == m.size(),
                      std::string("mirror placement count differs for ") + piece_name(p) + ": " +
                          std::to_string(a.size()) + " vs " + std::to_string(m.size()));

            // Stronger: the mirrored placements must match cell for cell.
            std::set<std::tuple<int, int, int, int>> as, ms;
            for (const auto& act : a) {
                Offset c[4];
                piece_cells(act.piece_state(), c);
                std::vector<std::pair<int, int>> v;
                for (int k = 0; k < 4; ++k) v.push_back({9 - c[k].x, c[k].y});
                std::sort(v.begin(), v.end());
                as.insert({v[0].first * 100 + v[0].second, v[1].first * 100 + v[1].second,
                           v[2].first * 100 + v[2].second, v[3].first * 100 + v[3].second});
            }
            for (const auto& act : m) {
                Offset c[4];
                piece_cells(act.piece_state(), c);
                std::vector<std::pair<int, int>> v;
                for (int k = 0; k < 4; ++k) v.push_back({c[k].x, c[k].y});
                std::sort(v.begin(), v.end());
                ms.insert({v[0].first * 100 + v[0].second, v[1].first * 100 + v[1].second,
                           v[2].first * 100 + v[2].second, v[3].first * 100 + v[3].second});
            }
            CHECK_MSG(as == ms, "mirrored placements must cover the same cells");
        }
    }
}

TEST(movegen_mirror_asymmetry_comes_only_from_180) {
    // Pins the exact scope of the asymmetry so the training pipeline knows when
    // the mirror augmentation of spec 14 is safe: with 180 rotations enabled a
    // small number of positions differ, purely because TETR.IO's 180 kick table
    // is directionally biased. With 180 disabled the generator is perfectly
    // mirror-symmetric.
    auto twin = [](Piece p) {
        switch (p) {
            case Piece::J: return Piece::L;
            case Piece::L: return Piece::J;
            case Piece::S: return Piece::Z;
            case Piece::Z: return Piece::S;
            default: return p;
        }
    };
    auto count_mismatches = [&](bool allow_180) {
        RulesetConfig cfg = league();
        cfg.movement.allow_180 = allow_180;
        MoveGenerator gen;
        Rng rng(777);
        int diff = 0;
        for (int trial = 0; trial < 25; ++trial) {
            Board b(10, 40);
            for (int y = 0; y < 7; ++y)
                for (int x = 0; x < 10; ++x)
                    if (rng.chance(1, 3)) b.fill_cell(x, y, false);
            const Board mb = b.mirrored();
            for (int pi = 0; pi < PIECE_COUNT; ++pi) {
                const Piece p = static_cast<Piece>(pi);
                if (gen.generate_for_piece(b, p, cfg, false).size() !=
                    gen.generate_for_piece(mb, twin(p), cfg, false).size())
                    ++diff;
            }
        }
        return diff;
    };
    CHECK_EQ(count_mismatches(false), 0);
    CHECK_MSG(count_mismatches(true) > 0,
              "180 kicks are expected to introduce mirror asymmetry");
}

TEST(evaluate_placement_reports_line_clears) {
    const RulesetConfig cfg = league();
    // A row missing exactly the four cells an I piece fills.
    Board b = board_from({"XXXXXX...."});
    ActivePiece p;
    p.type = Piece::I;
    p.rot = Rot::N;
    p.x = 6;
    p.y = -2;  // the I's cells sit at row y+2
    // Find the resting position properly instead of guessing.
    p = spawn_piece(Piece::I, cfg);
    p.x = 6;
    p.y -= hard_drop_distance(b, p);
    const PlacementOutcome oc = evaluate_placement(b, p, cfg);
    CHECK_EQ(oc.cleared_lines, 1);
    CHECK(oc.board.empty());
    CHECK(oc.all_clear);
}

TEST(evaluate_placement_detects_garbage_clears) {
    const RulesetConfig cfg = league();
    Board b(10, 40);
    // A garbage row with a single hole at column 3.
    for (int x = 0; x < 10; ++x)
        if (x != 3) b.fill_cell(x, 0, /*garbage=*/true);
    ActivePiece p = spawn_piece(Piece::I, cfg);
    p.rot = Rot::R;  // vertical, occupies one column
    p.x = 3 - shape_of(Piece::I, Rot::R).cells[0].x;
    p.y -= hard_drop_distance(b, p);
    const PlacementOutcome oc = evaluate_placement(b, p, cfg);
    CHECK_EQ(oc.cleared_lines, 1);
    CHECK(oc.cleared_garbage);
}

TEST(movegen_scales_to_a_tall_messy_board) {
    // Performance / robustness guard: the BFS must terminate and stay sane on
    // a high, hole-ridden board.
    const RulesetConfig cfg = league();
    Rng rng(24680);
    Board b(10, 40);
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 10; ++x)
            if (rng.chance(3, 5)) b.fill_cell(x, y, false);
    }
    MoveGenerator gen;
    for (int pi = 0; pi < PIECE_COUNT; ++pi) {
        const auto acts = gen.generate_for_piece(b, static_cast<Piece>(pi), cfg, false);
        for (const auto& a : acts) {
            CHECK(!collides(b, a.piece_state()));
            CHECK(grounded(b, a.piece_state()));
        }
    }
}
