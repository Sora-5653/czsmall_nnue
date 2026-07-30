// SPDX-License-Identifier: MIT
// Board / bitboard behaviour (spec 18.3 property tests).
#include "test_util.hpp"
#include "tetra/bitboard.hpp"
#include "tetra/rng.hpp"

using namespace tetra;

namespace {

Board make_board(std::initializer_list<const char*> rows_top_first, int width = 10,
                 int height = 40) {
    // Rows are given visually, top row first, like the debug renderer prints.
    Board b(width, height);
    const int n = static_cast<int>(rows_top_first.size());
    int y = n - 1;
    for (const char* row : rows_top_first) {
        for (int x = 0; x < width; ++x) {
            if (row[x] == 'X') b.fill_cell(x, y, false);
            else if (row[x] == 'G') b.fill_cell(x, y, true);
        }
        --y;
    }
    return b;
}

}  // namespace

TEST(board_starts_empty) {
    Board b(10, 40);
    CHECK(b.empty());
    CHECK_EQ(b.stack_height(), 0);
    CHECK_EQ(b.hole_count(), 0);
    for (int x = 0; x < 10; ++x) CHECK_EQ(b.column_height(x), 0);
}

TEST(board_out_of_bounds_is_solid) {
    Board b(10, 40);
    CHECK(b.is_solid(-1, 0));
    CHECK(b.is_solid(10, 0));
    CHECK(b.is_solid(0, -1));
    // Above the internal ceiling is empty, not solid: pieces may stick out.
    CHECK(!b.is_solid(0, 40));
    CHECK(!b.is_solid(0, 100));
}

TEST(board_clear_full_rows_shifts_down) {
    Board b = make_board({
        "X.........",
        "XXXXXXXXXX",
        "..........",
        "XXXXXXXXXX",
    });
    CHECK_EQ(b.stack_height(), 4);
    const std::vector<int> cleared = b.clear_full_rows();
    CHECK_EQ(static_cast<int>(cleared.size()), 2);
    // Bottom-up indices of the cleared rows.
    CHECK_EQ(cleared[0], 0);
    CHECK_EQ(cleared[1], 2);
    // What is left: the empty row, then the single block on top.
    CHECK_EQ(b.stack_height(), 2);
    CHECK_EQ(b.row(0), 0u);
    CHECK_EQ(b.row(1), 1u);
}

TEST(board_garbage_plane_is_subset_of_occupancy) {
    Board b(10, 40);
    b.fill_cell(3, 0, /*garbage=*/true);
    CHECK(b.is_solid(3, 0));
    CHECK(b.is_garbage_cell(3, 0));
    b.fill_cell(4, 0, /*garbage=*/false);
    CHECK(b.is_solid(4, 0));
    CHECK(!b.is_garbage_cell(4, 0));
}

TEST(board_insert_garbage_pushes_stack_up) {
    Board b = make_board({"X........."});
    CHECK_EQ(b.stack_height(), 1);
    const bool ok = b.insert_garbage_bottom({4, 4});
    CHECK(ok);
    CHECK_EQ(b.stack_height(), 3);
    // The original block moved up two rows.
    CHECK(b.is_solid(0, 2));
    // The garbage rows are full except for the hole column.
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 10; ++x) {
            if (x == 4) CHECK(!b.is_solid(x, y));
            else CHECK_MSG(b.is_solid(x, y), "garbage row should be filled");
        }
        CHECK(b.is_garbage_cell(0, y));
    }
}

TEST(board_insert_garbage_reports_overflow) {
    Board b(10, 8);
    for (int y = 0; y < 7; ++y) b.fill_cell(0, y, false);
    CHECK_EQ(b.stack_height(), 7);
    CHECK(b.insert_garbage_bottom({3}));       // 7 + 1 == 8, still fits
    CHECK(!b.insert_garbage_bottom({3, 3}));   // would overflow
}

TEST(board_hole_count) {
    Board b = make_board({
        "XXXXXXXXXX",
        "X.XXXXXXXX",
    });
    // One covered empty cell in column 1.
    CHECK_EQ(b.hole_count(), 1);
}

TEST(board_mirror_is_an_involution) {
    Rng rng(12345);
    for (int trial = 0; trial < 50; ++trial) {
        Board b(10, 20);
        for (int y = 0; y < 10; ++y)
            for (int x = 0; x < 10; ++x)
                if (rng.chance(1, 3)) b.fill_cell(x, y, rng.chance(1, 2));
        const Board m = b.mirrored();
        CHECK(m.mirrored() == b);
        // Mirroring preserves the aggregate statistics used for evaluation.
        CHECK_EQ(m.stack_height(), b.stack_height());
        CHECK_EQ(m.hole_count(), b.hole_count());
        CHECK_EQ(m.column_height(0), b.column_height(9));
    }
}

TEST(board_all_clear_detection) {
    Board b = make_board({"XXXXXXXXXX"});
    b.clear_full_rows();
    CHECK(b.empty());
}
