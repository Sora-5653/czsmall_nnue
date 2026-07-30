// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- M0 rule core: playfield representation.
#pragma once

#include "tetra/types.hpp"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace tetra {

// Playfield stored as one bitmask row per line. Row 0 is the BOTTOM row.
// Bit i of a row mask is column i, counted from the left (x = 0 .. width-1).
//
// Two parallel planes are kept (spec 7.1 PlayerState.field):
//   occupancy : any filled cell
//   garbage   : cells that came from received garbage lines
// The garbage plane is always a subset of the occupancy plane.
class Board {
public:
    static constexpr int MAX_WIDTH  = 32;
    static constexpr int MAX_HEIGHT = 64;

    Board() = default;
    Board(int width, int height) { reset(width, height); }

    void reset(int width, int height) {
        assert(width > 0 && width <= MAX_WIDTH);
        assert(height > 0 && height <= MAX_HEIGHT);
        width_ = width;
        height_ = height;
        full_mask_ = (width == 32) ? 0xFFFFFFFFu : ((1u << width) - 1u);
        occ_.assign(static_cast<size_t>(height), 0u);
        gar_.assign(static_cast<size_t>(height), 0u);
    }

    void clear() {
        std::fill(occ_.begin(), occ_.end(), 0u);
        std::fill(gar_.begin(), gar_.end(), 0u);
    }

    int width() const { return width_; }
    int height() const { return height_; }
    std::uint32_t full_mask() const { return full_mask_; }

    std::uint32_t row(int y) const {
        return (y >= 0 && y < height_) ? occ_[static_cast<size_t>(y)] : 0u;
    }
    std::uint32_t garbage_row(int y) const {
        return (y >= 0 && y < height_) ? gar_[static_cast<size_t>(y)] : 0u;
    }

    void set_row(int y, std::uint32_t mask) {
        assert(y >= 0 && y < height_);
        occ_[static_cast<size_t>(y)] = mask & full_mask_;
        gar_[static_cast<size_t>(y)] &= occ_[static_cast<size_t>(y)];
    }
    void set_garbage_row(int y, std::uint32_t occ_mask, std::uint32_t gar_mask) {
        assert(y >= 0 && y < height_);
        occ_[static_cast<size_t>(y)] = occ_mask & full_mask_;
        gar_[static_cast<size_t>(y)] = gar_mask & occ_[static_cast<size_t>(y)];
    }

    // Out-of-bounds below the floor and to the sides counts as solid, so that
    // collision tests need no separate bounds checks. Above the ceiling is
    // empty (pieces may stick out of the top of the internal field).
    bool is_solid(int x, int y) const {
        if (x < 0 || x >= width_) return true;
        if (y < 0) return true;
        if (y >= height_) return false;
        return (occ_[static_cast<size_t>(y)] >> x) & 1u;
    }

    bool is_garbage_cell(int x, int y) const {
        if (x < 0 || x >= width_ || y < 0 || y >= height_) return false;
        return (gar_[static_cast<size_t>(y)] >> x) & 1u;
    }

    void fill_cell(int x, int y, bool garbage = false) {
        assert(x >= 0 && x < width_);
        if (y < 0 || y >= height_) return;  // silently drop; caller checks top-out
        occ_[static_cast<size_t>(y)] |= (1u << x);
        if (garbage) gar_[static_cast<size_t>(y)] |= (1u << x);
    }

    bool row_full(int y) const { return row(y) == full_mask_; }
    bool row_empty(int y) const { return row(y) == 0u; }

    bool empty() const {
        for (int y = 0; y < height_; ++y)
            if (occ_[static_cast<size_t>(y)] != 0u) return false;
        return true;
    }

    // Height of the highest occupied row + 1 (0 when the board is empty).
    int stack_height() const {
        for (int y = height_ - 1; y >= 0; --y)
            if (occ_[static_cast<size_t>(y)] != 0u) return y + 1;
        return 0;
    }

    // Height of a single column: 1 + index of its topmost filled cell.
    int column_height(int x) const {
        for (int y = height_ - 1; y >= 0; --y)
            if ((occ_[static_cast<size_t>(y)] >> x) & 1u) return y + 1;
        return 0;
    }

    // Empty cells that have at least one filled cell somewhere above them.
    int hole_count() const {
        int holes = 0;
        for (int x = 0; x < width_; ++x) {
            bool seen = false;
            for (int y = height_ - 1; y >= 0; --y) {
                bool filled = (occ_[static_cast<size_t>(y)] >> x) & 1u;
                if (filled) seen = true;
                else if (seen) ++holes;
            }
        }
        return holes;
    }

    // Remove every full row, shifting the rows above down. Returns the row
    // indices (bottom-up, pre-clear coordinates) that were removed.
    std::vector<int> clear_full_rows() {
        std::vector<int> cleared;
        int write = 0;
        for (int y = 0; y < height_; ++y) {
            if (occ_[static_cast<size_t>(y)] == full_mask_) {
                cleared.push_back(y);
                continue;
            }
            if (write != y) {
                occ_[static_cast<size_t>(write)] = occ_[static_cast<size_t>(y)];
                gar_[static_cast<size_t>(write)] = gar_[static_cast<size_t>(y)];
            }
            ++write;
        }
        for (; write < height_; ++write) {
            occ_[static_cast<size_t>(write)] = 0u;
            gar_[static_cast<size_t>(write)] = 0u;
        }
        return cleared;
    }

    // Insert `count` garbage rows at the bottom. Each entry of `holes` is the
    // hole column for the corresponding inserted line (bottom-most first).
    // Returns true if the push-up did NOT overflow the internal ceiling.
    bool insert_garbage_bottom(const std::vector<int>& holes) {
        const int count = static_cast<int>(holes.size());
        if (count <= 0) return true;
        // Anything that would be pushed past the ceiling means a garbage-out.
        const int h = stack_height();
        const bool ok = (h + count) <= height_;
        for (int y = height_ - 1; y >= count; --y) {
            occ_[static_cast<size_t>(y)] = occ_[static_cast<size_t>(y - count)];
            gar_[static_cast<size_t>(y)] = gar_[static_cast<size_t>(y - count)];
        }
        for (int i = 0; i < count && i < height_; ++i) {
            const int hole = holes[static_cast<size_t>(i)];
            std::uint32_t mask = full_mask_;
            if (hole >= 0 && hole < width_) mask &= ~(1u << hole);
            occ_[static_cast<size_t>(i)] = mask;
            gar_[static_cast<size_t>(i)] = mask;
        }
        return ok;
    }

    bool operator==(const Board& o) const {
        return width_ == o.width_ && height_ == o.height_ && occ_ == o.occ_ && gar_ == o.gar_;
    }
    bool operator!=(const Board& o) const { return !(*this == o); }

    // Mirror the field left-to-right (spec 14: data augmentation, and 18.3
    // property tests).
    Board mirrored() const {
        Board b(width_, height_);
        for (int y = 0; y < height_; ++y) {
            std::uint32_t o = 0, g = 0;
            for (int x = 0; x < width_; ++x) {
                if ((occ_[static_cast<size_t>(y)] >> x) & 1u) o |= 1u << (width_ - 1 - x);
                if ((gar_[static_cast<size_t>(y)] >> x) & 1u) g |= 1u << (width_ - 1 - x);
            }
            b.occ_[static_cast<size_t>(y)] = o;
            b.gar_[static_cast<size_t>(y)] = g;
        }
        return b;
    }

    // Debug rendering: bottom row printed last. '.' empty, 'X' block, 'G' garbage.
    std::string to_string(int max_rows = 24) const {
        std::string s;
        int top = stack_height();
        if (top < 1) top = 1;
        if (top > max_rows) top = max_rows;
        for (int y = top - 1; y >= 0; --y) {
            for (int x = 0; x < width_; ++x) {
                if (is_garbage_cell(x, y)) s += 'G';
                else if (is_solid(x, y)) s += 'X';
                else s += '.';
            }
            s += '\n';
        }
        return s;
    }

    const std::vector<std::uint32_t>& rows() const { return occ_; }
    const std::vector<std::uint32_t>& garbage_rows() const { return gar_; }

private:
    int width_ = 0;
    int height_ = 0;
    std::uint32_t full_mask_ = 0;
    std::vector<std::uint32_t> occ_;
    std::vector<std::uint32_t> gar_;
};

}  // namespace tetra
