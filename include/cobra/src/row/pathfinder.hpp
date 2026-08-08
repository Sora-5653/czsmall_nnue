#pragma once

#include "../header.hpp"
#include "../gen.hpp"
#include "../ruleset.hpp"
#include "board.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <utility>
#include <vector>

namespace Cobra {

namespace PathFinder {

enum class Input : uint8_t {
    NO_INPUT, SHIFT_LEFT, SHIFT_RIGHT, DAS_LEFT, DAS_RIGHT, ROTATE_CW, ROTATE_CCW, ROTATE_FLIP, SOFT_DROP, HARD_DROP
};

using Inputs = std::vector<Input>;

template <typename RulesT, Piece p>
requires Ruleset<RulesT>
std::vector<std::pair<Move, Inputs>> get_all_input(const Board<>& b, const bool useFinesse,
                                                   const int force = 0) {
    static_assert(p.is_ok());
    constexpr bool checkTspin = (p == Piece::T) && (RulesT::SPINS == Policy::SpinRule::TSPIN);
    constexpr auto spinMul = checkTspin ? SpinType::size : 1;
    constexpr int SPAWN_Y = RulesT::SPAWN_Y;
    // Spell these sizes out locally.  GCC 15 ICEs while substituting the
    // consteval helper calls here when this pathfinder is instantiated from a
    // templated move-list callback, even though the same helpers compile in
    // the move generator.
    constexpr size_t cSize = (p.value == Piece::O) ? 1 :
                              ((p.value == Piece::I || p.value == Piece::S ||
                                p.value == Piece::Z) ? 2 : 4);
    constexpr size_t sSize = (p.value == Piece::O) ? 1 : 4;

    const Gen::SmearedBoard<Board<>, cSize> usable = Gen::usable_map<Board<>, p>(b);

    assert(force >= 0);
    const int threshold = std::min(SPAWN_Y + force + 1, Board<>::H);
    const int spawn = [&] {
        int s = SPAWN_Y;
        for (; s < threshold && !usable[Rotation::NORTH].get(Gen::SPAWN_X, s); ++s);
        return s;
    }();

    if (spawn == threshold)
        return {};

    struct PathNode {
        Input input;
        uint16_t prev;
    };
    struct GhostMove {
        Rotation r;
        SpinType s;
        int8_t x;
        int8_t y;
        uint16_t prev;

        GhostMove(Rotation r, int x, int y, uint16_t i, SpinType s) :
            r(r), s(s), x(static_cast<int8_t>(x)), y(static_cast<int8_t>(y)), prev(i) {}

        static constexpr uint16_t root() { return std::numeric_limits<uint16_t>::max(); }
    };

    std::array<Gen::SmearedBoard<Board<>, sSize>, spinMul> searched{};
    std::deque<GhostMove> leaf;
    std::vector<PathNode> internal;
    internal.reserve(256);
    std::vector<std::pair<Move, Inputs>> result;
    result.reserve(128);
    std::array<std::array<std::array<std::array<bool, Board<>::H>, Board<>::W>,
                            Rotation::size>, spinMul>
        emitted{};

    const auto spin_index = [&](const SpinType s) {
        if constexpr (checkTspin)
            return s;
        else {
            (void)s;
            return static_cast<size_t>(0);
        }
    };

    leaf.push_back({Rotation::NORTH, Gen::SPAWN_X, spawn, GhostMove::root(), SpinType::NONE});
    searched[spin_index(SpinType::NONE)][Rotation::NORTH].set(Gen::SPAWN_X, spawn);

    while (!leaf.empty()) {
        GhostMove m = leaf.front();
        leaf.pop_front();

        auto update = [&]<Input input>(const GhostMove& l) {
            assert(is_ok_x(l.x));
            assert(is_ok_y(l.y));

            auto& entry = searched[spin_index(l.s)][l.r];

            if (!entry.get(l.x, l.y)) {
                entry.set(l.x, l.y);
                internal.push_back({input, l.prev});
                leaf.push_back({l.r, l.x, l.y, static_cast<uint16_t>(internal.size() - 1), l.s});
            }
        };

        // Harddrop
        {
            GhostMove l = m;
            while (is_ok_y(l.y - 1) && usable[Gen::canonical_r<p>(l.r)].get(l.x, l.y - 1)) // No clean sonic drop api yet
                l.y -= 1;

            if constexpr (checkTspin)
                if (l.y != m.y)
                    l.s = SpinType::NONE;

            const Rotation canonical = Gen::canonical_r<p>(l.r);
            const size_t si = spin_index(l.s);
            const size_t ri = static_cast<size_t>(canonical);
            if (!emitted[si][ri][static_cast<size_t>(l.x)][static_cast<size_t>(l.y)]) {
                emitted[si][ri][static_cast<size_t>(l.x)][static_cast<size_t>(l.y)] = true;
                Inputs path;
                for (uint16_t i = l.prev; i != GhostMove::root(); i = internal[i].prev)
                    path.push_back(internal[i].input);
                std::ranges::reverse(path);
                path.push_back(Input::HARD_DROP);
                result.emplace_back(Move{p, canonical, l.x, l.y, l.s}, std::move(path));
            }
        }

        if constexpr (checkTspin)
            m.s = SpinType::NONE;

        // Rotation
        if constexpr (p != Piece::O) {
            constexpr size_t kickIndex = Gen::kick_index<RulesT, p>();
            constexpr size_t kick180Index = Gen::kick180_index<p>();

            m.r.route([&]<Rotation r>{
                auto rotate = [&]<Gen::Direction d, const auto& kickTable>{
                    constexpr Input input = []{
                        switch (d) {
                            case Gen::Direction::CW: return Input::ROTATE_CW;
                            case Gen::Direction::CCW: return Input::ROTATE_CCW;
                            case Gen::Direction::FLIP: return Input::ROTATE_FLIP;
                            default: std::unreachable();
                        }
                    }();
                    constexpr Rotation r1 = Gen::rotate<d>(r);
                    constexpr Rotation r1c = Gen::canonical_r<p>(r1);
                    constexpr auto off = Gen::canonical_offset<p>(r) - Gen::canonical_offset<p>(r1);
                    constexpr size_t kickSize = Gen::kick_size<RulesT, d, kickTable[r].size()>();

                    GhostMove l = m;
                    l.r = r1;

                    const auto obstructed = [&](const int x, const int y) {
                        return !is_ok_x(x) || !is_ok_y(y) || b.get(x, y);
                    };

                    [&]<size_t... i>(std::index_sequence<i...>) {
                        ([&]{
                            constexpr auto kick = kickTable[r][i] + off;
                            l.x = m.x + kick.x;
                            l.y = m.y + kick.y;

                            if (!is_ok_x(l.x) || !is_ok_y(l.y) || !(usable[r1c].get(l.x, l.y)))
                                return true;

                            if constexpr (checkTspin) {
                                const std::array corner = {
                                    obstructed(l.x - 1, l.y + 1),
                                    obstructed(l.x + 1, l.y + 1),
                                    obstructed(l.x + 1, l.y - 1),
                                    obstructed(l.x - 1, l.y - 1)
                                };
                                if (corner[0] + corner[1] + corner[2] + corner[3] >= 3)
                                    l.s = (i >= 4 || (corner[r1] && corner[Gen::rotate<Gen::Direction::CW>(r1)])) ? SpinType::FULL : SpinType::MINI;
                                else
                                    l.s = SpinType::NONE;
                            }

                            update.template operator()<input>(l);
                            return false;
                        }() && ...);
                    }(std::make_index_sequence<kickSize>());
                };
                rotate.template operator()<Gen::Direction::CW, Gen::kicks[kickIndex][Gen::Direction::CW]>();
                rotate.template operator()<Gen::Direction::CCW, Gen::kicks[kickIndex][Gen::Direction::CCW]>();
                if constexpr (RulesT::ENABLE_180)
                    rotate.template operator()<Gen::Direction::FLIP, Gen::kicks180[kick180Index]>();

            });
        }

        // Shift
        {
            auto shift = [&]<int dx, Input input>{
                static_assert(dx == 1 || dx == -1);
                GhostMove l = m;
                l.x += dx;

                if (is_ok_x(l.x) && usable[Gen::canonical_r<p>(l.r)].get(l.x, l.y)) {
                    update.template operator()<input>(l);

                    if (useFinesse) { // DAS
                        const int x = l.x;
                        while (is_ok_x(l.x + dx) && usable[Gen::canonical_r<p>(l.r)].get(l.x + dx, l.y))
                            l.x += dx;

                        if (l.x != x)
                            update.template operator()<dx == 1 ? Input::DAS_RIGHT : Input::DAS_LEFT>(l);
                    }
                }
            };
            shift.template operator()<-1, Input::SHIFT_LEFT>();
            shift.template operator()<1, Input::SHIFT_RIGHT>();
        }

        // Softdrop
        {
            GhostMove l = m;
            l.y -= 1;

            if (is_ok_y(l.y) && usable[Gen::canonical_r<p>(l.r)].get(l.x, l.y))
                update.template operator()<Input::SOFT_DROP>(l);
        }
    }

    return result;
}

template <typename RulesT, Piece p>
requires Ruleset<RulesT>
Inputs get_input(const Board<>& b, const Move& target, const bool useFinesse,
                 const int force = 0) {
    for (auto& [move, inputs] : get_all_input<RulesT, p>(b, useFinesse, force)) {
        if (move.rotation == target.rotation && move.x == target.x && move.y == target.y &&
            move.spin == target.spin)
            return std::move(inputs);
    }
    return {};
}

} // namespace PathFinder

} // namespace Cobra
