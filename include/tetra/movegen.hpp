// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- Cobra-backed legal placement generation.
//
// Cobra is the sole movement/placement backend.  The public Tetra action
// representation remains here so the simulator, evaluator, replay, and
// timing layers do not need to know about Cobra's coordinate types.
#pragma once

#include "tetra/movegen_types.hpp"

#include <algorithm>
#include <vector>

namespace tetra {

class MoveGenerator {
public:
    struct Options {
        bool include_hold = true;
        bool allow_180 = true;
        bool merge_equivalent = true;
        // Retained for source compatibility. Cobra bounds its own fixed-size
        // search and does not expose a project-side state limit.
        int max_states = 20000;
        bool enforce_gravity = true;
        Tick gravity_check_threshold = 8;
    };

    MoveGenerator() = default;
    explicit MoveGenerator(const Options& options) : options_(options) {}

    std::vector<PlacementAction> generate_for_piece(const Board& board, Piece piece,
                                                    const RulesetConfig& cfg,
                                                    bool use_hold,
                                                    bool allow_clutch = false) const;

    std::vector<PlacementAction> generate(const Board& board, Piece current, Piece hold_piece,
                                          Piece next_after_hold,
                                          const RulesetConfig& cfg,
                                          bool allow_clutch = false) const {
        std::vector<PlacementAction> out =
            generate_for_piece(board, current, cfg, false, allow_clutch);
        if (!options_.include_hold || !cfg.randomizer.hold_enabled) return out;

        const Piece swapped = (hold_piece != Piece::None) ? hold_piece : next_after_hold;
        if (swapped == Piece::None || swapped == current) return out;

        std::vector<PlacementAction> held =
            generate_for_piece(board, swapped, cfg, true, allow_clutch);
        out.insert(out.end(), held.begin(), held.end());
        return out;
    }

    static std::vector<PlacementAction> expand_delay_bins(
        const std::vector<PlacementAction>& actions, const RulesetConfig& cfg, Tick now,
        Tick next_garbage_activation = TICK_NEVER, Tick opponent_next_lock = TICK_NEVER,
        const std::vector<DelayBin>& bins = default_delay_bins()) {
        std::vector<PlacementAction> out;
        out.reserve(actions.size() * bins.size());

        const HandlingModel h = HandlingModel::from(cfg);
        for (const auto& base : actions) {
            std::vector<Tick> seen_waits;
            for (const DelayBin bin : bins) {
                Tick wait = delay_bin_ticks(bin);
                if (bin == DelayBin::WaitForEvent) {
                    // WAIT_FOR_EVENT is an *extra* delay on top of the base
                    // execution time. Target the placement's lock at the next
                    // event rather than waiting until the event and only then
                    // beginning the movement. If the fastest execution already
                    // reaches/passes the event, there is no distinct wait action.
                    const Tick event_offset = resolve_wait_for_event(
                        now, next_garbage_activation, opponent_next_lock, h);
                    wait = std::max<Tick>(0, event_offset - base.base_duration);
                    if (wait <= 0) continue;
                }
                if (std::find(seen_waits.begin(), seen_waits.end(), wait) != seen_waits.end())
                    continue;
                seen_waits.push_back(wait);

                PlacementAction a = base;
                a.delay_bin = bin;
                a.delay_ticks = wait;
                out.push_back(std::move(a));
            }
        }
        return out;
    }

    const Options& options() const { return options_; }

private:
    Options options_{};
};

}  // namespace tetra
