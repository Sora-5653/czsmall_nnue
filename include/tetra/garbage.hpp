// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- pending garbage queue (spec 7.2).
#pragma once

#include "tetra/ruleset.hpp"
#include "tetra/rng.hpp"
#include "tetra/types.hpp"

#include <cstdint>
#include <vector>

namespace tetra {

// One group of garbage lines in flight or waiting in the queue.
struct GarbageEntry {
    int lines = 0;
    Tick sent_at = 0;
    Tick arrival_at = 0;     // becomes cancellable/visible in the queue
    Tick activation_at = 0;  // may rise into the field from this tick on
    bool cancellable = true;
    bool tankable = true;
    int hole_column = -1;    // hidden from the observation until it rises
    int source_player = -1;
    std::uint32_t rule_metadata = 0;

    bool arrived(Tick now) const { return now >= arrival_at; }
    bool active(Tick now) const { return now >= activation_at; }
};

// FIFO of pending garbage with cancellation and activation logic.
class GarbageQueue {
public:
    void reset() { entries_.clear(); }

    void push(const GarbageEntry& e) { entries_.push_back(e); }

    // Total lines still pending (regardless of activation state).
    int total_lines() const {
        int n = 0;
        for (const auto& e : entries_) n += e.lines;
        return n;
    }

    // Lines that have arrived and are eligible to rise at `now`.
    int active_lines(Tick now) const {
        int n = 0;
        for (const auto& e : entries_)
            if (e.active(now)) n += e.lines;
        return n;
    }

    // Lines that have arrived in the queue (visible to the player) at `now`.
    int arrived_lines(Tick now) const {
        int n = 0;
        for (const auto& e : entries_)
            if (e.arrived(now)) n += e.lines;
        return n;
    }

    // Cancel up to `amount` lines from the front of the queue.
    // Returns the number of lines actually cancelled.
    //
    // Spec 6 / TETR.IO: garbage still in flight (before arrival_at) cannot be
    // cancelled when passthrough is enabled; with passthrough off, everything
    // pending is cancellable.
    int cancel(int amount, Tick now, const GarbageCfg& cfg) {
        if (amount <= 0 || cfg.blocking_mode == GarbageBlocking::None) return 0;
        int cancelled = 0;
        size_t i = 0;
        while (i < entries_.size() && amount > 0) {
            GarbageEntry& e = entries_[i];
            const bool eligible = e.cancellable && cfg.cancellable &&
                                  (!cfg.passthrough || e.arrived(now));
            if (!eligible) { ++i; continue; }
            const int take = std::min(amount, e.lines);
            e.lines -= take;
            amount -= take;
            cancelled += take;
            if (e.lines == 0) entries_.erase(entries_.begin() + static_cast<long>(i));
            else ++i;
        }
        return cancelled;
    }

    // Pop up to `cap` active lines for insertion into the field, expanding them
    // into per-line hole columns according to the messiness rules.
    std::vector<int> take_active(Tick now, const RulesetConfig& cfg, Rng& rng) {
        std::vector<int> holes;
        const GarbageCfg& g = cfg.garbage;
        int budget = (g.cap > 0) ? g.cap : INT32_MAX;

        while (!entries_.empty() && budget > 0) {
            GarbageEntry& e = entries_.front();
            if (!e.active(now)) break;
            const int take = std::min(budget, e.lines);

            int col = e.hole_column;
            if (col < 0) col = static_cast<int>(rng.below(static_cast<std::uint32_t>(cfg.geometry.width)));

            for (int i = 0; i < take; ++i) {
                if (g.hole_change_rule == GarbageHoleRule::PerLine) {
                    col = static_cast<int>(rng.below(static_cast<std::uint32_t>(cfg.geometry.width)));
                } else if (g.messiness_percent > 0 && i > 0) {
                    if (rng.chance(g.messiness_percent, 100))
                        col = static_cast<int>(rng.below(static_cast<std::uint32_t>(cfg.geometry.width)));
                }
                holes.push_back(col);
            }

            e.lines -= take;
            budget -= take;
            if (e.lines == 0) entries_.erase(entries_.begin());
            else e.hole_column = col;
        }
        return holes;
    }

    // Earliest tick at which any queued garbage becomes active.
    Tick next_activation(Tick now) const {
        Tick best = TICK_NEVER;
        for (const auto& e : entries_) {
            if (e.lines <= 0) continue;
            if (e.activation_at > now && e.activation_at < best) best = e.activation_at;
        }
        return best;
    }

    const std::vector<GarbageEntry>& entries() const { return entries_; }
    bool empty() const { return entries_.empty(); }

private:
    std::vector<GarbageEntry> entries_;
};

}  // namespace tetra