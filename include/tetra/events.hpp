// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- event log (spec 7.3).
#pragma once

#include "tetra/types.hpp"

#include <cstdint>
#include <deque>

namespace tetra {

enum class EventType : std::uint8_t {
    PieceSpawn = 0,
    PieceLock,
    LineClear,
    AttackSent,
    AttackCancelled,
    GarbageArrived,
    GarbageRaised,
    Topout,
    Hold,
};

inline const char* event_name(EventType t) {
    switch (t) {
        case EventType::PieceSpawn: return "piece_spawn";
        case EventType::PieceLock: return "piece_lock";
        case EventType::LineClear: return "line_clear";
        case EventType::AttackSent: return "attack_sent";
        case EventType::AttackCancelled: return "attack_cancelled";
        case EventType::GarbageArrived: return "garbage_arrived";
        case EventType::GarbageRaised: return "garbage_raised";
        case EventType::Topout: return "topout";
        case EventType::Hold: return "hold";
    }
    return "?";
}

struct Event {
    Tick timestamp = 0;
    Tick timestamp_delta = 0;  // ticks since the previous event of this player
    int actor = 0;             // player index
    EventType type = EventType::PieceSpawn;
    Piece piece = Piece::None;
    SpinType spin = SpinType::None;
    int lines = 0;
    Tick duration = 0;
};

// Bounded history buffer: spec 7.3 keeps the most recent 16-64 events.
class EventLog {
public:
    explicit EventLog(size_t capacity = 64) : capacity_(capacity) {}

    void reset() {
        events_.clear();
        last_ts_ = 0;
    }

    void push(Event e) {
        e.timestamp_delta = e.timestamp - last_ts_;
        last_ts_ = e.timestamp;
        events_.push_back(e);
        while (events_.size() > capacity_) events_.pop_front();
    }

    const std::deque<Event>& events() const { return events_; }
    size_t size() const { return events_.size(); }
    void set_capacity(size_t c) { capacity_ = c; }

private:
    size_t capacity_;
    std::deque<Event> events_;
    Tick last_ts_ = 0;
};

}  // namespace tetra