// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- single-player simulation (spec 7.1 PlayerState).
#pragma once

#include "tetra/attack.hpp"
#include "tetra/bitboard.hpp"
#include "tetra/events.hpp"
#include "tetra/garbage.hpp"
#include "tetra/piece_state.hpp"
#include "tetra/pieces.hpp"
#include "tetra/rng.hpp"
#include "tetra/ruleset.hpp"
#include "tetra/types.hpp"

#include <memory>
#include <vector>

namespace tetra {

// Result of committing one placement.
struct LockResult {
    bool ok = false;                 // false if the placement was illegal
    ClearDescriptor clear{};
    AttackResult attack{};
    int garbage_cancelled = 0;
    int garbage_received = 0;        // lines that rose into the field
    bool topped_out = false;
    TopoutReason topout_reason = TopoutReason::None;
    Tick duration = 0;               // ticks consumed by this placement
};

// A single player's board and all of its per-player state. The simulator owns
// the RNG; observations are produced by a separate masking layer so hidden
// state can never leak (spec 3.2, 18.3).
class Player {
public:
    Player() = default;

    void reset(const RulesetConfig& cfg, std::uint64_t seed, int index) {
        cfg_ = cfg;
        index_ = index;
        board_.reset(cfg.geometry.width, cfg.geometry.internal_height);
        queue_.reset(cfg.randomizer, seed);
        garbage_rng_.reseed(seed ^ 0xA5A5A5A5A5A5A5A5ull);
        attack_rng_.reseed(seed ^ 0x5A5A5A5A5A5A5A5Aull);
        garbage_.reset();
        events_.reset();
        state_.reset();
        hold_ = Piece::None;
        hold_used_ = false;
        alive_ = true;
        topout_ = TopoutReason::None;
        now_ = 0;
        lines_sent_ = 0;
        lines_received_ = 0;
        lines_cleared_ = 0;
        spawn_next();
    }

    // --- accessors ---------------------------------------------------------
    const Board& board() const { return board_; }
    Board& mutable_board() { return board_; }
    const ActivePiece& active() const { return active_; }
    Piece hold() const { return hold_; }
    bool hold_used() const { return hold_used_; }
    bool alive() const { return alive_; }
    TopoutReason topout_reason() const { return topout_; }
    Tick now() const { return now_; }
    void set_now(Tick t) { now_ = t; }
    const AttackState& attack_state() const { return state_; }
    AttackState& mutable_attack_state() { return state_; }
    const GarbageQueue& garbage() const { return garbage_; }
    GarbageQueue& mutable_garbage() { return garbage_; }
    const EventLog& events() const { return events_; }
    const PieceQueue& queue() const { return queue_; }
    PieceQueue& mutable_queue() { return queue_; }
    const RulesetConfig& ruleset() const { return cfg_; }
    int index() const { return index_; }
    std::int64_t lines_sent() const { return lines_sent_; }
    std::int64_t lines_received() const { return lines_received_; }
    std::int64_t lines_cleared() const { return lines_cleared_; }
    std::vector<Piece> visible_next() const { return queue_.visible_next(); }

    // --- hold --------------------------------------------------------------
    // Swap the active piece with hold. Returns false if hold is unavailable.
    bool do_hold() {
        if (!cfg_.randomizer.hold_enabled || hold_used_ || !active_.valid()) return false;
        const Piece cur = active_.type;
        if (hold_ == Piece::None) {
            hold_ = cur;
            if (!spawn_next()) return true;  // top-out recorded inside
        } else {
            const Piece swap = hold_;
            hold_ = cur;
            if (!spawn_specific(swap)) return true;
        }
        hold_used_ = true;
        log(EventType::Hold, cur, SpinType::None, 0, 0);
        return true;
    }

    // --- placement ---------------------------------------------------------
    // Commit the active piece at its current position: lock, clear lines,
    // compute attack, cancel pending garbage, then let the remainder rise.
    //
    // `duration` is the number of ticks this placement consumed, which the
    // caller (movegen / controller) supplies. `outgoing` receives the attack
    // that should be delivered to opponents.
    LockResult lock_piece(Tick duration, int* outgoing_lines) {
        LockResult r;
        if (!alive_ || !active_.valid()) return r;

        // Reject illegal placements outright: a caller (movegen, replay, or a
        // search that applied a stale action) must never be able to merge a
        // piece that overlaps the stack. `ok` stays false so the mistake is
        // visible instead of silently corrupting the field.
        if (collides(board_, active_)) {
            r.ok = false;
            return r;
        }

        // Detect the spin BEFORE the piece is merged into the field.
        const SpinType spin = detect_spin(board_, active_, cfg_);

        Offset cells[4];
        piece_cells(active_, cells);

        // Lock out: the whole piece finished above the visible ceiling.
        bool any_visible = false;
        for (const auto& c : cells)
            if (c.y < cfg_.geometry.visible_height) any_visible = true;

        for (const auto& c : cells) {
            if (c.y >= cfg_.geometry.internal_height) {
                // Piece sticking out of the internal field: immediate top-out.
                die(TopoutReason::LockOut);
                r.topped_out = true;
                r.topout_reason = TopoutReason::LockOut;
                return r;
            }
            board_.fill_cell(c.x, c.y, /*garbage=*/false);
        }

        now_ += duration;
        r.duration = duration;
        log(EventType::PieceLock, active_.type, spin, 0, duration);

        // --- line clears ---
        int garbage_rows_cleared = 0;
        std::vector<int> cleared_rows;
        for (int y = 0; y < cfg_.geometry.internal_height; ++y) {
            if (board_.row_full(y)) {
                cleared_rows.push_back(y);
                if (board_.garbage_row(y) != 0u) ++garbage_rows_cleared;
            }
        }
        if (!cleared_rows.empty()) board_.clear_full_rows();

        ClearDescriptor clear;
        clear.lines = static_cast<int>(cleared_rows.size());
        clear.spin = spin;
        clear.piece = active_.type;
        clear.cleared_garbage = garbage_rows_cleared > 0;
        clear.all_clear = clear.lines > 0 && board_.empty();
        r.clear = clear;
        lines_cleared_ += clear.lines;

        if (clear.lines > 0) {
            now_ += cfg_.clear_rules.line_clear_delay;
            r.duration += cfg_.clear_rules.line_clear_delay;
            log(EventType::LineClear, clear.piece, spin, clear.lines, 0);
        }

        if (!any_visible && clear.lines == 0) {
            die(TopoutReason::LockOut);
            r.topped_out = true;
            r.topout_reason = TopoutReason::LockOut;
            return r;
        }

        // --- attack ---
        Rng* rng = (cfg_.attack.rounding_mode == RoundingMode::Rng) ? &attack_rng_ : nullptr;
        r.attack = compute_attack(clear, state_, cfg_, rng);
        ++state_.pieces_placed;

        int outgoing = r.attack.lines;

        // --- cancellation (spec 12: cancelling is a consequence of timing,
        // never a hardcoded strategy) ---
        if (outgoing > 0 && cfg_.garbage.blocking_mode != GarbageBlocking::None) {
            int budget = outgoing;
            if (cfg_.attack.opener_phase_enabled &&
                state_.pieces_placed <= cfg_.attack.opener_phase_pieces &&
                outgoing < garbage_.total_lines()) {
                budget = outgoing * 2;  // opener phase cancels at double rate
            }
            const int cancelled = garbage_.cancel(budget, now_, cfg_.garbage);
            r.garbage_cancelled = cancelled;
            if (cancelled > 0) log(EventType::AttackCancelled, Piece::None, SpinType::None, cancelled, 0);
            // Cancelled lines are consumed from the outgoing attack.
            const int consumed = std::min(outgoing, cancelled);
            outgoing -= consumed;
        }

        if (outgoing > 0) {
            lines_sent_ += outgoing;
            log(EventType::AttackSent, clear.piece, spin, outgoing, 0);
        }
        if (outgoing_lines) *outgoing_lines = outgoing;

        // --- receive garbage ---
        // Garbage only rises when the placement produced no line clear, which
        // is what makes "eat the garbage on purpose" a real decision.
        if (clear.lines == 0) {
            const std::vector<int> holes = garbage_.take_active(now_, cfg_, garbage_rng_);
            if (!holes.empty()) {
                const bool ok = board_.insert_garbage_bottom(holes);
                r.garbage_received = static_cast<int>(holes.size());
                lines_received_ += r.garbage_received;
                log(EventType::GarbageRaised, Piece::None, SpinType::None, r.garbage_received, 0);
                if (!ok) {
                    die(TopoutReason::GarbageOut);
                    r.topped_out = true;
                    r.topout_reason = TopoutReason::GarbageOut;
                    return r;
                }
            }
        }

        // --- next piece ---
        hold_used_ = false;
        if (!spawn_next()) {
            r.topped_out = true;
            r.topout_reason = topout_;
            return r;
        }

        r.ok = true;
        return r;
    }

    // Receive an attack from another player: enqueue it with travel and
    // activation delays taken from the ruleset.
    void receive_attack(int lines, Tick sent_at, int source) {
        if (lines <= 0 || !alive_) return;
        GarbageEntry e;
        e.lines = lines;
        e.sent_at = sent_at;
        e.arrival_at = sent_at + cfg_.garbage.travel_time;
        e.activation_at = e.arrival_at + cfg_.garbage.activation_delay;
        e.source_player = source;
        if (cfg_.garbage.hole_change_rule != GarbageHoleRule::PerLine) {
            e.hole_column =
                static_cast<int>(garbage_rng_.below(static_cast<std::uint32_t>(cfg_.geometry.width)));
            if (mirror_board_ && cfg_.geometry.width > 0) {
                e.hole_column = cfg_.geometry.width - 1 - e.hole_column;
            }
        }
        garbage_.push(e);
        log(EventType::GarbageArrived, Piece::None, SpinType::None, lines, 0);
    }

    void set_mirror(bool m) {
        mirror_board_ = m;
        queue_.set_mirror(m);
    }

    // Resample hidden state so a search cannot read the future (spec 11.3).
    //
    // Called on a *copy* of the player at the search root: the visible preview
    // and the bag are preserved, everything beyond them is redrawn. The garbage
    // RNG is reseeded too, so undetermined hole columns are sampled rather than
    // known in advance.
    void determinize(std::uint64_t seed) {
        queue_.resample_hidden(seed);
        garbage_rng_.reseed(seed ^ 0xD1CED1CED1CED1CEull);
    }

    // Directly set the active piece (used by movegen when applying an action).
    void set_active(const ActivePiece& p) { active_ = p; }

    void die(TopoutReason reason) {
        if (!alive_) return;
        alive_ = false;
        topout_ = reason;
        log(EventType::Topout, Piece::None, SpinType::None, 0, 0);
    }

private:
    bool spawn_next() { return spawn_specific(queue_.pop()); }

    bool spawn_specific(Piece p) {
        active_ = spawn_piece(p, cfg_);
        if (collides(board_, active_)) {
            // Guideline block-out: try nudging up once (TETR.IO's clutch spawn).
            ActivePiece up = active_;
            up.y += 1;
            if (cfg_.movement.spawn_above_stack && !collides(board_, up)) {
                active_ = up;
            } else {
                die(TopoutReason::BlockOut);
                return false;
            }
        }
        log(EventType::PieceSpawn, active_.type, SpinType::None, 0, 0);
        return true;
    }

    void log(EventType t, Piece p, SpinType s, int lines, Tick dur) {
        Event e;
        e.timestamp = now_;
        e.actor = index_;
        e.type = t;
        e.piece = p;
        e.spin = s;
        e.lines = lines;
        e.duration = dur;
        events_.push(e);
    }

    RulesetConfig cfg_{};
    int index_ = 0;
    Board board_;
    ActivePiece active_{};
    Piece hold_ = Piece::None;
    bool hold_used_ = false;
    PieceQueue queue_;
    GarbageQueue garbage_;
    AttackState state_{};
    EventLog events_{64};
    Rng garbage_rng_{0};
    Rng attack_rng_{0};
    bool alive_ = true;
    TopoutReason topout_ = TopoutReason::None;
    Tick now_ = 0;
    std::int64_t lines_sent_ = 0;
    std::int64_t lines_received_ = 0;
    std::int64_t lines_cleared_ = 0;
    bool mirror_board_ = false;
};

}  // namespace tetra