// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- M1 tokenizer (spec 9.2, 9.3).
//
// Turns an Observation into the token sequence the Transformer consumes.
// Feeding raw cells would cost 400 tokens per board; instead each board is
// summarised as H row tokens + W column tokens + 1 board summary token, which
// keeps global attention cheap enough for repeated MCTS evaluation while
// preserving the per-row and per-column structure the game is played on.
#pragma once

#include "tetra/movegen.hpp"
#include "tetra/observation.hpp"
#include "tetra/schema.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace tetra {

// Token kinds carry a modality id (spec 15.1) so the model can learn
// modality-specific biases.
enum class TokenKind : std::uint8_t {
    Row = 0,
    Column = 1,
    BoardSummary = 2,
    Active = 3,
    Hold = 4,
    Next = 5,
    Garbage = 6,
    Counters = 7,
    Event = 8,
    Rule = 9,
    Time = 10,
    OpponentRow = 11,
    OpponentColumn = 12,
    OpponentSummary = 13,
    Missing = 14,
    Bag = 15,
    OpponentCounters = 16,
};

inline const char* token_kind_name(TokenKind k) {
    switch (k) {
        case TokenKind::Row: return "row";
        case TokenKind::Column: return "col";
        case TokenKind::BoardSummary: return "board";
        case TokenKind::Active: return "active";
        case TokenKind::Hold: return "hold";
        case TokenKind::Next: return "next";
        case TokenKind::Garbage: return "garbage";
        case TokenKind::Counters: return "counters";
        case TokenKind::Event: return "event";
        case TokenKind::Rule: return "rule";
        case TokenKind::Time: return "time";
        case TokenKind::OpponentRow: return "opp_row";
        case TokenKind::OpponentColumn: return "opp_col";
        case TokenKind::OpponentSummary: return "opp_board";
        case TokenKind::Missing: return "missing";
        case TokenKind::Bag: return "bag";
        case TokenKind::OpponentCounters: return "opp_counters";
    }
    return "?";
}

// Fixed-width feature vector per token. Keeping this a compile-time constant
// lets the encoder be a single dense matmul.
inline constexpr int TOKEN_FEATURES = 24;

struct Token {
    TokenKind kind = TokenKind::Missing;
    int player = 0;      // 0 = self, 1 = opponent
    int index = 0;       // row/column index, queue position, ...
    std::array<float, TOKEN_FEATURES> f{};
};

struct TokenizedObservation {
    std::vector<Token> tokens;
    int self_rows = 0;
    int self_cols = 0;

    size_t size() const { return tokens.size(); }
};

namespace detail {

inline int popcount32(std::uint32_t v) {
    int n = 0;
    while (v) {
        v &= v - 1;
        ++n;
    }
    return n;
}

// Squash an unbounded count into [0,1] so the encoder sees a stable range.
inline float squash(float v, float scale) { return v / (v + scale); }

}  // namespace detail

class Tokenizer {
public:
    struct Options {
        int max_events = 16;      // spec 7.3 keeps 16-64
        bool include_columns = true;
        bool include_rows = true;
        int max_rows = 24;        // rows above this are almost always empty
    };

    Tokenizer() = default;
    explicit Tokenizer(const Options& o) : opt_(o) {}

    TokenizedObservation encode(const Observation& obs, const RulesetConfig& cfg) const {
        TokenizedObservation out;
        out.tokens.reserve(96);

        encode_board(out, obs.board, cfg, /*player=*/0, TokenKind::Row, TokenKind::Column,
                     TokenKind::BoardSummary);
        out.self_rows = std::min(opt_.max_rows, obs.board.height());
        out.self_cols = obs.board.width();

        encode_active(out, obs, cfg);
        encode_hold(out, obs);
        encode_next(out, obs, cfg);
        encode_bag(out, obs, cfg);
        encode_garbage(out, obs, cfg);
        encode_counters(out, obs, cfg);
        encode_rule(out, cfg);
        encode_time(out, obs, cfg);
        encode_events(out, obs, cfg);

        if (obs.has_opponent) {
            encode_board(out, obs.opponent_board, cfg, /*player=*/1, TokenKind::OpponentRow,
                         TokenKind::OpponentColumn, TokenKind::OpponentSummary);
            encode_opponent_counters(out, obs);
        } else {
            // Spec 9.3: an explicit Missing token beats silently zero-padding,
            // because the model can then tell "no opponent" from "empty board".
            Token t;
            t.kind = TokenKind::Missing;
            t.player = 1;
            t.f[0] = 1.0f;
            out.tokens.push_back(t);
        }
        return out;
    }

    const Options& options() const { return opt_; }

private:
    void encode_board(TokenizedObservation& out, const Board& b, const RulesetConfig& cfg,
                      int player, TokenKind row_kind, TokenKind col_kind,
                      TokenKind summary_kind) const {
        const int W = b.width();
        const int H = std::min(opt_.max_rows, b.height());

        // --- row tokens ---
        if (opt_.include_rows) {
            for (int y = 0; y < H; ++y) {
                Token t;
                t.kind = row_kind;
                t.player = player;
                t.index = y;
                const std::uint32_t occ = b.row(y);
                const std::uint32_t gar = b.garbage_row(y);
                const int filled = detail::popcount32(occ);
                t.f[0] = static_cast<float>(filled) / static_cast<float>(W);
                t.f[1] = static_cast<float>(detail::popcount32(gar)) / static_cast<float>(W);
                t.f[2] = (occ == b.full_mask()) ? 1.0f : 0.0f;
                t.f[3] = (occ == 0) ? 1.0f : 0.0f;
                // One-away-from-full is the single most actionable row feature.
                t.f[4] = (filled == W - 1) ? 1.0f : 0.0f;
                t.f[5] = static_cast<float>(y) / static_cast<float>(cfg.geometry.visible_height);
                t.f[6] = (y < cfg.geometry.visible_height) ? 1.0f : 0.0f;
                // Transitions: how ragged the row is.
                int transitions = 0;
                for (int x = 0; x + 1 < W; ++x)
                    if (((occ >> x) & 1u) != ((occ >> (x + 1)) & 1u)) ++transitions;
                t.f[7] = static_cast<float>(transitions) / static_cast<float>(W);
                // Raw low bits let the model recover the exact pattern for the
                // narrow boards used in practice (W <= 16).
                for (int x = 0; x < W && x < 16; ++x)
                    t.f[8 + x] = ((occ >> x) & 1u) ? 1.0f : 0.0f;
                out.tokens.push_back(t);
            }
        }

        // --- column tokens ---
        if (opt_.include_columns) {
            for (int x = 0; x < W; ++x) {
                Token t;
                t.kind = col_kind;
                t.player = player;
                t.index = x;
                const int h = b.column_height(x);
                t.f[0] = static_cast<float>(h) / static_cast<float>(cfg.geometry.visible_height);
                int holes = 0, covered = 0;
                bool seen = false;
                for (int y = b.height() - 1; y >= 0; --y) {
                    const bool filled = b.is_solid(x, y);
                    if (filled) seen = true;
                    else if (seen) ++holes;
                    if (seen && filled) ++covered;
                }
                t.f[1] = detail::squash(static_cast<float>(holes), 4.0f);
                t.f[2] = detail::squash(static_cast<float>(covered), 8.0f);
                // Relative height against the neighbours: wells and towers.
                // The side walls act as infinitely tall neighbours, otherwise a
                // well in column 0 or W-1 (the classic place to keep the I
                // slot) would never register as a well at all.
                const int wall = cfg.geometry.internal_height;
                const int hl = (x > 0) ? b.column_height(x - 1) : wall;
                const int hr = (x + 1 < W) ? b.column_height(x + 1) : wall;
                // The signed differences stay clamped so an edge column does
                // not saturate the feature.
                t.f[3] = std::max(-1.0f, std::min(1.0f, static_cast<float>(h - hl) / 8.0f));
                t.f[4] = std::max(-1.0f, std::min(1.0f, static_cast<float>(h - hr) / 8.0f));
                // A well deep enough for a vertical I is a first-class concept.
                const int depth = std::min(hl, hr) - h;
                t.f[5] = (depth >= 3) ? 1.0f : 0.0f;
                t.f[6] = std::max(-1.0f, std::min(1.0f, static_cast<float>(depth) / 8.0f));
                t.f[7] = static_cast<float>(x) / static_cast<float>(W);
                t.f[8] = (x == 0 || x == W - 1) ? 1.0f : 0.0f;
                out.tokens.push_back(t);
            }
        }

        // --- board summary token ---
        {
            Token t;
            t.kind = summary_kind;
            t.player = player;
            const int height = b.stack_height();
            const int holes = b.hole_count();
            t.f[0] = static_cast<float>(height) / static_cast<float>(cfg.geometry.visible_height);
            t.f[1] = detail::squash(static_cast<float>(holes), 8.0f);
            int bumpiness = 0, maxh = 0, minh = 999;
            for (int x = 0; x < W; ++x) {
                const int h = b.column_height(x);
                maxh = std::max(maxh, h);
                minh = std::min(minh, h);
                if (x + 1 < W) bumpiness += std::abs(h - b.column_height(x + 1));
            }
            t.f[2] = detail::squash(static_cast<float>(bumpiness), 20.0f);
            t.f[3] = static_cast<float>(maxh - (minh == 999 ? 0 : minh)) / 20.0f;
            int garbage_rows = 0;
            for (int y = 0; y < b.height(); ++y)
                if (b.garbage_row(y) != 0u) ++garbage_rows;
            t.f[4] = detail::squash(static_cast<float>(garbage_rows), 8.0f);
            t.f[5] = b.empty() ? 1.0f : 0.0f;
            // Danger: how close the stack is to the ceiling.
            t.f[6] = static_cast<float>(height) /
                     static_cast<float>(std::max(1, cfg.geometry.internal_height));
            out.tokens.push_back(t);
        }
    }

    void encode_active(TokenizedObservation& out, const Observation& obs,
                       const RulesetConfig& cfg) const {
        Token t;
        t.kind = TokenKind::Active;
        if (obs.active.valid()) {
            t.f[static_cast<size_t>(obs.active.type)] = 1.0f;  // one-hot piece (7)
            t.f[7 + static_cast<size_t>(obs.active.rot)] = 1.0f;  // one-hot rotation (4)
            t.f[11] = static_cast<float>(obs.active.x) / static_cast<float>(cfg.geometry.width);
            t.f[12] = static_cast<float>(obs.active.y) /
                      static_cast<float>(cfg.geometry.visible_height);
            t.f[13] = (obs.active.last_action == LastAction::Rotate) ? 1.0f : 0.0f;
            t.f[14] = static_cast<float>(obs.active.last_kick) / 5.0f;
        } else {
            t.f[15] = 1.0f;  // missing flag
        }
        out.tokens.push_back(t);
    }

    void encode_hold(TokenizedObservation& out, const Observation& obs) const {
        Token t;
        t.kind = TokenKind::Hold;
        if (obs.hold != Piece::None) t.f[static_cast<size_t>(obs.hold)] = 1.0f;
        else t.f[15] = 1.0f;  // hold is empty
        t.f[7] = obs.hold_used ? 1.0f : 0.0f;
        out.tokens.push_back(t);
    }

    void encode_next(TokenizedObservation& out, const Observation& obs,
                     const RulesetConfig& cfg) const {
        // Exactly preview_count tokens: a shorter queue is padded with an
        // explicit missing flag rather than a fake piece.
        for (int i = 0; i < cfg.randomizer.preview_count; ++i) {
            Token t;
            t.kind = TokenKind::Next;
            t.index = i;
            if (i < static_cast<int>(obs.next.size()))
                t.f[static_cast<size_t>(obs.next[static_cast<size_t>(i)])] = 1.0f;
            else
                t.f[15] = 1.0f;
            t.f[8] = static_cast<float>(i) /
                     static_cast<float>(std::max(1, cfg.randomizer.preview_count));
            out.tokens.push_back(t);
        }
    }

    void encode_bag(TokenizedObservation& out, const Observation& obs,
                    const RulesetConfig& cfg) const {
        Token t;
        t.kind = TokenKind::Bag;
        t.index = -1;

        // The remaining contents are public state for bag randomizers. For
        // memoryless/debug randomizers, emit an explicit missing flag instead
        // of making an empty vector look like an exhausted bag.
        const bool is_bag = cfg.randomizer.type == RandomizerType::Bag7 ||
                            cfg.randomizer.type == RandomizerType::Bag14;
        if (!is_bag) {
            t.f[15] = 1.0f;
            out.tokens.push_back(t);
            return;
        }

        std::array<int, PIECE_COUNT> counts{};
        for (Piece piece : obs.bag_remaining) {
            const int index = static_cast<int>(piece);
            if (index >= 0 && index < PIECE_COUNT) ++counts[static_cast<size_t>(index)];
        }
        const int max_bag = cfg.randomizer.type == RandomizerType::Bag14 ? 14 : 7;
        for (int i = 0; i < PIECE_COUNT; ++i)
            // Bag7 has at most one copy and Bag14 has at most two, so this
            // preserves the exact public count without leaving [-4, 4].
            t.f[static_cast<size_t>(i)] =
                static_cast<float>(counts[static_cast<size_t>(i)]) /
                static_cast<float>(cfg.randomizer.type == RandomizerType::Bag14 ? 2 : 1);
        t.f[7] = static_cast<float>(obs.bag_remaining.size()) /
                 static_cast<float>(max_bag);
        t.f[8] = obs.bag_remaining.empty() ? 1.0f : 0.0f;
        int distinct = 0;
        for (int count : counts)
            if (count > 0) ++distinct;
        t.f[9] = static_cast<float>(distinct) / static_cast<float>(PIECE_COUNT);
        out.tokens.push_back(t);
    }

    void encode_garbage(TokenizedObservation& out, const Observation& obs,
                        const RulesetConfig& cfg) const {
        // One token summarising the whole queue, plus up to four group tokens.
        {
            Token t;
            t.kind = TokenKind::Garbage;
            t.index = -1;
            t.f[0] = detail::squash(static_cast<float>(obs.pending_lines), 6.0f);
            t.f[1] = detail::squash(static_cast<float>(obs.active_garbage_lines), 6.0f);
            t.f[2] = (obs.pending_lines > 0) ? 1.0f : 0.0f;
            t.f[3] = static_cast<float>(std::min(obs.pending_lines, cfg.garbage.cap > 0
                                                                        ? cfg.garbage.cap
                                                                        : obs.pending_lines)) /
                     8.0f;
            out.tokens.push_back(t);
        }
        int emitted = 0;
        for (const auto& g : obs.pending_garbage) {
            if (emitted >= 4) break;
            Token t;
            t.kind = TokenKind::Garbage;
            t.index = emitted;
            t.f[0] = detail::squash(static_cast<float>(g.lines), 4.0f);
            // Timing is what makes deliberate non-cancelling learnable.
            t.f[1] = detail::squash(static_cast<float>(std::max<Tick>(0, g.ticks_until_arrival)),
                                    30.0f);
            t.f[2] = detail::squash(
                static_cast<float>(std::max<Tick>(0, g.ticks_until_activation)), 30.0f);
            t.f[3] = (g.ticks_until_activation <= 0) ? 1.0f : 0.0f;
            t.f[4] = g.cancellable ? 1.0f : 0.0f;
            out.tokens.push_back(t);
            ++emitted;
        }
    }

    void encode_counters(TokenizedObservation& out, const Observation& obs,
                         const RulesetConfig& cfg) const {
        Token t;
        t.kind = TokenKind::Counters;
        t.f[0] = detail::squash(static_cast<float>(std::max(0, obs.combo)), 4.0f);
        t.f[1] = (obs.combo >= 0) ? 1.0f : 0.0f;
        t.f[2] = detail::squash(static_cast<float>(obs.b2b_streak), 4.0f);
        t.f[3] = (obs.b2b_streak > 0) ? 1.0f : 0.0f;
        t.f[4] = detail::squash(static_cast<float>(obs.surge), 8.0f);
        t.f[5] = (obs.surge > 0) ? 1.0f : 0.0f;
        t.f[6] = (obs.b2b_streak >= cfg.attack.surge_start_streak) ? 1.0f : 0.0f;
        t.f[7] = detail::squash(static_cast<float>(obs.pieces_placed), 50.0f);
        // Opener phase changes cancellation maths, so the model must see it.
        t.f[8] = (cfg.attack.opener_phase_enabled &&
                  obs.pieces_placed <= cfg.attack.opener_phase_pieces)
                     ? 1.0f
                     : 0.0f;
        out.tokens.push_back(t);
    }

    void encode_opponent_counters(TokenizedObservation& out, const Observation& obs) const {
        Token t;
        t.kind = TokenKind::OpponentCounters;
        t.player = 1;
        t.index = -1;
        t.f[0] = detail::squash(static_cast<float>(std::max(0, obs.opponent_pending_lines)), 6.0f);
        t.f[1] = (obs.opponent_pending_lines > 0) ? 1.0f : 0.0f;
        t.f[2] = detail::squash(static_cast<float>(std::max(0, obs.opponent_combo)), 4.0f);
        t.f[3] = (obs.opponent_combo >= 0) ? 1.0f : 0.0f;
        t.f[4] = detail::squash(static_cast<float>(std::max(0, obs.opponent_b2b)), 4.0f);
        t.f[5] = (obs.opponent_b2b > 0) ? 1.0f : 0.0f;
        t.f[6] = obs.opponent_alive ? 1.0f : 0.0f;
        t.f[7] = 1.0f;  // this token is present only when an opponent exists
        out.tokens.push_back(t);
    }

    void encode_rule(TokenizedObservation& out, const RulesetConfig& cfg) const {
        // Spec 9.3 / 14: a rule token plus ruleset randomisation is what lets
        // one model generalise across patches and custom rooms.
        Token t;
        t.kind = TokenKind::Rule;
        t.f[0] = static_cast<float>(cfg.geometry.width) / 10.0f;
        t.f[1] = static_cast<float>(cfg.geometry.visible_height) / 20.0f;
        t.f[2] = static_cast<float>(cfg.randomizer.preview_count) / 5.0f;
        t.f[3] = cfg.randomizer.hold_enabled ? 1.0f : 0.0f;
        t.f[4] = cfg.movement.allow_180 ? 1.0f : 0.0f;
        t.f[5] = static_cast<float>(static_cast<int>(cfg.movement.kick_table)) / 3.0f;
        t.f[6] = static_cast<float>(static_cast<int>(cfg.clear_rules.spin_detection)) / 3.0f;
        t.f[7] = static_cast<float>(static_cast<int>(cfg.attack.b2b_mode)) / 2.0f;
        t.f[8] = static_cast<float>(cfg.attack.quad) / 4.0f;
        t.f[9] = static_cast<float>(cfg.attack.all_clear) / 10.0f;
        t.f[10] = static_cast<float>(cfg.attack.surge_base) / 4.0f;
        t.f[11] = (cfg.attack.rounding_mode == RoundingMode::Rng) ? 1.0f : 0.0f;
        t.f[12] = detail::squash(static_cast<float>(cfg.garbage.travel_time), 30.0f);
        t.f[13] = detail::squash(static_cast<float>(cfg.garbage.activation_delay), 30.0f);
        t.f[14] = static_cast<float>(cfg.garbage.cap) / 8.0f;
        t.f[15] = static_cast<float>(cfg.garbage.messiness_percent) / 100.0f;
        t.f[16] = cfg.garbage.passthrough ? 1.0f : 0.0f;
        t.f[17] = cfg.attack.opener_phase_enabled ? 1.0f : 0.0f;
        out.tokens.push_back(t);
    }

    void encode_time(TokenizedObservation& out, const Observation& obs,
                     const RulesetConfig& cfg) const {
        Token t;
        t.kind = TokenKind::Time;
        const double seconds =
            static_cast<double>(obs.timestamp) / static_cast<double>(std::max(1, cfg.tick_rate));
        t.f[0] = detail::squash(static_cast<float>(seconds), 60.0f);
        t.f[1] = static_cast<float>(std::sin(seconds * 0.1));
        t.f[2] = static_cast<float>(std::cos(seconds * 0.1));
        t.f[3] = obs.alive ? 1.0f : 0.0f;
        out.tokens.push_back(t);
    }

    void encode_events(TokenizedObservation& out, const Observation& obs,
                       const RulesetConfig& cfg) const {
        // Most recent events first, capped at max_events.
        const int n = static_cast<int>(obs.recent_events.size());
        const int start = std::max(0, n - opt_.max_events);
        for (int i = n - 1; i >= start; --i) {
            const Event& e = obs.recent_events[static_cast<size_t>(i)];
            Token t;
            t.kind = TokenKind::Event;
            t.index = n - 1 - i;  // 0 == most recent
            t.player = e.actor;
            t.f[static_cast<size_t>(e.type)] = 1.0f;  // one-hot event type (9)
            t.f[9] = detail::squash(static_cast<float>(e.lines), 4.0f);
            t.f[10] = static_cast<float>(static_cast<int>(e.spin)) / 2.0f;
            t.f[11] = detail::squash(
                static_cast<float>(std::max<Tick>(0, obs.timestamp - e.timestamp)),
                static_cast<float>(cfg.tick_rate));
            t.f[12] = detail::squash(static_cast<float>(e.duration), 30.0f);
            if (e.piece != Piece::None) t.f[13] = static_cast<float>(e.piece) / 6.0f;
            out.tokens.push_back(t);
        }
    }

    Options opt_{};
};

// ---------------------------------------------------------------------------
// Action embedding (spec 10.1)
// ---------------------------------------------------------------------------
// Each legal placement becomes a query vector that cross-attends to the state
// tokens, which is what makes the policy head variable-length.
inline constexpr int ACTION_FEATURES = 24;

struct ActionEmbedding {
    std::array<float, ACTION_FEATURES> f{};
};

inline ActionEmbedding embed_action(const PlacementAction& a, const Board& before,
                                    const RulesetConfig& cfg) {
    ActionEmbedding e;
    e.f[static_cast<size_t>(a.final_piece)] = 1.0f;                 // 0..6 piece
    e.f[7 + static_cast<size_t>(a.final_rotation)] = 1.0f;          // 7..10 rotation
    e.f[11] = a.use_hold ? 1.0f : 0.0f;
    e.f[12] = static_cast<float>(a.final_x) / static_cast<float>(cfg.geometry.width);
    e.f[13] = static_cast<float>(a.final_y) / static_cast<float>(cfg.geometry.visible_height);
    e.f[14] = static_cast<float>(a.cleared_lines) / 4.0f;
    e.f[15] = (a.cleared_lines == 4) ? 1.0f : 0.0f;
    e.f[16] = static_cast<float>(static_cast<int>(a.spin)) / 2.0f;
    e.f[17] = a.cleared_garbage ? 1.0f : 0.0f;
    e.f[18] = a.all_clear ? 1.0f : 0.0f;
    e.f[19] = static_cast<float>(a.last_kick) / 5.0f;
    e.f[20] = static_cast<float>(a.canonical_input_sequence.size()) / 12.0f;
    e.f[21] = static_cast<float>(a.delay_bin) / 5.0f;

    // Resulting surface summary: the cheapest signal that separates a good
    // placement from a hole-making one.
    const PlacementOutcome oc = evaluate_placement(before, a.piece_state(), cfg);
    const int holes_before = before.hole_count();
    const int holes_after = oc.board.hole_count();
    e.f[22] = detail::squash(static_cast<float>(std::max(0, holes_after - holes_before)), 3.0f);
    e.f[23] = static_cast<float>(oc.board.stack_height()) /
              static_cast<float>(cfg.geometry.visible_height);
    return e;
}

}  // namespace tetra
