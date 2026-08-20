// SPDX-License-Identifier: MIT
// Human TETR.IO replay import adapter.
//
// Python owns the unstable .ttrm JSON surface and emits a small line protocol.
// C++ remains authoritative for board geometry, rotations, legal move generation,
// tokenization, action embeddings, and .tetradat serialization.
#pragma once

#include "tetra/batch.hpp"
#include "tetra/dataset.hpp"
#include "tetra/movegen.hpp"
#include "tetra/observation.hpp"
#include "tetra/piece_state.hpp"
#include "tetra/replay_buffer.hpp"
#include "tetra/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tetra {

struct HumanReplayState {
    Board board;
    Piece current = Piece::None;
    Piece hold = Piece::None;
    std::vector<Piece> queue;
    int combo = -1;
    int b2b = 0;
    bool valid = false;

    HumanReplayState() : board(Board::MAX_WIDTH, Board::MAX_HEIGHT) {}
};

struct HumanReplayTurn {
    Tick frame = 0;
    int player = 0;
    std::vector<std::string> keys;

    // Exact-v19 path. Legacy TURN records leave these at defaults and retain
    // the historical key re-execution adapter for backwards compatibility.
    std::int64_t order_time10 = 0;
    bool exact = false;
    HumanReplayState exact_before;
    HumanReplayState exact_after;
    Board exact_placement_board;
    bool exact_used_hold = false;
    Piece exact_final_piece = Piece::None;
    int exact_final_x = 0;   // diagnostic only; Python uses centre coordinates
    int exact_final_y = 0;   // diagnostic only; board match is authoritative
    Rot exact_final_rotation = Rot::N;
};

struct HumanReplayGame {
    std::string source_id;
    int round_index = 0;
    std::array<float, 2> outcome{0.0f, 0.0f};
    std::array<HumanReplayState, 2> initial;
    std::vector<HumanReplayTurn> turns;
};

struct HumanReplayImportStats {
    std::size_t games = 0;
    std::size_t turns = 0;
    std::size_t imported = 0;
    std::size_t skipped_invalid_state = 0;
    std::size_t skipped_execution = 0;
    std::size_t skipped_no_legal_match = 0;
    std::vector<std::string> unmatched_examples;
};

namespace human_detail {

inline std::vector<std::string> split(std::string_view text, char delimiter) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t at = text.find(delimiter, start);
        const std::size_t end = at == std::string_view::npos ? text.size() : at;
        out.emplace_back(text.substr(start, end - start));
        if (at == std::string_view::npos) break;
        start = at + 1;
    }
    return out;
}

inline Piece parse_piece(std::string_view text) {
    if (text.empty() || text == "-" || text == "none") return Piece::None;
    Piece piece = Piece::None;
    const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(text.front())));
    if (!piece_from_char(c, piece)) return Piece::None;
    return piece;
}

inline std::vector<Piece> parse_queue(std::string_view text) {
    std::vector<Piece> out;
    if (text == "-" || text.empty()) return out;
    for (const char raw : text) {
        if (raw == ',' || raw == ' ' || raw == ';') continue;
        Piece piece = Piece::None;
        const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(raw)));
        if (piece_from_char(c, piece)) out.push_back(piece);
    }
    return out;
}

inline bool parse_rows(std::string_view text, Board& board, const RulesetConfig& rules) {
    Board parsed(rules.geometry.width, rules.geometry.internal_height);
    const std::vector<std::string> rows = split(text, ',');
    if (rows.empty() || rows.size() > static_cast<std::size_t>(rules.geometry.internal_height))
        return false;
    for (std::size_t y = 0; y < rows.size(); ++y) {
        if (rows[y].empty()) return false;
        char* end = nullptr;
        const unsigned long mask = std::strtoul(rows[y].c_str(), &end, 10);
        if (!end || *end != '\0') return false;
        for (int x = 0; x < rules.geometry.width; ++x) {
            if ((mask & (1ul << x)) != 0ul)
                parsed.fill_cell(x, static_cast<int>(y), false);
        }
    }
    board = std::move(parsed);
    return true;
}

inline bool parse_rows_with_garbage(std::string_view occupied_text,
                                    std::string_view garbage_text,
                                    Board& board,
                                    const RulesetConfig& rules) {
    Board occupied;
    Board garbage;
    if (!parse_rows(occupied_text, occupied, rules) ||
        !parse_rows(garbage_text, garbage, rules))
        return false;
    for (int y = 0; y < rules.geometry.internal_height; ++y) {
        const std::uint32_t occ = occupied.row(y);
        const std::uint32_t gar = garbage.row(y);
        if ((gar & ~occ) != 0u) return false;
        occupied.set_garbage_row(y, occ, gar);
    }
    board = std::move(occupied);
    return true;
}

inline bool same_board(const Board& a, const Board& b) {
    if (a.width() != b.width() || a.height() != b.height()) return false;
    for (int y = 0; y < a.height(); ++y) {
        if (a.row(y) != b.row(y) || a.garbage_row(y) != b.garbage_row(y))
            return false;
    }
    return true;
}

inline std::uint64_t stable_source_hash(const std::string& source_id, int round_index) {
    std::uint64_t hash = 1469598103934665603ull;
    for (unsigned char c : source_id) {
        hash ^= c;
        hash *= 1099511628211ull;
    }
    for (int i = 0; i < 4; ++i) {
        hash ^= static_cast<unsigned char>((static_cast<unsigned int>(round_index) >> (i * 8)) & 0xffu);
        hash *= 1099511628211ull;
    }
    return hash;
}

inline bool same_placement(const PlacementAction& action, const ActivePiece& piece,
                           bool used_hold, const PlacementOutcome& executed,
                           const Board& before, const RulesetConfig& rules) {
    if (action.use_hold != used_hold || action.final_piece != piece.type) return false;
    if (action.final_x == piece.x && action.final_y == piece.y &&
        action.final_rotation == piece.rot)
        return true;
    const PlacementOutcome candidate = evaluate_placement(before, action.piece_state(), rules);
    if (candidate.board_hash != executed.board_hash) return false;
    for (int y = 0; y < candidate.board.height(); ++y)
        if (candidate.board.row(y) != executed.board.row(y)) return false;
    return true;
}

struct TurnExecution {
    bool ok = false;
    bool used_hold = false;
    Piece hold_after = Piece::None;
    std::size_t queue_consumed_for_hold = 0;
    ActivePiece piece;
    PlacementOutcome outcome;
};

inline TurnExecution execute_turn(const HumanReplayState& state,
                                  const HumanReplayTurn& turn,
                                  const RulesetConfig& rules) {
    TurnExecution result;
    if (!state.valid || state.current == Piece::None) return result;

    Piece moving = state.current;
    Piece hold_after = state.hold;
    ActivePiece active = spawn_piece(moving, rules);
    bool used_hold = false;
    bool locked = false;

    auto sonic_drop = [&]() {
        while (true) {
            ActivePiece next = active;
            next.y -= 1;
            if (collides(state.board, next)) break;
            next.last_action = LastAction::Drop;
            next.last_kick = 0;
            active = next;
        }
    };

    for (const std::string& key : turn.keys) {
        if (key == "hold") {
            if (used_hold) continue;
            used_hold = true;
            const Piece old_current = moving;
            if (state.hold != Piece::None) {
                moving = state.hold;
                hold_after = old_current;
            } else {
                if (state.queue.empty()) return result;
                moving = state.queue.front();
                hold_after = old_current;
                result.queue_consumed_for_hold = 1;
            }
            active = spawn_piece(moving, rules);
            continue;
        }
        if (key == "moveLeft" || key == "moveRight") {
            ActivePiece next = active;
            next.x += key == "moveLeft" ? -1 : 1;
            if (!collides(state.board, next)) {
                next.last_action = LastAction::Move;
                next.last_kick = 0;
                active = next;
            }
            continue;
        }
        if (key == "rotateCW") {
            (void)try_rotate(state.board, active, rot_cw(active.rot), rules);
            continue;
        }
        if (key == "rotateCCW") {
            (void)try_rotate(state.board, active, rot_ccw(active.rot), rules);
            continue;
        }
        if (key == "rotate180") {
            (void)try_rotate(state.board, active, rot_180(active.rot), rules);
            continue;
        }
        if (key == "softDrop") {
            // A replay keydown records the control activation rather than every
            // gravity tick. Treat it as a sonic drop; later rotations/taps still
            // allow floor kicks and tucks before the hard drop.
            sonic_drop();
            continue;
        }
        if (key == "hardDrop") {
            sonic_drop();
            locked = true;
            break;
        }
    }

    if (!locked || collides(state.board, active)) return result;
    result.ok = true;
    result.used_hold = used_hold;
    result.hold_after = hold_after;
    result.piece = active;
    result.outcome = evaluate_placement(state.board, active, rules);
    return result;
}

inline Observation observation_from_state(const HumanReplayState& state,
                                          const HumanReplayState& opponent,
                                          Tick frame,
                                          const RulesetConfig& rules) {
    Observation obs;
    obs.ruleset_hash = rules.hash();
    obs.ruleset = rules;
    obs.timestamp = frame;
    obs.board = state.board;
    obs.active = spawn_piece(state.current, rules);
    obs.hold = state.hold;
    obs.hold_used = false;
    const std::size_t preview = static_cast<std::size_t>(std::max(0, rules.randomizer.preview_count));
    for (std::size_t i = 0; i < std::min(preview, state.queue.size()); ++i)
        obs.next.push_back(state.queue[i]);
    obs.combo = state.combo;
    obs.b2b_streak = state.b2b;
    obs.alive = state.valid;
    obs.has_opponent = opponent.valid;
    if (opponent.valid) {
        obs.opponent_board = opponent.board;
        obs.opponent_combo = opponent.combo;
        obs.opponent_b2b = opponent.b2b;
        obs.opponent_alive = true;
    }
    return obs;
}

inline void advance_state(HumanReplayState& state, const TurnExecution& execution) {
    state.board = execution.outcome.board;
    state.hold = execution.hold_after;

    std::size_t consume = execution.queue_consumed_for_hold;
    if (consume > state.queue.size()) {
        state.current = Piece::None;
        state.queue.clear();
        state.valid = false;
        return;
    }
    if (consume > 0)
        state.queue.erase(state.queue.begin(), state.queue.begin() + static_cast<std::ptrdiff_t>(consume));

    if (!state.queue.empty()) {
        state.current = state.queue.front();
        state.queue.erase(state.queue.begin());
    } else {
        state.current = Piece::None;
    }

    if (execution.outcome.cleared_lines > 0) {
        state.combo = state.combo < 0 ? 0 : state.combo + 1;
        const bool difficult = execution.outcome.cleared_lines == 4 || execution.outcome.spin != SpinType::None;
        if (difficult)
            ++state.b2b;
        else
            state.b2b = 0;
    } else {
        state.combo = -1;
    }
}

inline bool parse_state_fields(const std::vector<std::string>& fields,
                               std::size_t base,
                               HumanReplayState& state,
                               const RulesetConfig& rules) {
    // board, current, hold, queue, combo, b2b
    if (fields.size() < base + 6) return false;
    if (!parse_rows(fields[base], state.board, rules)) return false;
    state.current = parse_piece(fields[base + 1]);
    state.hold = parse_piece(fields[base + 2]);
    state.queue = parse_queue(fields[base + 3]);
    try {
        state.combo = std::stoi(fields[base + 4]);
        state.b2b = std::stoi(fields[base + 5]);
    } catch (...) {
        return false;
    }
    state.valid = state.current != Piece::None;
    return state.valid;
}

inline bool parse_game_header(const std::vector<std::string>& fields, HumanReplayGame& game) {
    if (fields.size() != 5 || fields[0] != "GAME") return false;
    game.source_id = fields[1];
    try {
        game.round_index = std::stoi(fields[2]);
        game.outcome[0] = std::stof(fields[3]);
        game.outcome[1] = std::stof(fields[4]);
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace human_detail

inline bool read_human_replay_protocol(const std::string& path,
                                       const RulesetConfig& rules,
                                       std::vector<HumanReplayGame>& games,
                                       std::string* error = nullptr) {
    std::ifstream in(path);
    if (!in) {
        if (error) *error = "cannot open normalized replay protocol";
        return false;
    }

    HumanReplayGame current;
    bool in_game = false;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(in, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> fields = human_detail::split(line, '\t');
        if (fields.empty()) continue;
        if (fields[0] == "GAME") {
            if (in_game) {
                if (error) *error = "nested GAME at line " + std::to_string(line_number);
                return false;
            }
            current = HumanReplayGame{};
            if (!human_detail::parse_game_header(fields, current)) {
                if (error) *error = "invalid GAME at line " + std::to_string(line_number);
                return false;
            }
            in_game = true;
            continue;
        }
        if (!in_game) {
            if (error) *error = "record outside GAME at line " + std::to_string(line_number);
            return false;
        }
        if (fields[0] == "INIT") {
            if (fields.size() != 8) {
                if (error) *error = "invalid INIT width at line " + std::to_string(line_number);
                return false;
            }
            int player = -1;
            try { player = std::stoi(fields[1]); } catch (...) { player = -1; }
            if (player < 0 || player > 1 ||
                !human_detail::parse_state_fields(fields, 2, current.initial[static_cast<std::size_t>(player)], rules)) {
                if (error) *error = "invalid INIT at line " + std::to_string(line_number);
                return false;
            }
            continue;
        }
        if (fields[0] == "TURN") {
            if (fields.size() != 4) {
                if (error) *error = "invalid TURN width at line " + std::to_string(line_number);
                return false;
            }
            HumanReplayTurn turn;
            try {
                turn.frame = static_cast<Tick>(std::stoll(fields[1]));
                turn.player = std::stoi(fields[2]);
                turn.order_time10 = static_cast<std::int64_t>(turn.frame) * 10;
            } catch (...) {
                if (error) *error = "invalid TURN numbers at line " + std::to_string(line_number);
                return false;
            }
            if (turn.player < 0 || turn.player > 1) {
                if (error) *error = "invalid TURN player at line " + std::to_string(line_number);
                return false;
            }
            turn.keys = human_detail::split(fields[3], ',');
            current.turns.push_back(std::move(turn));
            continue;
        }
        if (fields[0] == "XTURN") {
            if (fields.size() != 22) {
                if (error) *error = "invalid XTURN width at line " + std::to_string(line_number);
                return false;
            }
            HumanReplayTurn turn;
            turn.exact = true;
            try {
                turn.order_time10 = std::stoll(fields[1]);
                turn.frame = static_cast<Tick>(std::stoll(fields[2]));
                turn.player = std::stoi(fields[3]);
                turn.exact_before.combo = std::stoi(fields[9]);
                turn.exact_before.b2b = std::stoi(fields[10]);
                turn.exact_used_hold = std::stoi(fields[11]) != 0;
                turn.exact_final_x = std::stoi(fields[13]);
                turn.exact_final_y = std::stoi(fields[14]);
                const int rotation = std::stoi(fields[15]);
                if (rotation < 0 || rotation > 3) throw std::out_of_range("rotation");
                turn.exact_final_rotation = static_cast<Rot>(rotation);
                turn.exact_after.combo = std::stoi(fields[20]);
                turn.exact_after.b2b = std::stoi(fields[21]);
            } catch (...) {
                if (error) *error = "invalid XTURN numbers at line " + std::to_string(line_number);
                return false;
            }
            if (turn.player < 0 || turn.player > 1 ||
                !human_detail::parse_rows_with_garbage(fields[4], fields[5], turn.exact_before.board, rules) ||
                !human_detail::parse_rows_with_garbage(fields[16], fields[17], turn.exact_placement_board, rules) ||
                !human_detail::parse_rows_with_garbage(fields[18], fields[19], turn.exact_after.board, rules)) {
                if (error) *error = "invalid XTURN board/player at line " + std::to_string(line_number);
                return false;
            }
            turn.exact_before.current = human_detail::parse_piece(fields[6]);
            turn.exact_before.hold = human_detail::parse_piece(fields[7]);
            turn.exact_before.queue = human_detail::parse_queue(fields[8]);
            turn.exact_final_piece = human_detail::parse_piece(fields[12]);
            turn.exact_before.valid = turn.exact_before.current != Piece::None;
            turn.exact_after.valid = true;  // only board/combo/B2B are used as opponent context
            if (!turn.exact_before.valid || turn.exact_final_piece == Piece::None) {
                if (error) *error = "invalid XTURN pieces at line " + std::to_string(line_number);
                return false;
            }
            current.turns.push_back(std::move(turn));
            continue;
        }
        if (fields[0] == "END") {
            if (!current.initial[0].valid || !current.initial[1].valid) {
                if (error) *error = "GAME missing INIT before END at line " + std::to_string(line_number);
                return false;
            }
            std::stable_sort(current.turns.begin(), current.turns.end(),
                             [](const HumanReplayTurn& a, const HumanReplayTurn& b) {
                                 const std::int64_t at = a.exact ? a.order_time10
                                                                  : static_cast<std::int64_t>(a.frame) * 10;
                                 const std::int64_t bt = b.exact ? b.order_time10
                                                                  : static_cast<std::int64_t>(b.frame) * 10;
                                 if (at != bt) return at < bt;
                                 return a.player < b.player;
                             });
            games.push_back(std::move(current));
            current = HumanReplayGame{};
            in_game = false;
            continue;
        }
        if (error) *error = "unknown record at line " + std::to_string(line_number);
        return false;
    }
    if (in_game) {
        if (error) *error = "unterminated GAME";
        return false;
    }
    return true;
}

inline HumanReplayImportStats build_human_replay_samples(
    const std::vector<HumanReplayGame>& games,
    const RulesetConfig& rules,
    std::uint32_t model_version,
    std::vector<TrainingSample>& samples) {
    HumanReplayImportStats stats;
    Tokenizer tokenizer;
    MoveGenerator movegen;

    for (const HumanReplayGame& game : games) {
        ++stats.games;
        std::array<HumanReplayState, 2> state = game.initial;
        std::array<std::uint32_t, 2> move_number{0, 0};
        const std::uint64_t source_hash = human_detail::stable_source_hash(game.source_id, game.round_index);

        for (const HumanReplayTurn& turn : game.turns) {
            ++stats.turns;
            const int player = turn.player;
            const int opponent = 1 - player;

            if (turn.exact) {
                const HumanReplayState& active_state = turn.exact_before;
                if (!active_state.valid || active_state.current == Piece::None) {
                    ++stats.skipped_invalid_state;
                    state[static_cast<std::size_t>(player)] = turn.exact_after;
                    continue;
                }

                const Observation obs = human_detail::observation_from_state(
                    active_state, state[static_cast<std::size_t>(opponent)], turn.frame, rules);
                const Piece next_after_hold = active_state.queue.empty()
                                                  ? Piece::None
                                                  : active_state.queue.front();
                const std::vector<PlacementAction> actions = movegen.generate(
                    active_state.board, active_state.current, active_state.hold,
                    next_after_hold, rules, active_state.combo >= 0);
                int chosen = -1;
                Tick best_duration = TICK_NEVER;
                for (std::size_t i = 0; i < actions.size(); ++i) {
                    const PlacementAction& action = actions[i];
                    const bool same_piece_hold_is_canonical_no_hold =
                        turn.exact_used_hold && !action.use_hold &&
                        active_state.hold == active_state.current &&
                        action.final_piece == active_state.current;
                    if ((!same_piece_hold_is_canonical_no_hold &&
                         action.use_hold != turn.exact_used_hold) ||
                        action.final_piece != turn.exact_final_piece)
                        continue;
                    const PlacementOutcome candidate =
                        evaluate_placement(active_state.board, action.piece_state(), rules);
                    if (!human_detail::same_board(candidate.board, turn.exact_placement_board))
                        continue;
                    if (chosen < 0 || action.base_duration < best_duration) {
                        chosen = static_cast<int>(i);
                        best_duration = action.base_duration;
                    }
                }
                if (chosen < 0) {
                    ++stats.skipped_no_legal_match;
                    if (stats.unmatched_examples.size() < 16) {
                        int best_diff = 1'000'000;
                        const PlacementAction* best_action = nullptr;
                        bool replay_origin_present = false;
                        int replay_origin_diff = -1;
                        for (const PlacementAction& action : actions) {
                            const bool same_piece_hold_is_canonical_no_hold =
                                turn.exact_used_hold && !action.use_hold &&
                                active_state.hold == active_state.current &&
                                action.final_piece == active_state.current;
                            if ((!same_piece_hold_is_canonical_no_hold &&
                                 action.use_hold != turn.exact_used_hold) ||
                                action.final_piece != turn.exact_final_piece)
                                continue;
                            const PlacementOutcome candidate =
                                evaluate_placement(active_state.board, action.piece_state(), rules);
                            int diff = 0;
                            for (int y = 0; y < candidate.board.height(); ++y) {
                                diff += std::popcount(candidate.board.row(y) ^ turn.exact_placement_board.row(y));
                                diff += std::popcount(candidate.board.garbage_row(y) ^
                                                      turn.exact_placement_board.garbage_row(y));
                            }
                            if (turn.exact_final_piece != Piece::I &&
                                turn.exact_final_piece != Piece::O &&
                                action.final_x == turn.exact_final_x - 1 &&
                                action.final_y == turn.exact_final_y - 1 &&
                                action.final_rotation == turn.exact_final_rotation) {
                                replay_origin_present = true;
                                replay_origin_diff = diff;
                            }
                            if (diff < best_diff) {
                                best_diff = diff;
                                best_action = &action;
                            }
                        }
                        std::ostringstream detail;
                        detail << "round=" << game.round_index
                               << " frame=" << turn.frame
                               << " player=" << player
                               << " current=" << piece_name(active_state.current)
                               << " final=" << piece_name(turn.exact_final_piece)
                               << " hold=" << (turn.exact_used_hold ? 1 : 0)
                               << " x=" << turn.exact_final_x
                               << " y=" << turn.exact_final_y
                               << " r=" << static_cast<int>(turn.exact_final_rotation)
                               << " actions=" << actions.size()
                               << " target_present=" << (replay_origin_present ? 1 : 0)
                               << " target_diff=" << replay_origin_diff
                               << " best_diff=" << best_diff;
                        if (best_action) {
                            detail << " best_action=(x=" << best_action->final_x
                                   << ",y=" << best_action->final_y
                                   << ",r=" << static_cast<int>(best_action->final_rotation)
                                   << ",dur=" << best_action->base_duration << ")";
                        }
                        stats.unmatched_examples.push_back(detail.str());
                    }
                    state[static_cast<std::size_t>(player)] = turn.exact_after;
                    ++move_number[static_cast<std::size_t>(player)];
                    continue;
                }

                const PlacementAction& chosen_action = actions[static_cast<std::size_t>(chosen)];
                const ActivePiece replay_start = spawn_piece_with_clutch(
                    active_state.board, chosen_action.final_piece, rules,
                    active_state.combo >= 0);
                const ExecutionResult reproduced = execute_inputs(
                    active_state.board, replay_start,
                    chosen_action.canonical_input_sequence, rules);
                if (!reproduced.ok || reproduced.piece.x != chosen_action.final_x ||
                    reproduced.piece.y != chosen_action.final_y ||
                    reproduced.piece.rot != chosen_action.final_rotation) {
                    ++stats.skipped_execution;
                    if (stats.unmatched_examples.size() < 16) {
                        std::ostringstream detail;
                        detail << "sequence mismatch round=" << game.round_index
                               << " frame=" << turn.frame << " player=" << player
                               << " target=(" << chosen_action.final_x << ','
                               << chosen_action.final_y << ','
                               << static_cast<int>(chosen_action.final_rotation) << ")"
                               << " replayed=(" << reproduced.piece.x << ','
                               << reproduced.piece.y << ','
                               << static_cast<int>(reproduced.piece.rot) << ") seq=";
                        for (const Input input : chosen_action.canonical_input_sequence)
                            detail << input_name(input) << ',';
                        stats.unmatched_examples.push_back(detail.str());
                    }
                    state[static_cast<std::size_t>(player)] = turn.exact_after;
                    ++move_number[static_cast<std::size_t>(player)];
                    continue;
                }

                TrainingSample sample;
                sample.tokens = tokenizer.encode(obs, rules).tokens;
                sample.action_embeddings.reserve(actions.size());
                for (const PlacementAction& action : actions)
                    sample.action_embeddings.push_back(embed_action(action, obs.board, rules));
                sample.search_policy.assign(actions.size(), 0.0f);
                sample.search_policy[static_cast<std::size_t>(chosen)] = 1.0f;
                sample.outcome = std::clamp(game.outcome[static_cast<std::size_t>(player)], -1.0f, 1.0f);
                sample.n_step_return = sample.outcome;
                sample.search_value = 0.0f;
                sample.aux_target_schema_version = schema::AUX_TARGET_SCHEMA_VERSION;
                sample.aux_targets.fill(0.0f);
                sample.aux_valid.fill(0);
                sample.termination_reason = TerminationReason::Terminated;
                sample.player_index = player;
                sample.value_perspective = 1;
                sample.timestamp = turn.frame;
                sample.trajectory_index = move_number[static_cast<std::size_t>(player)];
                sample.ruleset_hash = rules.hash();
                sample.game_seed = source_hash;
                sample.model_version = model_version;
                sample.move_number = move_number[static_cast<std::size_t>(player)];
                sample.chosen_action = chosen;
                samples.push_back(std::move(sample));
                ++stats.imported;

                state[static_cast<std::size_t>(player)] = turn.exact_after;
                ++move_number[static_cast<std::size_t>(player)];
                continue;
            }

            HumanReplayState& active_state = state[static_cast<std::size_t>(player)];
            if (!active_state.valid || active_state.current == Piece::None) {
                ++stats.skipped_invalid_state;
                continue;
            }

            const Observation obs = human_detail::observation_from_state(
                active_state, state[static_cast<std::size_t>(opponent)], turn.frame, rules);
            const human_detail::TurnExecution execution =
                human_detail::execute_turn(active_state, turn, rules);
            if (!execution.ok) {
                ++stats.skipped_execution;
                continue;
            }

            const Piece next_after_hold = active_state.queue.empty()
                                              ? Piece::None
                                              : active_state.queue.front();
            const std::vector<PlacementAction> actions = movegen.generate(
                active_state.board, active_state.current, active_state.hold,
                next_after_hold, rules, active_state.combo >= 0);
            int chosen = -1;
            Tick best_duration = TICK_NEVER;
            for (std::size_t i = 0; i < actions.size(); ++i) {
                if (!human_detail::same_placement(actions[i], execution.piece,
                                                  execution.used_hold, execution.outcome,
                                                  active_state.board, rules))
                    continue;
                if (chosen < 0 || actions[i].base_duration < best_duration) {
                    chosen = static_cast<int>(i);
                    best_duration = actions[i].base_duration;
                }
            }
            if (chosen < 0) {
                ++stats.skipped_no_legal_match;
                // Advance the reconstructed game anyway; one unrepresentable
                // placement must not poison every later board in the replay.
                human_detail::advance_state(active_state, execution);
                ++move_number[static_cast<std::size_t>(player)];
                continue;
            }

            TrainingSample sample;
            sample.tokens = tokenizer.encode(obs, rules).tokens;
            sample.action_embeddings.reserve(actions.size());
            for (const PlacementAction& action : actions)
                sample.action_embeddings.push_back(embed_action(action, obs.board, rules));
            sample.search_policy.assign(actions.size(), 0.0f);
            sample.search_policy[static_cast<std::size_t>(chosen)] = 1.0f;
            sample.outcome = std::clamp(game.outcome[static_cast<std::size_t>(player)], -1.0f, 1.0f);
            sample.n_step_return = sample.outcome;
            sample.search_value = 0.0f;
            sample.aux_target_schema_version = schema::AUX_TARGET_SCHEMA_VERSION;
            sample.aux_targets.fill(0.0f);
            sample.aux_valid.fill(0);
            sample.termination_reason = TerminationReason::Terminated;
            sample.player_index = player;
            sample.value_perspective = 1;
            sample.timestamp = turn.frame;
            sample.trajectory_index = move_number[static_cast<std::size_t>(player)];
            sample.ruleset_hash = rules.hash();
            sample.game_seed = source_hash;
            sample.model_version = model_version;
            sample.move_number = move_number[static_cast<std::size_t>(player)];
            sample.chosen_action = chosen;
            samples.push_back(std::move(sample));
            ++stats.imported;

            human_detail::advance_state(active_state, execution);
            ++move_number[static_cast<std::size_t>(player)];
        }
    }
    return stats;
}

inline bool import_human_replay_protocol(const std::string& input_path,
                                         const std::string& output_path,
                                         const RulesetConfig& rules,
                                         std::uint32_t model_version,
                                         HumanReplayImportStats* stats_out = nullptr,
                                         std::string* error = nullptr) {
    std::vector<HumanReplayGame> games;
    if (!read_human_replay_protocol(input_path, rules, games, error)) return false;

    std::vector<TrainingSample> samples;
    const HumanReplayImportStats stats =
        build_human_replay_samples(games, rules, model_version, samples);
    if (samples.empty()) {
        if (error) *error = "no human replay samples could be imported";
        if (stats_out) *stats_out = stats;
        return false;
    }

    std::vector<const TrainingSample*> pointers;
    pointers.reserve(samples.size());
    for (const TrainingSample& sample : samples) pointers.push_back(&sample);
    const TensorBatch batch = make_training_batch(pointers);
    DatasetContract contract = default_dataset_contract(batch);
    contract.self_play_seed = 0;
    contract.termination_reason = TerminationReason::Terminated;
    const bool ok = write_dataset_file(output_path, batch, rules.hash(), model_version, contract);
    if (!ok && error) *error = "failed to write human .tetradat shard";
    if (stats_out) *stats_out = stats;
    return ok;
}

}  // namespace tetra
