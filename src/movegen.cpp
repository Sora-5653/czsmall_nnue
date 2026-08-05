// SPDX-License-Identifier: Apache-2.0
// Adaptation boundary for Kixenon/cobra-movegen. The vendored implementation
// itself is retained under include/cobra/src with its upstream source text.
//
// Copyright 2026 Kixenon
// Modifications: this translation unit connects the upstream row backend to
// Tetra's fixed geometry, action, input, spin, and timing interfaces.
#include "tetra/movegen.hpp"

#include "cobra/src/row/board.hpp"
#include "cobra/src/row/movegen.hpp"
#include "cobra/src/row/pathfinder.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tetra {

template <Cobra::Policy::KickRule Kick, bool Enable180, bool TrackTSpin>
struct CobraRules : Cobra::RulesetBase {
    static constexpr Cobra::Policy::KickRule KICKS = Kick;
    static constexpr Cobra::Policy::SpinRule SPINS =
        TrackTSpin ? Cobra::Policy::SpinRule::TSPIN : Cobra::Policy::SpinRule::NONE;
    static constexpr int SPAWN_Y = 20;
    static constexpr bool ENABLE_180 = Enable180;
};

namespace {

using CobraBoard = Cobra::Board<>;

struct BoardView {
    CobraBoard board{};
};

// Cobra is the only supported movement backend.  The adapter intentionally
// accepts only the standard 10 x 40 Tetra field; custom room geometries are
// outside Cobra's fixed board contract and return no actions.
inline bool cobra_geometry(const Board& board, const RulesetConfig& cfg) {
    return cfg.geometry.width == Cobra::COL_NB && board.width() == Cobra::COL_NB &&
           board.height() == 40 && cfg.geometry.internal_height == 40 &&
           cfg.geometry.visible_height == 20 && cfg.movement.kick_table != KickTableId::None;
}

inline BoardView to_cobra_board(const Board& source) {
    BoardView view;
    for (int y = 0; y < source.height(); ++y) {
        const std::uint32_t row = source.row(y);
        for (int x = 0; x < source.width(); ++x) {
            if ((row & (std::uint32_t{1} << x)) != 0)
                view.board.set(x, y);
        }
    }
    return view;
}

inline bool map_cobra_input(Cobra::PathFinder::Input in, Input& out) {
    switch (in) {
        case Cobra::PathFinder::Input::SHIFT_LEFT: out = Input::Left; return true;
        case Cobra::PathFinder::Input::SHIFT_RIGHT: out = Input::Right; return true;
        case Cobra::PathFinder::Input::DAS_LEFT: out = Input::DasLeft; return true;
        case Cobra::PathFinder::Input::DAS_RIGHT: out = Input::DasRight; return true;
        case Cobra::PathFinder::Input::ROTATE_CW: out = Input::Cw; return true;
        case Cobra::PathFinder::Input::ROTATE_CCW: out = Input::Ccw; return true;
        case Cobra::PathFinder::Input::ROTATE_FLIP: out = Input::Flip; return true;
        case Cobra::PathFinder::Input::SOFT_DROP: out = Input::SoftDrop; return true;
        case Cobra::PathFinder::Input::HARD_DROP: out = Input::HardDrop; return true;
        case Cobra::PathFinder::Input::NO_INPUT: return false;
    }
    return false;
}

inline bool gravity_reachable(const Board& board, Piece piece, const std::vector<Input>& sequence,
                              const RulesetConfig& cfg,
                              const MoveGenerator::Options& options) {
    const HandlingModel h = HandlingModel::from(cfg);
    if (!options.enforce_gravity || h.gravity_num <= 0 ||
        h.ticks_per_cell() > options.gravity_check_threshold)
        return true;

    const ActivePiece start = spawn_piece(piece, cfg);
    std::vector<Input> prefix;
    prefix.reserve(sequence.size());
    for (const Input in : sequence) {
        prefix.push_back(in);
        if (in == Input::HardDrop) break;

        const ExecutionResult r = execute_inputs(board, start, prefix, cfg);
        if (!r.ok) return false;
        const Tick movement_cost = std::max<Tick>(0, r.cost - h.are);
        const Tick required_y = static_cast<Tick>(start.y) - h.gravity_fall(movement_cost);
        if (static_cast<Tick>(r.piece.y) > required_y) return false;
    }
    return true;
}

inline std::uint64_t action_key(const PlacementAction& a) {
    std::uint64_t k = a.resulting_board_hash;
    k = k * 1099511628211ull + static_cast<std::uint64_t>(a.cleared_lines);
    k = k * 1099511628211ull + static_cast<std::uint64_t>(a.cleared_garbage ? 1 : 0);
    k = k * 1099511628211ull + static_cast<std::uint64_t>(a.all_clear ? 1 : 0);
    k = k * 1099511628211ull + static_cast<std::uint64_t>(a.spin);
    k = k * 1099511628211ull + static_cast<std::uint64_t>(a.use_hold ? 1 : 0);
    k = k * 1099511628211ull + static_cast<std::uint64_t>(a.final_piece);
    return k;
}

inline void append_action(PlacementAction action, const MoveGenerator::Options& options,
                          std::vector<PlacementAction>& out,
                          std::unordered_map<std::uint64_t, size_t>& by_outcome) {
    if (!options.merge_equivalent) {
        out.push_back(std::move(action));
        return;
    }

    const std::uint64_t key = action_key(action);
    const auto it = by_outcome.find(key);
    if (it == by_outcome.end()) {
        by_outcome.emplace(key, out.size());
        out.push_back(std::move(action));
        return;
    }

    PlacementAction& kept = out[it->second];
    const bool cheaper = action.base_duration < kept.base_duration;
    const bool same_cost_shorter =
        action.base_duration == kept.base_duration &&
        action.canonical_input_sequence.size() < kept.canonical_input_sequence.size();
    if (cheaper || same_cost_shorter) kept = std::move(action);
}

using CobraPathMap = std::unordered_map<std::uint64_t, Cobra::PathFinder::Inputs>;

inline std::uint64_t cobra_target_key(const Cobra::Move& target) {
    std::uint64_t k = static_cast<std::uint64_t>(target.x);
    k = k * Cobra::ROW_NB + static_cast<std::uint64_t>(target.y);
    k = k * Cobra::Rotation::size + static_cast<std::uint64_t>(target.rotation.value);
    k = k * Cobra::SpinType::size + static_cast<std::uint64_t>(target.spin.value);
    return k;
}

template <typename PathRules, Cobra::Piece CP>
CobraPathMap cobra_paths(const CobraBoard& board, const bool use_finesse) {
    CobraPathMap paths;
    for (auto& [target, inputs] :
         Cobra::PathFinder::get_all_input<PathRules, CP>(board, use_finesse))
        paths.emplace(cobra_target_key(target), std::move(inputs));
    return paths;
}

inline bool map_cobra_path(const Cobra::PathFinder::Inputs& path, bool use_hold,
                           std::vector<Input>& out) {
    if (path.empty()) return false;
    out.clear();
    out.reserve(path.size() + (use_hold ? 1u : 0u));
    if (use_hold) out.push_back(Input::Hold);
    for (const Cobra::PathFinder::Input in : path) {
        Input mapped = Input::HardDrop;
        if (!map_cobra_input(in, mapped)) return false;
        out.push_back(mapped);
    }
    return true;
}

template <typename MoveRules, typename PathRules, Cobra::Piece CP>
void collect_piece(const BoardView& view, const Board& source, Piece tetra_piece,
                   bool use_hold, const RulesetConfig& cfg,
                   const MoveGenerator::Options& options, std::vector<PlacementAction>& out,
                   std::unordered_map<std::uint64_t, size_t>& by_outcome,
                   const bool spin_only) {
    const CobraPathMap tap_paths = cobra_paths<PathRules, CP>(view.board, false);
    const CobraPathMap finesse_paths = cobra_paths<PathRules, CP>(view.board, true);
    const Cobra::MoveList<MoveRules, CP, CobraBoard> moves(view.board, view.board.max_y());
    moves.for_each_move([&]<Cobra::Rotation R>(const int x, const int y,
                                                const Cobra::SpinType spin) {
        if (spin_only && spin == Cobra::SpinType::NONE) return;
        const Cobra::Move target{CP, R, x, y, spin};
        const auto tap_it = tap_paths.find(cobra_target_key(target));
        const auto finesse_it = finesse_paths.find(cobra_target_key(target));
        std::vector<Input> tap_sequence, finesse_sequence, sequence;
        const bool have_tap = tap_it != tap_paths.end() &&
                              map_cobra_path(tap_it->second, use_hold, tap_sequence);
        const bool have_finesse = finesse_it != finesse_paths.end() &&
                                  map_cobra_path(finesse_it->second, use_hold, finesse_sequence);
        ExecutionResult tap_result, finesse_result;
        if (have_tap)
            tap_result = execute_inputs(source, spawn_piece(tetra_piece, cfg), tap_sequence, cfg);
        if (have_finesse)
            finesse_result =
                execute_inputs(source, spawn_piece(tetra_piece, cfg), finesse_sequence, cfg);
        if (!tap_result.ok && !finesse_result.ok) return;
        if (!tap_result.ok ||
            (finesse_result.ok &&
             (finesse_result.cost < tap_result.cost ||
              (finesse_result.cost == tap_result.cost &&
               finesse_sequence.size() < tap_sequence.size())))) {
            sequence = std::move(finesse_sequence);
        } else {
            sequence = std::move(tap_sequence);
        }
        if (!gravity_reachable(source, tetra_piece, sequence, cfg, options)) return;

        const ExecutionResult full =
            execute_inputs(source, spawn_piece(tetra_piece, cfg), sequence, cfg);
        if (!full.ok) return;

        ActivePiece landed = full.piece;

        const PlacementOutcome outcome = evaluate_placement(source, landed, cfg);
        PlacementAction action;
        action.use_hold = use_hold;
        action.final_piece = landed.type;
        action.final_x = landed.x;
        action.final_y = landed.y;
        action.final_rotation = landed.rot;
        action.spin = outcome.spin;
        action.last_kick = landed.last_kick;
        action.canonical_input_sequence = std::move(sequence);
        action.base_duration = full.cost +
                               (outcome.cleared_lines > 0 ? cfg.clear_rules.line_clear_delay : 0);
        action.cleared_lines = outcome.cleared_lines;
        action.cleared_garbage = outcome.cleared_garbage;
        action.all_clear = outcome.all_clear;
        action.resulting_board_hash = outcome.board_hash;
        append_action(std::move(action), options, out, by_outcome);
    });
}

template <typename MoveRules, typename PathRules>
void collect_piece(const BoardView& view, const Board& source, Piece piece, bool use_hold,
                   const RulesetConfig& cfg, const MoveGenerator::Options& options,
                   std::vector<PlacementAction>& out,
                   std::unordered_map<std::uint64_t, size_t>& by_outcome,
                   const bool spin_only) {
    switch (piece) {
        case Piece::I:
            collect_piece<MoveRules, PathRules, Cobra::Piece::I>(
                view, source, piece, use_hold, cfg, options, out, by_outcome, spin_only);
            break;
        case Piece::J:
            collect_piece<MoveRules, PathRules, Cobra::Piece::J>(
                view, source, piece, use_hold, cfg, options, out, by_outcome, spin_only);
            break;
        case Piece::L:
            collect_piece<MoveRules, PathRules, Cobra::Piece::L>(
                view, source, piece, use_hold, cfg, options, out, by_outcome, spin_only);
            break;
        case Piece::O:
            collect_piece<MoveRules, PathRules, Cobra::Piece::O>(
                view, source, piece, use_hold, cfg, options, out, by_outcome, spin_only);
            break;
        case Piece::S:
            collect_piece<MoveRules, PathRules, Cobra::Piece::S>(
                view, source, piece, use_hold, cfg, options, out, by_outcome, spin_only);
            break;
        case Piece::T:
            collect_piece<MoveRules, PathRules, Cobra::Piece::T>(
                view, source, piece, use_hold, cfg, options, out, by_outcome, spin_only);
            break;
        case Piece::Z:
            collect_piece<MoveRules, PathRules, Cobra::Piece::Z>(
                view, source, piece, use_hold, cfg, options, out, by_outcome, spin_only);
            break;
        case Piece::None: break;
    }
}

template <Cobra::Policy::KickRule Kick, bool Enable180>
void collect_with_kick(const BoardView& view, const Board& source, Piece piece, bool use_hold,
                       const RulesetConfig& cfg, const MoveGenerator::Options& options,
                       std::vector<PlacementAction>& out,
                       std::unordered_map<std::uint64_t, size_t>& by_outcome) {
    using PathRules = CobraRules<Kick, Enable180, true>;
    if (piece == Piece::T && cfg.clear_rules.spin_detection != SpinDetection::None) {
        // The non-spin MoveList is the complete reachable-placement set.  The
        // spin-aware list is an additional Cobra pass for alternate paths that
        // lock the same cells as a T-spin.  Keeping these passes separate is
        // important: Cobra's spin reachability bitmap is provenance-aware and
        // must not suppress a plain landing which is mirror-equivalent to a
        // spin landing on another board.
        collect_piece<CobraRules<Kick, Enable180, false>,
                      PathRules>(
            view, source, piece, use_hold, cfg, options, out, by_outcome, false);
        collect_piece<CobraRules<Kick, Enable180, true>, PathRules>(
            view, source, piece, use_hold, cfg, options, out, by_outcome, true);
    } else {
        collect_piece<CobraRules<Kick, Enable180, false>,
                      CobraRules<Kick, Enable180, false>>(
            view, source, piece, use_hold, cfg, options, out, by_outcome, false);
    }
}

}  // namespace

std::vector<PlacementAction> MoveGenerator::generate_for_piece(const Board& board, Piece piece,
                                                                const RulesetConfig& cfg,
                                                                bool use_hold) const {
    std::vector<PlacementAction> out;
    if (!cobra_geometry(board, cfg) || piece == Piece::None) return out;

    const BoardView view = to_cobra_board(board);
    std::unordered_map<std::uint64_t, size_t> by_outcome;

    switch (cfg.movement.kick_table) {
        case KickTableId::SRS:
            if (options_.allow_180 && cfg.movement.allow_180)
                collect_with_kick<Cobra::Policy::KickRule::SRS, true>(
                    view, board, piece, use_hold, cfg, options_, out, by_outcome);
            else
                collect_with_kick<Cobra::Policy::KickRule::SRS, false>(
                    view, board, piece, use_hold, cfg, options_, out, by_outcome);
            break;
        case KickTableId::SRS_PLUS:
        case KickTableId::SRS_X:
            if (options_.allow_180 && cfg.movement.allow_180)
                collect_with_kick<Cobra::Policy::KickRule::SRS_PLUS, true>(
                    view, board, piece, use_hold, cfg, options_, out, by_outcome);
            else
                collect_with_kick<Cobra::Policy::KickRule::SRS_PLUS, false>(
                    view, board, piece, use_hold, cfg, options_, out, by_outcome);
            break;
        case KickTableId::None:
            break;
    }
    return out;
}

}  // namespace tetra
