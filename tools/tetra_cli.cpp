// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- developer CLI for the M0/M1 core.
//
// Subcommands:
//   ruleset [name]        print a ruleset and its hash
//   moves <piece> [seed]  enumerate legal placements on a random board
//   selfplay [seed] [n]   run a scripted greedy game and print statistics
//   bench [n]             movegen throughput benchmark
//   determinism [seed]    verify that a seed replays identically
//
// This binary is a local development tool only. It has no network code and
// cannot connect to TETR.IO (see docs/POLICY.md).
#include "tetra/movegen.hpp"
#include "tetra/player.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

using namespace tetra;

namespace {

RulesetConfig preset(const std::string& name) {
    if (name == "quickplay") return RulesetConfig::quick_play();
    if (name == "guideline") return RulesetConfig::guideline();
    return RulesetConfig::tetra_league();
}

void print_ruleset(const RulesetConfig& c) {
    std::printf("id                 %s\n", c.id.c_str());
    std::printf("version            %d\n", c.version);
    std::printf("ruleset_hash       %s\n", c.hash_hex().c_str());
    std::printf("geometry           %dx%d (visible %d)\n", c.geometry.width,
                c.geometry.internal_height, c.geometry.visible_height);
    std::printf("randomizer         type=%d preview=%d hold=%d\n",
                static_cast<int>(c.randomizer.type), c.randomizer.preview_count,
                c.randomizer.hold_enabled ? 1 : 0);
    std::printf("kicks              table=%d allow_180=%d\n",
                static_cast<int>(c.movement.kick_table), c.movement.allow_180 ? 1 : 0);
    std::printf("spin_detection     %d\n", static_cast<int>(c.clear_rules.spin_detection));
    std::printf("attack             S/D/T/Q = %d/%d/%d/%d  TSS/TSD/TST = %d/%d/%d\n",
                c.attack.single, c.attack.doubl, c.attack.triple, c.attack.quad,
                c.attack.tspin_single, c.attack.tspin_double, c.attack.tspin_triple);
    std::printf("b2b                mode=%d surge_base=%d start=%d segments=%d\n",
                static_cast<int>(c.attack.b2b_mode), c.attack.surge_base,
                c.attack.surge_start_streak, c.attack.surge_segments);
    std::printf("combo              1 + %d/%d * c   (log for zero base: %d)\n",
                c.attack.combo_multiplier_num, c.attack.combo_multiplier_den,
                c.attack.combo_log_for_zero_base ? 1 : 0);
    std::printf("rounding           %s\n",
                c.attack.rounding_mode == RoundingMode::Down ? "down" : "rng");
    std::printf("garbage            travel=%lld activation=%lld cap=%d messiness=%d%%\n",
                static_cast<long long>(c.garbage.travel_time),
                static_cast<long long>(c.garbage.activation_delay), c.garbage.cap,
                c.garbage.messiness_percent);
    std::printf("opener_phase       %d pieces (enabled=%d)\n", c.attack.opener_phase_pieces,
                c.attack.opener_phase_enabled ? 1 : 0);
}

// Greedy scripted policy used by the tools: clear as much as possible, then
// keep the stack low and avoid making holes.
size_t greedy_pick(const std::vector<PlacementAction>& acts, const Board& board,
                   const RulesetConfig& cfg) {
    size_t best = 0;
    double best_score = -1e18;
    for (size_t i = 0; i < acts.size(); ++i) {
        const PlacementOutcome oc = evaluate_placement(board, acts[i].piece_state(), cfg);
        double score = 0;
        score += 12.0 * oc.cleared_lines;
        score -= 8.0 * oc.board.hole_count();
        score -= 1.2 * oc.board.stack_height();
        int bumpiness = 0;
        for (int x = 0; x + 1 < oc.board.width(); ++x)
            bumpiness += std::abs(oc.board.column_height(x) - oc.board.column_height(x + 1));
        score -= 0.4 * bumpiness;
        if (oc.spin != SpinType::None) score += 3.0;
        if (oc.all_clear) score += 25.0;
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    return best;
}

int cmd_ruleset(int argc, char** argv) {
    print_ruleset(preset(argc > 2 ? argv[2] : "league"));
    return 0;
}

int cmd_moves(int argc, char** argv) {
    const RulesetConfig cfg = RulesetConfig::tetra_league();
    Piece piece = Piece::T;
    if (argc > 2) piece_from_char(static_cast<char>(std::toupper(argv[2][0])), piece);
    const std::uint64_t seed = (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : 1;

    Board b(cfg.geometry.width, cfg.geometry.internal_height);
    Rng rng(seed);
    for (int y = 0; y < 6; ++y)
        for (int x = 0; x < cfg.geometry.width; ++x)
            if (rng.chance(1, 3)) b.fill_cell(x, y, false);

    std::printf("board (seed %llu):\n%s\n", static_cast<unsigned long long>(seed),
                b.to_string().c_str());

    MoveGenerator gen;
    const auto acts = gen.generate_for_piece(b, piece, cfg, false);
    std::printf("piece %s: %zu legal placements\n\n", piece_name(piece), acts.size());
    for (size_t i = 0; i < acts.size() && i < 40; ++i) {
        const auto& a = acts[i];
        std::printf("  [%2zu] x=%2d y=%2d rot=%s spin=%-4s clears=%d  inputs:", i, a.final_x,
                    a.final_y, rot_name(a.final_rotation), spin_name(a.spin), a.cleared_lines);
        for (Input in : a.canonical_input_sequence) std::printf(" %s", input_name(in));
        std::printf("\n");
    }
    if (acts.size() > 40) std::printf("  ... %zu more\n", acts.size() - 40);
    return 0;
}

int cmd_selfplay(int argc, char** argv) {
    const std::uint64_t seed = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 1;
    const int pieces = (argc > 3) ? std::atoi(argv[3]) : 300;
    const RulesetConfig cfg = RulesetConfig::tetra_league();

    Player p;
    p.reset(cfg, seed, 0);
    MoveGenerator gen;

    int placed = 0, quads = 0, spins = 0;
    for (int i = 0; i < pieces && p.alive(); ++i) {
        const auto acts = gen.generate(p.board(), p.active().type, p.hold(),
                                       p.visible_next().empty() ? Piece::None : p.visible_next()[0],
                                       cfg);
        if (acts.empty()) break;
        const size_t pick = greedy_pick(acts, p.board(), cfg);
        if (acts[pick].use_hold && !p.do_hold()) continue;
        p.set_active(acts[pick].piece_state());
        int out = 0;
        const LockResult r = p.lock_piece(20, &out);
        ++placed;
        if (r.clear.lines >= 4) ++quads;
        if (r.clear.spin != SpinType::None) ++spins;
        if (!r.ok) break;
    }

    std::printf("seed             %llu\n", static_cast<unsigned long long>(seed));
    std::printf("ruleset_hash     %s\n", cfg.hash_hex().c_str());
    std::printf("pieces placed    %d\n", placed);
    std::printf("lines cleared    %lld\n", static_cast<long long>(p.lines_cleared()));
    std::printf("lines sent       %lld\n", static_cast<long long>(p.lines_sent()));
    std::printf("quads / spins    %d / %d\n", quads, spins);
    std::printf("attack per piece %.3f\n",
                placed ? static_cast<double>(p.lines_sent()) / placed : 0.0);
    std::printf("alive            %d (%s)\n", p.alive() ? 1 : 0, topout_name(p.topout_reason()));
    std::printf("final board:\n%s\n", p.board().to_string().c_str());
    return 0;
}

int cmd_bench(int argc, char** argv) {
    const int iters = (argc > 2) ? std::atoi(argv[2]) : 20000;
    const RulesetConfig cfg = RulesetConfig::tetra_league();
    MoveGenerator gen;
    Rng rng(1234);

    Board b(cfg.geometry.width, cfg.geometry.internal_height);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < cfg.geometry.width; ++x)
            if (rng.chance(2, 5)) b.fill_cell(x, y, false);

    const auto t0 = std::chrono::steady_clock::now();
    size_t total = 0;
    for (int i = 0; i < iters; ++i)
        total += gen.generate_for_piece(b, static_cast<Piece>(i % PIECE_COUNT), cfg, false).size();
    const auto t1 = std::chrono::steady_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("movegen: %d calls in %.1f ms  (%.1f us/call, %zu placements total)\n", iters, ms,
                ms * 1000.0 / iters, total);
    return 0;
}

int cmd_determinism(int argc, char** argv) {
    const std::uint64_t seed = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 1;
    const RulesetConfig cfg = RulesetConfig::tetra_league();
    MoveGenerator gen;

    auto run = [&]() {
        Player p;
        p.reset(cfg, seed, 0);
        std::uint64_t h = 1469598103934665603ull;
        for (int i = 0; i < 200 && p.alive(); ++i) {
            const auto acts = gen.generate_for_piece(p.board(), p.active().type, cfg, false);
            if (acts.empty()) break;
            const size_t pick = greedy_pick(acts, p.board(), cfg);
            p.set_active(acts[pick].piece_state());
            int out = 0;
            const LockResult r = p.lock_piece(20, &out);
            h ^= static_cast<std::uint64_t>(out + 1);
            h *= 1099511628211ull;
            h ^= detail::board_hash(p.board());
            h *= 1099511628211ull;
            if (!r.ok) break;
        }
        return h;
    };

    const std::uint64_t a = run(), b = run(), c = run();
    std::printf("run hashes: %016llx %016llx %016llx\n", static_cast<unsigned long long>(a),
                static_cast<unsigned long long>(b), static_cast<unsigned long long>(c));
    if (a == b && b == c) {
        std::printf("OK: deterministic\n");
        return 0;
    }
    std::printf("FAIL: non-deterministic\n");
    return 1;
}

void usage() {
    std::printf(
        "tetra_cli -- TetraFormer M0/M1 developer tool (local simulator only)\n\n"
        "  ruleset [league|quickplay|guideline]\n"
        "  moves <piece> [seed]\n"
        "  selfplay [seed] [pieces]\n"
        "  bench [iterations]\n"
        "  determinism [seed]\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    const std::string cmd = argv[1];
    if (cmd == "ruleset") return cmd_ruleset(argc, argv);
    if (cmd == "moves") return cmd_moves(argc, argv);
    if (cmd == "selfplay") return cmd_selfplay(argc, argv);
    if (cmd == "bench") return cmd_bench(argc, argv);
    if (cmd == "determinism") return cmd_determinism(argc, argv);
    usage();
    return 1;
}
