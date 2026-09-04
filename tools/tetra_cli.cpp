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
#include "tetra/replay.hpp"
#include "tetra/search.hpp"
#include "tetra/dataset.hpp"
#include "tetra/nnue.hpp"
#include "tetra/selfplay.hpp"
#include "tetra/arena.hpp"
#include "tetra/stats.hpp"
#include "tetra/gpu_evaluator.hpp"
#include "tetra/reanalyse.hpp"
#include "tetra/human_replay.hpp"

#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <memory>
#include <fstream>
#include <numeric>
#include <string>

using namespace tetra;

namespace {

std::unique_ptr<Evaluator> load_evaluator_from_name(const std::string& name) {
    if (name.empty() || name == "heuristic") {
        return std::make_unique<HeuristicEvaluator>();
    }
    if (name == "uniform") {
        return std::make_unique<UniformEvaluator>();
    }
    NnueWeights weights;
    std::string err;
    if (!weights.load(name, &err)) {
        std::fprintf(stderr, "cannot load weights %s: %s\n", name.c_str(), err.c_str());
        return nullptr;
    }
    std::printf("loaded weights %s (width %u, %u layers)\n",
                name.c_str(), weights.config().width, weights.config().layers);
    return std::make_unique<TetraFormerEvaluator>(std::move(weights));
}

std::unique_ptr<Evaluator> parse_cli_evaluator(int argc, char** argv,
                                               std::vector<std::string>* pos_args = nullptr,
                                               std::string* weights_path_out = nullptr) {
    std::string wpath;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc) {
            wpath = argv[++i];
        } else if (arg.rfind("--weights=", 0) == 0) {
            wpath = arg.substr(10);
        } else if (arg != "--v1" && arg != "--compact" && pos_args) {
            pos_args->push_back(arg);
        }
    }
    if (weights_path_out) *weights_path_out = wpath;
    return load_evaluator_from_name(wpath);
}

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
        // Spec 8.4: charge the action's real execution cost, not a constant.
        const LockResult r = p.lock_piece(acts[pick].total_duration(), &out);
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
    const double seconds = static_cast<double>(p.now()) / static_cast<double>(cfg.tick_rate);
    std::printf("elapsed          %.2f s (%lld ticks)\n", seconds,
                static_cast<long long>(p.now()));
    std::printf("pieces / second  %.2f\n", seconds > 0 ? placed / seconds : 0.0);
    std::printf("attack / second  %.3f\n",
                seconds > 0 ? static_cast<double>(p.lines_sent()) / seconds : 0.0);
    std::printf("APM              %.2f\n",
                attacks_per_minute(p.lines_sent(), p.now(), cfg));
    std::printf("APP              %.3f\n",
                attacks_per_piece(p.lines_sent(), placed));
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
            const LockResult r = p.lock_piece(acts[pick].total_duration(), &out);
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

int cmd_timing(int argc, char** argv) {
    // Show how action cost and the delay bins behave for one position.
    const RulesetConfig cfg = RulesetConfig::tetra_league();
    Piece piece = Piece::T;
    if (argc > 2) piece_from_char(static_cast<char>(std::toupper(argv[2][0])), piece);

    Board b(cfg.geometry.width, cfg.geometry.internal_height);
    MoveGenerator gen;
    const auto acts = gen.generate_for_piece(b, piece, cfg, false);

    const HandlingModel h = HandlingModel::from(cfg);
    std::printf("handling: tap=%lld das=%lld arr=%lld sdf=%lld are=%lld lock_delay=%lld\n\n",
                static_cast<long long>(h.tap), static_cast<long long>(h.das),
                static_cast<long long>(h.arr), static_cast<long long>(h.sdf),
                static_cast<long long>(h.are), static_cast<long long>(h.lock_delay));

    Tick cheapest = TICK_NEVER, dearest = 0;
    for (const auto& a : acts) {
        cheapest = std::min(cheapest, a.base_duration);
        dearest = std::max(dearest, a.base_duration);
    }
    std::printf("piece %s: %zu placements, cost %lld..%lld ticks\n\n", piece_name(piece),
                acts.size(), static_cast<long long>(cheapest),
                static_cast<long long>(dearest));

    for (size_t i = 0; i < acts.size() && i < 12; ++i) {
        const auto& a = acts[i];
        std::printf("  x=%2d rot=%s cost=%2lld  inputs:", a.final_x, rot_name(a.final_rotation),
                    static_cast<long long>(a.base_duration));
        for (Input in : a.canonical_input_sequence) std::printf(" %s", input_name(in));
        std::printf("\n");
    }

    // Delay bins against a pending garbage activation 45 ticks away.
    std::printf("\ndelay bins for the first placement (garbage activates at t=45):\n");
    const std::vector<PlacementAction> one{acts.front()};
    for (const auto& a : MoveGenerator::expand_delay_bins(one, cfg, /*now=*/0,
                                                          /*next_garbage=*/45)) {
        std::printf("  %-15s wait=%2lld total=%2lld\n", delay_bin_name(a.delay_bin),
                    static_cast<long long>(a.delay_ticks),
                    static_cast<long long>(a.total_duration()));
    }
    return 0;
}

int cmd_record(int argc, char** argv) {
    // Play a scripted game and write it to disk.
    const std::string path = (argc > 2) ? argv[2] : "game.tetrarep";
    const std::uint64_t seed = (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : 1;
    const int pieces = (argc > 4) ? std::atoi(argv[4]) : 300;
    const RulesetConfig cfg = RulesetConfig::tetra_league();

    Player p;
    p.reset(cfg, seed, 0);
    MoveGenerator gen;
    ReplayRecorder rec(cfg, seed, 0, /*checkpoint_interval=*/16);
    rec.note("tetra_cli record");

    for (int i = 0; i < pieces && p.alive(); ++i) {
        const auto acts = gen.generate(p.board(), p.active().type, p.hold(),
                                       p.visible_next().empty() ? Piece::None
                                                                : p.visible_next()[0],
                                       cfg);
        if (acts.empty()) break;
        const size_t pick = greedy_pick(acts, p.board(), cfg);
        if (acts[pick].use_hold && !p.do_hold()) continue;
        p.set_active(acts[pick].piece_state());
        int out = 0;
        const LockResult r = p.lock_piece(acts[pick].total_duration(), &out);
        rec.record(acts[pick], p);
        if (!r.ok) break;
    }

    const Replay replay = rec.finish(p);
    if (!write_replay_file(path, replay)) {
        std::printf("failed to write %s\n", path.c_str());
        return 1;
    }
    const size_t bytes = serialize(replay).size();
    std::printf("wrote %s\n", path.c_str());
    std::printf("  ruleset      %s (%s)\n", replay.ruleset_id.c_str(),
                cfg.hash_hex().c_str());
    std::printf("  seed         %llu\n", static_cast<unsigned long long>(replay.seed));
    std::printf("  placements   %zu\n", replay.placements.size());
    std::printf("  size         %zu bytes (%.1f per placement)\n", bytes,
                replay.placements.empty()
                    ? 0.0
                    : static_cast<double>(bytes) /
                          static_cast<double>(replay.placements.size()));
    std::printf("  lines sent   %lld\n", static_cast<long long>(replay.final_lines_sent));
    std::printf("  result       %s\n",
                replay.final_alive ? "survived" : topout_name(replay.final_topout));
    return 0;
}

int cmd_verify(int argc, char** argv) {
    // Re-simulate a replay and report the first divergence, if any.
    if (argc < 3) {
        std::printf("usage: tetra_cli verify <file.tetrarep>\n");
        return 1;
    }
    const DeserializeResult rd = read_replay_file(argv[2]);
    if (!rd.ok) {
        std::printf("cannot read replay: %s\n", rd.error.c_str());
        return 1;
    }
    const Replay& r = rd.replay;

    // Pick the preset whose hash matches the replay.
    const RulesetConfig presets[] = {RulesetConfig::tetra_league(), RulesetConfig::quick_play(),
                                     RulesetConfig::guideline()};
    const RulesetConfig* cfg = nullptr;
    for (const auto& c : presets)
        if (c.hash() == r.ruleset_hash) cfg = &c;
    if (!cfg) {
        std::printf("no known ruleset matches hash %016llx (replay says \"%s\")\n",
                    static_cast<unsigned long long>(r.ruleset_hash), r.ruleset_id.c_str());
        return 1;
    }

    const VerifyResult v = verify_replay(r, *cfg);
    std::printf("replay       %s\n", argv[2]);
    std::printf("ruleset      %s (%s)\n", r.ruleset_id.c_str(), cfg->hash_hex().c_str());
    std::printf("placements   %zu\n", r.placements.size());
    if (v.ok) {
        std::printf("RESULT       OK: %d placements re-simulated exactly\n",
                    v.placements_applied);
        return 0;
    }
    std::printf("RESULT       DIVERGED at placement %d\n", v.first_divergence);
    std::printf("  %s\n", v.error.c_str());
    if (v.expected_hash || v.actual_hash)
        std::printf("  expected board %016llx, got %016llx\n",
                    static_cast<unsigned long long>(v.expected_hash),
                    static_cast<unsigned long long>(v.actual_hash));
    return 1;
}

int cmd_search(int argc, char** argv) {
    // Inspect one search: candidates, timing and batch efficiency.
    const int sims = (argc > 2) ? std::atoi(argv[2]) : 64;
    const bool gumbel = (argc > 3) ? (std::atoi(argv[3]) != 0) : true;
    const std::uint64_t seed = (argc > 4) ? std::strtoull(argv[4], nullptr, 10) : 42;
    const RulesetConfig cfg = RulesetConfig::tetra_league();

    Player p;
    p.reset(cfg, seed, 0);
    MoveGenerator gen;
    for (int i = 0; i < 25 && p.alive(); ++i) {
        const auto a = gen.generate_for_piece(p.board(), p.active().type, cfg, false);
        if (a.empty()) break;
        size_t best = 0;
        for (size_t k = 1; k < a.size(); ++k)
            if (a[k].final_y < a[best].final_y) best = k;
        p.set_active(a[best].piece_state());
        int out = 0;
        if (!p.lock_piece(a[best].total_duration(), &out).ok) break;
    }

    std::printf("board:\n%s\n", p.board().to_string(14).c_str());
    std::printf("active %s   hold %s\n\n", piece_name(p.active().type), piece_name(p.hold()));

    HeuristicEvaluator ev;
    SearchConfig sc;
    sc.simulations = sims;
    sc.max_depth = 8;
    sc.use_gumbel = gumbel;
    sc.batch_size = 16;
    sc.seed = 1;
    Searcher s(ev, sc);

    const auto t0 = std::chrono::steady_clock::now();
    const SearchResult r = s.search(p);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();

    const auto acts = gen.generate(p.board(), p.active().type, p.hold(),
                                   p.visible_next().empty() ? Piece::None : p.visible_next()[0],
                                   cfg);
    std::printf("mode            %s\n", gumbel ? "gumbel" : "puct");
    std::printf("simulations     %d (ran %d)\n", sims, r.simulations_run);
    std::printf("elapsed         %.1f ms\n", ms);
    std::printf("nodes           %d\n", r.nodes_created);
    std::printf("evaluator calls %d for %d positions (mean batch %.1f)\n", r.evaluator_calls,
                r.positions_evaluated, r.mean_batch_size);
    std::printf("transposition   %d hits\n", r.transposition_hits);
    std::printf("value (WDL)     %.3f / %.3f / %.3f\n", r.value.win, r.value.draw, r.value.loss);

    std::vector<SearchCandidate> top = r.candidates;
    std::sort(top.begin(), top.end(),
              [](const SearchCandidate& a, const SearchCandidate& b) { return a.visits > b.visits; });
    std::printf("\ntop candidates:\n");
    for (size_t i = 0; i < top.size() && i < 8; ++i) {
        const PlacementAction& a = acts[static_cast<size_t>(top[i].action_index)];
        std::printf("  %s x=%2d rot=%s clears=%d spin=%-4s  visits=%3d  q=%+.3f  prior=%.3f%s\n",
                    piece_name(a.final_piece), a.final_x, rot_name(a.final_rotation),
                    a.cleared_lines, spin_name(a.spin), top[i].visits, top[i].q_value,
                    top[i].prior,
                    top[i].action_index == r.best_action ? "  <- chosen" : "");
    }
    return 0;
}

int cmd_selfplay_gen(int argc, char** argv) {
    // Run the self-play loop and report what the trainer would receive.
    std::vector<std::string> args;
    auto ev = parse_cli_evaluator(argc, argv, &args);
    if (!ev) return 1;
    const int games = (args.size() > 0) ? std::atoi(args[0].c_str()) : 5;
    const int pieces = (args.size() > 1) ? std::atoi(args[1].c_str()) : 100;
    const int sims = (args.size() > 2) ? std::atoi(args[2].c_str()) : 16;
    const RulesetConfig rules = RulesetConfig::tetra_league();

    SelfPlayConfig cfg;
    cfg.max_pieces = pieces;
    cfg.search.simulations = sims;
    cfg.search.max_depth = 6;
    cfg.search.use_gumbel = true;
    cfg.search.batch_size = 16;
    cfg.search.root_noise_fraction = 0.25f;
    cfg.search.root_noise_alpha = 0.3f;
    cfg.search.determinizations = 2;
    cfg.garbage_style = GarbageStyle::Steady;
    cfg.garbage_period = 8;
    cfg.garbage_lines = 2;
    cfg.truncation_is_draw = false;

    SelfPlayWorker worker(*ev, cfg);
    ReplayBuffer buffer(200000);

    std::printf("ruleset %s (%s), %d games x %d pieces, %d sims\n\n", rules.id.c_str(),
                rules.hash_hex().c_str(), games, pieces, sims);
    std::printf("%5s %8s %8s %7s %7s %7s %6s %8s %s\n", "game", "pieces", "cleared",
                "sent", "recv", "APM", "APP", "outcome", "result");

    int total_pieces = 0, survived = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < games; ++g) {
        SelfPlayStats st;
        auto samples = worker.play(rules, static_cast<std::uint64_t>(g), &st);
        total_pieces += st.pieces;
        if (st.survived) ++survived;
        const char* result = st.outcome > 0.5f ? "win" :
                             (st.outcome < -0.5f ? "loss" : "draw/truncated");
        std::printf("%5d %8d %8d %7d %7d %7.1f %6.3f %+8.0f %s\n", g, st.pieces,
                    st.lines_cleared, st.lines_sent, st.lines_received,
                    attacks_per_minute(st.lines_sent, st.duration, rules),
                    attacks_per_piece(st.lines_sent, st.pieces),
                    static_cast<double>(st.outcome),
                    result);
        buffer.push_game(std::move(samples));
    }
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::printf("\nbuffer     %zu samples (capacity %zu)\n", buffer.size(), buffer.capacity());
    std::printf("rulesets   %zu distinct\n", buffer.ruleset_hashes().size());
    std::printf("survived   %d/%d\n", survived, games);
    std::printf("throughput %.1f placements/s over %.1f s\n",
                secs > 0 ? total_pieces / secs : 0.0, secs);

    Rng rng(1);
    const auto batch = buffer.sample(32, rng);
    if (!batch.empty()) {
        size_t tokens = 0, actions = 0;
        for (const auto* s : batch) {
            tokens += s->tokens.size();
            actions += s->action_embeddings.size();
        }
        std::printf("\ntraining batch of %zu:\n", batch.size());
        std::printf("  mean tokens/sample  %.1f\n",
                    static_cast<double>(tokens) / static_cast<double>(batch.size()));
        std::printf("  mean actions/sample %.1f\n",
                    static_cast<double>(actions) / static_cast<double>(batch.size()));
        std::printf("  token features      %d\n", TOKEN_FEATURES);
        std::printf("  action features     %d\n", ACTION_FEATURES);
    }
    return 0;
}

int cmd_export(int argc, char** argv) {
    // Self-play straight into a .tetradat file for the Python trainer.
    std::vector<std::string> args;
    auto ev = parse_cli_evaluator(argc, argv, &args);
    if (!ev) return 1;
    bool compact = false;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--compact=1" || std::string(argv[i]) == "--compact")
            compact = true;
    }
    const std::string path = (args.size() > 0) ? args[0] : "train.tetradat";
    const int games = (args.size() > 1) ? std::atoi(args[1].c_str()) : 10;
    const int pieces = (args.size() > 2) ? std::atoi(args[2].c_str()) : 100;
    const int sims = (args.size() > 3) ? std::atoi(args[3].c_str()) : 16;
    const RulesetConfig rules = RulesetConfig::tetra_league();

    SelfPlayConfig cfg;
    cfg.max_pieces = pieces;
    cfg.search.simulations = sims;
    cfg.search.max_depth = 6;
    cfg.search.use_gumbel = true;
    cfg.search.batch_size = 16;
    cfg.search.root_noise_fraction = 0.25f;
    cfg.search.root_noise_alpha = 0.3f;
    cfg.search.determinizations = 2;
    cfg.garbage_style = GarbageStyle::Steady;
    cfg.garbage_period = 8;
    cfg.garbage_lines = 2;
    cfg.truncation_is_draw = true;

    SelfPlayWorker worker(*ev, cfg);
    ReplayBuffer buffer(1000000);

    const auto t0 = std::chrono::steady_clock::now();
    int survived = 0;
    for (int g = 0; g < games; ++g) {
        SelfPlayStats st;
        buffer.push_game(worker.play(rules, static_cast<std::uint64_t>(g), &st));
        if (st.survived) ++survived;
    }
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    if (!export_buffer(path, buffer, /*model_version=*/0, 0, 0, compact)) {
        std::printf("export failed\n");
        return 1;
    }
    const DatasetReadResult back = read_dataset_file(path);
    if (!back.ok) {
        std::printf("wrote %s but could not read it back: %s\n", path.c_str(),
                    back.error.c_str());
        return 1;
    }

    std::printf("wrote %s\n", path.c_str());
    std::printf("  samples       %u\n", back.header.samples);
    std::printf("  tokens        [%u, %u]\n", back.header.max_tokens,
                back.header.token_features);
    std::printf("  actions       [%u, %u]\n", back.header.max_actions,
                back.header.action_features);
    std::printf("  ruleset_hash  %016llx\n",
                static_cast<unsigned long long>(back.header.ruleset_hash));
    std::printf("  games         %d (%d survived) in %.1f s\n", games, survived, secs);
    std::printf("\ntrain with:\n  python trainer/train.py %s --steps 300\n", path.c_str());
    return 0;
}

int cmd_play(int argc, char** argv) {
    // Play with trained weights inside the C++ search: the closed loop.
    if (argc < 3) {
        std::printf("usage: tetra_cli play <weights.tetrawts> [pieces] [sims]\n");
        return 1;
    }
    const int pieces = (argc > 3) ? std::atoi(argv[3]) : 100;
    const int sims = (argc > 4) ? std::atoi(argv[4]) : 16;

    NnueWeights weights;
    std::string err;
    if (!weights.load(argv[2], &err)) {
        std::printf("cannot load weights: %s\n", err.c_str());
        return 1;
    }
    std::printf("weights    %s (width %u, %u layers, %u heads)\n", argv[2],
                weights.config().width, weights.config().layers, weights.config().heads);

    TetraFormerEvaluator ev(std::move(weights));
    const RulesetConfig rules = RulesetConfig::tetra_league();

    SelfPlayConfig cfg;
    cfg.max_pieces = pieces;
    cfg.search.simulations = sims;
    cfg.search.max_depth = 4;
    cfg.search.use_gumbel = true;
    cfg.search.batch_size = 16;
    cfg.garbage_style = GarbageStyle::Steady;

    SelfPlayWorker worker(ev, cfg);
    SelfPlayStats st;
    const auto t0 = std::chrono::steady_clock::now();
    worker.play(rules, 1, &st);
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::printf("pieces     %d\n", st.pieces);
    std::printf("cleared    %d\n", st.lines_cleared);
    std::printf("sent       %d\n", st.lines_sent);
    std::printf("received   %d\n", st.lines_received);
    std::printf("result     %s\n", st.survived ? "survived" : topout_name(st.topout));
    std::printf("APM        %.2f\n",
                attacks_per_minute(st.lines_sent, st.duration, rules));
    std::printf("APP        %.3f\n", attacks_per_piece(st.lines_sent, st.pieces));
    std::printf("speed      %.1f placements/s\n", secs > 0 ? st.pieces / secs : 0.0);
    return 0;
}

// Binary child process used by trainer/gpu_match.py.  stdout is reserved for
// the protocol; diagnostics must stay off the stream or the parent cannot
// decode the next frame.
int cmd_gpu_play_protocol(int argc, char** argv) {
    const int pieces = (argc > 2) ? std::atoi(argv[2]) : 100;
    const int sims = (argc > 3) ? std::atoi(argv[3]) : 16;
    const int batch = (argc > 4) ? std::atoi(argv[4]) : 16;
    const std::uint64_t seed = (argc > 5) ? std::strtoull(argv[5], nullptr, 10) : 1;
    const int determinizations = (argc > 6) ? std::atoi(argv[6]) : 1;
    const bool use_gumbel = (argc > 7) ? std::atoi(argv[7]) != 0 : false;
    enable_gpu_protocol_stdio();

    const RulesetConfig rules = RulesetConfig::tetra_league();
    SelfPlayConfig cfg;
    cfg.max_pieces = std::max(1, pieces);
    cfg.search.simulations = std::max(1, sims);
    cfg.search.max_depth = 6;
    cfg.search.use_gumbel = use_gumbel;
    cfg.search.batch_size = std::max(1, batch);
    cfg.search.root_noise_fraction = 0.25f;
    cfg.search.root_noise_alpha = 0.3f;
    cfg.search.determinizations = std::max(1, determinizations);
    cfg.garbage_style = GarbageStyle::Steady;
    cfg.garbage_period = 8;
    cfg.garbage_lines = 2;
    cfg.truncation_is_draw = true;

    RemoteGpuEvaluator ev(stdin, stdout, cfg.search.batch_size, 0);
    SelfPlayWorker worker(ev, cfg);
    SelfPlayStats st;
    worker.play(rules, seed, &st);
    write_gpu_game_result(stdout, st.pieces, st.lines_cleared, st.garbage_lines_cleared,
                          st.lines_sent, st.lines_received, st.survived, static_cast<int>(st.topout),
                          rules.tick_rate, st.outcome, st.duration, ev.positions_evaluated(),
                          ev.batches_issued());
    return 0;
}

// GPU-backed self-play data generation.  The child owns the rules, search and
// compact dataset; trainer/gpu_selfplay.py only serves batched network calls.
int cmd_gpu_export_protocol(int argc, char** argv) {
    const std::string path = (argc > 2) ? argv[2] : "train.tetradat";
    const int games = (argc > 3) ? std::atoi(argv[3]) : 10;
    const int pieces = (argc > 4) ? std::atoi(argv[4]) : 100;
    const int sims = (argc > 5) ? std::atoi(argv[5]) : 32;
    const int batch = (argc > 6) ? std::atoi(argv[6]) : 16;
    const std::uint64_t seed =
        (argc > 7) ? std::strtoull(argv[7], nullptr, 10) : 1;
    const std::uint32_t model_version =
        (argc > 8) ? static_cast<std::uint32_t>(std::strtoul(argv[8], nullptr, 10)) : 1;
    const int determinizations = (argc > 9) ? std::atoi(argv[9]) : 2;
    const bool use_gumbel = (argc > 10) ? std::atoi(argv[10]) != 0 : true;
    const float root_noise_fraction =
        (argc > 11) ? std::strtof(argv[11], nullptr) : 0.25f;
    const bool enable_timing_actions =
        (argc > 12) ? std::atoi(argv[12]) != 0 : false;
    const bool no_attack_delivery =
        (argc > 13) ? std::atoi(argv[13]) != 0 : false;
    enable_gpu_protocol_stdio();

    const RulesetConfig rules = RulesetConfig::tetra_league();
    SelfPlayConfig cfg;
    cfg.max_pieces = std::max(1, pieces);
    cfg.model_version = model_version;
    cfg.search.simulations = std::max(0, sims);
    cfg.search.max_depth = 6;
    cfg.search.use_gumbel = use_gumbel;
    cfg.search.batch_size = std::max(1, batch);
    cfg.search.root_noise_fraction =
        std::max(0.0f, std::min(1.0f, root_noise_fraction));
    cfg.search.root_noise_alpha = 0.3f;
    cfg.search.determinizations = std::max(1, determinizations);
    cfg.search.enable_timing_actions = enable_timing_actions;
    cfg.garbage_style = no_attack_delivery ? GarbageStyle::None : GarbageStyle::Steady;
    cfg.garbage_period = 8;
    cfg.garbage_lines = 2;
    cfg.truncation_is_draw = true;

    RemoteGpuEvaluator ev(stdin, stdout, cfg.search.batch_size, 0);
    SelfPlayWorker worker(ev, cfg);
    ReplayBuffer buffer(2u * static_cast<size_t>(std::max(1, games)) *
                        static_cast<size_t>(std::max(1, pieces)));
    for (int g = 0; g < std::max(1, games); ++g) {
        SelfPlayStats st;
        buffer.push_game(worker.play(rules, seed + static_cast<std::uint64_t>(g), &st));
        write_gpu_game_result(stdout, st.pieces, st.lines_cleared, st.garbage_lines_cleared,
                              st.lines_sent, st.lines_received, st.survived,
                              static_cast<int>(st.topout), rules.tick_rate, st.outcome,
                              st.duration, ev.positions_evaluated(), ev.batches_issued());
    }

    // Diagnose whether the policy target actually identifies the action the
    // search selected. This is especially important for Gumbel sequential
    // halving: the played action is the final survivor, while the historical
    // training target is the normalised root visit count. Keep this diagnostic
    // out of stdout because stdout is the binary GPU protocol stream.
    std::size_t chosen_valid = 0;
    std::size_t chosen_at_visit_max = 0;
    double chosen_mass_sum = 0.0;
    for (std::size_t i = 0; i < buffer.size(); ++i) {
        const TrainingSample& sample = buffer.at(i);
        if (sample.chosen_action < 0 ||
            static_cast<std::size_t>(sample.chosen_action) >= sample.search_policy.size() ||
            sample.search_policy.empty())
            continue;
        const float chosen_mass =
            sample.search_policy[static_cast<std::size_t>(sample.chosen_action)];
        const float max_mass =
            *std::max_element(sample.search_policy.begin(), sample.search_policy.end());
        ++chosen_valid;
        chosen_mass_sum += static_cast<double>(chosen_mass);
        if (chosen_mass + 1e-6f >= max_mass) ++chosen_at_visit_max;
    }
    if (chosen_valid > 0) {
        std::fprintf(stderr,
                     "search-target alignment: chosen_at_visit_max=%zu/%zu (%.2f%%), "
                     "mean_chosen_mass=%.6f\n",
                     chosen_at_visit_max, chosen_valid,
                     100.0 * static_cast<double>(chosen_at_visit_max) /
                         static_cast<double>(chosen_valid),
                     chosen_mass_sum / static_cast<double>(chosen_valid));
    }

    // GPU self-play records both player perspectives while each side changes
    // the other's incoming-garbage state. Compact Replay+ metadata does not
    // contain that opponent event stream, so its standalone reconstruction
    // cannot reproduce these observations reliably. Keep the already
    // tokenized samples in the rectangular v1 format for this path.
    if (!export_buffer(path, buffer, model_version, 0, 0, false))
        throw std::runtime_error("GPU self-play dataset export failed");
    gpu_protocol::write_exact(stdout, gpu_protocol::EXPORT_MAGIC,
                              sizeof(gpu_protocol::EXPORT_MAGIC));
    gpu_protocol::write_u32(stdout, static_cast<std::uint32_t>(std::max(1, games)));
    gpu_protocol::write_u32(stdout, static_cast<std::uint32_t>(buffer.size()));
    gpu_protocol::write_u64(stdout, ev.positions_evaluated());
    gpu_protocol::write_u64(stdout, ev.batches_issued());
    if (std::fflush(stdout) != 0) throw std::runtime_error("GPU export flush failed");
    return 0;
}

void write_json_float_array(std::ofstream& out, const float* values, int count) {
    out << '[';
    for (int i = 0; i < count; ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

void write_json_float_array(std::ofstream& out, const std::vector<float>& values) {
    write_json_float_array(out, values.data(), static_cast<int>(values.size()));
}

// Reconstruct historical roots from v4 provenance, rank them with a cheap
// current-network pass, and spend full search only on the most stale rows.
int cmd_gpu_reanalyse_protocol(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
                     "usage: gpu-reanalyse-protocol <input> <output> <audit.jsonl> "
                     "<select-count> [sims] [batch] [determinizations] [gumbel] "
                     "[no-attack-delivery] [timing-actions] [teacher-model-version]\n");
        return 1;
    }
    const std::string input = argv[2];
    const std::string output = argv[3];
    const std::string audit_path = argv[4];
    const int requested = std::max(1, std::atoi(argv[5]));
    const int sims = (argc > 6) ? std::max(1, std::atoi(argv[6])) : 128;
    const int batch = (argc > 7) ? std::max(1, std::atoi(argv[7])) : 16;
    const int determinizations = (argc > 8) ? std::max(1, std::atoi(argv[8])) : 2;
    const bool use_gumbel = (argc > 9) ? std::atoi(argv[9]) != 0 : true;
    const bool no_attack_delivery = (argc > 10) ? std::atoi(argv[10]) != 0 : false;
    const bool timing_actions = (argc > 11) ? std::atoi(argv[11]) != 0 : false;
    const std::uint32_t teacher_model_version =
        (argc > 12) ? static_cast<std::uint32_t>(std::strtoul(argv[12], nullptr, 10)) : 0;
    enable_gpu_protocol_stdio();

    const DatasetReadResult source = read_dataset_file(input);
    if (!source.ok) throw std::runtime_error("cannot read source dataset: " + source.error);
    if (source.header.version != DatasetHeader::VERSION)
        throw std::runtime_error("reanalyse requires a rectangular v4 dataset with chosen actions");
    bool rules_found = false;
    const RulesetConfig rules = ruleset_from_hash(source.header.ruleset_hash, &rules_found);
    if (!rules_found)
        throw std::runtime_error("source ruleset hash is not a known preset");

    const bool deliver_attacks = !no_attack_delivery;
    ReconstructionReport reconstructed = reconstruct_historical_roots(
        source.batch, rules, deliver_attacks, timing_actions);
    if (!reconstructed.ok())
        throw std::runtime_error("historical-state reconstruction failed: " + reconstructed.error);

    RemoteGpuEvaluator ev(stdin, stdout, batch, 0);
    struct Candidate {
        std::size_t root_index = 0;
        double score = 0.0;
        Evaluation raw;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(reconstructed.roots.size());
    for (std::size_t begin = 0; begin < reconstructed.roots.size();
         begin += static_cast<std::size_t>(batch)) {
        const std::size_t end = std::min(reconstructed.roots.size(),
                                         begin + static_cast<std::size_t>(batch));
        std::vector<EvalRequest> requests;
        requests.reserve(end - begin);
        for (std::size_t i = begin; i < end; ++i) {
            ReanalyseRoot& root = reconstructed.roots[i];
            requests.push_back(EvalRequest{&root.observation, &root.actions});
        }
        std::vector<Evaluation> evaluations;
        ev.evaluate(requests, evaluations);
        if (evaluations.size() != requests.size())
            throw std::runtime_error("teacher returned wrong cheap-screen batch size");
        for (std::size_t i = begin; i < end; ++i) {
            const std::size_t local = i - begin;
            const ReanalyseRoot& root = reconstructed.roots[i];
            const float* old_policy = source.batch.policy_target.data() +
                root.source_row * static_cast<std::size_t>(source.batch.max_actions);
            const double score = policy_kl(old_policy, evaluations[local].policy,
                                           static_cast<int>(root.actions.size()));
            candidates.push_back(Candidate{i, score, std::move(evaluations[local])});
        }
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Candidate& a, const Candidate& b) {
                         if (a.score != b.score) return a.score > b.score;
                         return a.root_index < b.root_index;
                     });
    candidates.resize(std::min(candidates.size(), static_cast<std::size_t>(requested)));

    SearchConfig search_cfg;
    search_cfg.simulations = sims;
    search_cfg.max_depth = 6;
    search_cfg.use_gumbel = use_gumbel;
    search_cfg.batch_size = batch;
    search_cfg.root_noise_fraction = 0.0f;
    search_cfg.root_noise_alpha = 0.3f;
    search_cfg.determinizations = determinizations;
    search_cfg.enable_timing_actions = timing_actions;
    Searcher searcher(ev, search_cfg);
    std::vector<std::size_t> selected_rows;
    std::vector<std::vector<float>> refreshed_policies;
    selected_rows.reserve(candidates.size());
    refreshed_policies.reserve(candidates.size());
    std::ofstream audit(audit_path, std::ios::binary | std::ios::trunc);
    if (!audit) throw std::runtime_error("cannot open reanalyse audit output");
    audit.precision(9);
    for (const Candidate& candidate : candidates) {
        const ReanalyseRoot& root = reconstructed.roots[candidate.root_index];
        search_cfg.seed = source.batch.game_seed[root.source_row] * 0x9E3779B97F4A7C15ull +
                          source.batch.move_number[root.source_row];
        searcher.set_config(search_cfg);
        const SearchResult result = searcher.search(root.active, &root.inactive, &ev,
                                                    deliver_attacks);
        if (result.search_policy.size() != root.actions.size())
            throw std::runtime_error("reanalysis search returned wrong policy width");
        selected_rows.push_back(root.source_row);
        refreshed_policies.push_back(result.search_policy);

        const int action_count = static_cast<int>(root.actions.size());
        const float* old_policy = source.batch.policy_target.data() +
            root.source_row * static_cast<std::size_t>(source.batch.max_actions);
        audit << "{\"source_row\":" << root.source_row
              << ",\"game_seed\":" << source.batch.game_seed[root.source_row]
              << ",\"move_number\":" << source.batch.move_number[root.source_row]
              << ",\"player_perspective\":" << source.batch.player_perspective[root.source_row]
              << ",\"selection_score_policy_kl\":" << candidate.score
              << ",\"historical_chosen_action\":" << source.batch.chosen_action[root.source_row]
              << ",\"reanalysis_best_action\":" << result.best_action
              << ",\"terminal_value_target\":" << source.batch.value_target[root.source_row]
              << ",\"current_raw_value\":" << candidate.raw.value.scalar()
              << ",\"reanalysis_search_value\":" << result.value.scalar()
              << ",\"historical_policy\":";
        write_json_float_array(audit, old_policy, action_count);
        audit << ",\"current_raw_policy\":";
        write_json_float_array(audit, candidate.raw.policy);
        audit << ",\"reanalysis_policy\":";
        write_json_float_array(audit, result.search_policy);
        audit << "}\n";
    }
    audit.close();

    TensorBatch refreshed = select_reanalysed_rows(source.batch, selected_rows,
                                                    refreshed_policies);
    DatasetContract contract;
    contract.contract_version = source.header.contract_version;
    contract.tokenizer_schema_version = source.header.tokenizer_schema_version;
    contract.tokenizer_schema_hash = source.header.tokenizer_schema_hash;
    contract.observation_schema_hash = source.header.observation_schema_hash;
    contract.action_schema_version = source.header.action_schema_version;
    contract.aux_target_schema_version = source.header.aux_target_schema_version;
    contract.randomizer_type = source.header.randomizer_type;
    contract.termination_reason = static_cast<TerminationReason>(source.header.termination_reason);
    contract.self_play_seed = source.header.self_play_seed;
    contract.token_kind_order_hash = source.header.token_kind_order_hash;
    const std::uint32_t output_model_version = teacher_model_version > 0
        ? teacher_model_version : source.header.model_version;
    if (!write_dataset_file(output, refreshed, source.header.ruleset_hash,
                            output_model_version, contract))
        throw std::runtime_error("cannot write reanalysed dataset");

    gpu_protocol::write_exact(stdout, "RANL", 4);
    gpu_protocol::write_u32(stdout, static_cast<std::uint32_t>(source.batch.batch));
    gpu_protocol::write_u32(stdout, static_cast<std::uint32_t>(selected_rows.size()));
    gpu_protocol::write_u32(stdout, static_cast<std::uint32_t>(reconstructed.token_rows_verified));
    gpu_protocol::write_u32(stdout, static_cast<std::uint32_t>(reconstructed.action_rows_verified));
    gpu_protocol::write_u64(stdout, ev.positions_evaluated());
    gpu_protocol::write_u64(stdout, ev.batches_issued());
    if (std::fflush(stdout) != 0) throw std::runtime_error("reanalyse protocol flush failed");
    return 0;
}

// GPU-backed Arena.  The C++ side still owns the paired-game protocol and
// simulator; Python only supplies candidate (model 0) and champion (model 1)
// evaluations through the same GPU bridge.
int cmd_gpu_arena_protocol(int argc, char** argv) {
    const int pairs = (argc > 2) ? std::atoi(argv[2]) : 10;
    const int sims = (argc > 3) ? std::atoi(argv[3]) : 16;
    const int pieces = (argc > 4) ? std::atoi(argv[4]) : 300;
    const int batch = (argc > 5) ? std::atoi(argv[5]) : 16;
    const int determinizations = (argc > 6) ? std::atoi(argv[6]) : 1;
    const bool use_gumbel = (argc > 7) ? std::atoi(argv[7]) != 0 : false;
    const std::uint64_t base_seed =
        (argc > 8) ? std::strtoull(argv[8], nullptr, 10) : 42;
    const int candidate_sims = (argc > 9) ? std::atoi(argv[9]) : -1;
    const int champion_sims = (argc > 10) ? std::atoi(argv[10]) : -1;
    const int candidate_gumbel = (argc > 11) ? std::atoi(argv[11]) : -1;
    const int champion_gumbel = (argc > 12) ? std::atoi(argv[12]) : -1;
    const float gumbel_c_scale = (argc > 13) ? std::strtof(argv[13], nullptr) : 0.01f;
    const float gumbel_noise_scale = (argc > 14) ? std::strtof(argv[14], nullptr) : 0.05f;
    const int candidate_timing_actions = (argc > 15) ? std::atoi(argv[15]) : -1;
    const int champion_timing_actions = (argc > 16) ? std::atoi(argv[16]) : -1;
    const float candidate_gumbel_noise_scale =
        (argc > 17) ? std::strtof(argv[17], nullptr) : -1.0f;
    const float champion_gumbel_noise_scale =
        (argc > 18) ? std::strtof(argv[18], nullptr) : -1.0f;
    const float candidate_time_budget_ms =
        (argc > 19) ? std::strtof(argv[19], nullptr) : -1.0f;
    const float champion_time_budget_ms =
        (argc > 20) ? std::strtof(argv[20], nullptr) : -1.0f;
    const int candidate_node_budget = (argc > 21) ? std::atoi(argv[21]) : -1;
    const int champion_node_budget = (argc > 22) ? std::atoi(argv[22]) : -1;
    const int garbage_style = (argc > 23) ? std::atoi(argv[23]) : -1;
    const int garbage_period = (argc > 24) ? std::atoi(argv[24]) : 8;
    const int garbage_lines = (argc > 25) ? std::atoi(argv[25]) : 2;
    enable_gpu_protocol_stdio();

    RemoteGpuEvaluator candidate(stdin, stdout, std::max(1, batch), 0);
    RemoteGpuEvaluator champion(stdin, stdout, std::max(1, batch), 1);

    ArenaConfig cfg;
    cfg.pairs = std::max(1, pairs);
    cfg.max_pieces = std::max(1, pieces);
    cfg.search.simulations = std::max(0, sims);
    cfg.candidate_simulations = candidate_sims;
    cfg.champion_simulations = champion_sims;
    cfg.candidate_time_budget_ms = candidate_time_budget_ms;
    cfg.champion_time_budget_ms = champion_time_budget_ms;
    cfg.candidate_node_budget = candidate_node_budget;
    cfg.champion_node_budget = champion_node_budget;
    cfg.candidate_gumbel = candidate_gumbel;
    cfg.champion_gumbel = champion_gumbel;
    cfg.candidate_gumbel_noise_scale = candidate_gumbel_noise_scale;
    cfg.champion_gumbel_noise_scale = champion_gumbel_noise_scale;
    cfg.candidate_timing_actions = candidate_timing_actions;
    cfg.champion_timing_actions = champion_timing_actions;
    cfg.search.max_depth = 6;
    cfg.search.use_gumbel = use_gumbel;
    cfg.search.gumbel_c_scale = std::max(0.0f, gumbel_c_scale);
    cfg.search.gumbel_noise_scale = std::max(0.0f, gumbel_noise_scale);
    cfg.search.batch_size = std::max(1, batch);
    cfg.search.determinizations = std::max(1, determinizations);
    if (garbage_style >= 0 && garbage_style <= static_cast<int>(GarbageStyle::Scripted))
        cfg.garbage_style = static_cast<GarbageStyle>(garbage_style);
    cfg.garbage_period = std::max(1, garbage_period);
    cfg.garbage_lines = std::max(0, garbage_lines);

    Arena arena(candidate, champion, cfg);
    const ArenaResult r = arena.evaluate(RulesetConfig::tetra_league(), base_seed);
    write_gpu_arena_result(
        stdout, static_cast<std::uint32_t>(r.games_played),
        static_cast<std::uint32_t>(r.candidate_wins),
        static_cast<std::uint32_t>(r.champion_wins), static_cast<std::uint32_t>(r.draws),
        r.win_rate, r.ci_lower, r.ci_upper, r.candidate_vs, r.champion_vs,
        r.candidate_apm, r.champion_apm, r.candidate_app, r.champion_app,
        r.candidate_pps, r.champion_pps,
        r.candidate_avg_pieces, r.champion_avg_pieces,
        r.candidate_avg_seconds, r.champion_avg_seconds,
        r.candidate_survival_rate, r.champion_survival_rate,
        r.candidate_sent_per_game, r.champion_sent_per_game,
        r.candidate_garbage_cleared_per_game, r.champion_garbage_cleared_per_game,
        r.candidate_received_per_game, r.champion_received_per_game,
        r.candidate_blockout_rate, r.champion_blockout_rate,
        r.candidate_lockout_rate, r.champion_lockout_rate,
        r.candidate_garbageout_rate, r.champion_garbageout_rate, r.promoted);
    auto write_diagnostics = [](const char* name, const SearchSideDiagnostics& d,
                                float budget) {
        std::fprintf(stderr,
                     "\"%s\":{\"budget_ms\":%.9g,\"decisions\":%llu,"
                     "\"simulations\":%llu,\"nodes\":%llu,\"evaluator_calls\":%llu,"
                     "\"positions_evaluated\":%llu,\"evaluation_flushes\":%llu,"
                     "\"node_budget_cutoffs\":%llu,\"time_budget_exhaustions\":%llu,"
                     "\"raw_policy_matches\":%llu,\"searched_action_changes\":%llu,"
                     "\"elapsed_ms\":%.9g,\"overshoot_ms\":%.9g,"
                     "\"evaluator_elapsed_ms\":%.9g,\"depth_sum\":%.9g,"
                     "\"root_setup_us\":%.9g,\"gather_us\":%.9g,"
                     "\"backup_us\":%.9g,\"finalize_us\":%.9g,"
                     "\"node_allocation_us\":%.9g,"
                     "\"legal_action_generation_us\":%.9g,"
                     "\"state_transition_us\":%.9g,\"selection_us\":%.9g,"
                     "\"depth_samples\":%llu,\"max_depth\":%d,"
                     "\"legal_actions\":%.9g,\"root_total_visits\":%.9g,"
                     "\"root_top1_visit_share\":%.9g,\"root_top1_q\":%.9g,"
                     "\"root_visit_entropy\":%.9g,\"decision_latencies_ms\":[",
                     name, budget,
                     static_cast<unsigned long long>(d.decisions),
                     static_cast<unsigned long long>(d.simulations),
                     static_cast<unsigned long long>(d.nodes),
                     static_cast<unsigned long long>(d.evaluator_calls),
                     static_cast<unsigned long long>(d.positions_evaluated),
                     static_cast<unsigned long long>(d.evaluation_flushes),
                     static_cast<unsigned long long>(d.node_budget_cutoffs),
                     static_cast<unsigned long long>(d.time_budget_exhaustions),
                     static_cast<unsigned long long>(d.raw_policy_matches),
                     static_cast<unsigned long long>(d.searched_action_changes),
                     d.elapsed_ms, d.overshoot_ms, d.evaluator_elapsed_ms, d.depth_sum,
                     d.root_setup_us, d.gather_us, d.backup_us, d.finalize_us,
                     d.node_allocation_us, d.legal_action_generation_us,
                     d.state_transition_us, d.selection_us,
                     static_cast<unsigned long long>(d.depth_samples), d.max_depth,
                     d.legal_actions, d.root_total_visits, d.root_top1_visit_share,
                     d.root_top1_q, d.root_visit_entropy);
        for (size_t i = 0; i < d.decision_latencies_ms.size(); ++i) {
            if (i != 0) std::fputc(',', stderr);
            std::fprintf(stderr, "%.9g", d.decision_latencies_ms[i]);
        }
        std::fputs("]}", stderr);
    };
    std::fputs("{\"arena_diagnostics\":{", stderr);
    write_diagnostics("candidate", r.diagnostics.candidate, candidate_time_budget_ms);
    std::fputc(',', stderr);
    write_diagnostics("champion", r.diagnostics.champion, champion_time_budget_ms);
    std::fprintf(stderr, "},\"garbage_style\":%d,\"garbage_period\":%d,\"garbage_lines\":%d}\n",
                 garbage_style >= 0 ? garbage_style : static_cast<int>(cfg.garbage_style),
                 cfg.garbage_period, cfg.garbage_lines);
    return 0;
}

int cmd_arena(int argc, char** argv) {
    if (argc < 3) {
        std::printf(
            "usage: tetra_cli arena <candidate> [champion=heuristic] [pairs=10] [sims=16] "
            "[pieces=300]\n");
        return 1;
    }
    const std::string cand_name = argv[2];
    const std::string champ_name = (argc > 3) ? argv[3] : "heuristic";
    const int pairs = (argc > 4) ? std::atoi(argv[4]) : 10;
    const int sims = (argc > 5) ? std::atoi(argv[5]) : 16;
    const int pieces = (argc > 6) ? std::atoi(argv[6]) : 300;

    auto cand = load_evaluator_from_name(cand_name);
    auto champ = load_evaluator_from_name(champ_name);
    if (!cand || !champ) return 1;

    ArenaConfig cfg;
    cfg.pairs = pairs;
    cfg.max_pieces = pieces;
    cfg.search.simulations = sims;
    cfg.search.max_depth = 6;
    cfg.search.use_gumbel = true;
    cfg.search.batch_size = 16;
    cfg.search.determinizations = 2;

    std::printf("Arena: Candidate (%s) vs Champion (%s)\n", cand_name.c_str(),
                champ_name.c_str());
    std::printf("Running %d paired games (%d games total, sims=%d, max_pieces=%d)...\n\n",
                pairs, pairs * 2, sims, pieces);

    Arena arena(*cand, *champ, cfg);
    const auto t0 = std::chrono::steady_clock::now();
    const ArenaResult r = arena.evaluate(RulesetConfig::tetra_league(), 42);
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::printf("%4s  %8s  %7s  %7s  %7s  %6s  %6s  %7s  %7s  %7s  %6s  %6s  %7s  %5s\n",
                "pair", "seed", "c_pcs", "c_sent", "c_vs", "c_apm", "c_app", "h_pcs", "h_sent",
                "h_vs", "h_apm", "h_app", "mirror", "score");
    for (size_t i = 0; i < r.games.size(); ++i) {
        const auto& g = r.games[i];
        const RulesetConfig stats_rules = RulesetConfig::tetra_league();
        std::printf("%4d  %016llx  %7d  %7lld  %7.1f  %6.1f  %6.3f  %7d  %7lld  %7.1f  %6.1f  %6.3f  %7s  %5.1f\n",
                    g.pair_index, static_cast<unsigned long long>(g.seed), g.candidate_pieces,
                    static_cast<long long>(g.candidate_sent),
                    versus_score(g.candidate_sent, g.candidate_garbage_cleared,
                                 g.candidate_pieces, g.candidate_duration, stats_rules),
                    attacks_per_minute(g.candidate_sent, g.candidate_duration, stats_rules),
                    attacks_per_piece(g.candidate_sent, g.candidate_pieces),
                    g.champion_pieces, static_cast<long long>(g.champion_sent),
                    versus_score(g.champion_sent, g.champion_garbage_cleared,
                                 g.champion_pieces, g.champion_duration, stats_rules),
                    attacks_per_minute(g.champion_sent, g.champion_duration, stats_rules),
                    attacks_per_piece(g.champion_sent, g.champion_pieces),
                    g.is_mirrored ? "yes" : "no", static_cast<double>(g.candidate_score));
    }

    std::printf("\nResult over %d games in %.1f s:\n", r.games_played, secs);
    std::printf("  Candidate wins : %d\n", r.candidate_wins);
    std::printf("  Champion wins  : %d\n", r.champion_wins);
    std::printf("  Draws          : %d\n", r.draws);
    std::printf("  VS             : %.1f / %.1f (candidate/champion)\n",
                r.candidate_vs, r.champion_vs);
    std::printf("  Win Rate       : %.1f%% (95%% CI: %.1f%% - %.1f%%)\n", r.win_rate * 100.0f,
                r.ci_lower * 100.0f, r.ci_upper * 100.0f);
    std::printf("  Threshold      : %.1f%%\n", cfg.promotion_threshold * 100.0f);
    std::printf(
        "  Status         : %s\n",
        r.promoted ? "\033[32mPROMOTED\033[0m" : "\033[33mRETAINED (no promotion)\033[0m");
    return 0;
}

// Diagnostic-only local evaluator control.  This keeps the C++ Arena/search
// path intact while removing both Python and GPU work.  Its output is not a
// strength result; it is only a nodes/ms ceiling measurement.
int cmd_arena_diagnostic(int argc, char** argv) {
    if (argc < 3) {
        std::printf(
            "usage: tetra_cli arena-diagnostic <uniform|heuristic> "
            "[champion=uniform] [pairs=1] [sims=100000] [pieces=20] "
            "[budget_ms=40] [garbage_style=1] [seed=42]\n");
        return 1;
    }
    const std::string candidate_name = argv[2];
    const std::string champion_name = (argc > 3) ? argv[3] : "uniform";
    const int pairs = (argc > 4) ? std::atoi(argv[4]) : 1;
    const int sims = (argc > 5) ? std::atoi(argv[5]) : 100000;
    const int pieces = (argc > 6) ? std::atoi(argv[6]) : 20;
    const float budget_ms = (argc > 7) ? std::strtof(argv[7], nullptr) : 40.0f;
    const int garbage_style = (argc > 8) ? std::atoi(argv[8]) : 1;
    const std::uint64_t seed = (argc > 9) ? std::strtoull(argv[9], nullptr, 10) : 42;

    auto candidate = load_evaluator_from_name(candidate_name);
    auto champion = load_evaluator_from_name(champion_name);
    if (!candidate || !champion) return 1;

    ArenaConfig cfg;
    cfg.pairs = std::max(1, pairs);
    cfg.max_pieces = std::max(1, pieces);
    cfg.search.simulations = std::max(0, sims);
    cfg.search.time_budget_ms = std::max(0.0f, budget_ms);
    cfg.search.max_depth = 6;
    cfg.search.batch_size = 16;
    cfg.search.use_gumbel = true;
    cfg.search.determinizations = 1;
    if (garbage_style >= 0 && garbage_style <= static_cast<int>(GarbageStyle::Scripted))
        cfg.garbage_style = static_cast<GarbageStyle>(garbage_style);

    Arena arena(*candidate, *champion, cfg);
    const ArenaResult result = arena.evaluate(RulesetConfig::tetra_league(), seed);
    auto write_side = [](const SearchSideDiagnostics& d) {
        std::printf(
            "{\"decisions\":%llu,\"nodes\":%llu,\"positions_evaluated\":%llu,"
            "\"evaluator_calls\":%llu,\"elapsed_ms\":%.9g,"
            "\"evaluator_elapsed_ms\":%.9g,\"overshoot_ms\":%.9g,"
            "\"root_setup_us\":%.9g,\"gather_us\":%.9g,"
            "\"backup_us\":%.9g,\"finalize_us\":%.9g,"
            "\"node_allocation_us\":%.9g,\"legal_action_generation_us\":%.9g,"
            "\"state_transition_us\":%.9g,\"selection_us\":%.9g,"
            "\"max_depth\":%d}",
            static_cast<unsigned long long>(d.decisions),
            static_cast<unsigned long long>(d.nodes),
            static_cast<unsigned long long>(d.positions_evaluated),
            static_cast<unsigned long long>(d.evaluator_calls),
            d.elapsed_ms, d.evaluator_elapsed_ms, d.overshoot_ms,
            d.root_setup_us, d.gather_us, d.backup_us, d.finalize_us,
            d.node_allocation_us, d.legal_action_generation_us,
            d.state_transition_us, d.selection_us, d.max_depth);
    };
    std::printf(
        "{\"local_dummy_diagnostics\":{\"candidate_evaluator\":\"%s\","
        "\"champion_evaluator\":\"%s\",\"garbage_style\":%d,"
        "\"budget_ms\":%.9g,\"games\":%d,\"candidate\":",
        candidate_name.c_str(), champion_name.c_str(), garbage_style, budget_ms,
        result.games_played);
    write_side(result.diagnostics.candidate);
    std::printf(",\"champion\":");
    write_side(result.diagnostics.champion);
    std::printf("}}\n");
    return 0;
}

int cmd_import_human_replay(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: import-human-replay <normalized.replay> <output.tetradat> "
                     "[model_version] [league|quickplay|guideline]\n");
        return 2;
    }
    const std::uint32_t model_version = argc > 4
        ? static_cast<std::uint32_t>(std::strtoul(argv[4], nullptr, 10))
        : 0u;
    const RulesetConfig rules = preset(argc > 5 ? argv[5] : "league");
    HumanReplayImportStats stats;
    std::string error;
    if (!import_human_replay_protocol(argv[2], argv[3], rules, model_version,
                                      &stats, &error)) {
        std::fprintf(stderr, "human replay import failed: %s\n", error.c_str());
        std::fprintf(stderr,
                     "  games=%zu turns=%zu imported=%zu invalid=%zu execution=%zu unmatched=%zu\n",
                     stats.games, stats.turns, stats.imported,
                     stats.skipped_invalid_state, stats.skipped_execution,
                     stats.skipped_no_legal_match);
        return 1;
    }
    std::printf(
        "human replay import: games=%zu turns=%zu imported=%zu invalid=%zu execution=%zu unmatched=%zu\n",
        stats.games, stats.turns, stats.imported,
        stats.skipped_invalid_state, stats.skipped_execution,
        stats.skipped_no_legal_match);
    std::printf("wrote %s under ruleset %s (%s), model_version=%u\n",
                argv[3], rules.id.c_str(), rules.hash_hex().c_str(), model_version);
    return 0;
}

int cmd_decode_dataset(int argc, char** argv) {
    if (argc < 3) {
        std::printf(
            "usage: tetra_cli decode-dataset <input.tetradat> [output.tetradat]\n");
        return 1;
    }
    const std::string in_path = argv[2];
    const DatasetReadResult r = read_dataset_file(in_path);
    if (!r.ok) {
        std::fprintf(stderr, "cannot decode %s: %s\n", in_path.c_str(), r.error.c_str());
        return 1;
    }
    const std::vector<std::uint8_t> v1_bytes =
        serialize_dataset(r.batch, r.header.ruleset_hash, r.header.model_version);

    if (argc > 3) {
        const std::string out_path = argv[3];
        std::FILE* f = std::fopen(out_path.c_str(), "wb");
        if (!f) {
            std::fprintf(stderr, "cannot open %s for write\n", out_path.c_str());
            return 1;
        }
        std::fwrite(v1_bytes.data(), 1, v1_bytes.size(), f);
        std::fclose(f);
        std::printf("decoded %s -> %s (%u samples, %zu bytes)\n", in_path.c_str(),
                    out_path.c_str(), r.header.samples, v1_bytes.size());
        return 0;
    }

    // Python consumes this branch as a binary pipe.  Windows otherwise
    // translates every 0x0a byte to CRLF and silently corrupts the tensors.
    enable_gpu_protocol_stdio();
    const size_t nw = std::fwrite(v1_bytes.data(), 1, v1_bytes.size(), stdout);
    std::fflush(stdout);
    if (nw != v1_bytes.size()) return 1;
    return 0;
}

void usage() {
    std::printf(
        "tetra_cli -- TetraFormer M0/M1 developer tool (local simulator only)\n\n"
        "  ruleset [league|quickplay|guideline]\n"
        "  moves <piece> [seed]\n"
        "  selfplay [seed] [pieces]\n"
        "  timing <piece>\n"
        "  record [file] [seed] [pieces]\n"
        "  verify <file>\n"
        "  search [sims] [gumbel:0|1] [seed]\n"
        "  selfplay-gen [games] [pieces] [sims]\n"
        "  export <file> [games] [pieces] [sims] [--weights ...]\n"
        "  play <weights.tetrawts|--weights=...> [pieces] [sims]\n"
        "  gpu-play-protocol [pieces] [sims] [batch] [seed] [determinizations] [gumbel]\n"
        "  gpu-export-protocol <file> [games] [pieces] [sims] [batch] [seed] [model_version] "
        "[determinizations] [gumbel]\n"
        "  gpu-reanalyse-protocol <input> <output> <audit.jsonl> <select-count> "
        "[sims] [batch] [determinizations] [gumbel] [no-attack] [timing] [model-version]\n"
        "  gpu-arena-protocol [pairs] [sims] [pieces] [batch] [determinizations] [gumbel] [seed] "
        "[candidate_sims] [champion_sims] [candidate_gumbel] [champion_gumbel] "
        "[gumbel_c_scale] [gumbel_noise_scale] [candidate_timing] [champion_timing] "
        "[candidate_gumbel_noise_scale] [champion_gumbel_noise_scale] "
        "[candidate_time_ms] [champion_time_ms] [candidate_node_budget] "
        "[champion_node_budget] [garbage_style] [garbage_period] [garbage_lines]\n"
        "  arena <candidate> [champion=heuristic] [pairs] [sims] [pieces]\n"
        "  arena-diagnostic <uniform|heuristic> [champion=uniform] [pairs] [sims] [pieces] "
        "[budget_ms] [garbage_style] [seed]\n"
        "  import-human-replay <normalized.replay> <output.tetradat> [model_version] "
        "[league|quickplay|guideline]\n"
        "  decode-dataset <input.tetradat> [output.tetradat]\n"
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
    if (cmd == "timing") return cmd_timing(argc, argv);
    if (cmd == "record") return cmd_record(argc, argv);
    if (cmd == "verify") return cmd_verify(argc, argv);
    if (cmd == "search") return cmd_search(argc, argv);
    if (cmd == "selfplay-gen") return cmd_selfplay_gen(argc, argv);
    if (cmd == "export") return cmd_export(argc, argv);
    if (cmd == "play") return cmd_play(argc, argv);
    if (cmd == "gpu-play-protocol") return cmd_gpu_play_protocol(argc, argv);
    if (cmd == "gpu-export-protocol") return cmd_gpu_export_protocol(argc, argv);
    if (cmd == "gpu-reanalyse-protocol") return cmd_gpu_reanalyse_protocol(argc, argv);
    if (cmd == "gpu-arena-protocol") return cmd_gpu_arena_protocol(argc, argv);
    if (cmd == "arena") return cmd_arena(argc, argv);
    if (cmd == "arena-diagnostic") return cmd_arena_diagnostic(argc, argv);
    if (cmd == "import-human-replay") return cmd_import_human_replay(argc, argv);
    if (cmd == "decode-dataset") return cmd_decode_dataset(argc, argv);
    if (cmd == "bench") return cmd_bench(argc, argv);
    if (cmd == "determinism") return cmd_determinism(argc, argv);
    usage();
    return 1;
}
