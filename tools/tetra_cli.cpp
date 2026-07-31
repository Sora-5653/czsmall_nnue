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

#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <memory>
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
    cfg.garbage_style = GarbageStyle::Steady;
    cfg.garbage_period = 8;
    cfg.garbage_lines = 2;

    SelfPlayWorker worker(*ev, cfg);
    ReplayBuffer buffer(200000);

    std::printf("ruleset %s (%s), %d games x %d pieces, %d sims\n\n", rules.id.c_str(),
                rules.hash_hex().c_str(), games, pieces, sims);
    std::printf("%5s %8s %8s %7s %7s %8s %s\n", "game", "pieces", "cleared", "sent", "recv",
                "outcome", "result");

    int total_pieces = 0, survived = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < games; ++g) {
        SelfPlayStats st;
        auto samples = worker.play(rules, static_cast<std::uint64_t>(g), &st);
        total_pieces += st.pieces;
        if (st.survived) ++survived;
        std::printf("%5d %8d %8d %7d %7d %+8.0f %s\n", g, st.pieces, st.lines_cleared,
                    st.lines_sent, st.lines_received, static_cast<double>(st.outcome),
                    st.survived ? "truncated" : topout_name(st.topout));
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
    bool compact = true;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--v1" || std::string(argv[i]) == "--compact=0")
            compact = false;
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
    cfg.garbage_style = GarbageStyle::Steady;
    cfg.garbage_period = 8;
    cfg.garbage_lines = 2;

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
    std::printf("result     %s\n", st.survived ? "survived" : topout_name(st.topout));
    std::printf("speed      %.1f placements/s\n", secs > 0 ? st.pieces / secs : 0.0);
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

    std::printf("Arena: Candidate (%s) vs Champion (%s)\n", cand_name.c_str(),
                champ_name.c_str());
    std::printf("Running %d paired games (%d games total, sims=%d, max_pieces=%d)...\n\n",
                pairs, pairs * 2, sims, pieces);

    Arena arena(*cand, *champ, cfg);
    const auto t0 = std::chrono::steady_clock::now();
    const ArenaResult r = arena.evaluate(RulesetConfig::tetra_league(), 42);
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::printf("%4s  %8s  %7s  %7s  %8s  %7s  %7s  %5s\n", "pair", "seed", "c_pcs", "c_sent",
                "h_pcs", "h_sent", "mirror", "score");
    for (size_t i = 0; i < r.games.size(); ++i) {
        const auto& g = r.games[i];
        std::printf("%4d  %016llx  %7d  %7lld  %7d  %7lld  %7s  %5.1f\n", g.pair_index,
                    static_cast<unsigned long long>(g.seed), g.candidate_pieces,
                    static_cast<long long>(g.candidate_sent), g.champion_pieces,
                    static_cast<long long>(g.champion_sent), g.is_mirrored ? "yes" : "no",
                    static_cast<double>(g.candidate_score));
    }

    std::printf("\nResult over %d games in %.1f s:\n", r.games_played, secs);
    std::printf("  Candidate wins : %d\n", r.candidate_wins);
    std::printf("  Champion wins  : %d\n", r.champion_wins);
    std::printf("  Draws          : %d\n", r.draws);
    std::printf("  Win Rate       : %.1f%% (95%% CI: %.1f%% - %.1f%%)\n", r.win_rate * 100.0f,
                r.ci_lower * 100.0f, r.ci_upper * 100.0f);
    std::printf("  Threshold      : %.1f%%\n", cfg.promotion_threshold * 100.0f);
    std::printf(
        "  Status         : %s\n",
        r.promoted ? "\033[32mPROMOTED\033[0m" : "\033[33mRETAINED (no promotion)\033[0m");
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
        "  arena <candidate> [champion=heuristic] [pairs] [sims] [pieces]\n"
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
    if (cmd == "arena") return cmd_arena(argc, argv);
    if (cmd == "decode-dataset") return cmd_decode_dataset(argc, argv);
    if (cmd == "bench") return cmd_bench(argc, argv);
    if (cmd == "determinism") return cmd_determinism(argc, argv);
    usage();
    return 1;
}