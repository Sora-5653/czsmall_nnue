// SPDX-License-Identifier: MIT
// Replay serialisation and verification (spec 17 /protocol, 18.1, 22).
#include "test_util.hpp"
#include "tetra/replay.hpp"

#include <cstdio>
#include <string>

using namespace tetra;

namespace {

RulesetConfig league() { return RulesetConfig::tetra_league(); }

// Play a scripted game and return the recorded replay alongside the player.
struct Recorded {
    Replay replay;
    std::uint64_t final_hash = 0;
    std::int64_t sent = 0;
    int placements = 0;
};

Recorded play_and_record(const RulesetConfig& cfg, std::uint64_t seed, int max_pieces,
                         int garbage_every = 0, int garbage_lines = 0,
                         int checkpoint_interval = 16) {
    Player p;
    p.reset(cfg, seed, 0);
    MoveGenerator gen;
    ReplayRecorder rec(cfg, seed, 0, checkpoint_interval);
    rec.note("test");

    Recorded out;
    for (int i = 0; i < max_pieces && p.alive(); ++i) {
        if (garbage_every > 0 && i > 0 && (i % garbage_every) == 0) {
            rec.record_garbage(garbage_lines, p.now(), 1);
            p.receive_attack(garbage_lines, p.now(), 1);
        }

        const auto acts = gen.generate_for_piece(p.board(), p.active().type, cfg, false);
        if (acts.empty()) break;
        // Deterministic policy with some variety, so spins and clears occur.
        size_t best = 0;
        for (size_t k = 1; k < acts.size(); ++k) {
            const auto& a = acts[k];
            const auto& b = acts[best];
            if (a.cleared_lines > b.cleared_lines ||
                (a.cleared_lines == b.cleared_lines && a.final_y < b.final_y))
                best = k;
        }
        p.set_active(acts[best].piece_state());
        int sent = 0;
        const LockResult r = p.lock_piece(acts[best].total_duration(), &sent);
        rec.record(acts[best], p);
        ++out.placements;
        if (!r.ok) break;
    }

    out.replay = rec.finish(p);
    out.final_hash = detail::board_hash(p.board());
    out.sent = p.lines_sent();
    return out;
}

}  // namespace

TEST(replay_round_trips_through_bytes) {
    const RulesetConfig cfg = league();
    const Recorded rec = play_and_record(cfg, 4242, 80);
    CHECK(rec.placements > 20);

    const std::vector<std::uint8_t> bytes = serialize(rec.replay);
    const DeserializeResult rt = deserialize(bytes);
    CHECK_MSG(rt.ok, "deserialize failed: " + rt.error);

    const Replay& a = rec.replay;
    const Replay& b = rt.replay;
    CHECK_EQ(a.ruleset_id, b.ruleset_id);
    CHECK_EQ(a.ruleset_hash, b.ruleset_hash);
    CHECK_EQ(a.seed, b.seed);
    CHECK_EQ(a.placements.size(), b.placements.size());
    CHECK_EQ(a.final_board_hash, b.final_board_hash);
    CHECK_EQ(a.final_lines_sent, b.final_lines_sent);
    CHECK_EQ(a.final_timestamp, b.final_timestamp);
    CHECK(a.final_alive == b.final_alive);

    for (size_t i = 0; i < a.placements.size() && i < b.placements.size(); ++i) {
        const auto& x = a.placements[i];
        const auto& y = b.placements[i];
        CHECK(x.piece == y.piece);
        CHECK_EQ(x.x, y.x);
        CHECK_EQ(x.y, y.y);
        CHECK(x.rot == y.rot);
        CHECK(x.spin == y.spin);
        CHECK(x.use_hold == y.use_hold);
        CHECK_EQ(x.duration, y.duration);
        CHECK(x.has_checkpoint == y.has_checkpoint);
        if (x.has_checkpoint) {
            CHECK_EQ(x.board_hash, y.board_hash);
            CHECK_EQ(x.lines_sent, y.lines_sent);
            CHECK_EQ(x.timestamp, y.timestamp);
        }
    }

    // Serialising the round-tripped replay must be byte-identical.
    CHECK(serialize(rt.replay) == bytes);
}

TEST(replay_verifies_against_the_engine) {
    // The core guarantee: a replay stores only inputs, so re-running it is a
    // genuine test of the simulator rather than a playback of cached results.
    const RulesetConfig cfg = league();
    for (std::uint64_t seed : {1ull, 42ull, 4242ull, 999ull}) {
        const Recorded rec = play_and_record(cfg, seed, 120);
        const VerifyResult v = verify_replay(rec.replay, cfg);
        CHECK_MSG(v.ok, "verification failed: " + v.error);
        CHECK_EQ(v.first_divergence, -1);
        CHECK_EQ(v.placements_applied, static_cast<int>(rec.replay.placements.size()));
    }
}

TEST(replay_with_garbage_verifies) {
    RulesetConfig cfg = league();
    cfg.garbage.travel_time = 0;
    cfg.garbage.activation_delay = 0;
    const Recorded rec = play_and_record(cfg, 77, 120, /*garbage_every=*/4, /*lines=*/2);
    CHECK(!rec.replay.garbage.empty());
    CHECK(rec.replay.final_lines_received > 0);

    const VerifyResult v = verify_replay(rec.replay, cfg);
    CHECK_MSG(v.ok, "verification with garbage failed: " + v.error);

    // This particular stream buries the greedy scripted policy, so the replay
    // also covers the top-out path: a replay that ends in a loss must still
    // verify exactly, including the recorded top-out reason.
    CHECK(!rec.replay.final_alive);
    CHECK(rec.replay.final_topout != TopoutReason::None);

    // A lighter stream that the policy survives must verify too.
    const Recorded light = play_and_record(cfg, 77, 120, /*garbage_every=*/12, /*lines=*/1);
    CHECK(!light.replay.garbage.empty());
    CHECK_MSG(verify_replay(light.replay, cfg).ok, "light garbage stream must verify");
}

TEST(replay_records_spin_provenance) {
    // Regression test. A piece that slid into a notch and one that was rotated
    // into the same notch occupy identical cells but score differently, so the
    // spin class cannot be re-derived from the final position. An earlier
    // version assumed "rotated" on playback and fabricated spins -- and the
    // extra attack that comes with them -- diverging at placement 16 of a
    // 53-placement game.
    const RulesetConfig cfg = league();
    const Recorded rec = play_and_record(cfg, 4242, 80);

    int rotated = 0, dropped = 0;
    for (const auto& p : rec.replay.placements) {
        if (p.spin != SpinType::None) ++rotated;
        else ++dropped;
    }
    CHECK(dropped > 0);

    // Corrupting the recorded spin must be detected, not silently absorbed.
    Replay tampered = rec.replay;
    bool flipped = false;
    for (auto& p : tampered.placements) {
        if (p.spin == SpinType::None && !flipped) {
            p.spin = SpinType::Full;
            flipped = true;
        }
    }
    if (flipped) {
        const VerifyResult v = verify_replay(tampered, cfg);
        CHECK_MSG(!v.ok, "a fabricated spin must be rejected");
        CHECK(v.first_divergence >= 0);
    }
}

TEST(verification_reports_the_first_divergence) {
    // Spec 22 wants replay-diff testing to localise a regression. Corrupt a
    // known placement and check the reported index.
    const RulesetConfig cfg = league();
    const Recorded rec = play_and_record(cfg, 4242, 80, 0, 0, /*checkpoint_interval=*/1);
    CHECK(rec.replay.placements.size() > 30);

    Replay tampered = rec.replay;
    const size_t target = 20;
    tampered.placements[target].x += 1;  // move the piece one column over

    const VerifyResult v = verify_replay(tampered, cfg);
    CHECK(!v.ok);
    CHECK_MSG(v.first_divergence == static_cast<int>(target),
              "expected divergence at " + std::to_string(target) + ", got " +
                  std::to_string(v.first_divergence));
}

TEST(verification_rejects_a_ruleset_hash_mismatch) {
    // A replay must never be silently reinterpreted under different rules.
    const RulesetConfig cfg = league();
    const Recorded rec = play_and_record(cfg, 1, 40);

    RulesetConfig changed = cfg;
    changed.attack.quad = 9;  // any play-affecting change
    CHECK(changed.hash() != cfg.hash());

    const VerifyResult v = verify_replay(rec.replay, changed);
    CHECK(!v.ok);
    CHECK_MSG(v.error.find("ruleset hash mismatch") != std::string::npos,
              "expected a hash mismatch error, got: " + v.error);
}

TEST(verification_detects_a_randomizer_change) {
    // Changing the piece sequence must be caught by the recorded piece types,
    // even before any board state diverges.
    const RulesetConfig cfg = league();
    const Recorded rec = play_and_record(cfg, 4242, 60);

    Replay wrong_seed = rec.replay;
    wrong_seed.seed = rec.replay.seed + 1;
    const VerifyResult v = verify_replay(wrong_seed, cfg);
    CHECK(!v.ok);
    CHECK(v.first_divergence >= 0);
}

TEST(corrupt_bytes_are_rejected) {
    const RulesetConfig cfg = league();
    const Recorded rec = play_and_record(cfg, 5, 40);
    const std::vector<std::uint8_t> good = serialize(rec.replay);

    // Bad magic.
    {
        std::vector<std::uint8_t> b = good;
        b[0] = 'X';
        const auto r = deserialize(b);
        CHECK(!r.ok);
        CHECK(r.error == "bad magic");
    }
    // Truncation.
    for (size_t cut : {size_t(4), size_t(20), good.size() / 2, good.size() - 9}) {
        std::vector<std::uint8_t> b(good.begin(), good.begin() + static_cast<long>(cut));
        const auto r = deserialize(b);
        CHECK_MSG(!r.ok, "truncated input must be rejected at " + std::to_string(cut));
    }
    // A single flipped bit anywhere in the body.
    for (size_t i = 12; i < good.size() - 8; i += 17) {
        std::vector<std::uint8_t> b = good;
        b[i] ^= 0x40;
        const auto r = deserialize(b);
        CHECK_MSG(!r.ok, "bit flip at " + std::to_string(i) + " must fail the checksum");
    }
    // Unsupported version.
    {
        std::vector<std::uint8_t> b = good;
        b[8] = 99;
        // Recompute the checksum so the version check is what rejects it.
        std::uint64_t sum = 1469598103934665603ull;
        for (size_t i = 0; i + 8 < b.size(); ++i) {
            sum ^= b[i];
            sum *= 1099511628211ull;
        }
        for (int k = 0; k < 8; ++k)
            b[b.size() - 8 + static_cast<size_t>(k)] =
                static_cast<std::uint8_t>((sum >> (k * 8)) & 0xFF);
        const auto r = deserialize(b);
        CHECK(!r.ok);
        CHECK_MSG(r.error.find("unsupported version") != std::string::npos,
                  "expected a version error, got: " + r.error);
    }
}

TEST(empty_replay_round_trips) {
    const RulesetConfig cfg = league();
    Player p;
    p.reset(cfg, 1, 0);
    ReplayRecorder rec(cfg, 1, 0);
    const Replay r = rec.finish(p);

    const auto rt = deserialize(serialize(r));
    CHECK(rt.ok);
    CHECK_EQ(static_cast<int>(rt.replay.placements.size()), 0);
    const VerifyResult v = verify_replay(rt.replay, cfg);
    CHECK_MSG(v.ok, "an empty replay should verify: " + v.error);
}

TEST(replay_survives_a_file_round_trip) {
    const RulesetConfig cfg = league();
    const Recorded rec = play_and_record(cfg, 31337, 100);

    const std::string path = "build/test_replay.tetrarep";
    CHECK_MSG(write_replay_file(path, rec.replay), "failed to write the replay file");

    const DeserializeResult rt = read_replay_file(path);
    CHECK_MSG(rt.ok, "failed to read back: " + rt.error);
    CHECK_EQ(rt.replay.placements.size(), rec.replay.placements.size());
    CHECK_EQ(rt.replay.final_board_hash, rec.replay.final_board_hash);

    const VerifyResult v = verify_replay(rt.replay, cfg);
    CHECK_MSG(v.ok, "file round trip failed verification: " + v.error);

    std::remove(path.c_str());
}

TEST(missing_file_is_reported_not_crashed) {
    const auto r = read_replay_file("build/definitely_not_here.tetrarep");
    CHECK(!r.ok);
    CHECK(!r.error.empty());
}

TEST(replays_verify_across_all_presets) {
    // A replay must be self-describing enough to verify under any ruleset it
    // was actually recorded with.
    for (const RulesetConfig& cfg :
         {RulesetConfig::tetra_league(), RulesetConfig::quick_play(),
          RulesetConfig::guideline()}) {
        const Recorded rec = play_and_record(cfg, 2024, 60);
        CHECK_EQ(rec.replay.ruleset_hash, cfg.hash());
        const VerifyResult v = verify_replay(rec.replay, cfg);
        CHECK_MSG(v.ok, std::string("preset ") + cfg.id + " failed: " + v.error);
    }
}

TEST(replay_is_compact) {
    // Self-play generates these by the million, so size matters.
    const RulesetConfig cfg = league();
    const Recorded rec = play_and_record(cfg, 8, 200);
    const size_t bytes = serialize(rec.replay).size();
    const double per_placement =
        static_cast<double>(bytes) / static_cast<double>(rec.replay.placements.size());
    CHECK_MSG(per_placement < 40.0,
              "replay too large: " + std::to_string(per_placement) + " B/placement");
}

TEST(checkpoint_interval_trades_size_for_locality) {
    const RulesetConfig cfg = league();
    const Recorded dense = play_and_record(cfg, 11, 100, 0, 0, /*interval=*/1);
    const Recorded sparse = play_and_record(cfg, 11, 100, 0, 0, /*interval=*/32);

    CHECK(serialize(dense.replay).size() > serialize(sparse.replay).size());
    // Both must verify.
    CHECK(verify_replay(dense.replay, cfg).ok);
    CHECK(verify_replay(sparse.replay, cfg).ok);

    // Dense checkpoints localise a fault precisely.
    Replay tampered = dense.replay;
    tampered.placements[10].x += 1;
    const VerifyResult v = verify_replay(tampered, cfg);
    CHECK(!v.ok);
    CHECK_EQ(v.first_divergence, 10);
}

TEST(verification_is_deterministic) {
    const RulesetConfig cfg = league();
    const Recorded rec = play_and_record(cfg, 606, 90);
    for (int i = 0; i < 5; ++i) {
        const VerifyResult v = verify_replay(rec.replay, cfg);
        CHECK(v.ok);
        CHECK_EQ(v.placements_applied, static_cast<int>(rec.replay.placements.size()));
    }
}

TEST(replay_captures_hold_usage) {
    const RulesetConfig cfg = league();
    Player p;
    p.reset(cfg, 4242, 0);
    MoveGenerator gen;
    ReplayRecorder rec(cfg, 4242, 0, 4);

    int holds = 0;
    for (int i = 0; i < 60 && p.alive(); ++i) {
        const auto acts =
            gen.generate(p.board(), p.active().type, p.hold(),
                         p.visible_next().empty() ? Piece::None : p.visible_next()[0], cfg);
        if (acts.empty()) break;
        // Alternate between holding and not, to exercise both paths.
        size_t pick = 0;
        const bool want_hold = (i % 3) == 0;
        for (size_t k = 0; k < acts.size(); ++k) {
            if (acts[k].use_hold == want_hold) {
                pick = k;
                break;
            }
        }
        if (acts[pick].use_hold) {
            if (!p.do_hold()) continue;
            ++holds;
        }
        p.set_active(acts[pick].piece_state());
        int sent = 0;
        const LockResult r = p.lock_piece(acts[pick].total_duration(), &sent);
        rec.record(acts[pick], p);
        if (!r.ok) break;
    }

    const Replay replay = rec.finish(p);
    CHECK_MSG(holds > 0, "the scripted game should have used hold");
    int recorded_holds = 0;
    for (const auto& pl : replay.placements)
        if (pl.use_hold) ++recorded_holds;
    CHECK_EQ(recorded_holds, holds);

    const VerifyResult v = verify_replay(replay, cfg);
    CHECK_MSG(v.ok, "a replay using hold must verify: " + v.error);
}