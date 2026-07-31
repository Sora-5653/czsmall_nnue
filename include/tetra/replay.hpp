// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- replay format and verification (spec 17 /protocol,
// 18.1 simulator consistency, 22 "replay diff testing").
//
// A replay is the ground truth artefact of this project: it is how a rule
// change is proven safe, how a self-play game becomes a training sample, and
// how a divergence between two builds is located. The format is therefore
// designed around three requirements:
//
//   1. It records the `ruleset_hash`, so a replay can never be silently
//      re-interpreted under different rules (spec 6).
//   2. It records only *inputs* (seed + placements), never derived state, so
//      replaying it is a real test of the simulator rather than a playback of
//      cached results.
//   3. It interleaves periodic state checkpoints, so a divergence is reported
//      at the placement where it first appears instead of at the end.
//
// Spec 17 asks for Protobuf eventually. This is a self-describing, versioned
// binary chunk with the same fields: no third-party dependency, and the
// on-disk layout is explicit little-endian so it is portable between builds.
#pragma once

#include "tetra/movegen.hpp"
#include "tetra/player.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace tetra {

// ---------------------------------------------------------------------------
// Byte-level helpers
// ---------------------------------------------------------------------------
namespace detail {

inline void put_u8(std::vector<std::uint8_t>& b, std::uint8_t v) { b.push_back(v); }

inline void put_u32(std::vector<std::uint8_t>& b, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<std::uint8_t>((v >> (i * 8)) & 0xFF));
}

inline void put_u64(std::vector<std::uint8_t>& b, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<std::uint8_t>((v >> (i * 8)) & 0xFF));
}

inline void put_i64(std::vector<std::uint8_t>& b, std::int64_t v) {
    put_u64(b, static_cast<std::uint64_t>(v));
}

inline void put_str(std::vector<std::uint8_t>& b, const std::string& s) {
    put_u32(b, static_cast<std::uint32_t>(s.size()));
    for (char c : s) b.push_back(static_cast<std::uint8_t>(c));
}

// Bounds-checked reader. Any truncation or overrun sets `ok` to false and every
// subsequent read is a no-op, so a corrupt file can never be half-applied.
struct Reader {
    const std::uint8_t* p = nullptr;
    size_t n = 0;
    size_t at = 0;
    bool ok = true;

    Reader(const std::uint8_t* data, size_t size) : p(data), n(size) {}

    bool need(size_t k) {
        if (!ok || at + k > n) {
            ok = false;
            return false;
        }
        return true;
    }
    std::uint8_t u8() {
        if (!need(1)) return 0;
        return p[at++];
    }
    std::uint32_t u32() {
        if (!need(4)) return 0;
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(p[at + i]) << (i * 8);
        at += 4;
        return v;
    }
    std::uint64_t u64() {
        if (!need(8)) return 0;
        std::uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(p[at + i]) << (i * 8);
        at += 8;
        return v;
    }
    std::int64_t i64() { return static_cast<std::int64_t>(u64()); }
    std::string str() {
        const std::uint32_t len = u32();
        if (!need(len)) return {};
        std::string s(reinterpret_cast<const char*>(p + at), len);
        at += len;
        return s;
    }
};

}  // namespace detail

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

// One committed placement. This is the input to the simulator, not its output:
// only what the player decided is stored.
struct ReplayPlacement {
    bool use_hold = false;
    Piece piece = Piece::None;   // the piece as it locked (after any hold swap)
    int x = 0;
    int y = 0;
    Rot rot = Rot::N;
    DelayBin delay_bin = DelayBin::Fastest;
    Tick duration = 0;           // total ticks the action occupied

    // Spin provenance. This CANNOT be re-derived from the final position: a
    // piece that slid into a notch and a piece that was rotated into the same
    // notch occupy identical cells but score differently. Assuming "rotated"
    // on replay fabricates spins (and the attack that comes with them), so the
    // classification is recorded explicitly and re-imposed on playback.
    SpinType spin = SpinType::None;
    std::uint8_t last_kick = 0;

    // Periodic integrity checkpoint. When `has_checkpoint` is set, the values
    // must match after applying this placement, which is what turns a replay
    // into a regression test.
    bool has_checkpoint = false;
    std::uint64_t board_hash = 0;
    std::int64_t lines_sent = 0;
    std::int64_t lines_received = 0;
    Tick timestamp = 0;
};

// An attack injected from outside (the opponent, or a scripted garbage stream
// in the M3 curriculum). Recorded against the placement index it precedes so
// the ordering is unambiguous on replay.
struct ReplayGarbage {
    std::uint32_t before_placement = 0;
    int lines = 0;
    Tick sent_at = 0;
    int source_player = -1;
};

struct Replay {
    static constexpr char MAGIC[8] = {'T', 'E', 'T', 'R', 'A', 'R', 'E', 'P'};
    static constexpr std::uint32_t VERSION = 1;

    // Header
    std::string ruleset_id;
    std::uint64_t ruleset_hash = 0;
    std::uint32_t ruleset_version = 0;
    std::uint64_t seed = 0;
    std::uint32_t player_index = 0;
    std::string engine_note;  // free-form provenance, e.g. a build id

    // Body
    std::vector<ReplayPlacement> placements;
    std::vector<ReplayGarbage> garbage;

    // Trailer: the final state, always checked.
    std::uint64_t final_board_hash = 0;
    std::int64_t final_lines_sent = 0;
    std::int64_t final_lines_received = 0;
    std::int64_t final_lines_cleared = 0;
    Tick final_timestamp = 0;
    bool final_alive = true;
    TopoutReason final_topout = TopoutReason::None;

    size_t size() const { return placements.size(); }
};

// ---------------------------------------------------------------------------
// Serialisation
// ---------------------------------------------------------------------------

inline std::vector<std::uint8_t> serialize(const Replay& r) {
    using namespace detail;
    std::vector<std::uint8_t> b;
    b.reserve(64 + r.placements.size() * 24);

    for (char c : Replay::MAGIC) b.push_back(static_cast<std::uint8_t>(c));
    put_u32(b, Replay::VERSION);

    put_str(b, r.ruleset_id);
    put_u64(b, r.ruleset_hash);
    put_u32(b, r.ruleset_version);
    put_u64(b, r.seed);
    put_u32(b, r.player_index);
    put_str(b, r.engine_note);

    put_u32(b, static_cast<std::uint32_t>(r.placements.size()));
    for (const auto& p : r.placements) {
        std::uint8_t flags = 0;
        if (p.use_hold) flags |= 1u;
        if (p.has_checkpoint) flags |= 2u;
        put_u8(b, flags);
        put_u8(b, static_cast<std::uint8_t>(p.piece));
        put_u8(b, static_cast<std::uint8_t>(p.rot));
        put_u8(b, static_cast<std::uint8_t>(p.delay_bin));
        // x and y can legitimately be negative (a bounding box may hang off the
        // left edge), so they are biased rather than truncated.
        put_u8(b, static_cast<std::uint8_t>(p.x + 16));
        put_u8(b, static_cast<std::uint8_t>(p.y + 16));
        put_u8(b, static_cast<std::uint8_t>(p.spin));
        put_u8(b, p.last_kick);
        put_i64(b, p.duration);
        if (p.has_checkpoint) {
            put_u64(b, p.board_hash);
            put_i64(b, p.lines_sent);
            put_i64(b, p.lines_received);
            put_i64(b, p.timestamp);
        }
    }

    put_u32(b, static_cast<std::uint32_t>(r.garbage.size()));
    for (const auto& g : r.garbage) {
        put_u32(b, g.before_placement);
        put_u32(b, static_cast<std::uint32_t>(g.lines));
        put_i64(b, g.sent_at);
        put_u32(b, static_cast<std::uint32_t>(g.source_player + 1));
    }

    put_u64(b, r.final_board_hash);
    put_i64(b, r.final_lines_sent);
    put_i64(b, r.final_lines_received);
    put_i64(b, r.final_lines_cleared);
    put_i64(b, r.final_timestamp);
    put_u8(b, r.final_alive ? 1 : 0);
    put_u8(b, static_cast<std::uint8_t>(r.final_topout));

    // Trailing checksum over everything above, so truncation and bit rot are
    // caught before the replay is trusted.
    std::uint64_t sum = 1469598103934665603ull;
    for (std::uint8_t v : b) {
        sum ^= v;
        sum *= 1099511628211ull;
    }
    put_u64(b, sum);
    return b;
}

struct DeserializeResult {
    bool ok = false;
    Replay replay;
    std::string error;
};

inline DeserializeResult deserialize(const std::uint8_t* data, size_t size) {
    using namespace detail;
    DeserializeResult res;

    if (size < 8 + 4 + 8) {
        res.error = "too short";
        return res;
    }
    if (std::memcmp(data, Replay::MAGIC, 8) != 0) {
        res.error = "bad magic";
        return res;
    }

    // Verify the checksum before interpreting anything.
    std::uint64_t sum = 1469598103934665603ull;
    for (size_t i = 0; i + 8 < size; ++i) {
        sum ^= data[i];
        sum *= 1099511628211ull;
    }
    std::uint64_t stored = 0;
    for (int i = 0; i < 8; ++i)
        stored |= static_cast<std::uint64_t>(data[size - 8 + i]) << (i * 8);
    if (sum != stored) {
        res.error = "checksum mismatch";
        return res;
    }

    Reader rd(data, size - 8);
    rd.at = 8;
    const std::uint32_t version = rd.u32();
    if (version != Replay::VERSION) {
        res.error = "unsupported version " + std::to_string(version);
        return res;
    }

    Replay r;
    r.ruleset_id = rd.str();
    r.ruleset_hash = rd.u64();
    r.ruleset_version = rd.u32();
    r.seed = rd.u64();
    r.player_index = rd.u32();
    r.engine_note = rd.str();

    const std::uint32_t n = rd.u32();
    if (!rd.ok || n > (1u << 24)) {
        res.error = "implausible placement count";
        return res;
    }
    r.placements.reserve(n);
    for (std::uint32_t i = 0; i < n && rd.ok; ++i) {
        ReplayPlacement p;
        const std::uint8_t flags = rd.u8();
        p.use_hold = (flags & 1u) != 0;
        p.has_checkpoint = (flags & 2u) != 0;
        p.piece = static_cast<Piece>(rd.u8());
        p.rot = static_cast<Rot>(rd.u8() & 3);
        p.delay_bin = static_cast<DelayBin>(rd.u8());
        p.x = static_cast<int>(rd.u8()) - 16;
        p.y = static_cast<int>(rd.u8()) - 16;
        p.spin = static_cast<SpinType>(rd.u8());
        p.last_kick = rd.u8();
        p.duration = rd.i64();
        if (p.has_checkpoint) {
            p.board_hash = rd.u64();
            p.lines_sent = rd.i64();
            p.lines_received = rd.i64();
            p.timestamp = rd.i64();
        }
        r.placements.push_back(p);
    }

    const std::uint32_t gn = rd.u32();
    if (!rd.ok || gn > (1u << 24)) {
        res.error = "implausible garbage count";
        return res;
    }
    for (std::uint32_t i = 0; i < gn && rd.ok; ++i) {
        ReplayGarbage g;
        g.before_placement = rd.u32();
        g.lines = static_cast<int>(rd.u32());
        g.sent_at = rd.i64();
        g.source_player = static_cast<int>(rd.u32()) - 1;
        r.garbage.push_back(g);
    }

    r.final_board_hash = rd.u64();
    r.final_lines_sent = rd.i64();
    r.final_lines_received = rd.i64();
    r.final_lines_cleared = rd.i64();
    r.final_timestamp = rd.i64();
    r.final_alive = rd.u8() != 0;
    r.final_topout = static_cast<TopoutReason>(rd.u8());

    if (!rd.ok) {
        res.error = "truncated";
        return res;
    }
    res.ok = true;
    res.replay = std::move(r);
    return res;
}

inline DeserializeResult deserialize(const std::vector<std::uint8_t>& bytes) {
    return deserialize(bytes.data(), bytes.size());
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

inline bool write_replay_file(const std::string& path, const Replay& r) {
    const std::vector<std::uint8_t> bytes = serialize(r);
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const size_t written = std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    return written == bytes.size();
}

inline DeserializeResult read_replay_file(const std::string& path) {
    DeserializeResult res;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        res.error = "cannot open " + path;
        return res;
    }
    std::vector<std::uint8_t> bytes;
    std::uint8_t buf[4096];
    size_t got = 0;
    while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0)
        bytes.insert(bytes.end(), buf, buf + got);
    std::fclose(f);
    return deserialize(bytes);
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

// Accumulates a replay while a game is played. The caller drives the game; the
// recorder only observes, so recording can never change the outcome.
class ReplayRecorder {
public:
    ReplayRecorder(const RulesetConfig& cfg, std::uint64_t seed, int player_index = 0,
                   int checkpoint_interval = 16) {
        replay_.ruleset_id = cfg.id;
        replay_.ruleset_hash = cfg.hash();
        replay_.ruleset_version = static_cast<std::uint32_t>(cfg.version);
        replay_.seed = seed;
        replay_.player_index = static_cast<std::uint32_t>(player_index);
        interval_ = checkpoint_interval;
    }

    void note(const std::string& s) { replay_.engine_note = s; }

    // Record an attack that is about to be delivered to the player.
    void record_garbage(int lines, Tick sent_at, int source_player) {
        ReplayGarbage g;
        g.before_placement = static_cast<std::uint32_t>(replay_.placements.size());
        g.lines = lines;
        g.sent_at = sent_at;
        g.source_player = source_player;
        replay_.garbage.push_back(g);
    }

    // Record a placement immediately AFTER it has been applied to `p`, so the
    // checkpoint reflects the post-placement state.
    void record(const PlacementAction& a, const Player& p) {
        ReplayPlacement rec;
        rec.use_hold = a.use_hold;
        rec.piece = a.final_piece;
        rec.x = a.final_x;
        rec.y = a.final_y;
        rec.rot = a.final_rotation;
        rec.delay_bin = a.delay_bin;
        rec.duration = a.total_duration();
        rec.spin = a.spin;
        rec.last_kick = static_cast<std::uint8_t>(a.last_kick);

        const size_t index = replay_.placements.size();
        if (interval_ > 0 && (index % static_cast<size_t>(interval_)) == 0) {
            rec.has_checkpoint = true;
            rec.board_hash = detail::board_hash(p.board());
            rec.lines_sent = p.lines_sent();
            rec.lines_received = p.lines_received();
            rec.timestamp = p.now();
        }
        replay_.placements.push_back(rec);
    }

    // Seal the replay with the final state.
    Replay finish(const Player& p) {
        replay_.final_board_hash = detail::board_hash(p.board());
        replay_.final_lines_sent = p.lines_sent();
        replay_.final_lines_received = p.lines_received();
        replay_.final_lines_cleared = p.lines_cleared();
        replay_.final_timestamp = p.now();
        replay_.final_alive = p.alive();
        replay_.final_topout = p.topout_reason();
        return replay_;
    }

    const Replay& current() const { return replay_; }

private:
    Replay replay_;
    int interval_ = 16;
};

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

// The result of re-simulating a replay. This is the harness spec 22 asks for:
// when a rule change breaks something, `first_divergence` says exactly where.
struct VerifyResult {
    bool ok = false;
    int placements_applied = 0;
    int first_divergence = -1;  // placement index, or -1 if none
    std::string error;

    // Populated on divergence so the failure message can be specific.
    std::uint64_t expected_hash = 0;
    std::uint64_t actual_hash = 0;
};

// Re-run a replay against the current engine and compare it to the recorded
// checkpoints. `cfg` must be the ruleset the replay was recorded under; the
// hash is checked so a mismatch is reported rather than silently tolerated.
inline VerifyResult verify_replay(const Replay& r, const RulesetConfig& cfg) {
    VerifyResult v;

    if (cfg.hash() != r.ruleset_hash) {
        v.error = "ruleset hash mismatch: replay " + std::to_string(r.ruleset_hash) +
                  ", engine " + std::to_string(cfg.hash());
        return v;
    }

    Player p;
    p.reset(cfg, r.seed, static_cast<int>(r.player_index));

    size_t gi = 0;
    for (size_t i = 0; i < r.placements.size(); ++i) {
        // Deliver any garbage scheduled before this placement.
        while (gi < r.garbage.size() && r.garbage[gi].before_placement == i) {
            p.receive_attack(r.garbage[gi].lines, r.garbage[gi].sent_at,
                             r.garbage[gi].source_player);
            ++gi;
        }

        const ReplayPlacement& rec = r.placements[i];
        if (!p.alive()) {
            v.error = "replay continues past a top-out at placement " + std::to_string(i);
            v.first_divergence = static_cast<int>(i);
            return v;
        }

        if (rec.use_hold && !p.do_hold()) {
            v.error = "hold refused at placement " + std::to_string(i);
            v.first_divergence = static_cast<int>(i);
            return v;
        }

        // The recorded piece must be the one actually on the board: this is
        // what catches a randomizer or hold-ordering regression.
        if (p.active().type != rec.piece) {
            v.error = std::string("piece mismatch at placement ") + std::to_string(i) +
                      ": replay says " + piece_name(rec.piece) + ", engine has " +
                      piece_name(p.active().type);
            v.first_divergence = static_cast<int>(i);
            return v;
        }

        ActivePiece piece;
        piece.type = rec.piece;
        piece.x = rec.x;
        piece.y = rec.y;
        piece.rot = rec.rot;
        // Restore the recorded provenance rather than guessing it. Marking
        // every replayed piece as "rotated" would let the engine detect spins
        // that the original game never scored.
        piece.last_action =
            (rec.spin != SpinType::None) ? LastAction::Rotate : LastAction::Drop;
        piece.last_kick = static_cast<int>(rec.last_kick);
        if (collides(p.board(), piece)) {
            v.error = "recorded placement collides at " + std::to_string(i);
            v.first_divergence = static_cast<int>(i);
            return v;
        }
        p.set_active(piece);

        // The engine must independently agree with the recorded spin class:
        // this is the check that catches a spin-detection regression.
        const SpinType derived = detect_spin(p.board(), piece, cfg);
        if (derived != rec.spin) {
            v.error = std::string("spin mismatch at placement ") + std::to_string(i) +
                      ": replay says " + spin_name(rec.spin) + ", engine derives " +
                      spin_name(derived);
            v.first_divergence = static_cast<int>(i);
            return v;
        }

        int out = 0;
        const LockResult lr = p.lock_piece(rec.duration, &out);
        if (!lr.ok && !lr.topped_out) {
            v.error = "placement rejected at " + std::to_string(i);
            v.first_divergence = static_cast<int>(i);
            return v;
        }
        ++v.placements_applied;

        if (rec.has_checkpoint) {
            const std::uint64_t h = detail::board_hash(p.board());
            if (h != rec.board_hash || p.lines_sent() != rec.lines_sent ||
                p.lines_received() != rec.lines_received || p.now() != rec.timestamp) {
                v.first_divergence = static_cast<int>(i);
                v.expected_hash = rec.board_hash;
                v.actual_hash = h;
                v.error = "checkpoint mismatch at placement " + std::to_string(i);
                return v;
            }
        }
    }

    if (detail::board_hash(p.board()) != r.final_board_hash ||
        p.lines_sent() != r.final_lines_sent || p.lines_received() != r.final_lines_received ||
        p.lines_cleared() != r.final_lines_cleared || p.now() != r.final_timestamp ||
        p.alive() != r.final_alive) {
        v.first_divergence = static_cast<int>(r.placements.size());
        v.expected_hash = r.final_board_hash;
        v.actual_hash = detail::board_hash(p.board());
        v.error = "final state mismatch";
        return v;
    }

    v.ok = true;
    return v;
}

}  // namespace tetra