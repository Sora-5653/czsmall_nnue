// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- training-set export (spec 13.5, 17 /trainer).
//
// The handover point between the C++ engine and the Python trainer.
//
// Format: a small self-describing header followed by the flat float32 buffers
// of a `TensorBatch`. That is deliberately the *same* rectangular layout the
// C++ inference path uses, so the trainer and the engine cannot disagree about
// shapes or padding, and the file can be wrapped by numpy with no reshaping:
//
//     tokens        float32 [N, T, TOKEN_FEATURES]
//     token_mask    float32 [N, T]
//     actions       float32 [N, A, ACTION_FEATURES]
//     action_mask   float32 [N, A]
//     policy_target float32 [N, A]
//     value_target  float32 [N]
//     aux_target    float32 [N, AUX_TARGETS]
//     aux_valid_mask float32 [N, AUX_TARGETS] (v3)
//     provenance    int/uint arrays (v3)
//
// Everything is little-endian float32 (or int32 for the header), which is what
// both numpy and libtorch expect natively on the platforms this runs on.
#pragma once

#include "tetra/batch.hpp"
#include "tetra/movegen.hpp"
#include "tetra/player.hpp"
#include "tetra/replay_buffer.hpp"
#include "tetra/ruleset.hpp"
#include "tetra/schema.hpp"
#include "tetra/tokenizer.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace tetra {

struct DatasetHeader {
    static constexpr char MAGIC[8] = {'T', 'E', 'T', 'R', 'A', 'D', 'A', 'T'};
    static constexpr std::uint32_t VERSION_LEGACY = 1;
    static constexpr std::uint32_t VERSION_COMPACT = 2;
    static constexpr std::uint32_t VERSION = 3;
    static constexpr std::uint32_t CONTRACT_VERSION = 1;

    std::uint32_t version = VERSION;
    std::uint32_t samples = 0;
    std::uint32_t max_tokens = 0;
    std::uint32_t max_actions = 0;
    std::uint32_t token_features = TOKEN_FEATURES;
    std::uint32_t action_features = ACTION_FEATURES;
    std::uint32_t aux_targets = TensorBatch::AUX_TARGETS;
    std::uint64_t ruleset_hash = 0;
    std::uint32_t model_version = 0;

    // Version-3 contract extension.  The human-readable names and ordered
    // lists are emitted by the manifest; the binary header carries their
    // stable hashes so a shard cannot pass validation on width alone.
    std::uint32_t contract_version = 0;
    std::uint32_t tokenizer_schema_version =
        schema::TOKENIZER_SCHEMA_VERSION;
    std::uint64_t tokenizer_schema_hash = schema::TOKENIZER_SCHEMA_HASH;
    std::uint64_t observation_schema_hash = schema::OBSERVATION_SCHEMA_HASH;
    std::uint32_t action_schema_version = schema::ACTION_SCHEMA_VERSION;
    std::uint32_t aux_target_schema_version =
        schema::LEGACY_AUX_TARGET_SCHEMA_VERSION;
    std::uint32_t randomizer_type = 0;
    std::uint32_t termination_reason =
        static_cast<std::uint32_t>(TerminationReason::Unknown);
    std::uint64_t self_play_seed = 0;
    std::uint64_t token_kind_order_hash = schema::TOKENIZER_SCHEMA_HASH;
};

struct DatasetContract {
    std::uint32_t contract_version = DatasetHeader::CONTRACT_VERSION;
    std::uint32_t tokenizer_schema_version = schema::TOKENIZER_SCHEMA_VERSION;
    std::uint64_t tokenizer_schema_hash = schema::TOKENIZER_SCHEMA_HASH;
    std::uint64_t observation_schema_hash = schema::OBSERVATION_SCHEMA_HASH;
    std::uint32_t action_schema_version = schema::ACTION_SCHEMA_VERSION;
    std::uint32_t aux_target_schema_version = schema::AUX_TARGET_SCHEMA_VERSION;
    std::uint32_t randomizer_type = 0;
    TerminationReason termination_reason = TerminationReason::Unknown;
    std::uint64_t self_play_seed = 0;
    std::uint64_t token_kind_order_hash = schema::TOKENIZER_SCHEMA_HASH;
};

namespace detail {

inline void put32(std::vector<std::uint8_t>& b, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<std::uint8_t>((v >> (i * 8)) & 0xFF));
}
inline void put64d(std::vector<std::uint8_t>& b, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<std::uint8_t>((v >> (i * 8)) & 0xFF));
}
inline void put_i32(std::vector<std::uint8_t>& b, std::int32_t v) {
    put32(b, static_cast<std::uint32_t>(v));
}
inline void put_floats(std::vector<std::uint8_t>& b, const std::vector<float>& v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(v.data());
    b.insert(b.end(), p, p + v.size() * sizeof(float));
}

inline void put_contract(std::vector<std::uint8_t>& b, const DatasetContract& c) {
    put32(b, c.contract_version);
    put32(b, c.tokenizer_schema_version);
    put64d(b, c.tokenizer_schema_hash);
    put64d(b, c.observation_schema_hash);
    put32(b, c.action_schema_version);
    put32(b, c.aux_target_schema_version);
    put32(b, c.randomizer_type);
    put32(b, static_cast<std::uint32_t>(c.termination_reason));
    put64d(b, c.self_play_seed);
    put64d(b, c.token_kind_order_hash);
}

inline constexpr size_t CONTRACT_BYTES = 4 + 4 + 8 + 8 + 4 + 4 + 4 + 4 + 8 + 8;

}  // namespace detail

// Serialise a training batch. The batch must have been built with targets.
inline std::vector<std::uint8_t> serialize_dataset(const TensorBatch& batch,
                                                   std::uint64_t ruleset_hash,
                                                   std::uint32_t model_version,
                                                   const DatasetContract& contract) {
    const std::uint32_t aux_width =
        batch.batch > 0 && batch.aux_target.size() % static_cast<size_t>(batch.batch) == 0
            ? static_cast<std::uint32_t>(batch.aux_target.size() /
                                         static_cast<size_t>(batch.batch))
            : static_cast<std::uint32_t>(TensorBatch::AUX_TARGETS);
    std::vector<std::uint8_t> out;
    for (char c : DatasetHeader::MAGIC) out.push_back(static_cast<std::uint8_t>(c));
    detail::put32(out, DatasetHeader::VERSION);
    detail::put32(out, static_cast<std::uint32_t>(batch.batch));
    detail::put32(out, static_cast<std::uint32_t>(batch.max_tokens));
    detail::put32(out, static_cast<std::uint32_t>(batch.max_actions));
    detail::put32(out, static_cast<std::uint32_t>(TOKEN_FEATURES));
    detail::put32(out, static_cast<std::uint32_t>(ACTION_FEATURES));
    detail::put32(out, aux_width);
    detail::put64d(out, ruleset_hash);
    detail::put32(out, model_version);
    detail::put_contract(out, contract);

    detail::put_floats(out, batch.tokens);
    detail::put_floats(out, batch.token_mask);
    detail::put_floats(out, batch.actions);
    detail::put_floats(out, batch.action_mask);
    detail::put_floats(out, batch.policy_target);
    detail::put_floats(out, batch.value_target);
    detail::put_floats(out, batch.aux_target);
    const size_t aux_count = static_cast<size_t>(batch.batch) *
                             static_cast<size_t>(aux_width);
    if (batch.aux_valid_mask.size() == aux_count) {
        detail::put_floats(out, batch.aux_valid_mask);
    } else {
        detail::put_floats(out, std::vector<float>(aux_count, 1.0f));
    }

    for (int i = 0; i < batch.batch; ++i) {
        const std::int32_t value =
            i < static_cast<int>(batch.player_perspective.size())
                ? batch.player_perspective[static_cast<size_t>(i)]
                : 1;
        detail::put_i32(out, value);
    }
    for (int i = 0; i < batch.batch; ++i) {
        const std::int32_t value =
            i < static_cast<int>(batch.termination_reason.size())
                ? batch.termination_reason[static_cast<size_t>(i)]
                : static_cast<std::int32_t>(TerminationReason::Unknown);
        detail::put_i32(out, value);
    }
    for (int i = 0; i < batch.batch; ++i) {
        const std::uint64_t value =
            i < static_cast<int>(batch.game_seed.size())
                ? batch.game_seed[static_cast<size_t>(i)]
                : 0;
        detail::put64d(out, value);
    }
    for (int i = 0; i < batch.batch; ++i) {
        const std::uint32_t value =
            i < static_cast<int>(batch.move_number.size())
                ? batch.move_number[static_cast<size_t>(i)]
                : 0;
        detail::put32(out, value);
    }
    return out;
}

inline DatasetContract default_dataset_contract(const TensorBatch& batch) {
    DatasetContract contract;
    const size_t aux_width = batch.batch > 0
                                 ? batch.aux_target.size() /
                                       static_cast<size_t>(batch.batch)
                                 : static_cast<size_t>(TensorBatch::AUX_TARGETS);
    contract.aux_target_schema_version =
        aux_width == static_cast<size_t>(schema::LEGACY_AUX_TARGET_COUNT)
            ? schema::LEGACY_AUX_TARGET_SCHEMA_VERSION
            : schema::AUX_TARGET_SCHEMA_VERSION;
    if (!batch.termination_reason.empty()) {
        const std::int32_t first = batch.termination_reason.front();
        bool same = true;
        for (std::int32_t value : batch.termination_reason)
            if (value != first) same = false;
        contract.termination_reason =
            same ? static_cast<TerminationReason>(first) : TerminationReason::Mixed;
    }
    if (!batch.game_seed.empty()) {
        const std::uint64_t first = batch.game_seed.front();
        bool same = true;
        for (std::uint64_t value : batch.game_seed)
            if (value != first) same = false;
        contract.self_play_seed = same ? first : 0;
    }
    return contract;
}

inline std::vector<std::uint8_t> serialize_dataset(const TensorBatch& batch,
                                                   std::uint64_t ruleset_hash,
                                                   std::uint32_t model_version) {
    return serialize_dataset(batch, ruleset_hash, model_version,
                             default_dataset_contract(batch));
}

inline bool write_dataset_file(const std::string& path, const TensorBatch& batch,
                               std::uint64_t ruleset_hash, std::uint32_t model_version,
                               const DatasetContract& contract) {
    const std::vector<std::uint8_t> bytes =
        serialize_dataset(batch, ruleset_hash, model_version, contract);
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const size_t n = std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    return n == bytes.size();
}

inline bool write_dataset_file(const std::string& path, const TensorBatch& batch,
                               std::uint64_t ruleset_hash, std::uint32_t model_version) {
    return write_dataset_file(path, batch, ruleset_hash, model_version,
                              default_dataset_contract(batch));
}

// Check whether a sample list can be safely serialized as Compact Replay+π (v2).
inline bool can_export_compact(const std::vector<const TrainingSample*>& samples) {
    if (samples.empty()) return false;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> seen;
    seen.reserve(samples.size());
    for (const auto* s : samples) {
        if (!s || s->ruleset_hash == 0 || s->chosen_action < 0)
            return false;
        if (s->aux_target_schema_version >= schema::AUX_TARGET_SCHEMA_VERSION)
            return false;
        // Compact Replay+ reconstruction only contains player 0's replay
        // stream. A two-board sample also contains the opponent board token
        // stream, which cannot be recovered from that metadata without
        // replaying the opponent. Keep those samples in rectangular v1,
        // where the already-masked observation is preserved exactly.
        for (const auto& token : s->tokens) {
            if (token.kind == TokenKind::OpponentRow ||
                token.kind == TokenKind::OpponentColumn ||
                token.kind == TokenKind::OpponentSummary ||
                token.kind == TokenKind::OpponentCounters)
                return false;
        }
        seen.push_back({s->game_seed, s->move_number});
    }
    std::sort(seen.begin(), seen.end());
    std::uint64_t cur_seed = ~0ull;
    std::uint32_t expected_move = 0;
    for (size_t i = 0; i < seen.size(); ++i) {
        if (i == 0 || seen[i].first != cur_seed) {
            cur_seed = seen[i].first;
            expected_move = 0;
        }
        if (seen[i].second != expected_move) return false;
        ++expected_move;
    }
    return true;
}

// Compact Replay+π serialization (version 2).
// Saves only provenance (ruleset, seed, move_number, chosen_action) and search
// targets (policy, outcome, aux_targets). Eliminates padded float32 tensors
// from disk storage, reducing I/O and disk size by ~100x (spec 13.5, ADR 0012).
inline std::vector<std::uint8_t> serialize_compact_dataset(
    const std::vector<const TrainingSample*>& samples,
    std::uint32_t model_version = 0) {
    std::vector<std::uint8_t> out;
    for (char c : DatasetHeader::MAGIC) out.push_back(static_cast<std::uint8_t>(c));
    detail::put32(out, DatasetHeader::VERSION_COMPACT);
    detail::put32(out, static_cast<std::uint32_t>(samples.size()));
    detail::put32(out, 0);  // max_tokens (computed on read)
    detail::put32(out, 0);  // max_actions (computed on read)
    detail::put32(out, static_cast<std::uint32_t>(TOKEN_FEATURES));
    detail::put32(out, static_cast<std::uint32_t>(ACTION_FEATURES));
    // Compact v2 is the legacy four-target format.  New interval targets
    // require the rectangular v3 payload because replay metadata alone
    // cannot reconstruct their event masks.
    detail::put32(out, static_cast<std::uint32_t>(schema::LEGACY_AUX_TARGET_COUNT));
    const std::uint64_t rhash = samples.empty() ? 0 : samples[0]->ruleset_hash;
    detail::put64d(out, rhash);
    detail::put32(out, model_version);

    for (const auto* s : samples) {
        detail::put64d(out, s->ruleset_hash);
        detail::put64d(out, s->game_seed);
        detail::put32(out, s->move_number);
        detail::put32(out, static_cast<std::uint32_t>(s->chosen_action));
        detail::put32(out, static_cast<std::uint32_t>(s->garbage_style));
        detail::put32(out, static_cast<std::uint32_t>(s->garbage_period));
        detail::put32(out, static_cast<std::uint32_t>(s->garbage_lines));
        detail::put32(out, static_cast<std::uint32_t>(s->search_policy.size()));
        detail::put_floats(out, s->search_policy);
        std::vector<float> scalars = {
            s->outcome,
            s->search_value,
            static_cast<float>(s->time_to_terminal),
            s->future_attack_1s,
            s->future_garbage_received,
            s->topped_out_within_4 ? 1.0f : 0.0f,
            s->topped_out_within_8 ? 1.0f : 0.0f,
        };
        detail::put_floats(out, scalars);
    }
    return out;
}

inline bool write_compact_dataset_file(
    const std::string& path,
    const std::vector<const TrainingSample*>& samples,
    std::uint32_t model_version = 0) {
    const std::vector<std::uint8_t> bytes =
        serialize_compact_dataset(samples, model_version);
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const size_t n = std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    return n == bytes.size();
}

struct DatasetReadResult {
    bool ok = false;
    DatasetHeader header;
    TensorBatch batch;
    std::string error;
};

inline DatasetReadResult deserialize_compact_dataset(
    const std::vector<std::uint8_t>& bytes, size_t at) {
    DatasetReadResult res;
    auto rd32 = [&]() {
        if (at + 4 > bytes.size()) return static_cast<std::uint32_t>(0);
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i)
            v |= static_cast<std::uint32_t>(bytes[at + static_cast<size_t>(i)])
                 << (i * 8);
        at += 4;
        return v;
    };
    auto rd64 = [&]() {
        if (at + 8 > bytes.size()) return static_cast<std::uint64_t>(0);
        std::uint64_t v = 0;
        for (int i = 0; i < 8; ++i)
            v |= static_cast<std::uint64_t>(bytes[at + static_cast<size_t>(i)])
                 << (i * 8);
        at += 8;
        return v;
    };

    DatasetHeader h;
    h.version = DatasetHeader::VERSION_COMPACT;
    h.samples = rd32();
    h.max_tokens = rd32();
    h.max_actions = rd32();
    h.token_features = rd32();
    h.action_features = rd32();
    h.aux_targets = rd32();
    h.ruleset_hash = rd64();
    h.model_version = rd32();

    struct V2Record {
        std::uint64_t ruleset_hash;
        std::uint64_t game_seed;
        std::uint32_t move_number;
        int chosen_action;
        std::uint8_t garbage_style;
        std::uint8_t garbage_period;
        std::uint8_t garbage_lines;
        std::vector<float> search_policy;
        float outcome;
        float search_value;
        int time_to_terminal;
        float future_attack_1s;
        float future_garbage_received;
        bool topped_out_within_4;
        bool topped_out_within_8;
        size_t original_index;
    };

    std::vector<V2Record> records;
    records.reserve(h.samples);

    for (size_t idx = 0; idx < h.samples; ++idx) {
        if (at + 8 + 8 + 4 + 4 + 4 + 4 + 4 + 4 > bytes.size()) {
            res.error = "truncated payload";
            return res;
        }
        V2Record r;
        r.ruleset_hash = rd64();
        r.game_seed = rd64();
        r.move_number = rd32();
        r.chosen_action = static_cast<int>(rd32());
        r.garbage_style = static_cast<std::uint8_t>(rd32());
        r.garbage_period = static_cast<std::uint8_t>(rd32());
        r.garbage_lines = static_cast<std::uint8_t>(rd32());
        const std::uint32_t psize = rd32();
        r.search_policy.resize(psize);
        if (at + psize * sizeof(float) > bytes.size()) {
            res.error = "truncated payload";
            return res;
        }
        std::memcpy(r.search_policy.data(), bytes.data() + at,
                    psize * sizeof(float));
        at += psize * sizeof(float);

        if (at + 7 * sizeof(float) > bytes.size()) {
            res.error = "truncated payload";
            return res;
        }
        float scalars[7];
        std::memcpy(scalars, bytes.data() + at, 7 * sizeof(float));
        at += 7 * sizeof(float);

        r.outcome = scalars[0];
        r.search_value = scalars[1];
        r.time_to_terminal = static_cast<int>(scalars[2]);
        r.future_attack_1s = scalars[3];
        r.future_garbage_received = scalars[4];
        r.topped_out_within_4 = (scalars[5] > 0.5f);
        r.topped_out_within_8 = (scalars[6] > 0.5f);
        r.original_index = idx;
        records.push_back(std::move(r));
    }

    std::map<std::pair<std::uint64_t, std::uint64_t>, std::vector<V2Record>> games;
    for (auto& r : records) {
        games[{r.ruleset_hash, r.game_seed}].push_back(std::move(r));
    }

    std::vector<TrainingSample> all_samples(h.samples);
    for (auto& [key, vec] : games) {
        std::sort(
            vec.begin(), vec.end(),
            [](const V2Record& a, const V2Record& b) { return a.move_number < b.move_number; });

        RulesetConfig rules = ruleset_from_hash(key.first);
        Player p;
        p.reset(rules, key.second, 0);
        MoveGenerator gen;
        Tokenizer tok;
        Rng garbage_rng(key.second ^ 0x5EEDFACEull);

        for (const auto& r : vec) {
            auto style = static_cast<GarbageStyle>(r.garbage_style);
            if (style != GarbageStyle::None && r.move_number > 0) {
                switch (style) {
                    case GarbageStyle::Steady:
                        if (r.move_number % std::max(1, static_cast<int>(r.garbage_period)) == 0)
                            p.receive_attack(r.garbage_lines, p.now(), 1);
                        break;
                    case GarbageStyle::FastSmall:
                        if (r.move_number % std::max(1, static_cast<int>(r.garbage_period) / 3) == 0)
                            p.receive_attack(1, p.now(), 1);
                        break;
                    case GarbageStyle::SlowLarge:
                        if (r.move_number % (std::max(1, static_cast<int>(r.garbage_period)) * 3) == 0)
                            p.receive_attack(r.garbage_lines * 3, p.now(), 1);
                        break;
                    case GarbageStyle::Burst:
                        if (garbage_rng.chance(1, std::max(2, static_cast<int>(r.garbage_period))))
                            p.receive_attack(r.garbage_lines + static_cast<int>(garbage_rng.below(4)), p.now(), 1);
                        break;
                    default:
                        break;
                }
            }

            const auto actions = gen.generate(
                p.board(), p.active().type, p.hold(),
                p.visible_next().empty() ? Piece::None : p.visible_next()[0],
                rules);
            if (actions.empty() || r.chosen_action < 0 ||
                r.chosen_action >= static_cast<int>(actions.size())) {
                res.error = "reconstruction divergence at move " +
                            std::to_string(r.move_number);
                return res;
            }

            TrainingSample s;
            const TokenizedObservation t = tok.encode(observe(p), rules);
            s.tokens = t.tokens;
            s.action_embeddings.reserve(actions.size());
            for (const auto& a : actions) {
                s.action_embeddings.push_back(embed_action(a, p.board(), rules));
            }
            s.search_policy = r.search_policy;
            s.outcome = r.outcome;
            s.n_step_return = r.outcome;
            s.search_value = r.search_value;
            s.time_to_terminal = r.time_to_terminal;
            s.future_attack_1s = r.future_attack_1s;
            s.future_garbage_received = r.future_garbage_received;
            s.topped_out_within_4 = r.topped_out_within_4;
            s.topped_out_within_8 = r.topped_out_within_8;
            s.ruleset_hash = r.ruleset_hash;
            s.model_version = h.model_version;
            s.move_number = r.move_number;
            s.chosen_action = r.chosen_action;

            all_samples[r.original_index] = std::move(s);

            const PlacementAction& chosen =
                actions[static_cast<size_t>(r.chosen_action)];
            if (chosen.use_hold && !p.do_hold()) break;
            p.set_active(chosen.piece_state());
            int sent = 0;
            p.lock_piece(chosen.total_duration(), &sent);
        }
    }

    std::vector<const TrainingSample*> ptrs(all_samples.size());
    for (size_t i = 0; i < all_samples.size(); ++i) ptrs[i] = &all_samples[i];
    res.batch = make_training_batch(ptrs);
    // make_training_batch uses the current schema width.  Compact v2 stores
    // only the legacy four targets, so expose exactly that legacy shape to
    // callers reading an old file.
    if (h.aux_targets != TensorBatch::AUX_TARGETS) {
        std::vector<float> legacy_target(
            static_cast<size_t>(h.samples) * h.aux_targets, 0.0f);
        std::vector<float> legacy_valid(
            static_cast<size_t>(h.samples) * h.aux_targets, 1.0f);
        for (std::uint32_t i = 0; i < h.samples; ++i) {
            for (std::uint32_t j = 0; j < h.aux_targets; ++j) {
                const size_t src = static_cast<size_t>(i) * TensorBatch::AUX_TARGETS + j;
                const size_t dst = static_cast<size_t>(i) * h.aux_targets + j;
                legacy_target[dst] = res.batch.aux_target[src];
            }
        }
        res.batch.aux_target = std::move(legacy_target);
        res.batch.aux_valid_mask = std::move(legacy_valid);
    }
    res.header = h;
    res.header.max_tokens = static_cast<std::uint32_t>(res.batch.max_tokens);
    res.header.max_actions = static_cast<std::uint32_t>(res.batch.max_actions);
    res.ok = true;
    return res;
}

// Read back a dataset. Used by the round-trip test, and by any C++ tooling that
// wants to inspect what the trainer will see.
inline DatasetReadResult deserialize_dataset(const std::vector<std::uint8_t>& bytes) {
    DatasetReadResult res;
    const size_t header_size = 8 + 4 * 7 + 8 + 4;
    if (bytes.size() < header_size) {
        res.error = "too short";
        return res;
    }
    if (std::memcmp(bytes.data(), DatasetHeader::MAGIC, 8) != 0) {
        res.error = "bad magic";
        return res;
    }

    size_t at = 8;
    auto rd32 = [&]() {
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(bytes[at + static_cast<size_t>(i)]) << (i * 8);
        at += 4;
        return v;
    };
    auto rd64 = [&]() {
        std::uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(bytes[at + static_cast<size_t>(i)]) << (i * 8);
        at += 8;
        return v;
    };

    DatasetHeader h;
    h.version = rd32();
    if (h.version == DatasetHeader::VERSION_COMPACT) {
        return deserialize_compact_dataset(bytes, at);
    }
    if (h.version != DatasetHeader::VERSION && h.version != DatasetHeader::VERSION_LEGACY) {
        res.error = "unsupported version";
        return res;
    }
    h.samples = rd32();
    h.max_tokens = rd32();
    h.max_actions = rd32();
    h.token_features = rd32();
    h.action_features = rd32();
    h.aux_targets = rd32();
    h.ruleset_hash = rd64();
    h.model_version = rd32();

    if (h.version == DatasetHeader::VERSION) {
        if (bytes.size() < at + detail::CONTRACT_BYTES) {
            res.error = "truncated contract header";
            return res;
        }
        h.contract_version = rd32();
        h.tokenizer_schema_version = rd32();
        h.tokenizer_schema_hash = rd64();
        h.observation_schema_hash = rd64();
        h.action_schema_version = rd32();
        h.aux_target_schema_version = rd32();
        h.randomizer_type = rd32();
        h.termination_reason = rd32();
        h.self_play_seed = rd64();
        h.token_kind_order_hash = rd64();
        if (h.contract_version != DatasetHeader::CONTRACT_VERSION) {
            res.error = "unsupported dataset contract version";
            return res;
        }
        if (h.tokenizer_schema_version != schema::TOKENIZER_SCHEMA_VERSION ||
            h.tokenizer_schema_hash != schema::TOKENIZER_SCHEMA_HASH ||
            h.token_kind_order_hash != schema::TOKENIZER_SCHEMA_HASH ||
            h.observation_schema_hash != schema::OBSERVATION_SCHEMA_HASH ||
            h.action_schema_version != schema::ACTION_SCHEMA_VERSION) {
            res.error = "tokenizer/observation/action schema mismatch";
            return res;
        }
        if (h.aux_target_schema_version == schema::AUX_TARGET_SCHEMA_VERSION &&
            h.aux_targets != static_cast<std::uint32_t>(schema::AUX_TARGET_COUNT)) {
            res.error = "aux target width does not match its schema";
            return res;
        }
        if (h.aux_target_schema_version == schema::LEGACY_AUX_TARGET_SCHEMA_VERSION &&
            h.aux_targets != static_cast<std::uint32_t>(schema::LEGACY_AUX_TARGET_COUNT)) {
            res.error = "legacy aux target width does not match its schema";
            return res;
        }
        if (h.aux_target_schema_version != schema::AUX_TARGET_SCHEMA_VERSION &&
            h.aux_target_schema_version != schema::LEGACY_AUX_TARGET_SCHEMA_VERSION) {
            res.error = "unknown aux target schema";
            return res;
        }
    } else {
        h.aux_target_schema_version = schema::LEGACY_AUX_TARGET_SCHEMA_VERSION;
    }

    if (h.token_features != TOKEN_FEATURES || h.action_features != ACTION_FEATURES) {
        res.error = "feature width mismatch: the engine and the file disagree";
        return res;
    }

    const size_t n = h.samples, t = h.max_tokens, a = h.max_actions;
    const size_t float_count =
        n * t * TOKEN_FEATURES + n * t + n * a * ACTION_FEATURES + n * a +
        n * a + n + n * h.aux_targets +
        (h.version == DatasetHeader::VERSION ? n * h.aux_targets : 0);
    const size_t metadata_bytes =
        h.version == DatasetHeader::VERSION
            ? n * sizeof(std::int32_t) + n * sizeof(std::int32_t) +
                  n * sizeof(std::uint64_t) + n * sizeof(std::uint32_t)
            : 0;
    const size_t need = float_count * sizeof(float) + metadata_bytes;
    if (bytes.size() < at + need) {
        res.error = "truncated payload";
        return res;
    }

    auto rdf = [&](std::vector<float>& dst, size_t count) {
        dst.resize(count);
        std::memcpy(dst.data(), bytes.data() + at, count * sizeof(float));
        at += count * sizeof(float);
    };

    TensorBatch b;
    b.batch = static_cast<int>(n);
    b.max_tokens = static_cast<int>(t);
    b.max_actions = static_cast<int>(a);
    rdf(b.tokens, n * t * TOKEN_FEATURES);
    rdf(b.token_mask, n * t);
    rdf(b.actions, n * a * ACTION_FEATURES);
    rdf(b.action_mask, n * a);
    rdf(b.policy_target, n * a);
    rdf(b.value_target, n);
    rdf(b.aux_target, n * h.aux_targets);

    if (h.version == DatasetHeader::VERSION) {
        rdf(b.aux_valid_mask, n * h.aux_targets);
        auto rdi32 = [&]() {
            const std::uint32_t value = rd32();
            return static_cast<std::int32_t>(value);
        };
        b.player_perspective.resize(n);
        for (size_t i = 0; i < n; ++i) b.player_perspective[i] = rdi32();
        b.termination_reason.resize(n);
        for (size_t i = 0; i < n; ++i) b.termination_reason[i] = rdi32();
        b.game_seed.resize(n);
        for (size_t i = 0; i < n; ++i) b.game_seed[i] = rd64();
        b.move_number.resize(n);
        for (size_t i = 0; i < n; ++i) b.move_number[i] = rd32();
    } else {
        b.aux_valid_mask.assign(n * h.aux_targets, 1.0f);
        b.player_perspective.assign(n, 1);
        b.termination_reason.assign(
            n, static_cast<std::int32_t>(TerminationReason::Unknown));
        b.game_seed.assign(n, 0);
        b.move_number.assign(n, 0);
    }

    res.ok = true;
    res.header = h;
    res.batch = std::move(b);
    return res;
}

inline DatasetReadResult read_dataset_file(const std::string& path) {
    DatasetReadResult res;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        res.error = "cannot open " + path;
        return res;
    }
    std::vector<std::uint8_t> bytes;
    std::uint8_t buf[65536];
    size_t got = 0;
    while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0)
        bytes.insert(bytes.end(), buf, buf + got);
    std::fclose(f);
    return deserialize_dataset(bytes);
}

inline DatasetContract dataset_contract_from_samples(
    const std::vector<const TrainingSample*>& samples) {
    DatasetContract contract;
    if (samples.empty()) return contract;

    const TrainingSample& first = *samples.front();
    contract.tokenizer_schema_version = first.tokenizer_schema_version;
    contract.aux_target_schema_version = first.aux_target_schema_version;
    contract.randomizer_type = first.randomizer_type;
    contract.termination_reason = first.termination_reason;
    contract.self_play_seed = first.game_seed;

    for (const TrainingSample* sample : samples) {
        if (!sample) continue;
        if (sample->tokenizer_schema_version != contract.tokenizer_schema_version)
            contract.tokenizer_schema_version = 0;
        if (sample->aux_target_schema_version != contract.aux_target_schema_version)
            contract.aux_target_schema_version = 0;
        if (sample->randomizer_type != contract.randomizer_type)
            contract.randomizer_type = 0;
        if (sample->termination_reason != contract.termination_reason)
            contract.termination_reason = TerminationReason::Mixed;
        if (sample->game_seed != contract.self_play_seed) contract.self_play_seed = 0;
    }

    if (contract.tokenizer_schema_version == 0) contract.tokenizer_schema_hash = 0;
    return contract;
}

// Convenience: export a whole replay buffer as one padded dataset.
// By default (`compact = true`), writes V2 Compact Replay + π format when
// possible, cutting disk usage and I/O by ~100x.
inline bool export_buffer(const std::string& path, const ReplayBuffer& buffer,
                          std::uint32_t model_version = 0, int pad_tokens = 0,
                          int pad_actions = 0, bool compact = true) {
    std::vector<const TrainingSample*> all;
    all.reserve(buffer.size());
    for (size_t i = 0; i < buffer.size(); ++i) all.push_back(&buffer.at(i));
    if (all.empty()) return false;
    if (compact && pad_tokens == 0 && pad_actions == 0 && can_export_compact(all)) {
        return write_compact_dataset_file(path, all, model_version);
    }
    const TensorBatch batch = make_training_batch(all, pad_tokens, pad_actions);
    DatasetContract contract = dataset_contract_from_samples(all);
    // Legacy samples are widened to the current rectangular layout with the
    // extra targets masked out.  Record that representation explicitly so
    // the v3 reader does not mistake a 36-wide payload for schema v1.
    if (contract.aux_target_schema_version ==
            schema::LEGACY_AUX_TARGET_SCHEMA_VERSION &&
        batch.aux_target.size() ==
            batch.batch * static_cast<std::size_t>(TensorBatch::AUX_TARGETS)) {
        contract.aux_target_schema_version = schema::AUX_TARGET_SCHEMA_VERSION;
    }
    if (contract.tokenizer_schema_version == 0 || contract.aux_target_schema_version == 0)
        return false;
    return write_dataset_file(path, batch, all.front()->ruleset_hash, model_version,
                              contract);
}

}  // namespace tetra
