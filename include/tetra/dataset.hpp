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
//
// Everything is little-endian float32 (or int32 for the header), which is what
// both numpy and libtorch expect natively on the platforms this runs on.
#pragma once

#include "tetra/batch.hpp"
#include "tetra/replay_buffer.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace tetra {

struct DatasetHeader {
    static constexpr char MAGIC[8] = {'T', 'E', 'T', 'R', 'A', 'D', 'A', 'T'};
    static constexpr std::uint32_t VERSION = 1;

    std::uint32_t version = VERSION;
    std::uint32_t samples = 0;
    std::uint32_t max_tokens = 0;
    std::uint32_t max_actions = 0;
    std::uint32_t token_features = TOKEN_FEATURES;
    std::uint32_t action_features = ACTION_FEATURES;
    std::uint32_t aux_targets = TensorBatch::AUX_TARGETS;
    std::uint64_t ruleset_hash = 0;
    std::uint32_t model_version = 0;
};

namespace detail {

inline void put32(std::vector<std::uint8_t>& b, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<std::uint8_t>((v >> (i * 8)) & 0xFF));
}
inline void put64d(std::vector<std::uint8_t>& b, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<std::uint8_t>((v >> (i * 8)) & 0xFF));
}
inline void put_floats(std::vector<std::uint8_t>& b, const std::vector<float>& v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(v.data());
    b.insert(b.end(), p, p + v.size() * sizeof(float));
}

}  // namespace detail

// Serialise a training batch. The batch must have been built with targets.
inline std::vector<std::uint8_t> serialize_dataset(const TensorBatch& batch,
                                                   std::uint64_t ruleset_hash,
                                                   std::uint32_t model_version) {
    std::vector<std::uint8_t> out;
    for (char c : DatasetHeader::MAGIC) out.push_back(static_cast<std::uint8_t>(c));
    detail::put32(out, DatasetHeader::VERSION);
    detail::put32(out, static_cast<std::uint32_t>(batch.batch));
    detail::put32(out, static_cast<std::uint32_t>(batch.max_tokens));
    detail::put32(out, static_cast<std::uint32_t>(batch.max_actions));
    detail::put32(out, static_cast<std::uint32_t>(TOKEN_FEATURES));
    detail::put32(out, static_cast<std::uint32_t>(ACTION_FEATURES));
    detail::put32(out, static_cast<std::uint32_t>(TensorBatch::AUX_TARGETS));
    detail::put64d(out, ruleset_hash);
    detail::put32(out, model_version);

    detail::put_floats(out, batch.tokens);
    detail::put_floats(out, batch.token_mask);
    detail::put_floats(out, batch.actions);
    detail::put_floats(out, batch.action_mask);
    detail::put_floats(out, batch.policy_target);
    detail::put_floats(out, batch.value_target);
    detail::put_floats(out, batch.aux_target);
    return out;
}

inline bool write_dataset_file(const std::string& path, const TensorBatch& batch,
                               std::uint64_t ruleset_hash, std::uint32_t model_version) {
    const std::vector<std::uint8_t> bytes =
        serialize_dataset(batch, ruleset_hash, model_version);
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
    if (h.version != DatasetHeader::VERSION) {
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

    if (h.token_features != TOKEN_FEATURES || h.action_features != ACTION_FEATURES) {
        res.error = "feature width mismatch: the engine and the file disagree";
        return res;
    }

    const size_t n = h.samples, t = h.max_tokens, a = h.max_actions;
    const size_t need = (n * t * TOKEN_FEATURES + n * t + n * a * ACTION_FEATURES + n * a +
                         n * a + n + n * h.aux_targets) *
                        sizeof(float);
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

// Convenience: export a whole replay buffer as one padded dataset.
inline bool export_buffer(const std::string& path, const ReplayBuffer& buffer,
                          std::uint32_t model_version = 0, int pad_tokens = 0,
                          int pad_actions = 0) {
    std::vector<const TrainingSample*> all;
    all.reserve(buffer.size());
    for (size_t i = 0; i < buffer.size(); ++i) all.push_back(&buffer.at(i));
    if (all.empty()) return false;
    const TensorBatch batch = make_training_batch(all, pad_tokens, pad_actions);
    return write_dataset_file(path, batch, all.front()->ruleset_hash, model_version);
}

}  // namespace tetra