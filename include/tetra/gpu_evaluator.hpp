// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- synchronous pipe bridge to a Python GPU evaluator.
//
// The rule/search core stays in C++; Python/Torch owns the accelerator.  The
// bridge is deliberately a small binary protocol so a whole Searcher leaf
// batch crosses the boundary at once instead of paying a process round trip per
// position.
#pragma once

#include "tetra/batch.hpp"
#include "tetra/evaluator.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace tetra {
namespace gpu_protocol {

inline constexpr char REQUEST_MAGIC[4] = {'T', 'G', 'P', 'U'};
inline constexpr char RESPONSE_MAGIC[4] = {'T', 'G', 'P', 'R'};
inline constexpr char RESULT_MAGIC[4] = {'T', 'G', 'S', 'T'};
inline constexpr char EXPORT_MAGIC[4] = {'T', 'G', 'E', 'D'};
inline constexpr char ARENA_MAGIC[4] = {'T', 'G', 'A', 'R'};

inline void write_exact(FILE* f, const void* data, size_t bytes) {
    if (bytes != 0 && std::fwrite(data, 1, bytes, f) != bytes)
        throw std::runtime_error("GPU evaluator pipe write failed");
}

inline void read_exact(FILE* f, void* data, size_t bytes) {
    if (bytes != 0 && std::fread(data, 1, bytes, f) != bytes)
        throw std::runtime_error("GPU evaluator pipe closed or truncated");
}

inline void write_u32(FILE* f, std::uint32_t v) {
    const std::uint8_t b[4] = {static_cast<std::uint8_t>(v),
                               static_cast<std::uint8_t>(v >> 8),
                               static_cast<std::uint8_t>(v >> 16),
                               static_cast<std::uint8_t>(v >> 24)};
    write_exact(f, b, sizeof(b));
}

inline std::uint32_t read_u32(FILE* f) {
    std::uint8_t b[4]{};
    read_exact(f, b, sizeof(b));
    return static_cast<std::uint32_t>(b[0]) |
           (static_cast<std::uint32_t>(b[1]) << 8) |
           (static_cast<std::uint32_t>(b[2]) << 16) |
           (static_cast<std::uint32_t>(b[3]) << 24);
}

inline void write_i64(FILE* f, std::int64_t v) {
    const std::uint64_t u = static_cast<std::uint64_t>(v);
    const std::uint8_t b[8] = {
        static_cast<std::uint8_t>(u), static_cast<std::uint8_t>(u >> 8),
        static_cast<std::uint8_t>(u >> 16), static_cast<std::uint8_t>(u >> 24),
        static_cast<std::uint8_t>(u >> 32), static_cast<std::uint8_t>(u >> 40),
        static_cast<std::uint8_t>(u >> 48), static_cast<std::uint8_t>(u >> 56)};
    write_exact(f, b, sizeof(b));
}

inline void write_u64(FILE* f, std::uint64_t u) {
    const std::uint8_t b[8] = {
        static_cast<std::uint8_t>(u), static_cast<std::uint8_t>(u >> 8),
        static_cast<std::uint8_t>(u >> 16), static_cast<std::uint8_t>(u >> 24),
        static_cast<std::uint8_t>(u >> 32), static_cast<std::uint8_t>(u >> 40),
        static_cast<std::uint8_t>(u >> 48), static_cast<std::uint8_t>(u >> 56)};
    write_exact(f, b, sizeof(b));
}

inline std::int64_t read_i64(FILE* f) {
    std::uint8_t b[8]{};
    read_exact(f, b, sizeof(b));
    std::uint64_t u = 0;
    for (int i = 0; i < 8; ++i)
        u |= static_cast<std::uint64_t>(b[i]) << (i * 8);
    return static_cast<std::int64_t>(u);
}

inline void write_float(FILE* f, float v) { write_exact(f, &v, sizeof(v)); }

inline float read_float(FILE* f) {
    float v = 0.0f;
    read_exact(f, &v, sizeof(v));
    return v;
}

inline void write_floats(FILE* f, const std::vector<float>& v) {
    write_exact(f, v.data(), v.size() * sizeof(float));
}

inline void read_floats(FILE* f, std::vector<float>& v) {
    read_exact(f, v.data(), v.size() * sizeof(float));
}

inline void check_magic(FILE* f, const char (&expected)[4]) {
    char got[4]{};
    read_exact(f, got, sizeof(got));
    for (int i = 0; i < 4; ++i) {
        if (got[i] != expected[i]) throw std::runtime_error("invalid GPU evaluator frame");
    }
}

}  // namespace gpu_protocol

// A synchronous Evaluator implementation whose other end is trainer/gpu_match.py.
// It is intended for local evaluation/self-play, not for an untrusted input
// stream: the child process is launched by the local wrapper itself.
class RemoteGpuEvaluator final : public Evaluator {
public:
    RemoteGpuEvaluator(FILE* input, FILE* output, int preferred_batch = 16,
                       std::uint32_t model_id = 0)
        : input_(input),
          output_(output),
          preferred_batch_(std::max(1, preferred_batch)),
          model_id_(model_id) {}

    void evaluate(const std::vector<EvalRequest>& batch,
                  std::vector<Evaluation>& out) override {
        account(batch.size());
        const TensorBatch packed = make_inference_batch(batch, tokenizer_);

        gpu_protocol::write_exact(output_, gpu_protocol::REQUEST_MAGIC,
                                  sizeof(gpu_protocol::REQUEST_MAGIC));
        // The ID lets one Python GPU server host both sides of an Arena while
        // keeping a single binary pipe and a single GPU context.
        gpu_protocol::write_u32(output_, model_id_);
        gpu_protocol::write_u32(output_, static_cast<std::uint32_t>(packed.batch));
        gpu_protocol::write_u32(output_, static_cast<std::uint32_t>(packed.max_tokens));
        gpu_protocol::write_u32(output_, static_cast<std::uint32_t>(packed.max_actions));
        gpu_protocol::write_floats(output_, packed.tokens);
        gpu_protocol::write_floats(output_, packed.token_mask);
        gpu_protocol::write_floats(output_, packed.actions);
        gpu_protocol::write_floats(output_, packed.action_mask);
        if (std::fflush(output_) != 0) throw std::runtime_error("GPU evaluator flush failed");

        gpu_protocol::check_magic(input_, gpu_protocol::RESPONSE_MAGIC);
        const std::uint32_t returned = gpu_protocol::read_u32(input_);
        if (returned != static_cast<std::uint32_t>(packed.batch))
            throw std::runtime_error("GPU evaluator batch mismatch");

        out.assign(batch.size(), Evaluation{});
        for (size_t i = 0; i < batch.size(); ++i) {
            const std::uint32_t action_count = gpu_protocol::read_u32(input_);
            const size_t expected = batch[i].actions ? batch[i].actions->size() : 0;
            if (action_count != expected)
                throw std::runtime_error("GPU evaluator action count mismatch");

            Evaluation& ev = out[i];
            ev.policy.resize(expected);
            gpu_protocol::read_floats(input_, ev.policy);
            ev.value.win = gpu_protocol::read_float(input_);
            ev.value.draw = gpu_protocol::read_float(input_);
            ev.value.loss = gpu_protocol::read_float(input_);
            ev.value.normalize();
            ev.aux.expected_net_attack_1s = gpu_protocol::read_float(input_);
            ev.aux.expected_received_garbage = gpu_protocol::read_float(input_);
            ev.aux.expected_time_to_ko = gpu_protocol::read_float(input_);
            ev.aux.topout_within_8_pieces = gpu_protocol::read_float(input_);
        }
    }

    int preferred_batch_size() const override { return preferred_batch_; }
    std::string name() const override { return "gpu-remote"; }

private:
    FILE* input_ = nullptr;
    FILE* output_ = nullptr;
    int preferred_batch_ = 16;
    std::uint32_t model_id_ = 0;
    Tokenizer tokenizer_;
};

inline void enable_gpu_protocol_stdio() {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    std::setvbuf(stdin, nullptr, _IONBF, 0);
    std::setvbuf(stdout, nullptr, _IONBF, 0);
}

// Final frame for the Python wrapper.  All game counters come from the C++
// simulator; Python only formats them and calculates display metrics.
inline void write_gpu_game_result(FILE* output, int pieces, int lines_cleared,
                                  int garbage_lines_cleared, int lines_sent,
                                  int lines_received, bool survived,
                                  int topout, int tick_rate, float outcome, Tick duration,
                                  std::uint64_t positions, std::uint64_t batches) {
    gpu_protocol::write_exact(output, gpu_protocol::RESULT_MAGIC,
                              sizeof(gpu_protocol::RESULT_MAGIC));
    gpu_protocol::write_u32(output, static_cast<std::uint32_t>(pieces));
    gpu_protocol::write_u32(output, static_cast<std::uint32_t>(lines_cleared));
    gpu_protocol::write_u32(output, static_cast<std::uint32_t>(garbage_lines_cleared));
    gpu_protocol::write_u32(output, static_cast<std::uint32_t>(lines_sent));
    gpu_protocol::write_u32(output, static_cast<std::uint32_t>(lines_received));
    gpu_protocol::write_u32(output, survived ? 1u : 0u);
    gpu_protocol::write_u32(output, static_cast<std::uint32_t>(topout));
    gpu_protocol::write_u32(output, static_cast<std::uint32_t>(tick_rate));
    gpu_protocol::write_float(output, outcome);
    gpu_protocol::write_i64(output, duration);
    gpu_protocol::write_u64(output, positions);
    gpu_protocol::write_u64(output, batches);
    if (std::fflush(output) != 0) throw std::runtime_error("GPU result flush failed");
}

// Final frame for the GPU self-play data generator.  The dataset itself is
// written by the C++ child, so only its summary crosses the pipe.
inline void write_gpu_export_result(FILE* output, std::uint32_t games,
                                    std::uint32_t samples, std::uint64_t positions,
                                    std::uint64_t batches) {
    gpu_protocol::write_exact(output, gpu_protocol::EXPORT_MAGIC,
                              sizeof(gpu_protocol::EXPORT_MAGIC));
    gpu_protocol::write_u32(output, games);
    gpu_protocol::write_u32(output, samples);
    gpu_protocol::write_u64(output, positions);
    gpu_protocol::write_u64(output, batches);
    if (std::fflush(output) != 0) throw std::runtime_error("GPU export flush failed");
}

// Final frame for the GPU-backed Arena.  The Arena itself remains in C++ so
// paired games, mirror seeds, and promotion arithmetic cannot drift from the
// CPU implementation; only the two model evaluations cross the pipe.
inline void write_gpu_arena_result(FILE* output, std::uint32_t games_played,
                                   std::uint32_t candidate_wins,
                                   std::uint32_t champion_wins, std::uint32_t draws,
                                   float win_rate, float ci_lower, float ci_upper,
                                   float candidate_vs, float champion_vs,
                                   float candidate_apm, float champion_apm,
                                   float candidate_app, float champion_app,
                                   float candidate_pps, float champion_pps,
                                   float candidate_avg_pieces, float champion_avg_pieces,
                                   float candidate_avg_seconds, float champion_avg_seconds,
                                   float candidate_survival_rate, float champion_survival_rate,
                                   float candidate_sent_per_game, float champion_sent_per_game,
                                   float candidate_garbage_cleared_per_game,
                                   float champion_garbage_cleared_per_game,
                                   float candidate_received_per_game,
                                   float champion_received_per_game,
                                   float candidate_blockout_rate, float champion_blockout_rate,
                                   float candidate_lockout_rate, float champion_lockout_rate,
                                   float candidate_garbageout_rate, float champion_garbageout_rate,
                                   bool promoted) {
    gpu_protocol::write_exact(output, gpu_protocol::ARENA_MAGIC,
                              sizeof(gpu_protocol::ARENA_MAGIC));
    gpu_protocol::write_u32(output, games_played);
    gpu_protocol::write_u32(output, candidate_wins);
    gpu_protocol::write_u32(output, champion_wins);
    gpu_protocol::write_u32(output, draws);
    gpu_protocol::write_float(output, win_rate);
    gpu_protocol::write_float(output, ci_lower);
    gpu_protocol::write_float(output, ci_upper);
    gpu_protocol::write_float(output, candidate_vs);
    gpu_protocol::write_float(output, champion_vs);
    gpu_protocol::write_float(output, candidate_apm);
    gpu_protocol::write_float(output, champion_apm);
    gpu_protocol::write_float(output, candidate_app);
    gpu_protocol::write_float(output, champion_app);
    gpu_protocol::write_float(output, candidate_pps);
    gpu_protocol::write_float(output, champion_pps);
    gpu_protocol::write_float(output, candidate_avg_pieces);
    gpu_protocol::write_float(output, champion_avg_pieces);
    gpu_protocol::write_float(output, candidate_avg_seconds);
    gpu_protocol::write_float(output, champion_avg_seconds);
    gpu_protocol::write_float(output, candidate_survival_rate);
    gpu_protocol::write_float(output, champion_survival_rate);
    gpu_protocol::write_float(output, candidate_sent_per_game);
    gpu_protocol::write_float(output, champion_sent_per_game);
    gpu_protocol::write_float(output, candidate_garbage_cleared_per_game);
    gpu_protocol::write_float(output, champion_garbage_cleared_per_game);
    gpu_protocol::write_float(output, candidate_received_per_game);
    gpu_protocol::write_float(output, champion_received_per_game);
    gpu_protocol::write_float(output, candidate_blockout_rate);
    gpu_protocol::write_float(output, champion_blockout_rate);
    gpu_protocol::write_float(output, candidate_lockout_rate);
    gpu_protocol::write_float(output, champion_lockout_rate);
    gpu_protocol::write_float(output, candidate_garbageout_rate);
    gpu_protocol::write_float(output, champion_garbageout_rate);
    gpu_protocol::write_u32(output, promoted ? 1u : 0u);
    if (std::fflush(output) != 0) throw std::runtime_error("GPU Arena flush failed");
}

}  // namespace tetra
