// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- C++ inference for trained weights (spec 16, 17).
//
// This closes the loop: train in PyTorch, export the weights, and run the same
// network inside the C++ search as an ordinary `Evaluator`.
//
// It is a from-scratch forward pass with **no third-party dependency**. That is
// a deliberate trade against ONNX Runtime or libtorch:
//
//   * `git clone && make` keeps working on any machine with a C++17 compiler,
//     which is what makes the engine easy to run and easy to test in CI;
//   * the ROCm/ONNX story for RDNA4 is still moving, and pinning the engine to
//     it would make the whole project hostage to that toolchain;
//   * self-play generation is latency-bound on small batches, where the
//     dispatch overhead of a heavyweight runtime is a real cost.
//
// The price is that this path is CPU-only and unoptimised, so *training* and
// any large-scale evaluation belong in Python on the GPU. `tests/test_nnue.cpp`
// pins the two implementations to agree numerically, so the C++ path cannot
// silently diverge from the model that was trained.
#pragma once

#include "tetra/batch.hpp"
#include "tetra/evaluator.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace tetra {

// ---------------------------------------------------------------------------
// Weight file
// ---------------------------------------------------------------------------
// Format: "TETRAWTS" | version | config | then every tensor in a fixed order,
// each as [name_len][name][ndim][dims...][float32 data]. Named tensors make the
// loader able to verify it received what it expected rather than trusting an
// offset table.
struct NnueConfig {
    std::uint32_t token_features = TOKEN_FEATURES;
    std::uint32_t action_features = ACTION_FEATURES;
    std::uint32_t width = 64;
    std::uint32_t layers = 2;
    std::uint32_t heads = 4;
    std::uint32_t ffn = 192;
    std::uint32_t aux_targets = TensorBatch::AUX_TARGETS;
};

struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;

    size_t rows() const { return shape.empty() ? 0 : static_cast<size_t>(shape[0]); }
    size_t cols() const { return shape.size() < 2 ? 1 : static_cast<size_t>(shape[1]); }
    bool is(std::initializer_list<int> s) const {
        return shape.size() == s.size() &&
               std::equal(shape.begin(), shape.end(), s.begin());
    }
};

namespace detail {

// out[m, n] = x[m, k] @ w[n, k]^T + b[n]   (PyTorch Linear layout)
inline void linear(const float* x, size_t m, size_t k, const Tensor& w, const Tensor* b,
                   float* out) {
    const size_t n = w.rows();
    for (size_t i = 0; i < m; ++i) {
        const float* xi = x + i * k;
        float* oi = out + i * n;
        for (size_t j = 0; j < n; ++j) {
            const float* wj = w.data.data() + j * k;
            float acc = b ? b->data[j] : 0.0f;
            for (size_t t = 0; t < k; ++t) acc += xi[t] * wj[t];
            oi[j] = acc;
        }
    }
}

inline void rmsnorm(float* x, size_t m, size_t d, const Tensor& weight, float eps = 1e-6f) {
    for (size_t i = 0; i < m; ++i) {
        float* xi = x + i * d;
        float sum = 0.0f;
        for (size_t j = 0; j < d; ++j) sum += xi[j] * xi[j];
        const float scale = 1.0f / std::sqrt(sum / static_cast<float>(d) + eps);
        for (size_t j = 0; j < d; ++j) xi[j] = xi[j] * scale * weight.data[j];
    }
}

inline float silu(float v) { return v / (1.0f + std::exp(-v)); }

// Multi-head attention with an additive key-padding mask.
// q: [mq, d], kv: [mk, d]. `key_mask` is 1 for real keys and 0 for padding.
inline void attention(const float* q_in, size_t mq, const float* kv_in, size_t mk, size_t d,
                      size_t heads, const Tensor& in_proj_w, const Tensor& in_proj_b,
                      const Tensor& out_proj_w, const Tensor& out_proj_b,
                      const float* key_mask, float* out, std::vector<float>& scratch) {
    const size_t hd = d / heads;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));

    // PyTorch packs Q, K, V into one [3d, d] projection.
    scratch.assign(mq * d + mk * d * 2, 0.0f);
    float* Q = scratch.data();
    float* K = Q + mq * d;
    float* V = K + mk * d;

    for (size_t i = 0; i < mq; ++i)
        for (size_t j = 0; j < d; ++j) {
            const float* w = in_proj_w.data.data() + j * d;
            float acc = in_proj_b.data[j];
            for (size_t t = 0; t < d; ++t) acc += q_in[i * d + t] * w[t];
            Q[i * d + j] = acc;
        }
    for (size_t i = 0; i < mk; ++i)
        for (size_t j = 0; j < d; ++j) {
            const float* wk = in_proj_w.data.data() + (d + j) * d;
            const float* wv = in_proj_w.data.data() + (2 * d + j) * d;
            float ak = in_proj_b.data[d + j], av = in_proj_b.data[2 * d + j];
            for (size_t t = 0; t < d; ++t) {
                ak += kv_in[i * d + t] * wk[t];
                av += kv_in[i * d + t] * wv[t];
            }
            K[i * d + j] = ak;
            V[i * d + j] = av;
        }

    std::vector<float> ctx(mq * d, 0.0f);
    std::vector<float> logits(mk);
    for (size_t h = 0; h < heads; ++h) {
        const size_t off = h * hd;
        for (size_t i = 0; i < mq; ++i) {
            float best = -1e30f;
            for (size_t j = 0; j < mk; ++j) {
                if (key_mask && key_mask[j] < 0.5f) {
                    logits[j] = -1e30f;
                    continue;
                }
                float dot = 0.0f;
                for (size_t t = 0; t < hd; ++t)
                    dot += Q[i * d + off + t] * K[j * d + off + t];
                logits[j] = dot * inv_sqrt;
                best = std::max(best, logits[j]);
            }
            float sum = 0.0f;
            for (size_t j = 0; j < mk; ++j) {
                logits[j] = (logits[j] <= -1e29f) ? 0.0f : std::exp(logits[j] - best);
                sum += logits[j];
            }
            if (sum <= 0.0f) sum = 1.0f;
            for (size_t j = 0; j < mk; ++j) {
                const float a = logits[j] / sum;
                if (a == 0.0f) continue;
                for (size_t t = 0; t < hd; ++t)
                    ctx[i * d + off + t] += a * V[j * d + off + t];
            }
        }
    }
    linear(ctx.data(), mq, d, out_proj_w, &out_proj_b, out);
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------
class NnueWeights {
public:
    static constexpr char MAGIC[8] = {'T', 'E', 'T', 'R', 'A', 'W', 'T', 'S'};
    static constexpr std::uint32_t VERSION = 1;

    bool load(const std::string& path, std::string* error = nullptr) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) {
            if (error) *error = "cannot open " + path;
            return false;
        }
        std::vector<std::uint8_t> b;
        std::uint8_t buf[65536];
        size_t got = 0;
        while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0) b.insert(b.end(), buf, buf + got);
        std::fclose(f);
        return parse(b, error);
    }

    bool parse(const std::vector<std::uint8_t>& b, std::string* error = nullptr) {
        auto fail = [&](const char* m) {
            if (error) *error = m;
            return false;
        };
        if (b.size() < 8 + 4 + 28) return fail("too short");
        if (std::memcmp(b.data(), MAGIC, 8) != 0) return fail("bad magic");

        size_t at = 8;
        auto u32 = [&]() {
            std::uint32_t v = 0;
            for (int i = 0; i < 4; ++i)
                v |= static_cast<std::uint32_t>(b[at + static_cast<size_t>(i)]) << (i * 8);
            at += 4;
            return v;
        };
        if (u32() != VERSION) return fail("unsupported version");

        cfg_.token_features = u32();
        cfg_.action_features = u32();
        cfg_.width = u32();
        cfg_.layers = u32();
        cfg_.heads = u32();
        cfg_.ffn = u32();
        cfg_.aux_targets = u32();

        if (cfg_.token_features != TOKEN_FEATURES || cfg_.action_features != ACTION_FEATURES)
            return fail("feature width mismatch between the weights and the engine");
        if (cfg_.width == 0 || cfg_.heads == 0 || cfg_.width % cfg_.heads != 0)
            return fail("invalid width/heads");

        const std::uint32_t count = u32();
        for (std::uint32_t i = 0; i < count; ++i) {
            if (at + 4 > b.size()) return fail("truncated");
            const std::uint32_t name_len = u32();
            if (at + name_len > b.size()) return fail("truncated name");
            std::string name(reinterpret_cast<const char*>(b.data() + at), name_len);
            at += name_len;

            const std::uint32_t ndim = u32();
            Tensor t;
            size_t n = 1;
            for (std::uint32_t d = 0; d < ndim; ++d) {
                const std::uint32_t dim = u32();
                t.shape.push_back(static_cast<int>(dim));
                n *= dim;
            }
            if (at + n * sizeof(float) > b.size()) return fail("truncated tensor data");
            t.data.resize(n);
            std::memcpy(t.data.data(), b.data() + at, n * sizeof(float));
            at += n * sizeof(float);
            tensors_[name] = std::move(t);
        }
        return validate(error);
    }

    const NnueConfig& config() const { return cfg_; }
    bool has(const std::string& n) const { return tensors_.count(n) != 0; }
    const Tensor& get(const std::string& n) const {
        static const Tensor empty;
        auto it = tensors_.find(n);
        return it == tensors_.end() ? empty : it->second;
    }
    size_t tensor_count() const { return tensors_.size(); }

private:
    bool validate(std::string* error) {
        auto need = [&](const std::string& n) {
            if (!has(n)) {
                if (error) *error = "missing tensor: " + n;
                return false;
            }
            return true;
        };
        if (!need("token_in.weight") || !need("token_in.bias")) return false;
        if (!need("action_in.weight") || !need("policy_out.weight")) return false;
        if (!need("norm.weight") || !need("policy_norm.weight")) return false;
        for (std::uint32_t l = 0; l < cfg_.layers; ++l) {
            const std::string p = "blocks." + std::to_string(l) + ".";
            for (const char* s : {"n1.weight", "n2.weight", "attn.in_proj_weight",
                                  "attn.in_proj_bias", "attn.out_proj.weight",
                                  "attn.out_proj.bias", "ffn.gate.weight", "ffn.up.weight",
                                  "ffn.down.weight"})
                if (!need(p + s)) return false;
        }
        return true;
    }

    NnueConfig cfg_{};
    std::map<std::string, Tensor> tensors_;
};

// ---------------------------------------------------------------------------
// The evaluator
// ---------------------------------------------------------------------------
class TetraFormerEvaluator : public Evaluator {
public:
    explicit TetraFormerEvaluator(NnueWeights weights, std::string name = "tetraformer")
        : w_(std::move(weights)), name_(std::move(name)) {}

    static bool load(const std::string& path, TetraFormerEvaluator** out,
                     std::string* error = nullptr) {
        NnueWeights w;
        if (!w.load(path, error)) return false;
        *out = new TetraFormerEvaluator(std::move(w), path);
        return true;
    }

    void evaluate(const std::vector<EvalRequest>& batch, std::vector<Evaluation>& out) override {
        account(batch.size());
        out.assign(batch.size(), Evaluation{});
        for (size_t i = 0; i < batch.size(); ++i) {
            if (!batch[i].observation || !batch[i].actions) {
                out[i].value.draw = 1.0f;
                continue;
            }
            forward(*batch[i].observation, *batch[i].actions, out[i]);
        }
    }

    std::string name() const override { return name_; }
    int preferred_batch_size() const override { return 16; }
    const NnueConfig& config() const { return w_.config(); }

private:
    void forward(const Observation& obs, const std::vector<PlacementAction>& actions,
                 Evaluation& ev) {
        const NnueConfig& c = w_.config();
        const size_t d = c.width;

        // --- tokenize ---
        const TokenizedObservation tokens = tok_.encode(obs, obs.ruleset);
        const size_t T = tokens.tokens.size();
        const size_t A = actions.size();
        if (T == 0) {
            ev.value.draw = 1.0f;
            ev.policy.assign(A, A ? 1.0f / static_cast<float>(A) : 0.0f);
            return;
        }

        std::vector<float> tok_in(T * TOKEN_FEATURES, 0.0f);
        for (size_t i = 0; i < T; ++i)
            for (int k = 0; k < TOKEN_FEATURES; ++k)
                tok_in[i * TOKEN_FEATURES + static_cast<size_t>(k)] =
                    tokens.tokens[i].f[static_cast<size_t>(k)];

        // --- encoder ---
        std::vector<float> x(T * d);
        detail::linear(tok_in.data(), T, TOKEN_FEATURES, w_.get("token_in.weight"),
                       &w_.get("token_in.bias"), x.data());

        std::vector<float> h(T * d), attn_out(T * d), gate(T * c.ffn), up(T * c.ffn),
            ffn_out(T * d), scratch;
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            const std::string p = "blocks." + std::to_string(l) + ".";

            h = x;
            detail::rmsnorm(h.data(), T, d, w_.get(p + "n1.weight"));
            detail::attention(h.data(), T, h.data(), T, d, c.heads,
                              w_.get(p + "attn.in_proj_weight"), w_.get(p + "attn.in_proj_bias"),
                              w_.get(p + "attn.out_proj.weight"), w_.get(p + "attn.out_proj.bias"),
                              /*key_mask=*/nullptr, attn_out.data(), scratch);
            for (size_t i = 0; i < T * d; ++i) x[i] += attn_out[i];

            h = x;
            detail::rmsnorm(h.data(), T, d, w_.get(p + "n2.weight"));
            detail::linear(h.data(), T, d, w_.get(p + "ffn.gate.weight"), nullptr, gate.data());
            detail::linear(h.data(), T, d, w_.get(p + "ffn.up.weight"), nullptr, up.data());
            for (size_t i = 0; i < T * c.ffn; ++i) gate[i] = detail::silu(gate[i]) * up[i];
            detail::linear(gate.data(), T, c.ffn, w_.get(p + "ffn.down.weight"), nullptr,
                           ffn_out.data());
            for (size_t i = 0; i < T * d; ++i) x[i] += ffn_out[i];
        }
        detail::rmsnorm(x.data(), T, d, w_.get("norm.weight"));

        // --- value and auxiliary heads (masked mean pool) ---
        std::vector<float> pooled(d, 0.0f);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d; ++j) pooled[j] += x[i * d + j];
        for (size_t j = 0; j < d; ++j) pooled[j] /= static_cast<float>(T);

        ev.value = head_wdl(pooled);
        fill_aux(pooled, ev.aux);

        // --- variable-length policy head ---
        if (A == 0) return;
        std::vector<float> act_in(A * ACTION_FEATURES, 0.0f);
        for (size_t i = 0; i < A; ++i) {
            const ActionEmbedding e = embed_action(actions[i], obs.board, obs.ruleset);
            for (int k = 0; k < ACTION_FEATURES; ++k)
                act_in[i * ACTION_FEATURES + static_cast<size_t>(k)] = e.f[static_cast<size_t>(k)];
        }
        std::vector<float> q(A * d);
        detail::linear(act_in.data(), A, ACTION_FEATURES, w_.get("action_in.weight"),
                       &w_.get("action_in.bias"), q.data());

        std::vector<float> pattn(A * d);
        detail::attention(q.data(), A, x.data(), T, d, c.heads,
                          w_.get("policy_attn.in_proj_weight"),
                          w_.get("policy_attn.in_proj_bias"),
                          w_.get("policy_attn.out_proj.weight"),
                          w_.get("policy_attn.out_proj.bias"),
                          /*key_mask=*/nullptr, pattn.data(), scratch);
        for (size_t i = 0; i < A * d; ++i) pattn[i] += q[i];
        detail::rmsnorm(pattn.data(), A, d, w_.get("policy_norm.weight"));

        std::vector<float> logits(A);
        detail::linear(pattn.data(), A, d, w_.get("policy_out.weight"),
                       w_.has("policy_out.bias") ? &w_.get("policy_out.bias") : nullptr,
                       logits.data());

        float best = -1e30f;
        for (float v : logits) best = std::max(best, v);
        float sum = 0.0f;
        ev.policy.resize(A);
        for (size_t i = 0; i < A; ++i) {
            ev.policy[i] = std::exp(logits[i] - best);
            sum += ev.policy[i];
        }
        for (float& v : ev.policy) v /= (sum > 0.0f ? sum : 1.0f);
    }

    ValueWDL head_wdl(const std::vector<float>& pooled) {
        const std::vector<float> out = mlp(pooled, "value_head");
        if (out.size() != 3) return ValueWDL::from_scalar(0.0f);
        float best = std::max(out[0], std::max(out[1], out[2]));
        float e0 = std::exp(out[0] - best), e1 = std::exp(out[1] - best),
              e2 = std::exp(out[2] - best);
        const float s = e0 + e1 + e2;
        ValueWDL v;
        v.win = e0 / s;
        v.draw = e1 / s;
        v.loss = e2 / s;
        return v;
    }

    void fill_aux(const std::vector<float>& pooled, AuxPredictions& aux) {
        const std::vector<float> out = mlp(pooled, "aux_head");
        if (out.size() >= 4) {
            aux.expected_net_attack_1s = out[0];
            aux.expected_received_garbage = out[1];
            aux.expected_time_to_ko = out[2];
            aux.topout_within_8_pieces = out[3];
        }
    }

    // Two-layer MLP with SiLU, matching nn.Sequential(Linear, SiLU, Linear).
    std::vector<float> mlp(const std::vector<float>& in, const std::string& prefix) {
        const std::string w0 = prefix + ".0.weight", b0 = prefix + ".0.bias";
        const std::string w2 = prefix + ".2.weight", b2 = prefix + ".2.bias";
        if (!w_.has(w0) || !w_.has(w2)) return {};
        const Tensor& W0 = w_.get(w0);
        std::vector<float> hidden(W0.rows());
        detail::linear(in.data(), 1, in.size(), W0, w_.has(b0) ? &w_.get(b0) : nullptr,
                       hidden.data());
        for (float& v : hidden) v = detail::silu(v);
        const Tensor& W2 = w_.get(w2);
        std::vector<float> out(W2.rows());
        detail::linear(hidden.data(), 1, hidden.size(), W2, w_.has(b2) ? &w_.get(b2) : nullptr,
                       out.data());
        return out;
    }

    NnueWeights w_;
    std::string name_;
    Tokenizer tok_;
};

}  // namespace tetra