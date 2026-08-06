// SPDX-License-Identifier: MIT
// TetraFormer / TetraZero -- deterministic RNG and piece randomizer.
//
// Spec 18.1 requires bit-exact reproducibility from a seed, so the generator is
// a fixed-algorithm splitmix64/xoshiro pair rather than anything from <random>
// (whose engines are portable but whose distributions are not).
#pragma once

#include "tetra/ruleset.hpp"
#include "tetra/types.hpp"

#include <cstdint>
#include <deque>
#include <vector>

namespace tetra {

class Rng {
public:
    explicit Rng(std::uint64_t seed = 0) { reseed(seed); }

    void reseed(std::uint64_t seed) {
        // splitmix64 expansion of the seed into the xoshiro256** state.
        std::uint64_t z = seed + 0x9E3779B97F4A7C15ull;
        for (int i = 0; i < 4; ++i) {
            std::uint64_t x = (z += 0x9E3779B97F4A7C15ull);
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
            s_[i] = x ^ (x >> 31);
        }
        if (!s_[0] && !s_[1] && !s_[2] && !s_[3]) s_[0] = 0x9E3779B97F4A7C15ull;
    }

    std::uint64_t next_u64() {
        const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Unbiased integer in [0, n).
    std::uint32_t below(std::uint32_t n) {
        if (n <= 1) return 0;
        const std::uint32_t limit = 0xFFFFFFFFu - (0xFFFFFFFFu % n);
        std::uint32_t v;
        do {
            v = static_cast<std::uint32_t>(next_u64() >> 32);
        } while (v >= limit);
        return v % n;
    }

    // True with probability num/den.
    bool chance(int num, int den) {
        if (num <= 0) return false;
        if (num >= den) return true;
        return static_cast<int>(below(static_cast<std::uint32_t>(den))) < num;
    }

    // Exact serialisable state, so a search can snapshot and restore.
    std::array<std::uint64_t, 4> state() const { return s_; }
    void set_state(const std::array<std::uint64_t, 4>& s) { s_ = s; }

private:
    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    std::array<std::uint64_t, 4> s_{};
};

// ---------------------------------------------------------------------------
// Piece randomizer / queue
// ---------------------------------------------------------------------------
// The simulator owns the full queue; the observation layer (spec 7.1) only ever
// exposes the first `preview_count` entries, so a bot cannot see hidden pieces.
class PieceQueue {
public:
    PieceQueue() = default;
    PieceQueue(const RandomizerCfg& cfg, std::uint64_t seed) { reset(cfg, seed); }

    void reset(const RandomizerCfg& cfg, std::uint64_t seed) {
        cfg_ = cfg;
        rng_.reseed(seed);
        queue_.clear();
        bag_.clear();
        pieces_generated_ = 0;
        refill(cfg.preview_count + 2);
    }

    Piece pop() {
        refill(cfg_.preview_count + 2);
        Piece p = queue_.front();
        queue_.pop_front();
        return p;
    }

    Piece peek(int i) const {
        return (i >= 0 && i < static_cast<int>(queue_.size())) ? queue_[static_cast<size_t>(i)]
                                                               : Piece::None;
    }

    // Pieces the player is allowed to see.
    std::vector<Piece> visible_next() const {
        std::vector<Piece> v;
        for (int i = 0; i < cfg_.preview_count; ++i) {
            Piece p = peek(i);
            if (p == Piece::None) break;
            v.push_back(p);
        }
        return v;
    }

    // Remaining contents of the current bag: legitimate public knowledge in a
    // 7-bag game, and the basis for `bag_belief` in the observation.
    std::vector<Piece> bag_remaining() const { return bag_; }

    // Full buffered queue, including the preview and the hidden lookahead.
    // Search transposition keys use this only to distinguish simulator states;
    // observations must continue to expose `visible_next()` only.
    std::vector<Piece> buffered_pieces() const {
        return std::vector<Piece>(queue_.begin(), queue_.end());
    }

    std::uint64_t pieces_generated() const { return pieces_generated_; }
    bool mirrored() const { return mirror_; }
    Rng& rng() { return rng_; }
    const Rng& rng() const { return rng_; }

    // Resample everything the player cannot legitimately see (spec 11.3).
    //
    // A search that copies the true queue would be reading the future: beyond
    // `preview_count` the sequence is hidden information, and using it inflates
    // the value of setups that only work because the engine knows what is
    // coming. This discards the unseen tail and regenerates it from a fresh
    // seed, keeping:
    //
    //   * the visible preview exactly as it is, and
    //   * the bag state, so 7-bag counting constraints still hold -- a resample
    //     that ignored the bag would be *less* informed than a human player.
    //
    // The result is one sample from the set of futures consistent with the
    // observation, which is what a chance node needs.
    void resample_hidden(std::uint64_t seed) {
        const int keep = std::max(0, cfg_.preview_count);
        // Discard the pieces the player cannot see, returning each one to the
        // bag it came from. Without this the bag is left short by however many
        // lookahead pieces the queue had buffered, and regeneration draws from
        // a depleted bag -- observably breaking the 7-bag guarantee (measured:
        // three O and one L in a 14-piece window).
        while (static_cast<int>(queue_.size()) > keep) {
            bag_.push_back(queue_.back());
            queue_.pop_back();
        }
        rng_.reseed(seed);
        // Reshuffle what is left of the bag so the returned pieces are not
        // simply drawn back in the order they were removed.
        for (int i = static_cast<int>(bag_.size()) - 1; i > 0; --i) {
            const int j = static_cast<int>(rng_.below(static_cast<std::uint32_t>(i + 1)));
            std::swap(bag_[static_cast<size_t>(i)], bag_[static_cast<size_t>(j)]);
        }
        refill(cfg_.preview_count + 2);
    }

    void set_mirror(bool m) { mirror_ = m; }

    // True when the queue holds pieces beyond what the player may see, which is
    // exactly the state a search must not exploit.
    bool has_hidden_lookahead() const {
        return static_cast<int>(queue_.size()) > std::max(0, cfg_.preview_count);
    }

private:
    void refill(int want) {
        while (static_cast<int>(queue_.size()) < want) generate_one();
    }

    void generate_one() {
        switch (cfg_.type) {
            case RandomizerType::Uniform: {
                Piece p = static_cast<Piece>(rng_.below(PIECE_COUNT));
                if (mirror_) p = mirror_piece(p);
                queue_.push_back(p);
                break;
            }
            case RandomizerType::OnePiece:
                queue_.push_back(Piece::I);
                break;
            case RandomizerType::Bag14:
                if (bag_.empty()) fill_bag(2);
                take_from_bag();
                break;
            case RandomizerType::Bag7:
            default:
                if (bag_.empty()) fill_bag(1);
                take_from_bag();
                break;
        }
        ++pieces_generated_;
    }

    void fill_bag(int copies) {
        bag_.clear();
        for (int c = 0; c < copies; ++c)
            for (int i = 0; i < PIECE_COUNT; ++i) bag_.push_back(static_cast<Piece>(i));
        // Fisher-Yates with our own RNG so the order is reproducible.
        for (int i = static_cast<int>(bag_.size()) - 1; i > 0; --i) {
            const int j = static_cast<int>(rng_.below(static_cast<std::uint32_t>(i + 1)));
            std::swap(bag_[static_cast<size_t>(i)], bag_[static_cast<size_t>(j)]);
        }
    }

    void take_from_bag() {
        Piece p = bag_.back();
        bag_.pop_back();
        if (mirror_) p = mirror_piece(p);
        queue_.push_back(p);
    }

    RandomizerCfg cfg_{};
    Rng rng_{0};
    std::deque<Piece> queue_;
    std::vector<Piece> bag_;
    std::uint64_t pieces_generated_ = 0;
    bool mirror_ = false;
};

}  // namespace tetra
