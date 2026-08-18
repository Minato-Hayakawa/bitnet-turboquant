#pragma once
// turboquant/rng.hpp
//
// Lightweight, dependency-free RNG helpers used across the library.
// We use a fixed, seedable PRNG (splitmix64 -> xorshift128+) so that
// quantizer/dequantizer pairs can regenerate the *same* random rotation
// and projection matrices from a small integer seed instead of storing
// them, which keeps the compressed representation tiny (this mirrors
// the "data-oblivious" design of TurboQuant: the randomness is a
// pseudo-random function of the seed, not learned/stored data).

#include <cstdint>
#include <cmath>

namespace turboquant {

// splitmix64: used only to seed xorshift128+ well from a single uint64.
inline uint64_t splitmix64_next(uint64_t &state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

class Rng {
public:
    explicit Rng(uint64_t seed = 0) { reseed(seed); }

    void reseed(uint64_t seed) {
        uint64_t sm = seed;
        s0_ = splitmix64_next(sm);
        s1_ = splitmix64_next(sm);
        if (s0_ == 0 && s1_ == 0) s1_ = 1; // avoid degenerate all-zero state
    }

    // xorshift128+
    uint64_t next_u64() {
        uint64_t x = s0_;
        const uint64_t y = s1_;
        s0_ = y;
        x ^= x << 23;
        x ^= x >> 17;
        x ^= y ^ (y >> 26);
        s1_ = x;
        return x + y;
    }

    // Uniform double in [0, 1)
    double next_uniform() {
        // 53 bits of mantissa precision
        return (next_u64() >> 11) * (1.0 / 9007199254740992.0);
    }

    // +1 or -1 with equal probability (used for the random sign / Rademacher
    // diagonal in the Randomized Hadamard Transform, and for the random
    // Gaussian... actually sign is used directly here; Gaussian is separate).
    int8_t next_sign() {
        return (next_u64() & 1u) ? int8_t(1) : int8_t(-1);
    }

    // Standard normal sample via Box-Muller (sufficient for our purposes;
    // we don't need cryptographic quality, just good statistical spread).
    double next_gaussian() {
        // Box-Muller, cached second value for efficiency.
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        double u1, u2, s;
        do {
            u1 = 2.0 * next_uniform() - 1.0;
            u2 = 2.0 * next_uniform() - 1.0;
            s = u1 * u1 + u2 * u2;
        } while (s >= 1.0 || s == 0.0);
        double mul = std::sqrt(-2.0 * std::log(s) / s);
        spare_ = u2 * mul;
        has_spare_ = true;
        return u1 * mul;
    }

private:
    uint64_t s0_, s1_;
    bool has_spare_ = false;
    double spare_ = 0.0;
};

} // namespace turboquant
