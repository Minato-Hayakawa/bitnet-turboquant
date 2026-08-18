#pragma once
// turboquant/hadamard.hpp
//
// Randomized Hadamard Transform (RHT) = random sign flip (Rademacher
// diagonal) followed by a Fast Walsh-Hadamard Transform (FWHT), scaled by
// 1/sqrt(d). This is Stage 1's preprocessing step in TurboQuant: it takes
// an arbitrary unit vector and (in expectation) spreads its energy evenly
// across coordinates, which is what makes a *scalar* (per-coordinate)
// quantizer near-optimal afterwards -- the whole point of the "data-
// oblivious" design (no calibration data needed, works for any input
// distribution because the random rotation makes coordinates behave like
// i.i.d. Gaussians regardless of the original distribution).
//
// Requires dim to be a power of two. TurboQuant pads to the next power of
// two internally (see turboquant.hpp) so callers never have to worry about
// this.

#include <cstdint>
#include <vector>
#include <cstring>
#include "rng.hpp"

namespace turboquant {

inline bool is_power_of_two(size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

inline size_t next_power_of_two(size_t n) {
    if (n == 0) return 1;
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// In-place Fast Walsh-Hadamard Transform on `data[0..n)`, n must be a power
// of two. This is the *unnormalized* transform (entries of the implicit
// Hadamard matrix are +-1, not +-1/sqrt(n)); callers normalize themselves.
inline void fwht_inplace(float *data, size_t n) {
    for (size_t len = 1; len < n; len <<= 1) {
        for (size_t i = 0; i < n; i += (len << 1)) {
            for (size_t j = i; j < i + len; ++j) {
                float a = data[j];
                float b = data[j + len];
                data[j] = a + b;
                data[j + len] = a - b;
            }
        }
    }
}

// Generates a length-`dim` vector of +-1 signs from `seed`. Both the
// quantizer and dequantizer call this with the same seed to regenerate the
// identical diagonal without ever storing it.
inline std::vector<int8_t> random_sign_vector(size_t dim, uint64_t seed) {
    Rng rng(seed);
    std::vector<int8_t> signs(dim);
    for (size_t i = 0; i < dim; ++i) signs[i] = rng.next_sign();
    return signs;
}

// Forward RHT: out = H * diag(signs) * in / sqrt(n)   (n = padded dim)
// `in` has length `orig_dim` (<= n); zero-padded internally.
inline void randomized_hadamard_forward(const float *in, size_t orig_dim,
                                         size_t padded_dim,
                                         const std::vector<int8_t> &signs,
                                         float *out /* length padded_dim */) {
    for (size_t i = 0; i < padded_dim; ++i) {
        float v = (i < orig_dim) ? in[i] : 0.0f;
        out[i] = v * static_cast<float>(signs[i]);
    }
    fwht_inplace(out, padded_dim);
    const float norm = 1.0f / std::sqrt(static_cast<float>(padded_dim));
    for (size_t i = 0; i < padded_dim; ++i) out[i] *= norm;
}

// Inverse RHT. The Hadamard transform is (up to the 1/sqrt(n) scaling)
// self-inverse (H*H = n*I), and diag(signs) is self-inverse since signs are
// +-1, so the inverse is: apply FWHT again, normalize, then undo the sign
// flip. Only the first `orig_dim` outputs are written (the rest is padding).
inline void randomized_hadamard_inverse(const float *in /* length padded_dim */,
                                         size_t orig_dim, size_t padded_dim,
                                         const std::vector<int8_t> &signs,
                                         float *out /* length orig_dim */) {
    std::vector<float> tmp(in, in + padded_dim);
    fwht_inplace(tmp.data(), padded_dim);
    const float norm = 1.0f / std::sqrt(static_cast<float>(padded_dim));
    for (size_t i = 0; i < orig_dim; ++i) {
        out[i] = tmp[i] * norm * static_cast<float>(signs[i]);
    }
}

} // namespace turboquant
