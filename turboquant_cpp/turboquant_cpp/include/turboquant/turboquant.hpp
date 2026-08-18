#pragma once
// turboquant/turboquant.hpp
//
// C++ port of the TurboQuant two-stage online vector quantizer
// (Zandieh, Daliri, Hadian, Mirrokni -- ICLR 2026, arXiv:2504.19874),
// following the structure of the PyPI `turboquant-torch` reference
// implementation, but with no PyTorch/Python dependency so it can be
// linked directly into a C++ inference engine such as bitnet.cpp.
//
// Pipeline (matches the paper's two-stage design):
//   Stage 1 (MSE-optimal, b-1 bits/coord):
//     x -> normalize to unit vector -> Randomized Hadamard Transform
//       -> Lloyd-Max scalar quantizer -> codes + ||x||
//   Stage 2 (QJL, 1 bit/coord, optional "unbiased" refinement):
//     residual r = x - dequant_stage1(codes, norm)
//       -> seeded structured (Hadamard) JL sketch -> sign bits + ||r||
//
// HONEST CAVEAT (please read):
//   The original QJL construction is designed primarily to give an
//   *unbiased estimator of inner products* (e.g. query . key in attention)
//   directly from the sign bits, without ever materializing a residual
//   vector. Here we instead use the sign bits to produce a *point estimate
//   of the residual vector itself* (via the classical 1-bit compressed
//   structured one-bit residual reconstruction, which is simpler to reason
//   about and test, and is
//   useful when you want an actual dequantized tensor (e.g. to feed back
//   into standard attention kernels) rather than only inner-product
//   estimates. This is a deliberate, documented engineering simplification,
//   not a claim of exact paper-equivalence -- validate against your own
//   accuracy targets before relying on it.

#include <vector>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <numeric>

#include "rng.hpp"
#include "hadamard.hpp"
#include "codebook.hpp"
#include "bitpack.hpp"

namespace turboquant {

// M_PI is not guaranteed by the standard (e.g. MSVC without
// _USE_MATH_DEFINES), so define our own to keep this header portable.
constexpr double kPi = 3.14159265358979323846;

struct QuantizedVector {
    std::vector<uint8_t> codes_packed;   // Stage-1 Lloyd-Max codes, (bit_width-1) bits each
    std::vector<uint8_t> signs_packed;   // Stage-2 QJL sign bits, 1 bit each (empty if disabled)
    float norm = 0.0f;                   // ||x||_2
    float residual_norm = 0.0f;          // ||residual||_2 (only meaningful if Stage 2 enabled)
    size_t orig_dim = 0;

    // Total bits used to store this vector (codes + signs + the two fp32 norms).
    size_t bit_size() const {
        return codes_packed.size() * 8 + signs_packed.size() * 8 +
               32 /* norm */ + (signs_packed.empty() ? 0 : 32 /* residual_norm */);
    }
};

class TurboQuant {
public:
    // dim:        vector dimensionality (need not be a power of two)
    // bit_width:  total bits/coordinate budget, b. Stage 1 uses (b-1) bits,
    //             Stage 2 (if enabled) uses 1 bit. Must be >= 2 if
    //             unbiased=true, >= 1 otherwise.
    // unbiased:   whether to run Stage 2 (QJL residual refinement). Matches
    //             the `unbiased=True` flag in the Python reference API.
    // seed:       fixes the random Hadamard rotation + QJL projection so
    //             quantize()/dequantize() (and repeated runs) agree. Two
    //             TurboQuant instances with the same (dim, seed) always
    //             generate the same random operators.
    TurboQuant(size_t dim, int bit_width, bool unbiased = true, uint64_t seed = 0x54554251ULL)
        : dim_(dim), bit_width_(bit_width), unbiased_(unbiased), seed_(seed) {
        if (dim_ == 0) throw std::invalid_argument("TurboQuant: dim must be > 0");
        if (unbiased_ && bit_width_ < 2)
            throw std::invalid_argument("TurboQuant: bit_width must be >= 2 when unbiased=true");
        if (!unbiased_ && bit_width_ < 1)
            throw std::invalid_argument("TurboQuant: bit_width must be >= 1");

        padded_dim_ = next_power_of_two(dim_);
        stage1_levels_ = 1u << static_cast<unsigned>(unbiased_ ? bit_width_ - 1 : bit_width_);
        hadamard_signs_ = random_sign_vector(padded_dim_, seed_);

        // A seeded randomized Hadamard transform is used for the residual
        // sketch.  Unlike the old dense Gaussian matrix this has O(d log d)
        // work and O(d) state, which is essential for KV-cache use.
        if (unbiased_) qjl_signs_ = random_sign_vector(padded_dim_, seed_ ^ 0x514A4C5F53454544ULL);
    }

    QuantizedVector quantize(const float *x, size_t n) const {
        if (n != dim_) throw std::invalid_argument("TurboQuant::quantize: dimension mismatch");

        QuantizedVector out;
        out.orig_dim = dim_;

        // --- norm + unit vector ---
        double sumsq = 0.0;
        for (size_t i = 0; i < dim_; ++i) sumsq += static_cast<double>(x[i]) * x[i];
        float norm = static_cast<float>(std::sqrt(sumsq));
        out.norm = norm;

        std::vector<float> u(dim_);
        if (norm > 0.0f) {
            for (size_t i = 0; i < dim_; ++i) u[i] = x[i] / norm;
        } // else u stays all-zero; codes will all point at the near-zero codeword

        // --- Stage 1: Randomized Hadamard Transform + Lloyd-Max ---
        std::vector<float> y(padded_dim_);
        randomized_hadamard_forward(u.data(), dim_, padded_dim_, hadamard_signs_, y.data());

        const Codebook &cb = Codebook::get(static_cast<int>(stage1_levels_));
        const float sigma = 1.0f / std::sqrt(static_cast<float>(padded_dim_));
        std::vector<uint32_t> codes(padded_dim_);
        for (size_t i = 0; i < padded_dim_; ++i) {
            codes[i] = static_cast<uint32_t>(cb.encode(y[i] / sigma));
        }
        out.codes_packed = pack_bits(codes, bits_per_stage1_code());

        if (!unbiased_) return out;

        // --- reconstruct Stage-1 estimate to compute the residual ---
        std::vector<float> y_hat(padded_dim_);
        for (size_t i = 0; i < padded_dim_; ++i) y_hat[i] = cb.decode(static_cast<int>(codes[i])) * sigma;

        std::vector<float> u_hat(dim_);
        randomized_hadamard_inverse(y_hat.data(), dim_, padded_dim_, hadamard_signs_, u_hat.data());

        std::vector<float> residual(dim_);
        for (size_t i = 0; i < dim_; ++i) residual[i] = x[i] - u_hat[i] * norm;

        double res_sumsq = 0.0;
        for (size_t i = 0; i < dim_; ++i) res_sumsq += static_cast<double>(residual[i]) * residual[i];
        out.residual_norm = static_cast<float>(std::sqrt(res_sumsq));

        // --- Stage 2: structured one-bit JL residual sketch ---
        // The randomized Hadamard transform is an orthogonal, sub-Gaussian JL
        // transform. Its signs are fixed from seed_ and therefore need not be
        // stored per token. This replaces the former dense Gaussian QJL matrix
        // with an O(d log d) transform suitable for a KV-cache hot path.
        std::vector<float> sketch(padded_dim_);
        randomized_hadamard_forward(residual.data(), dim_, padded_dim_, qjl_signs_, sketch.data());
        std::vector<int8_t> signs(padded_dim_);
        for (size_t i = 0; i < padded_dim_; ++i) {
            signs[i] = (sketch[i] >= 0.0f) ? int8_t(1) : int8_t(-1);
        }
        out.signs_packed = pack_signs(signs);

        return out;
    }

    std::vector<float> dequantize(const QuantizedVector &q) const {
        if (q.orig_dim != dim_) throw std::invalid_argument("TurboQuant::dequantize: dimension mismatch");

        const Codebook &cb = Codebook::get(static_cast<int>(stage1_levels_));
        auto codes = unpack_bits(q.codes_packed, padded_dim_, bits_per_stage1_code());

        const float sigma = 1.0f / std::sqrt(static_cast<float>(padded_dim_));
        std::vector<float> y_hat(padded_dim_);
        for (size_t i = 0; i < padded_dim_; ++i) y_hat[i] = cb.decode(static_cast<int>(codes[i])) * sigma;

        std::vector<float> u_hat(dim_);
        randomized_hadamard_inverse(y_hat.data(), dim_, padded_dim_, hadamard_signs_, u_hat.data());

        std::vector<float> x_hat(dim_);
        for (size_t i = 0; i < dim_; ++i) x_hat[i] = u_hat[i] * q.norm;

        if (unbiased_ && !q.signs_packed.empty()) {
            auto signs = unpack_signs(q.signs_packed, padded_dim_);
            // For an orthonormal random transform, each coefficient has
            // E|y_i| ~= ||r|| sqrt(2/(pi*d)).  Reuse that magnitude with the
            // stored signs and invert the same seeded transform.
            const float scale = q.residual_norm * static_cast<float>(std::sqrt(2.0 / (kPi * padded_dim_)));
            std::vector<float> signed_sketch(padded_dim_);
            for (size_t i = 0; i < padded_dim_; ++i) signed_sketch[i] = signs[i] * scale;
            std::vector<float> r_hat(dim_);
            randomized_hadamard_inverse(signed_sketch.data(), dim_, padded_dim_, qjl_signs_, r_hat.data());
            for (size_t j = 0; j < dim_; ++j) x_hat[j] += r_hat[j];
        }

        return x_hat;
    }

    // Compression ratio vs. storing the vector as fp32 (32 bits/coord).
    double compression_ratio() const {
        double orig_bits = static_cast<double>(dim_) * 32.0;
        double comp_bits = static_cast<double>(padded_dim_) * bits_per_stage1_code() + 32.0;
        if (unbiased_) comp_bits += static_cast<double>(padded_dim_) * 1.0 + 32.0;
        return orig_bits / comp_bits;
    }

    size_t dim() const { return dim_; }
    size_t padded_dim() const { return padded_dim_; }
    int bit_width() const { return bit_width_; }
    bool unbiased() const { return unbiased_; }

private:
    int bits_per_stage1_code() const {
        return unbiased_ ? bit_width_ - 1 : bit_width_;
    }

    size_t dim_;
    size_t padded_dim_;
    int bit_width_;
    bool unbiased_;
    uint64_t seed_;
    unsigned stage1_levels_;
    std::vector<int8_t> hadamard_signs_;
    std::vector<int8_t> qjl_signs_; // fixed signs for the structured residual sketch
};

} // namespace turboquant
