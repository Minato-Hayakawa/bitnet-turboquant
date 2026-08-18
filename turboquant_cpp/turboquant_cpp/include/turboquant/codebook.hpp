#pragma once
// turboquant/codebook.hpp
//
// Lloyd-Max scalar quantizer for a standard-normal source.
//
// After the Randomized Hadamard Transform, each coordinate of a unit-norm
// vector behaves approximately like an i.i.d. N(0, 1/d) sample (d = padded
// dimension). We therefore only need ONE codebook per bit-width, built once
// for a *standard* normal N(0,1) and rescaled at quantize time by the
// per-vector empirical std (equivalently, by 1/sqrt(d) since the RHT output
// of a unit vector has variance ~1/d) -- this is the "MSE-optimal scalar
// quantizer" referenced in the TurboQuant paper, built via Lloyd's
// algorithm (1D k-means), which converges to the Lloyd-Max optimum for a
// given source density.
//
// Codebooks are cached the first time a given (bits) value is requested, so
// repeated TurboQuant(...) construction is cheap after the first call.

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include "rng.hpp"

namespace turboquant {

class Codebook {
public:
    // levels: number of quantization levels (2^bits)
    // iters: Lloyd-Max refinement iterations
    // n_samples: Monte-Carlo samples of N(0,1) used to fit the codebook
    static const Codebook &get(int levels, int iters = 60, size_t n_samples = 200000) {
        static std::mutex mtx;
        static std::unordered_map<int, Codebook> cache;
        std::lock_guard<std::mutex> lock(mtx);
        auto it = cache.find(levels);
        if (it != cache.end()) return it->second;
        Codebook cb;
        cb.build(levels, iters, n_samples);
        auto res = cache.emplace(levels, std::move(cb));
        return res.first->second;
    }

    // Quantize a standard-normal-scaled value x (i.e. x / sigma) to the
    // nearest codeword index using binary search over the (sorted)
    // boundaries. Returns index in [0, levels).
    inline int encode(float x_normalized) const {
        // boundaries_.size() == levels_ - 1
        auto pos = std::upper_bound(boundaries_.begin(), boundaries_.end(), x_normalized);
        return static_cast<int>(pos - boundaries_.begin());
    }

    inline float decode(int index) const {
        return codewords_[index];
    }

    int levels() const { return levels_; }
    int bits() const { return bits_; }

private:
    void build(int levels, int iters, size_t n_samples) {
        levels_ = levels;
        bits_ = 0;
        for (int l = levels; l > 1; l >>= 1) bits_++;

        // Draw fixed Monte-Carlo samples from N(0,1) with a dedicated,
        // deterministic seed so the codebook is 100% reproducible across
        // runs/platforms (no dependence on wall-clock or hardware RNG).
        Rng rng(0xC0DEBEEFULL ^ static_cast<uint64_t>(levels));
        std::vector<float> samples(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            samples[i] = static_cast<float>(rng.next_gaussian());
        }
        std::sort(samples.begin(), samples.end());

        // Initialize codewords at the quantiles of N(0,1) (better than
        // uniform spacing -- speeds up Lloyd-Max convergence a lot).
        codewords_.resize(levels);
        for (int i = 0; i < levels; ++i) {
            double q = (i + 0.5) / levels;
            codewords_[i] = static_cast<float>(inv_norm_cdf(q));
        }
        std::sort(codewords_.begin(), codewords_.end());

        boundaries_.resize(levels - 1);

        for (int it = 0; it < iters; ++it) {
            // Boundaries = midpoints between adjacent codewords (Lloyd's
            // nearest-neighbor / "Voronoi" condition).
            for (int i = 0; i < levels - 1; ++i) {
                boundaries_[i] = 0.5f * (codewords_[i] + codewords_[i + 1]);
            }
            // Recompute each codeword as the centroid (mean) of samples
            // assigned to it (Lloyd's "centroid" condition == MSE-optimal
            // reconstruction point for its cell).
            std::vector<double> sum(levels, 0.0);
            std::vector<size_t> count(levels, 0);
            size_t lo = 0;
            for (int cell = 0; cell < levels; ++cell) {
                float upper = (cell < levels - 1) ? boundaries_[cell]
                                                   : std::numeric_limits<float>::infinity();
                size_t hi = lo;
                while (hi < samples.size() && samples[hi] < upper) hi++;
                for (size_t k = lo; k < hi; ++k) sum[cell] += samples[k];
                count[cell] = hi - lo;
                lo = hi;
            }
            for (int cell = 0; cell < levels; ++cell) {
                if (count[cell] > 0) {
                    codewords_[cell] = static_cast<float>(sum[cell] / static_cast<double>(count[cell]));
                }
                // else: keep previous codeword (empty cell, rare with
                // quantile init + enough samples)
            }
        }
        // Final boundaries consistent with converged codewords.
        for (int i = 0; i < levels - 1; ++i) {
            boundaries_[i] = 0.5f * (codewords_[i] + codewords_[i + 1]);
        }
    }

    // Acklam's algorithm approximation of the inverse standard normal CDF.
    // Accurate to ~1e-9, more than sufficient for codebook initialization.
    static double inv_norm_cdf(double p) {
        static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                                    -2.759285104469687e+02, 1.383577518672690e+02,
                                    -3.066479806614716e+01, 2.506628277459239e+00};
        static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                                    -1.556989798598866e+02, 6.680131188771972e+01,
                                    -1.328068155288572e+01};
        static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                    -2.400758277161838e+00, -2.549732539343734e+00,
                                    4.374664141464968e+00, 2.938163982698783e+00};
        static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
                                    2.445134137142996e+00, 3.754408661907416e+00};
        const double p_low = 0.02425, p_high = 1 - p_low;
        double q, r;
        if (p < p_low) {
            q = std::sqrt(-2 * std::log(p));
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
        } else if (p <= p_high) {
            q = p - 0.5; r = q*q;
            return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
        } else {
            q = std::sqrt(-2 * std::log(1 - p));
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
        }
    }

    int levels_ = 0;
    int bits_ = 0;
    std::vector<float> codewords_;
    std::vector<float> boundaries_;
};

} // namespace turboquant
