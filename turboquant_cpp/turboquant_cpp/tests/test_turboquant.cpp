// Basic correctness + distortion sanity checks for the C++ TurboQuant port.
// Not a rigorous unit test suite -- just enough to verify the pipeline is
// wired correctly and produces sane, paper-ballpark numbers before you rely
// on it for anything real.

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

#include "turboquant/turboquant.hpp"
#include "turboquant/kv_cache.hpp"

using namespace turboquant;

static double mse(const std::vector<float> &a, const std::vector<float> &b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return s / a.size();
}

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("  [FAIL] %s\n", msg); failures++; } \
    else { std::printf("  [ OK ] %s\n", msg); } \
} while (0)

int main() {
    std::mt19937 gen(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::printf("== Test 1: single-vector roundtrip shape/range ==\n");
    {
        const size_t dim = 128;
        TurboQuant tq(dim, 3, /*unbiased=*/true);
        std::vector<float> x(dim);
        for (auto &v : x) v = nd(gen);

        auto q = tq.quantize(x.data(), dim);
        auto x_hat = tq.dequantize(q);

        CHECK(x_hat.size() == dim, "dequantized size matches input dim");
        double m = mse(x, x_hat);
        std::printf("        MSE (unit-scale random vector) = %.6f\n", m);
        CHECK(m < 1.0, "MSE is reasonably small (< 1.0 for O(1)-scale input)");
        std::printf("        compression ratio = %.2fx\n", tq.compression_ratio());
        CHECK(tq.compression_ratio() > 5.0, "compression ratio > 5x at 3 bits/coord");
    }

    std::printf("\n== Test 2: distortion decreases as bit_width increases ==\n");
    {
        const size_t dim = 128;
        const int n_trials = 20;
        double prev_mse = 1e9;
        bool monotonic = true;
        for (int bits : {2, 3, 4, 5}) {
            TurboQuant tq(dim, bits, true);
            double total_mse = 0.0;
            for (int t = 0; t < n_trials; ++t) {
                std::vector<float> x(dim);
                for (auto &v : x) v = nd(gen);
                auto q = tq.quantize(x.data(), dim);
                auto x_hat = tq.dequantize(q);
                total_mse += mse(x, x_hat);
            }
            double avg = total_mse / n_trials;
            std::printf("        bits=%d  avg_MSE=%.6f  ratio=%.2fx\n", bits, avg, tq.compression_ratio());
            if (avg > prev_mse) monotonic = false;
            prev_mse = avg;
        }
        CHECK(monotonic, "MSE is non-increasing as bit_width grows (2->3->4->5)");
    }

    std::printf("\n== Test 3: zero-vector edge case ==\n");
    {
        const size_t dim = 64;
        TurboQuant tq(dim, 3, true);
        std::vector<float> x(dim, 0.0f);
        auto q = tq.quantize(x.data(), dim);
        auto x_hat = tq.dequantize(q);
        double m = mse(x, x_hat);
        std::printf("        MSE on all-zero input = %.8f\n", m);
        CHECK(m < 1e-4, "near-zero MSE on the degenerate zero vector");
    }

    std::printf("\n== Test 4: non-power-of-two dim ==\n");
    {
        const size_t dim = 100; // not a power of two -- exercises padding path
        TurboQuant tq(dim, 3, true);
        std::vector<float> x(dim);
        for (auto &v : x) v = nd(gen);
        auto q = tq.quantize(x.data(), dim);
        auto x_hat = tq.dequantize(q);
        CHECK(x_hat.size() == dim, "output size matches non-power-of-two input dim");
        std::printf("        padded_dim = %zu (from dim = %zu)\n", tq.padded_dim(), tq.dim());
    }

    std::printf("\n== Test 5: KV cache batch compress/decompress ==\n");
    {
        const size_t head_dim = 128;
        const size_t seq_len = 64;
        TurboQuantKVCache cache(head_dim, 3, true);

        std::vector<float> keys(seq_len * head_dim), values(seq_len * head_dim);
        for (auto &v : keys) v = nd(gen);
        for (auto &v : values) v = nd(gen);

        auto compressed = cache.compress(keys.data(), values.data(), seq_len);

        std::vector<float> keys_hat(seq_len * head_dim), values_hat(seq_len * head_dim);
        cache.decompress_keys(compressed, keys_hat.data());
        cache.decompress_values(compressed, values_hat.data());

        double key_mse = mse(keys, keys_hat);
        double val_mse = mse(values, values_hat);
        std::printf("        key MSE=%.6f  value MSE=%.6f\n", key_mse, val_mse);
        CHECK(key_mse < 1.0 && val_mse < 1.0, "KV cache roundtrip MSE is bounded");

        auto sav = cache.memory_savings(seq_len);
        std::printf("        memory: %.0f bytes -> %.0f bytes (%.2fx)\n",
                     sav.original_bytes, sav.compressed_bytes, sav.ratio);
        CHECK(sav.ratio > 5.0, "KV cache memory savings > 5x at 3 bits/coord");
    }

    std::printf("\n== Test 6: unbiased=false (Stage 1 only) still works ==\n");
    {
        const size_t dim = 64;
        TurboQuant tq(dim, 2, /*unbiased=*/false);
        std::vector<float> x(dim);
        for (auto &v : x) v = nd(gen);
        auto q = tq.quantize(x.data(), dim);
        CHECK(q.signs_packed.empty(), "no sign bits stored when unbiased=false");
        auto x_hat = tq.dequantize(q);
        double m = mse(x, x_hat);
        std::printf("        Stage-1-only MSE (2 bits) = %.6f, ratio=%.2fx\n", m, tq.compression_ratio());
        CHECK(tq.compression_ratio() > 10.0, "Stage-1-only compression ratio > 10x at 2 bits/coord");
    }

    std::printf("\n%s: %d failure(s)\n", failures == 0 ? "SUMMARY: ALL PASSED" : "SUMMARY: FAILURES FOUND", failures);
    return failures == 0 ? 0 : 1;
}
