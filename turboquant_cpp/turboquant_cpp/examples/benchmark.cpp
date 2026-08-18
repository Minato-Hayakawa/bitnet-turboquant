// Benchmarks compression ratio, distortion, and rough throughput across a
// few (head_dim, bit_width) configurations typical of real transformer
// attention heads -- mirrors the "Benchmarks on Real Models" table in the
// Python reference's README, but generated locally against synthetic
// Gaussian data (no HuggingFace/model dependency here).

#include <cstdio>
#include <vector>
#include <random>
#include <chrono>

#include "turboquant/kv_cache.hpp"

using namespace turboquant;
using Clock = std::chrono::high_resolution_clock;

int main() {
    std::mt19937 gen(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    struct Config { size_t head_dim; int bits; const char *label; };
    std::vector<Config> configs = {
        {64,  2, "head_dim=64  (e.g. small models), 2-bit"},
        {64,  3, "head_dim=64  (e.g. small models), 3-bit"},
        {128, 3, "head_dim=128 (e.g. Llama/Qwen),   3-bit"},
        {128, 4, "head_dim=128 (e.g. Llama/Qwen),   4-bit"},
        {256, 3, "head_dim=256 (e.g. larger models), 3-bit"},
    };

    const size_t seq_len = 2048;

    std::printf("%-45s %10s %10s %10s %12s\n", "config", "MSE", "ratio", "orig(MB)", "comp(MB)");
    std::printf("%s\n", std::string(95, '-').c_str());

    for (auto &cfg : configs) {
        TurboQuantKVCache cache(cfg.head_dim, cfg.bits, true);

        std::vector<float> keys(seq_len * cfg.head_dim), values(seq_len * cfg.head_dim);
        for (auto &v : keys) v = nd(gen);
        for (auto &v : values) v = nd(gen);

        auto t0 = Clock::now();
        auto compressed = cache.compress(keys.data(), values.data(), seq_len);
        auto t1 = Clock::now();

        std::vector<float> keys_hat(seq_len * cfg.head_dim);
        cache.decompress_keys(compressed, keys_hat.data());
        auto t2 = Clock::now();

        double mse = 0.0;
        for (size_t i = 0; i < keys.size(); ++i) {
            double d = keys[i] - keys_hat[i];
            mse += d * d;
        }
        mse /= keys.size();

        auto sav = cache.memory_savings(seq_len);
        double compress_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double decompress_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        std::printf("%-45s %10.5f %9.2fx %11.3f %11.3f\n", cfg.label, mse, sav.ratio,
                    sav.original_bytes / (1024.0 * 1024.0), sav.compressed_bytes / (1024.0 * 1024.0));
        std::printf("%-45s   compress: %.2f ms (%.1f Melem/s)   decompress: %.2f ms\n", "",
                    compress_ms, (seq_len * cfg.head_dim) / (compress_ms * 1000.0), decompress_ms);
    }

    return 0;
}
