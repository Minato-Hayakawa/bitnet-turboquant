#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "bitnet_turboquant_kv.hpp"

namespace {
struct Options {
    int layers = 36, kv_heads = 72, head_dim = 144, context = 8192;
    int fidelity_tokens = 8, bit_width = 3;
    double baseline_mb = 10368.0;
    int baseline_context = 8192;
    uint32_t seed = 42;
    std::string csv_path;
};

void usage(const char *p) {
    std::cerr << "Usage: " << p << " [--bit-width 2|3|4] [--layers N] [--kv-heads N]"
              << " [--head-dim N] [--context N] [--baseline-mb N] [--baseline-context N] [--fidelity-tokens N]"
              << " [--csv FILE]\n"
              << "Runs a deterministic synthetic-KV fidelity probe.  For end-to-end token"
              << " accuracy, feed model logits to the companion Python runner.\n";
}
bool take_int(int &i, int argc, char **argv, int &out) {
    if (++i >= argc) return false; out = std::atoi(argv[i]); return true;
}
bool take_double(int &i, int argc, char **argv, double &out) {
    if (++i >= argc) return false; out = std::atof(argv[i]); return true;
}
double dot(const float *a, const float *b, int n) {
    double s = 0; for (int i = 0; i < n; ++i) s += double(a[i]) * b[i]; return s;
}
double cosine(const std::vector<float> &a, const std::vector<float> &b) {
    const double aa = dot(a.data(), a.data(), int(a.size()));
    const double bb = dot(b.data(), b.data(), int(b.size()));
    return aa == 0 || bb == 0 ? 0 : dot(a.data(), b.data(), int(a.size())) / std::sqrt(aa * bb);
}
std::vector<float> attention(const std::vector<float> &q, const std::vector<float> &k,
                             const std::vector<float> &v, int tokens, int dim) {
    std::vector<double> score(tokens); double max_s = -INFINITY;
    for (int t = 0; t < tokens; ++t) max_s = std::max(max_s, dot(q.data(), k.data() + t * dim, dim) / std::sqrt(double(dim)));
    double z = 0; for (int t = 0; t < tokens; ++t) z += (score[t] = std::exp(dot(q.data(), k.data() + t * dim, dim) / std::sqrt(double(dim)) - max_s));
    std::vector<float> out(dim, 0);
    for (int t = 0; t < tokens; ++t) for (int d = 0; d < dim; ++d) out[d] += float(score[t] / z) * v[t * dim + d];
    return out;
}
double encoded_bytes_per_vector(int dim, int bits) {
    const int padded = 1 << int(std::ceil(std::log2(std::max(1, dim))));
    // Stage 1 codes + two FP32 norms + one QJL sign bit per padded coordinate.
    return (padded * (bits - 1) + 32 + padded + 32) / 8.0;
}
} // namespace

int main(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i]; bool ok = true;
        if (a == "--bit-width") ok = take_int(i, argc, argv, o.bit_width);
        else if (a == "--layers") ok = take_int(i, argc, argv, o.layers);
        else if (a == "--kv-heads") ok = take_int(i, argc, argv, o.kv_heads);
        else if (a == "--head-dim") ok = take_int(i, argc, argv, o.head_dim);
        else if (a == "--context") ok = take_int(i, argc, argv, o.context);
        else if (a == "--baseline-context") ok = take_int(i, argc, argv, o.baseline_context);
        else if (a == "--fidelity-tokens") ok = take_int(i, argc, argv, o.fidelity_tokens);
        else if (a == "--baseline-mb") ok = take_double(i, argc, argv, o.baseline_mb);
        else if (a == "--csv") { if (++i < argc) o.csv_path = argv[i]; else ok = false; }
        else { usage(argv[0]); return 2; }
        if (!ok) { usage(argv[0]); return 2; }
    }
    if (o.bit_width < 2 || o.bit_width > 4 || o.layers < 1 || o.kv_heads < 1 || o.head_dim < 1 || o.context < 1 || o.baseline_context < 1) return 2;

    const double fp16_bytes = 2.0 * o.layers * o.kv_heads * o.context * o.head_dim * 2.0;
    const double scaled_baseline_mb = o.baseline_mb * double(o.context) / o.baseline_context;
    const double actual_ratio = 16.0 / (encoded_bytes_per_vector(o.head_dim, o.bit_width) * 8.0 / o.head_dim);
    const double compressed_mb = scaled_baseline_mb / actual_ratio;
    const double ideal_ratio = 16.0 / o.bit_width;
    const double ideal_compressed_mb = scaled_baseline_mb / ideal_ratio;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "compression,bit_width,context,baseline_mb,compressed_mb,ratio,ideal_compressed_mb,ideal_ratio\n";
    std::cout << "kv_cache," << o.bit_width << ',' << o.context << ',' << scaled_baseline_mb << ',' << compressed_mb << ',' << actual_ratio << ',' << ideal_compressed_mb << ',' << ideal_ratio << "\n";
    std::cout << "# derived FP16 size for " << o.layers << " layers x " << o.kv_heads << " KV heads x " << o.context
              << " tokens x " << o.head_dim << " dims: " << fp16_bytes / 1024.0 / 1024.0 << " MiB\n";
    if (std::abs(fp16_bytes / 1024.0 / 1024.0 - scaled_baseline_mb) > 1.0)
        std::cout << "# WARNING: --baseline-mb is an external reference and does not match the supplied geometry.\n";

    // Deterministic probe: each head is an independent KV slice; this measures
    // attention-output fidelity after the same quantize/dequantize path used by the adapter.
    const int t = std::min(o.context, o.fidelity_tokens);
    std::mt19937 gen(o.seed); std::normal_distribution<float> normal(0.f, 1.f);
    double cos_sum = 0; int samples = 0;
    for (int h = 0; h < o.kv_heads; ++h) {
        std::vector<float> k(t * o.head_dim), v(t * o.head_dim), q(o.head_dim), kh(k.size()), vh(v.size());
        for (float &x : k) x = normal(gen); for (float &x : v) x = normal(gen); for (float &x : q) x = normal(gen);
        bitnet_turboquant::KVCacheAdapter adapter(o.head_dim, o.bit_width, true);
        adapter.append(k.data(), v.data(), t); adapter.read_keys(kh.data()); adapter.read_values(vh.data());
        cos_sum += cosine(attention(q, k, v, t, o.head_dim), attention(q, kh, vh, t, o.head_dim)); ++samples;
    }
    const double attention_cos = cos_sum / samples;
    std::cout << "attention_fidelity,bit_width,layers_configured,kv_heads,probe_tokens,cosine_similarity,top1_token_match,top5_token_match\n";
    std::cout << "attention," << o.bit_width << ',' << o.layers << ',' << o.kv_heads << ',' << t << ',' << attention_cos << ",NA,NA\n";
    std::cout << "# Top-k needs model logits; NA prevents reporting fabricated token-accuracy values.\n";
    if (!o.csv_path.empty()) {
        std::ofstream f(o.csv_path);
        if (!f) { std::cerr << "cannot write " << o.csv_path << "\n"; return 1; }
        f << "kind,bit_width,context,baseline_mb,compressed_mb,ratio,ideal_compressed_mb,ideal_ratio,layers,kv_heads,probe_tokens,cosine_similarity,top1,top5\n";
        f << "compression," << o.bit_width << ',' << o.context << ',' << scaled_baseline_mb << ',' << compressed_mb << ',' << actual_ratio << ',' << ideal_compressed_mb << ',' << ideal_ratio << ",,,,,,\n";
        f << "attention," << o.bit_width << ",,,,,,,," << o.layers << ',' << o.kv_heads << ',' << t << ',' << attention_cos << ",NA,NA\n";
    }
}
