#pragma once
// turboquant/kv_cache.hpp
//
// Thin batch wrapper around TurboQuant for compressing a KV cache: a
// sequence of key/value vectors of fixed `head_dim`, laid out row-major
// as (n_vectors x head_dim) -- i.e. exactly the shape you get from one
// (batch, head) slice of a standard transformer KV cache.
//
// SCOPE (please read before wiring into bitnet.cpp / llama.cpp):
//   This mirrors the *core* compress/decompress functionality of the
//   Python reference's `TurboQuantKVCache`. It intentionally does NOT
//   reimplement every feature of the Python reference (sliding-window
//   fp16 residual buffer for recent tokens, outlier-channel routing,
//   GQA/MQA-aware bit bumping, per-layer adaptive bit allocation). Those
//   are all layered policy on top of the same core quantizer and can be
//   added incrementally; they are not required to get a working,
//   measurable compression path end-to-end.
//
//   Wiring this into bitnet.cpp itself is a separate, larger task: that
//   engine stores its KV cache as `ggml_tensor` buffers managed by
//   llama.cpp's `llama_kv_cache` machinery, and swapping the underlying
//   storage/attention read path for a quantized one means touching
//   llama.cpp internals (a custom ggml buffer type or a modified
//   attention kernel that decompresses per block), not just linking this
//   library. See the project README for concrete pointers on where to
//   start looking in llama.cpp/bitnet.cpp.

#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <algorithm>

#include "turboquant.hpp"

namespace turboquant {

struct CompressedKV {
    std::vector<QuantizedVector> keys;   // one entry per token position
    std::vector<QuantizedVector> values;
};

class TurboQuantKVCache {
public:
    TurboQuantKVCache(size_t head_dim, int bit_width, bool unbiased = true,
                       uint64_t seed_keys = 0x4B455953ULL /* "KEYS" */,
                       uint64_t seed_values = 0x56414C53ULL /* "VALS" */)
        : head_dim_(head_dim),
          tq_keys_(head_dim, bit_width, unbiased, seed_keys),
          tq_values_(head_dim, bit_width, unbiased, seed_values) {}

    // keys/values: row-major (n_vectors x head_dim_)
    CompressedKV compress(const float *keys, const float *values, size_t n_vectors) const {
        CompressedKV out;
        out.keys.reserve(n_vectors);
        out.values.reserve(n_vectors);
        for (size_t t = 0; t < n_vectors; ++t) {
            out.keys.push_back(tq_keys_.quantize(keys + t * head_dim_, head_dim_));
            out.values.push_back(tq_values_.quantize(values + t * head_dim_, head_dim_));
        }
        return out;
    }

    // Writes n_vectors * head_dim_ floats into `out` (caller-allocated).
    void decompress_keys(const CompressedKV &c, float *out) const {
        decompress_into(tq_keys_, c.keys, out);
    }
    void decompress_values(const CompressedKV &c, float *out) const {
        decompress_into(tq_values_, c.values, out);
    }

    // Returns {original_bytes, compressed_bytes, ratio} for n_vectors of
    // BOTH keys and values (matches the Python API's memory_savings()).
    struct MemorySavings {
        double original_bytes;
        double compressed_bytes;
        double ratio;
    };
    MemorySavings memory_savings(size_t n_vectors) const {
        double orig = 2.0 * n_vectors * head_dim_ * 4.0; // fp32, keys+values
        double comp_per_vec = (static_cast<double>(tq_keys_.padded_dim()) *
                                    (tq_keys_.unbiased() ? tq_keys_.bit_width() - 1 : tq_keys_.bit_width()) +
                                32.0 + (tq_keys_.unbiased() ? tq_keys_.padded_dim() + 32.0 : 0.0)) / 8.0;
        double comp = 2.0 * n_vectors * comp_per_vec; // keys+values, same config
        MemorySavings ms;
        ms.original_bytes = orig;
        ms.compressed_bytes = comp;
        ms.ratio = orig / comp;
        return ms;
    }

    size_t head_dim() const { return head_dim_; }

private:
    static void decompress_into(const TurboQuant &tq, const std::vector<QuantizedVector> &vecs, float *out) {
        size_t d = tq.dim();
        for (size_t t = 0; t < vecs.size(); ++t) {
            auto v = tq.dequantize(vecs[t]);
            std::copy(v.begin(), v.end(), out + t * d);
        }
    }

    size_t head_dim_;
    TurboQuant tq_keys_;
    TurboQuant tq_values_;
};

} // namespace turboquant
