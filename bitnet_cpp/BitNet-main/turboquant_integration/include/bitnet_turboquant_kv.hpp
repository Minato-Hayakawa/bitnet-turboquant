#pragma once

// Boundary adapter for the llama.cpp KV-cache write/read path.  The llama.cpp
// submodule is intentionally not included in this repository checkout, so this
// class has no ggml dependency and can be exercised independently.  At the
// integration point, call append() when a FP16 KV vector is evicted/offloaded,
// and read_keys()/read_values() immediately before attention consumes it.

#include <cstddef>
#include <vector>

#include "turboquant/kv_cache.hpp"

namespace bitnet_turboquant {

class KVCacheAdapter {
public:
    KVCacheAdapter(size_t head_dim, int bit_width, bool unbiased = true)
        : codec_(head_dim, bit_width, unbiased), head_dim_(head_dim) {}

    void append(const float *keys, const float *values, size_t tokens) {
        compressed_ = codec_.compress(keys, values, tokens);
        tokens_ = tokens;
    }

    void read_keys(float *out) const { codec_.decompress_keys(compressed_, out); }
    void read_values(float *out) const { codec_.decompress_values(compressed_, out); }
    size_t tokens() const { return tokens_; }
    turboquant::TurboQuantKVCache::MemorySavings memory_savings() const {
        return codec_.memory_savings(tokens_);
    }

private:
    turboquant::TurboQuantKVCache codec_;
    turboquant::CompressedKV compressed_;
    size_t head_dim_;
    size_t tokens_ = 0;
};

} // namespace bitnet_turboquant
