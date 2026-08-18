#pragma once
// turboquant/bitpack.hpp
//
// Packs arrays of small unsigned integers (arbitrary bit width, e.g. the
// Stage-1 codebook indices) and single bits (Stage-2 sign bits) into
// contiguous byte buffers. This is what actually realizes the "b bits per
// coordinate" compression -- without packing, an int storing an index for
// a 2-bit code would itself take 32/64 bits and defeat the purpose.

#include <cstdint>
#include <vector>
#include <cstddef>

namespace turboquant {

// Packs `values[0..n)`, each assumed to fit in `bits_per_value` bits, into
// a tightly packed byte buffer (LSB-first within each byte, values written
// in order starting from bit 0 of byte 0).
inline std::vector<uint8_t> pack_bits(const std::vector<uint32_t> &values, int bits_per_value) {
    const size_t n = values.size();
    const size_t total_bits = n * static_cast<size_t>(bits_per_value);
    std::vector<uint8_t> out((total_bits + 7) / 8, 0);
    size_t bit_pos = 0;
    for (size_t i = 0; i < n; ++i) {
        uint32_t v = values[i];
        for (int b = 0; b < bits_per_value; ++b) {
            if (v & (1u << b)) {
                out[bit_pos >> 3] |= static_cast<uint8_t>(1u << (bit_pos & 7));
            }
            bit_pos++;
        }
    }
    return out;
}

inline std::vector<uint32_t> unpack_bits(const std::vector<uint8_t> &packed, size_t n, int bits_per_value) {
    std::vector<uint32_t> out(n, 0);
    size_t bit_pos = 0;
    for (size_t i = 0; i < n; ++i) {
        uint32_t v = 0;
        for (int b = 0; b < bits_per_value; ++b) {
            size_t byte_idx = bit_pos >> 3;
            if (byte_idx < packed.size() && (packed[byte_idx] & (1u << (bit_pos & 7)))) {
                v |= (1u << b);
            }
            bit_pos++;
        }
        out[i] = v;
    }
    return out;
}

// Convenience specializations for 1-bit sign arrays (Stage 2 QJL bits),
// stored as bool-ish 0/1 rather than +-1 to keep packing generic.
inline std::vector<uint8_t> pack_signs(const std::vector<int8_t> &signs) {
    std::vector<uint32_t> bits(signs.size());
    for (size_t i = 0; i < signs.size(); ++i) bits[i] = (signs[i] > 0) ? 1u : 0u;
    return pack_bits(bits, 1);
}

inline std::vector<int8_t> unpack_signs(const std::vector<uint8_t> &packed, size_t n) {
    auto bits = unpack_bits(packed, n, 1);
    std::vector<int8_t> signs(n);
    for (size_t i = 0; i < n; ++i) signs[i] = bits[i] ? int8_t(1) : int8_t(-1);
    return signs;
}

} // namespace turboquant
