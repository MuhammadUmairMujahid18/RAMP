#include "bm8_quantizer.h"

#include <math.h>
#include <stdint.h>

#define BM8_MAN_BITS 7
#define BM8_MAN_LEVELS ((1 << BM8_MAN_BITS) - 1)  // Maximun Mantissa Value
#define BM8_INV_MAN_LEVELS (1.0f / 127.0f)
#define BM8_MAX_EXP 7
#define BM8_MIN_EXP -16

static uint32_t bm8_lfsr = 0xBEEF1234u;

static inline float bm8_sanitize_value(float val) {
    return isfinite(val) ? val : 0.0f;
}

static inline int bm8_select_shared_exp(float max_val) {
    if (max_val <= 0.0f) {
        return 0;
    }
    int exp = (int)floorf(log2f(max_val));
    if (exp > BM8_MAX_EXP) exp = BM8_MAX_EXP;
    if (exp < BM8_MIN_EXP) exp = BM8_MIN_EXP;
    return exp;
}

static inline float bm8_stochastic_round(float val) {
    uint32_t bit = ((bm8_lfsr >> 0) ^ (bm8_lfsr >> 1) ^ (bm8_lfsr >> 21) ^ (bm8_lfsr >> 31)) & 1u;
    bm8_lfsr = (bm8_lfsr >> 1) | (bit << 31);
    return floorf(val + (float)bit);
}

static float bm8_pow2_int(int exp) {
#pragma HLS INLINE
    float value = 1.0f;

    if (exp >= 0) {
        for (int i = 0; i < BM8_MAX_EXP; ++i) {
#pragma HLS UNROLL
            if (i < exp) {
                value *= 2.0f;
            }
        }
    } else {
        const int neg_exp = -exp;
        for (int i = 0; i < -BM8_MIN_EXP; ++i) {
#pragma HLS UNROLL
            if (i < neg_exp) {
                value *= 0.5f;
            }
        }
    }

    return value;
}

static float bm8_block_max_abs(const float block_in[], int block_size) {
    float max_val = 0.0f;

    for (int i = 0; i < block_size; ++i) {
#pragma HLS PIPELINE
        float clean_val = bm8_sanitize_value(block_in[i]);
        float abs_val = (clean_val >= 0.0f) ? clean_val : -clean_val;
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }

    return max_val;
}

static Bm8PackedBlock bm8_block_pack(
    const float block_in[],
    int block_size,
    bool use_stochastic
) {
    Bm8PackedBlock packed_block;
    packed_block.shared_exp = 0;
    packed_block.valid_count = block_size;
    packed_block.block_span = block_size;
    packed_block.packed_values = 0;

    float max_val = bm8_block_max_abs(block_in, block_size);
    int exp = bm8_select_shared_exp(max_val);
    packed_block.shared_exp = exp;

    // Use shift-style power-of-two scaling instead of powf(2.0f, (float)exp).
    // Vitis 2025.2 can otherwise generate mismatched signed/unsigned sitofp IP
    // names during RTL cosimulation for this small BM8 kernel.
    float scale = bm8_pow2_int(-exp);

    for (int i = 0; i < block_size; ++i) {
#pragma HLS PIPELINE
        float scaled = bm8_sanitize_value(block_in[i]) * scale;
        int sign = (scaled < 0.0f) ? 1 : 0;
        float abs_val = (scaled >= 0.0f) ? scaled : -scaled;
        if (abs_val > 1.0f) {
            abs_val = 1.0f;
        }

        float quant_input = abs_val * 127.0f;
        float mantissa = use_stochastic ? bm8_stochastic_round(quant_input) : roundf(quant_input);
        if (mantissa > 127.0f) {
            mantissa = 127.0f;
        }

        ap_uint<8> encoded = 0;
        encoded[7] = sign;
        encoded.range(6, 0) = (ap_uint<7>)mantissa;
        packed_block.packed_values.range((i * 8) + 7, i * 8) = encoded;
    }

    return packed_block;
}

static void bm8_block_unpack(
    const Bm8PackedBlock &packed_block,
    float block_out[],
    int block_size
) {
    int exp = (int)packed_block.shared_exp;
    float recon_scale = bm8_pow2_int(exp);

    for (int i = 0; i < block_size; ++i) {
#pragma HLS PIPELINE
        ap_uint<8> encoded = packed_block.packed_values.range((i * 8) + 7, i * 8);
        int sign = encoded[7];
        ap_uint<7> mantissa = encoded.range(6, 0);
        float frac = mantissa.to_uint() * BM8_INV_MAN_LEVELS;
        float recon = frac * recon_scale;
        block_out[i] = sign ? -recon : recon;
    }
}

void bm8_tensor_quantize(
    hls::stream<float> &in_stream,
    hls::stream<float> &out_stream,
    int tensor_length,
    bool use_stochastic
) {
    hls::stream<Bm8PackedBlock> packed_stream("bm8_packed_stream");
#pragma HLS STREAM variable=packed_stream depth=4

    bm8_tensor_quantize_packed(in_stream, packed_stream, tensor_length, use_stochastic);
    bm8_tensor_dequantize_packed(packed_stream, out_stream, tensor_length);
}

void bm8_tensor_quantize_buffer(
    const float input[],
    float output[],
    int tensor_length,
    bool use_stochastic
) {
    Bm8PackedBlock packed_blocks[BM8_MAX_PACKED_BLOCKS];

    bm8_tensor_quantize_packed_buffer(input, packed_blocks, tensor_length, use_stochastic);
    bm8_tensor_dequantize_packed_buffer(packed_blocks, output, tensor_length);
}

void bm8_tensor_quantize_packed(
    hls::stream<float> &in_stream,
    hls::stream<Bm8PackedBlock> &packed_stream,
    int tensor_length,
    bool use_stochastic
) {
    bm8_tensor_quantize_packed(in_stream, packed_stream, tensor_length, use_stochastic, BM8_BLOCK_SIZE);
}

void bm8_tensor_quantize_packed(
    hls::stream<float> &in_stream,
    hls::stream<Bm8PackedBlock> &packed_stream,
    int tensor_length,
    bool use_stochastic,
    int configured_block_size
) {
    float block_in[BM8_BLOCK_SIZE];
    const int selected_block_size = (configured_block_size > 0 && configured_block_size <= BM8_BLOCK_SIZE)
        ? configured_block_size
        : BM8_BLOCK_SIZE;

    for (int base = 0; base < tensor_length; base += selected_block_size) {
        int block_size = selected_block_size;
        if (base + block_size > tensor_length) {
            block_size = tensor_length - base;
        }

        for (int i = 0; i < block_size; ++i) {
#pragma HLS PIPELINE
            block_in[i] = in_stream.read();
        }

        packed_stream.write(bm8_block_pack(block_in, block_size, use_stochastic));
    }
}

void bm8_tensor_dequantize_packed(
    hls::stream<Bm8PackedBlock> &packed_stream,
    hls::stream<float> &out_stream,
    int tensor_length
) {
    float block_out[BM8_BLOCK_SIZE];

    for (int base = 0; base < tensor_length; ) {
        Bm8PackedBlock packed_block = packed_stream.read();
        int block_size = (int)packed_block.valid_count;

        bm8_block_unpack(packed_block, block_out, block_size);

        for (int i = 0; i < block_size; ++i) {
#pragma HLS PIPELINE
            out_stream.write(block_out[i]);
        }

        base += (int)packed_block.block_span;
    }
}

void bm8_tensor_quantize_packed_buffer(
    const float input[],
    Bm8PackedBlock packed_blocks[],
    int tensor_length,
    bool use_stochastic
) {
    bm8_tensor_quantize_packed_buffer(input, packed_blocks, tensor_length, use_stochastic, BM8_BLOCK_SIZE);
}

void bm8_tensor_quantize_packed_buffer(
    const float input[],
    Bm8PackedBlock packed_blocks[],
    int tensor_length,
    bool use_stochastic,
    int configured_block_size
) {
    float block_in[BM8_BLOCK_SIZE];
    int block_index = 0;
    const int selected_block_size = (configured_block_size > 0 && configured_block_size <= BM8_BLOCK_SIZE)
        ? configured_block_size
        : BM8_BLOCK_SIZE;

    for (int base = 0; base < tensor_length; base += selected_block_size) {
        int block_size = selected_block_size;
        if (base + block_size > tensor_length) {
            block_size = tensor_length - base;
        }

        for (int i = 0; i < block_size; ++i) {
#pragma HLS PIPELINE
            block_in[i] = input[base + i];
        }

        packed_blocks[block_index++] = bm8_block_pack(block_in, block_size, use_stochastic);
    }
}

void bm8_tensor_dequantize_packed_buffer(
    const Bm8PackedBlock packed_blocks[],
    float output[],
    int tensor_length
) {
    float block_out[BM8_BLOCK_SIZE];
    int block_index = 0;

    for (int base = 0; base < tensor_length; ) {
        const Bm8PackedBlock &packed_block = packed_blocks[block_index++];
        int block_size = (int)packed_block.valid_count;

        bm8_block_unpack(packed_block, block_out, block_size);

        for (int i = 0; i < block_size; ++i) {
#pragma HLS PIPELINE
            output[base + i] = block_out[i];
        }

        base += (int)packed_block.block_span;
    }
}
