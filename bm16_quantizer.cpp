#include "bm16_quantizer.h"

#include <math.h>
#include <stdint.h>

#define BM16_MAN_BITS 15
#define BM16_MAN_LEVELS ((1 << BM16_MAN_BITS) - 1)
#define BM16_MAX_EXP 15
#define BM16_MIN_EXP -16

static uint32_t bm16_lfsr = 0x13579BDFu;

static inline float bm16_sanitize_value(float val) {
    return isfinite(val) ? val : 0.0f;
}

static inline int bm16_select_shared_exp(float max_val) {
    if (max_val <= 0.0f) {
        return 0;
    }
    // Use ceil so the scaled block maximum stays within [0, 1] and avoids saturation.
    int exp = (int)ceilf(log2f(max_val));
    if (exp > BM16_MAX_EXP) exp = BM16_MAX_EXP;
    if (exp < BM16_MIN_EXP) exp = BM16_MIN_EXP;
    return exp;
}

static inline float bm16_stochastic_round(float val) {
    uint32_t bit = ((bm16_lfsr >> 0) ^ (bm16_lfsr >> 2) ^ (bm16_lfsr >> 3) ^ (bm16_lfsr >> 5)) & 1u;
    bm16_lfsr = (bm16_lfsr >> 1) | (bit << 31);
    return floorf(val + (float)bit);
}

static float bm16_block_max_abs(const float block_in[], int block_size) {
    float max_val = 0.0f;

    for (int i = 0; i < block_size; ++i) {
#pragma HLS PIPELINE
        float clean_val = bm16_sanitize_value(block_in[i]);
        float abs_val = (clean_val >= 0.0f) ? clean_val : -clean_val;
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }

    return max_val;
}

static Bm16PackedBlock bm16_block_pack(
    const float block_in[],
    int block_size,
    bool use_stochastic
) {
    Bm16PackedBlock packed_block;
    packed_block.shared_exp = 0;
    packed_block.valid_count = block_size;
    packed_block.block_span = block_size;
    packed_block.packed_values = 0;

    float max_val = bm16_block_max_abs(block_in, block_size);
    int exp = bm16_select_shared_exp(max_val);
    packed_block.shared_exp = exp;

    float scale = powf(2.0f, (float)(-exp));

    for (int i = 0; i < block_size; ++i) {
#pragma HLS PIPELINE
        float scaled = bm16_sanitize_value(block_in[i]) * scale;
        int sign = (scaled < 0.0f) ? 1 : 0;
        float abs_val = (scaled >= 0.0f) ? scaled : -scaled;
        if (abs_val > 1.0f) {
            abs_val = 1.0f;
        }

        float quant_input = abs_val * (float)BM16_MAN_LEVELS;
        float mantissa = use_stochastic ? bm16_stochastic_round(quant_input) : roundf(quant_input);
        if (mantissa > (float)BM16_MAN_LEVELS) {
            mantissa = (float)BM16_MAN_LEVELS;
        }

        ap_uint<16> encoded = 0;
        encoded[15] = sign;
        encoded.range(14, 0) = (ap_uint<15>)mantissa;
        packed_block.packed_values.range((i * 16) + 15, i * 16) = encoded;
    }

    return packed_block;
}

static void bm16_block_unpack(
    const Bm16PackedBlock &packed_block,
    float block_out[],
    int block_size
) {
    int exp = (int)packed_block.shared_exp;
    float recon_scale = powf(2.0f, (float)exp);

    for (int i = 0; i < block_size; ++i) {
#pragma HLS PIPELINE
        ap_uint<16> encoded = packed_block.packed_values.range((i * 16) + 15, i * 16);
        int sign = encoded[15];
        int mantissa = (int)encoded.range(14, 0);
        float frac = (float)mantissa / (float)BM16_MAN_LEVELS;
        float recon = frac * recon_scale;
        block_out[i] = sign ? -recon : recon;
    }
}

void bm16_tensor_quantize(
    hls::stream<float> &in_stream,
    hls::stream<float> &out_stream,
    int tensor_length,
    bool use_stochastic
) {
    hls::stream<Bm16PackedBlock> packed_stream("bm16_packed_stream");
#pragma HLS STREAM variable=packed_stream depth=4

    bm16_tensor_quantize_packed(in_stream, packed_stream, tensor_length, use_stochastic);
    bm16_tensor_dequantize_packed(packed_stream, out_stream, tensor_length);
}

void bm16_tensor_quantize_buffer(
    const float input[],
    float output[],
    int tensor_length,
    bool use_stochastic
) {
    Bm16PackedBlock packed_blocks[BM16_MAX_PACKED_BLOCKS];

    bm16_tensor_quantize_packed_buffer(input, packed_blocks, tensor_length, use_stochastic);
    bm16_tensor_dequantize_packed_buffer(packed_blocks, output, tensor_length);
}

void bm16_tensor_quantize_packed(
    hls::stream<float> &in_stream,
    hls::stream<Bm16PackedBlock> &packed_stream,
    int tensor_length,
    bool use_stochastic
) {
    bm16_tensor_quantize_packed(in_stream, packed_stream, tensor_length, use_stochastic, BM16_BLOCK_SIZE);
}

void bm16_tensor_quantize_packed(
    hls::stream<float> &in_stream,
    hls::stream<Bm16PackedBlock> &packed_stream,
    int tensor_length,
    bool use_stochastic,
    int configured_block_size
) {
    float block_in[BM16_BLOCK_SIZE];
    const int selected_block_size = (configured_block_size > 0 && configured_block_size <= BM16_BLOCK_SIZE)
        ? configured_block_size
        : BM16_BLOCK_SIZE;

    for (int base = 0; base < tensor_length; base += selected_block_size) {
        int block_size = selected_block_size;
        if (base + block_size > tensor_length) {
            block_size = tensor_length - base;
        }

        for (int i = 0; i < block_size; ++i) {
#pragma HLS PIPELINE
            block_in[i] = in_stream.read();
        }

        packed_stream.write(bm16_block_pack(block_in, block_size, use_stochastic));
    }
}

void bm16_tensor_dequantize_packed(
    hls::stream<Bm16PackedBlock> &packed_stream,
    hls::stream<float> &out_stream,
    int tensor_length
) {
    float block_out[BM16_BLOCK_SIZE];

    for (int base = 0; base < tensor_length; ) {
        Bm16PackedBlock packed_block = packed_stream.read();
        int block_size = (int)packed_block.valid_count;

        bm16_block_unpack(packed_block, block_out, block_size);

        for (int i = 0; i < block_size; ++i) {
#pragma HLS PIPELINE
            out_stream.write(block_out[i]);
        }

        base += (int)packed_block.block_span;
    }
}

void bm16_tensor_quantize_packed_buffer(
    const float input[],
    Bm16PackedBlock packed_blocks[],
    int tensor_length,
    bool use_stochastic
) {
    bm16_tensor_quantize_packed_buffer(input, packed_blocks, tensor_length, use_stochastic, BM16_BLOCK_SIZE);
}

void bm16_tensor_quantize_packed_buffer(
    const float input[],
    Bm16PackedBlock packed_blocks[],
    int tensor_length,
    bool use_stochastic,
    int configured_block_size
) {
    float block_in[BM16_BLOCK_SIZE];
    int block_index = 0;
    const int selected_block_size = (configured_block_size > 0 && configured_block_size <= BM16_BLOCK_SIZE)
        ? configured_block_size
        : BM16_BLOCK_SIZE;

    for (int base = 0; base < tensor_length; base += selected_block_size) {
        int block_size = selected_block_size;
        if (base + block_size > tensor_length) {
            block_size = tensor_length - base;
        }

        for (int i = 0; i < block_size; ++i) {
#pragma HLS PIPELINE
            block_in[i] = input[base + i];
        }

        packed_blocks[block_index++] = bm16_block_pack(block_in, block_size, use_stochastic);
    }
}

void bm16_tensor_dequantize_packed_buffer(
    const Bm16PackedBlock packed_blocks[],
    float output[],
    int tensor_length
) {
    float block_out[BM16_BLOCK_SIZE];
    int block_index = 0;

    for (int base = 0; base < tensor_length; ) {
        const Bm16PackedBlock &packed_block = packed_blocks[block_index++];
        int block_size = (int)packed_block.valid_count;

        bm16_block_unpack(packed_block, block_out, block_size);

        for (int i = 0; i < block_size; ++i) {
#pragma HLS PIPELINE
            output[base + i] = block_out[i];
        }

        base += (int)packed_block.block_span;
    }
}
