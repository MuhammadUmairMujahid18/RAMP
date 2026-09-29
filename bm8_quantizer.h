#ifndef BM8_QUANTIZER_H
#define BM8_QUANTIZER_H

#include <ap_int.h>
#include <hls_stream.h>

#define BM8_BLOCK_SIZE 32
#define BM8_MAX_BLOCK_SIZE BM8_BLOCK_SIZE
#define BM8_BITS_PER_VALUE 8
#define BM8_PACKED_WIDTH (BM8_BLOCK_SIZE * BM8_BITS_PER_VALUE)
#define BM8_MAX_TENSOR_SIZE 8192
#define BM8_MAX_PACKED_BLOCKS ((BM8_MAX_TENSOR_SIZE + BM8_BLOCK_SIZE - 1) / BM8_BLOCK_SIZE)

struct Bm8PackedBlock {
    ap_int<8> shared_exp;
    ap_uint<16> valid_count;
    ap_uint<8> block_span;
    ap_uint<BM8_PACKED_WIDTH> packed_values;
};

void bm8_tensor_quantize(
    hls::stream<float> &in_stream,
    hls::stream<float> &out_stream,
    int tensor_length,
    bool use_stochastic
);

void bm8_tensor_quantize_buffer(
    const float input[],
    float output[],
    int tensor_length,
    bool use_stochastic
);

void bm8_tensor_quantize_packed(
    hls::stream<float> &in_stream,
    hls::stream<Bm8PackedBlock> &packed_stream,
    int tensor_length,
    bool use_stochastic
);

void bm8_tensor_quantize_packed(
    hls::stream<float> &in_stream,
    hls::stream<Bm8PackedBlock> &packed_stream,
    int tensor_length,
    bool use_stochastic,
    int block_size
);

void bm8_tensor_dequantize_packed(
    hls::stream<Bm8PackedBlock> &packed_stream,
    hls::stream<float> &out_stream,
    int tensor_length
);

void bm8_tensor_quantize_packed_buffer(
    const float input[],
    Bm8PackedBlock packed_blocks[],
    int tensor_length,
    bool use_stochastic
);

void bm8_tensor_quantize_packed_buffer(
    const float input[],
    Bm8PackedBlock packed_blocks[],
    int tensor_length,
    bool use_stochastic,
    int block_size
);

void bm8_tensor_dequantize_packed_buffer(
    const Bm8PackedBlock packed_blocks[],
    float output[],
    int tensor_length
);

#endif
