#ifndef BM16_QUANTIZER_H
#define BM16_QUANTIZER_H

#include <ap_int.h>
#include <hls_stream.h>

#define BM16_BLOCK_SIZE 16
#define BM16_MAX_BLOCK_SIZE BM16_BLOCK_SIZE
#define BM16_BITS_PER_VALUE 16
#define BM16_PACKED_WIDTH (BM16_BLOCK_SIZE * BM16_BITS_PER_VALUE)
#define BM16_MAX_TENSOR_SIZE 8192
#define BM16_MAX_PACKED_BLOCKS ((BM16_MAX_TENSOR_SIZE + BM16_BLOCK_SIZE - 1) / BM16_BLOCK_SIZE)

struct Bm16PackedBlock {
    ap_int<8> shared_exp;
    ap_uint<16> valid_count;
    ap_uint<8> block_span;
    ap_uint<BM16_PACKED_WIDTH> packed_values;
};

void bm16_tensor_quantize(
    hls::stream<float> &in_stream,
    hls::stream<float> &out_stream,
    int tensor_length,
    bool use_stochastic
);

void bm16_tensor_quantize_buffer(
    const float input[],
    float output[],
    int tensor_length,
    bool use_stochastic
);

void bm16_tensor_quantize_packed(
    hls::stream<float> &in_stream,
    hls::stream<Bm16PackedBlock> &packed_stream,
    int tensor_length,
    bool use_stochastic
);

void bm16_tensor_quantize_packed(
    hls::stream<float> &in_stream,
    hls::stream<Bm16PackedBlock> &packed_stream,
    int tensor_length,
    bool use_stochastic,
    int block_size
);

void bm16_tensor_dequantize_packed(
    hls::stream<Bm16PackedBlock> &packed_stream,
    hls::stream<float> &out_stream,
    int tensor_length
);

void bm16_tensor_quantize_packed_buffer(
    const float input[],
    Bm16PackedBlock packed_blocks[],
    int tensor_length,
    bool use_stochastic
);

void bm16_tensor_quantize_packed_buffer(
    const float input[],
    Bm16PackedBlock packed_blocks[],
    int tensor_length,
    bool use_stochastic,
    int block_size
);

void bm16_tensor_dequantize_packed_buffer(
    const Bm16PackedBlock packed_blocks[],
    float output[],
    int tensor_length
);

#endif
