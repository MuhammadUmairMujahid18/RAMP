#ifndef TOP_TRAIN_BM8_H
#define TOP_TRAIN_BM8_H

#include <hls_stream.h>

void top_train_bm8(
    hls::stream<float> &input_stream,
    hls::stream<float> &target_stream,
    hls::stream<float> &prediction_stream,
    int tensor_length,
    bool reset_state,
    float learning_rate,
    float momentum,
    float &loss,
    float &weight,
    float &bias,
    float &velocity_weight,
    float &velocity_bias
);

#endif
