#include <hls_stream.h>

#include "resnet18_cifar.h"

// Dedicated BM8 training top: full ResNet-18 with BM8-emulated training state.
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
) {
#pragma HLS INTERFACE axis port=input_stream
#pragma HLS INTERFACE axis port=target_stream
#pragma HLS INTERFACE axis port=prediction_stream
#pragma HLS INTERFACE s_axilite port=tensor_length bundle=control
#pragma HLS INTERFACE s_axilite port=reset_state bundle=control
#pragma HLS INTERFACE s_axilite port=learning_rate bundle=control
#pragma HLS INTERFACE s_axilite port=momentum bundle=control
#pragma HLS INTERFACE s_axilite port=loss bundle=control
#pragma HLS INTERFACE s_axilite port=weight bundle=control
#pragma HLS INTERFACE s_axilite port=bias bundle=control
#pragma HLS INTERFACE s_axilite port=velocity_weight bundle=control
#pragma HLS INTERFACE s_axilite port=velocity_bias bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static ResNet18Params params;
    static ResNet18State opt_state;
    static bool initialized = false;
    float input_buffer[RESNET18_BATCH_SIZE][RESNET18_INPUT_SIZE];
    float logits[RESNET18_BATCH_SIZE][RESNET18_NUM_CLASSES];
    ResNet18StepResult batch_results[RESNET18_BATCH_SIZE];
    int labels[RESNET18_BATCH_SIZE];
    const int valid_length = (tensor_length > RESNET18_INPUT_SIZE) ? RESNET18_INPUT_SIZE : tensor_length;

    if (!initialized) {
        init_resnet18_params(params);
        reset_resnet18_state(opt_state);
        initialized = true;
    }

    if (reset_state) {
        init_resnet18_params(params);
        reset_resnet18_state(opt_state);
    }

    for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
        for (int i = 0; i < valid_length; ++i) {
#pragma HLS PIPELINE
            input_buffer[sample][i] = input_stream.read();
        }
        labels[sample] = (int)target_stream.read();
        if (labels[sample] < 0) {
            labels[sample] = 0;
        }
        if (labels[sample] >= RESNET18_NUM_CLASSES) {
            labels[sample] = RESNET18_NUM_CLASSES - 1;
        }
    }

    resnet18_train_batch_bm8(
        input_buffer,
        labels,
        learning_rate,
        momentum,
        params,
        opt_state,
        logits,
        batch_results,
        loss
    );

    for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
        for (int i = 0; i < RESNET18_NUM_CLASSES; ++i) {
#pragma HLS PIPELINE
            prediction_stream.write(logits[sample][i]);
        }
    }

    weight = params.fc_weight[0];
    bias = params.fc_bias[0];
    velocity_weight = opt_state.velocity_fc_weight[0];
    velocity_bias = opt_state.velocity_fc_bias[0];
}
