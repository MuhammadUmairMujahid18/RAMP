#include "resnet18_cifar.h"

#include <math.h>

namespace {

static float scratch0[RESNET18_MAX_ACTIVATION_ELEMENTS];
static float scratch1[RESNET18_MAX_ACTIVATION_ELEMENTS];
static float scratch2[RESNET18_MAX_ACTIVATION_ELEMENTS];
static ResNet18BatchTrainingWorkspace default_batch_workspace;

inline int feature_index(int c, int y, int x, int height, int width) {
    return ((c * height) + y) * width + x;
}

inline int conv_index(int oc, int ic, int ky, int kx, int in_channels) {
    return (((oc * in_channels) + ic) * RESNET18_KERNEL_SIZE + ky) * RESNET18_KERNEL_SIZE + kx;
}

inline int proj_index(int oc, int ic, int in_channels) {
    return (oc * in_channels) + ic;
}

inline int fc_index(int cls, int channel) {
    return (cls * RESNET18_STAGE4_CHANNELS) + channel;
}

float relu(float value) {
#pragma HLS INLINE
    return (value > 0.0f) ? value : 0.0f;
}

float clamp_gradient(float value) {
#pragma HLS INLINE
    if (value > RESNET18_GRAD_CLIP_VALUE) {
        return RESNET18_GRAD_CLIP_VALUE;
    }
    if (value < -RESNET18_GRAD_CLIP_VALUE) {
        return -RESNET18_GRAD_CLIP_VALUE;
    }
    return value;
}

void zero_buffer(float buffer[], int length) {
#pragma HLS INLINE
    for (int i = 0; i < length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=65536 avg=8192
#pragma HLS PIPELINE II=1
        buffer[i] = 0.0f;
    }
}

void copy_buffer(const float input[], float output[], int length) {
#pragma HLS INLINE
    for (int i = 0; i < length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=65536 avg=8192
#pragma HLS PIPELINE II=1
        output[i] = input[i];
    }
}

void relu_inplace(float buffer[], int length) {
#pragma HLS INLINE
    for (int i = 0; i < length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=65536 avg=8192
#pragma HLS PIPELINE II=1
        buffer[i] = relu(buffer[i]);
    }
}

void apply_relu_gradient(const float activation[], float gradient[], int length) {
#pragma HLS INLINE
    for (int i = 0; i < length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=65536 avg=8192
#pragma HLS PIPELINE II=1
        gradient[i] = (activation[i] > 0.0f) ? gradient[i] : 0.0f;
    }
}

void compute_batchnorm_stats(
    float batch_data[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    int channels,
    int height,
    int width,
    BatchNormStats &stats
) {
    const int spatial = height * width;
    const float count = (float)(RESNET18_BATCH_SIZE * spatial);
    const float epsilon = 1e-5f;
#pragma HLS INLINE off

    for (int c = 0; c < channels; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=512 avg=256
        float sum = 0.0f;
        for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
            for (int idx = 0; idx < spatial; ++idx) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=1024 avg=256
#pragma HLS PIPELINE II=4
                sum += batch_data[sample][(c * spatial) + idx];
            }
        }
        stats.mean[c] = sum / count;
    }

    for (int c = 0; c < channels; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=512 avg=256
        float sq_sum = 0.0f;
        for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
            for (int idx = 0; idx < spatial; ++idx) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=1024 avg=256
#pragma HLS PIPELINE II=4
                float centered = batch_data[sample][(c * spatial) + idx] - stats.mean[c];
                sq_sum += centered * centered;
            }
        }
        stats.inv_std[c] = 1.0f / sqrtf((sq_sum / count) + epsilon);
    }
}

void apply_batchnorm_inplace(
    float batch_data[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    int channels,
    int height,
    int width,
    const BatchNormStats &stats,
    const float gamma[],
    const float beta[]
) {
    const int spatial = height * width;
#pragma HLS INLINE off
    for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
        for (int c = 0; c < channels; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=512 avg=256
            for (int idx = 0; idx < spatial; ++idx) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=1024 avg=256
#pragma HLS PIPELINE II=1
                int flat_index = (c * spatial) + idx;
                float normalized =
                    (batch_data[sample][flat_index] - stats.mean[c]) * stats.inv_std[c];
                batch_data[sample][flat_index] = normalized * gamma[c] + beta[c];
            }
        }
    }
}

void batchnorm_backward_from_input(
    float input_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    float grad_output_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    int channels,
    int height,
    int width,
    const BatchNormStats &stats,
    const float gamma[],
    float grad_input_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    float grad_gamma[],
    float grad_beta[]
) {
    const int spatial = height * width;
    const float count = (float)(RESNET18_BATCH_SIZE * spatial);
#pragma HLS INLINE off

    for (int c = 0; c < channels; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=512 avg=256
        float sum_grad = 0.0f;
        float sum_grad_xhat = 0.0f;

        for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
            for (int idx = 0; idx < spatial; ++idx) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=1024 avg=256
#pragma HLS PIPELINE II=4
                int flat_index = (c * spatial) + idx;
                float xhat = (input_batch[sample][flat_index] - stats.mean[c]) * stats.inv_std[c];
                float grad = grad_output_batch[sample][flat_index];
                sum_grad += grad;
                sum_grad_xhat += grad * xhat;
            }
        }

        grad_beta[c] += sum_grad;
        grad_gamma[c] += sum_grad_xhat;

        for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
            for (int idx = 0; idx < spatial; ++idx) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=1024 avg=256
#pragma HLS PIPELINE II=1
                int flat_index = (c * spatial) + idx;
                float xhat = (input_batch[sample][flat_index] - stats.mean[c]) * stats.inv_std[c];
                float grad = grad_output_batch[sample][flat_index];
                grad_input_batch[sample][flat_index] =
                    ((gamma[c] * stats.inv_std[c]) / count) *
                    ((count * grad) - sum_grad - (xhat * sum_grad_xhat));
            }
        }
    }
}

void conv3x3_forward(
    const float input[],
    int in_channels,
    int in_height,
    int in_width,
    const float weight[],
    const float bias[],
    int out_channels,
    int stride,
    float output[]
) {
    const int out_height = (in_height + stride - 1) / stride;
    const int out_width = (in_width + stride - 1) / stride;
#pragma HLS INLINE off

    for (int oc = 0; oc < out_channels; ++oc) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=512 avg=256
        for (int oy = 0; oy < out_height; ++oy) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=32 avg=16
            for (int ox = 0; ox < out_width; ++ox) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=32 avg=16
                float acc = bias[oc];
                for (int ic = 0; ic < in_channels; ++ic) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=512 avg=128
                    for (int ky = 0; ky < RESNET18_KERNEL_SIZE; ++ky) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3 avg=3
                        for (int kx = 0; kx < RESNET18_KERNEL_SIZE; ++kx) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3 avg=3
#pragma HLS PIPELINE II=3
                            int iy = (oy * stride) + ky - 1;
                            int ix = (ox * stride) + kx - 1;
                            if (iy >= 0 && iy < in_height && ix >= 0 && ix < in_width) {
                                acc += weight[conv_index(oc, ic, ky, kx, in_channels)] *
                                    input[feature_index(ic, iy, ix, in_height, in_width)];
                            }
                        }
                    }
                }
                output[feature_index(oc, oy, ox, out_height, out_width)] = acc;
            }
        }
    }
}

void conv1x1_forward(
    const float input[],
    int in_channels,
    int in_height,
    int in_width,
    const float weight[],
    const float bias[],
    int out_channels,
    int stride,
    float output[]
) {
    const int out_height = (in_height + stride - 1) / stride;
    const int out_width = (in_width + stride - 1) / stride;
#pragma HLS INLINE off

    for (int oc = 0; oc < out_channels; ++oc) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=512 avg=256
        for (int oy = 0; oy < out_height; ++oy) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=32 avg=16
            for (int ox = 0; ox < out_width; ++ox) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=32 avg=16
                float acc = bias[oc];
                int iy = oy * stride;
                int ix = ox * stride;
                for (int ic = 0; ic < in_channels; ++ic) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=512 avg=256
#pragma HLS PIPELINE II=4
                    acc += weight[proj_index(oc, ic, in_channels)] *
                        input[feature_index(ic, iy, ix, in_height, in_width)];
                }
                output[feature_index(oc, oy, ox, out_height, out_width)] = acc;
            }
        }
    }
}

void conv3x3_backward(
    const float input[],
    int in_channels,
    int in_height,
    int in_width,
    const float grad_output[],
    int out_channels,
    int stride,
    const float weight[],
    float grad_input[],
    float grad_weight[],
    float grad_bias[]
) {
    const int out_height = (in_height + stride - 1) / stride;
    const int out_width = (in_width + stride - 1) / stride;
    const int input_length = in_channels * in_height * in_width;
#pragma HLS INLINE off

    zero_buffer(grad_input, input_length);

    for (int oc = 0; oc < out_channels; ++oc) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=512 avg=256
        for (int oy = 0; oy < out_height; ++oy) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=32 avg=16
            for (int ox = 0; ox < out_width; ++ox) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=32 avg=16
                float grad = grad_output[feature_index(oc, oy, ox, out_height, out_width)];
                grad_bias[oc] += grad;
                for (int ic = 0; ic < in_channels; ++ic) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=512 avg=128
                    for (int ky = 0; ky < RESNET18_KERNEL_SIZE; ++ky) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3 avg=3
                        for (int kx = 0; kx < RESNET18_KERNEL_SIZE; ++kx) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3 avg=3
#pragma HLS PIPELINE II=4
                            int iy = (oy * stride) + ky - 1;
                            int ix = (ox * stride) + kx - 1;
                            if (iy >= 0 && iy < in_height && ix >= 0 && ix < in_width) {
                                int input_idx = feature_index(ic, iy, ix, in_height, in_width);
                                int weight_idx = conv_index(oc, ic, ky, kx, in_channels);
                                grad_weight[weight_idx] += grad * input[input_idx];
                                grad_input[input_idx] += grad * weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

void conv1x1_backward(
    const float input[],
    int in_channels,
    int in_height,
    int in_width,
    const float grad_output[],
    int out_channels,
    int stride,
    const float weight[],
    float grad_input[],
    float grad_weight[],
    float grad_bias[]
) {
    const int out_height = (in_height + stride - 1) / stride;
    const int out_width = (in_width + stride - 1) / stride;
    const int input_length = in_channels * in_height * in_width;
#pragma HLS INLINE off

    zero_buffer(grad_input, input_length);

    for (int oc = 0; oc < out_channels; ++oc) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=512 avg=256
        for (int oy = 0; oy < out_height; ++oy) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=32 avg=16
            for (int ox = 0; ox < out_width; ++ox) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=32 avg=16
                float grad = grad_output[feature_index(oc, oy, ox, out_height, out_width)];
                grad_bias[oc] += grad;
                int iy = oy * stride;
                int ix = ox * stride;
                for (int ic = 0; ic < in_channels; ++ic) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=512 avg=256
#pragma HLS PIPELINE II=4
                    int input_idx = feature_index(ic, iy, ix, in_height, in_width);
                    int weight_idx = proj_index(oc, ic, in_channels);
                    grad_weight[weight_idx] += grad * input[input_idx];
                    grad_input[input_idx] += grad * weight[weight_idx];
                }
            }
        }
    }
}

void basic_block_forward(
    const float input[],
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int stride,
    bool use_projection,
    const float conv1_weight[],
    const float conv1_bias[],
    const float conv2_weight[],
    const float conv2_bias[],
    const float proj_weight[],
    const float proj_bias[],
    float conv1_out[],
    float output[]
) {
    const int out_height = (in_height + stride - 1) / stride;
    const int out_width = (in_width + stride - 1) / stride;
    const int output_length = out_channels * out_height * out_width;
#pragma HLS INLINE off

    conv3x3_forward(input, in_channels, in_height, in_width, conv1_weight, conv1_bias, out_channels, stride, conv1_out);
    relu_inplace(conv1_out, output_length);
    conv3x3_forward(conv1_out, out_channels, out_height, out_width, conv2_weight, conv2_bias, out_channels, 1, output);

    if (use_projection) {
        conv1x1_forward(input, in_channels, in_height, in_width, proj_weight, proj_bias, out_channels, stride, scratch0);
    } else {
        copy_buffer(input, scratch0, output_length);
    }

    for (int i = 0; i < output_length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=8192 max=65536 avg=32768
#pragma HLS PIPELINE II=1
        output[i] = relu(output[i] + scratch0[i]);
    }
}

void basic_block_backward(
    const float input[],
    const float conv1_out[],
    const float block_output[],
    const float grad_output[],
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int stride,
    bool use_projection,
    const float conv1_weight[],
    const float conv2_weight[],
    const float proj_weight[],
    float grad_input[],
    float grad_conv1_weight[],
    float grad_conv1_bias[],
    float grad_conv2_weight[],
    float grad_conv2_bias[],
    float grad_proj_weight[],
    float grad_proj_bias[]
) {
    const int out_height = (in_height + stride - 1) / stride;
    const int out_width = (in_width + stride - 1) / stride;
    const int output_length = out_channels * out_height * out_width;
    const int input_length = in_channels * in_height * in_width;
#pragma HLS INLINE off

    zero_buffer(grad_input, input_length);

    for (int i = 0; i < output_length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=8192 max=65536 avg=32768
#pragma HLS PIPELINE II=1
        scratch0[i] = (block_output[i] > 0.0f) ? grad_output[i] : 0.0f;
    }

    conv3x3_backward(
        conv1_out,
        out_channels,
        out_height,
        out_width,
        scratch0,
        out_channels,
        1,
        conv2_weight,
        scratch1,
        grad_conv2_weight,
        grad_conv2_bias
    );
    apply_relu_gradient(conv1_out, scratch1, output_length);
    conv3x3_backward(
        input,
        in_channels,
        in_height,
        in_width,
        scratch1,
        out_channels,
        stride,
        conv1_weight,
        scratch2,
        grad_conv1_weight,
        grad_conv1_bias
    );

    for (int i = 0; i < input_length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=8192 max=65536 avg=32768
#pragma HLS PIPELINE II=1
        grad_input[i] += scratch2[i];
    }

    if (use_projection) {
        conv1x1_backward(
            input,
            in_channels,
            in_height,
            in_width,
            scratch0,
            out_channels,
            stride,
            proj_weight,
            scratch2,
            grad_proj_weight,
            grad_proj_bias
        );
        for (int i = 0; i < input_length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=8192 max=65536 avg=32768
#pragma HLS PIPELINE II=1
            grad_input[i] += scratch2[i];
        }
    } else {
        for (int i = 0; i < input_length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=8192 max=65536 avg=32768
#pragma HLS PIPELINE II=1
            grad_input[i] += scratch0[i];
        }
    }
}

void basic_block_forward_batchnorm(
    const float input_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int stride,
    bool use_projection,
    const float conv1_weight[],
    const float conv1_bias[],
    const float conv1_bn_gamma[],
    const float conv1_bn_beta[],
    const float conv2_weight[],
    const float conv2_bias[],
    const float conv2_bn_gamma[],
    const float conv2_bn_beta[],
    const float proj_weight[],
    const float proj_bias[],
    const float proj_bn_gamma[],
    const float proj_bn_beta[],
    BatchNormStats &conv1_stats,
    BatchNormStats &conv2_stats,
    BatchNormStats *proj_stats,
    float conv1_out_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    float output_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    ResNet18BatchTrainingWorkspace &workspace
) {
    const int out_height = (in_height + stride - 1) / stride;
    const int out_width = (in_width + stride - 1) / stride;
    const int output_length = out_channels * out_height * out_width;
    float (*conv2_out_batch)[RESNET18_MAX_ACTIVATION_ELEMENTS] = workspace.batch_tmp0;
    float (*shortcut_batch)[RESNET18_MAX_ACTIVATION_ELEMENTS] = workspace.batch_tmp1;
#pragma HLS INLINE off

    for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
        conv3x3_forward(
            input_batch[sample],
            in_channels,
            in_height,
            in_width,
            conv1_weight,
            conv1_bias,
            out_channels,
            stride,
            conv1_out_batch[sample]
        );
    }

    compute_batchnorm_stats(conv1_out_batch, out_channels, out_height, out_width, conv1_stats);
    apply_batchnorm_inplace(
        conv1_out_batch,
        out_channels,
        out_height,
        out_width,
        conv1_stats,
        conv1_bn_gamma,
        conv1_bn_beta
    );
    for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
        relu_inplace(conv1_out_batch[sample], output_length);
        conv3x3_forward(
            conv1_out_batch[sample],
            out_channels,
            out_height,
            out_width,
            conv2_weight,
            conv2_bias,
            out_channels,
            1,
            conv2_out_batch[sample]
        );
    }

    compute_batchnorm_stats(conv2_out_batch, out_channels, out_height, out_width, conv2_stats);
    apply_batchnorm_inplace(
        conv2_out_batch,
        out_channels,
        out_height,
        out_width,
        conv2_stats,
        conv2_bn_gamma,
        conv2_bn_beta
    );

    if (use_projection) {
        for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
            conv1x1_forward(
                input_batch[sample],
                in_channels,
                in_height,
                in_width,
                proj_weight,
                proj_bias,
                out_channels,
                stride,
                shortcut_batch[sample]
            );
        }
        compute_batchnorm_stats(shortcut_batch, out_channels, out_height, out_width, *proj_stats);
        apply_batchnorm_inplace(
            shortcut_batch,
            out_channels,
            out_height,
            out_width,
            *proj_stats,
            proj_bn_gamma,
            proj_bn_beta
        );
    }

    for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
        for (int i = 0; i < output_length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=8192 max=65536 avg=32768
#pragma HLS PIPELINE II=1
            float shortcut = use_projection ? shortcut_batch[sample][i] : input_batch[sample][i];
            output_batch[sample][i] = relu(conv2_out_batch[sample][i] + shortcut);
        }
    }
}

void basic_block_backward_batchnorm(
    const float input_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    float conv1_out_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    float block_output_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    float grad_output_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int stride,
    bool use_projection,
    const float conv1_weight[],
    const float conv1_bias[],
    const float conv1_bn_gamma[],
    const float conv2_weight[],
    const float conv2_bias[],
    const float conv2_bn_gamma[],
    const float proj_weight[],
    const float proj_bias[],
    const float proj_bn_gamma[],
    const BatchNormStats &conv1_stats,
    const BatchNormStats &conv2_stats,
    const BatchNormStats *proj_stats,
    float grad_input_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    float grad_conv1_weight[],
    float grad_conv1_bias[],
    float grad_conv1_bn_gamma[],
    float grad_conv1_bn_beta[],
    float grad_conv2_weight[],
    float grad_conv2_bias[],
    float grad_conv2_bn_gamma[],
    float grad_conv2_bn_beta[],
    float grad_proj_weight[],
    float grad_proj_bias[],
    float grad_proj_bn_gamma[],
    float grad_proj_bn_beta[],
    ResNet18BatchTrainingWorkspace &workspace
) {
    const int out_height = (in_height + stride - 1) / stride;
    const int out_width = (in_width + stride - 1) / stride;
    const int output_length = out_channels * out_height * out_width;
    const int input_length = in_channels * in_height * in_width;
    float (*relu_grad_batch)[RESNET18_MAX_ACTIVATION_ELEMENTS] = workspace.batch_tmp0;
    float (*recomputed_batch)[RESNET18_MAX_ACTIVATION_ELEMENTS] = workspace.batch_tmp1;
    float (*bn_grad_batch)[RESNET18_MAX_ACTIVATION_ELEMENTS] = workspace.batch_tmp2;
    float (*conv_grad_batch)[RESNET18_MAX_ACTIVATION_ELEMENTS] = workspace.batch_tmp3;
#pragma HLS INLINE off

    for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
        zero_buffer(grad_input_batch[sample], input_length);
        for (int i = 0; i < output_length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=8192 max=65536 avg=32768
#pragma HLS PIPELINE II=1
            relu_grad_batch[sample][i] =
                (block_output_batch[sample][i] > 0.0f) ? grad_output_batch[sample][i] : 0.0f;
        }
    }

    for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
        conv3x3_forward(
            conv1_out_batch[sample],
            out_channels,
            out_height,
            out_width,
            conv2_weight,
            conv2_bias,
            out_channels,
            1,
            recomputed_batch[sample]
        );
    }
    batchnorm_backward_from_input(
        recomputed_batch,
        relu_grad_batch,
        out_channels,
        out_height,
        out_width,
        conv2_stats,
        conv2_bn_gamma,
        bn_grad_batch
        ,
        grad_conv2_bn_gamma,
        grad_conv2_bn_beta
    );
    for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
        conv3x3_backward(
            conv1_out_batch[sample],
            out_channels,
            out_height,
            out_width,
            bn_grad_batch[sample],
            out_channels,
            1,
            conv2_weight,
            conv_grad_batch[sample],
            grad_conv2_weight,
            grad_conv2_bias
        );
        apply_relu_gradient(conv1_out_batch[sample], conv_grad_batch[sample], output_length);
    }

    for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
        conv3x3_forward(
            input_batch[sample],
            in_channels,
            in_height,
            in_width,
            conv1_weight,
            conv1_bias,
            out_channels,
            stride,
            recomputed_batch[sample]
        );
    }
    batchnorm_backward_from_input(
        recomputed_batch,
        conv_grad_batch,
        out_channels,
        out_height,
        out_width,
        conv1_stats,
        conv1_bn_gamma,
        bn_grad_batch
        ,
        grad_conv1_bn_gamma,
        grad_conv1_bn_beta
    );
    for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
        conv3x3_backward(
            input_batch[sample],
            in_channels,
            in_height,
            in_width,
            bn_grad_batch[sample],
            out_channels,
            stride,
            conv1_weight,
            grad_input_batch[sample],
            grad_conv1_weight,
            grad_conv1_bias
        );
    }

    if (use_projection) {
        for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
            conv1x1_forward(
                input_batch[sample],
                in_channels,
                in_height,
                in_width,
                proj_weight,
                proj_bias,
                out_channels,
                stride,
                recomputed_batch[sample]
            );
        }
        batchnorm_backward_from_input(
            recomputed_batch,
            relu_grad_batch,
            out_channels,
            out_height,
            out_width,
            *proj_stats,
            proj_bn_gamma,
            bn_grad_batch,
            grad_proj_bn_gamma,
            grad_proj_bn_beta
        );
        for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
            conv1x1_backward(
                input_batch[sample],
                in_channels,
                in_height,
                in_width,
                bn_grad_batch[sample],
                out_channels,
                stride,
                proj_weight,
                conv_grad_batch[sample],
                grad_proj_weight,
                grad_proj_bias
            );
            for (int i = 0; i < input_length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=8192 max=65536 avg=32768
#pragma HLS PIPELINE II=1
                grad_input_batch[sample][i] += conv_grad_batch[sample][i];
            }
        }
    } else {
        for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
            for (int i = 0; i < input_length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=8192 max=65536 avg=32768
#pragma HLS PIPELINE II=1
                grad_input_batch[sample][i] += relu_grad_batch[sample][i];
            }
        }
    }
}

void initialize_conv_array(float weights[], int length, float base_scale, int seed) {
#pragma HLS INLINE off
    for (int i = 0; i < length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2359296 avg=65536
#pragma HLS PIPELINE II=1
        int pattern = ((i * 17) + seed * 13) % 23;
        weights[i] = base_scale * (float)(pattern - 11);
    }
}

void initialize_bias_array(float bias[], int length, float scale, int seed) {
#pragma HLS INLINE off
    for (int i = 0; i < length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512 avg=128
#pragma HLS PIPELINE II=1
        int pattern = ((i * 5) + seed * 3) % 7;
        bias[i] = scale * (float)(pattern - 3);
    }
}

void initialize_batchnorm_affine(float gamma[], float beta[], int length) {
#pragma HLS INLINE off
    for (int i = 0; i < length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=512 avg=256
#pragma HLS PIPELINE II=1
        gamma[i] = 1.0f;
        beta[i] = 0.0f;
    }
}

void initialize_projection_identity(float weights[], float bias[], int in_channels, int out_channels) {
#pragma HLS INLINE off
    for (int oc = 0; oc < out_channels; ++oc) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=512 avg=256
        bias[oc] = 0.0f;
        for (int ic = 0; ic < in_channels; ++ic) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=256 avg=128
#pragma HLS PIPELINE II=1
            weights[proj_index(oc, ic, in_channels)] = (oc == ic) ? 1.0f : 0.0f;
        }
    }
}

void initialize_block(
    float conv1_weight[],
    int conv1_length,
    float conv1_bias[],
    int out_channels,
    float conv2_weight[],
    int conv2_length,
    float conv2_bias[],
    float scale,
    int seed
) {
#pragma HLS INLINE off
    initialize_conv_array(conv1_weight, conv1_length, scale, seed);
    initialize_bias_array(conv1_bias, out_channels, 0.002f, seed);
    initialize_conv_array(conv2_weight, conv2_length, scale * 0.8f, seed + 7);
    initialize_bias_array(conv2_bias, out_channels, 0.002f, seed + 7);
}

void argmax_logits(const float logits[RESNET18_NUM_CLASSES], int &predicted_label) {
#pragma HLS INLINE
    predicted_label = 0;
    float best = logits[0];
    for (int cls = 1; cls < RESNET18_NUM_CLASSES; ++cls) {
#pragma HLS LOOP_TRIPCOUNT min=9 max=9 avg=9
#pragma HLS PIPELINE II=1
        if (logits[cls] > best) {
            best = logits[cls];
            predicted_label = cls;
        }
    }
}

void accumulate_gradient_array(float accum[], const float grad[], int length) {
#pragma HLS INLINE
    for (int i = 0; i < length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2359296 avg=65536
#pragma HLS PIPELINE II=1
        accum[i] += grad[i];
    }
}

void scale_gradient_array(float grad[], int length, float scale) {
#pragma HLS INLINE
    for (int i = 0; i < length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2359296 avg=65536
#pragma HLS PIPELINE II=1
        grad[i] *= scale;
    }
}

void clip_gradient_array(float grad[], int length) {
#pragma HLS INLINE
    // Cap unusually large gradients before the optimizer step to keep training numerically stable.
    for (int i = 0; i < length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2359296 avg=65536
#pragma HLS PIPELINE II=1
        grad[i] = clamp_gradient(grad[i]);
    }
}

}  // namespace

void init_resnet18_params(ResNet18Params &params) {
#pragma HLS INLINE off
    initialize_conv_array(
        params.stem_weight,
        RESNET18_STEM_CHANNELS * RESNET18_IMAGE_CHANNELS * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE,
        0.015f,
        1
    );
    initialize_bias_array(params.stem_bias, RESNET18_STEM_CHANNELS, 0.001f, 1);
    initialize_batchnorm_affine(params.stem_bn_gamma, params.stem_bn_beta, RESNET18_STEM_CHANNELS);

    initialize_block(
        params.layer1_block0_conv1_weight,
        RESNET18_STAGE1_CHANNELS * RESNET18_STAGE1_CHANNELS * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE,
        params.layer1_block0_conv1_bias,
        RESNET18_STAGE1_CHANNELS,
        params.layer1_block0_conv2_weight,
        RESNET18_STAGE1_CHANNELS * RESNET18_STAGE1_CHANNELS * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE,
        params.layer1_block0_conv2_bias,
        0.010f,
        3
    );
    initialize_batchnorm_affine(
        params.layer1_block0_conv1_bn_gamma,
        params.layer1_block0_conv1_bn_beta,
        RESNET18_STAGE1_CHANNELS
    );
    initialize_batchnorm_affine(
        params.layer1_block0_conv2_bn_gamma,
        params.layer1_block0_conv2_bn_beta,
        RESNET18_STAGE1_CHANNELS
    );
    initialize_block(
        params.layer1_block1_conv1_weight,
        RESNET18_STAGE1_CHANNELS * RESNET18_STAGE1_CHANNELS * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE,
        params.layer1_block1_conv1_bias,
        RESNET18_STAGE1_CHANNELS,
        params.layer1_block1_conv2_weight,
        RESNET18_STAGE1_CHANNELS * RESNET18_STAGE1_CHANNELS * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE,
        params.layer1_block1_conv2_bias,
        0.010f,
        5
    );
    initialize_batchnorm_affine(
        params.layer1_block1_conv1_bn_gamma,
        params.layer1_block1_conv1_bn_beta,
        RESNET18_STAGE1_CHANNELS
    );
    initialize_batchnorm_affine(
        params.layer1_block1_conv2_bn_gamma,
        params.layer1_block1_conv2_bn_beta,
        RESNET18_STAGE1_CHANNELS
    );
    initialize_block(
        params.layer2_block0_conv1_weight,
        RESNET18_STAGE2_CHANNELS * RESNET18_STAGE1_CHANNELS * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE,
        params.layer2_block0_conv1_bias,
        RESNET18_STAGE2_CHANNELS,
        params.layer2_block0_conv2_weight,
        RESNET18_STAGE2_CHANNELS * RESNET18_STAGE2_CHANNELS * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE,
        params.layer2_block0_conv2_bias,
        0.008f,
        7
    );
    initialize_batchnorm_affine(
        params.layer2_block0_conv1_bn_gamma,
        params.layer2_block0_conv1_bn_beta,
        RESNET18_STAGE2_CHANNELS
    );
    initialize_batchnorm_affine(
        params.layer2_block0_conv2_bn_gamma,
        params.layer2_block0_conv2_bn_beta,
        RESNET18_STAGE2_CHANNELS
    );
    initialize_projection_identity(
        params.layer2_block0_proj_weight,
        params.layer2_block0_proj_bias,
        RESNET18_STAGE1_CHANNELS,
        RESNET18_STAGE2_CHANNELS
    );
    initialize_batchnorm_affine(
        params.layer2_block0_proj_bn_gamma,
        params.layer2_block0_proj_bn_beta,
        RESNET18_STAGE2_CHANNELS
    );
    initialize_block(
        params.layer2_block1_conv1_weight,
        RESNET18_STAGE2_CHANNELS * RESNET18_STAGE2_CHANNELS * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE,
        params.layer2_block1_conv1_bias,
        RESNET18_STAGE2_CHANNELS,
        params.layer2_block1_conv2_weight,
        RESNET18_STAGE2_CHANNELS * RESNET18_STAGE2_CHANNELS * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE,
        params.layer2_block1_conv2_bias,
        0.008f,
        9
    );
    initialize_batchnorm_affine(
        params.layer2_block1_conv1_bn_gamma,
        params.layer2_block1_conv1_bn_beta,
        RESNET18_STAGE2_CHANNELS
    );
    initialize_batchnorm_affine(
        params.layer2_block1_conv2_bn_gamma,
        params.layer2_block1_conv2_bn_beta,
        RESNET18_STAGE2_CHANNELS
    );
    initialize_block(
        params.layer3_block0_conv1_weight,
        RESNET18_STAGE3_CHANNELS * RESNET18_STAGE2_CHANNELS * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE,
        params.layer3_block0_conv1_bias,
        RESNET18_STAGE3_CHANNELS,
        params.layer3_block0_conv2_weight,
        RESNET18_STAGE3_CHANNELS * RESNET18_STAGE3_CHANNELS * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE,
        params.layer3_block0_conv2_bias,
        0.006f,
        11
    );
    initialize_batchnorm_affine(
        params.layer3_block0_conv1_bn_gamma,
        params.layer3_block0_conv1_bn_beta,
        RESNET18_STAGE3_CHANNELS
    );
    initialize_batchnorm_affine(
        params.layer3_block0_conv2_bn_gamma,
        params.layer3_block0_conv2_bn_beta,
        RESNET18_STAGE3_CHANNELS
    );
    initialize_projection_identity(
        params.layer3_block0_proj_weight,
        params.layer3_block0_proj_bias,
        RESNET18_STAGE2_CHANNELS,
        RESNET18_STAGE3_CHANNELS
    );
    initialize_batchnorm_affine(
        params.layer3_block0_proj_bn_gamma,
        params.layer3_block0_proj_bn_beta,
        RESNET18_STAGE3_CHANNELS
    );
    initialize_block(
        params.layer3_block1_conv1_weight,
        RESNET18_STAGE3_CHANNELS * RESNET18_STAGE3_CHANNELS * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE,
        params.layer3_block1_conv1_bias,
        RESNET18_STAGE3_CHANNELS,
        params.layer3_block1_conv2_weight,
        RESNET18_STAGE3_CHANNELS * RESNET18_STAGE3_CHANNELS * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE,
        params.layer3_block1_conv2_bias,
        0.006f,
        13
    );
    initialize_batchnorm_affine(
        params.layer3_block1_conv1_bn_gamma,
        params.layer3_block1_conv1_bn_beta,
        RESNET18_STAGE3_CHANNELS
    );
    initialize_batchnorm_affine(
        params.layer3_block1_conv2_bn_gamma,
        params.layer3_block1_conv2_bn_beta,
        RESNET18_STAGE3_CHANNELS
    );
    initialize_block(
        params.layer4_block0_conv1_weight,
        RESNET18_STAGE4_CHANNELS * RESNET18_STAGE3_CHANNELS * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE,
        params.layer4_block0_conv1_bias,
        RESNET18_STAGE4_CHANNELS,
        params.layer4_block0_conv2_weight,
        RESNET18_STAGE4_CHANNELS * RESNET18_STAGE4_CHANNELS * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE,
        params.layer4_block0_conv2_bias,
        0.004f,
        15
    );
    initialize_batchnorm_affine(
        params.layer4_block0_conv1_bn_gamma,
        params.layer4_block0_conv1_bn_beta,
        RESNET18_STAGE4_CHANNELS
    );
    initialize_batchnorm_affine(
        params.layer4_block0_conv2_bn_gamma,
        params.layer4_block0_conv2_bn_beta,
        RESNET18_STAGE4_CHANNELS
    );
    initialize_projection_identity(
        params.layer4_block0_proj_weight,
        params.layer4_block0_proj_bias,
        RESNET18_STAGE3_CHANNELS,
        RESNET18_STAGE4_CHANNELS
    );
    initialize_batchnorm_affine(
        params.layer4_block0_proj_bn_gamma,
        params.layer4_block0_proj_bn_beta,
        RESNET18_STAGE4_CHANNELS
    );
    initialize_block(
        params.layer4_block1_conv1_weight,
        RESNET18_STAGE4_CHANNELS * RESNET18_STAGE4_CHANNELS * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE,
        params.layer4_block1_conv1_bias,
        RESNET18_STAGE4_CHANNELS,
        params.layer4_block1_conv2_weight,
        RESNET18_STAGE4_CHANNELS * RESNET18_STAGE4_CHANNELS * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE,
        params.layer4_block1_conv2_bias,
        0.004f,
        17
    );
    initialize_batchnorm_affine(
        params.layer4_block1_conv1_bn_gamma,
        params.layer4_block1_conv1_bn_beta,
        RESNET18_STAGE4_CHANNELS
    );
    initialize_batchnorm_affine(
        params.layer4_block1_conv2_bn_gamma,
        params.layer4_block1_conv2_bn_beta,
        RESNET18_STAGE4_CHANNELS
    );

    for (int cls = 0; cls < RESNET18_NUM_CLASSES; ++cls) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
        params.fc_bias[cls] = 0.0f;
        for (int channel = 0; channel < RESNET18_STAGE4_CHANNELS; ++channel) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512 avg=512
#pragma HLS PIPELINE II=1
            int pattern = ((cls * 19) + (channel * 7)) % 29;
            params.fc_weight[fc_index(cls, channel)] = 0.003f * (float)(pattern - 14);
        }
    }
}

void reset_resnet18_state(ResNet18State &state) {
#pragma HLS INLINE off
#define ZERO_STATE_ARRAY(name) zero_buffer(state.name, (int)(sizeof(state.name) / sizeof(float)));
    RESNET18_STATE_ARRAYS(ZERO_STATE_ARRAY)
#undef ZERO_STATE_ARRAY
}

void zero_resnet18_gradients(ResNet18Gradients &gradients) {
#pragma HLS INLINE off
#define ZERO_GRAD_ARRAY(name) zero_buffer(gradients.name, (int)(sizeof(gradients.name) / sizeof(float)));
    RESNET18_GRAD_ARRAYS(ZERO_GRAD_ARRAY)
#undef ZERO_GRAD_ARRAY
}

void resnet18_forward(
    const float image[RESNET18_INPUT_SIZE],
    const ResNet18Params &params,
    ResNet18Activations &activations
) {
#pragma HLS INLINE off
    conv3x3_forward(
        image,
        RESNET18_IMAGE_CHANNELS,
        RESNET18_IMAGE_HEIGHT,
        RESNET18_IMAGE_WIDTH,
        params.stem_weight,
        params.stem_bias,
        RESNET18_STEM_CHANNELS,
        1,
        activations.stem_out
    );
    relu_inplace(activations.stem_out, RESNET18_STAGE1_CHANNELS * RESNET18_STAGE1_HEIGHT * RESNET18_STAGE1_WIDTH);

    basic_block_forward(
        activations.stem_out,
        RESNET18_STAGE1_CHANNELS,
        RESNET18_STAGE1_HEIGHT,
        RESNET18_STAGE1_WIDTH,
        RESNET18_STAGE1_CHANNELS,
        1,
        false,
        params.layer1_block0_conv1_weight,
        params.layer1_block0_conv1_bias,
        params.layer1_block0_conv2_weight,
        params.layer1_block0_conv2_bias,
        0,
        0,
        activations.layer1_block0_conv1_out,
        activations.layer1_block0_out
    );
    basic_block_forward(
        activations.layer1_block0_out,
        RESNET18_STAGE1_CHANNELS,
        RESNET18_STAGE1_HEIGHT,
        RESNET18_STAGE1_WIDTH,
        RESNET18_STAGE1_CHANNELS,
        1,
        false,
        params.layer1_block1_conv1_weight,
        params.layer1_block1_conv1_bias,
        params.layer1_block1_conv2_weight,
        params.layer1_block1_conv2_bias,
        0,
        0,
        activations.layer1_block1_conv1_out,
        activations.layer1_block1_out
    );
    basic_block_forward(
        activations.layer1_block1_out,
        RESNET18_STAGE1_CHANNELS,
        RESNET18_STAGE1_HEIGHT,
        RESNET18_STAGE1_WIDTH,
        RESNET18_STAGE2_CHANNELS,
        2,
        true,
        params.layer2_block0_conv1_weight,
        params.layer2_block0_conv1_bias,
        params.layer2_block0_conv2_weight,
        params.layer2_block0_conv2_bias,
        params.layer2_block0_proj_weight,
        params.layer2_block0_proj_bias,
        activations.layer2_block0_conv1_out,
        activations.layer2_block0_out
    );
    basic_block_forward(
        activations.layer2_block0_out,
        RESNET18_STAGE2_CHANNELS,
        RESNET18_STAGE2_HEIGHT,
        RESNET18_STAGE2_WIDTH,
        RESNET18_STAGE2_CHANNELS,
        1,
        false,
        params.layer2_block1_conv1_weight,
        params.layer2_block1_conv1_bias,
        params.layer2_block1_conv2_weight,
        params.layer2_block1_conv2_bias,
        0,
        0,
        activations.layer2_block1_conv1_out,
        activations.layer2_block1_out
    );
    basic_block_forward(
        activations.layer2_block1_out,
        RESNET18_STAGE2_CHANNELS,
        RESNET18_STAGE2_HEIGHT,
        RESNET18_STAGE2_WIDTH,
        RESNET18_STAGE3_CHANNELS,
        2,
        true,
        params.layer3_block0_conv1_weight,
        params.layer3_block0_conv1_bias,
        params.layer3_block0_conv2_weight,
        params.layer3_block0_conv2_bias,
        params.layer3_block0_proj_weight,
        params.layer3_block0_proj_bias,
        activations.layer3_block0_conv1_out,
        activations.layer3_block0_out
    );
    basic_block_forward(
        activations.layer3_block0_out,
        RESNET18_STAGE3_CHANNELS,
        RESNET18_STAGE3_HEIGHT,
        RESNET18_STAGE3_WIDTH,
        RESNET18_STAGE3_CHANNELS,
        1,
        false,
        params.layer3_block1_conv1_weight,
        params.layer3_block1_conv1_bias,
        params.layer3_block1_conv2_weight,
        params.layer3_block1_conv2_bias,
        0,
        0,
        activations.layer3_block1_conv1_out,
        activations.layer3_block1_out
    );
    basic_block_forward(
        activations.layer3_block1_out,
        RESNET18_STAGE3_CHANNELS,
        RESNET18_STAGE3_HEIGHT,
        RESNET18_STAGE3_WIDTH,
        RESNET18_STAGE4_CHANNELS,
        2,
        true,
        params.layer4_block0_conv1_weight,
        params.layer4_block0_conv1_bias,
        params.layer4_block0_conv2_weight,
        params.layer4_block0_conv2_bias,
        params.layer4_block0_proj_weight,
        params.layer4_block0_proj_bias,
        activations.layer4_block0_conv1_out,
        activations.layer4_block0_out
    );
    basic_block_forward(
        activations.layer4_block0_out,
        RESNET18_STAGE4_CHANNELS,
        RESNET18_STAGE4_HEIGHT,
        RESNET18_STAGE4_WIDTH,
        RESNET18_STAGE4_CHANNELS,
        1,
        false,
        params.layer4_block1_conv1_weight,
        params.layer4_block1_conv1_bias,
        params.layer4_block1_conv2_weight,
        params.layer4_block1_conv2_bias,
        0,
        0,
        activations.layer4_block1_conv1_out,
        activations.layer4_block1_out
    );

    for (int channel = 0; channel < RESNET18_STAGE4_CHANNELS; ++channel) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512 avg=512
        float sum = 0.0f;
        for (int y = 0; y < RESNET18_STAGE4_HEIGHT; ++y) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
            for (int x = 0; x < RESNET18_STAGE4_WIDTH; ++x) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
#pragma HLS PIPELINE II=1
                sum += activations.layer4_block1_out[feature_index(channel, y, x, RESNET18_STAGE4_HEIGHT, RESNET18_STAGE4_WIDTH)];
            }
        }
        activations.pooled[channel] = sum / (float)(RESNET18_STAGE4_HEIGHT * RESNET18_STAGE4_WIDTH);
    }

    for (int cls = 0; cls < RESNET18_NUM_CLASSES; ++cls) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
        float acc = params.fc_bias[cls];
        for (int channel = 0; channel < RESNET18_STAGE4_CHANNELS; ++channel) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512 avg=512
#pragma HLS PIPELINE II=1
            acc += params.fc_weight[fc_index(cls, channel)] * activations.pooled[channel];
        }
        activations.logits[cls] = acc;
    }
}

float resnet18_softmax_cross_entropy(
    const float logits[RESNET18_NUM_CLASSES],
    int label,
    float grad_logits[RESNET18_NUM_CLASSES]
) {
#pragma HLS INLINE off
    float max_logit = logits[0];
    for (int cls = 1; cls < RESNET18_NUM_CLASSES; ++cls) {
#pragma HLS LOOP_TRIPCOUNT min=9 max=9 avg=9
#pragma HLS PIPELINE II=1
        if (logits[cls] > max_logit) {
            max_logit = logits[cls];
        }
    }

    float sum_exp = 0.0f;
    for (int cls = 0; cls < RESNET18_NUM_CLASSES; ++cls) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
#pragma HLS PIPELINE II=1
        scratch0[cls] = expf(logits[cls] - max_logit);
        sum_exp += scratch0[cls];
    }

    float label_prob = 1e-6f;
    for (int cls = 0; cls < RESNET18_NUM_CLASSES; ++cls) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
#pragma HLS PIPELINE II=1
        float probability = scratch0[cls] / sum_exp;
        grad_logits[cls] = probability;
        if (cls == label) {
            label_prob = (probability > 1e-6f) ? probability : 1e-6f;
            grad_logits[cls] -= 1.0f;
        }
    }

    return -logf(label_prob);
}

void resnet18_backward(
    const float image[RESNET18_INPUT_SIZE],
    int label,
    const ResNet18Params &params,
    const ResNet18Activations &activations,
    ResNet18Gradients &gradients
) {
#pragma HLS INLINE off
    zero_resnet18_gradients(gradients);

    float grad_logits[RESNET18_NUM_CLASSES];
    float grad_pooled[RESNET18_STAGE4_CHANNELS];
    const float pooled_scale = 1.0f / (float)(RESNET18_STAGE4_HEIGHT * RESNET18_STAGE4_WIDTH);

    zero_buffer(grad_pooled, RESNET18_STAGE4_CHANNELS);
    (void)resnet18_softmax_cross_entropy(activations.logits, label, grad_logits);

    for (int cls = 0; cls < RESNET18_NUM_CLASSES; ++cls) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
        gradients.fc_bias[cls] += grad_logits[cls];
        for (int channel = 0; channel < RESNET18_STAGE4_CHANNELS; ++channel) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512 avg=512
#pragma HLS PIPELINE II=1
            gradients.fc_weight[fc_index(cls, channel)] += grad_logits[cls] * activations.pooled[channel];
            grad_pooled[channel] += params.fc_weight[fc_index(cls, channel)] * grad_logits[cls];
        }
    }

    for (int channel = 0; channel < RESNET18_STAGE4_CHANNELS; ++channel) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512 avg=512
        float grad_value = grad_pooled[channel] * pooled_scale;
        for (int y = 0; y < RESNET18_STAGE4_HEIGHT; ++y) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
            for (int x = 0; x < RESNET18_STAGE4_WIDTH; ++x) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
#pragma HLS PIPELINE II=1
                scratch0[feature_index(channel, y, x, RESNET18_STAGE4_HEIGHT, RESNET18_STAGE4_WIDTH)] = grad_value;
            }
        }
    }

    basic_block_backward(
        activations.layer4_block0_out,
        activations.layer4_block1_conv1_out,
        activations.layer4_block1_out,
        scratch0,
        RESNET18_STAGE4_CHANNELS,
        RESNET18_STAGE4_HEIGHT,
        RESNET18_STAGE4_WIDTH,
        RESNET18_STAGE4_CHANNELS,
        1,
        false,
        params.layer4_block1_conv1_weight,
        params.layer4_block1_conv2_weight,
        0,
        scratch1,
        gradients.layer4_block1_conv1_weight,
        gradients.layer4_block1_conv1_bias,
        gradients.layer4_block1_conv2_weight,
        gradients.layer4_block1_conv2_bias,
        0,
        0
    );
    basic_block_backward(
        activations.layer3_block1_out,
        activations.layer4_block0_conv1_out,
        activations.layer4_block0_out,
        scratch1,
        RESNET18_STAGE3_CHANNELS,
        RESNET18_STAGE3_HEIGHT,
        RESNET18_STAGE3_WIDTH,
        RESNET18_STAGE4_CHANNELS,
        2,
        true,
        params.layer4_block0_conv1_weight,
        params.layer4_block0_conv2_weight,
        params.layer4_block0_proj_weight,
        scratch0,
        gradients.layer4_block0_conv1_weight,
        gradients.layer4_block0_conv1_bias,
        gradients.layer4_block0_conv2_weight,
        gradients.layer4_block0_conv2_bias,
        gradients.layer4_block0_proj_weight,
        gradients.layer4_block0_proj_bias
    );
    basic_block_backward(
        activations.layer3_block0_out,
        activations.layer3_block1_conv1_out,
        activations.layer3_block1_out,
        scratch0,
        RESNET18_STAGE3_CHANNELS,
        RESNET18_STAGE3_HEIGHT,
        RESNET18_STAGE3_WIDTH,
        RESNET18_STAGE3_CHANNELS,
        1,
        false,
        params.layer3_block1_conv1_weight,
        params.layer3_block1_conv2_weight,
        0,
        scratch1,
        gradients.layer3_block1_conv1_weight,
        gradients.layer3_block1_conv1_bias,
        gradients.layer3_block1_conv2_weight,
        gradients.layer3_block1_conv2_bias,
        0,
        0
    );
    basic_block_backward(
        activations.layer2_block1_out,
        activations.layer3_block0_conv1_out,
        activations.layer3_block0_out,
        scratch1,
        RESNET18_STAGE2_CHANNELS,
        RESNET18_STAGE2_HEIGHT,
        RESNET18_STAGE2_WIDTH,
        RESNET18_STAGE3_CHANNELS,
        2,
        true,
        params.layer3_block0_conv1_weight,
        params.layer3_block0_conv2_weight,
        params.layer3_block0_proj_weight,
        scratch0,
        gradients.layer3_block0_conv1_weight,
        gradients.layer3_block0_conv1_bias,
        gradients.layer3_block0_conv2_weight,
        gradients.layer3_block0_conv2_bias,
        gradients.layer3_block0_proj_weight,
        gradients.layer3_block0_proj_bias
    );
    basic_block_backward(
        activations.layer2_block0_out,
        activations.layer2_block1_conv1_out,
        activations.layer2_block1_out,
        scratch0,
        RESNET18_STAGE2_CHANNELS,
        RESNET18_STAGE2_HEIGHT,
        RESNET18_STAGE2_WIDTH,
        RESNET18_STAGE2_CHANNELS,
        1,
        false,
        params.layer2_block1_conv1_weight,
        params.layer2_block1_conv2_weight,
        0,
        scratch1,
        gradients.layer2_block1_conv1_weight,
        gradients.layer2_block1_conv1_bias,
        gradients.layer2_block1_conv2_weight,
        gradients.layer2_block1_conv2_bias,
        0,
        0
    );
    basic_block_backward(
        activations.layer1_block1_out,
        activations.layer2_block0_conv1_out,
        activations.layer2_block0_out,
        scratch1,
        RESNET18_STAGE1_CHANNELS,
        RESNET18_STAGE1_HEIGHT,
        RESNET18_STAGE1_WIDTH,
        RESNET18_STAGE2_CHANNELS,
        2,
        true,
        params.layer2_block0_conv1_weight,
        params.layer2_block0_conv2_weight,
        params.layer2_block0_proj_weight,
        scratch0,
        gradients.layer2_block0_conv1_weight,
        gradients.layer2_block0_conv1_bias,
        gradients.layer2_block0_conv2_weight,
        gradients.layer2_block0_conv2_bias,
        gradients.layer2_block0_proj_weight,
        gradients.layer2_block0_proj_bias
    );
    basic_block_backward(
        activations.layer1_block0_out,
        activations.layer1_block1_conv1_out,
        activations.layer1_block1_out,
        scratch0,
        RESNET18_STAGE1_CHANNELS,
        RESNET18_STAGE1_HEIGHT,
        RESNET18_STAGE1_WIDTH,
        RESNET18_STAGE1_CHANNELS,
        1,
        false,
        params.layer1_block1_conv1_weight,
        params.layer1_block1_conv2_weight,
        0,
        scratch1,
        gradients.layer1_block1_conv1_weight,
        gradients.layer1_block1_conv1_bias,
        gradients.layer1_block1_conv2_weight,
        gradients.layer1_block1_conv2_bias,
        0,
        0
    );
    basic_block_backward(
        activations.stem_out,
        activations.layer1_block0_conv1_out,
        activations.layer1_block0_out,
        scratch1,
        RESNET18_STAGE1_CHANNELS,
        RESNET18_STAGE1_HEIGHT,
        RESNET18_STAGE1_WIDTH,
        RESNET18_STAGE1_CHANNELS,
        1,
        false,
        params.layer1_block0_conv1_weight,
        params.layer1_block0_conv2_weight,
        0,
        scratch0,
        gradients.layer1_block0_conv1_weight,
        gradients.layer1_block0_conv1_bias,
        gradients.layer1_block0_conv2_weight,
        gradients.layer1_block0_conv2_bias,
        0,
        0
    );

    apply_relu_gradient(activations.stem_out, scratch0, RESNET18_STAGE1_CHANNELS * RESNET18_STAGE1_HEIGHT * RESNET18_STAGE1_WIDTH);
    conv3x3_backward(
        image,
        RESNET18_IMAGE_CHANNELS,
        RESNET18_IMAGE_HEIGHT,
        RESNET18_IMAGE_WIDTH,
        scratch0,
        RESNET18_STEM_CHANNELS,
        1,
        params.stem_weight,
        scratch1,
        gradients.stem_weight,
        gradients.stem_bias
    );
}

void resnet18_sgd_momentum_update(
    ResNet18Params &params,
    ResNet18State &state,
    const ResNet18Gradients &gradients,
    float learning_rate,
    float momentum
) {
#pragma HLS INLINE off
#define UPDATE_PARAM(param_name, state_name, grad_name) \
    for (int i = 0; i < (int)(sizeof(params.param_name) / sizeof(float)); ++i) { \
        _Pragma("HLS LOOP_TRIPCOUNT min=1 max=2359296 avg=65536") \
        _Pragma("HLS PIPELINE II=1") \
        float clipped_grad = clamp_gradient(gradients.grad_name[i]); \
        state.state_name[i] = momentum * state.state_name[i] - learning_rate * clipped_grad; \
        params.param_name[i] += state.state_name[i]; \
    }

    UPDATE_PARAM(stem_weight, velocity_stem_weight, stem_weight);
    UPDATE_PARAM(stem_bias, velocity_stem_bias, stem_bias);
    UPDATE_PARAM(stem_bn_gamma, velocity_stem_bn_gamma, stem_bn_gamma);
    UPDATE_PARAM(stem_bn_beta, velocity_stem_bn_beta, stem_bn_beta);
    UPDATE_PARAM(layer1_block0_conv1_weight, velocity_layer1_block0_conv1_weight, layer1_block0_conv1_weight);
    UPDATE_PARAM(layer1_block0_conv1_bias, velocity_layer1_block0_conv1_bias, layer1_block0_conv1_bias);
    UPDATE_PARAM(layer1_block0_conv1_bn_gamma, velocity_layer1_block0_conv1_bn_gamma, layer1_block0_conv1_bn_gamma);
    UPDATE_PARAM(layer1_block0_conv1_bn_beta, velocity_layer1_block0_conv1_bn_beta, layer1_block0_conv1_bn_beta);
    UPDATE_PARAM(layer1_block0_conv2_weight, velocity_layer1_block0_conv2_weight, layer1_block0_conv2_weight);
    UPDATE_PARAM(layer1_block0_conv2_bias, velocity_layer1_block0_conv2_bias, layer1_block0_conv2_bias);
    UPDATE_PARAM(layer1_block0_conv2_bn_gamma, velocity_layer1_block0_conv2_bn_gamma, layer1_block0_conv2_bn_gamma);
    UPDATE_PARAM(layer1_block0_conv2_bn_beta, velocity_layer1_block0_conv2_bn_beta, layer1_block0_conv2_bn_beta);
    UPDATE_PARAM(layer1_block1_conv1_weight, velocity_layer1_block1_conv1_weight, layer1_block1_conv1_weight);
    UPDATE_PARAM(layer1_block1_conv1_bias, velocity_layer1_block1_conv1_bias, layer1_block1_conv1_bias);
    UPDATE_PARAM(layer1_block1_conv1_bn_gamma, velocity_layer1_block1_conv1_bn_gamma, layer1_block1_conv1_bn_gamma);
    UPDATE_PARAM(layer1_block1_conv1_bn_beta, velocity_layer1_block1_conv1_bn_beta, layer1_block1_conv1_bn_beta);
    UPDATE_PARAM(layer1_block1_conv2_weight, velocity_layer1_block1_conv2_weight, layer1_block1_conv2_weight);
    UPDATE_PARAM(layer1_block1_conv2_bias, velocity_layer1_block1_conv2_bias, layer1_block1_conv2_bias);
    UPDATE_PARAM(layer1_block1_conv2_bn_gamma, velocity_layer1_block1_conv2_bn_gamma, layer1_block1_conv2_bn_gamma);
    UPDATE_PARAM(layer1_block1_conv2_bn_beta, velocity_layer1_block1_conv2_bn_beta, layer1_block1_conv2_bn_beta);
    UPDATE_PARAM(layer2_block0_conv1_weight, velocity_layer2_block0_conv1_weight, layer2_block0_conv1_weight);
    UPDATE_PARAM(layer2_block0_conv1_bias, velocity_layer2_block0_conv1_bias, layer2_block0_conv1_bias);
    UPDATE_PARAM(layer2_block0_conv1_bn_gamma, velocity_layer2_block0_conv1_bn_gamma, layer2_block0_conv1_bn_gamma);
    UPDATE_PARAM(layer2_block0_conv1_bn_beta, velocity_layer2_block0_conv1_bn_beta, layer2_block0_conv1_bn_beta);
    UPDATE_PARAM(layer2_block0_conv2_weight, velocity_layer2_block0_conv2_weight, layer2_block0_conv2_weight);
    UPDATE_PARAM(layer2_block0_conv2_bias, velocity_layer2_block0_conv2_bias, layer2_block0_conv2_bias);
    UPDATE_PARAM(layer2_block0_conv2_bn_gamma, velocity_layer2_block0_conv2_bn_gamma, layer2_block0_conv2_bn_gamma);
    UPDATE_PARAM(layer2_block0_conv2_bn_beta, velocity_layer2_block0_conv2_bn_beta, layer2_block0_conv2_bn_beta);
    UPDATE_PARAM(layer2_block0_proj_weight, velocity_layer2_block0_proj_weight, layer2_block0_proj_weight);
    UPDATE_PARAM(layer2_block0_proj_bias, velocity_layer2_block0_proj_bias, layer2_block0_proj_bias);
    UPDATE_PARAM(layer2_block0_proj_bn_gamma, velocity_layer2_block0_proj_bn_gamma, layer2_block0_proj_bn_gamma);
    UPDATE_PARAM(layer2_block0_proj_bn_beta, velocity_layer2_block0_proj_bn_beta, layer2_block0_proj_bn_beta);
    UPDATE_PARAM(layer2_block1_conv1_weight, velocity_layer2_block1_conv1_weight, layer2_block1_conv1_weight);
    UPDATE_PARAM(layer2_block1_conv1_bias, velocity_layer2_block1_conv1_bias, layer2_block1_conv1_bias);
    UPDATE_PARAM(layer2_block1_conv1_bn_gamma, velocity_layer2_block1_conv1_bn_gamma, layer2_block1_conv1_bn_gamma);
    UPDATE_PARAM(layer2_block1_conv1_bn_beta, velocity_layer2_block1_conv1_bn_beta, layer2_block1_conv1_bn_beta);
    UPDATE_PARAM(layer2_block1_conv2_weight, velocity_layer2_block1_conv2_weight, layer2_block1_conv2_weight);
    UPDATE_PARAM(layer2_block1_conv2_bias, velocity_layer2_block1_conv2_bias, layer2_block1_conv2_bias);
    UPDATE_PARAM(layer2_block1_conv2_bn_gamma, velocity_layer2_block1_conv2_bn_gamma, layer2_block1_conv2_bn_gamma);
    UPDATE_PARAM(layer2_block1_conv2_bn_beta, velocity_layer2_block1_conv2_bn_beta, layer2_block1_conv2_bn_beta);
    UPDATE_PARAM(layer3_block0_conv1_weight, velocity_layer3_block0_conv1_weight, layer3_block0_conv1_weight);
    UPDATE_PARAM(layer3_block0_conv1_bias, velocity_layer3_block0_conv1_bias, layer3_block0_conv1_bias);
    UPDATE_PARAM(layer3_block0_conv1_bn_gamma, velocity_layer3_block0_conv1_bn_gamma, layer3_block0_conv1_bn_gamma);
    UPDATE_PARAM(layer3_block0_conv1_bn_beta, velocity_layer3_block0_conv1_bn_beta, layer3_block0_conv1_bn_beta);
    UPDATE_PARAM(layer3_block0_conv2_weight, velocity_layer3_block0_conv2_weight, layer3_block0_conv2_weight);
    UPDATE_PARAM(layer3_block0_conv2_bias, velocity_layer3_block0_conv2_bias, layer3_block0_conv2_bias);
    UPDATE_PARAM(layer3_block0_conv2_bn_gamma, velocity_layer3_block0_conv2_bn_gamma, layer3_block0_conv2_bn_gamma);
    UPDATE_PARAM(layer3_block0_conv2_bn_beta, velocity_layer3_block0_conv2_bn_beta, layer3_block0_conv2_bn_beta);
    UPDATE_PARAM(layer3_block0_proj_weight, velocity_layer3_block0_proj_weight, layer3_block0_proj_weight);
    UPDATE_PARAM(layer3_block0_proj_bias, velocity_layer3_block0_proj_bias, layer3_block0_proj_bias);
    UPDATE_PARAM(layer3_block0_proj_bn_gamma, velocity_layer3_block0_proj_bn_gamma, layer3_block0_proj_bn_gamma);
    UPDATE_PARAM(layer3_block0_proj_bn_beta, velocity_layer3_block0_proj_bn_beta, layer3_block0_proj_bn_beta);
    UPDATE_PARAM(layer3_block1_conv1_weight, velocity_layer3_block1_conv1_weight, layer3_block1_conv1_weight);
    UPDATE_PARAM(layer3_block1_conv1_bias, velocity_layer3_block1_conv1_bias, layer3_block1_conv1_bias);
    UPDATE_PARAM(layer3_block1_conv1_bn_gamma, velocity_layer3_block1_conv1_bn_gamma, layer3_block1_conv1_bn_gamma);
    UPDATE_PARAM(layer3_block1_conv1_bn_beta, velocity_layer3_block1_conv1_bn_beta, layer3_block1_conv1_bn_beta);
    UPDATE_PARAM(layer3_block1_conv2_weight, velocity_layer3_block1_conv2_weight, layer3_block1_conv2_weight);
    UPDATE_PARAM(layer3_block1_conv2_bias, velocity_layer3_block1_conv2_bias, layer3_block1_conv2_bias);
    UPDATE_PARAM(layer3_block1_conv2_bn_gamma, velocity_layer3_block1_conv2_bn_gamma, layer3_block1_conv2_bn_gamma);
    UPDATE_PARAM(layer3_block1_conv2_bn_beta, velocity_layer3_block1_conv2_bn_beta, layer3_block1_conv2_bn_beta);
    UPDATE_PARAM(layer4_block0_conv1_weight, velocity_layer4_block0_conv1_weight, layer4_block0_conv1_weight);
    UPDATE_PARAM(layer4_block0_conv1_bias, velocity_layer4_block0_conv1_bias, layer4_block0_conv1_bias);
    UPDATE_PARAM(layer4_block0_conv1_bn_gamma, velocity_layer4_block0_conv1_bn_gamma, layer4_block0_conv1_bn_gamma);
    UPDATE_PARAM(layer4_block0_conv1_bn_beta, velocity_layer4_block0_conv1_bn_beta, layer4_block0_conv1_bn_beta);
    UPDATE_PARAM(layer4_block0_conv2_weight, velocity_layer4_block0_conv2_weight, layer4_block0_conv2_weight);
    UPDATE_PARAM(layer4_block0_conv2_bias, velocity_layer4_block0_conv2_bias, layer4_block0_conv2_bias);
    UPDATE_PARAM(layer4_block0_conv2_bn_gamma, velocity_layer4_block0_conv2_bn_gamma, layer4_block0_conv2_bn_gamma);
    UPDATE_PARAM(layer4_block0_conv2_bn_beta, velocity_layer4_block0_conv2_bn_beta, layer4_block0_conv2_bn_beta);
    UPDATE_PARAM(layer4_block0_proj_weight, velocity_layer4_block0_proj_weight, layer4_block0_proj_weight);
    UPDATE_PARAM(layer4_block0_proj_bias, velocity_layer4_block0_proj_bias, layer4_block0_proj_bias);
    UPDATE_PARAM(layer4_block0_proj_bn_gamma, velocity_layer4_block0_proj_bn_gamma, layer4_block0_proj_bn_gamma);
    UPDATE_PARAM(layer4_block0_proj_bn_beta, velocity_layer4_block0_proj_bn_beta, layer4_block0_proj_bn_beta);
    UPDATE_PARAM(layer4_block1_conv1_weight, velocity_layer4_block1_conv1_weight, layer4_block1_conv1_weight);
    UPDATE_PARAM(layer4_block1_conv1_bias, velocity_layer4_block1_conv1_bias, layer4_block1_conv1_bias);
    UPDATE_PARAM(layer4_block1_conv1_bn_gamma, velocity_layer4_block1_conv1_bn_gamma, layer4_block1_conv1_bn_gamma);
    UPDATE_PARAM(layer4_block1_conv1_bn_beta, velocity_layer4_block1_conv1_bn_beta, layer4_block1_conv1_bn_beta);
    UPDATE_PARAM(layer4_block1_conv2_weight, velocity_layer4_block1_conv2_weight, layer4_block1_conv2_weight);
    UPDATE_PARAM(layer4_block1_conv2_bias, velocity_layer4_block1_conv2_bias, layer4_block1_conv2_bias);
    UPDATE_PARAM(layer4_block1_conv2_bn_gamma, velocity_layer4_block1_conv2_bn_gamma, layer4_block1_conv2_bn_gamma);
    UPDATE_PARAM(layer4_block1_conv2_bn_beta, velocity_layer4_block1_conv2_bn_beta, layer4_block1_conv2_bn_beta);
    UPDATE_PARAM(fc_weight, velocity_fc_weight, fc_weight);
    UPDATE_PARAM(fc_bias, velocity_fc_bias, fc_bias);

#undef UPDATE_PARAM
}

void resnet18_train_step(
    const float image[RESNET18_INPUT_SIZE],
    int label,
    float learning_rate,
    float momentum,
    ResNet18Params &params,
    ResNet18State &state,
    float logits[RESNET18_NUM_CLASSES],
    ResNet18StepResult &result
) {
#pragma HLS INLINE off
    static ResNet18Activations activations;
    static ResNet18Gradients gradients;
    float grad_logits[RESNET18_NUM_CLASSES];

    resnet18_forward(image, params, activations);
    result.loss = resnet18_softmax_cross_entropy(activations.logits, label, grad_logits);
    argmax_logits(activations.logits, result.predicted_label);

    for (int cls = 0; cls < RESNET18_NUM_CLASSES; ++cls) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
#pragma HLS PIPELINE II=1
        logits[cls] = activations.logits[cls];
    }

    resnet18_backward(image, label, params, activations, gradients);
    resnet18_sgd_momentum_update(params, state, gradients, learning_rate, momentum);
}

void resnet18_train_batch_extmem(
    const float images[RESNET18_BATCH_SIZE][RESNET18_INPUT_SIZE],
    const int labels[RESNET18_BATCH_SIZE],
    float learning_rate,
    float momentum,
    ResNet18Params &params,
    ResNet18State &state,
    ResNet18BatchTrainingWorkspace &workspace,
    float logits[RESNET18_BATCH_SIZE][RESNET18_NUM_CLASSES],
    ResNet18StepResult batch_results[RESNET18_BATCH_SIZE],
    float &average_loss
) {
#pragma HLS INLINE off
    ResNet18Activations *activations = workspace.activations;
    ResNet18Gradients &batch_gradients = workspace.batch_gradients;
    ResNet18BatchNormCache &batchnorm_cache = workspace.batchnorm_cache;
    float (*batch_scratch0)[RESNET18_MAX_ACTIVATION_ELEMENTS] = workspace.batch_scratch0;
    float (*batch_scratch1)[RESNET18_MAX_ACTIVATION_ELEMENTS] = workspace.batch_scratch1;
    float (*batch_scratch2)[RESNET18_MAX_ACTIVATION_ELEMENTS] = workspace.batch_scratch2;
    float (*batch_scratch3)[RESNET18_MAX_ACTIVATION_ELEMENTS] = workspace.batch_scratch3;
    float (*batch_grad0)[RESNET18_MAX_ACTIVATION_ELEMENTS] = workspace.batch_grad0;
    float (*batch_grad1)[RESNET18_MAX_ACTIVATION_ELEMENTS] = workspace.batch_grad1;
    float (*batch_tmp0)[RESNET18_MAX_ACTIVATION_ELEMENTS] = workspace.batch_tmp0;
    float grad_logits[RESNET18_NUM_CLASSES];
    const int stem_length = RESNET18_STEM_CHANNELS * RESNET18_STAGE1_HEIGHT * RESNET18_STAGE1_WIDTH;
    const int stage1_length = RESNET18_STAGE1_CHANNELS * RESNET18_STAGE1_HEIGHT * RESNET18_STAGE1_WIDTH;
    const int stage2_length = RESNET18_STAGE2_CHANNELS * RESNET18_STAGE2_HEIGHT * RESNET18_STAGE2_WIDTH;
    const int stage3_length = RESNET18_STAGE3_CHANNELS * RESNET18_STAGE3_HEIGHT * RESNET18_STAGE3_WIDTH;
    const int stage4_length = RESNET18_STAGE4_CHANNELS * RESNET18_STAGE4_HEIGHT * RESNET18_STAGE4_WIDTH;

    zero_resnet18_gradients(batch_gradients);
    average_loss = 0.0f;

#define COPY_FIELD_TO_BATCH(dst, field, length) \
    for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) { \
        _Pragma("HLS LOOP_TRIPCOUNT min=10 max=10 avg=10") \
        copy_buffer(activations[sample].field, dst[sample], length); \
    }

#define COPY_BATCH_TO_FIELD(field, src, length) \
    for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) { \
        _Pragma("HLS LOOP_TRIPCOUNT min=10 max=10 avg=10") \
        copy_buffer(src[sample], activations[sample].field, length); \
    }

    for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
        conv3x3_forward(
            images[sample],
            RESNET18_IMAGE_CHANNELS,
            RESNET18_IMAGE_HEIGHT,
            RESNET18_IMAGE_WIDTH,
            params.stem_weight,
            params.stem_bias,
            RESNET18_STEM_CHANNELS,
            1,
            batch_scratch0[sample]
        );
    }

    compute_batchnorm_stats(
        batch_scratch0,
        RESNET18_STEM_CHANNELS,
        RESNET18_STAGE1_HEIGHT,
        RESNET18_STAGE1_WIDTH,
        batchnorm_cache.stem
    );
    apply_batchnorm_inplace(
        batch_scratch0,
        RESNET18_STEM_CHANNELS,
        RESNET18_STAGE1_HEIGHT,
        RESNET18_STAGE1_WIDTH,
        batchnorm_cache.stem,
        params.stem_bn_gamma,
        params.stem_bn_beta
    );
    for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
        relu_inplace(batch_scratch0[sample], stem_length);
        copy_buffer(batch_scratch0[sample], activations[sample].stem_out, stem_length);
    }

    COPY_FIELD_TO_BATCH(batch_scratch0, stem_out, stage1_length);
    basic_block_forward_batchnorm(
        batch_scratch0,
        RESNET18_STAGE1_CHANNELS,
        RESNET18_STAGE1_HEIGHT,
        RESNET18_STAGE1_WIDTH,
        RESNET18_STAGE1_CHANNELS,
        1,
        false,
        params.layer1_block0_conv1_weight,
        params.layer1_block0_conv1_bias,
        params.layer1_block0_conv1_bn_gamma,
        params.layer1_block0_conv1_bn_beta,
        params.layer1_block0_conv2_weight,
        params.layer1_block0_conv2_bias,
        params.layer1_block0_conv2_bn_gamma,
        params.layer1_block0_conv2_bn_beta,
        0,
        0,
        0,
        0,
        batchnorm_cache.layer1_block0_conv1,
        batchnorm_cache.layer1_block0_conv2,
        0,
        batch_scratch1,
        batch_scratch2,
        workspace
    );
    COPY_BATCH_TO_FIELD(layer1_block0_conv1_out, batch_scratch1, stage1_length);
    COPY_BATCH_TO_FIELD(layer1_block0_out, batch_scratch2, stage1_length);

    COPY_FIELD_TO_BATCH(batch_scratch0, layer1_block0_out, stage1_length);
    basic_block_forward_batchnorm(
        batch_scratch0,
        RESNET18_STAGE1_CHANNELS,
        RESNET18_STAGE1_HEIGHT,
        RESNET18_STAGE1_WIDTH,
        RESNET18_STAGE1_CHANNELS,
        1,
        false,
        params.layer1_block1_conv1_weight,
        params.layer1_block1_conv1_bias,
        params.layer1_block1_conv1_bn_gamma,
        params.layer1_block1_conv1_bn_beta,
        params.layer1_block1_conv2_weight,
        params.layer1_block1_conv2_bias,
        params.layer1_block1_conv2_bn_gamma,
        params.layer1_block1_conv2_bn_beta,
        0,
        0,
        0,
        0,
        batchnorm_cache.layer1_block1_conv1,
        batchnorm_cache.layer1_block1_conv2,
        0,
        batch_scratch1,
        batch_scratch2,
        workspace
    );
    COPY_BATCH_TO_FIELD(layer1_block1_conv1_out, batch_scratch1, stage1_length);
    COPY_BATCH_TO_FIELD(layer1_block1_out, batch_scratch2, stage1_length);

    COPY_FIELD_TO_BATCH(batch_scratch0, layer1_block1_out, stage1_length);
    basic_block_forward_batchnorm(
        batch_scratch0,
        RESNET18_STAGE1_CHANNELS,
        RESNET18_STAGE1_HEIGHT,
        RESNET18_STAGE1_WIDTH,
        RESNET18_STAGE2_CHANNELS,
        2,
        true,
        params.layer2_block0_conv1_weight,
        params.layer2_block0_conv1_bias,
        params.layer2_block0_conv1_bn_gamma,
        params.layer2_block0_conv1_bn_beta,
        params.layer2_block0_conv2_weight,
        params.layer2_block0_conv2_bias,
        params.layer2_block0_conv2_bn_gamma,
        params.layer2_block0_conv2_bn_beta,
        params.layer2_block0_proj_weight,
        params.layer2_block0_proj_bias,
        params.layer2_block0_proj_bn_gamma,
        params.layer2_block0_proj_bn_beta,
        batchnorm_cache.layer2_block0_conv1,
        batchnorm_cache.layer2_block0_conv2,
        &batchnorm_cache.layer2_block0_proj,
        batch_scratch1,
        batch_scratch2,
        workspace
    );
    COPY_BATCH_TO_FIELD(layer2_block0_conv1_out, batch_scratch1, stage2_length);
    COPY_BATCH_TO_FIELD(layer2_block0_out, batch_scratch2, stage2_length);

    COPY_FIELD_TO_BATCH(batch_scratch0, layer2_block0_out, stage2_length);
    basic_block_forward_batchnorm(
        batch_scratch0,
        RESNET18_STAGE2_CHANNELS,
        RESNET18_STAGE2_HEIGHT,
        RESNET18_STAGE2_WIDTH,
        RESNET18_STAGE2_CHANNELS,
        1,
        false,
        params.layer2_block1_conv1_weight,
        params.layer2_block1_conv1_bias,
        params.layer2_block1_conv1_bn_gamma,
        params.layer2_block1_conv1_bn_beta,
        params.layer2_block1_conv2_weight,
        params.layer2_block1_conv2_bias,
        params.layer2_block1_conv2_bn_gamma,
        params.layer2_block1_conv2_bn_beta,
        0,
        0,
        0,
        0,
        batchnorm_cache.layer2_block1_conv1,
        batchnorm_cache.layer2_block1_conv2,
        0,
        batch_scratch1,
        batch_scratch2,
        workspace
    );
    COPY_BATCH_TO_FIELD(layer2_block1_conv1_out, batch_scratch1, stage2_length);
    COPY_BATCH_TO_FIELD(layer2_block1_out, batch_scratch2, stage2_length);

    COPY_FIELD_TO_BATCH(batch_scratch0, layer2_block1_out, stage2_length);
    basic_block_forward_batchnorm(
        batch_scratch0,
        RESNET18_STAGE2_CHANNELS,
        RESNET18_STAGE2_HEIGHT,
        RESNET18_STAGE2_WIDTH,
        RESNET18_STAGE3_CHANNELS,
        2,
        true,
        params.layer3_block0_conv1_weight,
        params.layer3_block0_conv1_bias,
        params.layer3_block0_conv1_bn_gamma,
        params.layer3_block0_conv1_bn_beta,
        params.layer3_block0_conv2_weight,
        params.layer3_block0_conv2_bias,
        params.layer3_block0_conv2_bn_gamma,
        params.layer3_block0_conv2_bn_beta,
        params.layer3_block0_proj_weight,
        params.layer3_block0_proj_bias,
        params.layer3_block0_proj_bn_gamma,
        params.layer3_block0_proj_bn_beta,
        batchnorm_cache.layer3_block0_conv1,
        batchnorm_cache.layer3_block0_conv2,
        &batchnorm_cache.layer3_block0_proj,
        batch_scratch1,
        batch_scratch2,
        workspace
    );
    COPY_BATCH_TO_FIELD(layer3_block0_conv1_out, batch_scratch1, stage3_length);
    COPY_BATCH_TO_FIELD(layer3_block0_out, batch_scratch2, stage3_length);

    COPY_FIELD_TO_BATCH(batch_scratch0, layer3_block0_out, stage3_length);
    basic_block_forward_batchnorm(
        batch_scratch0,
        RESNET18_STAGE3_CHANNELS,
        RESNET18_STAGE3_HEIGHT,
        RESNET18_STAGE3_WIDTH,
        RESNET18_STAGE3_CHANNELS,
        1,
        false,
        params.layer3_block1_conv1_weight,
        params.layer3_block1_conv1_bias,
        params.layer3_block1_conv1_bn_gamma,
        params.layer3_block1_conv1_bn_beta,
        params.layer3_block1_conv2_weight,
        params.layer3_block1_conv2_bias,
        params.layer3_block1_conv2_bn_gamma,
        params.layer3_block1_conv2_bn_beta,
        0,
        0,
        0,
        0,
        batchnorm_cache.layer3_block1_conv1,
        batchnorm_cache.layer3_block1_conv2,
        0,
        batch_scratch1,
        batch_scratch2,
        workspace
    );
    COPY_BATCH_TO_FIELD(layer3_block1_conv1_out, batch_scratch1, stage3_length);
    COPY_BATCH_TO_FIELD(layer3_block1_out, batch_scratch2, stage3_length);

    COPY_FIELD_TO_BATCH(batch_scratch0, layer3_block1_out, stage3_length);
    basic_block_forward_batchnorm(
        batch_scratch0,
        RESNET18_STAGE3_CHANNELS,
        RESNET18_STAGE3_HEIGHT,
        RESNET18_STAGE3_WIDTH,
        RESNET18_STAGE4_CHANNELS,
        2,
        true,
        params.layer4_block0_conv1_weight,
        params.layer4_block0_conv1_bias,
        params.layer4_block0_conv1_bn_gamma,
        params.layer4_block0_conv1_bn_beta,
        params.layer4_block0_conv2_weight,
        params.layer4_block0_conv2_bias,
        params.layer4_block0_conv2_bn_gamma,
        params.layer4_block0_conv2_bn_beta,
        params.layer4_block0_proj_weight,
        params.layer4_block0_proj_bias,
        params.layer4_block0_proj_bn_gamma,
        params.layer4_block0_proj_bn_beta,
        batchnorm_cache.layer4_block0_conv1,
        batchnorm_cache.layer4_block0_conv2,
        &batchnorm_cache.layer4_block0_proj,
        batch_scratch1,
        batch_scratch2,
        workspace
    );
    COPY_BATCH_TO_FIELD(layer4_block0_conv1_out, batch_scratch1, stage4_length);
    COPY_BATCH_TO_FIELD(layer4_block0_out, batch_scratch2, stage4_length);

    COPY_FIELD_TO_BATCH(batch_scratch0, layer4_block0_out, stage4_length);
    basic_block_forward_batchnorm(
        batch_scratch0,
        RESNET18_STAGE4_CHANNELS,
        RESNET18_STAGE4_HEIGHT,
        RESNET18_STAGE4_WIDTH,
        RESNET18_STAGE4_CHANNELS,
        1,
        false,
        params.layer4_block1_conv1_weight,
        params.layer4_block1_conv1_bias,
        params.layer4_block1_conv1_bn_gamma,
        params.layer4_block1_conv1_bn_beta,
        params.layer4_block1_conv2_weight,
        params.layer4_block1_conv2_bias,
        params.layer4_block1_conv2_bn_gamma,
        params.layer4_block1_conv2_bn_beta,
        0,
        0,
        0,
        0,
        batchnorm_cache.layer4_block1_conv1,
        batchnorm_cache.layer4_block1_conv2,
        0,
        batch_scratch1,
        batch_scratch2,
        workspace
    );
    COPY_BATCH_TO_FIELD(layer4_block1_conv1_out, batch_scratch1, stage4_length);
    COPY_BATCH_TO_FIELD(layer4_block1_out, batch_scratch2, stage4_length);

    for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
        zero_buffer(batch_grad0[sample], stage4_length);
        for (int channel = 0; channel < RESNET18_STAGE4_CHANNELS; ++channel) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512 avg=512
            float sum = 0.0f;
            for (int y = 0; y < RESNET18_STAGE4_HEIGHT; ++y) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
                for (int x = 0; x < RESNET18_STAGE4_WIDTH; ++x) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
#pragma HLS PIPELINE II=1
                    sum += activations[sample].layer4_block1_out[
                        feature_index(channel, y, x, RESNET18_STAGE4_HEIGHT, RESNET18_STAGE4_WIDTH)
                    ];
                }
            }
            activations[sample].pooled[channel] =
                sum / (float)(RESNET18_STAGE4_HEIGHT * RESNET18_STAGE4_WIDTH);
        }

        for (int cls = 0; cls < RESNET18_NUM_CLASSES; ++cls) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
            float acc = params.fc_bias[cls];
            for (int channel = 0; channel < RESNET18_STAGE4_CHANNELS; ++channel) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512 avg=512
#pragma HLS PIPELINE II=1
                acc += params.fc_weight[fc_index(cls, channel)] * activations[sample].pooled[channel];
            }
            activations[sample].logits[cls] = acc;
            logits[sample][cls] = acc;
        }

        batch_results[sample].loss =
            resnet18_softmax_cross_entropy(activations[sample].logits, labels[sample], grad_logits);
        argmax_logits(activations[sample].logits, batch_results[sample].predicted_label);
        average_loss += batch_results[sample].loss;

        for (int cls = 0; cls < RESNET18_NUM_CLASSES; ++cls) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
            batch_gradients.fc_bias[cls] += grad_logits[cls];
            for (int channel = 0; channel < RESNET18_STAGE4_CHANNELS; ++channel) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512 avg=512
#pragma HLS PIPELINE II=1
                batch_gradients.fc_weight[fc_index(cls, channel)] +=
                    grad_logits[cls] * activations[sample].pooled[channel];
                batch_grad0[sample][feature_index(channel, 0, 0, 1, 1)] +=
                    params.fc_weight[fc_index(cls, channel)] * grad_logits[cls];
            }
        }

        for (int channel = 0; channel < RESNET18_STAGE4_CHANNELS; ++channel) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512 avg=512
            float grad_value =
                batch_grad0[sample][feature_index(channel, 0, 0, 1, 1)] /
                (float)(RESNET18_STAGE4_HEIGHT * RESNET18_STAGE4_WIDTH);
            for (int y = 0; y < RESNET18_STAGE4_HEIGHT; ++y) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
                for (int x = 0; x < RESNET18_STAGE4_WIDTH; ++x) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
#pragma HLS PIPELINE II=1
                    batch_grad0[sample][feature_index(channel, y, x, RESNET18_STAGE4_HEIGHT, RESNET18_STAGE4_WIDTH)] =
                        grad_value;
                }
            }
        }
    }

    COPY_FIELD_TO_BATCH(batch_scratch0, layer4_block0_out, stage4_length);
    COPY_FIELD_TO_BATCH(batch_scratch1, layer4_block1_conv1_out, stage4_length);
    COPY_FIELD_TO_BATCH(batch_scratch2, layer4_block1_out, stage4_length);
    basic_block_backward_batchnorm(
        batch_scratch0,
        batch_scratch1,
        batch_scratch2,
        batch_grad0,
        RESNET18_STAGE4_CHANNELS,
        RESNET18_STAGE4_HEIGHT,
        RESNET18_STAGE4_WIDTH,
        RESNET18_STAGE4_CHANNELS,
        1,
        false,
        params.layer4_block1_conv1_weight,
        params.layer4_block1_conv1_bias,
        params.layer4_block1_conv1_bn_gamma,
        params.layer4_block1_conv2_weight,
        params.layer4_block1_conv2_bias,
        params.layer4_block1_conv2_bn_gamma,
        0,
        0,
        0,
        batchnorm_cache.layer4_block1_conv1,
        batchnorm_cache.layer4_block1_conv2,
        0,
        batch_grad1,
        batch_gradients.layer4_block1_conv1_weight,
        batch_gradients.layer4_block1_conv1_bias,
        batch_gradients.layer4_block1_conv1_bn_gamma,
        batch_gradients.layer4_block1_conv1_bn_beta,
        batch_gradients.layer4_block1_conv2_weight,
        batch_gradients.layer4_block1_conv2_bias,
        batch_gradients.layer4_block1_conv2_bn_gamma,
        batch_gradients.layer4_block1_conv2_bn_beta,
        0,
        0,
        0,
        0,
        workspace
    );

    COPY_FIELD_TO_BATCH(batch_scratch0, layer3_block1_out, stage3_length);
    COPY_FIELD_TO_BATCH(batch_scratch1, layer4_block0_conv1_out, stage4_length);
    COPY_FIELD_TO_BATCH(batch_scratch2, layer4_block0_out, stage4_length);
    basic_block_backward_batchnorm(
        batch_scratch0,
        batch_scratch1,
        batch_scratch2,
        batch_grad1,
        RESNET18_STAGE3_CHANNELS,
        RESNET18_STAGE3_HEIGHT,
        RESNET18_STAGE3_WIDTH,
        RESNET18_STAGE4_CHANNELS,
        2,
        true,
        params.layer4_block0_conv1_weight,
        params.layer4_block0_conv1_bias,
        params.layer4_block0_conv1_bn_gamma,
        params.layer4_block0_conv2_weight,
        params.layer4_block0_conv2_bias,
        params.layer4_block0_conv2_bn_gamma,
        params.layer4_block0_proj_weight,
        params.layer4_block0_proj_bias,
        params.layer4_block0_proj_bn_gamma,
        batchnorm_cache.layer4_block0_conv1,
        batchnorm_cache.layer4_block0_conv2,
        &batchnorm_cache.layer4_block0_proj,
        batch_grad0,
        batch_gradients.layer4_block0_conv1_weight,
        batch_gradients.layer4_block0_conv1_bias,
        batch_gradients.layer4_block0_conv1_bn_gamma,
        batch_gradients.layer4_block0_conv1_bn_beta,
        batch_gradients.layer4_block0_conv2_weight,
        batch_gradients.layer4_block0_conv2_bias,
        batch_gradients.layer4_block0_conv2_bn_gamma,
        batch_gradients.layer4_block0_conv2_bn_beta,
        batch_gradients.layer4_block0_proj_weight,
        batch_gradients.layer4_block0_proj_bias,
        batch_gradients.layer4_block0_proj_bn_gamma,
        batch_gradients.layer4_block0_proj_bn_beta,
        workspace
    );

    COPY_FIELD_TO_BATCH(batch_scratch0, layer3_block0_out, stage3_length);
    COPY_FIELD_TO_BATCH(batch_scratch1, layer3_block1_conv1_out, stage3_length);
    COPY_FIELD_TO_BATCH(batch_scratch2, layer3_block1_out, stage3_length);
    basic_block_backward_batchnorm(
        batch_scratch0,
        batch_scratch1,
        batch_scratch2,
        batch_grad0,
        RESNET18_STAGE3_CHANNELS,
        RESNET18_STAGE3_HEIGHT,
        RESNET18_STAGE3_WIDTH,
        RESNET18_STAGE3_CHANNELS,
        1,
        false,
        params.layer3_block1_conv1_weight,
        params.layer3_block1_conv1_bias,
        params.layer3_block1_conv1_bn_gamma,
        params.layer3_block1_conv2_weight,
        params.layer3_block1_conv2_bias,
        params.layer3_block1_conv2_bn_gamma,
        0,
        0,
        0,
        batchnorm_cache.layer3_block1_conv1,
        batchnorm_cache.layer3_block1_conv2,
        0,
        batch_grad1,
        batch_gradients.layer3_block1_conv1_weight,
        batch_gradients.layer3_block1_conv1_bias,
        batch_gradients.layer3_block1_conv1_bn_gamma,
        batch_gradients.layer3_block1_conv1_bn_beta,
        batch_gradients.layer3_block1_conv2_weight,
        batch_gradients.layer3_block1_conv2_bias,
        batch_gradients.layer3_block1_conv2_bn_gamma,
        batch_gradients.layer3_block1_conv2_bn_beta,
        0,
        0,
        0,
        0,
        workspace
    );

    COPY_FIELD_TO_BATCH(batch_scratch0, layer2_block1_out, stage2_length);
    COPY_FIELD_TO_BATCH(batch_scratch1, layer3_block0_conv1_out, stage3_length);
    COPY_FIELD_TO_BATCH(batch_scratch2, layer3_block0_out, stage3_length);
    basic_block_backward_batchnorm(
        batch_scratch0,
        batch_scratch1,
        batch_scratch2,
        batch_grad1,
        RESNET18_STAGE2_CHANNELS,
        RESNET18_STAGE2_HEIGHT,
        RESNET18_STAGE2_WIDTH,
        RESNET18_STAGE3_CHANNELS,
        2,
        true,
        params.layer3_block0_conv1_weight,
        params.layer3_block0_conv1_bias,
        params.layer3_block0_conv1_bn_gamma,
        params.layer3_block0_conv2_weight,
        params.layer3_block0_conv2_bias,
        params.layer3_block0_conv2_bn_gamma,
        params.layer3_block0_proj_weight,
        params.layer3_block0_proj_bias,
        params.layer3_block0_proj_bn_gamma,
        batchnorm_cache.layer3_block0_conv1,
        batchnorm_cache.layer3_block0_conv2,
        &batchnorm_cache.layer3_block0_proj,
        batch_grad0,
        batch_gradients.layer3_block0_conv1_weight,
        batch_gradients.layer3_block0_conv1_bias,
        batch_gradients.layer3_block0_conv1_bn_gamma,
        batch_gradients.layer3_block0_conv1_bn_beta,
        batch_gradients.layer3_block0_conv2_weight,
        batch_gradients.layer3_block0_conv2_bias,
        batch_gradients.layer3_block0_conv2_bn_gamma,
        batch_gradients.layer3_block0_conv2_bn_beta,
        batch_gradients.layer3_block0_proj_weight,
        batch_gradients.layer3_block0_proj_bias,
        batch_gradients.layer3_block0_proj_bn_gamma,
        batch_gradients.layer3_block0_proj_bn_beta,
        workspace
    );

    COPY_FIELD_TO_BATCH(batch_scratch0, layer2_block0_out, stage2_length);
    COPY_FIELD_TO_BATCH(batch_scratch1, layer2_block1_conv1_out, stage2_length);
    COPY_FIELD_TO_BATCH(batch_scratch2, layer2_block1_out, stage2_length);
    basic_block_backward_batchnorm(
        batch_scratch0,
        batch_scratch1,
        batch_scratch2,
        batch_grad0,
        RESNET18_STAGE2_CHANNELS,
        RESNET18_STAGE2_HEIGHT,
        RESNET18_STAGE2_WIDTH,
        RESNET18_STAGE2_CHANNELS,
        1,
        false,
        params.layer2_block1_conv1_weight,
        params.layer2_block1_conv1_bias,
        params.layer2_block1_conv1_bn_gamma,
        params.layer2_block1_conv2_weight,
        params.layer2_block1_conv2_bias,
        params.layer2_block1_conv2_bn_gamma,
        0,
        0,
        0,
        batchnorm_cache.layer2_block1_conv1,
        batchnorm_cache.layer2_block1_conv2,
        0,
        batch_grad1,
        batch_gradients.layer2_block1_conv1_weight,
        batch_gradients.layer2_block1_conv1_bias,
        batch_gradients.layer2_block1_conv1_bn_gamma,
        batch_gradients.layer2_block1_conv1_bn_beta,
        batch_gradients.layer2_block1_conv2_weight,
        batch_gradients.layer2_block1_conv2_bias,
        batch_gradients.layer2_block1_conv2_bn_gamma,
        batch_gradients.layer2_block1_conv2_bn_beta,
        0,
        0,
        0,
        0,
        workspace
    );

    COPY_FIELD_TO_BATCH(batch_scratch0, layer1_block1_out, stage1_length);
    COPY_FIELD_TO_BATCH(batch_scratch1, layer2_block0_conv1_out, stage2_length);
    COPY_FIELD_TO_BATCH(batch_scratch2, layer2_block0_out, stage2_length);
    basic_block_backward_batchnorm(
        batch_scratch0,
        batch_scratch1,
        batch_scratch2,
        batch_grad1,
        RESNET18_STAGE1_CHANNELS,
        RESNET18_STAGE1_HEIGHT,
        RESNET18_STAGE1_WIDTH,
        RESNET18_STAGE2_CHANNELS,
        2,
        true,
        params.layer2_block0_conv1_weight,
        params.layer2_block0_conv1_bias,
        params.layer2_block0_conv1_bn_gamma,
        params.layer2_block0_conv2_weight,
        params.layer2_block0_conv2_bias,
        params.layer2_block0_conv2_bn_gamma,
        params.layer2_block0_proj_weight,
        params.layer2_block0_proj_bias,
        params.layer2_block0_proj_bn_gamma,
        batchnorm_cache.layer2_block0_conv1,
        batchnorm_cache.layer2_block0_conv2,
        &batchnorm_cache.layer2_block0_proj,
        batch_grad0,
        batch_gradients.layer2_block0_conv1_weight,
        batch_gradients.layer2_block0_conv1_bias,
        batch_gradients.layer2_block0_conv1_bn_gamma,
        batch_gradients.layer2_block0_conv1_bn_beta,
        batch_gradients.layer2_block0_conv2_weight,
        batch_gradients.layer2_block0_conv2_bias,
        batch_gradients.layer2_block0_conv2_bn_gamma,
        batch_gradients.layer2_block0_conv2_bn_beta,
        batch_gradients.layer2_block0_proj_weight,
        batch_gradients.layer2_block0_proj_bias,
        batch_gradients.layer2_block0_proj_bn_gamma,
        batch_gradients.layer2_block0_proj_bn_beta,
        workspace
    );

    COPY_FIELD_TO_BATCH(batch_scratch0, layer1_block0_out, stage1_length);
    COPY_FIELD_TO_BATCH(batch_scratch1, layer1_block1_conv1_out, stage1_length);
    COPY_FIELD_TO_BATCH(batch_scratch2, layer1_block1_out, stage1_length);
    basic_block_backward_batchnorm(
        batch_scratch0,
        batch_scratch1,
        batch_scratch2,
        batch_grad1,
        RESNET18_STAGE1_CHANNELS,
        RESNET18_STAGE1_HEIGHT,
        RESNET18_STAGE1_WIDTH,
        RESNET18_STAGE1_CHANNELS,
        1,
        false,
        params.layer1_block1_conv1_weight,
        params.layer1_block1_conv1_bias,
        params.layer1_block1_conv1_bn_gamma,
        params.layer1_block1_conv2_weight,
        params.layer1_block1_conv2_bias,
        params.layer1_block1_conv2_bn_gamma,
        0,
        0,
        0,
        batchnorm_cache.layer1_block1_conv1,
        batchnorm_cache.layer1_block1_conv2,
        0,
        batch_grad0,
        batch_gradients.layer1_block1_conv1_weight,
        batch_gradients.layer1_block1_conv1_bias,
        batch_gradients.layer1_block1_conv1_bn_gamma,
        batch_gradients.layer1_block1_conv1_bn_beta,
        batch_gradients.layer1_block1_conv2_weight,
        batch_gradients.layer1_block1_conv2_bias,
        batch_gradients.layer1_block1_conv2_bn_gamma,
        batch_gradients.layer1_block1_conv2_bn_beta,
        0,
        0,
        0,
        0,
        workspace
    );

    COPY_FIELD_TO_BATCH(batch_scratch0, stem_out, stage1_length);
    COPY_FIELD_TO_BATCH(batch_scratch1, layer1_block0_conv1_out, stage1_length);
    COPY_FIELD_TO_BATCH(batch_scratch2, layer1_block0_out, stage1_length);
    basic_block_backward_batchnorm(
        batch_scratch0,
        batch_scratch1,
        batch_scratch2,
        batch_grad1,
        RESNET18_STAGE1_CHANNELS,
        RESNET18_STAGE1_HEIGHT,
        RESNET18_STAGE1_WIDTH,
        RESNET18_STAGE1_CHANNELS,
        1,
        false,
        params.layer1_block0_conv1_weight,
        params.layer1_block0_conv1_bias,
        params.layer1_block0_conv1_bn_gamma,
        params.layer1_block0_conv2_weight,
        params.layer1_block0_conv2_bias,
        params.layer1_block0_conv2_bn_gamma,
        0,
        0,
        0,
        batchnorm_cache.layer1_block0_conv1,
        batchnorm_cache.layer1_block0_conv2,
        0,
        batch_grad0,
        batch_gradients.layer1_block0_conv1_weight,
        batch_gradients.layer1_block0_conv1_bias,
        batch_gradients.layer1_block0_conv1_bn_gamma,
        batch_gradients.layer1_block0_conv1_bn_beta,
        batch_gradients.layer1_block0_conv2_weight,
        batch_gradients.layer1_block0_conv2_bias,
        batch_gradients.layer1_block0_conv2_bn_gamma,
        batch_gradients.layer1_block0_conv2_bn_beta,
        0,
        0,
        0,
        0,
        workspace
    );

    for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
        apply_relu_gradient(
            activations[sample].stem_out,
            batch_grad0[sample],
            stem_length
        );
        conv3x3_forward(
            images[sample],
            RESNET18_IMAGE_CHANNELS,
            RESNET18_IMAGE_HEIGHT,
            RESNET18_IMAGE_WIDTH,
            params.stem_weight,
            params.stem_bias,
            RESNET18_STEM_CHANNELS,
            1,
            batch_scratch1[sample]
        );
    }
    batchnorm_backward_from_input(
        batch_scratch1,
        batch_grad0,
        RESNET18_STEM_CHANNELS,
        RESNET18_STAGE1_HEIGHT,
        RESNET18_STAGE1_WIDTH,
        batchnorm_cache.stem,
        params.stem_bn_gamma,
        batch_grad1,
        batch_gradients.stem_bn_gamma,
        batch_gradients.stem_bn_beta
    );
    for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
        conv3x3_backward(
            images[sample],
            RESNET18_IMAGE_CHANNELS,
            RESNET18_IMAGE_HEIGHT,
            RESNET18_IMAGE_WIDTH,
            batch_grad1[sample],
            RESNET18_STEM_CHANNELS,
            1,
            params.stem_weight,
            batch_tmp0[sample],
            batch_gradients.stem_weight,
            batch_gradients.stem_bias
        );
    }

    average_loss /= (float)RESNET18_BATCH_SIZE;
    const float scale = 1.0f / (float)RESNET18_BATCH_SIZE;

#define SCALE_GRAD_FIELD(name) \
    scale_gradient_array(batch_gradients.name, (int)(sizeof(batch_gradients.name) / sizeof(float)), scale);
    RESNET18_GRAD_ARRAYS(SCALE_GRAD_FIELD)
#undef SCALE_GRAD_FIELD

    // Average first, then clip the batch gradients before applying the update.
#define CLIP_GRAD_FIELD(name) \
    clip_gradient_array(batch_gradients.name, (int)(sizeof(batch_gradients.name) / sizeof(float)));
    RESNET18_GRAD_ARRAYS(CLIP_GRAD_FIELD)
#undef CLIP_GRAD_FIELD

    resnet18_sgd_momentum_update(params, state, batch_gradients, learning_rate, momentum);

#undef COPY_FIELD_TO_BATCH
#undef COPY_BATCH_TO_FIELD
}

void resnet18_train_batch(
    const float images[RESNET18_BATCH_SIZE][RESNET18_INPUT_SIZE],
    const int labels[RESNET18_BATCH_SIZE],
    float learning_rate,
    float momentum,
    ResNet18Params &params,
    ResNet18State &state,
    float logits[RESNET18_BATCH_SIZE][RESNET18_NUM_CLASSES],
    ResNet18StepResult batch_results[RESNET18_BATCH_SIZE],
    float &average_loss
) {
#pragma HLS INLINE off
    resnet18_train_batch_extmem(
        images,
        labels,
        learning_rate,
        momentum,
        params,
        state,
        default_batch_workspace,
        logits,
        batch_results,
        average_loss
    );
}

void resnet18_basic_block_forward_bench_l3(
    const float input_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    ResNet18Params &params,
    ResNet18BatchTrainingWorkspace &workspace,
    float output_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    float &checksum
) {
#pragma HLS INLINE off
    const int output_length = RESNET18_STAGE3_CHANNELS * RESNET18_STAGE3_HEIGHT * RESNET18_STAGE3_WIDTH;

    basic_block_forward_batchnorm(
        input_batch,
        RESNET18_STAGE2_CHANNELS,
        RESNET18_STAGE2_HEIGHT,
        RESNET18_STAGE2_WIDTH,
        RESNET18_STAGE3_CHANNELS,
        2,
        true,
        params.layer3_block0_conv1_weight,
        params.layer3_block0_conv1_bias,
        params.layer3_block0_conv1_bn_gamma,
        params.layer3_block0_conv1_bn_beta,
        params.layer3_block0_conv2_weight,
        params.layer3_block0_conv2_bias,
        params.layer3_block0_conv2_bn_gamma,
        params.layer3_block0_conv2_bn_beta,
        params.layer3_block0_proj_weight,
        params.layer3_block0_proj_bias,
        params.layer3_block0_proj_bn_gamma,
        params.layer3_block0_proj_bn_beta,
        workspace.batchnorm_cache.layer3_block0_conv1,
        workspace.batchnorm_cache.layer3_block0_conv2,
        &workspace.batchnorm_cache.layer3_block0_proj,
        workspace.batch_scratch0,
        output_batch,
        workspace
    );

    checksum = 0.0f;
    for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
        for (int i = 0; i < output_length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=16384 max=16384 avg=16384
#pragma HLS PIPELINE II=1
            checksum += output_batch[sample][i];
        }
    }
}

void resnet18_basic_block_backward_bench_l3(
    const float input_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    float conv1_out_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    float block_output_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    float grad_output_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    ResNet18Params &params,
    ResNet18BatchTrainingWorkspace &workspace,
    float grad_input_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    float &checksum
) {
#pragma HLS INLINE off
    const int input_length = RESNET18_STAGE2_CHANNELS * RESNET18_STAGE2_HEIGHT * RESNET18_STAGE2_WIDTH;

    for (int channel = 0; channel < RESNET18_STAGE3_CHANNELS; ++channel) {
#pragma HLS LOOP_TRIPCOUNT min=256 max=256 avg=256
#pragma HLS PIPELINE II=1
        workspace.batchnorm_cache.layer3_block0_conv1.mean[channel] = 0.0f;
        workspace.batchnorm_cache.layer3_block0_conv1.inv_std[channel] = 1.0f;
        workspace.batchnorm_cache.layer3_block0_conv2.mean[channel] = 0.0f;
        workspace.batchnorm_cache.layer3_block0_conv2.inv_std[channel] = 1.0f;
        workspace.batchnorm_cache.layer3_block0_proj.mean[channel] = 0.0f;
        workspace.batchnorm_cache.layer3_block0_proj.inv_std[channel] = 1.0f;
    }

    zero_buffer(workspace.batch_gradients.layer3_block0_conv1_weight,
                (int)(sizeof(workspace.batch_gradients.layer3_block0_conv1_weight) / sizeof(float)));
    zero_buffer(workspace.batch_gradients.layer3_block0_conv1_bias,
                (int)(sizeof(workspace.batch_gradients.layer3_block0_conv1_bias) / sizeof(float)));
    zero_buffer(workspace.batch_gradients.layer3_block0_conv1_bn_gamma,
                (int)(sizeof(workspace.batch_gradients.layer3_block0_conv1_bn_gamma) / sizeof(float)));
    zero_buffer(workspace.batch_gradients.layer3_block0_conv1_bn_beta,
                (int)(sizeof(workspace.batch_gradients.layer3_block0_conv1_bn_beta) / sizeof(float)));
    zero_buffer(workspace.batch_gradients.layer3_block0_conv2_weight,
                (int)(sizeof(workspace.batch_gradients.layer3_block0_conv2_weight) / sizeof(float)));
    zero_buffer(workspace.batch_gradients.layer3_block0_conv2_bias,
                (int)(sizeof(workspace.batch_gradients.layer3_block0_conv2_bias) / sizeof(float)));
    zero_buffer(workspace.batch_gradients.layer3_block0_conv2_bn_gamma,
                (int)(sizeof(workspace.batch_gradients.layer3_block0_conv2_bn_gamma) / sizeof(float)));
    zero_buffer(workspace.batch_gradients.layer3_block0_conv2_bn_beta,
                (int)(sizeof(workspace.batch_gradients.layer3_block0_conv2_bn_beta) / sizeof(float)));
    zero_buffer(workspace.batch_gradients.layer3_block0_proj_weight,
                (int)(sizeof(workspace.batch_gradients.layer3_block0_proj_weight) / sizeof(float)));
    zero_buffer(workspace.batch_gradients.layer3_block0_proj_bias,
                (int)(sizeof(workspace.batch_gradients.layer3_block0_proj_bias) / sizeof(float)));
    zero_buffer(workspace.batch_gradients.layer3_block0_proj_bn_gamma,
                (int)(sizeof(workspace.batch_gradients.layer3_block0_proj_bn_gamma) / sizeof(float)));
    zero_buffer(workspace.batch_gradients.layer3_block0_proj_bn_beta,
                (int)(sizeof(workspace.batch_gradients.layer3_block0_proj_bn_beta) / sizeof(float)));

    basic_block_backward_batchnorm(
        input_batch,
        conv1_out_batch,
        block_output_batch,
        grad_output_batch,
        RESNET18_STAGE2_CHANNELS,
        RESNET18_STAGE2_HEIGHT,
        RESNET18_STAGE2_WIDTH,
        RESNET18_STAGE3_CHANNELS,
        2,
        true,
        params.layer3_block0_conv1_weight,
        params.layer3_block0_conv1_bias,
        params.layer3_block0_conv1_bn_gamma,
        params.layer3_block0_conv2_weight,
        params.layer3_block0_conv2_bias,
        params.layer3_block0_conv2_bn_gamma,
        params.layer3_block0_proj_weight,
        params.layer3_block0_proj_bias,
        params.layer3_block0_proj_bn_gamma,
        workspace.batchnorm_cache.layer3_block0_conv1,
        workspace.batchnorm_cache.layer3_block0_conv2,
        &workspace.batchnorm_cache.layer3_block0_proj,
        grad_input_batch,
        workspace.batch_gradients.layer3_block0_conv1_weight,
        workspace.batch_gradients.layer3_block0_conv1_bias,
        workspace.batch_gradients.layer3_block0_conv1_bn_gamma,
        workspace.batch_gradients.layer3_block0_conv1_bn_beta,
        workspace.batch_gradients.layer3_block0_conv2_weight,
        workspace.batch_gradients.layer3_block0_conv2_bias,
        workspace.batch_gradients.layer3_block0_conv2_bn_gamma,
        workspace.batch_gradients.layer3_block0_conv2_bn_beta,
        workspace.batch_gradients.layer3_block0_proj_weight,
        workspace.batch_gradients.layer3_block0_proj_bias,
        workspace.batch_gradients.layer3_block0_proj_bn_gamma,
        workspace.batch_gradients.layer3_block0_proj_bn_beta,
        workspace
    );

    checksum = 0.0f;
    for (int sample = 0; sample < RESNET18_BATCH_SIZE; ++sample) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10 avg=10
        for (int i = 0; i < input_length; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=32768 max=32768 avg=32768
#pragma HLS PIPELINE II=1
            checksum += grad_input_batch[sample][i];
        }
    }
}
