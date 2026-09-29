#ifndef RESNET18_CIFAR_H
#define RESNET18_CIFAR_H

#include "bm8_quantizer.h"

#define RESNET18_IMAGE_CHANNELS 3
#define RESNET18_IMAGE_HEIGHT 32
#define RESNET18_IMAGE_WIDTH 32
#define RESNET18_INPUT_SIZE (RESNET18_IMAGE_CHANNELS * RESNET18_IMAGE_HEIGHT * RESNET18_IMAGE_WIDTH)
#define RESNET18_NUM_CLASSES 10
#define RESNET18_BATCH_SIZE 10
#define RESNET18_GRAD_CLIP_VALUE 1.0f

#define RESNET18_STEM_CHANNELS 64
#define RESNET18_STAGE1_CHANNELS 64
#define RESNET18_STAGE2_CHANNELS 128
#define RESNET18_STAGE3_CHANNELS 256
#define RESNET18_STAGE4_CHANNELS 512

#define RESNET18_STAGE1_HEIGHT 32
#define RESNET18_STAGE1_WIDTH 32
#define RESNET18_STAGE2_HEIGHT 16
#define RESNET18_STAGE2_WIDTH 16
#define RESNET18_STAGE3_HEIGHT 8
#define RESNET18_STAGE3_WIDTH 8
#define RESNET18_STAGE4_HEIGHT 4
#define RESNET18_STAGE4_WIDTH 4

#define RESNET18_KERNEL_SIZE 3
#define RESNET18_MAX_ACTIVATION_ELEMENTS (RESNET18_STAGE1_CHANNELS * RESNET18_STAGE1_HEIGHT * RESNET18_STAGE1_WIDTH)

#define RESNET18_BLOCK_PARAMS(prefix, in_ch, out_ch) \
    float prefix##_conv1_weight[(out_ch) * (in_ch) * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE]; \
    float prefix##_conv1_bias[(out_ch)]; \
    float prefix##_conv2_weight[(out_ch) * (out_ch) * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE]; \
    float prefix##_conv2_bias[(out_ch)]

#define RESNET18_BLOCK_STATE(prefix, in_ch, out_ch) \
    float velocity_##prefix##_conv1_weight[(out_ch) * (in_ch) * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE]; \
    float velocity_##prefix##_conv1_bias[(out_ch)]; \
    float velocity_##prefix##_conv2_weight[(out_ch) * (out_ch) * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE]; \
    float velocity_##prefix##_conv2_bias[(out_ch)]

#define RESNET18_BLOCK_GRADS(prefix, in_ch, out_ch) \
    float prefix##_conv1_weight[(out_ch) * (in_ch) * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE]; \
    float prefix##_conv1_bias[(out_ch)]; \
    float prefix##_conv2_weight[(out_ch) * (out_ch) * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE]; \
    float prefix##_conv2_bias[(out_ch)]

#define RESNET18_PROJ_PARAMS(prefix, in_ch, out_ch) \
    float prefix##_proj_weight[(out_ch) * (in_ch)]; \
    float prefix##_proj_bias[(out_ch)]

#define RESNET18_PROJ_STATE(prefix, in_ch, out_ch) \
    float velocity_##prefix##_proj_weight[(out_ch) * (in_ch)]; \
    float velocity_##prefix##_proj_bias[(out_ch)]

#define RESNET18_PROJ_GRADS(prefix, in_ch, out_ch) \
    float prefix##_proj_weight[(out_ch) * (in_ch)]; \
    float prefix##_proj_bias[(out_ch)]

#define RESNET18_BN_AFFINE_PARAMS(prefix, channels) \
    float prefix##_bn_gamma[(channels)]; \
    float prefix##_bn_beta[(channels)]

#define RESNET18_BN_AFFINE_STATE(prefix, channels) \
    float velocity_##prefix##_bn_gamma[(channels)]; \
    float velocity_##prefix##_bn_beta[(channels)]

#define RESNET18_BN_AFFINE_GRADS(prefix, channels) \
    float prefix##_bn_gamma[(channels)]; \
    float prefix##_bn_beta[(channels)]

#define RESNET18_PARAM_ARRAYS(X) \
    X(stem_weight) \
    X(stem_bias) \
    X(stem_bn_gamma) \
    X(stem_bn_beta) \
    X(layer1_block0_conv1_weight) \
    X(layer1_block0_conv1_bias) \
    X(layer1_block0_conv1_bn_gamma) \
    X(layer1_block0_conv1_bn_beta) \
    X(layer1_block0_conv2_weight) \
    X(layer1_block0_conv2_bias) \
    X(layer1_block0_conv2_bn_gamma) \
    X(layer1_block0_conv2_bn_beta) \
    X(layer1_block1_conv1_weight) \
    X(layer1_block1_conv1_bias) \
    X(layer1_block1_conv1_bn_gamma) \
    X(layer1_block1_conv1_bn_beta) \
    X(layer1_block1_conv2_weight) \
    X(layer1_block1_conv2_bias) \
    X(layer1_block1_conv2_bn_gamma) \
    X(layer1_block1_conv2_bn_beta) \
    X(layer2_block0_conv1_weight) \
    X(layer2_block0_conv1_bias) \
    X(layer2_block0_conv1_bn_gamma) \
    X(layer2_block0_conv1_bn_beta) \
    X(layer2_block0_conv2_weight) \
    X(layer2_block0_conv2_bias) \
    X(layer2_block0_conv2_bn_gamma) \
    X(layer2_block0_conv2_bn_beta) \
    X(layer2_block0_proj_weight) \
    X(layer2_block0_proj_bias) \
    X(layer2_block0_proj_bn_gamma) \
    X(layer2_block0_proj_bn_beta) \
    X(layer2_block1_conv1_weight) \
    X(layer2_block1_conv1_bias) \
    X(layer2_block1_conv1_bn_gamma) \
    X(layer2_block1_conv1_bn_beta) \
    X(layer2_block1_conv2_weight) \
    X(layer2_block1_conv2_bias) \
    X(layer2_block1_conv2_bn_gamma) \
    X(layer2_block1_conv2_bn_beta) \
    X(layer3_block0_conv1_weight) \
    X(layer3_block0_conv1_bias) \
    X(layer3_block0_conv1_bn_gamma) \
    X(layer3_block0_conv1_bn_beta) \
    X(layer3_block0_conv2_weight) \
    X(layer3_block0_conv2_bias) \
    X(layer3_block0_conv2_bn_gamma) \
    X(layer3_block0_conv2_bn_beta) \
    X(layer3_block0_proj_weight) \
    X(layer3_block0_proj_bias) \
    X(layer3_block0_proj_bn_gamma) \
    X(layer3_block0_proj_bn_beta) \
    X(layer3_block1_conv1_weight) \
    X(layer3_block1_conv1_bias) \
    X(layer3_block1_conv1_bn_gamma) \
    X(layer3_block1_conv1_bn_beta) \
    X(layer3_block1_conv2_weight) \
    X(layer3_block1_conv2_bias) \
    X(layer3_block1_conv2_bn_gamma) \
    X(layer3_block1_conv2_bn_beta) \
    X(layer4_block0_conv1_weight) \
    X(layer4_block0_conv1_bias) \
    X(layer4_block0_conv1_bn_gamma) \
    X(layer4_block0_conv1_bn_beta) \
    X(layer4_block0_conv2_weight) \
    X(layer4_block0_conv2_bias) \
    X(layer4_block0_conv2_bn_gamma) \
    X(layer4_block0_conv2_bn_beta) \
    X(layer4_block0_proj_weight) \
    X(layer4_block0_proj_bias) \
    X(layer4_block0_proj_bn_gamma) \
    X(layer4_block0_proj_bn_beta) \
    X(layer4_block1_conv1_weight) \
    X(layer4_block1_conv1_bias) \
    X(layer4_block1_conv1_bn_gamma) \
    X(layer4_block1_conv1_bn_beta) \
    X(layer4_block1_conv2_weight) \
    X(layer4_block1_conv2_bias) \
    X(layer4_block1_conv2_bn_gamma) \
    X(layer4_block1_conv2_bn_beta) \
    X(fc_weight) \
    X(fc_bias)

#define RESNET18_STATE_ARRAYS(X) \
    X(velocity_stem_weight) \
    X(velocity_stem_bias) \
    X(velocity_stem_bn_gamma) \
    X(velocity_stem_bn_beta) \
    X(velocity_layer1_block0_conv1_weight) \
    X(velocity_layer1_block0_conv1_bias) \
    X(velocity_layer1_block0_conv1_bn_gamma) \
    X(velocity_layer1_block0_conv1_bn_beta) \
    X(velocity_layer1_block0_conv2_weight) \
    X(velocity_layer1_block0_conv2_bias) \
    X(velocity_layer1_block0_conv2_bn_gamma) \
    X(velocity_layer1_block0_conv2_bn_beta) \
    X(velocity_layer1_block1_conv1_weight) \
    X(velocity_layer1_block1_conv1_bias) \
    X(velocity_layer1_block1_conv1_bn_gamma) \
    X(velocity_layer1_block1_conv1_bn_beta) \
    X(velocity_layer1_block1_conv2_weight) \
    X(velocity_layer1_block1_conv2_bias) \
    X(velocity_layer1_block1_conv2_bn_gamma) \
    X(velocity_layer1_block1_conv2_bn_beta) \
    X(velocity_layer2_block0_conv1_weight) \
    X(velocity_layer2_block0_conv1_bias) \
    X(velocity_layer2_block0_conv1_bn_gamma) \
    X(velocity_layer2_block0_conv1_bn_beta) \
    X(velocity_layer2_block0_conv2_weight) \
    X(velocity_layer2_block0_conv2_bias) \
    X(velocity_layer2_block0_conv2_bn_gamma) \
    X(velocity_layer2_block0_conv2_bn_beta) \
    X(velocity_layer2_block0_proj_weight) \
    X(velocity_layer2_block0_proj_bias) \
    X(velocity_layer2_block0_proj_bn_gamma) \
    X(velocity_layer2_block0_proj_bn_beta) \
    X(velocity_layer2_block1_conv1_weight) \
    X(velocity_layer2_block1_conv1_bias) \
    X(velocity_layer2_block1_conv1_bn_gamma) \
    X(velocity_layer2_block1_conv1_bn_beta) \
    X(velocity_layer2_block1_conv2_weight) \
    X(velocity_layer2_block1_conv2_bias) \
    X(velocity_layer2_block1_conv2_bn_gamma) \
    X(velocity_layer2_block1_conv2_bn_beta) \
    X(velocity_layer3_block0_conv1_weight) \
    X(velocity_layer3_block0_conv1_bias) \
    X(velocity_layer3_block0_conv1_bn_gamma) \
    X(velocity_layer3_block0_conv1_bn_beta) \
    X(velocity_layer3_block0_conv2_weight) \
    X(velocity_layer3_block0_conv2_bias) \
    X(velocity_layer3_block0_conv2_bn_gamma) \
    X(velocity_layer3_block0_conv2_bn_beta) \
    X(velocity_layer3_block0_proj_weight) \
    X(velocity_layer3_block0_proj_bias) \
    X(velocity_layer3_block0_proj_bn_gamma) \
    X(velocity_layer3_block0_proj_bn_beta) \
    X(velocity_layer3_block1_conv1_weight) \
    X(velocity_layer3_block1_conv1_bias) \
    X(velocity_layer3_block1_conv1_bn_gamma) \
    X(velocity_layer3_block1_conv1_bn_beta) \
    X(velocity_layer3_block1_conv2_weight) \
    X(velocity_layer3_block1_conv2_bias) \
    X(velocity_layer3_block1_conv2_bn_gamma) \
    X(velocity_layer3_block1_conv2_bn_beta) \
    X(velocity_layer4_block0_conv1_weight) \
    X(velocity_layer4_block0_conv1_bias) \
    X(velocity_layer4_block0_conv1_bn_gamma) \
    X(velocity_layer4_block0_conv1_bn_beta) \
    X(velocity_layer4_block0_conv2_weight) \
    X(velocity_layer4_block0_conv2_bias) \
    X(velocity_layer4_block0_conv2_bn_gamma) \
    X(velocity_layer4_block0_conv2_bn_beta) \
    X(velocity_layer4_block0_proj_weight) \
    X(velocity_layer4_block0_proj_bias) \
    X(velocity_layer4_block0_proj_bn_gamma) \
    X(velocity_layer4_block0_proj_bn_beta) \
    X(velocity_layer4_block1_conv1_weight) \
    X(velocity_layer4_block1_conv1_bias) \
    X(velocity_layer4_block1_conv1_bn_gamma) \
    X(velocity_layer4_block1_conv1_bn_beta) \
    X(velocity_layer4_block1_conv2_weight) \
    X(velocity_layer4_block1_conv2_bias) \
    X(velocity_layer4_block1_conv2_bn_gamma) \
    X(velocity_layer4_block1_conv2_bn_beta) \
    X(velocity_fc_weight) \
    X(velocity_fc_bias)

#define RESNET18_GRAD_ARRAYS(X) \
    X(stem_weight) \
    X(stem_bias) \
    X(stem_bn_gamma) \
    X(stem_bn_beta) \
    X(layer1_block0_conv1_weight) \
    X(layer1_block0_conv1_bias) \
    X(layer1_block0_conv1_bn_gamma) \
    X(layer1_block0_conv1_bn_beta) \
    X(layer1_block0_conv2_weight) \
    X(layer1_block0_conv2_bias) \
    X(layer1_block0_conv2_bn_gamma) \
    X(layer1_block0_conv2_bn_beta) \
    X(layer1_block1_conv1_weight) \
    X(layer1_block1_conv1_bias) \
    X(layer1_block1_conv1_bn_gamma) \
    X(layer1_block1_conv1_bn_beta) \
    X(layer1_block1_conv2_weight) \
    X(layer1_block1_conv2_bias) \
    X(layer1_block1_conv2_bn_gamma) \
    X(layer1_block1_conv2_bn_beta) \
    X(layer2_block0_conv1_weight) \
    X(layer2_block0_conv1_bias) \
    X(layer2_block0_conv1_bn_gamma) \
    X(layer2_block0_conv1_bn_beta) \
    X(layer2_block0_conv2_weight) \
    X(layer2_block0_conv2_bias) \
    X(layer2_block0_conv2_bn_gamma) \
    X(layer2_block0_conv2_bn_beta) \
    X(layer2_block0_proj_weight) \
    X(layer2_block0_proj_bias) \
    X(layer2_block0_proj_bn_gamma) \
    X(layer2_block0_proj_bn_beta) \
    X(layer2_block1_conv1_weight) \
    X(layer2_block1_conv1_bias) \
    X(layer2_block1_conv1_bn_gamma) \
    X(layer2_block1_conv1_bn_beta) \
    X(layer2_block1_conv2_weight) \
    X(layer2_block1_conv2_bias) \
    X(layer2_block1_conv2_bn_gamma) \
    X(layer2_block1_conv2_bn_beta) \
    X(layer3_block0_conv1_weight) \
    X(layer3_block0_conv1_bias) \
    X(layer3_block0_conv1_bn_gamma) \
    X(layer3_block0_conv1_bn_beta) \
    X(layer3_block0_conv2_weight) \
    X(layer3_block0_conv2_bias) \
    X(layer3_block0_conv2_bn_gamma) \
    X(layer3_block0_conv2_bn_beta) \
    X(layer3_block0_proj_weight) \
    X(layer3_block0_proj_bias) \
    X(layer3_block0_proj_bn_gamma) \
    X(layer3_block0_proj_bn_beta) \
    X(layer3_block1_conv1_weight) \
    X(layer3_block1_conv1_bias) \
    X(layer3_block1_conv1_bn_gamma) \
    X(layer3_block1_conv1_bn_beta) \
    X(layer3_block1_conv2_weight) \
    X(layer3_block1_conv2_bias) \
    X(layer3_block1_conv2_bn_gamma) \
    X(layer3_block1_conv2_bn_beta) \
    X(layer4_block0_conv1_weight) \
    X(layer4_block0_conv1_bias) \
    X(layer4_block0_conv1_bn_gamma) \
    X(layer4_block0_conv1_bn_beta) \
    X(layer4_block0_conv2_weight) \
    X(layer4_block0_conv2_bias) \
    X(layer4_block0_conv2_bn_gamma) \
    X(layer4_block0_conv2_bn_beta) \
    X(layer4_block0_proj_weight) \
    X(layer4_block0_proj_bias) \
    X(layer4_block0_proj_bn_gamma) \
    X(layer4_block0_proj_bn_beta) \
    X(layer4_block1_conv1_weight) \
    X(layer4_block1_conv1_bias) \
    X(layer4_block1_conv1_bn_gamma) \
    X(layer4_block1_conv1_bn_beta) \
    X(layer4_block1_conv2_weight) \
    X(layer4_block1_conv2_bias) \
    X(layer4_block1_conv2_bn_gamma) \
    X(layer4_block1_conv2_bn_beta) \
    X(fc_weight) \
    X(fc_bias)

struct ResNet18Params {
    float stem_weight[RESNET18_STEM_CHANNELS * RESNET18_IMAGE_CHANNELS * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE];
    float stem_bias[RESNET18_STEM_CHANNELS];
    RESNET18_BN_AFFINE_PARAMS(stem, RESNET18_STEM_CHANNELS);
    RESNET18_BLOCK_PARAMS(layer1_block0, RESNET18_STAGE1_CHANNELS, RESNET18_STAGE1_CHANNELS);
    RESNET18_BN_AFFINE_PARAMS(layer1_block0_conv1, RESNET18_STAGE1_CHANNELS);
    RESNET18_BN_AFFINE_PARAMS(layer1_block0_conv2, RESNET18_STAGE1_CHANNELS);
    RESNET18_BLOCK_PARAMS(layer1_block1, RESNET18_STAGE1_CHANNELS, RESNET18_STAGE1_CHANNELS);
    RESNET18_BN_AFFINE_PARAMS(layer1_block1_conv1, RESNET18_STAGE1_CHANNELS);
    RESNET18_BN_AFFINE_PARAMS(layer1_block1_conv2, RESNET18_STAGE1_CHANNELS);
    RESNET18_BLOCK_PARAMS(layer2_block0, RESNET18_STAGE1_CHANNELS, RESNET18_STAGE2_CHANNELS);
    RESNET18_BN_AFFINE_PARAMS(layer2_block0_conv1, RESNET18_STAGE2_CHANNELS);
    RESNET18_BN_AFFINE_PARAMS(layer2_block0_conv2, RESNET18_STAGE2_CHANNELS);
    RESNET18_PROJ_PARAMS(layer2_block0, RESNET18_STAGE1_CHANNELS, RESNET18_STAGE2_CHANNELS);
    RESNET18_BN_AFFINE_PARAMS(layer2_block0_proj, RESNET18_STAGE2_CHANNELS);
    RESNET18_BLOCK_PARAMS(layer2_block1, RESNET18_STAGE2_CHANNELS, RESNET18_STAGE2_CHANNELS);
    RESNET18_BN_AFFINE_PARAMS(layer2_block1_conv1, RESNET18_STAGE2_CHANNELS);
    RESNET18_BN_AFFINE_PARAMS(layer2_block1_conv2, RESNET18_STAGE2_CHANNELS);
    RESNET18_BLOCK_PARAMS(layer3_block0, RESNET18_STAGE2_CHANNELS, RESNET18_STAGE3_CHANNELS);
    RESNET18_BN_AFFINE_PARAMS(layer3_block0_conv1, RESNET18_STAGE3_CHANNELS);
    RESNET18_BN_AFFINE_PARAMS(layer3_block0_conv2, RESNET18_STAGE3_CHANNELS);
    RESNET18_PROJ_PARAMS(layer3_block0, RESNET18_STAGE2_CHANNELS, RESNET18_STAGE3_CHANNELS);
    RESNET18_BN_AFFINE_PARAMS(layer3_block0_proj, RESNET18_STAGE3_CHANNELS);
    RESNET18_BLOCK_PARAMS(layer3_block1, RESNET18_STAGE3_CHANNELS, RESNET18_STAGE3_CHANNELS);
    RESNET18_BN_AFFINE_PARAMS(layer3_block1_conv1, RESNET18_STAGE3_CHANNELS);
    RESNET18_BN_AFFINE_PARAMS(layer3_block1_conv2, RESNET18_STAGE3_CHANNELS);
    RESNET18_BLOCK_PARAMS(layer4_block0, RESNET18_STAGE3_CHANNELS, RESNET18_STAGE4_CHANNELS);
    RESNET18_BN_AFFINE_PARAMS(layer4_block0_conv1, RESNET18_STAGE4_CHANNELS);
    RESNET18_BN_AFFINE_PARAMS(layer4_block0_conv2, RESNET18_STAGE4_CHANNELS);
    RESNET18_PROJ_PARAMS(layer4_block0, RESNET18_STAGE3_CHANNELS, RESNET18_STAGE4_CHANNELS);
    RESNET18_BN_AFFINE_PARAMS(layer4_block0_proj, RESNET18_STAGE4_CHANNELS);
    RESNET18_BLOCK_PARAMS(layer4_block1, RESNET18_STAGE4_CHANNELS, RESNET18_STAGE4_CHANNELS);
    RESNET18_BN_AFFINE_PARAMS(layer4_block1_conv1, RESNET18_STAGE4_CHANNELS);
    RESNET18_BN_AFFINE_PARAMS(layer4_block1_conv2, RESNET18_STAGE4_CHANNELS);
    float fc_weight[RESNET18_NUM_CLASSES * RESNET18_STAGE4_CHANNELS];
    float fc_bias[RESNET18_NUM_CLASSES];
};

struct ResNet18State {
    float velocity_stem_weight[RESNET18_STEM_CHANNELS * RESNET18_IMAGE_CHANNELS * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE];
    float velocity_stem_bias[RESNET18_STEM_CHANNELS];
    RESNET18_BN_AFFINE_STATE(stem, RESNET18_STEM_CHANNELS);
    RESNET18_BLOCK_STATE(layer1_block0, RESNET18_STAGE1_CHANNELS, RESNET18_STAGE1_CHANNELS);
    RESNET18_BN_AFFINE_STATE(layer1_block0_conv1, RESNET18_STAGE1_CHANNELS);
    RESNET18_BN_AFFINE_STATE(layer1_block0_conv2, RESNET18_STAGE1_CHANNELS);
    RESNET18_BLOCK_STATE(layer1_block1, RESNET18_STAGE1_CHANNELS, RESNET18_STAGE1_CHANNELS);
    RESNET18_BN_AFFINE_STATE(layer1_block1_conv1, RESNET18_STAGE1_CHANNELS);
    RESNET18_BN_AFFINE_STATE(layer1_block1_conv2, RESNET18_STAGE1_CHANNELS);
    RESNET18_BLOCK_STATE(layer2_block0, RESNET18_STAGE1_CHANNELS, RESNET18_STAGE2_CHANNELS);
    RESNET18_BN_AFFINE_STATE(layer2_block0_conv1, RESNET18_STAGE2_CHANNELS);
    RESNET18_BN_AFFINE_STATE(layer2_block0_conv2, RESNET18_STAGE2_CHANNELS);
    RESNET18_PROJ_STATE(layer2_block0, RESNET18_STAGE1_CHANNELS, RESNET18_STAGE2_CHANNELS);
    RESNET18_BN_AFFINE_STATE(layer2_block0_proj, RESNET18_STAGE2_CHANNELS);
    RESNET18_BLOCK_STATE(layer2_block1, RESNET18_STAGE2_CHANNELS, RESNET18_STAGE2_CHANNELS);
    RESNET18_BN_AFFINE_STATE(layer2_block1_conv1, RESNET18_STAGE2_CHANNELS);
    RESNET18_BN_AFFINE_STATE(layer2_block1_conv2, RESNET18_STAGE2_CHANNELS);
    RESNET18_BLOCK_STATE(layer3_block0, RESNET18_STAGE2_CHANNELS, RESNET18_STAGE3_CHANNELS);
    RESNET18_BN_AFFINE_STATE(layer3_block0_conv1, RESNET18_STAGE3_CHANNELS);
    RESNET18_BN_AFFINE_STATE(layer3_block0_conv2, RESNET18_STAGE3_CHANNELS);
    RESNET18_PROJ_STATE(layer3_block0, RESNET18_STAGE2_CHANNELS, RESNET18_STAGE3_CHANNELS);
    RESNET18_BN_AFFINE_STATE(layer3_block0_proj, RESNET18_STAGE3_CHANNELS);
    RESNET18_BLOCK_STATE(layer3_block1, RESNET18_STAGE3_CHANNELS, RESNET18_STAGE3_CHANNELS);
    RESNET18_BN_AFFINE_STATE(layer3_block1_conv1, RESNET18_STAGE3_CHANNELS);
    RESNET18_BN_AFFINE_STATE(layer3_block1_conv2, RESNET18_STAGE3_CHANNELS);
    RESNET18_BLOCK_STATE(layer4_block0, RESNET18_STAGE3_CHANNELS, RESNET18_STAGE4_CHANNELS);
    RESNET18_BN_AFFINE_STATE(layer4_block0_conv1, RESNET18_STAGE4_CHANNELS);
    RESNET18_BN_AFFINE_STATE(layer4_block0_conv2, RESNET18_STAGE4_CHANNELS);
    RESNET18_PROJ_STATE(layer4_block0, RESNET18_STAGE3_CHANNELS, RESNET18_STAGE4_CHANNELS);
    RESNET18_BN_AFFINE_STATE(layer4_block0_proj, RESNET18_STAGE4_CHANNELS);
    RESNET18_BLOCK_STATE(layer4_block1, RESNET18_STAGE4_CHANNELS, RESNET18_STAGE4_CHANNELS);
    RESNET18_BN_AFFINE_STATE(layer4_block1_conv1, RESNET18_STAGE4_CHANNELS);
    RESNET18_BN_AFFINE_STATE(layer4_block1_conv2, RESNET18_STAGE4_CHANNELS);
    float velocity_fc_weight[RESNET18_NUM_CLASSES * RESNET18_STAGE4_CHANNELS];
    float velocity_fc_bias[RESNET18_NUM_CLASSES];
};

struct ResNet18Gradients {
    float stem_weight[RESNET18_STEM_CHANNELS * RESNET18_IMAGE_CHANNELS * RESNET18_KERNEL_SIZE * RESNET18_KERNEL_SIZE];
    float stem_bias[RESNET18_STEM_CHANNELS];
    RESNET18_BN_AFFINE_GRADS(stem, RESNET18_STEM_CHANNELS);
    RESNET18_BLOCK_GRADS(layer1_block0, RESNET18_STAGE1_CHANNELS, RESNET18_STAGE1_CHANNELS);
    RESNET18_BN_AFFINE_GRADS(layer1_block0_conv1, RESNET18_STAGE1_CHANNELS);
    RESNET18_BN_AFFINE_GRADS(layer1_block0_conv2, RESNET18_STAGE1_CHANNELS);
    RESNET18_BLOCK_GRADS(layer1_block1, RESNET18_STAGE1_CHANNELS, RESNET18_STAGE1_CHANNELS);
    RESNET18_BN_AFFINE_GRADS(layer1_block1_conv1, RESNET18_STAGE1_CHANNELS);
    RESNET18_BN_AFFINE_GRADS(layer1_block1_conv2, RESNET18_STAGE1_CHANNELS);
    RESNET18_BLOCK_GRADS(layer2_block0, RESNET18_STAGE1_CHANNELS, RESNET18_STAGE2_CHANNELS);
    RESNET18_BN_AFFINE_GRADS(layer2_block0_conv1, RESNET18_STAGE2_CHANNELS);
    RESNET18_BN_AFFINE_GRADS(layer2_block0_conv2, RESNET18_STAGE2_CHANNELS);
    RESNET18_PROJ_GRADS(layer2_block0, RESNET18_STAGE1_CHANNELS, RESNET18_STAGE2_CHANNELS);
    RESNET18_BN_AFFINE_GRADS(layer2_block0_proj, RESNET18_STAGE2_CHANNELS);
    RESNET18_BLOCK_GRADS(layer2_block1, RESNET18_STAGE2_CHANNELS, RESNET18_STAGE2_CHANNELS);
    RESNET18_BN_AFFINE_GRADS(layer2_block1_conv1, RESNET18_STAGE2_CHANNELS);
    RESNET18_BN_AFFINE_GRADS(layer2_block1_conv2, RESNET18_STAGE2_CHANNELS);
    RESNET18_BLOCK_GRADS(layer3_block0, RESNET18_STAGE2_CHANNELS, RESNET18_STAGE3_CHANNELS);
    RESNET18_BN_AFFINE_GRADS(layer3_block0_conv1, RESNET18_STAGE3_CHANNELS);
    RESNET18_BN_AFFINE_GRADS(layer3_block0_conv2, RESNET18_STAGE3_CHANNELS);
    RESNET18_PROJ_GRADS(layer3_block0, RESNET18_STAGE2_CHANNELS, RESNET18_STAGE3_CHANNELS);
    RESNET18_BN_AFFINE_GRADS(layer3_block0_proj, RESNET18_STAGE3_CHANNELS);
    RESNET18_BLOCK_GRADS(layer3_block1, RESNET18_STAGE3_CHANNELS, RESNET18_STAGE3_CHANNELS);
    RESNET18_BN_AFFINE_GRADS(layer3_block1_conv1, RESNET18_STAGE3_CHANNELS);
    RESNET18_BN_AFFINE_GRADS(layer3_block1_conv2, RESNET18_STAGE3_CHANNELS);
    RESNET18_BLOCK_GRADS(layer4_block0, RESNET18_STAGE3_CHANNELS, RESNET18_STAGE4_CHANNELS);
    RESNET18_BN_AFFINE_GRADS(layer4_block0_conv1, RESNET18_STAGE4_CHANNELS);
    RESNET18_BN_AFFINE_GRADS(layer4_block0_conv2, RESNET18_STAGE4_CHANNELS);
    RESNET18_PROJ_GRADS(layer4_block0, RESNET18_STAGE3_CHANNELS, RESNET18_STAGE4_CHANNELS);
    RESNET18_BN_AFFINE_GRADS(layer4_block0_proj, RESNET18_STAGE4_CHANNELS);
    RESNET18_BLOCK_GRADS(layer4_block1, RESNET18_STAGE4_CHANNELS, RESNET18_STAGE4_CHANNELS);
    RESNET18_BN_AFFINE_GRADS(layer4_block1_conv1, RESNET18_STAGE4_CHANNELS);
    RESNET18_BN_AFFINE_GRADS(layer4_block1_conv2, RESNET18_STAGE4_CHANNELS);
    float fc_weight[RESNET18_NUM_CLASSES * RESNET18_STAGE4_CHANNELS];
    float fc_bias[RESNET18_NUM_CLASSES];
};

struct ResNet18Activations {
    float stem_out[RESNET18_STAGE1_CHANNELS * RESNET18_STAGE1_HEIGHT * RESNET18_STAGE1_WIDTH];
    float layer1_block0_conv1_out[RESNET18_STAGE1_CHANNELS * RESNET18_STAGE1_HEIGHT * RESNET18_STAGE1_WIDTH];
    float layer1_block0_out[RESNET18_STAGE1_CHANNELS * RESNET18_STAGE1_HEIGHT * RESNET18_STAGE1_WIDTH];
    float layer1_block1_conv1_out[RESNET18_STAGE1_CHANNELS * RESNET18_STAGE1_HEIGHT * RESNET18_STAGE1_WIDTH];
    float layer1_block1_out[RESNET18_STAGE1_CHANNELS * RESNET18_STAGE1_HEIGHT * RESNET18_STAGE1_WIDTH];
    float layer2_block0_conv1_out[RESNET18_STAGE2_CHANNELS * RESNET18_STAGE2_HEIGHT * RESNET18_STAGE2_WIDTH];
    float layer2_block0_out[RESNET18_STAGE2_CHANNELS * RESNET18_STAGE2_HEIGHT * RESNET18_STAGE2_WIDTH];
    float layer2_block1_conv1_out[RESNET18_STAGE2_CHANNELS * RESNET18_STAGE2_HEIGHT * RESNET18_STAGE2_WIDTH];
    float layer2_block1_out[RESNET18_STAGE2_CHANNELS * RESNET18_STAGE2_HEIGHT * RESNET18_STAGE2_WIDTH];
    float layer3_block0_conv1_out[RESNET18_STAGE3_CHANNELS * RESNET18_STAGE3_HEIGHT * RESNET18_STAGE3_WIDTH];
    float layer3_block0_out[RESNET18_STAGE3_CHANNELS * RESNET18_STAGE3_HEIGHT * RESNET18_STAGE3_WIDTH];
    float layer3_block1_conv1_out[RESNET18_STAGE3_CHANNELS * RESNET18_STAGE3_HEIGHT * RESNET18_STAGE3_WIDTH];
    float layer3_block1_out[RESNET18_STAGE3_CHANNELS * RESNET18_STAGE3_HEIGHT * RESNET18_STAGE3_WIDTH];
    float layer4_block0_conv1_out[RESNET18_STAGE4_CHANNELS * RESNET18_STAGE4_HEIGHT * RESNET18_STAGE4_WIDTH];
    float layer4_block0_out[RESNET18_STAGE4_CHANNELS * RESNET18_STAGE4_HEIGHT * RESNET18_STAGE4_WIDTH];
    float layer4_block1_conv1_out[RESNET18_STAGE4_CHANNELS * RESNET18_STAGE4_HEIGHT * RESNET18_STAGE4_WIDTH];
    float layer4_block1_out[RESNET18_STAGE4_CHANNELS * RESNET18_STAGE4_HEIGHT * RESNET18_STAGE4_WIDTH];
    float pooled[RESNET18_STAGE4_CHANNELS];
    float logits[RESNET18_NUM_CLASSES];
};

struct ResNet18StepResult {
    float loss;
    int predicted_label;
};

struct BatchNormStats {
    float mean[RESNET18_STAGE4_CHANNELS];
    float inv_std[RESNET18_STAGE4_CHANNELS];
};

struct ResNet18BatchNormCache {
    BatchNormStats stem;
    BatchNormStats layer1_block0_conv1;
    BatchNormStats layer1_block0_conv2;
    BatchNormStats layer1_block1_conv1;
    BatchNormStats layer1_block1_conv2;
    BatchNormStats layer2_block0_conv1;
    BatchNormStats layer2_block0_conv2;
    BatchNormStats layer2_block0_proj;
    BatchNormStats layer2_block1_conv1;
    BatchNormStats layer2_block1_conv2;
    BatchNormStats layer3_block0_conv1;
    BatchNormStats layer3_block0_conv2;
    BatchNormStats layer3_block0_proj;
    BatchNormStats layer3_block1_conv1;
    BatchNormStats layer3_block1_conv2;
    BatchNormStats layer4_block0_conv1;
    BatchNormStats layer4_block0_conv2;
    BatchNormStats layer4_block0_proj;
    BatchNormStats layer4_block1_conv1;
    BatchNormStats layer4_block1_conv2;
};

struct ResNet18BatchTrainingWorkspace {
    float batch_scratch0[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS];
    float batch_scratch1[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS];
    float batch_scratch2[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS];
    float batch_scratch3[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS];
    float batch_grad0[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS];
    float batch_grad1[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS];
    float batch_tmp0[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS];
    float batch_tmp1[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS];
    float batch_tmp2[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS];
    float batch_tmp3[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS];
    ResNet18Activations activations[RESNET18_BATCH_SIZE];
    ResNet18Gradients batch_gradients;
    ResNet18BatchNormCache batchnorm_cache;
};

void init_resnet18_params(ResNet18Params &params);
void reset_resnet18_state(ResNet18State &state);
void zero_resnet18_gradients(ResNet18Gradients &gradients);

void resnet18_forward(
    const float image[RESNET18_INPUT_SIZE],
    const ResNet18Params &params,
    ResNet18Activations &activations
);

float resnet18_softmax_cross_entropy(
    const float logits[RESNET18_NUM_CLASSES],
    int label,
    float grad_logits[RESNET18_NUM_CLASSES]
);

void resnet18_backward(
    const float image[RESNET18_INPUT_SIZE],
    int label,
    const ResNet18Params &params,
    const ResNet18Activations &activations,
    ResNet18Gradients &gradients
);

void resnet18_sgd_momentum_update(
    ResNet18Params &params,
    ResNet18State &state,
    const ResNet18Gradients &gradients,
    float learning_rate,
    float momentum
);

void resnet18_train_step(
    const float image[RESNET18_INPUT_SIZE],
    int label,
    float learning_rate,
    float momentum,
    ResNet18Params &params,
    ResNet18State &state,
    float logits[RESNET18_NUM_CLASSES],
    ResNet18StepResult &result
);

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
);

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
);

void quantize_resnet18_params_bm8(
    const ResNet18Params &input,
    ResNet18Params &output
);

void quantize_resnet18_state_bm8(
    const ResNet18State &input,
    ResNet18State &output
);

void quantize_resnet18_gradients_bm8(
    const ResNet18Gradients &input,
    ResNet18Gradients &output
);

void quantize_resnet18_logits_bm8(
    const float input[RESNET18_NUM_CLASSES],
    float output[RESNET18_NUM_CLASSES]
);

void resnet18_train_step_bm8(
    const float image[RESNET18_INPUT_SIZE],
    int label,
    float learning_rate,
    float momentum,
    ResNet18Params &params,
    ResNet18State &state,
    float logits[RESNET18_NUM_CLASSES],
    ResNet18StepResult &result
);

void resnet18_train_batch_bm8(
    const float images[RESNET18_BATCH_SIZE][RESNET18_INPUT_SIZE],
    const int labels[RESNET18_BATCH_SIZE],
    float learning_rate,
    float momentum,
    ResNet18Params &params,
    ResNet18State &state,
    float logits[RESNET18_BATCH_SIZE][RESNET18_NUM_CLASSES],
    ResNet18StepResult batch_results[RESNET18_BATCH_SIZE],
    float &average_loss
);

void resnet18_train_batch_mixed_l3_bm8(
    const float images[RESNET18_BATCH_SIZE][RESNET18_INPUT_SIZE],
    const int labels[RESNET18_BATCH_SIZE],
    float learning_rate,
    float momentum,
    ResNet18Params &params,
    ResNet18State &state,
    float logits[RESNET18_BATCH_SIZE][RESNET18_NUM_CLASSES],
    ResNet18StepResult batch_results[RESNET18_BATCH_SIZE],
    float &average_loss
);

void resnet18_basic_block_forward_bench_l3(
    const float input_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    ResNet18Params &params,
    ResNet18BatchTrainingWorkspace &workspace,
    float output_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    float &checksum
);

void resnet18_basic_block_backward_bench_l3(
    const float input_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    float conv1_out_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    float block_output_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    float grad_output_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    ResNet18Params &params,
    ResNet18BatchTrainingWorkspace &workspace,
    float grad_input_batch[RESNET18_BATCH_SIZE][RESNET18_MAX_ACTIVATION_ELEMENTS],
    float &checksum
);

#endif
