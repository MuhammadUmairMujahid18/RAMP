#ifndef BM8_EUROSAT_DATA_H
#define BM8_EUROSAT_DATA_H

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "resnet18_cifar.h"

#define EUROSAT_TRAIN_IMAGES 27000
#define EUROSAT_ONE_PERCENT_IMAGES (EUROSAT_TRAIN_IMAGES / 100)
#define EUROSAT_RECORD_BYTES (1 + RESNET18_INPUT_SIZE)
#define EUROSAT_BINARY_MAGIC 0x45555354u

struct EurosatSample {
    float image[RESNET18_INPUT_SIZE];
    int label;
};

static unsigned int eurosat_read_be_u32(std::ifstream &file) {
    unsigned char bytes[4];
    file.read((char *)bytes, 4);
    return ((unsigned int)bytes[0] << 24) |
           ((unsigned int)bytes[1] << 16) |
           ((unsigned int)bytes[2] << 8) |
           (unsigned int)bytes[3];
}

static bool eurosat_file_exists(const std::string &path) {
    std::ifstream file(path.c_str(), std::ios::binary);
    return file.good();
}

static std::string resolve_eurosat_binary_path() {
    const char *candidates[] = {
        "datasets/eurosat_train_32x32.bin",
        "../fixed_precision_BM8/datasets/eurosat_train_32x32.bin",
        "../fp32_eurosat_resnet18/datasets/eurosat_train_32x32.bin",
        "../../fp32_eurosat_resnet18/datasets/eurosat_train_32x32.bin",
        "../../../fp32_eurosat_resnet18/datasets/eurosat_train_32x32.bin",
        "C:/Users/muma7778/Documents/postdoc/bm4_hls/experiments/fixed_precision_BM8/datasets/eurosat_train_32x32.bin",
        "C:/Users/muma7778/Documents/postdoc/bm4_hls/experiments/fp32_eurosat_resnet18/datasets/eurosat_train_32x32.bin"
    };

    for (int i = 0; i < (int)(sizeof(candidates) / sizeof(candidates[0])); ++i) {
        if (eurosat_file_exists(candidates[i])) {
            return std::string(candidates[i]);
        }
    }
    return std::string();
}

static void load_eurosat_one_percent(EurosatSample samples[EUROSAT_ONE_PERCENT_IMAGES]) {
    static bool loaded = false;
    if (loaded) {
        return;
    }

    std::string path = resolve_eurosat_binary_path();
    if (path.empty()) {
        std::cerr << "Could not locate EuroSAT binary." << std::endl;
        std::cerr << "Run: python host_tools\\preprocess_eurosat.py --download" << std::endl;
        std::cerr << "Expected local output: datasets/eurosat_train_32x32.bin" << std::endl;
        std::exit(1);
    }

    std::ifstream file(path.c_str(), std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open EuroSAT file: " << path << std::endl;
        std::exit(1);
    }

    unsigned int magic = eurosat_read_be_u32(file);
    unsigned int sample_count = eurosat_read_be_u32(file);
    if (magic != EUROSAT_BINARY_MAGIC) {
        std::cerr << "EuroSAT binary header mismatch." << std::endl;
        std::exit(1);
    }
    if (sample_count < EUROSAT_ONE_PERCENT_IMAGES) {
        std::cerr << "EuroSAT binary does not contain enough samples." << std::endl;
        std::exit(1);
    }

    unsigned char record[EUROSAT_RECORD_BYTES];
    for (int sample_index = 0; sample_index < EUROSAT_ONE_PERCENT_IMAGES; ++sample_index) {
        file.read((char *)record, EUROSAT_RECORD_BYTES);
        if (file.gcount() != EUROSAT_RECORD_BYTES) {
            std::cerr << "Failed to read EuroSAT sample " << sample_index << std::endl;
            std::exit(1);
        }

        samples[sample_index].label = (int)record[0];
        for (int i = 0; i < RESNET18_INPUT_SIZE; ++i) {
            samples[sample_index].image[i] = (((float)record[i + 1]) / 255.0f - 0.5f) * 2.0f;
        }
    }

    loaded = true;
}

#endif
