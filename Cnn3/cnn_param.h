//
// Created by gracexu on 2020/12/31.
//

#ifndef CNN3_CNN_PARAM_H
#define CNN3_CNN_PARAM_H
//#pragma GCC optimize(3)


struct ConvParam {
    int pad;
    int stride;
    int kernel_size;
    int in_channels;
    int out_channels;
    float* p_weight;
    float* p_bias;
};

struct FcParam {
    int in_features;
    int out_features;
    float* p_weight;
    float* p_bias;
};

#endif //CNN3_CNN_PARAM_H
