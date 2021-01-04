#include "human_detect_cnn.h"
#include "cnn_param.h"
#include <numeric>
//#pragma GCC optimize(3)

#include <cmath>
#include <math.h>
#include <cstring>
#include <iostream>
#include <chrono>
#define timePrint  std::cout\
<< "Slow calculations took "\
<< std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "μs ≈ "\
<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms ≈ "\
<< std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s.\n";

//float PartConv(const CnnMatrix &mat, const ConvParam &param, int i, int j, int k) {
//    float res = 0;
//    for (int l = 0; l < mat.channels_; ++l) {
//        for (int m = 0; m < param.kernel_size; ++m) {
//            for (int n = 0; n < param.kernel_size; ++n) {
//                res += mat.Get(l, j * param.stride - param.pad + m, k * param.stride - param.pad + n) *
//                       param.p_weight[i * mat.channels_ * param.kernel_size * param.kernel_size +
//                                      l * param.kernel_size * param.kernel_size + m * param.kernel_size + n];
//            }
//        }
//    }
//    return res;
//}
//
//void ConvLayer(const CnnMatrix &in, CnnMatrix &out, const ConvParam &param) {
//    out.init(param.out_channels, ((in.rows_ + 2 * param.pad - param.kernel_size) / param.stride + 1),
//             ((in.cols_ + 2 * param.pad - param.kernel_size) / param.stride + 1));
//    out.data_ = new float [out.Total()];
//    for (int i = 0; i < out.channels_; ++i) {
//        for (int j = 0; j < out.rows_; ++j) {
//            for (int k = 0; k < out.cols_; ++k) {
//                out.Set(PartConv(in, param, i, j, k) + param.p_bias[i], i, j, k);
//            }
//        }
//    }
//}

void Relu(CnnMatrix &mat) {
    for (int i = 0; i < mat.Total(); ++i) {
        if (mat.data_[i] < 0) {
            mat.data_[i] = 0;
        }
    }
}

inline float Max(float a, float b) {
    return a > b ? a : b;
}

float PartPooling(const CnnMatrix &mat, int i, int j, int k) {
    float res = Max(mat.Get(i, j * 2, k * 2), mat.Get(i, j * 2, k * 2 + 1));
    res = Max(res, mat.Get(i, j * 2 + 1, k * 2));
    res = Max(res, mat.Get(i, j * 2 + 1, k * 2 + 1));
    return res;
}

void MaxPoolingLayer(const CnnMatrix &in, CnnMatrix &out) {
    out.init(in.channels_, in.rows_ / 2, in.cols_ / 2);
    out.data_ = new float [out.Total()];
    for (int i = 0; i < out.channels_; ++i) {
        for (int j = 0; j < out.rows_; ++j) {
            for (int k = 0; k < out.cols_; ++k) {
                out.Set(PartPooling(in, i, j, k), i, j, k);
            }
        }
    }
    Relu(out);
}

float DotProduct(const CnnMatrix &in, const FcParam &param, int index) {
 //   std::cout<<in.data_<<" "<<in.Total()<<std::endl;
    return std::inner_product(in.data_, &in.data_[in.Total()], &param.p_weight[index * in.Total()], 0.0f);
}

void FcLayer(const CnnMatrix &in, CnnMatrix &out, const FcParam &param) {
    out.init(param.out_features, 1, 1);
    out.data_ = new float [out.Total()];
    for (int i = 0; i < out.channels_; ++i) {
        out.Set(DotProduct(in, param, i), i, 0, 0);
    }
//    float res1=0,res2=0;
//    for(int i=0;i<8*8*32;++i) {
//        res1+=in.data_[i]*param.p_weight[i];
//    }
//    for(int i=8*8*32;i<2*8*8*32;++i) {
//        res2+=in.data_[i-2048]*param.p_weight[i];
//    }

}

float GetScore(CnnMatrix &mat, int index) {
    float res = 0;
    for (int i = 0; i < mat.Total(); ++i) {
        res += std::exp(mat.Get(i, 0, 0));
//

    }
    return (std::exp(mat.Get(index, 0, 0)) / res);
}

float GetConfidenceScore128x128rbg(float *rgb_arr, int rows, int cols) {
    if (rows != 128 && cols != 128) {
        throw InputSizeException();
    }
     CnnMatrix origin{3, rows, cols, rgb_arr};
    CnnMatrix mats[6];
    extern ConvParam conv_params[3];
    extern FcParam fc_params[1];
    auto start = std::chrono::steady_clock::now();

    mats[0]=origin*conv_params[0];
    Relu(mats[0]);
    MaxPoolingLayer(mats[0], mats[1]);
    mats[2]=mats[1]*conv_params[1];
    Relu(mats[2]);
    MaxPoolingLayer(mats[2], mats[3]);
    mats[4]=mats[3]*conv_params[2];
    Relu(mats[4]);
    FcLayer(mats[4], mats[5], fc_params[0]);
    auto end = std::chrono::steady_clock::now();
    timePrint
    float res = GetScore(mats[5], 1);
    for (const auto & mat : mats) {
        mat.Destroy();
    }
    return res;
}