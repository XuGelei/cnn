//
// Created by gracexu on 2020/12/31.
//

#ifndef CNN3_HUMAN_DETECT_CNN_H
#define CNN3_HUMAN_DETECT_CNN_H
#include <omp.h>
//#pragma GCC optimize(3)

#include <exception>
#include <iostream>
#include "cnn_param.h"

class CnnMatrix {
public:
    float Get(int channel, int row, int col) const {
        if (row < 0 || row > rows_ - 1 || col < 0 || col > cols_ - 1) {
            return 0;
        }
        return data_[channel * cols_ * rows_ + row * cols_ + col];
    }

    float &Set(float val, int channel, int row, int col) const {
        return (data_[channel * cols_ * rows_ + row * cols_ + col] = val);
    }

    int Total() const {
        return channels_ * rows_ * cols_;
    }

    void Destroy() const {
        delete [] data_;
    }

    void init(int channels, int rows, int cols, float *data = nullptr) {
        channels_ = channels;
        rows_ = rows;
        cols_ = cols;
        data_ = data;
    }
    void print(){
        for(int i=0;i<channels_*rows_*cols_;++i) std::cout<<data_[i]<<" ";
    }
    float PartConv(const ConvParam &param, int i, int j, int k) {
        float res = 0;
        for (int l = 0; l < channels_; ++l) {
            for (int m = 0; m < param.kernel_size; ++m) {
                for (int n = 0; n < param.kernel_size; ++n) {
                    res += Get(l, j * param.stride - param.pad + m, k * param.stride - param.pad + n) *
                           param.p_weight[i * channels_ * param.kernel_size * param.kernel_size +
                                          l * param.kernel_size * param.kernel_size + m * param.kernel_size + n];
                }
            }
        }
        return res;
    }
    CnnMatrix  operator*( ConvParam param){
        CnnMatrix out;
        out.init(param.out_channels, ((rows_ + 2 * param.pad - param.kernel_size) / param.stride + 1),
                 ((cols_ + 2 * param.pad - param.kernel_size) / param.stride + 1));
        out.data_ = new float [out.Total()];
//#pragma omp parallel for

        for (int i = 0; i < out.channels_; ++i) {
            for (int j = 0; j < out.rows_; ++j) {
                for (int k = 0; k < out.cols_; ++k) {
                    out.Set(PartConv( param, i, j, k) + param.p_bias[i], i, j, k);
                }
            }
        }
        return out;
    }
    int channels_;
    int cols_;
    int rows_;
    float *data_;
};

class InputSizeException : public std::exception {
    const char *what() const noexcept override {
        return "Input size is not 3x128x128!";
    }
};

float GetConfidenceScore128x128rbg(float *rgb_arr, int rows, int cols);

#endif //CNN3_HUMAN_DETECT_CNN_H