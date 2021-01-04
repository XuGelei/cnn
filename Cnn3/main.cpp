//#pragma GCC optimize(3)
#include <iostream>
#include "human_detect_cnn.h"
#include <vector>
#include <chrono>
#include "opencv2/opencv.hpp"
//#include <io.h>
#define timePrint  std::cout\
<< "Slow calculations took "\
<< std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "μs ≈ "\
<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms ≈ "\
<< std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s.\n";

std::string GetFileName(std::string full_path) {
    full_path.replace(0, full_path.find_last_of('\\') + 1, "");
    return full_path;
}

//Get score by image in given full path and Print
void PrintImage(const std::string &name) {
    using namespace cv;
    Mat mat = imread(name, IMREAD_COLOR);
    std::vector<Mat> img_channels;
    split(mat, img_channels);
    auto *img_data = new float [(int)(img_channels[0].total() + img_channels[1].total()
                                      + img_channels[2].total())];
    for (int i = 0; i < img_channels[2].total(); ++i) {
        img_data[i] = (float)img_channels[2].data[i] / 255.0f;
    }
    for (int i = 0; i < img_channels[1].total(); ++i) {
        img_data[i + img_channels[2].total()] = (float)img_channels[1].data[i] / 255.0f;
    }
    for (int i = 0; i <img_channels[0].total(); ++i) {
        img_data[i + img_channels[2].total() + img_channels[1].total()] =
                (float)img_channels[0].data[i] / 255.0f;
   //     std::cout<<img_data[i + img_channels[2].total() + img_channels[1].total()]<<" ";
    }
    std::cout << "img_name: ";
    auto start = std::chrono::steady_clock::now();
    float out = GetConfidenceScore128x128rbg(img_data, img_channels[0].rows, img_channels[0].cols);
    auto end = std::chrono::steady_clock::now();
    std::cout << GetFileName(name) << "\nPossibility to be face: " << out << "\n";
   // timePrint
}



int main() {
    std::vector<std::string> file_names;

PrintImage("cat.jpg");
    std::cout<<std::endl;
    return 0;
}