// Stage-0 skeleton: image I/O round-trip.
//   defect_detector <input_image> <output_image>
// Reads any image, converts to grayscale (NEU-DET steel images are
// stored as 3-channel JPEG but semantically grayscale), saves the result.
// This is the scaffold on which preprocess -> segmentation -> features
// -> classify stages will be stacked, keeping every stage verifiable.

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "preprocess.h"

namespace {
void usage(const char* argv0) {
    std::cerr << "usage: " << argv0 << " <input_image> <output_image>\n";
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        usage(argv[0]);
        return 1;
    }
    const std::string in_path = argv[1];
    const std::string out_path = argv[2];

    cv::Mat img = cv::imread(in_path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "failed to read image: " << in_path << "\n";
        return 1;
    }
    std::cout << "input : " << in_path << "  " << img.cols << "x" << img.rows
              << "  channels=" << img.channels() << "\n";

    cv::Mat gray = preprocess::toGray(img);

    if (!cv::imwrite(out_path, gray)) {
        std::cerr << "failed to write image: " << out_path << "\n";
        return 1;
    }
    std::cout << "output: " << out_path << "  gray  " << gray.cols << "x"
              << gray.rows << "\n";
    return 0;
}
