// Stage-0 skeleton: image I/O round-trip.
//   defect_detector <input_image> <output_image>
// Reads any image, converts to grayscale (NEU-DET steel images are
// stored as 3-channel JPEG but semantically grayscale), saves the result.
// This is the scaffold on which preprocess -> segmentation -> features
// -> classify stages will be stacked, keeping every stage verifiable.
//
// v2: friendly diagnostics -- validate the output extension up-front
// (imwrite selects the encoder from the file extension, so a typo like
// ".jgp" must be reported in plain words) and catch cv::Exception so the
// program exits cleanly instead of aborting with a core dump.

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "preprocess.h"

namespace {

const std::vector<std::string> kSupportedExts = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"};

void usage(const char* argv0) {
    std::cerr << "usage: " << argv0 << " <input_image> <output_image>\n";
}

std::string extension(const std::string& path) {
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) {
        return "";
    }
    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return ext;
}

bool hasSupportedExt(const std::string& path) {
    const std::string ext = extension(path);
    return std::find(kSupportedExts.begin(), kSupportedExts.end(), ext) !=
           kSupportedExts.end();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            usage(argv[0]);
            return 1;
        }
        const std::string in_path = argv[1];
        const std::string out_path = argv[2];

        if (!hasSupportedExt(out_path)) {
            std::cerr << "unsupported output extension \"" << extension(out_path)
                      << "\" -- imwrite picks the encoder from the extension, "
                         "supported: "
                      << "png, jpg, jpeg, bmp, tif, tiff\n";
            return 1;
        }

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
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << "\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
