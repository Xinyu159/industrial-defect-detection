// defect_detector -- NEU-DET steel surface defect detection (C++/OpenCV)
//
// Commands:
//   defect_detector gray       <in> <out.png>          -- stage0: gray only
//   defect_detector preprocess <in> <out_prefix>       -- stage1: full chain
//
// Stage-1 chain (why this order matters):
//   1) toGray     -- NEU images are 3-channel JPEG but semantically gray
//   2) denoise    -- equalization amplifies noise, so remove it FIRST
//   3) enhance    -- global equalizeHist (full-frame) vs CLAHE (local,
//                    contrast-limited) -- the robust pick for uneven
//                    illumination on real production lines
// Every step is saved as <out_prefix>_<step>.png and the gray-level
// statistics (min/mean/max/std) are printed so the effect is verifiable
// numerically, not just by eye.

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
    std::cerr << "usage:\n"
              << "  " << argv0 << " gray       <input_image> <output_image>\n"
              << "  " << argv0 << " preprocess <input_image> <out_prefix>\n";
}

// min / mean / max / std of an 8-bit gray image -- numeric evidence that a
// preprocessing step changed the distribution the way we intended.
void reportStats(const char* tag, const cv::Mat& gray) {
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    double mn, mx;
    cv::minMaxLoc(gray, &mn, &mx);
    std::cout << "  " << tag << ": min=" << (int)mn << " mean=" << mean[0]
              << " max=" << (int)mx << " std=" << stddev[0] << "\n";
}

// Stage 1: gray -> median denoise -> global / CLAHE equalization.
// Returns 0 on success.
int runPreprocess(const std::string& in_path, const std::string& prefix) {
    cv::Mat img = cv::imread(in_path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "failed to read image: " << in_path << "\n";
        return 1;
    }
    std::cout << "input : " << in_path << "  " << img.cols << "x" << img.rows
              << "  channels=" << img.channels() << "\n";

    const cv::Mat gray = preprocess::toGray(img);
    reportStats("gray        ", gray);
    cv::imwrite(prefix + "_1_gray.png", gray);

    const cv::Mat denoised = preprocess::medianDenoise(gray, 5);
    reportStats("median d=5  ", denoised);
    cv::imwrite(prefix + "_2_median5.png", denoised);

    const cv::Mat eq_global = preprocess::equalizeGlobal(denoised);
    reportStats("eq global   ", eq_global);
    cv::imwrite(prefix + "_3_equalize_global.png", eq_global);

    const cv::Mat eq_clahe = preprocess::equalizeCLAHE(denoised, 2.0, 8);
    reportStats("eq CLAHE    ", eq_clahe);
    cv::imwrite(prefix + "_4_equalize_clahe.png", eq_clahe);

    std::cout << "saved: " << prefix << "_{1_gray,2_median5,3_equalize_global,"
                 "4_equalize_clahe}.png\n";
    return 0;
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
        if (argc < 3) {
            usage(argv[0]);
            return 1;
        }
        const std::string cmd = argv[1];
        if (cmd == "preprocess") {
            if (argc != 4) {
                usage(argv[0]);
                return 1;
            }
            return runPreprocess(argv[2], argv[3]);
        }
        // legacy / "gray" mode: <in> <out.png>
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
