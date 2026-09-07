// defect_detector -- NEU-DET steel surface defect detection (C++/OpenCV)
//
// Commands:
//   defect_detector gray       <in> <out.png>          -- stage0: gray only
//   defect_detector preprocess <in> <out_prefix>       -- stage1: full chain
//   defect_detector segment    <in> <out_prefix> [strategy]
//   defect_detector batch      <list.txt> <out_dir>    -- stage2.5: one image
//        per line, all strategies, masks only. Evaluation against the XML
//        ground truth lives in scripts/batch_hitrate.py -- the C++ core only
//        produces masks, a separate validation tool judges them. A single
//        image that fails to load is skipped, the rest still runs.
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
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "preprocess.h"
#include "segmentation.h"
#include "visualize.h"

namespace {

const std::vector<std::string> kSupportedExts = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"};

void usage(const char* argv0) {
    std::cerr << "usage:\n"
              << "  " << argv0 << " gray       <input_image> <output_image>\n"
              << "  " << argv0 << " preprocess <input_image> <out_prefix>\n"
              << "  " << argv0 << " segment    <input_image> <out_prefix> "
                 "[otsu|edge|hat|all]\n"
              << "  " << argv0 << " batch      <image_list.txt> <out_dir>\n";
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

// Run one segmentation strategy on an image, save mask + overlay + a
// component-color view, and report region stats so results can be judged
// numerically before moving on (HDevelop-style "drag & look" loop).
void runOneStrategy(const std::string& tag, const cv::Mat& gray,
                    const cv::Mat& mask, const std::string& prefix) {
    cv::Mat cleaned = segment::morphologyClean(mask, 3, 5);
    cv::imwrite(prefix + "_" + tag + "_mask.png", cleaned);

    cv::Mat overlay = visualize::overlayMask(gray, cleaned);
    cv::imwrite(prefix + "_" + tag + "_overlay.png", overlay);

    cv::Mat colored = visualize::colorizeComponents(cleaned);
    cv::imwrite(prefix + "_" + tag + "_components.png", colored);

    cv::Mat labels, stats, centroids;
    const int n =
        cv::connectedComponentsWithStats(cleaned, labels, stats, centroids, 8);
    const double fgShare =
        100.0 * cv::countNonZero(cleaned) / (double)cleaned.total();
    std::cout << "  [" << tag << "] fg=" << fgShare << "% regions=" << n - 1;
    if (n > 1) {
        std::vector<int> areas;
        for (int i = 1; i < n; ++i) {
            areas.push_back(stats.at<int>(i, cv::CC_STAT_AREA));
        }
        std::sort(areas.begin(), areas.end(), std::greater<int>());
        std::cout << " top areas=";
        const size_t k = std::min<size_t>(3, areas.size());
        for (size_t i = 0; i < k; ++i) {
            std::cout << areas[i] << (i + 1 < k ? "," : "");
        }
    }
    std::cout << "\n";
}

// Stage 2: try the three strategies on one image; each exploits a different
// separability dimension, then morphologyClean tidies the mask up.
int runSegment(const std::string& in_path, const std::string& prefix,
               const std::string& which) {
    cv::Mat img = cv::imread(in_path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "failed to read image: " << in_path << "\n";
        return 1;
    }
    const cv::Mat gray = preprocess::toGray(img);
    std::cout << "segment " << in_path << "\n";

    const bool all = (which == "all");
    if (all || which == "otsu") {
        cv::Mat m = segment::thresholdOtsu(gray, /*defectsAreDark=*/true);
        runOneStrategy("otsu", gray, m, prefix);
    }
    if (all || which == "edge") {
        cv::Mat m = segment::edgeConnect(gray, /*enhanceFirst=*/true);
        runOneStrategy("edge", gray, m, prefix);
    }
    if (all || which == "hat") {
        cv::Mat m = segment::blackHatDetect(gray, 15);
        runOneStrategy("hat", gray, m, prefix);
    }
    std::cout << "saved: " << prefix
              << "_{otsu,edge,hat}_{mask,overlay,components}.png\n";
    return 0;
}

// Stage 2.5: run every segmentation strategy on every image listed in
// <list.txt> (one path per line, '#' comments allowed) and write only the
// cleaned mask to <out_dir>/<stem>_<strategy>_mask.png. Runs all images in
// one process -- unlike the interactive `segment` command this must be fast
// enough for hundreds of frames. Masks use the same morphologyClean(3,5)
// post-processing as `segment`, so interactive and batch results match.
int runBatch(const std::string& list_path, const std::string& out_dir) {
    std::ifstream in(list_path);
    if (!in) {
        std::cerr << "cannot open list file: " << list_path << "\n";
        return 1;
    }
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);

    int ok = 0, skipped = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::string line;
    while (std::getline(in, line)) {
        // trim surrounding whitespace
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }
        const std::string path = line.substr(first, line.find_last_not_of(" \t\r\n") - first + 1);

        cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            std::cerr << "!! skip (cannot read): " << path << "\n";
            ++skipped;
            continue;
        }
        const cv::Mat gray = preprocess::toGray(img);
        const std::string stem = std::filesystem::path(path).stem().string();

        struct Run {
            const char* tag;
            cv::Mat (*make)(const cv::Mat&);
        } const runs[] = {
            {"otsu", [](const cv::Mat& g) { return segment::thresholdOtsu(g, true); }},
            {"edge", [](const cv::Mat& g) { return segment::edgeConnect(g, true); }},
            {"hat", [](const cv::Mat& g) { return segment::blackHatDetect(g, 15); }},
        };
        for (const auto& r : runs) {
            const cv::Mat cleaned =
                segment::morphologyClean(r.make(gray), 3, 5);
            cv::imwrite(out_dir + "/" + stem + "_" + r.tag + "_mask.png", cleaned);
        }
        ++ok;
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    std::cout << "batch done: " << ok << " processed, " << skipped
              << " skipped in " << ms << " ms\n";
    std::cout << "masks in:   " << out_dir << "\n";
    return 0;
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
        if (cmd == "segment") {
            const std::string which =
                (argc == 5) ? argv[4] : std::string("all");
            return runSegment(argv[2], argv[3], which);
        }
        if (cmd == "preprocess") {
            if (argc != 4) {
                usage(argv[0]);
                return 1;
            }
            return runPreprocess(argv[2], argv[3]);
        }
        if (cmd == "batch") {
            if (argc != 4) {
                usage(argv[0]);
                return 1;
            }
            return runBatch(argv[2], argv[3]);
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
