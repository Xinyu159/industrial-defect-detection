#include "visualize.h"

#include <opencv2/imgproc.hpp>

namespace visualize {

cv::Mat overlayMask(const cv::Mat& gray, const cv::Mat& mask,
                    const cv::Scalar& color) {
    CV_Assert(!gray.empty() && gray.channels() == 1);
    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(bgr, contours, -1, color, 2, cv::LINE_AA);
    return bgr;
}

cv::Mat colorizeComponents(const cv::Mat& mask) {
    CV_Assert(!mask.empty() && mask.channels() == 1);
    cv::Mat labels, stats, centroids;
    const int n =
        cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    cv::Mat colored(mask.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::RNG rng(20260907);  // fixed seed -> reproducible colors
    for (int i = 1; i < n; ++i) {  // 0 = background
        const cv::Vec3b color(static_cast<uchar>(rng.uniform(60, 255)),
                              static_cast<uchar>(rng.uniform(60, 255)),
                              static_cast<uchar>(rng.uniform(60, 255)));
        colored.setTo(color, labels == i);
    }
    return colored;
}

}  // namespace visualize
