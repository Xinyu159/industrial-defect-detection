#include "preprocess.h"

#include <opencv2/imgproc.hpp>

namespace preprocess {

cv::Mat toGray(const cv::Mat& src) {
    CV_Assert(!src.empty());
    if (src.channels() == 1) {
        return src.clone();
    }
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

}  // namespace preprocess
