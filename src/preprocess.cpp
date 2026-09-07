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

cv::Mat medianDenoise(const cv::Mat& gray, int ksize) {
    CV_Assert(!gray.empty() && gray.channels() == 1);
    cv::Mat out;
    cv::medianBlur(gray, out, ksize);
    return out;
}

cv::Mat gaussianDenoise(const cv::Mat& gray, int ksize, double sigma) {
    CV_Assert(!gray.empty() && gray.channels() == 1);
    cv::Mat out;
    cv::GaussianBlur(gray, out, cv::Size(ksize, ksize), sigma);
    return out;
}

cv::Mat equalizeGlobal(const cv::Mat& gray) {
    CV_Assert(!gray.empty() && gray.channels() == 1 && gray.depth() == CV_8U);
    cv::Mat out;
    cv::equalizeHist(gray, out);
    return out;
}

cv::Mat equalizeCLAHE(const cv::Mat& gray, double clipLimit, int tileSize) {
    CV_Assert(!gray.empty() && gray.channels() == 1 && gray.depth() == CV_8U);
    cv::Mat out;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, cv::Size(tileSize, tileSize));
    clahe->apply(gray, out);
    return out;
}

}  // namespace preprocess
