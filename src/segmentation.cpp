#include "segmentation.h"

#include <opencv2/imgproc.hpp>

#include "preprocess.h"

namespace segment {

cv::Mat thresholdOtsu(const cv::Mat& gray, bool defectsAreDark) {
    CV_Assert(!gray.empty() && gray.channels() == 1);
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    if (defectsAreDark) {
        // Otsu's 255 class is the BRIGHT one (background on steel); the
        // dark defects sit below the threshold -> invert to keep the
        // convention "255 = defect".
        cv::bitwise_not(binary, binary);
    }
    return binary;
}

cv::Mat edgeConnect(const cv::Mat& gray, bool enhanceFirst, double cannyLow,
                    double cannyHigh, int connectK) {
    CV_Assert(!gray.empty() && gray.channels() == 1);
    cv::Mat src = gray;
    if (enhanceFirst) {
        src = preprocess::equalizeCLAHE(gray, 2.0, 8);  // stage-1 boost
    }
    cv::Mat edges;
    cv::Canny(src, edges, cannyLow, cannyHigh);

    // Connect broken line fragments: close with a square kernel joins
    // segments a few px apart; opening then drops isolated specks.
    cv::Mat cleaned;
    const cv::Mat kConn = cv::getStructuringElement(cv::MORPH_RECT,
                                                    cv::Size(connectK, connectK));
    cv::morphologyEx(edges, cleaned, cv::MORPH_CLOSE, kConn);
    cv::Mat kSmall = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(cleaned, cleaned, cv::MORPH_OPEN, kSmall);
    return cleaned;
}

cv::Mat blackHatDetect(const cv::Mat& gray, int kernelSize) {
    CV_Assert(!gray.empty() && gray.channels() == 1);
    cv::Mat hat;
    const cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSize, kernelSize));
    cv::morphologyEx(gray, hat, cv::MORPH_BLACKHAT, kernel);

    // Hat response is strong where pixels are much darker than their
    // neighborhood; Otsu separates those dips from flat/high-texture bg.
    cv::Mat binary;
    cv::threshold(hat, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    return binary;
}

cv::Mat morphologyClean(const cv::Mat& mask, int openK, int closeK) {
    CV_Assert(!mask.empty() && mask.channels() == 1);
    cv::Mat out;
    const cv::Mat kOpen =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(openK, openK));
    const cv::Mat kClose =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(closeK, closeK));
    cv::morphologyEx(mask, out, cv::MORPH_OPEN, kOpen);
    cv::morphologyEx(out, out, cv::MORPH_CLOSE, kClose);
    return out;
}

}  // namespace segment
