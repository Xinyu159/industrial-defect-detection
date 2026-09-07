#pragma once

#include <opencv2/core.hpp>

namespace preprocess {

// Grayscale input is expected for NEU-DET steel surface images.
// Returns 8-bit single-channel image.
cv::Mat toGray(const cv::Mat& src);

}  // namespace preprocess
