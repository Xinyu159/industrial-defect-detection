#pragma once

#include <opencv2/core.hpp>

// Visual helpers: human-checkable views of masks on top of the source gray
// image -- the fastest way to tell "did the strategy find the defect?".
namespace visualize {

// Returns a BGR copy of `gray` with defect-mask contours drawn in green.
// Contours instead of a flood fill keep the original pixels readable.
cv::Mat overlayMask(const cv::Mat& gray, const cv::Mat& mask,
                    const cv::Scalar& color = cv::Scalar(0, 230, 0));

// Returns a labelized color view: each connected component gets a distinct
// hue -- good for eyeballing oversegmentation before feature extraction.
cv::Mat colorizeComponents(const cv::Mat& mask);

}  // namespace visualize
