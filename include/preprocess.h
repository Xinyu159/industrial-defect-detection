#pragma once

#include <opencv2/core.hpp>

namespace preprocess {

// Grayscale input is expected for NEU-DET steel surface images.
// Returns 8-bit single-channel image.
cv::Mat toGray(const cv::Mat& src);

// ---- Stage 1: denoising ----
// Median filter: replaces each pixel with the median of its ksize x ksize
// neighborhood. Edge-preserving and immune to outliers -- the right choice
// for impulse noise (dust, specular spikes). Kernel must be odd.
cv::Mat medianDenoise(const cv::Mat& gray, int ksize = 5);

// Gaussian filter: linear low-pass, weights decay with distance from center.
// Good for Gaussian (sensor) noise; trades some edge sharpness. sigma <= 0
// lets OpenCV derive sigma from the kernel size.
cv::Mat gaussianDenoise(const cv::Mat& gray, int ksize = 5, double sigma = 0.0);

// ---- Stage 1: contrast enhancement (low-contrast steel surfaces) ----
// Global histogram equalization: remaps gray levels so the cumulative
// histogram is (approx.) linear. Stretches the full-image contrast; can
// over-amplify noise in flat regions and does not handle uneven lighting.
cv::Mat equalizeGlobal(const cv::Mat& gray);

// CLAHE (Contrast Limited Adaptive Histogram Equalization): equalizes small
// local tiles, then limits contrast gain per pixel (clipLimit) and
// interpolates between tiles. The standard choice for industrial images
// with shading / uneven illumination -- local, not global.
cv::Mat equalizeCLAHE(const cv::Mat& gray, double clipLimit = 2.0, int tileSize = 8);

}  // namespace preprocess
