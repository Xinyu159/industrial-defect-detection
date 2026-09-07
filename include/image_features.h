#pragma once

#include <opencv2/core.hpp>

#include <vector>

// Stage 3: per-image feature extraction. Every function APPENDS its
// features to `out`, so callers can compose the groups they need in a
// fixed order. Features are chosen to be explainable -- each one names
// the defect property it measures:
//   gray group : brightness/shape of the gray histogram
//                (inclusions are dark blobs, patches are bright)
//   GLCM group : local texture energy/contrast from the co-occurrence
//                matrix (crazing nets are fine dense texture -- the
//                class that defeated every pixel-level strategy)
//   mask group : what the stage-2 segmentation found (scratches are
//                few elongated regions, pitting is many small ones)
namespace features {

// mean, std of the gray image (2 features)
void appendGrayStats(const cv::Mat& gray, std::vector<double>& out);

// Gray-level co-occurrence matrix statistics over 4 directions at
// distance 1, image quantized to 32 levels:
// contrast, energy, homogeneity, correlation (4 features)
void appendGlcm(const cv::Mat& gray, std::vector<double>& out);

// Foreground ratio, region count, mean and max region area of a
// (binary, cleaned) mask (4 features)
void appendMaskShape(const cv::Mat& mask, std::vector<double>& out);

}  // namespace features
