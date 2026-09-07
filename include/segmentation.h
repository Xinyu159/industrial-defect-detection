#pragma once

#include <opencv2/core.hpp>

// Stage 2: defect segmentation. Three strategies, each exploiting a
// different "separability dimension" (see docs): gray level (Otsu),
// gradient (edges), local relative gray (black-hat). All return a binary
// mask where 255 = candidate defect pixels.
namespace segment {

// Strategy 1 -- gray-level separability. Otsu finds the threshold that
// maximizes between-class variance; assumes a bimodal histogram.
// Works for dark blocks on brighter steel (RS rolled-in scale, Pa patches).
cv::Mat thresholdOtsu(const cv::Mat& gray, bool defectsAreDark = true);

// Strategy 2 -- gradient separability: Canny edges, then morphological
// closing/opening to connect broken segments into elongated regions and
// drop specks. For thin, low-contrast structures (Sc scratches, Cr crazing
// web) where gray levels overlap the background but gradients do not.
// `enhanceFirst` applies CLAHE before Canny -- the contrast boost that
// makes faint cracks detectable (verified visually in stage 1).
cv::Mat edgeConnect(const cv::Mat& gray, bool enhanceFirst = true,
                    double cannyLow = 60, double cannyHigh = 160,
                    int connectK = 5);

// Strategy 3 -- local-relative gray: black-hat = closing - original,
// i.e. structures DARKER than their immediate neighborhood, independent
// of global illumination. Then Otsu on the hat response keeps the strong
// local dips (PS pitting dots, In inclusions).
// kernelSize must exceed the defect size but stay below the background
// texture scale.
cv::Mat blackHatDetect(const cv::Mat& gray, int kernelSize = 15);

// Post-process a raw mask: opening removes isolated specks (noise),
// closing fills small holes inside defects. Returns cleaned mask.
cv::Mat morphologyClean(const cv::Mat& mask, int openK = 3, int closeK = 5);

}  // namespace segment
