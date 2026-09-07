#include "image_features.h"

#include <opencv2/imgproc.hpp>

#include <cmath>

namespace features {

namespace {

// Quantize 8-bit gray into `levels` bins: out is CV_8U with values 0..L-1.
cv::Mat quantize(const cv::Mat& gray, int levels) {
    cv::Mat f;
    gray.convertTo(f, CV_32F, (levels - 1) / 255.0);
    cv::Mat q;
    f.convertTo(q, CV_8U);  // saturating cast -> clamp to [0, 255]
    return q;
}

}  // namespace

void appendGrayStats(const cv::Mat& gray, std::vector<double>& out) {
    CV_Assert(!gray.empty() && gray.channels() == 1);
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    out.push_back(mean[0]);
    out.push_back(stddev[0]);
}

void appendGlcm(const cv::Mat& gray, std::vector<double>& out) {
    CV_Assert(!gray.empty() && gray.channels() == 1);
    const int L = 32;  // gray levels after quantization
    const cv::Mat q = quantize(gray, L);
    const int h = q.rows, w = q.cols;

    // Co-occurrence of gray value a with b at offset d=(1,0),(1,1),(0,1),
    //(-1,1). Matrix is symmetrized (add transpose) so direction does not
    // matter -- we want "how textured", not "which orientation".
    std::vector<std::vector<long>> glcm(L, std::vector<long>(L, 0));
    const int dx[4] = {1, 1, 0, -1};
    const int dy[4] = {0, 1, 1, 1};
    for (int y = 0; y < h; ++y) {
        const uchar* row = q.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            const int a = row[x];
            for (int k = 0; k < 4; ++k) {
                const int nx = x + dx[k], ny = y + dy[k];
                if (nx < 0 || nx >= w || ny < 0 || ny >= h) {
                    continue;
                }
                const int b = q.ptr<uchar>(ny)[nx];
                ++glcm[a][b];
            }
        }
    }
    long total = 0;
    for (int i = 0; i < L; ++i) {
        for (int j = i; j < L; ++j) {
            const long sym = glcm[i][j] + glcm[j][i];
            glcm[i][j] = glcm[j][i] = sym;
            total += sym;
        }
    }

    // Marginal distribution (same for rows and columns, matrix symmetric).
    std::vector<double> marg(L, 0.0);
    double mean = 0.0;
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            marg[i] += glcm[i][j] / static_cast<double>(total);
        }
        mean += i * marg[i];
    }
    double var = 0.0;
    for (int i = 0; i < L; ++i) {
        var += (i - mean) * (i - mean) * marg[i];
    }

    double contrast = 0.0, energy = 0.0, homogeneity = 0.0, correlation = 0.0;
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            const double p = glcm[i][j] / static_cast<double>(total);
            const int diff = i - j;
            contrast += p * diff * diff;
            energy += p * p;
            homogeneity += p / (1.0 + std::abs(diff));
            correlation += p * (i - mean) * (j - mean);
        }
    }
    // Correlation is the normalized covariance; a flat (zero-variance)
    // image has a degenerate matrix -> define it as 1 (fully correlated).
    out.push_back(contrast);
    out.push_back(energy);
    out.push_back(homogeneity);
    out.push_back(var < 1e-12 ? 1.0 : correlation / var);
}

void appendMaskShape(const cv::Mat& mask, std::vector<double>& out) {
    CV_Assert(!mask.empty() && mask.channels() == 1);
    const double fg =
        cv::countNonZero(mask) / static_cast<double>(mask.total());
    out.push_back(fg);

    cv::Mat labels, stats, centroids;
    const int n =
        cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    const int regions = n - 1;  // 0 = background
    out.push_back(regions);
    if (regions == 0) {
        out.push_back(0.0);
        out.push_back(0.0);
        return;
    }
    double area_sum = 0.0;
    int area_max = 0;
    for (int i = 1; i < n; ++i) {
        const int a = stats.at<int>(i, cv::CC_STAT_AREA);
        area_sum += a;
        area_max = std::max(area_max, a);
    }
    out.push_back(area_sum / regions);
    out.push_back(area_max);
}

}  // namespace features
