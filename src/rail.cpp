#include "rail.h"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <vector>

// Radon-style skew search: rotate the horizontal-edge map by each candidate angle and score how
// strongly the edge energy piles into a few rows. A long straight rail edge peaks sharply when it is
// horizontal; unlike HoughLinesP this gives sub-0.1 degree precision (Hough segments scattered by ~1 degree).
namespace {

double rowSharpness(const cv::Mat& edges, const cv::Rect& crop, double deg) {
    // Positive deg turns the image counter-clockwise on screen, which lowers atan2(dy,dx) (y down) by deg.
    cv::Mat m = cv::getRotationMatrix2D({edges.cols / 2.0f, edges.rows / 2.0f}, deg, 1.0);
    m.at<double>(0, 2) -= crop.x;
    m.at<double>(1, 2) -= crop.y;
    cv::Mat rotated, rows;
    cv::warpAffine(edges, rotated, m, crop.size(), cv::INTER_LINEAR);
    cv::reduce(rotated, rows, 1, cv::REDUCE_SUM, CV_32F);
    return rows.dot(rows);
}

}  // namespace

std::optional<double> railAngle(const cv::Mat& bgr) {
    if (bgr.empty() || (bgr.channels() != 1 && bgr.channels() != 3)) return std::nullopt;
    const double scale = 1000.0 / std::max(bgr.cols, bgr.rows);  // ~1000 px long side
    cv::Mat gray, edges;
    if (bgr.channels() == 3) cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY); else gray = bgr;
    cv::resize(gray, gray, {}, scale, scale, cv::INTER_AREA);
    cv::GaussianBlur(gray, gray, {5, 5}, 0);
    cv::Sobel(gray, edges, CV_32F, 0, 1, 3);
    edges = cv::abs(edges);

    // Central square of 0.7 of the short side stays inside the image for any rotation about the centre.
    const int side = cvRound(0.7 * std::min(edges.cols, edges.rows));
    const cv::Rect crop((edges.cols - side) / 2, (edges.rows - side) / 2, side, side);

    const double range = 15.0;  // rail must lie within +-15 degrees of horizontal
    std::vector<double> coarse;
    double bestDeg = 0, best = -1;
    for (double d = -range; d <= range + 1e-9; d += 0.5) {  // coarse step 0.5 degrees
        coarse.push_back(rowSharpness(edges, crop, d));
        if (coarse.back() > best) best = coarse.back(), bestDeg = d;
    }
    std::vector<double> sorted = coarse;
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
    const double median = sorted[sorted.size() / 2];
    if (std::abs(bestDeg) >= range) return std::nullopt;

    const double center = bestDeg;
    best = -1;
    for (int i = -10; i <= 10; ++i) {  // fine step 0.05 degrees around the coarse peak
        const double s = rowSharpness(edges, crop, center + 0.05 * i);
        if (s > best) best = s, bestDeg = center + 0.05 * i;
    }
    // Peak must stand 1.5% above the median score: pure noise gives 1.004, real photos >= 1.025.
    if (median <= 0 || best < 1.015 * median) return std::nullopt;
    return bestDeg;
}
