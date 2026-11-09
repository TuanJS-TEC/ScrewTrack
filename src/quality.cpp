#include "quality.h"

#include <opencv2/imgproc.hpp>

std::optional<std::string> rejectReason(const cv::Mat& bgrCrop) {
    constexpr int kMinSide = 150;         // px; below this a hex's edges are too few pixels to fit an angle
    constexpr double kMinStdDev = 15.0;   // grey-level std dev; labelled nuts span 22-60, a flat patch is ~0
    constexpr double kMinSharpness = 60.0;  // Laplacian variance at 256x256; labelled nuts >= 80, blur sigma 2 gives 39
    if (bgrCrop.empty() || bgrCrop.cols < kMinSide || bgrCrop.rows < kMinSide) return "too small";

    cv::Mat gray;
    if (bgrCrop.channels() == 1) gray = bgrCrop;
    else cv::cvtColor(bgrCrop, gray, cv::COLOR_BGR2GRAY);
    cv::Scalar mean, sd;
    cv::meanStdDev(gray, mean, sd);
    if (sd[0] < kMinStdDev) return "low contrast";

    // Fixed size so the sharpness figure does not depend on the crop size.
    cv::Mat small, lap;
    cv::resize(gray, small, cv::Size(256, 256), 0, 0, cv::INTER_AREA);
    cv::Laplacian(small, lap, CV_64F);
    cv::meanStdDev(lap, mean, sd);
    if (sd[0] * sd[0] < kMinSharpness) return "blurry";
    return std::nullopt;
}
