#pragma once
#include <opencv2/core.hpp>
#include <optional>
#include <string>

// nullopt when the BGR nut crop is usable for angle measurement; otherwise a short reason
// ("too small", "low contrast", "blurry"). Never throws on an empty Mat.
// Limit: only checks size, contrast and sharpness; a clear but covered/rusty nut still passes.
std::optional<std::string> rejectReason(const cv::Mat& bgrCrop);
