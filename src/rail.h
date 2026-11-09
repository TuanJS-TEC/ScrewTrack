#pragma once
#include <opencv2/core.hpp>
#include <optional>

// Direction of the dominant near-horizontal straight line (the rail) in a whole BGR photo, in degrees,
// atan2(dy, dx) with y down; nullopt when no dominant near-horizontal line within +-15 degrees is found.
// A returned angle is not proof that a rail is present: any strong horizontal edge structure can also give one.
// Accepts 3-channel BGR or 1-channel gray.
std::optional<double> railAngle(const cv::Mat& bgr);
