#pragma once
#include <opencv2/core.hpp>
#include <optional>

// Folds an angle in degrees into [-30, 30) (hex symmetry is 60 degrees).
double wrap60(double deg);

// Hex orientation of a nut crop (CV_8UC1) in degrees, folded into [-30, 30),
// atan2(dy, dx) convention with y down; nullopt when there is no clear 6-fold structure.
// The value is the gradient direction, i.e. the edge direction + 30 deg (mod 60): only differences are meaningful.
// Expects a box tight around the nut: only the ring at 0.38..0.47 of the short side is used.
std::optional<double> hexAngle(const cv::Mat& gray);

// A ring (annulus) in crop pixel coordinates.
struct Ring {
    cv::Point2f centre;
    float rIn, rOut;
};

// Ring around the nut in a crop that is only roughly around it (e.g. a detector box): tries centres within
// +-0.15 of the short side and ring scales 0.75..1.0. Accepts only if the most significant 6-fold ring passes
// hexAngle's rules and a significance floor (nullopt otherwise), then returns the most central passing ring within
// 5% of its significance that reads the same hex angle (within 2 deg).
std::optional<Ring> selectRing(const cv::Mat& gray);

// hexAngle's measurement and acceptance rules on a given ring.
std::optional<double> hexAngleAt(const cv::Mat& gray, const Ring& ring);

// The worst shift (deg) over the 24 leave-out windows of consistentTurn below, and the turn itself in `turn`.
double arcShift(const cv::Mat& refGray, const cv::Mat& sesGray, const Ring& ring, double& turn);

// False when the turn between two crops of the same nut (same size, same ring) is not carried by the whole ring:
// leaving out any 45-degree arc (session arcs follow the turn) moves the turn by more than 0.55 deg. A strip, stick
// or debris over part of the nut in one photo does that; synthetic clean turns of 0-4 deg pass in ~96% of cases.
// Clean turns of 6-30 deg pass far less often (the still background in the ring moves the arcs more the further the
// nut turned: 49 of 101 at 15 deg), so a larger shift is also accepted when the nut still reads loose with the shift
// taken off: |turn| - shift >= kLooseDeg + 1. Only the ok/loose answer matters there.
bool consistentTurn(const cv::Mat& refGray, const cv::Mat& sesGray, const Ring& ring);
