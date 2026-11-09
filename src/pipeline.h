#pragma once
#include <filesystem>
#include <vector>

#include "detector.h"
#include "types.h"

// One NutResult per reference image (*.jpg/*.JPG, sorted by name) and side (0 top, 1 bottom).
// The hex ring is chosen once on the reference's detector box; the session is cut at the same rectangle and
// measured on the same ring, moved by the placement shift (phase correlation).
// rotationDeg = wrap60((hex_s - rail_s) - (hex_r - rail_r)); Loose when |rotationDeg| >= kLooseDeg.
// Unmeasurable reasons: "image missing", "nut not found", or "unclear: <why>" (quality gate,
// "no clear hex", "no rail line", "misaligned", or "covered": the turn is not carried by the whole hex ring,
// e.g. something covers part of the nut in one photo). If both images fail, the reference's reason is reported.
std::vector<NutResult> measureSession(const NutDetector& det, const std::filesystem::path& refDir,
                                      const std::filesystem::path& sessionDir);

// Stores the rotation rounded to 0.01 deg (as the CSV prints it) and sets Ok/Loose from the rounded value.
void setRotation(NutResult& r, double deg);

// Rotation of one registered nut: `refBox` is where it sits on `side` of the registered photo `ref`, `sesBox` where the
// same nut was found in `ses`. Same measurement as measureSession for that side, without running the detector.
// Both photos must have the same size.
NutResult measureNutIn(const cv::Mat& ref, const cv::Rect& refBox, const cv::Mat& ses, const cv::Rect& sesBox, int side);
