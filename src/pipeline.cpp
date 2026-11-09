#include "pipeline.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "angle.h"
#include "quality.h"
#include "rail.h"

namespace {
namespace fs = std::filesystem;

// phaseCorrelate peak on a window about twice the box: a nut rotated 4-30 deg in IMG_0001 still gives >= 0.47;
// with the rail-difference de-rotation, whole photos turned 0.5-5 deg give >= 0.19 (157 nut windows), while
// 1570 unrelated photo pairs give <= 0.040.
constexpr double kMinResponse = 0.1;

// One photo, loaded and detected once.
struct Shot {
    cv::Mat img;                // empty for missing and zero-byte files
    std::array<cv::Rect, 2> box;  // per side; empty = no nut
    std::optional<double> rail;
};

Shot load(const NutDetector& det, const fs::path& file) {
    Shot s;
    s.img = cv::imread(file.string());
    if (s.img.empty()) return s;
    for (const cv::Rect& r : det.detect(s.img)) {
        cv::Rect& b = s.box[sideOf(r, s.img.rows)];
        if (r.area() > b.area()) b = r;  // two boxes on one side: keep the larger
    }
    s.rail = railAngle(s.img);
    return s;
}

cv::Mat grayOf(const cv::Mat& bgr) {
    cv::Mat g;
    cv::cvtColor(bgr, g, cv::COLOR_BGR2GRAY);
    return g;
}

// Shift of the session's content relative to the reference around `box` (device placement differs slightly).
// The window is about twice the box, centred on it: on the box alone the rotating nut dominates and
// the correlation peak of a 10 deg turn sinks to the level of unrelated photos. The session window is first
// turned back by the rail difference `railTurn` (session - reference, deg): without it a photo turned 2 deg peaked
// below 0.05 in 16 of 157 nut windows. Returns how far `pivot` (photo coordinates) moved; with the ring centre as
// pivot the ring keeps its place on the nut when the photo turns (a 3 deg photo turn then read -0.2 deg, with
// the box centre as pivot -0.46). nullopt when the move is implausibly large (> 0.15 of the box's short side)
// or the peak is weak.
std::optional<cv::Point2d> placementShift(const cv::Mat& ref, const cv::Mat& ses, const cv::Rect& box, double railTurn,
                                          cv::Point2d pivot) {
    // phaseCorrelate pads to getOptimalDFTSize; an odd padded size biases the shift by half a pixel (identical
    // photos gave 0.5 px), so the window has an even optimal size and is slid, not cut, to stay in the photo.
    auto evenDft = [](int n) {
        n = cv::getOptimalDFTSize(n);
        while (n % 2) n = cv::getOptimalDFTSize(n + 1);
        return n;
    };
    const cv::Size size(evenDft(2 * box.width), evenDft(2 * box.height));
    if (size.width > ref.cols || size.height > ref.rows) return std::nullopt;
    const cv::Point centre = (box.tl() + box.br()) / 2;
    const cv::Rect win(std::clamp(centre.x - size.width / 2, 0, ref.cols - size.width),
                       std::clamp(centre.y - size.height / 2, 0, ref.rows - size.height), size.width, size.height);
    if ((win & cv::Rect(0, 0, ses.cols, ses.rows)) != win) return std::nullopt;
    cv::Mat a, s, b, window, back;
    grayOf(ref(win)).convertTo(a, CV_32F);
    grayOf(ses(win)).convertTo(s, CV_32F);
    // Rotation is compensated by the measured rail angles, not estimated from the images.
    const cv::Mat turn = cv::getRotationMatrix2D(cv::Point2f(size.width / 2.0f, size.height / 2.0f), railTurn, 1.0);
    cv::warpAffine(s, b, turn, s.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    cv::createHanningWindow(window, a.size(), CV_32F);
    double response = 0;
    const cv::Point2d shift = cv::phaseCorrelate(a, b, window, &response);
    // The pivot sits at pivot + shift in the turned-back window; map it into the session window.
    const cv::Point2d c(pivot.x - win.x, pivot.y - win.y);
    cv::invertAffineTransform(turn, back);
    const cv::Point2d p = c + shift;
    const cv::Point2d moved(back.at<double>(0, 0) * p.x + back.at<double>(0, 1) * p.y + back.at<double>(0, 2) - c.x,
                            back.at<double>(1, 0) * p.x + back.at<double>(1, 1) * p.y + back.at<double>(1, 2) - c.y);
    if (std::hypot(moved.x, moved.y) > 0.15 * std::min(box.width, box.height) || response < kMinResponse)
        return std::nullopt;
    return moved;
}

// Angle (hex minus rail) of one nut in both photos, or the reason it cannot be measured.
struct Pair {
    std::optional<double> rotation;
    std::string reason;
};

Pair measureNut(const Shot& ref, const Shot& ses, int side) {
    if (ref.img.empty()) return {std::nullopt, "image missing"};
    const cv::Rect box = ref.box[side];
    if (box.empty()) return {std::nullopt, "nut not found"};
    if (const auto bad = rejectReason(ref.img(box))) return {std::nullopt, "unclear: " + *bad};
    const cv::Mat refGray = grayOf(ref.img(box));
    const auto ring = selectRing(refGray);
    if (!ring) return {std::nullopt, "unclear: no clear hex"};
    const auto hexR = hexAngleAt(refGray, *ring);
    if (!hexR) return {std::nullopt, "unclear: no clear hex"};
    if (!ref.rail) return {std::nullopt, "unclear: no rail line"};

    if (ses.img.empty()) return {std::nullopt, "image missing"};
    if (ses.box[side].empty()) return {std::nullopt, "nut not found"};
    if (!ses.rail) return {std::nullopt, "unclear: no rail line"};
    const auto shift =
        placementShift(ref.img, ses.img, box, *ses.rail - *ref.rail, cv::Point2d(box.tl()) + cv::Point2d(ring->centre));
    if (!shift) return {std::nullopt, "unclear: misaligned"};
    // Move the crop, not the ring: the ring then covers the same nut pixels with the same crop borders (moving
    // only the ring changes how it is clipped and blurred at the borders: 1 deg error for a pure 20 px shift).
    // Whole pixels only: a sub-pixel ring move alone changed identical photos by up to 0.1 deg.
    const cv::Rect sesBox = box + cv::Point(cvRound(shift->x), cvRound(shift->y));
    if ((sesBox & cv::Rect(0, 0, ses.img.cols, ses.img.rows)) != sesBox) return {std::nullopt, "unclear: misaligned"};
    if (const auto bad = rejectReason(ses.img(sesBox))) return {std::nullopt, "unclear: " + *bad};
    const cv::Mat sesGray = grayOf(ses.img(sesBox));
    const auto hexS = hexAngleAt(sesGray, *ring);
    if (!hexS) return {std::nullopt, "unclear: no clear hex"};
    // Strength and balance alone let a strip over 15% of a nut read -6.5 deg on a nut that did not turn.
    if (!consistentTurn(refGray, sesGray, *ring)) return {std::nullopt, "unclear: covered"};
    return {wrap60((*hexS - *ses.rail) - (*hexR - *ref.rail)), ""};
}

bool isJpg(const fs::path& p) {
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return std::tolower(c); });
    return e == ".jpg";
}
}  // namespace

void setRotation(NutResult& r, double deg) {
    // Rounded as the CSV prints it: unrounded, 2.996 printed "3.00,ok".
    r.rotationDeg = std::round(deg * 100.0) / 100.0;
    r.status = std::abs(*r.rotationDeg) >= kLooseDeg ? Status::Loose : Status::Ok;
}

std::vector<NutResult> measureSession(const NutDetector& det, const fs::path& refDir, const fs::path& sessionDir) {
    std::vector<fs::path> names;
    for (const auto& e : fs::directory_iterator(refDir))
        if (e.is_regular_file() && isJpg(e.path())) names.push_back(e.path().filename());
    std::sort(names.begin(), names.end());

    std::vector<NutResult> rows;
    for (const fs::path& name : names) {
        const Shot ref = load(det, refDir / name);
        const Shot ses = load(det, sessionDir / name);
        for (int s = 0; s < 2; ++s) {
            const Pair p = measureNut(ref, ses, s);
            NutResult r{name.string(), s, Status::Unmeasurable, std::nullopt, p.reason};
            if (p.rotation) setRotation(r, *p.rotation);
            rows.push_back(r);
        }
    }
    return rows;
}

NutResult measureNutIn(const cv::Mat& ref, const cv::Rect& refBox, const cv::Mat& ses, const cv::Rect& sesBox, int side) {
    Shot r, s;  // shortcut: the rail angle is measured again for each nut, not once per photo
    r.img = ref;
    r.box[side] = refBox;
    r.rail = railAngle(ref);
    s.img = ses;
    s.box[side] = sesBox;
    s.rail = railAngle(ses);
    const Pair p = measureNut(r, s, side);
    NutResult res{"", side, Status::Unmeasurable, std::nullopt, p.reason};
    if (p.rotation) setRotation(res, *p.rotation);
    return res;
}
