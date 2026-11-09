#include "angle.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <vector>
#include <opencv2/imgproc.hpp>

#include "types.h"

double wrap60(double deg) {
    double r = std::fmod(deg + 30.0, 60.0);
    if (r < 0) r += 60.0;
    if (r >= 60.0) r -= 60.0;  // -tiny + 60 rounds to 60
    return r - 30.0;
}

namespace {
// 0.035: Gaussian-noise images (60 tried) peak at 0.028; the weak rusty IMG_0001 nut gives 0.043.
constexpr double kMinStrength = 0.035;
// Picking the best of 294 rings lets noise pass the strength rule: noise z peaks at ~10.2 (40 images), accepted
// detector-box nuts have z >= 16.6. shortcut: z grows with crop size; recalibrate if crops leave ~500-700 px.
constexpr double kMinZ = 13.0;

// Gradient field of a crop, computed once; h6 is each pixel's 6-fold vote, magnitude * exp(6i * direction).
struct Field {
    cv::Mat mag, ang;  // CV_32F; ang = atan2(gy, gx) in radians, y down
    std::vector<std::complex<double>> h6;
};

Field gradientField(const cv::Mat& gray) {
    CV_Assert(gray.type() == CV_8UC1);
    // Sigma 4 px suppresses rust grain and JPEG texture, which otherwise dominate the gradients.
    Field g;
    cv::Mat f, gx, gy;
    gray.convertTo(f, CV_32F);
    cv::GaussianBlur(f, f, cv::Size(), 4.0);
    cv::Sobel(f, gx, CV_32F, 1, 0);
    cv::Sobel(f, gy, CV_32F, 0, 1);
    cv::cartToPolar(gx, gy, g.mag, g.ang);
    const float* m = g.mag.ptr<float>();
    const float* a = g.ang.ptr<float>();
    g.h6.resize(g.mag.total());
    for (size_t i = 0; i < g.h6.size(); ++i) g.h6[i] = double(m[i]) * std::polar(1.0, 6.0 * double(a[i]));
    return g;
}

struct Annulus {
    double cx, cy, rIn, rOut;
};

// Ring 0.38..0.47 of `side` around a centre: on a tight nut box (side = short side) the hex flats (apothem ~0.39-0.42,
// corners ~0.45-0.48) cross it; the centre is the round bolt head and the rim holds clip and stones.
Annulus nutRing(double cx, double cy, double side) { return {cx, cy, 0.38 * side, 0.47 * side}; }

Annulus toAnnulus(const Ring& r) { return {r.centre.x, r.centre.y, r.rIn, r.rOut}; }

// Calls f(pixel index) for every field pixel on the ring; parts outside the crop are skipped.
template <class F>
void forRing(const Field& g, const Annulus& r, F&& f) {
    const double rIn2 = std::pow(r.rIn, 2), rOut2 = std::pow(r.rOut, 2);
    const int cols = g.mag.cols;
    const int y0 = std::max(0, int(std::floor(r.cy - r.rOut))), y1 = std::min(g.mag.rows - 1, int(std::ceil(r.cy + r.rOut)));
    const int x0 = std::max(0, int(std::floor(r.cx - r.rOut))), x1 = std::min(cols - 1, int(std::ceil(r.cx + r.rOut)));
    for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x) {
            const double d2 = (x - r.cx) * (x - r.cx) + (y - r.cy) * (y - r.cy);
            if (d2 >= rIn2 && d2 <= rOut2) f(size_t(y) * cols + x);
        }
}

struct Vote {
    std::complex<double> total;  // sum of 6-fold votes
    double strength;             // |total| / sum of magnitudes, 0..1
    double z;                    // |total| / sqrt(sum of squared magnitudes): significance, ~1 for random directions
};

Vote ringVote(const Field& g, const Annulus& r) {
    const float* m = g.mag.ptr<float>();
    std::complex<double> total;
    double sum = 0, sum2 = 0;
    forRing(g, r, [&](size_t i) { total += g.h6[i]; sum += m[i]; sum2 += double(m[i]) * m[i]; });
    return {total, sum > 0 ? std::abs(total) / sum : 0.0, sum2 > 0 ? std::abs(total) / std::sqrt(sum2) : 0.0};
}

// A straight bar or edge also has a full 6th harmonic, but from one gradient direction only. A hex
// has three flat directions (mod 180): split the 6-fold energy among them, each must carry its share.
bool balanced(const Field& g, const Annulus& r, std::complex<double> total) {
    const float* m = g.mag.ptr<float>();
    const float* a = g.ang.ptr<float>();
    const double t0 = std::arg(total) / 6.0;
    double share[3] = {0, 0, 0};  // sums to |total|
    forRing(g, r, [&](size_t i) {
        const int j = ((int(std::lround((a[i] - t0) / (CV_PI / 3))) % 3) + 3) % 3;
        share[j] += m[i] * std::cos(6.0 * (a[i] - t0));
    });
    // 0.3 of an even split: bars at any angle give 0, the hexagon 1.0, IMG_0001 0.65; keeps 92/134 labelled nuts.
    return 3 * std::min({share[0], share[1], share[2]}) >= 0.3 * std::abs(total);
}

// hexAngle's acceptance rules on one ring; the folded angle when they pass.
std::optional<double> angleOn(const Field& g, const Annulus& r) {
    const Vote v = ringVote(g, r);
    if (v.strength < kMinStrength || !balanced(g, r, v.total)) return std::nullopt;
    return wrap60(std::arg(v.total) * 180.0 / CV_PI / 6.0);
}
}  // namespace

std::optional<double> hexAngle(const cv::Mat& gray) {
    return angleOn(gradientField(gray), nutRing(gray.cols / 2.0, gray.rows / 2.0, std::min(gray.cols, gray.rows)));
}

std::optional<Ring> selectRing(const cv::Mat& gray) {
    const Field g = gradientField(gray);
    const double side = std::min(gray.cols, gray.rows);
    struct Candidate {
        double z;
        int off2;  // squared grid distance of the centre from the crop centre
        Ring ring;
    };
    std::vector<Candidate> cands;
    // Centres on a 0.05-side grid within +-0.15 of the short side; detector boxes are 1.0-1.35x the nut.
    for (double scale = 0.75; scale <= 1.001; scale += 0.05)
        for (int iy = -3; iy <= 3; ++iy)
            for (int ix = -3; ix <= 3; ++ix) {
                const Annulus a = nutRing(gray.cols / 2.0 + ix * 0.05 * side, gray.rows / 2.0 + iy * 0.05 * side, scale * side);
                const Ring r{cv::Point2f(float(a.cx), float(a.cy)), float(a.rIn), float(a.rOut)};
                // z, not strength: strength favours small rings, whose few pixels make it noisy.
                cands.push_back({ringVote(g, toAnnulus(r)).z, ix * ix + iy * iy, r});
            }
    const Candidate* top = &*std::max_element(
        cands.begin(), cands.end(), [](const Candidate& a, const Candidate& b) { return a.z < b.z; });
    // Accept only when the most significant ring itself passes: taking the best ring that passes, at any score,
    // accepted all 155 detector boxes with 3 errors > 10 deg.
    const auto topDeg = top->z >= kMinZ ? angleOn(g, toAnnulus(top->ring)) : std::nullopt;
    if (!topDeg) return std::nullopt;
    // Rings within 5% of the top score are a toss-up between the nut and the nut plus nearby edges; prefer the
    // most central (then the largest): an off-centre fixed ring drifts ~0.25 deg per degree of turn.
    // The alternative must also read the same hex (within 2 deg): without that, IMG_0075 swapped to a central
    // ring 11.5 deg off its label.
    const Candidate* best = top;
    for (const Candidate& c : cands) {
        const bool better = c.off2 < best->off2 || (c.off2 == best->off2 && c.ring.rOut > best->ring.rOut);
        if (c.z < 0.95 * top->z || !better) continue;
        const auto deg = angleOn(g, toAnnulus(c.ring));
        if (deg && std::abs(wrap60(*deg - *topDeg)) <= 2.0) best = &c;
    }
    return best->ring;
}

std::optional<double> hexAngleAt(const cv::Mat& gray, const Ring& ring) { return angleOn(gradientField(gray), toAnnulus(ring)); }

namespace {
constexpr int kArcs = 24;  // 15-degree arcs; a left-out window is 3 arcs (45 deg)
// Calibrated on 122 sample nuts, synthetic 0-6 deg turns vs strips/sticks/leaves over 5-40% of the box: 0.55 keeps
// 94% of clean turns, and no covered unturned nut then reads more than 1.21 deg (1304 of 9086 read >= 3 without it).
// 0.5 failed the r=1.0 disc 4 deg pipeline test (0.54); 0.6 kept a covered 3 deg turn 2.9 deg off.
constexpr double kMaxArcShift = 0.55;

// 6-fold vote per 15-degree arc of the ring, by position angle about the ring centre, arcs starting at startDeg.
std::array<std::complex<double>, kArcs> arcVotes(const Field& g, const Annulus& r, double startDeg) {
    std::array<std::complex<double>, kArcs> t{};
    const int cols = g.mag.cols;
    forRing(g, r, [&](size_t i) {
        const double pos = std::atan2(double(i / cols) - r.cy, double(i % cols) - r.cx) * 180.0 / CV_PI + 180.0 - startDeg;
        t[size_t(std::fmod(pos + 720.0, 360.0) / (360.0 / kArcs)) % kArcs] += g.h6[i];
    });
    return t;
}

// Turn (deg) from vote a to vote b, folded into (-30, 30].
double turnOf(std::complex<double> a, std::complex<double> b) { return std::arg(b * std::conj(a)) * 180.0 / CV_PI / 6.0; }
}  // namespace

double arcShift(const cv::Mat& refGray, const cv::Mat& sesGray, const Ring& ring, double& turn) {
    const Annulus a = toAnnulus(ring);
    const Field gr = gradientField(refGray), gs = gradientField(sesGray);
    const std::complex<double> totR = ringVote(gr, a).total, totS = ringVote(gs, a).total;
    turn = turnOf(totR, totS);
    // Session arcs follow the turn, so a window leaves out the same part of the nut in both photos: with fixed
    // arcs a clean 4 deg turn alone moved the estimate by 1.3 deg (95th percentile), with following arcs 0.5.
    const auto arcR = arcVotes(gr, a, 0.0), arcS = arcVotes(gs, a, turn);
    double worst = 0;
    for (int j = 0; j < kArcs; ++j) {
        std::complex<double> r = totR, s = totS;
        for (int k = j; k < j + 3; ++k) {
            r -= arcR[k % kArcs];
            s -= arcS[k % kArcs];
        }
        worst = std::max(worst, std::abs(wrap60(turnOf(r, s) - turn)));
    }
    return worst;
}

bool consistentTurn(const cv::Mat& refGray, const cv::Mat& sesGray, const Ring& ring) {
    double turn;
    const double shift = arcShift(refGray, sesGray, ring, turn);
    // Calibrated on 123 nuts turned 0-30 deg (clean and under a random bar): with margin 4 the old gate's 13-87 of
    // 123..77 clean turns at 6-30 deg become 76-118, and no covered unturned nut (156) reads loose, no covered
    // turned nut (6-30 deg) reads ok. Margin 3 let 2 covered unturned nuts read loose.
    return shift <= kMaxArcShift || std::abs(turn) - shift >= kLooseDeg + 1.0;
}
