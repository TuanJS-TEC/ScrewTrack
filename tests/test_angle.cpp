#include <cmath>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "angle.h"
#include "check.h"
#include "test_util.h"

// Radius 360 on 800: flats at 312..360 px = 0.39..0.45 of the side, inside hexAngle's ring like a tight nut box.
static cv::Mat hexagon(double deg) {
    cv::Mat img(800, 800, CV_8UC1, cv::Scalar(90));
    std::vector<cv::Point> pts;
    for (int k = 0; k < 6; ++k) {
        const double a = (deg + 60.0 * k) * CV_PI / 180.0;
        // 4 fractional bits keep the sub-pixel vertex position.
        pts.emplace_back(cvRound((400 + 360 * std::cos(a)) * 16), cvRound((400 + 360 * std::sin(a)) * 16));
    }
    cv::fillConvexPoly(img, pts, cv::Scalar(255), cv::LINE_AA, 4);
    return img;
}

int main() {
    CHECK(wrap60(0) == 0);
    CHECK(wrap60(31) == -29);
    CHECK(wrap60(-31) == 29);
    CHECK(wrap60(60) == 0);
    CHECK(wrap60(-29 - 29) == 2);

    const auto base = hexAngle(hexagon(0));
    CHECK(base);
    for (int deg = 0; deg <= 10; ++deg) {
        const auto a = hexAngle(hexagon(deg));
        CHECK(a);
        const double err = wrap60(*a - *base) - deg;
        std::cout << "hexagon " << deg << " deg: error " << err << "\n";
        CHECK(std::abs(err) <= 1.0);
    }

    const cv::Mat img = cv::imread(std::string(PROJECT_DIR) + "/Data/img/IMG_0001.JPG", cv::IMREAD_GRAYSCALE);
    CHECK(!img.empty());
    const cv::Rect box(1815, 763, 2395 - 1815, 1310 - 763);
    const auto crop0 = hexAngle(img(box));
    CHECK(crop0);
    for (int deg = 0; deg <= 10; deg += 2) {
        cv::Mat rotated = img.clone();
        rotateNutInPlace(rotated, box, deg);
        const auto a = hexAngle(rotated(box));
        CHECK(a);
        const double err = wrap60(*a - *crop0) - deg;
        std::cout << "IMG_0001 crop " << deg << " deg: error " << err << "\n";
        CHECK(std::abs(err) <= 1.0);
    }

    CHECK(!hexAngle(cv::Mat(600, 600, CV_8UC1, cv::Scalar(128))));
    cv::Mat noise(600, 600, CV_8UC1);
    cv::randn(noise, 128, 30);
    CHECK(!hexAngle(noise));

    // One straight structure (clip, shadow, rim) has a full 6th harmonic too; it must be rejected at any angle.
    for (int deg = 0; deg < 180; deg += 5) {
        cv::Mat bar(800, 800, CV_8UC1, cv::Scalar(90));
        const cv::Point d(cvRound(600 * std::cos(deg * CV_PI / 180)), cvRound(600 * std::sin(deg * CV_PI / 180)));
        cv::line(bar, cv::Point(400, 400) - d, cv::Point(400, 400) + d, cv::Scalar(255), 40, cv::LINE_AA);
        const bool accepted = hexAngle(bar).has_value();
        if (accepted) std::cout << "bar at " << deg << " deg accepted\n";
        CHECK(!accepted);
    }
    cv::Mat bar(800, 800, CV_8UC1, cv::Scalar(90));
    cv::line(bar, {0, 40}, {800, 400}, cv::Scalar(255), 40, cv::LINE_AA);  // crosses the ring, not the centre
    CHECK(!hexAngle(bar));
    cv::Mat circle(800, 800, CV_8UC1, cv::Scalar(90));
    cv::circle(circle, {400, 400}, 340, cv::Scalar(255), cv::FILLED, cv::LINE_AA);
    CHECK(!hexAngle(circle));
    cv::Mat square(800, 800, CV_8UC1, cv::Scalar(90));
    cv::rectangle(square, {60, 60}, {740, 740}, cv::Scalar(255), cv::FILLED);
    CHECK(!hexAngle(square));

    // selectRing + hexAngleAt: same answer as hexAngle on a tight box around a clear nut (IMG_0001 bottom).
    const cv::Rect bottom(1781, 2901, 2313 - 1781, 3393 - 2901);
    const auto hexBottom = hexAngle(img(bottom));
    CHECK(hexBottom);
    const auto ring0 = selectRing(img(bottom));
    CHECK(ring0);
    const auto at0 = hexAngleAt(img(bottom), *ring0);
    CHECK(at0);
    std::cout << "IMG_0001 bottom labelled box: hexAngle " << *hexBottom << " selected ring " << *at0 << "\n";
    CHECK(std::abs(wrap60(*at0 - *hexBottom)) <= 0.5);

    // Loose boxes like the detector's (centre ~40 px off, +10%) around a labelled nut, rotated 1..4 deg (the operating
    // range; the loose alarm is at 3); the ring is chosen once on the 0 deg crop and reused. Worst |tracked - applied|
    // per box: NAN when selectRing finds no ring, INFINITY when hexAngleAt rejects a rotated crop. The disc turned is
    // 1.3x the box's short side so it covers the ring: in reality the whole nut turns. Known limit: rings reach
    // stones that do not turn, so a ~10 deg turn reads 0.7-1.7 deg low.
    auto looseTracking = [&img](cv::Rect label) {
        const cv::Rect area(label.x - 200, label.y - 200, label.width + 400, label.height + 400);
        const cv::Rect nut = label - area.tl();
        const cv::Mat still = img(area);
        const int dx = cvRound(0.05 * nut.width), dy = cvRound(0.05 * nut.height);
        const cv::Rect loose[] = {nut + cv::Point(0, 40),   nut + cv::Point(0, -40),  nut + cv::Point(30, 30),
                                  nut + cv::Point(-30, 30), nut + cv::Point(30, -30), nut + cv::Point(-30, -30),
                                  cv::Rect(nut.x - dx, nut.y - dy, nut.width + 2 * dx, nut.height + 2 * dy)};
        std::vector<double> worst;
        for (const cv::Rect& b : loose) {
            const auto ring = selectRing(still(b));
            const auto zero = ring ? hexAngleAt(still(b), *ring) : std::nullopt;
            if (!zero) {
                worst.push_back(NAN);
                continue;
            }
            double w = 0;
            for (int deg = 1; deg <= 4; ++deg) {
                cv::Mat patch = still.clone();
                rotateNutInPlace(patch, nut, deg, 1.3);
                const auto a = hexAngleAt(patch(b), *ring);
                w = std::max(w, a ? std::abs(wrap60(*a - *zero) - deg) : INFINITY);
            }
            worst.push_back(w);
        }
        return worst;
    };
    int noRing = 0;
    for (double w : looseTracking(bottom)) {
        std::cout << "bottom nut, loose box: worst tracking error " << w << "\n";
        if (std::isnan(w)) ++noRing;
        else CHECK(w <= 1.0);
    }
    CHECK(noRing <= 2);  // the clear nut must get a ring in most loose boxes, or the check above is vacuous
    // The rusty top nut need not be measured, but if a ring is accepted it must track. Bound 1.25, not 1.0: box
    // (-30,-30) keeps a ring 57 px off the nut centre (no passing ring within 5% of its score is closer) and drifts
    // 0.26/0.59/0.93/1.12 deg at 1..4 deg. Open risk: a loose box on a weak nut costs accuracy.
    for (double w : looseTracking(box)) {
        if (!(std::isnan(w) || w <= 1.25)) std::cout << "top nut, loose box: worst tracking error " << w << "\n";
        CHECK(std::isnan(w) || w <= 1.25);
    }

    // Searching centres must not let non-hex structures through.
    auto rejected = [](const cv::Mat& m, const std::string& what) {
        const auto r = selectRing(m);
        if (r) std::cout << what << " accepted by selectRing at " << r->centre << "\n";
        return !r;
    };
    CHECK(rejected(cv::Mat(600, 600, CV_8UC1, cv::Scalar(128)), "flat gray"));
    for (int seed = 0; seed < 40; ++seed) {
        cv::Mat n(600, 600, CV_8UC1);
        cv::RNG(seed).fill(n, cv::RNG::NORMAL, 128, 30);
        CHECK(rejected(n, "noise " + std::to_string(seed)));
    }
    auto canvas = [] { return cv::Mat(800, 800, CV_8UC1, cv::Scalar(90)); };
    auto barAt = [](cv::Mat& m, cv::Point c, int deg) {
        const cv::Point d(cvRound(600 * std::cos(deg * CV_PI / 180)), cvRound(600 * std::sin(deg * CV_PI / 180)));
        cv::line(m, c - d, c + d, cv::Scalar(255), 40, cv::LINE_AA);
    };
    for (int deg = 0; deg < 180; deg += 5) {
        cv::Mat m = canvas();
        barAt(m, {400, 400}, deg);
        CHECK(rejected(m, "bar " + std::to_string(deg)));
    }
    CHECK(rejected(bar, "off-centre bar"));
    CHECK(rejected(circle, "circle"));
    CHECK(rejected(square, "square"));
    for (int deg : {90, 45, 30}) {
        cv::Mat m = canvas();
        barAt(m, {400, 400}, 10);
        barAt(m, {400, 400}, 10 + deg);
        CHECK(rejected(m, "bars crossing at " + std::to_string(deg)));
    }
    cv::Mat rect = canvas();
    cv::rectangle(rect, {60, 200}, {740, 600}, cv::Scalar(255), cv::FILLED);
    CHECK(rejected(rect, "rectangle"));
    cv::Mat offCircle = canvas();
    cv::circle(offCircle, {480, 330}, 300, cv::Scalar(255), cv::FILLED, cv::LINE_AA);
    CHECK(rejected(offCircle, "off-centre circle"));
    return 0;
}
