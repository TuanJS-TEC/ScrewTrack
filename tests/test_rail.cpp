#include <cmath>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "check.h"
#include "rail.h"

int main() {
    const cv::Mat img = cv::imread(std::string(PROJECT_DIR) + "/Data/img/IMG_0001.JPG");
    CHECK(!img.empty());

    // Labelled rail_ref lines of IMG_0001 measure 1.30 and 1.48 degrees.
    const auto base = railAngle(img);
    CHECK(base);
    std::cout << "IMG_0001 rail angle " << *base << "\n";
    CHECK(std::abs(*base - 1.4) <= 0.7);

    // warpAffine(+3) in OpenCV turns the image counter-clockwise, so -3 is +3 degrees in atan2(dy,dx), y down.
    cv::Mat rotated;
    cv::warpAffine(img, rotated, cv::getRotationMatrix2D({img.cols / 2.0f, img.rows / 2.0f}, -3.0, 1.0), img.size(),
                   cv::INTER_LINEAR, cv::BORDER_REFLECT);
    const auto turned = railAngle(rotated);
    CHECK(turned);
    std::cout << "rotated +3 deg: " << *turned << " (delta " << *turned - *base << ")\n";
    CHECK(std::abs(*turned - *base - 3.0) <= 0.5);

    CHECK(!railAngle(cv::Mat(1000, 1000, CV_8UC3, cv::Scalar(128, 128, 128))));

    // Structureless noise has no dominant line either.
    cv::Mat noise(1000, 1000, CV_8UC3);
    cv::theRNG().state = 1;
    cv::randu(noise, 0, 255);
    CHECK(!railAngle(noise));

    // Strong stripes outside +-15 degrees of horizontal must not yield an angle: one 30 degree diagonal stripe and vertical stripes.
    cv::Mat diagonal(1000, 1000, CV_8UC3, cv::Scalar(30, 30, 30)), vertical = diagonal.clone();
    const double t = std::tan(30.0 * CV_PI / 180.0);
    cv::line(diagonal, {-1000, cvRound(300 - 1000 * t)}, {2000, cvRound(300 + 2000 * t)}, cv::Scalar(220, 220, 220), 40);
    for (int x = 20; x < 1000; x += 80) cv::line(vertical, {x, 0}, {x, 999}, cv::Scalar(220, 220, 220), 12);
    CHECK(!railAngle(diagonal));
    CHECK(!railAngle(vertical));

    // 1-channel input works and matches the colour result.
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    const auto g = railAngle(gray);
    CHECK(g);
    CHECK(std::abs(*g - *base) <= 0.1);
    return 0;
}
