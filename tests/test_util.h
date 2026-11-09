#pragma once
#include <algorithm>
#include <opencv2/imgproc.hpp>

// Rotates the circular patch (diameter = radiusScale * min box side) about the box centre, in place.
// Sign: getRotationMatrix2D(-deg) turns the patch clockwise on screen, i.e. +deg in atan2(dy,dx) with y down.
inline void rotateNutInPlace(cv::Mat& img, cv::Rect box, double deg, double radiusScale = 1.0) {
    const cv::Point2f c(box.x + box.width / 2.0f, box.y + box.height / 2.0f);
    cv::Mat rotated, mask = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::warpAffine(img, rotated, cv::getRotationMatrix2D(c, -deg, 1.0), img.size(), cv::INTER_CUBIC);
    cv::circle(mask, c, int(radiusScale * std::min(box.width, box.height) / 2), 255, cv::FILLED);
    rotated.copyTo(img, mask);
}
