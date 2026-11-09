#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "check.h"
#include "quality.h"

int main() {
    const cv::Mat img = cv::imread(std::string(PROJECT_DIR) + "/Data/img/IMG_0001.JPG");
    CHECK(!img.empty());
    const cv::Mat crop = img(cv::Rect(cv::Point(1815, 763), cv::Point(2395, 1310))).clone();

    const auto sharp = rejectReason(crop);
    std::cout << "sharp crop: " << (sharp ? *sharp : "usable") << "\n";
    CHECK(!sharp);

    cv::Mat blurred;
    cv::GaussianBlur(crop, blurred, cv::Size(0, 0), 15);
    const auto blur = rejectReason(blurred);
    CHECK(blur && blur->find("blurry") != std::string::npos);

    const auto small = rejectReason(crop(cv::Rect(0, 0, 20, 20)));
    CHECK(small && small->find("too small") != std::string::npos);

    const auto flat = rejectReason(cv::Mat(300, 300, CV_8UC3, cv::Scalar(128, 128, 128)));
    CHECK(flat && flat->find("low contrast") != std::string::npos);

    const auto empty = rejectReason(cv::Mat());
    CHECK(empty && empty->find("too small") != std::string::npos);
    return 0;
}
