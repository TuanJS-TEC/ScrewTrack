#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "check.h"
#include "detector.h"

static double iou(const cv::Rect& a, const cv::Rect& b) {
    const double i = (a & b).area();
    return i / (a.area() + b.area() - i);
}

int main() {
    const std::string root = PROJECT_DIR;
    // A missing model names the export command instead of a raw cv::Exception.
    try {
        NutDetector missing(root + "/model/no_such_model.onnx");
        CHECK(false);
    } catch (const std::runtime_error& e) {
        CHECK(std::string(e.what()).find("export it with") != std::string::npos);
    }
    // Throws the same message (and the test aborts with it) when model/best.onnx was not exported.
    NutDetector det(root + "/model/best.onnx");

    CHECK(sideOf(cv::Rect(0, 10, 50, 50), 100) == 0);
    CHECK(sideOf(cv::Rect(0, 70, 50, 50), 100) == 1);

    cv::Mat img = cv::imread(root + "/Data/img/IMG_0001.JPG");
    CHECK(!img.empty());
    const cv::Rect nutTop(1815, 763, 2395 - 1815, 1310 - 763);
    const cv::Rect nutBottom(1781, 2900, 2312 - 1781, 3393 - 2900);

    auto boxes = det.detect(img);
    for (auto& b : boxes) std::cout << "IMG_0001 box " << b << "\n";
    CHECK(boxes.size() >= 2);
    for (const cv::Rect& gt : {nutTop, nutBottom}) {
        double best = 0;
        for (auto& b : boxes) best = std::max(best, iou(b, gt));
        std::cout << "best IoU " << best << "\n";
        CHECK(best >= 0.7);
        CHECK(sideOf(gt, img.rows) == (gt == nutTop ? 0 : 1));
    }

    // Shift so the top nut touches x=0; every box must stay inside the image.
    cv::Mat shifted, M = (cv::Mat_<double>(2, 3) << 1, 0, -nutTop.x, 0, 1, 0);
    cv::warpAffine(img, shifted, M, img.size());
    auto edgeBoxes = det.detect(shifted);
    CHECK(!edgeBoxes.empty());
    const cv::Rect image(0, 0, shifted.cols, shifted.rows);
    for (auto& b : edgeBoxes) CHECK((b & image) == b);

    cv::Mat img50 = cv::imread(root + "/Data/img/IMG_0050.JPG");
    CHECK(!img50.empty());
    auto b50 = det.detect(img50);
    for (auto& b : b50) std::cout << "IMG_0050 box " << b << "\n";
    CHECK(b50.size() == 1);
    CHECK(sideOf(b50[0], img50.rows) == 1);
    return 0;
}
