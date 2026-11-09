#pragma once
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct Detection {
    cv::Rect box;
    float score;
};

class NutDetector {
public:
    explicit NutDetector(const std::string& onnxPath);
    // Class-3 (nut) boxes, confidence >= 0.25, clamped to the image.
    std::vector<cv::Rect> detect(const cv::Mat& bgr) const;
    // For tracking: screw spike (class 3) and abnormal screw spike (class 1) with their score, >= minConf.
    std::vector<Detection> detectScored(const cv::Mat& bgr, float minConf) const;

private:
    std::vector<Detection> run(const cv::Mat& bgr, float minConf, const std::vector<int>& classes) const;
    mutable cv::dnn::Net net_;
};

// 0 if the box centre is in the top half of the image, else 1.
int sideOf(const cv::Rect& box, int imageHeight);
