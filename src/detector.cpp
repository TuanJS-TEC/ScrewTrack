#include "detector.h"

#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace {
constexpr int kInput = 640;
constexpr int kNutClass = 3;
constexpr int kAbnormalNutClass = 1;
constexpr float kConf = 0.25f;
constexpr float kNmsIou = 0.45f;
}  // namespace

NutDetector::NutDetector(const std::string& onnxPath) {
    try {
        net_ = cv::dnn::readNetFromONNX(onnxPath);
    } catch (const cv::Exception&) {  // missing or unreadable file: report it below with the export hint
    }
    if (net_.empty())
        throw std::runtime_error("cannot load ONNX model: " + onnxPath +
                                 " (export it with: .venv/bin/yolo export model=model/best.pt format=onnx imgsz=640, see README.md)");
}

std::vector<cv::Rect> NutDetector::detect(const cv::Mat& bgr) const {
    std::vector<cv::Rect> boxes;
    for (const Detection& d : run(bgr, kConf, {kNutClass})) boxes.push_back(d.box);
    return boxes;
}

std::vector<Detection> NutDetector::detectScored(const cv::Mat& bgr, float minConf) const {
    return run(bgr, minConf, {kNutClass, kAbnormalNutClass});
}

std::vector<Detection> NutDetector::run(const cv::Mat& bgr, float minConf, const std::vector<int>& classes) const {
    // Letterbox: scale to fit 640x640, centre, pad with grey 114.
    const float scale = std::min(float(kInput) / bgr.cols, float(kInput) / bgr.rows);
    const int w = int(std::round(bgr.cols * scale)), h = int(std::round(bgr.rows * scale));
    const int padX = (kInput - w) / 2, padY = (kInput - h) / 2;
    cv::Mat resized, canvas(kInput, kInput, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(bgr, resized, {w, h}, 0, 0, cv::INTER_LINEAR);
    resized.copyTo(canvas(cv::Rect(padX, padY, w, h)));

    net_.setInput(cv::dnn::blobFromImage(canvas, 1.0 / 255.0, {kInput, kInput}, cv::Scalar(), true, false));
    cv::Mat out = net_.forward();  // [1, 4+classes, anchors]
    const int rows = out.size[1], anchors = out.size[2];
    if (rows <= 4 + kNutClass) throw std::runtime_error("unexpected ONNX output shape");
    const float* d = out.ptr<float>();
    auto at = [&](int r, int a) { return d[r * anchors + a]; };

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    for (int a = 0; a < anchors; ++a) {
        float conf = 0;
        for (int c : classes) conf = std::max(conf, at(4 + c, a));
        if (conf < minConf) continue;
        const float cx = at(0, a), cy = at(1, a), bw = at(2, a), bh = at(3, a);
        boxes.emplace_back(int(std::round((cx - bw / 2 - padX) / scale)),
                           int(std::round((cy - bh / 2 - padY) / scale)),
                           int(std::round(bw / scale)), int(std::round(bh / scale)));
        scores.push_back(conf);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, minConf, kNmsIou, keep);

    std::vector<Detection> result;
    const cv::Rect image(0, 0, bgr.cols, bgr.rows);
    for (int i : keep) {
        const cv::Rect r = boxes[i] & image;
        if (!r.empty()) result.push_back({r, scores[i]});
    }
    return result;
}

int sideOf(const cv::Rect& box, int imageHeight) {
    return (box.y + box.height / 2.0 < imageHeight / 2.0) ? 0 : 1;
}
