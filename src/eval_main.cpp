// eval_labels <model.onnx> <Data_dir>: checks detector, quality gate and hex angle against the labelme labels.
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "angle.h"
#include "detector.h"
#include "quality.h"

namespace fs = std::filesystem;

struct Shape {
    std::string label;
    int group = -1;
    std::vector<cv::Point2d> pts;
};

static std::vector<Shape> readShapes(const fs::path& json, std::string& imagePath) {
    std::vector<Shape> out;
    cv::FileStorage f(json.string(), cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!f.isOpened()) return out;
    f["imagePath"] >> imagePath;
    for (const cv::FileNode& n : f["shapes"]) {
        Shape s;
        n["label"] >> s.label;
        cv::FileNode g = n["group_id"];
        if (g.isInt()) s.group = (int)g;
        for (const cv::FileNode& p : n["points"]) s.pts.emplace_back((double)p[0], (double)p[1]);
        out.push_back(std::move(s));
    }
    return out;
}

static double iou(const cv::Rect& a, const cv::Rect& b) {
    double i = (a & b).area();
    return i / (a.area() + b.area() - i);
}

// hexAngle averages gradient directions, which are normal to the hex edges, so it reads the edge direction + 30 deg
// (mod 60). The pipeline only uses differences between two measurements, so the offset cancels; remove it here.
static const double kGradientOffset = 30.0;

// Circular mean mod 60 deg of the segment directions (atan2(dy,dx), y down), plus the gradient-normal offset.
static double labelAngle(const std::vector<cv::Point2d>& p) {
    double sx = 0, sy = 0;
    for (size_t i = 1; i < p.size(); i++) {
        double t = std::atan2(p[i].y - p[i - 1].y, p[i].x - p[i - 1].x) * 6;
        sx += std::cos(t);
        sy += std::sin(t);
    }
    return std::atan2(sy, sx) / 6 * 180 / CV_PI + kGradientOffset;
}

static void stats(const char* name, std::vector<double> e, int total) {
    if (e.empty()) { std::printf("  %-34s n=0\n", name); return; }
    std::sort(e.begin(), e.end());
    double m = 0;
    for (double v : e) m += v;
    auto q = [&](double f) { return e[std::min(e.size() - 1, (size_t)(f * (e.size() - 1) + 0.5))]; };
    int g3 = 0, g10 = 0;
    for (double v : e) { g3 += v > 3; g10 += v > 10; }
    std::printf("  %-34s n=%zu/%d mean %.2f median %.2f p90 %.2f max %.2f  >3deg: %d  >10deg: %d\n", name, e.size(), total,
                m / e.size(), q(0.5), q(0.9), e.back(), g3, g10);
}

static cv::Mat grayCrop(const cv::Mat& bgr, const cv::Rect& r) {
    cv::Mat g;
    cv::cvtColor(bgr(r), g, cv::COLOR_BGR2GRAY);
    return g;
}

static bool inside(const std::vector<cv::Point2d>& p, const cv::Rect& r) {
    for (auto& q : p)
        if (!r.contains(cv::Point((int)q.x, (int)q.y))) return false;
    return !p.empty();
}

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: eval_labels <model.onnx> <Data_dir>\n"); return 2; }
    NutDetector det(argv[1]);
    fs::path dir = argv[2];
    std::vector<fs::path> jsons;
    for (auto& e : fs::directory_iterator(dir / "js"))
        if (e.path().extension() == ".json") jsons.push_back(e.path());
    std::sort(jsons.begin(), jsons.end());

    int files = 0, noImage = 0, parseFail = 0, nuts = 0, matched = 0, noEdges = 0;
    int ringOk = 0, measured = 0, tightOk = 0;
    std::map<std::string, int> reasons;
    std::vector<double> errDet, errTight, errTightSame;  // errTightSame: tight box, on nuts the detector path accepted
    int detEdgeTotal = 0, tightEdgeTotal = 0;
    struct Unclear { std::string file; bool detected; std::string reject; bool ring; bool angle; bool tightAngle; };
    std::vector<Unclear> unclear;

    for (auto& jp : jsons) {
        std::string imgName;
        std::vector<Shape> shapes;
        try {
            shapes = readShapes(jp, imgName);
        } catch (const cv::Exception&) {  // truncated/malformed JSON: skip and count
            parseFail++;
            continue;
        }
        if (shapes.empty() || imgName.empty()) { parseFail++; continue; }
        cv::Mat img = cv::imread((dir / "img" / imgName).string());
        if (img.empty()) { noImage++; continue; }
        files++;
        std::vector<cv::Rect> boxes = det.detect(img);
        const cv::Rect full(0, 0, img.cols, img.rows);

        // Runs the detector-box path on a label rect; returns nullopt if no detector box matches.
        struct Res { std::string reject; bool ring = false; std::optional<double> angle; };
        auto viaDetector = [&](const cv::Rect& lab) -> std::optional<Res> {
            const cv::Rect* best = nullptr;
            double bi = 0.5;
            for (auto& b : boxes)
                if (iou(b, lab) >= bi) { bi = iou(b, lab); best = &b; }
            if (!best) return std::nullopt;
            Res r;
            if (auto why = rejectReason(img(*best))) r.reject = *why;
            cv::Mat g = grayCrop(img, *best);
            if (auto ring = selectRing(g)) { r.ring = true; r.angle = hexAngleAt(g, *ring); }
            return r;
        };

        for (auto& s : shapes) {
            if (s.label == "nut_unclear" && s.pts.size() == 2) {
                cv::Rect lab = cv::Rect(cv::Point((int)s.pts[0].x, (int)s.pts[0].y), cv::Point((int)s.pts[1].x, (int)s.pts[1].y)) & full;
                Unclear u{jp.filename().string(), false, "", false, false, false};
                if (auto r = viaDetector(lab)) { u.detected = true; u.reject = r->reject; u.ring = r->ring; u.angle = r->angle.has_value(); }
                if (lab.area() > 0) u.tightAngle = hexAngle(grayCrop(img, lab)).has_value();
                unclear.push_back(u);
                continue;
            }
            if (s.label != "nut" || s.pts.size() != 2) continue;
            nuts++;
            cv::Rect lab = cv::Rect(cv::Point((int)s.pts[0].x, (int)s.pts[0].y), cv::Point((int)s.pts[1].x, (int)s.pts[1].y)) & full;
            if (lab.area() <= 0) continue;
            const Shape* edges = nullptr;
            for (auto& e : shapes)
                if (e.label == "nut_edges" && e.group == s.group && e.pts.size() >= 2 && inside(e.pts, lab)) { edges = &e; break; }
            if (!edges) noEdges++;
            double la = edges ? labelAngle(edges->pts) : 0;

            std::optional<double> tight = hexAngle(grayCrop(img, lab));
            if (tight) tightOk++;
            if (edges) {
                tightEdgeTotal++;
                if (tight) errTight.push_back(std::abs(wrap60(*tight - la)));
            }

            auto r = viaDetector(lab);
            if (!r) continue;
            matched++;
            if (!r->reject.empty()) reasons[r->reject]++;
            if (r->ring) ringOk++;
            if (r->ring && r->reject.empty()) measured++;
            if (edges) {
                detEdgeTotal++;
                if (r->angle) {
                    errDet.push_back(std::abs(wrap60(*r->angle - la)));
                    if (tight) errTightSame.push_back(std::abs(wrap60(*tight - la)));
                }
            }
        }
    }

    std::printf("files %d (skipped: %d no image, %d unparsable)\n", files, noImage, parseFail);
    std::printf("(a) detection: %d/%d labelled nuts have a detector box with IoU>=0.5 (%.1f%%)\n", matched, nuts, nuts ? 100.0 * matched / nuts : 0.0);
    std::printf("(b) rejectReason on detector-box crop (of %d matched):", matched);
    int rej = 0;
    for (auto& [k, v] : reasons) { std::printf(" '%s'=%d", k.c_str(), v); rej += v; }
    std::printf("%s rejected=%d usable=%d\n", reasons.empty() ? " none" : "", rej, matched - rej);
    std::printf("(c) selectRing+hexAngleAt on detector-box crop: accepted %d/%d; accepted and not rejected by quality gate (pipeline measures) %d/%d\n",
                ringOk, matched, measured, matched);
    std::printf("    nuts with no nut_edges label inside their box (excluded from angle stats): %d of %d\n", noEdges, nuts);
    std::printf("    angle error |wrap60(measured - label)| in degrees, nuts with an edge label:\n");
    stats("detector box (selectRing+hexAngleAt)", errDet, detEdgeTotal);
    stats("tight labelled box (hexAngle)", errTight, tightEdgeTotal);
    stats("tight box, same nuts as detector row", errTightSame, detEdgeTotal);
    std::printf("(d) tight labelled box hexAngle accepted: %d/%d\n", tightOk, nuts);
    std::printf("(e) nut_unclear: %zu\n", unclear.size());
    for (auto& u : unclear)
        std::printf("    %s: detector box %s, quality reject '%s', ring %s, detector-path angle %s; tight-box hexAngle %s\n", u.file.c_str(),
                    u.detected ? "yes" : "no", u.reject.empty() ? "-" : u.reject.c_str(), u.ring ? "selected" : "none",
                    u.angle ? "GIVEN" : "none", u.tightAngle ? "GIVEN" : "none");
    return 0;
}
