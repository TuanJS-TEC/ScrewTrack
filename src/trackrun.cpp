#include "trackrun.h"

#include <algorithm>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <stdexcept>

#include "pipeline.h"

namespace fs = std::filesystem;

namespace {
constexpr float kImageConf = 0.25f;  // as NutDetector::detect: at 0.5 IMG_0001's top nut and all of IMG_0050 are lost
// A registered nut is looked for this low near its old place. IMG_0005's bottom nut scored 0.28, then 0.25 once the
// nut was turned, so it flipped in and out at 0.25. Below 0.10 most boxes are not nuts.
constexpr float kAnchorConf = 0.10f;
constexpr double kAnchorIou = 0.2;  // shortcut: the nut may move about a quarter of its size; upgrade if photos drift more

bool isImage(const fs::path& p) {
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return std::tolower(c); });
    return e == ".jpg" || e == ".jpeg" || e == ".png";
}

double iou(const cv::Rect& a, const cv::Rect& b) {
    const double i = (a & b).area();
    return i / (a.area() + b.area() - i);
}

void note(TrackRun& run, const std::vector<TrackedBox>& tracks, int imageHeight, int photo) {
    for (const TrackedBox& t : tracks) {
        run.seen.insert(t.id);
        run.first.try_emplace(t.id, RegNut{t.id, sideOf(t.box, imageHeight), t.box.x, t.box.y, t.box.width, t.box.height, photo});
    }
}

TrackRun runVideo(const NutDetector& det, const std::string& path, const FrameFn& onFrame) {
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) throw std::runtime_error("cannot open video: " + path);
    TrackRun run;
    ByteTracker tracker;
    cv::Mat frame;
    for (int f = 0; cap.read(frame); ++f) {
        const auto tracks = tracker.update(det.detectScored(frame, kTrackLow), frame.rows);
        note(run, tracks, frame.rows, -1);
        if (!onFrame(f, frame, tracks)) break;
    }
    return run;
}

TrackRun runImages(const NutDetector& det, std::vector<fs::path> files, const FrameFn& onFrame, const Registry& reg) {
    std::sort(files.begin(), files.end());
    TrackRun run;
    for (const fs::path& f : files) run.photos.push_back(fs::absolute(f).string());
    const bool anchored = !reg.photos.empty();
    int nextId = 1;
    if (anchored)
        for (const RegNut& n : reg.nuts) nextId = std::max(nextId, n.id + 1);

    for (int f = 0; f < int(files.size()); ++f) {
        const cv::Mat img = cv::imread(files[f].string());
        // Skipping it would shift every later id.
        if (img.empty()) throw std::runtime_error("cannot read image: " + files[f].string());
        const std::vector<Detection> dets = det.detectScored(img, anchored ? kAnchorConf : kImageConf);
        std::vector<char> used(dets.size());
        std::vector<TrackedBox> tracks;
        if (anchored)
            for (const RegNut& n : reg.nuts) {
                if (n.photo != f) continue;
                int best = -1;
                double bestIou = kAnchorIou;
                for (int i = 0; i < int(dets.size()); ++i) {
                    if (used[i] || sideOf(dets[i].box, img.rows) != n.side) continue;
                    if (const double v = iou(dets[i].box, cv::Rect(n.x, n.y, n.w, n.h)); v >= bestIou) best = i, bestIou = v;
                }
                if (best >= 0) {
                    used[best] = 1;
                    tracks.push_back({n.id, dets[best].box, dets[best].score});
                }
            }
        std::vector<Detection> rest;  // not a registered nut, and sure enough to count
        for (int i = 0; i < int(dets.size()); ++i)
            if (!used[i] && dets[i].score >= kImageConf) rest.push_back(dets[i]);
        std::sort(rest.begin(), rest.end(), [&](const Detection& a, const Detection& b) {
            return std::make_pair(sideOf(a.box, img.rows), a.box.x) < std::make_pair(sideOf(b.box, img.rows), b.box.x);
        });
        for (const Detection& d : rest) tracks.push_back({nextId++, d.box, d.score});
        note(run, tracks, img.rows, f);
        if (!onFrame(f, img, tracks)) break;
    }
    return run;
}
}  // namespace

TrackRun runSource(const NutDetector& det, const std::string& path, const FrameFn& onFrame, const Registry& registered) {
    std::vector<fs::path> files;
    if (fs::is_directory(path)) {
        for (const auto& e : fs::directory_iterator(path))
            if (e.is_regular_file() && isImage(e.path())) files.push_back(e.path());
        if (files.empty()) throw std::runtime_error("no images (*.jpg, *.jpeg, *.png) in folder: " + path);
        return runImages(det, files, onFrame, registered);
    }
    if (isImage(path)) return runImages(det, {path}, onFrame, registered);
    return runVideo(det, path, onFrame);
}

std::map<int, NutResult> measureRotations(const Registry& reg, const cv::Mat& ses, const std::vector<TrackedBox>& tracks) {
    std::map<int, NutResult> out;
    std::map<int, cv::Mat> refs;  // each registered photo is read once per call
    for (const TrackedBox& t : tracks) {
        const auto n = std::find_if(reg.nuts.begin(), reg.nuts.end(), [&](const RegNut& r) { return r.id == t.id; });
        if (n == reg.nuts.end() || n->photo < 0 || n->photo >= int(reg.photos.size())) continue;
        cv::Mat& ref = refs[n->photo];
        if (ref.empty()) ref = cv::imread(reg.photos[n->photo]);
        NutResult r{"", n->side, Status::Unmeasurable, std::nullopt, ""};
        if (ref.empty()) r.reason = "image missing";
        else if (ref.size() != ses.size()) r.reason = "unclear: size differs from the registered photo";
        else r = measureNutIn(ref, cv::Rect(n->x, n->y, n->w, n->h), ses, t.box, n->side);
        out.emplace(t.id, r);
    }
    return out;
}
