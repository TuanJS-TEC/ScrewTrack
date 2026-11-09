#include "bytetrack.h"

#include <algorithm>

namespace {
constexpr float kHigh = 0.5f;      // first-pass detections
constexpr float kNewTrack = 0.6f;  // a track can only start from a box this sure
constexpr float kIouHigh = 0.2f, kIouLow = 0.5f;
constexpr int kMinHits = 3;        // frames before a track gets an id
constexpr int kMaxLost = 30;       // frames a track survives without a match

cv::Mat measure(const cv::Rect& b) {
    return (cv::Mat_<float>(4, 1) << b.x + b.width / 2.0f, b.y + b.height / 2.0f, float(b.width), float(b.height));
}

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float i = (a & b).area();
    const float u = a.area() + b.area() - i;
    return u > 0 ? i / u : 0;
}
}  // namespace

ByteTracker::Track::Track(const Detection& d) : box(d.box), score(d.score) {
    cv::setIdentity(kf.transitionMatrix);
    for (int i = 0; i < 4; ++i) kf.transitionMatrix.at<float>(i, 4 + i) = 1;
    kf.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
    for (int i = 0; i < 4; ++i) kf.measurementMatrix.at<float>(i, i) = 1;
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1.0));
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(10.0));
    measure(d.box).copyTo(kf.statePost.rowRange(0, 4));
    pred = cv::Rect2f(d.box);
}

void ByteTracker::Track::predict() {
    const cv::Mat s = kf.predict();
    const float w = std::max(1.0f, s.at<float>(2)), h = std::max(1.0f, s.at<float>(3));
    pred = cv::Rect2f(s.at<float>(0) - w / 2, s.at<float>(1) - h / 2, w, h);
    // predict() restarts from statePost; keep the prediction when no box corrects it
    kf.statePre.copyTo(kf.statePost);
    kf.errorCovPre.copyTo(kf.errorCovPost);
}

void ByteTracker::Track::correct(const Detection& d) {
    kf.correct(measure(d.box));
    box = d.box;
    score = d.score;
    ++hits;
    lostFor = 0;
    seen = true;
}

// Greedy by IoU, best pairs first. shortcut: not the optimal (Hungarian) assignment, upgrade if ids swap
// between boxes that overlap. Matched entries are removed from `tracks` and `dets`.
void ByteTracker::associate(std::vector<int>& tracks, std::vector<int>& dets, const std::vector<Detection>& all,
                            float minIou) {
    struct Pair {
        float iou;
        int t, d;
    };
    std::vector<Pair> pairs;
    for (int t : tracks)
        for (int d : dets)
            if (const float v = iou(tracks_[t].pred, cv::Rect2f(all[d].box)); v >= minIou) pairs.push_back({v, t, d});
    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.iou > b.iou; });
    std::vector<char> usedT(tracks_.size()), usedD(all.size());
    for (const Pair& p : pairs) {
        if (usedT[p.t] || usedD[p.d]) continue;
        usedT[p.t] = usedD[p.d] = 1;
        tracks_[p.t].correct(all[p.d]);
    }
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [&](int t) { return usedT[t]; }), tracks.end());
    dets.erase(std::remove_if(dets.begin(), dets.end(), [&](int d) { return usedD[d]; }), dets.end());
}

std::vector<TrackedBox> ByteTracker::update(const std::vector<Detection>& dets, int frameHeight) {
    std::vector<int> high, low, ts;
    for (int i = 0; i < int(dets.size()); ++i) {
        if (dets[i].score >= kHigh) high.push_back(i);
        else if (dets[i].score >= kTrackLow) low.push_back(i);
    }
    for (int t = 0; t < int(tracks_.size()); ++t) {
        tracks_[t].predict();
        tracks_[t].seen = false;
        ts.push_back(t);
    }

    associate(ts, high, dets, kIouHigh);
    std::vector<int> stillTracked;  // the low-score pass only rescues tracks that were not already lost
    for (int t : ts)
        if (tracks_[t].lostFor == 0) stillTracked.push_back(t);
    associate(stillTracked, low, dets, kIouLow);

    for (int d : high)
        if (dets[d].score >= kNewTrack) tracks_.emplace_back(dets[d]);

    std::vector<int> confirmed;
    for (int t = 0; t < int(tracks_.size()); ++t)
        if (tracks_[t].id == 0 && tracks_[t].seen && tracks_[t].hits >= kMinHits) confirmed.push_back(t);
    std::sort(confirmed.begin(), confirmed.end(), [&](int a, int b) {
        const cv::Rect &x = tracks_[a].box, &y = tracks_[b].box;
        return std::make_pair(sideOf(x, frameHeight), x.x) < std::make_pair(sideOf(y, frameHeight), y.x);
    });
    for (int t : confirmed) tracks_[t].id = nextId_++;

    std::vector<TrackedBox> out;
    for (const Track& t : tracks_)
        if (t.id && t.seen) out.push_back({t.id, t.box, t.score});

    for (Track& t : tracks_)
        if (!t.seen) ++t.lostFor;
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [](const Track& t) { return t.lostFor > kMaxLost || (t.id == 0 && !t.seen); }),
                  tracks_.end());
    return out;
}
