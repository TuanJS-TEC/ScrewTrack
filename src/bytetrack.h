#pragma once
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

#include "detector.h"

constexpr float kTrackLow = 0.1f;  // lowest detection score ByteTracker uses; feed it detections down to here

struct TrackedBox {
    int id;
    cv::Rect box;
    float score;
};

// ByteTrack (Zhang et al. 2022): high-score boxes first, then low-score boxes rescue tracks that would be lost.
// Ids are 1, 2, 3... in the order tracks are confirmed (seen in kMinHits frames), so the same route gives the
// same ids. Boxes confirmed in one frame are numbered top side first, then left to right.
class ByteTracker {
public:
    // Confirmed tracks that were matched to a detection in this frame.
    std::vector<TrackedBox> update(const std::vector<Detection>& dets, int frameHeight);

private:
    struct Track {
        cv::KalmanFilter kf{8, 4, 0, CV_32F};  // state cx cy w h + velocities; measurement cx cy w h
        int id = 0;                            // 0 until confirmed
        int hits = 1, lostFor = 0;
        bool seen = true;
        cv::Rect box;
        float score = 0;
        cv::Rect2f pred;
        explicit Track(const Detection& d);
        void predict();
        void correct(const Detection& d);
    };
    void associate(std::vector<int>& tracks, std::vector<int>& dets, const std::vector<Detection>& all, float minIou);

    std::vector<Track> tracks_;
    int nextId_ = 1;
};
