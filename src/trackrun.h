#pragma once
#include <functional>
#include <map>
#include <opencv2/core.hpp>
#include <set>
#include <string>
#include <vector>

#include "bytetrack.h"
#include "nutdb.h"
#include "types.h"

struct TrackRun {
    std::map<int, RegNut> first;       // where each id was first seen
    std::set<int> seen;
    std::vector<std::string> photos;   // images: the photos processed, in order (absolute paths)
};

// Called after every frame (or image); return false to stop.
using FrameFn = std::function<bool(int frame, const cv::Mat& img, const std::vector<TrackedBox>& tracks)>;

// `path` is a video (file or stream url), an image, or a folder of images (*.jpg, *.jpeg, *.png).
// Video: ByteTrack ids, in the order nuts are confirmed. Photos are separate places, not frames of one scene, so
// nothing is tracked: images are taken in file name order and nuts get ids 1, 2, 3... top side first, then left to right.
// `registered` (from an earlier register run, empty when registering): for photos, the k-th photo is matched with the
// k-th registered photo, and each nut registered there is looked for near where it was, down to a low score, so a nut
// whose score wobbles around the usual 0.25 keeps its id. Other nuts (score >= 0.25) get new ids after the registered ones.
// Throws when the path cannot be opened or a folder has no images.
TrackRun runSource(const NutDetector& det, const std::string& path, const FrameFn& onFrame,
                   const Registry& registered = Registry());

// For photos and a registry made from photos: rotation of each tracked, registered nut against its registered photo.
// Empty for video, and no entry for nuts that are not registered.
std::map<int, NutResult> measureRotations(const Registry& reg, const cv::Mat& ses, const std::vector<TrackedBox>& tracks);
