#include <filesystem>
#include <opencv2/imgcodecs.hpp>

#include "check.h"
#include "test_util.h"
#include "trackrun.h"

namespace fs = std::filesystem;

static const auto kNone = [](int, const cv::Mat&, const std::vector<TrackedBox>&) { return true; };

int main() {
    const std::string root = PROJECT_DIR;
    NutDetector det(root + "/model/best.onnx");
    const fs::path dir = fs::temp_directory_path() / "trackrun_test";
    fs::remove_all(dir);
    fs::create_directories(dir / "ref");
    // Names sort IMG_0001 < IMG_0050: IMG_0001 has a top and a bottom nut, IMG_0050 one bottom nut.
    fs::copy_file(root + "/Data/img/IMG_0050.JPG", dir / "ref" / "IMG_0050.JPG");
    fs::copy_file(root + "/Data/img/IMG_0001.JPG", dir / "ref" / "IMG_0001.JPG");

    // Register: ids by file name, then top first.
    std::vector<std::vector<TrackedBox>> perImage;
    const TrackRun run = runSource(det, (dir / "ref").string(), [&](int, const cv::Mat&, const std::vector<TrackedBox>& t) {
        perImage.push_back(t);
        return true;
    });
    CHECK(perImage.size() == 2);
    CHECK(perImage[0].size() == 2 && perImage[0][0].id == 1 && perImage[0][1].id == 2);
    CHECK(perImage[1].size() == 1 && perImage[1][0].id == 3);
    CHECK(run.first.size() == 3 && run.first.at(1).side == 0 && run.first.at(2).side == 1 && run.first.at(3).side == 1);
    CHECK(run.first.at(1).photo == 0 && run.first.at(3).photo == 1 && run.photos.size() == 2);
    CHECK(fs::path(run.photos[0]).filename() == "IMG_0001.JPG");

    // One image file works the same way.
    int n = 0;
    runSource(det, (dir / "ref" / "IMG_0001.JPG").string(), [&](int, const cv::Mat&, const std::vector<TrackedBox>& t) {
        n += int(t.size());
        return true;
    });
    CHECK(n == 2);

    Registry reg{{}, run.photos};
    for (const auto& [id, nut] : run.first) reg.nuts.push_back(nut);

    // Track: the same nuts get the same ids, whatever the file is called.
    fs::create_directories(dir / "ses");
    fs::copy_file(root + "/Data/img/IMG_0001.JPG", dir / "ses" / "a.JPG");
    fs::copy_file(root + "/Data/img/IMG_0050.JPG", dir / "ses" / "b.JPG");
    perImage.clear();
    runSource(det, (dir / "ses").string(), [&](int, const cv::Mat&, const std::vector<TrackedBox>& t) {
        perImage.push_back(t);
        return true;
    }, reg);
    CHECK(perImage.size() == 2 && perImage[0].size() == 2 && perImage[0][0].id == 1 && perImage[0][1].id == 2);
    CHECK(perImage[1].size() == 1 && perImage[1][0].id == 3);

    // A registered nut scoring under 0.25 is still found near its place (IMG_0089's top nut scores ~0.15), and a sure
    // detection that is not registered gets an id after the registered ones.
    fs::create_directories(dir / "low");
    fs::copy_file(root + "/Data/img/IMG_0089.JPG", dir / "low" / "IMG_0089.JPG");
    Registry low{{{7, 0, 1847, 711, 539, 631, 0}}, {"x.jpg"}};
    perImage.clear();
    runSource(det, (dir / "low").string(), [&](int, const cv::Mat&, const std::vector<TrackedBox>& t) {
        perImage.push_back(t);
        return true;
    }, low);
    CHECK(perImage.size() == 1 && perImage[0].size() >= 1 && perImage[0][0].id == 7 && perImage[0][0].score < 0.25f);
    for (size_t i = 1; i < perImage[0].size(); ++i) CHECK(perImage[0][i].id > 7 && perImage[0][i].score >= 0.25f);

    // Rotation: the top nut of IMG_0001 turned 4 deg against its registered photo.
    cv::Mat turned = cv::imread(root + "/Data/img/IMG_0001.JPG");
    rotateNutInPlace(turned, cv::Rect(cv::Point(1815, 763), cv::Point(2395, 1310)), 4.0);
    std::vector<TrackedBox> top;
    runSource(det, (dir / "ses" / "a.JPG").string(), [&](int, const cv::Mat&, const std::vector<TrackedBox>& t) {
        top = t;
        return true;
    }, reg);
    const auto rot = measureRotations(reg, turned, top);
    CHECK(rot.count(1) && rot.at(1).rotationDeg && std::abs(*rot.at(1).rotationDeg - 4.0) <= 1.0);
    CHECK(rot.at(1).status == Status::Loose);
    CHECK(measureRotations(reg, turned, {{99, cv::Rect(), 1.0f}}).empty());  // unregistered: no entry
    cv::Mat small;
    cv::resize(turned, small, {1000, 1000});
    CHECK(measureRotations(reg, small, top).at(1).status == Status::Unmeasurable);  // size differs

    fs::path empty = dir / "empty";
    fs::create_directories(empty);
    try {
        runSource(det, empty.string(), kNone);
        CHECK(false);
    } catch (const std::runtime_error&) {
    }
    fs::remove_all(dir);
    return 0;
}
