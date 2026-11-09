#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>

#include "check.h"
#include "pipeline.h"
#include "report.h"
#include "test_util.h"

namespace fs = std::filesystem;
static const std::string kImg = std::string(PROJECT_DIR) + "/Data/img/";

static void print(const char* what, const std::vector<NutResult>& rows) {
    for (const auto& r : rows)
        std::cout << what << " side " << r.side << ": " << r.reason << " "
                  << (r.rotationDeg ? *r.rotationDeg : 0.0) << "\n";
}

int main() {
    // CSV text.
    std::ostringstream csv;
    writeCsv(csv, "s1",
             {{"IMG_0001.JPG", 0, Status::Ok, 0.5, ""},
              {"IMG_0001.JPG", 1, Status::Unmeasurable, std::nullopt, "nut not found"},
              {"IMG_0002.JPG", 0, Status::Loose, -4.25, ""}});
    CHECK(csv.str() ==
          "session,image,side,status,rotation_deg,reason\n"
          "s1,IMG_0001.JPG,top,ok,0.50,\n"
          "s1,IMG_0001.JPG,bottom,unmeasurable,,nut not found\n"
          "s1,IMG_0002.JPG,top,loose,-4.25,\n");

    // Status follows the printed (0.01 deg) value: 2.996 prints 3.00 and is loose, 2.994 prints 2.99 and is ok.
    {
        NutResult a{"a.JPG", 0, Status::Unmeasurable, std::nullopt, ""}, b = a, c = a;
        setRotation(a, 2.996);
        setRotation(b, 2.994);
        setRotation(c, -2.996);
        CHECK(a.status == Status::Loose && b.status == Status::Ok && c.status == Status::Loose);
        std::ostringstream out;
        writeCsv(out, "s", {a, b});
        CHECK(out.str() == "session,image,side,status,rotation_deg,reason\ns,a.JPG,top,loose,3.00,\ns,a.JPG,top,ok,2.99,\n");
    }

    const NutDetector det(std::string(PROJECT_DIR) + "/model/best.onnx");
    const fs::path tmp = fs::temp_directory_path() / "screwtrack_pipeline_test";
    fs::remove_all(tmp);
    fs::create_directories(tmp / "ref");
    fs::create_directories(tmp / "s_deg");
    fs::create_directories(tmp / "s_empty");

    // Rotated nut: IMG_0001 top nut, labelled box.
    const cv::Rect box(cv::Point(1815, 763), cv::Point(2395, 1310));
    fs::copy_file(kImg + "IMG_0001.JPG", tmp / "ref" / "IMG_0001.JPG");
    cv::Mat img = cv::imread(kImg + "IMG_0001.JPG");
    CHECK(!img.empty());
    for (double deg : {4.0, 1.0}) {
        cv::Mat rot = img.clone();
        rotateNutInPlace(rot, box, deg);
        cv::imwrite((tmp / "s_deg" / "IMG_0001.JPG").string(), rot);
        const auto rows = measureSession(det, tmp / "ref", tmp / "s_deg");
        print(deg == 4.0 ? "rot 4" : "rot 1", rows);
        CHECK(rows.size() == 2);
        CHECK(rows[0].side == 0 && rows[1].side == 1);
        CHECK(rows[0].rotationDeg);
        if (deg == 4.0) {
            CHECK(rows[0].status == Status::Loose);
            CHECK(std::abs(*rows[0].rotationDeg - 4.0) <= 1.0);
        } else {
            CHECK(rows[0].status == Status::Ok);
            CHECK(std::abs(*rows[0].rotationDeg - 1.0) <= 1.0);
        }
    }

    // Nut not turned, but a dark strip covers the left 15% of its box in the session: read -6.5 deg (loose) before
    // the covering check, must be unmeasurable now.
    {
        cv::Mat covered = img.clone();
        cv::rectangle(covered, cv::Rect(box.x, box.y, int(0.15 * box.width), box.height), cv::Scalar(60, 60, 60), cv::FILLED);
        cv::imwrite((tmp / "s_deg" / "IMG_0001.JPG").string(), covered);
        const auto rows = measureSession(det, tmp / "ref", tmp / "s_deg");
        print("covered", rows);
        CHECK(rows.size() == 2);
        CHECK(rows[0].status == Status::Unmeasurable && rows[0].reason == "unclear: covered" && !rows[0].rotationDeg);
    }

    // A big turn (the covering check used to reject it as "covered": the still background moves the arcs more the
    // further the nut turned). Only loose-or-not is exact here, the value a few deg. Positive only: the synthetic -15
    // patch fails "no clear hex" (a seam artifact: real photos turned either way measure, see the 15 deg pairs).
    for (double deg : {15.0}) {
        cv::Mat rot = img.clone();
        rotateNutInPlace(rot, box, deg, 1.6);
        cv::imwrite((tmp / "s_deg" / "IMG_0001.JPG").string(), rot);
        const auto rows = measureSession(det, tmp / "ref", tmp / "s_deg");
        print("big turn", rows);
        CHECK(rows.size() == 2);
        CHECK(rows[0].rotationDeg && rows[0].status == Status::Loose);
        CHECK(std::abs(*rows[0].rotationDeg - deg) <= 3.0);
    }

    // Rotated 3 deg and the whole photo shifted 20 px (placement drift between sessions).
    {
        cv::Mat rot = img.clone(), moved;
        rotateNutInPlace(rot, box, 3.0);
        cv::warpAffine(rot, moved, cv::Mat((cv::Mat_<double>(2, 3) << 1, 0, 16, 0, 1, 12)), rot.size(), cv::INTER_NEAREST,
                       cv::BORDER_REPLICATE);
        cv::imwrite((tmp / "s_deg" / "IMG_0001.JPG").string(), moved);
        const auto rows = measureSession(det, tmp / "ref", tmp / "s_deg");
        print("rot 3 + shift", rows);
        CHECK(rows.size() == 2);
        CHECK(rows[0].rotationDeg);
        CHECK(std::abs(*rows[0].rotationDeg - 3.0) <= 1.0);
    }

    // Whole photo turned about its centre (device placement), nut not turned / turned 3 deg. Without the rail
    // de-rotation in the shift search, the 3 deg photo came out "misaligned"; with the box centre instead of the ring
    // centre as pivot, the unturned bottom nut of the 3 deg photo came out "covered".
    for (const auto [photo, deg] : {std::pair{1.5, 0.0}, {1.5, 3.0}, {3.0, 0.0}}) {
        cv::Mat rot = img.clone(), turned;
        rotateNutInPlace(rot, box, deg, 1.3);
        cv::warpAffine(rot, turned, cv::getRotationMatrix2D(cv::Point2f(img.cols / 2.0f, img.rows / 2.0f), -photo, 1.0), img.size(),
                       cv::INTER_CUBIC, cv::BORDER_REFLECT);
        cv::imwrite((tmp / "s_deg" / "IMG_0001.JPG").string(), turned);
        const auto rows = measureSession(det, tmp / "ref", tmp / "s_deg");
        print(("photo " + std::to_string(photo) + " + rot " + std::to_string(deg)).c_str(), rows);
        CHECK(rows.size() == 2);
        CHECK(rows[0].rotationDeg);
        CHECK(std::abs(*rows[0].rotationDeg - deg) < 1.0);
        CHECK(rows[1].rotationDeg && std::abs(*rows[1].rotationDeg) < 1.0);  // bottom nut never turned
    }

    // A different photo under the same name: the placement shift cannot be found, so no angle.
    {
        fs::create_directories(tmp / "s_other");
        fs::copy_file(kImg + "IMG_0002.JPG", tmp / "s_other" / "IMG_0001.JPG");
        const auto rows = measureSession(det, tmp / "ref", tmp / "s_other");
        print("other photo", rows);
        CHECK(rows.size() == 2);
        for (const auto& r : rows) CHECK(r.status == Status::Unmeasurable && r.reason == "unclear: misaligned" && !r.rotationDeg);
    }

    // Missing and zero-byte session files.
    auto rows = measureSession(det, tmp / "ref", tmp / "s_empty");
    CHECK(rows.size() == 2);
    for (const auto& r : rows) CHECK(r.status == Status::Unmeasurable && r.reason == "image missing" && !r.rotationDeg);
    std::ofstream(tmp / "s_empty" / "IMG_0001.JPG").close();
    rows = measureSession(det, tmp / "ref", tmp / "s_empty");
    for (const auto& r : rows) CHECK(r.status == Status::Unmeasurable && r.reason == "image missing");

    // No nuts in IMG_0090: both rows not found.
    fs::create_directories(tmp / "ref90");
    fs::copy_file(kImg + "IMG_0090.JPG", tmp / "ref90" / "IMG_0090.JPG");
    rows = measureSession(det, tmp / "ref90", tmp / "ref90");
    CHECK(rows.size() == 2);
    for (const auto& r : rows) CHECK(r.status == Status::Unmeasurable && r.reason == "nut not found");

    // IMG_0050: top nut absent, bottom nut present.
    fs::create_directories(tmp / "ref50");
    fs::copy_file(kImg + "IMG_0050.JPG", tmp / "ref50" / "IMG_0050.JPG");
    rows = measureSession(det, tmp / "ref50", tmp / "ref50");
    print("img50", rows);
    CHECK(rows.size() == 2);
    CHECK(rows[0].status == Status::Unmeasurable && rows[0].reason == "nut not found");
    CHECK(rows[1].reason != "nut not found");

    fs::remove_all(tmp);
    return 0;
}
