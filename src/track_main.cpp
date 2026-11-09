#include <cmath>
#include <cstdio>
#include <iostream>

#include "trackrun.h"

int main(int argc, char** argv) {
    const std::string mode = argc == 5 ? argv[3] : "";
    if (mode != "register" && mode != "track") {
        std::cerr << "usage: nuttrack <model.onnx> <db> register|track <video|image|image_dir>\n"
                     "  register: give ids to the nuts and store them in <db>\n"
                     "  track:    give the nuts the ids stored in <db>; for photos also measure how far each nut turned\n";
        return 2;
    }
    try {
        const NutDetector det(argv[1]);
        NutDb db(argv[2]);
        const Registry reg = mode == "track" ? db.load() : Registry();
        std::set<int> registered;
        for (const RegNut& n : reg.nuts) registered.insert(n.id);

        std::cout << "frame,id,x,y,w,h,score,registered,status,rotation_deg,reason\n";
        const TrackRun run = runSource(det, argv[4], [&](int f, const cv::Mat& img, const std::vector<TrackedBox>& tracks) {
            const auto rot = measureRotations(reg, img, tracks);
            for (const TrackedBox& t : tracks) {
                std::cout << f << ',' << t.id << ',' << t.box.x << ',' << t.box.y << ',' << t.box.width << ',' << t.box.height
                          << ',' << t.score << ',' << registered.count(t.id);
                if (const auto r = rot.find(t.id); r != rot.end()) {
                    char deg[32] = "";
                    if (r->second.rotationDeg)
                        std::snprintf(deg, sizeof deg, "%.2f", std::fabs(*r->second.rotationDeg) < 0.005 ? 0.0 : *r->second.rotationDeg);
                    std::cout << ',' << statusName(r->second.status) << ',' << deg << ',' << r->second.reason;
                } else {
                    std::cout << ",,,";
                }
                std::cout << '\n';
            }
            std::cout << std::flush;  // realtime: one frame's rows are out before the next frame is read
            return true;
        }, reg);
        if (mode == "register") {
            Registry out{{}, run.photos};
            for (const auto& [id, n] : run.first) out.nuts.push_back(n);
            db.save(out);
            std::cerr << "registered " << out.nuts.size() << " nuts\n";
        } else {
            for (int id : registered)
                if (!run.seen.count(id)) std::cerr << "not seen: id " << id << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
