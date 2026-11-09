#include <map>

#include "bytetrack.h"
#include "check.h"

// Three nuts slide right 5 px/frame; nut A (top) disappears for 5 frames, then is only weakly detected.
static Detection nut(int x, int y, float s = 0.9f) { return {cv::Rect(x, y, 100, 100), s}; }

int main() {
    ByteTracker tr;
    std::map<char, int> id;  // first id seen per nut
    auto note = [&](char name, int y, const std::vector<TrackedBox>& out, int x) {
        for (const TrackedBox& t : out)
            if (t.box.x == x && t.box.y == y) {
                if (id.count(name)) CHECK(id[name] == t.id);  // never changes
                id[name] = t.id;
            }
        return 0;
    };
    for (int f = 0; f < 40; ++f) {
        std::vector<Detection> d;
        const bool aGone = f >= 15 && f < 20;
        const float aScore = (f >= 20 && f < 23) ? 0.3f : 0.9f;  // low score: still rescued by the second pass
        if (!aGone) d.push_back(nut(10 + 5 * f, 100, aScore));
        d.push_back(nut(300 + 5 * f, 900));               // B, bottom side
        if (f >= 10) d.push_back(nut(700 + 5 * f, 100));  // C enters later
        if (f == 5) d.push_back(nut(1500, 500, 0.3f));    // a lone weak box never becomes a track
        const auto out = tr.update(d, 1000);
        note('A', 100, out, 10 + 5 * f);
        note('B', 900, out, 300 + 5 * f);
        if (f >= 10) note('C', 100, out, 700 + 5 * f);
        for (const TrackedBox& t : out) CHECK(!(t.box.x == 1500));
    }
    // A (top) and B (bottom) are confirmed in the same frame: top first. C later.
    CHECK(id['A'] == 1 && id['B'] == 2 && id['C'] == 3);
    return 0;
}
