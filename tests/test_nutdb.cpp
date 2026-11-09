#include <cstdio>
#include <sqlite3.h>

#include <filesystem>

#include "check.h"
#include "nutdb.h"

int main() {
    const std::string path = (std::filesystem::temp_directory_path() / "nutdb_test.sqlite").string();
    std::remove(path.c_str());
    {
        NutDb db(path);
        CHECK(db.load().nuts.empty());
        db.save({{{1, 0, 10, 20, 90, 80, 0}, {2, 1, 11, 21, 100, 95, 1}}, {"/a/x.jpg", "/a/empty.jpg", "/a/y.jpg"}});
    }
    NutDb db(path);  // reopened: the registration survived
    Registry r = db.load();
    CHECK(r.nuts.size() == 2 && r.nuts[1].id == 2 && r.nuts[1].side == 1 && r.nuts[1].x == 11 && r.nuts[1].y == 21);
    CHECK(r.nuts[1].w == 100 && r.nuts[1].h == 95 && r.nuts[1].photo == 1);
    CHECK(r.photos.size() == 3 && r.photos[1] == "/a/empty.jpg");  // a photo without a nut keeps its place
    db.save({{{1, 0, 1, 1, 1, 1, -1}}, {}});  // saving again replaces
    CHECK(db.load().nuts.size() == 1 && db.load().photos.empty());
    std::remove(path.c_str());

    // A database from before positions were stored: load says so, and registering upgrades it.
    {
        sqlite3* raw = nullptr;
        CHECK(sqlite3_open(path.c_str(), &raw) == SQLITE_OK);
        CHECK(sqlite3_exec(raw, "CREATE TABLE nut(id INTEGER PRIMARY KEY, side INTEGER, w INTEGER, h INTEGER); INSERT INTO nut VALUES(1,0,9,9)", nullptr, nullptr, nullptr) == SQLITE_OK);
        sqlite3_close(raw);
    }
    {
        NutDb old(path);
        try {
            old.load();
            CHECK(false);
        } catch (const std::runtime_error& e) {
            CHECK(std::string(e.what()).find("old format") != std::string::npos);
        }
        old.save({{{1, 0, 1, 2, 3, 4, 0}}, {"p.jpg"}});
        CHECK(old.load().nuts.size() == 1);
    }
    std::remove(path.c_str());
    return 0;
}
