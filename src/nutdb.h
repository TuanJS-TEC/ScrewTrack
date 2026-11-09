#pragma once
#include <string>
#include <vector>

struct sqlite3;

// A nut registered on the first run. id = order the nut was found in.
struct RegNut {
    int id;
    int side;        // 0 top, 1 bottom
    int x, y, w, h;  // box at first sight
    int photo;       // index in Registry::photos, -1 when registered from video
};

struct Registry {
    std::vector<RegNut> nuts;          // ordered by id
    std::vector<std::string> photos;   // registered photos in order (also those without a nut), absolute paths
};

class NutDb {
public:
    explicit NutDb(const std::string& path);  // creates the file and tables if missing
    ~NutDb();
    NutDb(const NutDb&) = delete;
    NutDb& operator=(const NutDb&) = delete;

    void save(const Registry& reg);  // replaces everything registered before (also a database in an old format)
    Registry load() const;           // throws for a database in an old format: register again

private:
    sqlite3* db_ = nullptr;
};
