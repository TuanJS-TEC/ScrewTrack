#include "nutdb.h"

#include <sqlite3.h>

#include <stdexcept>

namespace {
const char* kSchema =
    "CREATE TABLE IF NOT EXISTS nut(id INTEGER PRIMARY KEY, side INTEGER, x INTEGER, y INTEGER, w INTEGER, h INTEGER, "
    "photo INTEGER);"
    "CREATE TABLE IF NOT EXISTS photo(idx INTEGER PRIMARY KEY, path TEXT);";

void check(sqlite3* db, int rc) {
    if (rc != SQLITE_OK && rc != SQLITE_ROW && rc != SQLITE_DONE) throw std::runtime_error(sqlite3_errmsg(db));
}

struct Stmt {
    sqlite3_stmt* st = nullptr;
    Stmt(sqlite3* db, const char* sql) { check(db, sqlite3_prepare_v2(db, sql, -1, &st, nullptr)); }
    ~Stmt() { sqlite3_finalize(st); }
};
}  // namespace

NutDb::NutDb(const std::string& path) {
    if (sqlite3_open(path.c_str(), &db_) != SQLITE_OK) {
        const std::string msg = db_ ? sqlite3_errmsg(db_) : "out of memory";
        sqlite3_close(db_);
        throw std::runtime_error("cannot open database " + path + ": " + msg);
    }
    check(db_, sqlite3_exec(db_, kSchema, nullptr, nullptr, nullptr));
}

NutDb::~NutDb() { sqlite3_close(db_); }

void NutDb::save(const Registry& reg) {
    check(db_, sqlite3_exec(db_, "BEGIN", nullptr, nullptr, nullptr));
    try {
        check(db_, sqlite3_exec(db_, "DROP TABLE IF EXISTS nut; DROP TABLE IF EXISTS photo;", nullptr, nullptr, nullptr));
        check(db_, sqlite3_exec(db_, kSchema, nullptr, nullptr, nullptr));
        Stmt nut(db_, "INSERT INTO nut VALUES(?,?,?,?,?,?,?)");
        for (const RegNut& n : reg.nuts) {
            sqlite3_reset(nut.st);
            const int v[] = {n.id, n.side, n.x, n.y, n.w, n.h, n.photo};
            for (int i = 0; i < 7; ++i) sqlite3_bind_int(nut.st, i + 1, v[i]);
            check(db_, sqlite3_step(nut.st));
        }
        Stmt photo(db_, "INSERT INTO photo VALUES(?,?)");
        for (int i = 0; i < int(reg.photos.size()); ++i) {
            sqlite3_reset(photo.st);
            sqlite3_bind_int(photo.st, 1, i);
            sqlite3_bind_text(photo.st, 2, reg.photos[i].c_str(), -1, SQLITE_TRANSIENT);
            check(db_, sqlite3_step(photo.st));
        }
        check(db_, sqlite3_exec(db_, "COMMIT", nullptr, nullptr, nullptr));
    } catch (...) {
        sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
        throw;
    }
}

Registry NutDb::load() const {
    Registry reg;
    try {
        Stmt nut(db_, "SELECT id, side, x, y, w, h, photo FROM nut ORDER BY id");
        while (sqlite3_step(nut.st) == SQLITE_ROW)
            reg.nuts.push_back({sqlite3_column_int(nut.st, 0), sqlite3_column_int(nut.st, 1), sqlite3_column_int(nut.st, 2),
                                sqlite3_column_int(nut.st, 3), sqlite3_column_int(nut.st, 4), sqlite3_column_int(nut.st, 5),
                                sqlite3_column_int(nut.st, 6)});
        Stmt photo(db_, "SELECT path FROM photo ORDER BY idx");
        while (sqlite3_step(photo.st) == SQLITE_ROW)
            reg.photos.emplace_back(reinterpret_cast<const char*>(sqlite3_column_text(photo.st, 0)));
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("database is in an old format, register again (") + e.what() + ")");
    }
    return reg;
}
