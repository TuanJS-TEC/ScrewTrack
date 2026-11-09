#pragma once
#include <optional>
#include <string>

enum class Status { Ok, Loose, Unmeasurable };

constexpr double kLooseDeg = 3.0;

inline const char* statusName(Status s) { return s == Status::Ok ? "ok" : s == Status::Loose ? "loose" : "unmeasurable"; }

struct NutResult {
    std::string image;
    int side;  // 0 top, 1 bottom
    Status status;
    std::optional<double> rotationDeg;
    std::string reason;
};
