#include "report.h"

#include <cmath>
#include <cstdio>

void writeCsv(std::ostream& out, const std::string& sessionName, const std::vector<NutResult>& rows) {
    out << "session,image,side,status,rotation_deg,reason\n";
    for (const NutResult& r : rows) {
        char deg[32] = "";
        if (r.rotationDeg) std::snprintf(deg, sizeof deg, "%.2f", std::fabs(*r.rotationDeg) < 0.005 ? 0.0 : *r.rotationDeg);
        out << sessionName << ',' << r.image << ',' << (r.side == 0 ? "top" : "bottom") << ',' << statusName(r.status) << ','
            << deg << ',' << r.reason << '\n';
    }
}
