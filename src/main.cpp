#include <filesystem>
#include <iostream>
#include <sstream>

#include "pipeline.h"
#include "report.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: screwtrack <model.onnx> <ref_dir> <session_dir>...\n";
        return 2;
    }
    for (int i = 2; i < argc; ++i)
        if (!std::filesystem::is_directory(argv[i])) {
            std::cerr << "error: not a directory: " << argv[i] << "\n";
            return 2;
        }
    try {
        const NutDetector det(argv[1]);
        for (int i = 3; i < argc; ++i) {
            const std::filesystem::path dir(argv[i]);
            const auto rows = measureSession(det, argv[2], dir);
            if (rows.empty()) {  // rows come from the reference's images, so it has none
                std::cerr << "error: no *.jpg images in the reference directory: " << argv[2] << "\n";
                return 2;
            }
            std::ostringstream csv;
            writeCsv(csv, dir.filename().empty() ? dir.parent_path().filename().string() : dir.filename().string(), rows);
            const std::string text = csv.str();
            // one header for the whole output: skip it after the first session
            std::cout << (i == 3 ? text : text.substr(text.find('\n') + 1));
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
