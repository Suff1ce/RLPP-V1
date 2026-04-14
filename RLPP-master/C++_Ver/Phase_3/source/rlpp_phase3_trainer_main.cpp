// Phase 3: minimal end-to-end RLPP episode loop in C++ (parity vs Python-exported training_case/).

#include "rlpp_phase3_training_case_parity.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    std::string case_dir;
    double tol = 1e-9;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            std::cout << "Usage: RLPP_Phase_3_Trainer --case DIR [--tol X]\n"
                         "  Run C++ RLPP episode loop vs Python-exported training_case.\n";
            return 0;
        }
        if (a == "--case" && i + 1 < argc) {
            case_dir = argv[++i];
            continue;
        }
        if (a == "--tol" && i + 1 < argc) {
            tol = std::stod(argv[++i]);
            continue;
        }
        std::cerr << "Unknown arg: " << a << "\n";
        return 1;
    }

    if (case_dir.empty()) {
        fs::path guess = fs::path("d:/RLPP-master/C++_Ver/Phase_3/training_case");
        if (fs::is_directory(guess)) {
            case_dir = guess.string();
        }
    }
    if (case_dir.empty() || !fs::is_directory(case_dir)) {
        std::cerr << "Missing case dir. Run: python Python_Ver/export_phase3_training_case.py\n";
        return 1;
    }

    try {
        return rlpp::run_phase3_training_case_parity(case_dir, tol);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
