#pragma once

#include <string>

namespace rlpp {

/// Full-episode RL parity vs Python `export_phase3_training_case.py` (deterministic u01_ep*.csv forward).
/// Returns 0 on success, non-zero on mismatch or I/O error (prints to std::cout/std::cerr by convention of caller).
int run_phase3_training_case_parity(const std::string& training_case_dir, double tol);

} // namespace rlpp
