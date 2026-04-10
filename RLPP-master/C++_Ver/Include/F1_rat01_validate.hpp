#pragma once

#include "F1_rat01_bundle.hpp"

// Throws std::runtime_error with a descriptive message if any check fails.
// Pass his from metadata if you have it; pass -1 to skip the his+1 check.
void validate_f1_bundle_or_throw(const F1Bundle& b, int his);