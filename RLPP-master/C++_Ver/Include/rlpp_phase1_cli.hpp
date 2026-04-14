#pragma once

#include <string>

enum class RlppPhase1Mode {
    All,                     // parity + online + deterministic replay + sampled sweep
    ParityOnly,              // four parity checks only
    ParityAndOnline,         // parity + RLPPInference deterministic online test
    ReplayAll,               // deterministic + sampled replay
    ReplayDeterministicOnly,
    ReplaySampledOnly,
    /// Phase 2.5: parity + online + deterministic replay + hardware I/O contract + sign-off report
    HwPrep,
};

struct RlppPhase1CliConfig {
    RlppPhase1Mode mode = RlppPhase1Mode::All;

    std::string bundle_dir = "D:/rlpp_f1_bundle_rat01";

    double tau_bins = 150.0;
    int valid_col_start = 0;
    int valid_col_count = 100000;
    int seed_start = 0;
    int seed_count = 10;

    /// Base directory for replay CSV logs (subfolders deterministic/ and sampled_seedN/)
    std::string replay_out_base = "D:/RLPP-master/C++_Ver/Phase_2/logs/Release";

    /// When true, deterministic/sampled replay also writes hardware_trace_v1.bin (concatenated v1 frames)
    bool dump_hardware_trace_v1 = false;
};

/// Parses argv. Exits process with 0 after printing help if --help or -h.
RlppPhase1CliConfig parse_rlpp_phase1_cli(int argc, char** argv);

void print_rlpp_phase1_cli_help(const char* argv0);
