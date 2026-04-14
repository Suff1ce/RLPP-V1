#include "rlpp_phase1_cli.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

static std::string next_arg(int& i, int argc, char** argv, const char* flag_name) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value after ") + flag_name);
    }
    return std::string(argv[++i]);
}

static RlppPhase1Mode parse_mode(const std::string& s) {
    if (s == "all") return RlppPhase1Mode::All;
    if (s == "parity") return RlppPhase1Mode::ParityOnly;
    if (s == "parity-online") return RlppPhase1Mode::ParityAndOnline;
    if (s == "replay") return RlppPhase1Mode::ReplayAll;
    if (s == "replay-det") return RlppPhase1Mode::ReplayDeterministicOnly;
    if (s == "replay-sampled") return RlppPhase1Mode::ReplaySampledOnly;
    if (s == "hw-prep") return RlppPhase1Mode::HwPrep;
    throw std::runtime_error(
        "Unknown --mode value: \"" + s + "\". Use --help for list."
    );
}

void print_rlpp_phase1_cli_help(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --help, -h              Show this help and exit\n"
        << "  --bundle PATH           F1 CSV bundle directory (decoder_xoffset/gain/ymin + bundle CSVs)\n"
        << "  --mode NAME             Run subset (default: all)\n"
        << "      all                 Parity + online inference test + deterministic replay + sampled sweep\n"
        << "      parity              Encoder/generator/decoder parity checks only\n"
        << "      parity-online       parity + RLPPInference deterministic online test\n"
        << "      replay              deterministic + sampled replay\n"
        << "      replay-det          deterministic replay only\n"
        << "      replay-sampled      sampled replay seed sweep only\n"
        << "      hw-prep             Phase 2.5: parity + online + det replay + hardware I/O contract + report\n"
        << "  --tau-bins N            Exponential encoder tau (default 150)\n"
        << "  --valid-col-start N     Replay window start (0-based valid column index)\n"
        << "  --valid-col-count N     Replay window length (-1 = all remaining)\n"
        << "  --seed-start N          First RNG seed for sampled replay\n"
        << "  --seed-count N          Number of seeds for sampled replay\n"
        << "  --replay-out-base PATH  Base directory for replay logs (subdir deterministic/ sampled_seedK/)\n"
        << "  --dump-hardware-trace   Also write hardware_trace_v1.bin during replay (concatenated v1 frames)\n"
        << "\n"
        << "With no arguments, defaults match the previous hardcoded Phase_1 behavior.\n";
}

RlppPhase1CliConfig parse_rlpp_phase1_cli(int argc, char** argv) {
    RlppPhase1CliConfig cfg;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--help") == 0 || std::strcmp(a, "-h") == 0) {
            print_rlpp_phase1_cli_help(argv[0]);
            std::exit(0);
        }
        if (std::strcmp(a, "--bundle") == 0) {
            cfg.bundle_dir = next_arg(i, argc, argv, "--bundle");
            continue;
        }
        if (std::strcmp(a, "--mode") == 0) {
            cfg.mode = parse_mode(next_arg(i, argc, argv, "--mode"));
            continue;
        }
        if (std::strcmp(a, "--tau-bins") == 0) {
            cfg.tau_bins = std::stod(next_arg(i, argc, argv, "--tau-bins"));
            continue;
        }
        if (std::strcmp(a, "--valid-col-start") == 0) {
            cfg.valid_col_start = std::stoi(next_arg(i, argc, argv, "--valid-col-start"));
            continue;
        }
        if (std::strcmp(a, "--valid-col-count") == 0) {
            cfg.valid_col_count = std::stoi(next_arg(i, argc, argv, "--valid-col-count"));
            continue;
        }
        if (std::strcmp(a, "--seed-start") == 0) {
            cfg.seed_start = std::stoi(next_arg(i, argc, argv, "--seed-start"));
            continue;
        }
        if (std::strcmp(a, "--seed-count") == 0) {
            cfg.seed_count = std::stoi(next_arg(i, argc, argv, "--seed-count"));
            continue;
        }
        if (std::strcmp(a, "--replay-out-base") == 0) {
            cfg.replay_out_base = next_arg(i, argc, argv, "--replay-out-base");
            continue;
        }
        if (std::strcmp(a, "--dump-hardware-trace") == 0) {
            cfg.dump_hardware_trace_v1 = true;
            continue;
        }
        throw std::runtime_error(std::string("Unknown argument: ") + a + " (try --help)");
    }

    return cfg;
}
