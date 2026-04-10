#pragma once

#include <string>
#include <Eigen/Dense>
#include "F1_rat01_bundle.hpp"

// Phase-2 replay modes:
// - Deterministic: drive decoder history from downstream_spikes_ref (no RNG).
// - Sampled: sample spikes from generator probabilities (MATLAB/Python-style).
enum class ReplayMode {
    DeterministicFromBundleSpikes,
    SampledFromGeneratorProbs
};

struct ReplayConfig {
    ReplayMode mode = ReplayMode::DeterministicFromBundleSpikes;
    unsigned int rng_seed = 0;
    double tau_bins = 150.0; // must match what you used in Phase 1 parity

    // Logging
    std::string out_dir = "D:/RLPP-master/C++_Ver/Phase_2/logs";
    bool dump_decoder_logits = true;
    bool dump_labels = true;
    bool dump_latency_us = true;

    bool dump_generator_probs = false;
    bool dump_downstream_spikes = false;
    bool dump_decoder_features = false;

    // Latency options
    bool latency_only_valid_bins = true;

    // Window over valid columns (0-based; valid columns correspond to decoder_logits_ref columns)
    int valid_col_start = 0;   // skip first N valid columns
    int valid_col_count = -1;  // record this many valid columns; -1 = all remaining
};

struct ReplayResult {
    Eigen::MatrixXd decoder_logits; // [num_labels x T_valid]
    Eigen::VectorXi labels;         // [T_valid]
    double max_abs_err = 0.0;       // vs bundle.decoder_logits_ref
    double mean_abs_err = 0.0;
    int label_mismatches = 0;       // vs bundle.labels_ref
};

ReplayResult run_replay_or_throw(
    const F1Bundle& bundle,
    const Eigen::VectorXd& xoffset,
    const Eigen::VectorXd& gain,
    double ymin,
    const ReplayConfig& cfg
);