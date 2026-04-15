#pragma once

#include <Eigen/Dense>

#include <string>

/// After Phase 3 offline training, write a directory compatible with Phase 1 `load_f1_bundle()`
/// so `RLPP_realtime_udp_infer` (and `RLPP_Phase_1` replay) can load the same generator + decoder.
///
/// Writes:
/// - generator_W1.csv, generator_W2.csv (trained weights)
/// - sortedIndices.csv (same permutation used during training)
/// - decoder_*.csv and decoder_xoffset/gain/ymin (copied from decoder_prefix decoding NN CSVs)
/// - Synthetic reference CSVs (zeros/labels) sized to satisfy `load_f1_bundle` shape checks.
/// - phase3_export_manifest.txt (human-readable notes)
///
/// Encoder geometry: Phase 3 `inputEnsemble` has `feat` rows per time bin; Phase 1 encoder expects
/// `feature_dim = Nx * H_enc` where `Nx` is upstream channel count and `H_enc` is exponential history length.
/// You must have `feat == Nx * H_enc`. Defaults: `Nx = feat`, `H_enc = 1`.
void export_phase3_inference_bundle(
    const std::string& out_dir,
    const Eigen::MatrixXd& W1,
    const Eigen::MatrixXd& W2,
    const Eigen::VectorXi& sorted_indices_1based,
    int feat_rows_input_ensemble,
    const std::string& decoder_prefix,
    int export_nx,
    int export_encoder_h,
    const std::string& phase3_model_name,
    const std::string& phase3_case_dir
);
