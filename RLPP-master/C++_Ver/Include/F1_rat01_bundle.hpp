#pragma once

#include <string>
#include <Eigen/Dense>

struct F1Bundle {
    // raw input
    Eigen::MatrixXd upstream_spikes;      // [T x Nx]

    // generator weights (bias is last column in W1/W2)
    Eigen::MatrixXd generator_W1;         // [hidden x (feature_dim+1)]
    Eigen::MatrixXd generator_W2;         // [Ny x (hidden+1)]

    // optional decoder weights (may be empty if not NN decoder)
    Eigen::MatrixXd decoder_W1;
    Eigen::VectorXd decoder_b1;
    Eigen::MatrixXd decoder_W2;
    Eigen::VectorXd decoder_b2;

    // reference intermediates
    Eigen::MatrixXd encoder_features_ref;     // [feature_dim x T_valid]
    Eigen::MatrixXd generator_probs_ref;      // [Ny x T_valid]
    Eigen::MatrixXd downstream_spikes_ref;    // [Ny x T_valid]
    Eigen::MatrixXd decoder_features_ref;     // [(his+1)*Ny x T_valid]
    Eigen::MatrixXd decoder_logits_ref;       // [num_labels x T_valid]
    Eigen::VectorXi labels_ref;               // [T_valid]
    Eigen::VectorXi sorted_indices_1based;    // [Ny]
};

F1Bundle load_f1_bundle(const std::string& dir);