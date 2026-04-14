#pragma once

#include <Eigen/Dense>
#include <random>
#include "exponential_history_encoder.hpp"
#include "two_layer_mlp.hpp"
#include "decoder_history_buffer.hpp"
#include "decoding_model01.hpp"

enum class SpikeDriveMode {
    SampledBernoulli,   // faithful to MATLAB applynets: rand <= p
    ExpectedProb        // debug mode: use p as “spikes” (not faithful, but deterministic)
};

struct RLPPInferenceConfig {
    double tau_bins = 150.0;                // must match export / parity
    SpikeDriveMode spike_mode = SpikeDriveMode::SampledBernoulli;
    unsigned int rng_seed = 0;
};

struct RLPPInferenceStepOutput {
    bool valid = false;               // false until encoder.can_encode(t)

    Eigen::VectorXd encoder_x;        // [feature_dim], valid only if valid==true
    Eigen::VectorXd gen_prob;         // [Ny]
    Eigen::VectorXd gen_spikes;       // [Ny] (0/1 if sampled, else probs if ExpectedProb)

    Eigen::VectorXd decoder_ensemble; // [Ny*(his+1)] flattened (lag0 first)
    Eigen::VectorXd decoder_y;        // [num_labels] (softmax output)
    int label_1based = 0;             // 1..K when valid, else 0
};

class RLPPInference {
public:
    // sorted_indices_1based: MATLAB-style indices that reorder M1 channels before decoder history
    RLPPInference(
        int Nx,
        int H,
        int Ny,
        int num_lags,
        const Eigen::VectorXi& sorted_indices_1based,
        const Eigen::MatrixXd& gen_W1,
        const Eigen::MatrixXd& gen_W2,
        const Eigen::VectorXd& dec_xoffset,
        const Eigen::VectorXd& dec_gain,
        double dec_ymin,
        const Eigen::MatrixXd& dec_W1,
        const Eigen::VectorXd& dec_b1,
        const Eigen::MatrixXd& dec_W2,
        const Eigen::VectorXd& dec_b2,
        const RLPPInferenceConfig& cfg
    );

    // One time bin. t is 1-based (same convention your encoder uses).
    RLPPInferenceStepOutput step(const Eigen::VectorXd& upstream_bin, int t);

    // Test/debug helper: drive decoder history with known downstream spikes
    // (unsorted M1 order, matching downstream_spikes_ref columns in your bundle).
    // This lets you validate the online state machine without RNG differences.
    RLPPInferenceStepOutput step_with_downstream_spikes(
        const Eigen::VectorXd& upstream_bin,
        const Eigen::VectorXd& downstream_spikes_unsorted,
        int t
    );

    // Same as above, but reads column `col` only after upstream is observed and
    // encoder.can_encode(t) is true (safe for replay when col is not yet aligned).
    RLPPInferenceStepOutput step_with_downstream_spikes(
        const Eigen::VectorXd& upstream_bin,
        const Eigen::MatrixXd& downstream_spikes_ref,
        int col,
        int t
    );

    int start_time_bin() const { return encoder_.start_time_bin(); }

private:
    Eigen::VectorXd reorder_m1_if_needed(const Eigen::VectorXd& v) const;

    RLPPInferenceStepOutput complete_step_with_provided_downstream(
        const Eigen::VectorXd& downstream_spikes_unsorted,
        int t
    );

    RLPPInferenceConfig cfg_;

    int Nx_;
    int H_;
    int Ny_;
    int num_lags_;

    Eigen::VectorXi sorted_indices_1based_;

    ExponentialHistoryEncoder encoder_;
    TwoLayerMLP generator_;
    DecoderHistoryBuffer dec_hist_;

    // decoder mapminmax + weights
    Eigen::VectorXd dec_xoffset_;
    Eigen::VectorXd dec_gain_;
    double dec_ymin_;
    Eigen::MatrixXd dec_W1_;
    Eigen::VectorXd dec_b1_;
    Eigen::MatrixXd dec_W2_;
    Eigen::VectorXd dec_b2_;

    std::mt19937 rng_;
};