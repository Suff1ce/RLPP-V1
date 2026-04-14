#pragma once

#include <Eigen/Dense>
#include <random>

namespace rlpp {

/// Same as applynets_priori.py (RLPP.py training forward): prior-scaled p, then sample spikes.
struct ApplyNetsPrioriBatch {
    Eigen::MatrixXd p_output;     ///< after priori
    Eigen::MatrixXd hidden_unit;  ///< [(hidden+1) x n]
    Eigen::MatrixXd spk_out;      ///< Bernoulli vs p_output
};

/// input_unit: [(feature_dim+1) x n], last row all 1 (bias). W1: [hidden x (feature+1)], W2: [Ny x (hidden+1)]
ApplyNetsPrioriBatch applynets_priori_forward(
    const Eigen::MatrixXd& input_unit,
    const Eigen::MatrixXd& W1,
    const Eigen::MatrixXd& W2,
    int episode,
    double priori_m,
    double priori_n,
    std::mt19937& rng
);

/// Deterministic variant for exact parity: uses a pre-generated U(0,1) matrix `uniform_01` of shape [Ny x n].
/// Spike rule matches Python: choose_action = (uniform_01 <= p_output).
ApplyNetsPrioriBatch applynets_priori_forward_with_uniforms(
    const Eigen::MatrixXd& input_unit,
    const Eigen::MatrixXd& W1,
    const Eigen::MatrixXd& W2,
    int episode,
    double priori_m,
    double priori_n,
    const Eigen::MatrixXd& uniform_01
);

/// getgradient.py (RL): delta = reward * (spk - p)
void getgradient_rl(
    const Eigen::MatrixXd& reward_per_sample,  ///< [Ny x n] (broadcast row-wise from smoothed 1xn in Python)
    const Eigen::MatrixXd& p_output,
    const Eigen::MatrixXd& spk_out_predict,
    const Eigen::MatrixXd& hidden_unit,
    const Eigen::MatrixXd& input_unit,
    const Eigen::MatrixXd& weight_hidden_output,
    int num_samples,
    Eigen::MatrixXd& weight_delta1,
    Eigen::MatrixXd& weight_delta2
);

/// Python broadcasts smoothed reward (length n) across Ny rows: delta = reward * (spk - p)
void getgradient_rl_broadcast(
    const Eigen::RowVectorXd& smoothed_reward_n,  ///< length n
    const Eigen::MatrixXd& p_output,
    const Eigen::MatrixXd& spk_out_predict,
    const Eigen::MatrixXd& hidden_unit,
    const Eigen::MatrixXd& input_unit,
    const Eigen::MatrixXd& weight_hidden_output,
    int num_samples,
    Eigen::MatrixXd& weight_delta1,
    Eigen::MatrixXd& weight_delta2
);

double learning_rate_rl_simulations(int episode, int max_episode);
double learning_rate_rl_real(int episode, int max_episode);

/// RLPP.py reward pipeline: success (NaN where invalid), motorPerform in {1,2,3}, per-sample inner reward, conv, z-score.
Eigen::RowVectorXd compute_smoothed_reward_rlpp(
    const Eigen::RowVectorXd& success,       ///< may contain NaN
    const Eigen::VectorXi& motor_perform,    ///< length n, values 1..3
    double epsilon,
    int episode,
    int max_episode,
    double discount_factor,
    int discount_length
);

} // namespace rlpp
