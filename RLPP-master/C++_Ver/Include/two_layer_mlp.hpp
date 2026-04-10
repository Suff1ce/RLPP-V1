#pragma once

#include <Eigen/Dense>

class TwoLayerMLP {
public:
    // Faithful Python-style architecture:
    //
    //   input_with_bias  = [x; 1]
    //   hidden           = sigmoid(W1 * input_with_bias)
    //   hidden_with_bias = [hidden; 1]
    //   logits           = W2 * hidden_with_bias
    //   output           = sigmoid(logits)
    //
    // Shapes:
    //   W1: [hidden_dim x (feature_dim + 1)]
    //   W2: [output_dim x (hidden_dim + 1)]
    //
    // The caller supplies raw features x of length feature_dim.
    // This class appends both bias units internally.
    TwoLayerMLP(const Eigen::MatrixXd& W1,
                const Eigen::MatrixXd& W2);

    int feature_dim() const;  // raw feature dimension, excluding appended bias
    int input_dim_with_bias() const;
    int hidden_dim() const;
    int output_dim() const;

    Eigen::VectorXd forward(const Eigen::VectorXd& x) const;
    Eigen::VectorXd forward_logits(const Eigen::VectorXd& x) const;

    // X shape: [feature_dim x num_samples]
    Eigen::MatrixXd forward_batch(const Eigen::MatrixXd& X) const;
    Eigen::MatrixXd forward_logits_batch(const Eigen::MatrixXd& X) const;

    const Eigen::MatrixXd& weight_input_hidden() const;
    const Eigen::MatrixXd& weight_hidden_output() const;

private:
    Eigen::MatrixXd W1_; // [hidden_dim x (feature_dim + 1)]
    Eigen::MatrixXd W2_; // [output_dim x (hidden_dim + 1)]

};