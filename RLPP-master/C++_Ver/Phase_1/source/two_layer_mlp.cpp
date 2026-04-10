#include "two_layer_mlp.hpp"
#include "math_functions.hpp"

#include <cmath>
#include <stdexcept>

TwoLayerMLP::TwoLayerMLP(const Eigen::MatrixXd& W1,
                         const Eigen::MatrixXd& W2)
    : W1_(W1), W2_(W2) {
    if (W1_.rows() <= 0 || W1_.cols() <= 1) {
        throw std::runtime_error(
            "TwoLayerMLP: W1 must be non-empty and include an input bias column");
    }

    if (W2_.rows() <= 0 || W2_.cols() <= 1) {
        throw std::runtime_error(
            "TwoLayerMLP: W2 must be non-empty and include a hidden bias column");
    }

    const int hidden = static_cast<int>(W1_.rows());

    if (W2_.cols() != hidden + 1) {
        throw std::runtime_error(
            "TwoLayerMLP: W2 must have hidden_dim + 1 columns "
            "(extra column for appended hidden bias unit)");
    }
}

int TwoLayerMLP::feature_dim() const {
    return static_cast<int>(W1_.cols()) - 1;
}

int TwoLayerMLP::input_dim_with_bias() const {
    return static_cast<int>(W1_.cols());
}

int TwoLayerMLP::hidden_dim() const {
    return static_cast<int>(W1_.rows());
}

int TwoLayerMLP::output_dim() const {
    return static_cast<int>(W2_.rows());
}

Eigen::VectorXd TwoLayerMLP::forward_logits(const Eigen::VectorXd& x) const {
    if (x.size() != feature_dim()) {
        throw std::runtime_error(
            "TwoLayerMLP::forward_logits: raw feature size mismatch");
    }

    // input_with_bias = [x; 1]
    Eigen::VectorXd input_with_bias(feature_dim() + 1);
    input_with_bias.head(feature_dim()) = x;
    input_with_bias(feature_dim()) = 1.0;

    // hidden = sigmoid(W1 * input_with_bias)
    Eigen::VectorXd hidden = sigmoid_vector(W1_ * input_with_bias);

    // hidden_with_bias = [hidden; 1]
    Eigen::VectorXd hidden_with_bias(hidden_dim() + 1);
    hidden_with_bias.head(hidden_dim()) = hidden;
    hidden_with_bias(hidden_dim()) = 1.0;

    // logits = W2 * hidden_with_bias
    return W2_ * hidden_with_bias;
}

Eigen::VectorXd TwoLayerMLP::forward(const Eigen::VectorXd& x) const {
    return sigmoid_vector(forward_logits(x));
}

Eigen::MatrixXd TwoLayerMLP::forward_logits_batch(const Eigen::MatrixXd& X) const {
    if (X.rows() != feature_dim()) {
        throw std::runtime_error(
            "TwoLayerMLP::forward_logits_batch: raw feature row count mismatch");
    }

    const int n = static_cast<int>(X.cols());

    // input_with_bias = [X; ones]
    Eigen::MatrixXd input_with_bias(feature_dim() + 1, n);
    input_with_bias.topRows(feature_dim()) = X;
    input_with_bias.row(feature_dim()) = Eigen::RowVectorXd::Ones(n);

    // hidden = sigmoid(W1 * input_with_bias)
    Eigen::MatrixXd hidden = sigmoid_matrix(W1_ * input_with_bias); // [hidden_dim x n]

    // hidden_with_bias = [hidden; ones]
    Eigen::MatrixXd hidden_with_bias(hidden_dim() + 1, n);
    hidden_with_bias.topRows(hidden_dim()) = hidden;
    hidden_with_bias.row(hidden_dim()) = Eigen::RowVectorXd::Ones(n);

    // logits = W2 * hidden_with_bias)
    return W2_ * hidden_with_bias; // [output_dim x n]
}

Eigen::MatrixXd TwoLayerMLP::forward_batch(const Eigen::MatrixXd& X) const {
    return sigmoid_matrix(forward_logits_batch(X));
}

const Eigen::MatrixXd& TwoLayerMLP::weight_input_hidden() const {
    return W1_;
}

const Eigen::MatrixXd& TwoLayerMLP::weight_hidden_output() const {
    return W2_;
}