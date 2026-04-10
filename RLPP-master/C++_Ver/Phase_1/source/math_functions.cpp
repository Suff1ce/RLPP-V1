#include "math_functions.hpp"

#include <cmath>
#include <stdexcept>

double sigmoid_scalar(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

Eigen::VectorXd sigmoid_vector(const Eigen::VectorXd& v) {
    Eigen::VectorXd result(v.size());
    for (int i = 0; i < v.size(); i++) {
        result(i) = sigmoid_scalar(v(i));
    }
    return result;
}

Eigen::MatrixXd sigmoid_matrix(const Eigen::MatrixXd& M) {
    Eigen::MatrixXd result(M.rows(), M.cols());
    for (int i = 0; i < M.rows(); i++) {
        for (int j = 0; j < M.cols(); j++) {
            result(i, j) = sigmoid_scalar(M(i, j));
        }
    }
    return result;
}

int argmax_index(const Eigen::VectorXd& v) {
    if (v.size() == 0) {
        throw std::invalid_argument("Input vector is empty");
    }
    int max_index = 0;
    for (int i = 1; i < v.size(); i++) {
        if (v(i) > v(max_index)) {
            max_index = i;
        }
    }
    return max_index;
}

Eigen::VectorXd sample_Bernoulli(const Eigen::VectorXd& probabilities, std::mt19937& rng) {
    Eigen::VectorXd samples(probabilities.size());
    for (int i = 0; i < probabilities.size(); i++) {
        double p = probabilities(i);
        if (p < 0.0 || p > 1.0) {
            throw std::invalid_argument("Probabilities must be in the range [0, 1]");
        }
        std::bernoulli_distribution dist(probabilities(i));
        samples(i) = dist(rng) ? 1.0 : 0.0;
    }
    return samples;
}