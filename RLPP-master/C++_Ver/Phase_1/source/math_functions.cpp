#include "math_functions.hpp"

#include <cmath>
#include <stdexcept>

Eigen::VectorXd apply_sorted_indices_1based(
    const Eigen::VectorXd& v_unsorted,
    const Eigen::VectorXi& sorted_indices_1based
) {
    if (sorted_indices_1based.size() == 0) {
        return v_unsorted;
    }
    const int n = static_cast<int>(v_unsorted.size());
    if (sorted_indices_1based.size() != n) {
        throw std::runtime_error("sorted_indices_1based length != vector length");
    }
    Eigen::VectorXd out(n);
    for (int i = 0; i < n; ++i) {
        const int src = sorted_indices_1based(i) - 1;
        if (src < 0 || src >= n) {
            throw std::runtime_error("sorted index out of range");
        }
        out(i) = v_unsorted(src);
    }
    return out;
}

Eigen::MatrixXd apply_sorted_indices_1based_matrix(
    const Eigen::MatrixXd& spikes_Ny_by_T,
    const Eigen::VectorXi& sorted_indices_1based
) {
    if (sorted_indices_1based.size() == 0) {
        return spikes_Ny_by_T;
    }
    const int Ny = static_cast<int>(spikes_Ny_by_T.rows());
    const int T = static_cast<int>(spikes_Ny_by_T.cols());
    if (sorted_indices_1based.size() != Ny) {
        throw std::runtime_error(
            "sortedIndices length must equal Ny (got " + std::to_string(sorted_indices_1based.size()) +
            ", Ny=" + std::to_string(Ny) + ")");
    }
    Eigen::MatrixXd out(Ny, T);
    for (int c = 0; c < T; ++c) {
        Eigen::VectorXd col = spikes_Ny_by_T.col(c);
        out.col(c) = apply_sorted_indices_1based(col, sorted_indices_1based);
    }
    return out;
}

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