#pragma once

#include <Eigen/Dense>
#include <random>

// MATLAB 1-based sortedIndices: out(i) = v_unsorted(sorted_indices(i)-1). Empty indices => no reorder.
Eigen::VectorXd apply_sorted_indices_1based(
    const Eigen::VectorXd& v_unsorted,
    const Eigen::VectorXi& sorted_indices_1based
);

// Same row permutation applied to each column (e.g. Ny x T downstream spikes).
Eigen::MatrixXd apply_sorted_indices_1based_matrix(
    const Eigen::MatrixXd& spikes_Ny_by_T,
    const Eigen::VectorXi& sorted_indices_1based
);

double sigmoid_scalar(double x);
Eigen::VectorXd sigmoid_vector(const Eigen::VectorXd& v);
Eigen::MatrixXd sigmoid_matrix(const Eigen::MatrixXd& M);
int argmax_index(const Eigen::VectorXd& v);
Eigen::VectorXd sample_Bernoulli(const Eigen::VectorXd& probabilities, std::mt19937& rng);