#pragma once

#include <Eigen/Dense>
#include <random>
#include <string>
#include <vector>

namespace rlpp {

enum class DataLoaderMode {
    Train,
    Test,
    All
};

struct DataLoaderOpt {
    DataLoaderMode mode = DataLoaderMode::Train;

    // 1-based cursor like Python
    int data_loader_cursor = 1;

    bool shuffle_when_cursor_is_one = true;

    int batch_size = 20;
    int discount_length = 100;

    // trials (MATLAB-style labels: Trials vector values compared against these)
    std::vector<int> train_trials;
    int number_of_train_trials = 0;
    std::vector<int> test_trials;
};

struct DataLoaderBatch {
    Eigen::MatrixXd batch_input;      // [feature_dim x n_time]
    Eigen::MatrixXd batch_m1_truth;   // [Ny x n_time]
    Eigen::VectorXi batch_actions;    // [n_time]
};

/// Exact port of Python utils/DataLoader.py, including:
/// - train cursor logic (1-based), stop inclusive, shuffle when cursor==1
/// - time index expansion with offsets [-discountLength..0]
/// - numpy-style unique sorting
/// - numpy negative indexing semantics (indices < 0 wrap from end)
DataLoaderBatch dataloader_forward_exact(
    const Eigen::MatrixXd& input_ensemble,   // [feature_dim x T]
    const Eigen::MatrixXd& m1_truth,         // [Ny x T]
    const Eigen::VectorXi& actions,          // [T]
    const Eigen::VectorXi& trials,           // [T] values compared to trial IDs
    DataLoaderOpt& opt,
    std::mt19937& rng
);

} // namespace rlpp

