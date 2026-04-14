#pragma once

#include <Eigen/Dense>
#include <string>

namespace rlpp {

struct EmulatorResult {
    Eigen::RowVectorXd success;     // [1 x T], may contain NaN
    double rate = 0.0;              // nanmean of success (like python)
    Eigen::VectorXi motor_perform;  // [T], values 1..K
    Eigen::MatrixXd ensemble;       // [(his+1)*Ny x T] for real emulator; empty for simu
};

/// Exact port of Python decoding/emulator_simu.py (+ decodingModel_simulation.py)
EmulatorResult emulator_simu_exact(
    const Eigen::MatrixXd& spikes,        // [Ny x T] (Ny must be 2 for decodingModel_simulation)
    const Eigen::VectorXi& motor_expect,  // [T]
    const Eigen::VectorXi& indexes,       // length Ny
    int his,
    const std::string& model_name
);

/// Exact port of Python decoding/emulator_real.py for modelName=decodingModel_manual (no external params).
EmulatorResult emulator_real_manual_exact(
    const Eigen::MatrixXd& spikes,        // [Ny x T]
    const Eigen::VectorXi& motor_expect,  // [T]
    const Eigen::VectorXi& indexes,       // length Ny
    int his
);

/// Same as emulator_real with modelName decodingModel_01 or decodingModel_02: ensemble + trained net from CSV
/// (export Python_Ver/export_decoding_params_to_csv.py from decodingModel_*_para.mat).
EmulatorResult emulator_real_trained_nn_exact(
    const Eigen::MatrixXd& spikes,
    const Eigen::VectorXi& motor_expect,
    const Eigen::VectorXi& indexes,
    int his,
    const std::string& params_csv_prefix
);

} // namespace rlpp

