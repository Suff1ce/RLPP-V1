#pragma once

#include <Eigen/Dense>

class DecoderHistoryBuffer {
public:
    // num_lags includes lag 0 (current bin).
    // To match Python emulator_real.py, use:
    // num_lags = his + 1
    DecoderHistoryBuffer(int num_outputs, int num_lags);

    void push(const Eigen::VectorXd& spikes);

    const Eigen::MatrixXd& matrix() const;

    // Flatten in Python decoder order:
    // lag 0 first (current/newest), then lag 1, ..., oldest lag last.
    Eigen::VectorXd flatten_for_python_decoder() const;

    int num_outputs() const;
    int num_lags() const;
    int feature_size() const;

private:
    int Ny;
    int Hd; // total lag blocks, including current bin
    Eigen::MatrixXd history; // shape Ny x Hd, columns oldest -> newest
};