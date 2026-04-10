#include "decoder_history_buffer.hpp"

#include <stdexcept>

DecoderHistoryBuffer::DecoderHistoryBuffer(int num_outputs, int num_lags)
    : Ny(num_outputs),
      Hd(num_lags),
      history(Eigen::MatrixXd::Zero(num_outputs, num_lags)) {
    if (Ny <= 0) {
        throw std::runtime_error("DecoderHistoryBuffer: num_outputs must be positive");
    }
    if (Hd <= 0) {
        throw std::runtime_error("DecoderHistoryBuffer: num_lags must be positive");
    }
}

void DecoderHistoryBuffer::push(const Eigen::VectorXd& spikes) {
    if (spikes.size() != Ny) {
        throw std::runtime_error("DecoderHistoryBuffer::push: spike vector size mismatch");
    }

    if (Hd > 1) {
        history.leftCols(Hd - 1) = history.rightCols(Hd - 1);
    }
    history.col(Hd - 1) = spikes;
}

const Eigen::MatrixXd& DecoderHistoryBuffer::matrix() const {
    return history;
}

Eigen::VectorXd DecoderHistoryBuffer::flatten_for_python_decoder() const {
    Eigen::VectorXd flat(Ny * Hd);

    int k = 0;
    for (int col = Hd - 1; col >= 0; --col) {
        for (int row = 0; row < Ny; ++row) {
            flat(k++) = history(row, col);
        }
    }

    return flat;
}

int DecoderHistoryBuffer::num_outputs() const {
    return Ny;
}

int DecoderHistoryBuffer::num_lags() const {
    return Hd;
}

int DecoderHistoryBuffer::feature_size() const {
    return Ny * Hd;
}