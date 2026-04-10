#include "decoding_model01.hpp"

#include <stdexcept>

Eigen::MatrixXd decodingModel01_forward(const Eigen::MatrixXd& ensemble,
                                        const Eigen::VectorXd& xoffset,
                                        const Eigen::VectorXd& gain,
                                        double ymin,
                                        const Eigen::MatrixXd& IW1_1,
                                        const Eigen::VectorXd& b1,
                                        const Eigen::MatrixXd& LW2_1,
                                        const Eigen::VectorXd& b2
) {
    const int input_dim = static_cast<int>(ensemble.rows());
    const int T = static_cast<int>(ensemble.cols());

    if (input_dim <= 0 || T <= 0) {
        throw std::runtime_error("decodingModel01_forward: ensemble is empty");
    }
    if (xoffset.size() != input_dim || gain.size() != input_dim) {
        throw std::runtime_error("decodingModel01_forward: xoffset/gain size mismatch vs ensemble.rows()");
    }
    if (IW1_1.cols() != input_dim) {
        throw std::runtime_error("decodingModel01_forward: IW1_1.cols != input_dim");
    }
    if (b1.size() != IW1_1.rows()) {
        throw std::runtime_error("decodingModel01_forward: b1.size != IW1_1.rows");
    }
    if (LW2_1.cols() != IW1_1.rows()) {
        throw std::runtime_error("decodingModel01_forward: LW2_1.cols != hidden_dim");
    }
    if (b2.size() != LW2_1.rows()) {
        throw std::runtime_error("decodingModel01_forward: b2.size != num_labels");
    }

    // mapminmax: Xp(i,:) = (X(i,:) - xoffset(i)) * gain(i) + ymin
    Eigen::MatrixXd Xp = ensemble;
    for (int i = 0; i < input_dim; ++i) {
        Xp.row(i) = (Xp.row(i).array() - xoffset(i)) * gain(i) + ymin;
    }

    // Layer 1: a1 = tansig(IW1_1 * Xp + b1), tansig == tanh
    Eigen::MatrixXd n1 = IW1_1 * Xp;   // [hidden x T]
    n1.colwise() += b1;
    Eigen::MatrixXd a1 = n1.array().tanh().matrix();

    // Layer 2 pre-activation: n2 = LW2_1 * a1 + b2
    Eigen::MatrixXd n2 = LW2_1 * a1;   // [num_labels x T]
    n2.colwise() += b2;

    // Softmax per column
    Eigen::MatrixXd y(n2.rows(), n2.cols());
    for (int c = 0; c < n2.cols(); ++c) {
        const double m = n2.col(c).maxCoeff();
        Eigen::VectorXd ex = (n2.col(c).array() - m).exp();
        double den = ex.sum();
        if (den == 0.0) den = 1.0;
        y.col(c) = ex / den;
    }
    
    return y;
}