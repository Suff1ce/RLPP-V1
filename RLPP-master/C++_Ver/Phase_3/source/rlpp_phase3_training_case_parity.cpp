#include "rlpp_phase3_training_case_parity.hpp"

#include "csv_utils.hpp"
#include "rlpp_training_math.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace rlpp {

namespace {

double max_abs_diff(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    return (a - b).cwiseAbs().maxCoeff();
}

Eigen::VectorXi load_csv_vector_i32(const std::string& path) {
    Eigen::MatrixXd m = load_csv_matrix(path);
    Eigen::VectorXd v;
    if (m.rows() == 1 && m.cols() >= 1) {
        v = m.row(0).transpose();
    } else if (m.cols() == 1 && m.rows() >= 1) {
        v = m.col(0);
    } else {
        throw std::runtime_error("Expected vector CSV at " + path);
    }
    Eigen::VectorXi out(v.size());
    for (int i = 0; i < v.size(); ++i) {
        out(i) = static_cast<int>(std::llround(v(i)));
    }
    return out;
}

} // namespace

int run_phase3_training_case_parity(const std::string& dir, double tol) {
    const Eigen::MatrixXd input_unit = load_csv_matrix(dir + "/input_unit.csv");
    Eigen::MatrixXd W1 = load_csv_matrix(dir + "/W1_init.csv");
    Eigen::MatrixXd W2 = load_csv_matrix(dir + "/W2_init.csv");

    const Eigen::VectorXi motor_perform = load_csv_vector_i32(dir + "/motor_perform.csv");
    const Eigen::MatrixXd success_m = load_csv_matrix(dir + "/success.csv");
    if (success_m.rows() != 1) {
        throw std::runtime_error("success.csv must be 1 x n");
    }
    const Eigen::RowVectorXd success = success_m.row(0);

    const Eigen::MatrixXd reward_params = load_csv_matrix(dir + "/reward_params.csv");
    if (reward_params.rows() < 1 || reward_params.cols() < 4) {
        throw std::runtime_error("reward_params.csv must be 1x4: epsilon,max_episode,discount_factor,discount_length");
    }
    const double epsilon = reward_params(0, 0);
    const int max_episode = static_cast<int>(std::llround(reward_params(0, 1)));
    const double discount_factor = reward_params(0, 2);
    const int discount_length = static_cast<int>(std::llround(reward_params(0, 3)));

    const Eigen::MatrixXd priori_params = load_csv_matrix(dir + "/priori_params.csv");
    if (priori_params.rows() < 1 || priori_params.cols() < 2) {
        throw std::runtime_error("priori_params.csv must be 1x2: M,N");
    }
    const double priori_m = priori_params(0, 0);
    const double priori_n = priori_params(0, 1);

    const Eigen::MatrixXd train_params = load_csv_matrix(dir + "/train_params.csv");
    if (train_params.rows() < 1 || train_params.cols() < 2) {
        throw std::runtime_error("train_params.csv must be 1x2: episodes,is_simulations_flag");
    }
    const int episodes = static_cast<int>(std::llround(train_params(0, 0)));
    const bool is_sim = (train_params(0, 1) >= 0.5);

    const int n = static_cast<int>(input_unit.cols());
    if (motor_perform.size() != n || success.size() != n) {
        throw std::runtime_error("motor_perform/success length mismatch vs input_unit");
    }

    for (int ep = 0; ep < episodes; ++ep) {
        const Eigen::MatrixXd u01 = load_csv_matrix(dir + "/u01_ep" + std::to_string(ep) + ".csv");

        ApplyNetsPrioriBatch fwd = applynets_priori_forward_with_uniforms(
            input_unit, W1, W2, ep, priori_m, priori_n, u01
        );

        const Eigen::MatrixXd p_ref = load_csv_matrix(dir + "/p_ep" + std::to_string(ep) + ".csv");
        const Eigen::MatrixXd h_ref = load_csv_matrix(dir + "/hidden_ep" + std::to_string(ep) + ".csv");
        const Eigen::MatrixXd s_ref = load_csv_matrix(dir + "/spk_ep" + std::to_string(ep) + ".csv");

        const double p_err = max_abs_diff(fwd.p_output, p_ref);
        const double h_err = max_abs_diff(fwd.hidden_unit, h_ref);
        const double s_err = max_abs_diff(fwd.spk_out, s_ref);
        std::cout << "[ep " << ep << "] forward max_abs_err p=" << p_err
                  << " hidden=" << h_err << " spk=" << s_err << "\n";
        if (p_err > tol || h_err > tol || s_err > tol) {
            throw std::runtime_error("forward parity failed at episode " + std::to_string(ep));
        }

        const Eigen::RowVectorXd smoothed = compute_smoothed_reward_rlpp(
            success, motor_perform, epsilon, ep, max_episode, discount_factor, discount_length
        );
        const Eigen::MatrixXd sm_ref_m = load_csv_matrix(dir + "/smoothed_ep" + std::to_string(ep) + ".csv");
        const Eigen::RowVectorXd sm_ref = sm_ref_m.row(0);
        const double sm_err = (smoothed - sm_ref).cwiseAbs().maxCoeff();
        std::cout << "[ep " << ep << "] smoothed reward max_abs_err=" << sm_err << "\n";
        if (sm_err > tol) {
            throw std::runtime_error("smoothed reward parity failed at episode " + std::to_string(ep));
        }

        Eigen::MatrixXd dW2, dW1;
        getgradient_rl_broadcast(
            smoothed, fwd.p_output, fwd.spk_out, fwd.hidden_unit, input_unit, W2, n, dW2, dW1
        );

        const Eigen::MatrixXd dW2_ref = load_csv_matrix(dir + "/dW2_ep" + std::to_string(ep) + ".csv");
        const Eigen::MatrixXd dW1_ref = load_csv_matrix(dir + "/dW1_ep" + std::to_string(ep) + ".csv");
        const double dw2_err = max_abs_diff(dW2, dW2_ref);
        const double dw1_err = max_abs_diff(dW1, dW1_ref);
        std::cout << "[ep " << ep << "] grads max_abs_err dW2=" << dw2_err << " dW1=" << dw1_err << "\n";
        if (dw2_err > tol || dw1_err > tol) {
            throw std::runtime_error("gradient parity failed at episode " + std::to_string(ep));
        }

        const double lr = is_sim ? learning_rate_rl_simulations(ep, max_episode)
                                 : learning_rate_rl_real(ep, max_episode);
        W2 = W2 + lr * dW2;
        W1 = W1 + lr * dW1;

        const Eigen::MatrixXd W2_ref = load_csv_matrix(dir + "/W2_after_ep" + std::to_string(ep) + ".csv");
        const Eigen::MatrixXd W1_ref = load_csv_matrix(dir + "/W1_after_ep" + std::to_string(ep) + ".csv");
        const double w2_err = max_abs_diff(W2, W2_ref);
        const double w1_err = max_abs_diff(W1, W1_ref);
        std::cout << "[ep " << ep << "] weights max_abs_err W2=" << w2_err << " W1=" << w1_err << "\n";
        if (w2_err > tol || w1_err > tol) {
            throw std::runtime_error("weight update parity failed at episode " + std::to_string(ep));
        }
    }

    std::cout << "Phase 3 trainer parity OK (full episode updates)\n";
    return 0;
}

} // namespace rlpp
