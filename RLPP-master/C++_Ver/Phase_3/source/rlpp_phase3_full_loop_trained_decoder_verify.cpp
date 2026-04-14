// Full RLPP loop parity: applynets_priori(u01) -> emulator_real_trained_nn (decodingModel_01/02 CSV)
// vs Python_Ver/export_phase3_full_loop_trained_decoder_case.py

#include "csv_utils.hpp"
#include "rlpp_emulator.hpp"
#include "rlpp_training_math.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

static double max_abs_diff(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    return (a - b).cwiseAbs().maxCoeff();
}

static double max_abs_row_nan_equal(const Eigen::RowVectorXd& a, const Eigen::RowVectorXd& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("max_abs_row_nan_equal: size mismatch");
    }
    double m = 0.0;
    for (int i = 0; i < a.size(); ++i) {
        const double x = a(i);
        const double y = b(i);
        const bool xn = std::isnan(x);
        const bool yn = std::isnan(y);
        if (xn && yn) continue;
        if (xn != yn) return std::numeric_limits<double>::infinity();
        m = std::max(m, std::abs(x - y));
    }
    return m;
}

static Eigen::VectorXi load_csv_vec_i(const std::string& path) {
    Eigen::MatrixXd m = load_csv_matrix(path);
    Eigen::VectorXi out;
    if (m.cols() == 1) {
        out.resize(m.rows());
        for (int i = 0; i < m.rows(); ++i) out(i) = static_cast<int>(std::llround(m(i, 0)));
        return out;
    }
    if (m.rows() == 1) {
        out.resize(m.cols());
        for (int i = 0; i < m.cols(); ++i) out(i) = static_cast<int>(std::llround(m(0, i)));
        return out;
    }
    throw std::runtime_error("expected vector at " + path);
}

static Eigen::RowVectorXd load_success_row(const std::string& path) {
    Eigen::MatrixXd m = load_csv_matrix(path);
    if (m.rows() != 1) {
        throw std::runtime_error("success csv must be 1 x n: " + path);
    }
    return m.row(0);
}

static std::string load_decoder_prefix_file(const std::string& dir) {
    const std::string p = dir + "/decoder_prefix.txt";
    std::ifstream f(p);
    if (!f) {
        throw std::runtime_error("Missing decoder_prefix.txt in " + dir);
    }
    std::string line;
    if (!std::getline(f, line)) {
        throw std::runtime_error("Empty decoder_prefix.txt");
    }
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) {
        line.pop_back();
    }
    if (line.empty()) {
        throw std::runtime_error("decoder_prefix.txt has no path");
    }
    return line;
}

int main(int argc, char** argv) {
    std::string dir = "d:/RLPP-master/C++_Ver/Phase_3/full_loop_decoding01_case";
    if (argc >= 2) dir = argv[1];
    if (!fs::is_directory(dir)) {
        std::cerr << "Missing case dir. Run:\n"
                     "  python Python_Ver/export_phase3_full_loop_trained_decoder_case.py --model 01\n"
                     "  python Python_Ver/export_phase3_full_loop_trained_decoder_case.py --model 02\n";
        return 1;
    }
    const double tol = 1e-9;

    const std::string decoder_prefix = load_decoder_prefix_file(dir);

    const Eigen::MatrixXd input_unit = load_csv_matrix(dir + "/input_unit.csv");
    Eigen::MatrixXd W1 = load_csv_matrix(dir + "/W1_init.csv");
    Eigen::MatrixXd W2 = load_csv_matrix(dir + "/W2_init.csv");

    const Eigen::VectorXi batch_actions = load_csv_vec_i(dir + "/batchActions.csv");
    const Eigen::VectorXi indexes = load_csv_vec_i(dir + "/indexes.csv");
    const int his = static_cast<int>(std::llround(load_csv_matrix(dir + "/his.csv")(0, 0)));

    const Eigen::MatrixXd reward_params = load_csv_matrix(dir + "/reward_params.csv");
    const double epsilon = reward_params(0, 0);
    const int max_episode = static_cast<int>(std::llround(reward_params(0, 1)));
    const double discount_factor = reward_params(0, 2);
    const int discount_length = static_cast<int>(std::llround(reward_params(0, 3)));

    const Eigen::MatrixXd priori_params = load_csv_matrix(dir + "/priori_params.csv");
    const double priori_m = priori_params(0, 0);
    const double priori_n = priori_params(0, 1);

    const Eigen::MatrixXd train_params = load_csv_matrix(dir + "/train_params.csv");
    const int episodes = static_cast<int>(std::llround(train_params(0, 0)));
    const bool is_sim = (train_params(0, 1) >= 0.5);

    const int n = static_cast<int>(input_unit.cols());
    if (batch_actions.size() != n) {
        throw std::runtime_error("batchActions length mismatch");
    }

    try {
        for (int ep = 0; ep < episodes; ++ep) {
            const Eigen::MatrixXd u01 = load_csv_matrix(dir + "/u01_ep" + std::to_string(ep) + ".csv");

            rlpp::ApplyNetsPrioriBatch fwd = rlpp::applynets_priori_forward_with_uniforms(
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
                throw std::runtime_error("forward parity failed");
            }

            rlpp::EmulatorResult emu = rlpp::emulator_real_trained_nn_exact(
                fwd.spk_out, batch_actions, indexes, his, decoder_prefix
            );

            const Eigen::RowVectorXd success_ref = load_success_row(dir + "/success_ep" + std::to_string(ep) + ".csv");
            const Eigen::VectorXi motor_ref = load_csv_vec_i(dir + "/motor_perform_ep" + std::to_string(ep) + ".csv");

            const double es = max_abs_row_nan_equal(emu.success.row(0), success_ref);
            const int mm = (emu.motor_perform.array() != motor_ref.array()).count();
            std::cout << "[ep " << ep << "] trained emulator max_abs_err success=" << es << " motor_mism=" << mm << "\n";
            if (!std::isfinite(es) || es > tol || mm != 0) {
                throw std::runtime_error("trained emulator parity failed");
            }

            const Eigen::RowVectorXd smoothed = rlpp::compute_smoothed_reward_rlpp(
                emu.success.row(0), emu.motor_perform, epsilon, ep, max_episode, discount_factor, discount_length
            );
            const Eigen::MatrixXd sm_ref_m = load_csv_matrix(dir + "/smoothed_ep" + std::to_string(ep) + ".csv");
            const Eigen::RowVectorXd sm_ref = sm_ref_m.row(0);
            const double sm_err = (smoothed - sm_ref).cwiseAbs().maxCoeff();
            std::cout << "[ep " << ep << "] smoothed reward max_abs_err=" << sm_err << "\n";
            if (sm_err > tol) {
                throw std::runtime_error("smoothed reward parity failed");
            }

            Eigen::MatrixXd dW2, dW1;
            rlpp::getgradient_rl_broadcast(
                smoothed, fwd.p_output, fwd.spk_out, fwd.hidden_unit, input_unit, W2, n, dW2, dW1
            );

            const Eigen::MatrixXd dW2_ref = load_csv_matrix(dir + "/dW2_ep" + std::to_string(ep) + ".csv");
            const Eigen::MatrixXd dW1_ref = load_csv_matrix(dir + "/dW1_ep" + std::to_string(ep) + ".csv");
            const double dw2_err = max_abs_diff(dW2, dW2_ref);
            const double dw1_err = max_abs_diff(dW1, dW1_ref);
            std::cout << "[ep " << ep << "] grads max_abs_err dW2=" << dw2_err << " dW1=" << dw1_err << "\n";
            if (dw2_err > tol || dw1_err > tol) {
                throw std::runtime_error("gradient parity failed");
            }

            const double lr = is_sim ? rlpp::learning_rate_rl_simulations(ep, max_episode)
                                     : rlpp::learning_rate_rl_real(ep, max_episode);
            W2 = W2 + lr * dW2;
            W1 = W1 + lr * dW1;

            const Eigen::MatrixXd W2_ref = load_csv_matrix(dir + "/W2_after_ep" + std::to_string(ep) + ".csv");
            const Eigen::MatrixXd W1_ref = load_csv_matrix(dir + "/W1_after_ep" + std::to_string(ep) + ".csv");
            const double w2_err = max_abs_diff(W2, W2_ref);
            const double w1_err = max_abs_diff(W1, W1_ref);
            std::cout << "[ep " << ep << "] weights max_abs_err W2=" << w2_err << " W1=" << w1_err << "\n";
            if (w2_err > tol || w1_err > tol) {
                throw std::runtime_error("weight update parity failed");
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Full-loop trained-decoder case parity OK\n";
    return 0;
}
