// Verify C++ DataLoader + emulators against Python exports.

#include "csv_utils.hpp"
#include "rlpp_dataloader.hpp"
#include "rlpp_emulator.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

static double max_abs(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    return (a - b).cwiseAbs().maxCoeff();
}

static double max_abs_nan_equal(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        throw std::runtime_error("max_abs_nan_equal: shape mismatch");
    }
    double m = 0.0;
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < a.cols(); ++j) {
            const double x = a(i, j);
            const double y = b(i, j);
            const bool xn = std::isnan(x);
            const bool yn = std::isnan(y);
            if (xn && yn) continue;
            if (xn != yn) return std::numeric_limits<double>::infinity();
            m = std::max(m, std::abs(x - y));
        }
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

int main(int argc, char** argv) {
    std::string dir = "d:/RLPP-master/C++_Ver/Phase_3/loader_emulator_case";
    if (argc >= 2) dir = argv[1];
    if (!fs::is_directory(dir)) {
        std::cerr << "Missing case dir. Run: python Python_Ver/export_loader_emulator_reference.py\n";
        return 1;
    }

    // ----- DataLoader -----
    {
        const Eigen::MatrixXd input = load_csv_matrix(dir + "/inputEnsemble.csv");
        const Eigen::MatrixXd m1 = load_csv_matrix(dir + "/M1_truth.csv");
        const Eigen::VectorXi actions = load_csv_vec_i(dir + "/Actions.csv");
        const Eigen::VectorXi trials = load_csv_vec_i(dir + "/Trials.csv");
        const Eigen::VectorXi train_trials = load_csv_vec_i(dir + "/opt_trainTrials.csv");
        const Eigen::MatrixXd meta = load_csv_matrix(dir + "/opt_meta.csv");
        const Eigen::VectorXi trial_indexes_ref = load_csv_vec_i(dir + "/trialIndexes.csv");
        const Eigen::VectorXi time_indexes_ref = load_csv_vec_i(dir + "/time_indexes.csv");

        rlpp::DataLoaderOpt opt;
        opt.mode = rlpp::DataLoaderMode::Train;
        opt.data_loader_cursor = 1;
        opt.shuffle_when_cursor_is_one = false; // python already permuted trainTrials for this case
        opt.batch_size = static_cast<int>(std::llround(meta(0, 1)));
        opt.discount_length = static_cast<int>(std::llround(meta(0, 2)));
        opt.number_of_train_trials = static_cast<int>(std::llround(meta(0, 3)));
        opt.train_trials.resize(train_trials.size());
        for (int i = 0; i < train_trials.size(); ++i) opt.train_trials[static_cast<std::size_t>(i)] = train_trials(i);

        std::mt19937 rng(0); // not used for this parity check; shuffle already baked into opt_trainTrials.csv
        rlpp::DataLoaderBatch b = rlpp::dataloader_forward_exact(input, m1, actions, trials, opt, rng);

        const Eigen::MatrixXd ref_in = load_csv_matrix(dir + "/batchInput.csv");
        const Eigen::MatrixXd ref_m1 = load_csv_matrix(dir + "/batchM1_truth.csv");
        const Eigen::VectorXi ref_act = load_csv_vec_i(dir + "/batchActions.csv");

        const double e1 = max_abs(b.batch_input, ref_in);
        const double e2 = max_abs(b.batch_m1_truth, ref_m1);
        const int mism_act = (b.batch_actions.array() != ref_act.array()).count();
        std::cout << "DataLoader: max_abs_err input=" << e1 << " m1=" << e2
                  << " action_mism=" << mism_act << "\n";
        if (e1 > 0 || e2 > 0 || mism_act != 0) {
            std::cerr << "FAIL DataLoader parity\n";
            return 1;
        }
    }

    // ----- emulator_simu -----
    {
        const Eigen::MatrixXd spikes = load_csv_matrix(dir + "/simu_spikes.csv");
        const Eigen::VectorXi motor_expect = load_csv_vec_i(dir + "/simu_motorExpect.csv");
        const Eigen::VectorXi indexes = load_csv_vec_i(dir + "/simu_indexes.csv");
        const int his = static_cast<int>(std::llround(load_csv_matrix(dir + "/simu_his.csv")(0, 0)));

        rlpp::EmulatorResult r = rlpp::emulator_simu_exact(spikes, motor_expect, indexes, his, "decodingModel_simulation");

        const Eigen::MatrixXd ref_success_m = load_csv_matrix(dir + "/simu_success.csv");
        const Eigen::VectorXi ref_motor = load_csv_vec_i(dir + "/simu_motorPerform.csv");
        const double ref_rate = load_csv_matrix(dir + "/simu_rate.csv")(0, 0);

        const double es = max_abs_nan_equal(r.success, ref_success_m);
        const int mm = (r.motor_perform.array() != ref_motor.array()).count();
        const double er = std::abs(r.rate - ref_rate);
        std::cout << "emulator_simu: max_abs_err success=" << es << " motor_mism=" << mm << " rate_err=" << er << "\n";
        if (!std::isfinite(es) || es > 0 || mm != 0 || er > 0) {
            std::cerr << "FAIL emulator_simu parity\n";
            return 1;
        }
    }

    // ----- emulator_real manual -----
    {
        const Eigen::MatrixXd spikes = load_csv_matrix(dir + "/real_spikes.csv");
        const Eigen::VectorXi motor_expect = load_csv_vec_i(dir + "/real_motorExpect.csv");
        const Eigen::VectorXi indexes = load_csv_vec_i(dir + "/real_indexes.csv");
        const int his = static_cast<int>(std::llround(load_csv_matrix(dir + "/real_his.csv")(0, 0)));

        rlpp::EmulatorResult r = rlpp::emulator_real_manual_exact(spikes, motor_expect, indexes, his);

        const Eigen::MatrixXd ref_success_m = load_csv_matrix(dir + "/real_success.csv");
        const Eigen::VectorXi ref_motor = load_csv_vec_i(dir + "/real_motorPerform.csv");
        const double ref_rate = load_csv_matrix(dir + "/real_rate.csv")(0, 0);
        const Eigen::MatrixXd ref_ensemble = load_csv_matrix(dir + "/real_ensemble.csv");

        const double es = max_abs_nan_equal(r.success, ref_success_m);
        const int mm = (r.motor_perform.array() != ref_motor.array()).count();
        const double er = std::abs(r.rate - ref_rate);
        const double ee = max_abs(r.ensemble, ref_ensemble);

        std::cout << "emulator_real_manual: max_abs_err success=" << es << " motor_mism=" << mm
                  << " rate_err=" << er << " ensemble_err=" << ee << "\n";
        if (!std::isfinite(es) || es > 0 || mm != 0 || er > 0 || ee > 0) {
            std::cerr << "FAIL emulator_real_manual parity\n";
            return 1;
        }
    }

    // ----- emulator_real decodingModel_01 / 02 (trained net from CSV) -----
    {
        const Eigen::MatrixXd spikes = load_csv_matrix(dir + "/real01_spikes.csv");
        const Eigen::VectorXi motor_expect = load_csv_vec_i(dir + "/real01_motorExpect.csv");
        const Eigen::VectorXi indexes = load_csv_vec_i(dir + "/real01_indexes.csv");
        const int his = static_cast<int>(std::llround(load_csv_matrix(dir + "/real01_his.csv")(0, 0)));

        rlpp::EmulatorResult r1 =
            rlpp::emulator_real_trained_nn_exact(spikes, motor_expect, indexes, his, dir + "/decoding01");

        const Eigen::MatrixXd ref_success1 = load_csv_matrix(dir + "/real01_success.csv");
        const Eigen::VectorXi ref_motor1 = load_csv_vec_i(dir + "/real01_motorPerform.csv");
        const double ref_rate1 = load_csv_matrix(dir + "/real01_rate.csv")(0, 0);
        const Eigen::MatrixXd ref_ensemble1 = load_csv_matrix(dir + "/real01_ensemble.csv");

        const double es1 = max_abs_nan_equal(r1.success, ref_success1);
        const int mm1 = (r1.motor_perform.array() != ref_motor1.array()).count();
        const double er1 = std::abs(r1.rate - ref_rate1);
        const double ee1 = max_abs(r1.ensemble, ref_ensemble1);

        std::cout << "emulator_real d01: max_abs_err success=" << es1 << " motor_mism=" << mm1
                  << " rate_err=" << er1 << " ensemble_err=" << ee1 << "\n";
        if (!std::isfinite(es1) || es1 > 0 || mm1 != 0 || er1 > 0 || ee1 > 0) {
            std::cerr << "FAIL emulator_real decodingModel_01 parity\n";
            return 1;
        }

        const Eigen::MatrixXd spikes2 = load_csv_matrix(dir + "/real02_spikes.csv");
        const Eigen::VectorXi indexes2 = load_csv_vec_i(dir + "/real02_indexes.csv");
        const int his2 = static_cast<int>(std::llround(load_csv_matrix(dir + "/real02_his.csv")(0, 0)));

        rlpp::EmulatorResult r2 = rlpp::emulator_real_trained_nn_exact(
            spikes2, motor_expect, indexes2, his2, dir + "/decoding02"
        );

        const Eigen::MatrixXd ref_success2 = load_csv_matrix(dir + "/real02_success.csv");
        const Eigen::VectorXi ref_motor2 = load_csv_vec_i(dir + "/real02_motorPerform.csv");
        const double ref_rate2 = load_csv_matrix(dir + "/real02_rate.csv")(0, 0);
        const Eigen::MatrixXd ref_ensemble2 = load_csv_matrix(dir + "/real02_ensemble.csv");

        const double es2 = max_abs_nan_equal(r2.success, ref_success2);
        const int mm2 = (r2.motor_perform.array() != ref_motor2.array()).count();
        const double er2 = std::abs(r2.rate - ref_rate2);
        const double ee2 = max_abs(r2.ensemble, ref_ensemble2);

        std::cout << "emulator_real d02: max_abs_err success=" << es2 << " motor_mism=" << mm2
                  << " rate_err=" << er2 << " ensemble_err=" << ee2 << "\n";
        if (!std::isfinite(es2) || es2 > 0 || mm2 != 0 || er2 > 0 || ee2 > 0) {
            std::cerr << "FAIL emulator_real decodingModel_02 parity\n";
            return 1;
        }
    }

    std::cout << "Loader + emulator parity OK\n";
    return 0;
}

