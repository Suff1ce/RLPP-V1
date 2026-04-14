// Phase 3: RL training math parity vs Python (export_phase3_reference.py). No supervised path.

#include "csv_utils.hpp"
#include "rlpp_training_math.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

static double max_abs_diff(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    return (a - b).cwiseAbs().maxCoeff();
}

static int verify_testdata(const std::string& dir, double tol) {
    using rlpp::getgradient_rl_broadcast;
    using rlpp::compute_smoothed_reward_rlpp;

    const Eigen::MatrixXd W2 = load_csv_matrix(dir + "/W2.csv");
    const Eigen::MatrixXd input_unit = load_csv_matrix(dir + "/input_unit.csv");

    {
        const Eigen::MatrixXd p = load_csv_matrix(dir + "/rl_p_output.csv");
        const Eigen::MatrixXd hidden = load_csv_matrix(dir + "/rl_hidden_unit.csv");
        const Eigen::MatrixXd spk = load_csv_matrix(dir + "/rl_spk.csv");
        Eigen::MatrixXd smoothed_m = load_csv_matrix(dir + "/rl_smoothed_reward.csv");
        Eigen::RowVectorXd smoothed = smoothed_m.row(0);
        const int n = static_cast<int>(input_unit.cols());
        Eigen::MatrixXd d1, d2;
        getgradient_rl_broadcast(smoothed, p, spk, hidden, input_unit, W2, n, d1, d2);
        const Eigen::MatrixXd e1 = load_csv_matrix(dir + "/expected_rl_delta1.csv");
        const Eigen::MatrixXd e2 = load_csv_matrix(dir + "/expected_rl_delta2.csv");
        const double m1 = max_abs_diff(d1, e1);
        const double m2 = max_abs_diff(d2, e2);
        std::cout << "RL gradient max_abs_err delta1=" << m1 << " delta2=" << m2 << "\n";
        if (m1 > tol || m2 > tol) {
            std::cerr << "FAIL RL gradient parity\n";
            return 1;
        }
    }

    {
        Eigen::MatrixXd succ_m = load_csv_matrix(dir + "/success.csv");
        Eigen::RowVectorXd success = succ_m.row(0);
        Eigen::MatrixXd mp_m = load_csv_matrix(dir + "/motor_perform.csv");
        Eigen::VectorXi motor(mp_m.rows());
        for (int i = 0; i < motor.size(); ++i) {
            motor(i) = static_cast<int>(std::llround(mp_m(i, 0)));
        }
        Eigen::MatrixXd rp = load_csv_matrix(dir + "/reward_params.csv");
        const int episode = static_cast<int>(rp(0, 0));
        const int max_ep = static_cast<int>(rp(0, 1));
        const double epsilon = rp(0, 2);
        const double discount_factor = rp(0, 3);
        const int discount_length = static_cast<int>(rp(0, 4));

        Eigen::RowVectorXd sm = compute_smoothed_reward_rlpp(
            success,
            motor,
            epsilon,
            episode,
            max_ep,
            discount_factor,
            discount_length
        );
        const Eigen::MatrixXd ref_m = load_csv_matrix(dir + "/rl_smoothed_reward.csv");
        Eigen::RowVectorXd ref = ref_m.row(0);
        const double mr = (sm - ref).cwiseAbs().maxCoeff();
        std::cout << "smoothed reward max_abs_err=" << mr << "\n";
        if (mr > tol) {
            std::cerr << "FAIL smoothed reward parity\n";
            return 1;
        }
    }

    std::cout << "Phase 3 RL parity OK (gradients + reward)\n";
    return 0;
}

int main(int argc, char** argv) {
    std::string testdata;
    double tol = 1e-9;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            std::cout << "Usage: RLPP_Phase_3 --testdata DIR [--tol X]\n"
                         "  Verify C++ RL training math vs CSVs from Python_Ver/export_phase3_reference.py\n"
                         "\n"
                         "Other Phase 3 executables:\n"
                         "  RLPP_Phase_3_Trainer   full episode weight-update parity vs export_phase3_training_case.py\n"
                         "  RLPP_Phase_3_LoaderEmuVerify   DataLoader + emulator parity vs export_loader_emulator_reference.py\n"
                         "  RLPP_Phase_3_FullLoopVerify   u01 + emulator_manual + RL update vs export_phase3_full_loop_emulator_case.py\n"
                         "  RLPP_Phase_3_FullLoopTrainedVerify   u01 + decodingModel_01/02 (CSV) + RL update vs export_phase3_full_loop_trained_decoder_case.py\n"
                          "  RLPP_Phase_3_RunCaseVerify   exported runner-style batches + emulator + RL update vs export_phase3_run_case.py\n"
                         "  RLPP_Phase_3_Run   end-to-end RL on loader CSVs (RNG), or --training-case for trainer parity\n";
            return 0;
        }
        if (a == "--testdata" && i + 1 < argc) {
            testdata = argv[++i];
            continue;
        }
        if (a == "--tol" && i + 1 < argc) {
            tol = std::stod(argv[++i]);
            continue;
        }
        std::cerr << "Unknown arg: " << a << "\n";
        return 1;
    }

    if (testdata.empty()) {
        const fs::path here = fs::path(argv[0]).parent_path();
        fs::path guess = here / ".." / "Phase_3" / "testdata";
        if (fs::is_directory(guess)) {
            testdata = fs::canonical(guess).string();
        } else {
            guess = fs::path("d:/RLPP-master/C++_Ver/Phase_3/testdata");
            if (fs::is_directory(guess)) {
                testdata = guess.string();
            }
        }
    }

    if (testdata.empty() || !fs::is_directory(testdata)) {
        std::cerr << "Missing testdata. Run: python Python_Ver/export_phase3_reference.py\n"
                     "Then: RLPP_Phase_3 --testdata C++_Ver/Phase_3/testdata\n";
        return 1;
    }

    return verify_testdata(testdata, tol);
}
