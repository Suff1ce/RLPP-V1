#include <Eigen/Dense>
#include <iostream>
#include <exception>
#include <random>
#include "csv_utils.hpp"
#include "math_functions.hpp"
#include "two_layer_mlp.hpp"
#include "F1_rat01_bundle.hpp"
#include "F1_rat01_validate.hpp"
#include "F1_rat01_parity.hpp"
#include "replay_runner.hpp"
// int main() {
//     try {
//         std::cout << "Loading matrix from CSV file...\n\n";

//         Eigen::MatrixXd matrix = load_csv_matrix("C++_Ver/Phase_1/data/test.csv");
        
//         std::cout << "Matrix shape: " << matrix.rows() << " x " << matrix.cols() << "\n";
//         std::cout << "Matrix contents:\n" << matrix << "\n\n";

//         save_csv_matrix("C++_Ver/Phase_1/data/test_output.csv", matrix);
//         std::cout << "Matrix saved to data/test_output.csv\n\n";
//     }

//     catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << "\n";
//         return 1;
//     }

//     return 0;
// }

// int main() {
//     // Test sigmoid_scalar
//     std::cout << "sigmoid_scalar(0.0) = " << sigmoid_scalar(0.0) << "\n";

//     // Test vector sigmoid
//     Eigen::VectorXd v(3);
//     v << -1.0, 0.0, 1.0;

//     Eigen::VectorXd s = sigmoid_vector(v);
//     std::cout << "sigmoid([-1, 0, 1]) =\n" << s << "\n\n";

//     // Test argmax_index
//     Eigen::VectorXd a(3);
//     a << 1.0, 7.0, 3.0;

//     std::cout << "argmax_index([1, 7, 3]) = " << argmax_index(a) << "\n\n";

//     // Test Bernoulli sampling
//     Eigen::VectorXd probs(4);
//     probs << 0.0, 0.5, 1.0, 0.8;

//     std::mt19937 rng(42); // fixed seed for reproducibility
//     Eigen::VectorXd sample = sample_Bernoulli(probs, rng);

//     std::cout << "sample_Bernoulli([0, 0.5, 1, 0.8]) =\n" << sample << "\n";

//     return 0;
// }

// int main() {
//     // input_dim = 3, hidden_dim = 2, output_dim = 2
//     Eigen::MatrixXd W1 = Eigen::MatrixXd::Zero(3, 2);
//     Eigen::VectorXd b1 = Eigen::VectorXd::Zero(2);

//     Eigen::MatrixXd W2 = Eigen::MatrixXd::Zero(2, 2);
//     Eigen::VectorXd b2 = Eigen::VectorXd::Zero(2);

//     TwoLayerMLP mlp(W1, b1, W2, b2);

//     Eigen::VectorXd x(3);
//     x << 1.0, -2.0, 0.5;

//     Eigen::VectorXd y_sigmoid = mlp.forward_sigmoid_output(x);
//     Eigen::VectorXd y_linear = mlp.forward_linear_output(x);

//     std::cout << "Sigmoid output:\n" << y_sigmoid << "\n\n";
//     std::cout << "Linear output:\n" << y_linear << "\n";

//     return 0;
// }

// int main() {
//     ExponentialHistoryEncoder encoder(num_inputs, H, tau_bins);

//     for (int t_idx = 0; t_idx < T; ++t_idx) {
//         int t = t_idx + 1;  // 1-based to match Python

//         Eigen::VectorXd spike_bin = ...;  // size num_inputs
//         encoder.observe_bin(spike_bin, t);

//         if (!encoder.can_encode(t)) {
//             continue; // matches Python's idea of ignoring early bins
//         }

//         Eigen::VectorXd x = encoder.encode(t);

//         // use x as network input
//     }

//     return 0;
// }

int main() {
    try {
        Eigen::MatrixXd xoffset_m = load_csv_matrix("D:/rlpp_f1_bundle_rat01/decoder_xoffset.csv");
        Eigen::MatrixXd gain_m    = load_csv_matrix("D:/rlpp_f1_bundle_rat01/decoder_gain.csv");
        Eigen::MatrixXd ymin_m    = load_csv_matrix("D:/rlpp_f1_bundle_rat01/decoder_ymin.csv");

        auto to_vector = [](const Eigen::MatrixXd& M) -> Eigen::VectorXd {
            if (M.cols() == 1) return M.col(0);
            if (M.rows() == 1) return M.row(0).transpose();
            throw std::runtime_error("Expected a vector CSV (Nx1 or 1xN).");
        };
        
        Eigen::VectorXd xoffset = to_vector(xoffset_m);
        Eigen::VectorXd gain    = to_vector(gain_m);
        
        if (ymin_m.size() != 1) {
            throw std::runtime_error("decoder_ymin.csv must be a single value.");
        }
        double ymin = ymin_m(0, 0);
        
        F1Bundle bundle = load_f1_bundle("D:/rlpp_f1_bundle_rat01");
        validate_f1_bundle_or_throw(bundle, /*his=*/-1);

        run_encoder_parity_or_throw(bundle);
        std::cout << "F1 encoder parity PASSED.\n";

        run_generator_parity_or_throw(bundle);
        std::cout << "F1 generator parity PASSED.\n";

        run_decoder_feature_parity_or_throw(bundle);
        std::cout << "F1 decoder feature parity PASSED.\n";

        run_decoder_output_model01_parity_or_throw(bundle, xoffset, gain, ymin);
        std::cout << "F1 decoder output model01 parity PASSED.\n";

        // --- Replay 1: deterministic (golden wiring check) ---
        {
            ReplayConfig cfg;
            cfg.out_dir = "D:/RLPP-master/C++_Ver/Phase_2/logs/Release/deterministic";
            cfg.dump_latency_us = true;
            cfg.dump_generator_probs = false;
            cfg.dump_downstream_spikes = false;
            cfg.dump_decoder_features = false;

            cfg.mode = ReplayMode::DeterministicFromBundleSpikes;
            cfg.rng_seed = 0;
            cfg.valid_col_start = 0;
            cfg.valid_col_count = 100000;

            ReplayResult r = run_replay_or_throw(bundle, xoffset, gain, ymin, cfg);
            std::cout << "[det] Replay: max_abs_err=" << r.max_abs_err
                    << " mean_abs_err=" << r.mean_abs_err << "\n";
            std::cout << "[det] Replay: label mismatches=" << r.label_mismatches
                    << " / " << r.labels.size() << "\n";
        }

        // --- Replay 2: sampled (stochastic, Phase 2.3/2.4.3) ---
        {
            const int seed_start = 0;
            const int seed_count = 10;

            for (int s = seed_start; s < seed_start + seed_count; ++s) {
                ReplayConfig cfg;
                cfg.out_dir = "D:/RLPP-master/C++_Ver/Phase_2/logs/Release/sampled_seed" + std::to_string(s);
                cfg.dump_latency_us = true;
                cfg.dump_generator_probs = true;
                cfg.dump_downstream_spikes = true;
                cfg.dump_decoder_features = false;

                cfg.mode = ReplayMode::SampledFromGeneratorProbs;
                cfg.rng_seed = static_cast<unsigned int>(s);
                cfg.valid_col_start = 0;
                cfg.valid_col_count = 100000;

                ReplayResult r = run_replay_or_throw(bundle, xoffset, gain, ymin, cfg);

                std::cout << "[samp seed=" << s << "] mismatches="
                        << r.label_mismatches << "/" << r.labels.size()
                        << " rate=" << (static_cast<double>(r.label_mismatches) / r.labels.size())
                        << "\n";
            }
        }

        return 0;
    } 
    
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}