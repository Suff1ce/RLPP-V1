// Phase 3: end-to-end RLPP loop in C++ (RL-only).
//
// Mirrors Python_Ver/training/RLPP.py at a high level:
//   DataLoader -> applynets_priori -> emulator_simu/emulator_real -> smoothed reward -> RL gradients -> weight updates
//
// This runner is designed to be safe-by-default:
// - Defaults to decodingModel_01 (trained decoder) for "Real" mode
// - Validates Ny/his against decoder input dim before running

#include "csv_utils.hpp"
#include "rlpp_dataloader.hpp"
#include "rlpp_emulator.hpp"
#include "rlpp_training_math.hpp"
#include "rlpp_decoding_params.hpp"
#include "rlpp_phase3_training_case_parity.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

enum class DataIndex { Real, Simulations };
enum class MotorModel { Manual, Decoding01, Decoding02 };

static std::string to_string(DataIndex d) {
    return d == DataIndex::Simulations ? "Simulations" : "Real";
}

static std::string to_string(MotorModel m) {
    switch (m) {
        case MotorModel::Manual: return "decodingModel_manual";
        case MotorModel::Decoding01: return "decodingModel_01";
        case MotorModel::Decoding02: return "decodingModel_02";
    }
    return "unknown";
}

static DataIndex parse_data_index(const std::string& s) {
    if (s == "Real" || s == "real") return DataIndex::Real;
    if (s == "Simulations" || s == "simulations" || s == "Sim" || s == "sim") return DataIndex::Simulations;
    throw std::runtime_error("Unknown --data-index: " + s);
}

static MotorModel parse_model(const std::string& s) {
    if (s == "decodingModel_manual" || s == "manual") return MotorModel::Manual;
    if (s == "decodingModel_01" || s == "01" || s == "d01") return MotorModel::Decoding01;
    if (s == "decodingModel_02" || s == "02" || s == "d02") return MotorModel::Decoding02;
    throw std::runtime_error("Unknown --model: " + s);
}

static Eigen::VectorXi identity_indexes_1based(int Ny) {
    Eigen::VectorXi idx(Ny);
    for (int i = 0; i < Ny; ++i) idx(i) = i + 1;
    return idx;
}

static Eigen::VectorXi shuffled_indexes_1based(int Ny, std::mt19937& rng) {
    std::vector<int> v(static_cast<std::size_t>(Ny));
    std::iota(v.begin(), v.end(), 1);
    std::shuffle(v.begin(), v.end(), rng);
    Eigen::VectorXi out(Ny);
    for (int i = 0; i < Ny; ++i) out(i) = v[static_cast<std::size_t>(i)];
    return out;
}

struct NyHis {
    int Ny = 0;
    int his = 0;
};

static NyHis choose_ny_his_for_input_dim(int input_dim) {
    // Match the bundled MATLAB decoders by default.
    if (input_dim == 122) return NyHis{61, 1};
    if (input_dim == 183) return NyHis{61, 2};

    // Generic fallback: smallest his that factors the input.
    for (int his = 0; his <= 64; ++his) {
        const int denom = his + 1;
        if (input_dim % denom != 0) continue;
        const int Ny = input_dim / denom;
        if (Ny >= 1) return NyHis{Ny, his};
    }
    throw std::runtime_error("Cannot factor decoder input_dim into Ny*(his+1): " + std::to_string(input_dim));
}

static void usage() {
    std::cout
        << "Usage: RLPP_Phase_3_Run [options]\n"
        << "  End-to-end RL (DataLoader -> applynets_priori -> emulator -> reward -> grads -> update).\n"
        << "  For deterministic per-episode parity vs Python, use RLPP_Phase_3_Trainer + export_phase3_training_case.py.\n"
        << "\n"
        << "  Required CSVs under --case-dir: inputEnsemble.csv, M1_truth.csv, Actions.csv, Trials.csv\n"
        << "  Optional: opt_trainTrials.csv + opt_meta.csv (else K=max(Trials), trainTrials=1..K)\n"
        << "\n"
        << "  --case-dir DIR         CSV case dir with inputEnsemble/Actions/Trials (+ opt meta)\n"
        << "                         default: d:/RLPP-master/C++_Ver/Phase_3/loader_emulator_case\n"
        << "  --data-index Real|Simulations   default: Real\n"
        << "  --model decodingModel_01|decodingModel_02|decodingModel_manual   default: decodingModel_01\n"
        << "  --decoder-prefix PATH  prefix to decoder CSVs (no trailing _xoffset)\n"
        << "                         default: d:/RLPP-master/C++_Ver/Phase_3/decoding_params/decoding01\n"
        << "  --episodes N           default: 5\n"
        << "  --hidden H             default: 20\n"
        << "  --priori-m M           default: 1\n"
        << "  --priori-n N           default: 1\n"
        << "  --epsilon E            default: 0.3\n"
        << "  --max-episode N        default: 1000 (for epsilon schedule)\n"
        << "  --discount-factor D    default: 0.9\n"
        << "  --discount-length L    default: 5\n"
        << "  --shuffle-train-trials 0|1   shuffle trainTrials when cursor==1 (Python behavior), default: 1\n"
        << "  --seed S               default: 2026\n"
        << "\n"
        << "Deterministic parity (same as RLPP_Phase_3_Trainer; no DataLoader/emulator):\n"
        << "  --training-case DIR    run vs export_phase3_training_case.py CSVs (u01_ep*.csv forward)\n"
        << "  --tol X                tolerance for parity checks (with --training-case), default 1e-9\n"
        << "\n";
}

static int run_phase3_main(int argc, char** argv) {
    std::string training_case_dir;
    double parity_tol = 1e-9;

    std::string case_dir = "d:/RLPP-master/C++_Ver/Phase_3/loader_emulator_case";
    DataIndex data_index = DataIndex::Real;
    MotorModel model = MotorModel::Decoding01;
    std::string decoder_prefix = "d:/RLPP-master/C++_Ver/Phase_3/decoding_params/decoding01";

    int episodes = 5;
    int H = 20;
    double priori_m = 1.0;
    double priori_n = 1.0;

    double epsilon = 0.3;
    int max_episode_for_schedule = 1000;
    double discount_factor = 0.9;
    int discount_length = 5;

    bool shuffle_train_trials = true;

    unsigned int seed = 2026;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            usage();
            return 0;
        }
        auto need = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value after " + flag);
            return argv[++i];
        };

        if (a == "--training-case") {
            training_case_dir = need(a);
            continue;
        }
        if (a == "--tol") {
            parity_tol = std::stod(need(a));
            continue;
        }

        if (a == "--case-dir") { case_dir = need(a); continue; }
        if (a == "--data-index") { data_index = parse_data_index(need(a)); continue; }
        if (a == "--model") { model = parse_model(need(a)); continue; }
        if (a == "--decoder-prefix") { decoder_prefix = need(a); continue; }
        if (a == "--episodes") { episodes = std::stoi(need(a)); continue; }
        if (a == "--hidden") { H = std::stoi(need(a)); continue; }
        if (a == "--priori-m") { priori_m = std::stod(need(a)); continue; }
        if (a == "--priori-n") { priori_n = std::stod(need(a)); continue; }
        if (a == "--epsilon") { epsilon = std::stod(need(a)); continue; }
        if (a == "--max-episode") { max_episode_for_schedule = std::stoi(need(a)); continue; }
        if (a == "--discount-factor") { discount_factor = std::stod(need(a)); continue; }
        if (a == "--discount-length") { discount_length = std::stoi(need(a)); continue; }
        if (a == "--shuffle-train-trials") {
            const std::string v = need(a);
            if (v == "1" || v == "true" || v == "True") shuffle_train_trials = true;
            else if (v == "0" || v == "false" || v == "False") shuffle_train_trials = false;
            else throw std::runtime_error("Invalid --shuffle-train-trials (expected 0/1/true/false): " + v);
            continue;
        }
        if (a == "--seed") { seed = static_cast<unsigned int>(std::stoul(need(a))); continue; }

        throw std::runtime_error("Unknown arg: " + a);
    }

    if (!training_case_dir.empty()) {
        if (!fs::is_directory(training_case_dir)) {
            throw std::runtime_error("--training-case is not a directory: " + training_case_dir);
        }
        return rlpp::run_phase3_training_case_parity(training_case_dir, parity_tol);
    }

    if (!fs::is_directory(case_dir)) {
        throw std::runtime_error("case_dir is not a directory: " + case_dir);
    }

    // ---- Load dataset CSVs (same shape as Python export_loader_emulator_reference) ----
    const Eigen::MatrixXd inputEnsemble = load_csv_matrix(case_dir + "/inputEnsemble.csv");
    const Eigen::MatrixXd M1_truth = load_csv_matrix(case_dir + "/M1_truth.csv");
    const Eigen::VectorXi Actions = [&] {
        Eigen::MatrixXd a = load_csv_matrix(case_dir + "/Actions.csv");
        Eigen::VectorXi out;
        if (a.cols() == 1) {
            out.resize(a.rows());
            for (int r = 0; r < a.rows(); ++r) out(r) = static_cast<int>(std::llround(a(r, 0)));
            return out;
        }
        if (a.rows() == 1) {
            out.resize(a.cols());
            for (int c = 0; c < a.cols(); ++c) out(c) = static_cast<int>(std::llround(a(0, c)));
            return out;
        }
        throw std::runtime_error("Actions.csv must be vector");
    }();
    const Eigen::VectorXi Trials = [&] {
        Eigen::MatrixXd a = load_csv_matrix(case_dir + "/Trials.csv");
        Eigen::VectorXi out;
        if (a.cols() == 1) {
            out.resize(a.rows());
            for (int r = 0; r < a.rows(); ++r) out(r) = static_cast<int>(std::llround(a(r, 0)));
            return out;
        }
        if (a.rows() == 1) {
            out.resize(a.cols());
            for (int c = 0; c < a.cols(); ++c) out(c) = static_cast<int>(std::llround(a(0, c)));
            return out;
        }
        throw std::runtime_error("Trials.csv must be vector");
    }();

    // TrainTrials and meta are optional; if missing, we run on "all" indices.
    rlpp::DataLoaderOpt opt;
    opt.mode = rlpp::DataLoaderMode::Train;
    opt.shuffle_when_cursor_is_one = shuffle_train_trials;
    opt.data_loader_cursor = 1;
    opt.batch_size = std::min(3, static_cast<int>(Actions.size()));
    opt.discount_length = discount_length;

    if (fs::is_regular_file(case_dir + "/opt_trainTrials.csv") && fs::is_regular_file(case_dir + "/opt_meta.csv")) {
        const Eigen::VectorXi train_trials = [&] {
            Eigen::MatrixXd a = load_csv_matrix(case_dir + "/opt_trainTrials.csv");
            Eigen::VectorXi out;
            if (a.cols() == 1) {
                out.resize(a.rows());
                for (int r = 0; r < a.rows(); ++r) out(r) = static_cast<int>(std::llround(a(r, 0)));
                return out;
            }
            if (a.rows() == 1) {
                out.resize(a.cols());
                for (int c = 0; c < a.cols(); ++c) out(c) = static_cast<int>(std::llround(a(0, c)));
                return out;
            }
            throw std::runtime_error("opt_trainTrials.csv must be vector");
        }();
        const Eigen::MatrixXd meta = load_csv_matrix(case_dir + "/opt_meta.csv");
        if (meta.rows() < 1 || meta.cols() < 4) {
            throw std::runtime_error("opt_meta.csv must be 1x4: cursor,batchSize,discountLength,NumberOfTrainTrials");
        }
        opt.data_loader_cursor = static_cast<int>(std::llround(meta(0, 0)));
        opt.batch_size = static_cast<int>(std::llround(meta(0, 1)));
        opt.discount_length = static_cast<int>(std::llround(meta(0, 2)));
        opt.number_of_train_trials = static_cast<int>(std::llround(meta(0, 3)));
        opt.train_trials.resize(train_trials.size());
        for (int i = 0; i < train_trials.size(); ++i) opt.train_trials[static_cast<std::size_t>(i)] = train_trials(i);
    } else {
        // Infer K = max trial label in Trials (expects 1..K like Python); use sequential trainTrials 1..K.
        int K = 0;
        for (int i = 0; i < Trials.size(); ++i) {
            if (Trials(i) > K) K = Trials(i);
        }
        if (K <= 0) {
            throw std::runtime_error(
                "Missing opt_trainTrials.csv/opt_meta.csv and could not infer K from Trials.csv (max trial id <= 0)"
            );
        }
        opt.mode = rlpp::DataLoaderMode::Train;
        opt.number_of_train_trials = K;
        opt.train_trials.resize(static_cast<std::size_t>(K));
        for (int t = 0; t < K; ++t) opt.train_trials[static_cast<std::size_t>(t)] = t + 1;
        opt.data_loader_cursor = 1;
        opt.batch_size = std::min(opt.batch_size, K);
    }

    std::mt19937 rng(seed);

    // Determine Ny/his + decoder params (only for Real mode trained decoders).
    rlpp::DecodingNnParams dec;
    NyHis nh;
    if (data_index == DataIndex::Real && model != MotorModel::Manual) {
        dec = rlpp::load_decoding_nn_params_csv(decoder_prefix);
        nh = choose_ny_his_for_input_dim(static_cast<int>(dec.xoffset.size()));
    } else {
        // Minimal default for manual/simulation
        nh = NyHis{8, 2};
    }

    const int Ny = nh.Ny;
    const int his = nh.his;

    std::cout << "Phase3_Run config:\n"
              << "  case_dir=" << case_dir << "\n"
              << "  data_index=" << to_string(data_index) << "\n"
              << "  model=" << to_string(model) << "\n"
              << "  Ny=" << Ny << " his=" << his << "\n";
    if (data_index == DataIndex::Real && model != MotorModel::Manual) {
        std::cout << "  decoder_prefix=" << decoder_prefix
                  << " input_dim=" << dec.xoffset.size()
                  << " hidden=" << dec.IW1_1.rows()
                  << " out=" << dec.LW2_1.rows()
                  << "\n";
        if ((his + 1) * Ny != dec.xoffset.size()) {
            throw std::runtime_error("Ny/his mismatch: (his+1)*Ny must equal decoder input_dim");
        }
    }

    // ---- Initialize generator weights (same distribution as MATLAB/Python: U[-1,1]) ----
    const int feat = static_cast<int>(inputEnsemble.rows());
    const int Nx = feat + 1; // bias added like Python inputUnit = vstack([batchInput, ones])
    std::uniform_real_distribution<double> unif(-1.0, 1.0);

    Eigen::MatrixXd W1(H, Nx);
    for (int r = 0; r < W1.rows(); ++r)
        for (int c = 0; c < W1.cols(); ++c) W1(r, c) = unif(rng);

    Eigen::MatrixXd W2(Ny, H + 1);
    for (int r = 0; r < W2.rows(); ++r)
        for (int c = 0; c < W2.cols(); ++c) W2(r, c) = unif(rng);

    const Eigen::VectorXi indexes = shuffled_indexes_1based(Ny, rng);

    // ---- Episode loop ----
    for (int ep = 0; ep < episodes; ++ep) {
        // Batch selection (M1_truth must span the same time axis as inputEnsemble / Actions / Trials)
        rlpp::DataLoaderBatch batch =
            rlpp::dataloader_forward_exact(inputEnsemble, M1_truth, Actions, Trials, opt, rng);

        const int NumOfSamples = static_cast<int>(batch.batch_input.cols());
        if (NumOfSamples <= 0) throw std::runtime_error("Empty batch");

        Eigen::MatrixXd inputUnit(Nx, NumOfSamples);
        inputUnit.topRows(feat) = batch.batch_input;
        inputUnit.row(Nx - 1).setOnes();

        // Forward (generator) -> predicted spikes
        rlpp::ApplyNetsPrioriBatch fwd = rlpp::applynets_priori_forward(
            inputUnit, W1, W2, ep, priori_m, priori_n, rng
        );

        // Emulator -> motor + success
        rlpp::EmulatorResult emu;
        if (data_index == DataIndex::Simulations) {
            emu = rlpp::emulator_simu_exact(fwd.spk_out, batch.batch_actions, indexes, his, "decodingModel_simulation");
        } else {
            if (model == MotorModel::Manual) {
                emu = rlpp::emulator_real_manual_exact(fwd.spk_out, batch.batch_actions, indexes, his);
            } else {
                emu = rlpp::emulator_real_trained_nn_exact(fwd.spk_out, batch.batch_actions, indexes, his, decoder_prefix);
            }
        }

        // Smoothed reward (includes innerReward + epsilon schedule + normalization)
        Eigen::RowVectorXd smoothed = rlpp::compute_smoothed_reward_rlpp(
            emu.success, emu.motor_perform, epsilon, ep, max_episode_for_schedule, discount_factor, discount_length
        );

        // Gradients + update
        Eigen::MatrixXd dW2, dW1;
        rlpp::getgradient_rl_broadcast(
            smoothed, fwd.p_output, fwd.spk_out, fwd.hidden_unit, inputUnit, W2, NumOfSamples, dW2, dW1
        );

        const double lr = (data_index == DataIndex::Simulations)
                            ? rlpp::learning_rate_rl_simulations(ep, max_episode_for_schedule)
                            : rlpp::learning_rate_rl_real(ep, max_episode_for_schedule);

        W2 = W2 + lr * dW2;
        W1 = W1 + lr * dW1;

        std::cout << "[ep " << ep << "] "
                  << "sucRate=" << emu.rate
                  << " lr=" << lr
                  << " W1_norm=" << W1.norm()
                  << " W2_norm=" << W2.norm()
                  << "\n";
    }

    std::cout << "Phase3_Run done.\n";
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run_phase3_main(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "RLPP_Phase_3_Run error: " << e.what() << "\n";
        return 1;
    }
}

