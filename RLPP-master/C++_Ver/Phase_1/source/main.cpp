#include <Eigen/Dense>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "csv_utils.hpp"
#include "F1_rat01_bundle.hpp"
#include "F1_rat01_validate.hpp"
#include "F1_rat01_parity.hpp"
#include "replay_runner.hpp"
#include "RLPP_inference.hpp"
#include "rlpp_hardware_output.hpp"
#include "rlpp_phase1_cli.hpp"

namespace fs = std::filesystem;

static std::string bundle_file(const std::string& bundle_dir, const char* name) {
    return (fs::path(bundle_dir) / name).string();
}

static void load_decoder_mapminmax(
    const std::string& bundle_dir,
    Eigen::VectorXd& xoffset,
    Eigen::VectorXd& gain,
    double& ymin
) {
    Eigen::MatrixXd xoffset_m = load_csv_matrix(bundle_file(bundle_dir, "decoder_xoffset.csv"));
    Eigen::MatrixXd gain_m = load_csv_matrix(bundle_file(bundle_dir, "decoder_gain.csv"));
    Eigen::MatrixXd ymin_m = load_csv_matrix(bundle_file(bundle_dir, "decoder_ymin.csv"));

    auto to_vector = [](const Eigen::MatrixXd& M) -> Eigen::VectorXd {
        if (M.cols() == 1) return M.col(0);
        if (M.rows() == 1) return M.row(0).transpose();
        throw std::runtime_error("Expected a vector CSV (Nx1 or 1xN).");
    };

    xoffset = to_vector(xoffset_m);
    gain = to_vector(gain_m);
    if (ymin_m.size() != 1) {
        throw std::runtime_error("decoder_ymin.csv must be a single value.");
    }
    ymin = ymin_m(0, 0);
}

struct OnlineDetMetrics {
    double max_abs_err = 0.0;
    double mean_abs_err = 0.0;
    int label_mismatches = 0;
    int n_labels = 0;
};

static void run_parity_chain(
    const F1Bundle& bundle,
    const Eigen::VectorXd& xoffset,
    const Eigen::VectorXd& gain,
    double ymin,
    double tau_bins
) {
    run_encoder_parity_or_throw(bundle, tau_bins);
    std::cout << "F1 encoder parity PASSED.\n";

    run_generator_parity_or_throw(bundle);
    std::cout << "F1 generator parity PASSED.\n";

    run_decoder_feature_parity_or_throw(bundle);
    std::cout << "F1 decoder feature parity PASSED.\n";

    run_decoder_output_model01_parity_or_throw(bundle, xoffset, gain, ymin);
    std::cout << "F1 decoder output model01 parity PASSED.\n";
}

static OnlineDetMetrics run_online_inference_deterministic_test(
    const F1Bundle& bundle,
    const Eigen::VectorXd& xoffset,
    const Eigen::VectorXd& gain,
    double ymin,
    double tau_bins
) {
    const int T = static_cast<int>(bundle.upstream_spikes.rows());
    const int Nx = static_cast<int>(bundle.upstream_spikes.cols());
    const int feature_dim = static_cast<int>(bundle.encoder_features_ref.rows());
    if (feature_dim % Nx != 0) {
        throw std::runtime_error("online test: feature_dim not divisible by Nx");
    }
    const int H = feature_dim / Nx;
    const int Ny = static_cast<int>(bundle.downstream_spikes_ref.rows());
    if (bundle.decoder_features_ref.rows() % Ny != 0) {
        throw std::runtime_error("online test: decoder_features_ref rows not divisible by Ny");
    }
    const int num_lags = static_cast<int>(bundle.decoder_features_ref.rows() / Ny);
    const int T_valid = static_cast<int>(bundle.decoder_logits_ref.cols());
    const int num_labels = static_cast<int>(bundle.decoder_logits_ref.rows());

    RLPPInferenceConfig icfg;
    icfg.tau_bins = tau_bins;
    icfg.spike_mode = SpikeDriveMode::SampledBernoulli;
    icfg.rng_seed = 0;

    RLPPInference inf(
        Nx, H, Ny, num_lags,
        bundle.sorted_indices_1based,
        bundle.generator_W1, bundle.generator_W2,
        xoffset, gain, ymin,
        bundle.decoder_W1, bundle.decoder_b1,
        bundle.decoder_W2, bundle.decoder_b2,
        icfg
    );

    Eigen::MatrixXd Y_online(num_labels, T_valid);
    Eigen::VectorXi labels_online(T_valid);

    int col = 0;
    for (int t_idx = 0; t_idx < T && col < T_valid; ++t_idx) {
        const int t = t_idx + 1;
        Eigen::VectorXd u_t = bundle.upstream_spikes.row(t_idx).transpose();
        Eigen::VectorXd s_t = bundle.downstream_spikes_ref.col(col);

        RLPPInferenceStepOutput o = inf.step_with_downstream_spikes(u_t, s_t, t);
        if (!o.valid) {
            continue;
        }

        Y_online.col(col) = o.decoder_y;
        labels_online(col) = o.label_1based;
        ++col;
    }

    if (col != T_valid) {
        throw std::runtime_error("online test: produced " + std::to_string(col) +
                                 " valid cols, expected " + std::to_string(T_valid));
    }

    const double max_abs = (Y_online - bundle.decoder_logits_ref).cwiseAbs().maxCoeff();
    const double mean_abs = (Y_online - bundle.decoder_logits_ref).cwiseAbs().mean();
    int mism = 0;
    for (int i = 0; i < labels_online.size(); ++i) {
        if (labels_online(i) != bundle.labels_ref(i)) mism++;
    }

    std::cout << "Online step() deterministic test: max_abs_err=" << max_abs
              << " mean_abs_err=" << mean_abs << "\n";
    std::cout << "Online step() deterministic labels mismatches: " << mism
              << " / " << labels_online.size() << "\n";

    OnlineDetMetrics m;
    m.max_abs_err = max_abs;
    m.mean_abs_err = mean_abs;
    m.label_mismatches = mism;
    m.n_labels = static_cast<int>(labels_online.size());
    return m;
}

static void write_sample_hardware_frames_v1(
    const F1Bundle& bundle,
    const Eigen::VectorXd& xoffset,
    const Eigen::VectorXd& gain,
    double ymin,
    double tau_bins,
    const std::string& dir,
    int max_frames
) {
    namespace fs = std::filesystem;
    fs::create_directories(fs::path(dir));

    const int T = static_cast<int>(bundle.upstream_spikes.rows());
    const int Nx = static_cast<int>(bundle.upstream_spikes.cols());
    const int feature_dim = static_cast<int>(bundle.encoder_features_ref.rows());
    if (feature_dim % Nx != 0) {
        throw std::runtime_error("sample frames: feature_dim not divisible by Nx");
    }
    const int H = feature_dim / Nx;
    const int Ny = static_cast<int>(bundle.downstream_spikes_ref.rows());
    if (bundle.decoder_features_ref.rows() % Ny != 0) {
        throw std::runtime_error("sample frames: decoder_features_ref rows not divisible by Ny");
    }
    const int num_lags = static_cast<int>(bundle.decoder_features_ref.rows() / Ny);
    const int T_valid = static_cast<int>(bundle.decoder_logits_ref.cols());

    RLPPInferenceConfig icfg;
    icfg.tau_bins = tau_bins;
    icfg.spike_mode = SpikeDriveMode::SampledBernoulli;
    icfg.rng_seed = 0;

    RLPPInference inf(
        Nx, H, Ny, num_lags,
        bundle.sorted_indices_1based,
        bundle.generator_W1, bundle.generator_W2,
        xoffset, gain, ymin,
        bundle.decoder_W1, bundle.decoder_b1,
        bundle.decoder_W2, bundle.decoder_b2,
        icfg
    );

    int col = 0;
    int written = 0;
    std::uint64_t seq = 0;
    for (int t_idx = 0; t_idx < T && col < T_valid; ++t_idx) {
        const int t = t_idx + 1;
        Eigen::VectorXd u_t = bundle.upstream_spikes.row(t_idx).transpose();
        Eigen::VectorXd s_t = bundle.downstream_spikes_ref.col(col);

        RLPPInferenceStepOutput o = inf.step_with_downstream_spikes(u_t, s_t, t);
        if (!o.valid) {
            continue;
        }

        if (written < max_frames) {
            std::vector<std::uint8_t> packed;
            pack_rlpp_hardware_frame_v1(o, t, seq, packed);
            ++seq;
            const std::string fn =
                dir + "/sample_hardware_frame_v1_" + std::to_string(written) + ".bin";
            std::ofstream bf(fn, std::ios::binary);
            if (!bf) {
                throw std::runtime_error("sample frames: cannot write " + fn);
            }
            bf.write(reinterpret_cast<const char*>(packed.data()),
                     static_cast<std::streamsize>(packed.size()));
            ++written;
        }
        ++col;
    }
}

static void run_phase2_5_hw_prep(
    const F1Bundle& bundle,
    const Eigen::VectorXd& xoffset,
    const Eigen::VectorXd& gain,
    double ymin,
    const RlppPhase1CliConfig& cfg
) {
    namespace fs = std::filesystem;
    const fs::path out_root = fs::path(cfg.replay_out_base) / "phase2_5";
    fs::create_directories(out_root);

    const int num_labels = static_cast<int>(bundle.decoder_logits_ref.rows());

    std::cout << "\n=== Phase 2.5 hardware prep (hw-prep) ===\n"
              << "Output directory: " << out_root.string() << "\n";

    write_rlpp_hardware_io_contract_text((out_root / "hardware_io_contract_v1.txt").string(), num_labels);
    std::cout << "Wrote hardware_io_contract_v1.txt\n";

    run_parity_chain(bundle, xoffset, gain, ymin, cfg.tau_bins);
    OnlineDetMetrics online =
        run_online_inference_deterministic_test(bundle, xoffset, gain, ymin, cfg.tau_bins);

    ReplayConfig rc;
    rc.out_dir = (out_root / "replay_deterministic").string();
    rc.dump_latency_us = true;
    rc.dump_generator_probs = false;
    rc.dump_downstream_spikes = false;
    rc.dump_decoder_features = false;
    rc.mode = ReplayMode::DeterministicFromBundleSpikes;
    rc.rng_seed = 0;
    rc.tau_bins = cfg.tau_bins;
    rc.valid_col_start = cfg.valid_col_start;
    rc.valid_col_count = cfg.valid_col_count;
    rc.dump_hardware_trace_v1 = true;

    ReplayResult replay = run_replay_or_throw(bundle, xoffset, gain, ymin, rc);
    std::cout << "[phase2.5 det replay] max_abs_err=" << replay.max_abs_err
              << " mean_abs_err=" << replay.mean_abs_err << " label_mismatches="
              << replay.label_mismatches << "/" << replay.labels.size() << "\n";
    std::cout << "Wrote hardware_trace_v1.bin (full replay window) under replay_deterministic/\n";

    write_sample_hardware_frames_v1(
        bundle, xoffset, gain, ymin, cfg.tau_bins, (out_root / "sample_frames").string(), 3);
    std::cout << "Wrote sample binary frames under sample_frames/\n";

    const double logit_tol = 1e-8;
    const bool online_ok =
        online.label_mismatches == 0 && online.max_abs_err <= logit_tol;
    const bool replay_ok =
        replay.label_mismatches == 0 && replay.max_abs_err <= logit_tol;
    const bool pass = online_ok && replay_ok;

    {
        std::ofstream rep(out_root / "phase2_5_report.txt");
        if (!rep) {
            throw std::runtime_error("phase2_5: cannot write phase2_5_report.txt");
        }
        rep << "RLPP Phase 2.5 sign-off report\n";
        rep << "build_stamp: " << __DATE__ << " " << __TIME__ << "\n";
        rep << "bundle_dir: " << cfg.bundle_dir << "\n";
        rep << "tau_bins: " << cfg.tau_bins << "\n";
        rep << "valid_col_start: " << cfg.valid_col_start << "\n";
        rep << "valid_col_count: " << cfg.valid_col_count << "\n";
        rep << "num_labels_K: " << num_labels << "\n";
        rep << "hardware_frame_v1_bytes: " << (32 + 8 * num_labels) << "\n";
        rep << "hardware_trace_v1: replay_deterministic/hardware_trace_v1.bin (concatenated frames)\n";
        rep << "\nparity: PASS (four checks: encoder, generator, decoder features, decoder output)\n";
        rep << "\nonline_deterministic:\n";
        rep << "  max_abs_err: " << online.max_abs_err << "\n";
        rep << "  mean_abs_err: " << online.mean_abs_err << "\n";
        rep << "  label_mismatches: " << online.label_mismatches << " / " << online.n_labels << "\n";
        rep << "\nreplay_deterministic:\n";
        rep << "  max_abs_err: " << replay.max_abs_err << "\n";
        rep << "  mean_abs_err: " << replay.mean_abs_err << "\n";
        rep << "  label_mismatches: " << replay.label_mismatches << " / " << replay.labels.size()
            << "\n";
        rep << "\ncriteria: label_mismatches==0 and max_abs_err<=" << logit_tol
            << " (online and replay)\n";
        rep << "\nOVERALL_STATUS: " << (pass ? "PASS" : "FAIL") << "\n";
    }

    std::cout << "\nPhase 2.5 OVERALL_STATUS: " << (pass ? "PASS" : "FAIL") << "\n";
    std::cout << "Report: " << (out_root / "phase2_5_report.txt").string() << "\n";

    if (!pass) {
        throw std::runtime_error("Phase 2.5 sign-off FAILED (see phase2_5_report.txt)");
    }
}

static void run_replay_deterministic(
    const F1Bundle& bundle,
    const Eigen::VectorXd& xoffset,
    const Eigen::VectorXd& gain,
    double ymin,
    const RlppPhase1CliConfig& cfg
) {
    ReplayConfig rc;
    rc.out_dir = (fs::path(cfg.replay_out_base) / "deterministic").string();
    rc.dump_latency_us = true;
    rc.dump_generator_probs = false;
    rc.dump_downstream_spikes = false;
    rc.dump_decoder_features = false;
    rc.mode = ReplayMode::DeterministicFromBundleSpikes;
    rc.rng_seed = 0;
    rc.tau_bins = cfg.tau_bins;
    rc.valid_col_start = cfg.valid_col_start;
    rc.valid_col_count = cfg.valid_col_count;
    rc.dump_hardware_trace_v1 = cfg.dump_hardware_trace_v1;

    ReplayResult r = run_replay_or_throw(bundle, xoffset, gain, ymin, rc);
    std::cout << "[det] Replay: max_abs_err=" << r.max_abs_err
              << " mean_abs_err=" << r.mean_abs_err << "\n";
    std::cout << "[det] Replay: label mismatches=" << r.label_mismatches
              << " / " << r.labels.size() << "\n";
}

static void run_replay_sampled_sweep(
    const F1Bundle& bundle,
    const Eigen::VectorXd& xoffset,
    const Eigen::VectorXd& gain,
    double ymin,
    const RlppPhase1CliConfig& cfg
) {
    for (int s = cfg.seed_start; s < cfg.seed_start + cfg.seed_count; ++s) {
        ReplayConfig rc;
        rc.out_dir = (fs::path(cfg.replay_out_base) / ("sampled_seed" + std::to_string(s))).string();
        rc.dump_latency_us = true;
        rc.dump_generator_probs = true;
        rc.dump_downstream_spikes = true;
        rc.dump_decoder_features = false;
        rc.mode = ReplayMode::SampledFromGeneratorProbs;
        rc.rng_seed = static_cast<unsigned int>(s);
        rc.tau_bins = cfg.tau_bins;
        rc.valid_col_start = cfg.valid_col_start;
        rc.valid_col_count = cfg.valid_col_count;
        rc.dump_hardware_trace_v1 = cfg.dump_hardware_trace_v1;

        ReplayResult r = run_replay_or_throw(bundle, xoffset, gain, ymin, rc);

        std::cout << "[samp seed=" << s << "] mismatches="
                  << r.label_mismatches << "/" << r.labels.size()
                  << " rate=" << (static_cast<double>(r.label_mismatches) / r.labels.size())
                  << "\n";
    }
}

int main(int argc, char** argv) {
    try {
        RlppPhase1CliConfig cfg = parse_rlpp_phase1_cli(argc, argv);

        Eigen::VectorXd xoffset;
        Eigen::VectorXd gain;
        double ymin = 0.0;
        load_decoder_mapminmax(cfg.bundle_dir, xoffset, gain, ymin);

        F1Bundle bundle = load_f1_bundle(cfg.bundle_dir);
        validate_f1_bundle_or_throw(bundle, /*his=*/-1);

        switch (cfg.mode) {
            case RlppPhase1Mode::All:
                run_parity_chain(bundle, xoffset, gain, ymin, cfg.tau_bins);
                run_online_inference_deterministic_test(bundle, xoffset, gain, ymin, cfg.tau_bins);
                run_replay_deterministic(bundle, xoffset, gain, ymin, cfg);
                run_replay_sampled_sweep(bundle, xoffset, gain, ymin, cfg);
                break;
            case RlppPhase1Mode::ParityOnly:
                run_parity_chain(bundle, xoffset, gain, ymin, cfg.tau_bins);
                break;
            case RlppPhase1Mode::ParityAndOnline:
                run_parity_chain(bundle, xoffset, gain, ymin, cfg.tau_bins);
                run_online_inference_deterministic_test(bundle, xoffset, gain, ymin, cfg.tau_bins);
                break;
            case RlppPhase1Mode::HwPrep:
                run_phase2_5_hw_prep(bundle, xoffset, gain, ymin, cfg);
                break;
            case RlppPhase1Mode::ReplayAll:
                run_replay_deterministic(bundle, xoffset, gain, ymin, cfg);
                run_replay_sampled_sweep(bundle, xoffset, gain, ymin, cfg);
                break;
            case RlppPhase1Mode::ReplayDeterministicOnly:
                run_replay_deterministic(bundle, xoffset, gain, ymin, cfg);
                break;
            case RlppPhase1Mode::ReplaySampledOnly:
                run_replay_sampled_sweep(bundle, xoffset, gain, ymin, cfg);
                break;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
