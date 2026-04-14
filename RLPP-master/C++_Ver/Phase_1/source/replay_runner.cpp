#include "replay_runner.hpp"
#include "csv_utils.hpp"
#include "RLPP_inference.hpp"
#include "rlpp_hardware_output.hpp"
#include "two_layer_mlp.hpp"
#include "math_functions.hpp"

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <vector>

static void ensure_dir_exists_or_throw(const std::string& dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(fs::path(dir), ec);
    if (ec) {
        throw std::runtime_error("Could not create out_dir: " + dir + " (" + ec.message() + ")");
    }
}

static Eigen::VectorXd vectorxi_to_vectord(const Eigen::VectorXi& v) {
    Eigen::VectorXd out(v.size());
    for (int i = 0; i < v.size(); ++i) {
        out(i) = static_cast<double>(v(i));
    }
    return out;
}

static Eigen::VectorXd int64vec_to_vectord(const std::vector<long long>& v) {
    Eigen::VectorXd out(static_cast<int>(v.size()));
    for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        out(i) = static_cast<double>(v[i]);
    }
    return out;
}

static void print_latency_summary_us(const std::vector<long long>& us) {
    if (us.empty()) {
        std::cout << "Replay latency: no samples\n";
        return;
    }
    std::vector<long long> s = us;
    std::sort(s.begin(), s.end());

    auto pct = [&](double p) -> long long {
        const double idx = p * (s.size() - 1);
        const std::size_t i = static_cast<std::size_t>(idx);
        return s[i];
    };

    const long long minv = s.front();
    const long long maxv = s.back();
    const double mean = std::accumulate(s.begin(), s.end(), 0.0) / static_cast<double>(s.size());

    std::cout << "Replay latency (us):"
              << " n=" << s.size()
              << " min=" << minv
              << " mean=" << mean
              << " p50=" << pct(0.50)
              << " p95=" << pct(0.95)
              << " p99=" << pct(0.99)
              << " max=" << maxv
              << "\n";
}

static void print_spike_rate_calibration(
    const Eigen::MatrixXd& probs_Ny_by_T,
    const Eigen::MatrixXd& spikes_Ny_by_T
) {
    if (probs_Ny_by_T.size() == 0 || spikes_Ny_by_T.size() == 0) {
        std::cout << "Spike-rate calibration: missing probs/spikes dump\n";
        return;
    }
    if (probs_Ny_by_T.rows() != spikes_Ny_by_T.rows() ||
        probs_Ny_by_T.cols() != spikes_Ny_by_T.cols()) {
        std::cout << "Spike-rate calibration: shape mismatch\n";
        return;
    }

    Eigen::VectorXd mean_p = probs_Ny_by_T.rowwise().mean();
    Eigen::VectorXd mean_s = spikes_Ny_by_T.rowwise().mean();
    Eigen::VectorXd gap = (mean_s - mean_p).cwiseAbs();

    std::cout << "Spike-rate calibration:\n";
    std::cout << "  mean(|mean(spike)-mean(prob)|) = " << gap.mean() << "\n";
    std::cout << "  max (|mean(spike)-mean(prob)|) = " << gap.maxCoeff() << "\n";
}

static void print_confusion_matrix_1based_3class(
    const Eigen::VectorXi& pred_labels,
    const Eigen::VectorXi& ref_labels
) {
    long long cm[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    const int n = std::min(pred_labels.size(), ref_labels.size());
    for (int i = 0; i < n; ++i) {
        const int a = pred_labels(i);
        const int b = ref_labels(i);
        if (a >= 1 && a <= 3 && b >= 1 && b <= 3) {
            cm[a-1][b-1] += 1;
        }
    }
    std::cout << "Confusion matrix (pred rows, ref cols):\n";
    std::cout << cm[0][0] << " " << cm[0][1] << " " << cm[0][2] << "\n";
    std::cout << cm[1][0] << " " << cm[1][1] << " " << cm[1][2] << "\n";
    std::cout << cm[2][0] << " " << cm[2][1] << " " << cm[2][2] << "\n";
}

ReplayResult run_replay_or_throw(
    const F1Bundle& bundle,
    const Eigen::VectorXd& xoffset,
    const Eigen::VectorXd& gain,
    double ymin,
    const ReplayConfig& cfg
) {
    const int T  = static_cast<int>(bundle.upstream_spikes.rows());
    const int Nx = static_cast<int>(bundle.upstream_spikes.cols());
    const int feature_dim = static_cast<int>(bundle.encoder_features_ref.rows());
    if (feature_dim % Nx != 0) throw std::runtime_error("feature_dim not divisible by Nx");
    const int H = feature_dim / Nx;

    const int Ny = TwoLayerMLP(bundle.generator_W1, bundle.generator_W2).output_dim();

    if (bundle.decoder_features_ref.rows() % Ny != 0)
        throw std::runtime_error("decoder_features_ref rows not divisible by Ny");
    const int num_lags = static_cast<int>(bundle.decoder_features_ref.rows() / Ny);

    const int T_valid = static_cast<int>(bundle.decoder_logits_ref.cols());
    const int num_labels = static_cast<int>(bundle.decoder_logits_ref.rows());

    if (cfg.valid_col_start < 0 || cfg.valid_col_start > T_valid) {
        throw std::runtime_error("valid_col_start out of range");
    }

    const int remaining = T_valid - cfg.valid_col_start;
    const int out_cols = (cfg.valid_col_count < 0) ? remaining
                                                   : std::min(cfg.valid_col_count, remaining);
    if (out_cols < 0) {
        throw std::runtime_error("valid_col_count out of range");
    }

    RLPPInferenceConfig icfg;
    icfg.tau_bins = cfg.tau_bins;
    icfg.rng_seed = cfg.rng_seed;
    icfg.spike_mode = SpikeDriveMode::SampledBernoulli;

    RLPPInference inf(
        Nx, H, Ny, num_lags,
        bundle.sorted_indices_1based,
        bundle.generator_W1, bundle.generator_W2,
        xoffset, gain, ymin,
        bundle.decoder_W1, bundle.decoder_b1,
        bundle.decoder_W2, bundle.decoder_b2,
        icfg
    );

    ReplayResult out;
    out.decoder_logits.resize(num_labels, out_cols);
    out.labels.resize(out_cols);

    Eigen::MatrixXd probs_dump;
    Eigen::MatrixXd spikes_dump;
    Eigen::MatrixXd feats_dump;

    if (cfg.dump_generator_probs) {
        probs_dump.resize(Ny, out_cols);
    }
    if (cfg.dump_downstream_spikes) {
        spikes_dump.resize(Ny, out_cols);
    }
    if (cfg.dump_decoder_features) {
        feats_dump.resize(Ny * num_lags, out_cols);
    }

    std::vector<long long> latency_us;
    if (cfg.dump_latency_us) {
        latency_us.reserve(out_cols);
    }

    if (cfg.dump_decoder_logits || cfg.dump_labels || cfg.dump_latency_us ||
        cfg.dump_generator_probs || cfg.dump_downstream_spikes || cfg.dump_decoder_features ||
        cfg.dump_hardware_trace_v1) {
        ensure_dir_exists_or_throw(cfg.out_dir);
    }

    std::unique_ptr<std::ofstream> hardware_trace_out;
    if (cfg.dump_hardware_trace_v1) {
        hardware_trace_out = std::make_unique<std::ofstream>(
            cfg.out_dir + "/hardware_trace_v1.bin",
            std::ios::binary | std::ios::trunc
        );
        if (!*hardware_trace_out) {
            throw std::runtime_error("replay: cannot open hardware_trace_v1.bin for write");
        }
    }

    int col = 0;
    int rec_col = 0;
    for (int t_idx = 0; t_idx < T; ++t_idx) {
        const int t = t_idx + 1;

        Eigen::VectorXd u_t = bundle.upstream_spikes.row(t_idx).transpose();

        const bool record = (col >= cfg.valid_col_start) && (rec_col < out_cols);
        const bool time_this_bin = record && cfg.dump_latency_us;

        using clock = std::chrono::high_resolution_clock;
        clock::time_point t0{};
        clock::time_point t1{};

        RLPPInferenceStepOutput o;
        if (time_this_bin) {
            t0 = clock::now();
        }

        if (cfg.mode == ReplayMode::DeterministicFromBundleSpikes) {
            // Column is read inside RLPPInference only after observe + can_encode (ordering-safe).
            o = inf.step_with_downstream_spikes(u_t, bundle.downstream_spikes_ref, col, t);
        } else {
            o = inf.step(u_t, t);
        }

        if (time_this_bin) {
            t1 = clock::now();
        }

        if (!o.valid) {
            continue;
        }
        if (col >= T_valid) {
            throw std::runtime_error("replay produced too many valid cols");
        }

        if (time_this_bin) {
            latency_us.push_back(
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
            );
        }

        if (record) {
            out.decoder_logits.col(rec_col) = o.decoder_y;
            out.labels(rec_col) = o.label_1based;

            if (cfg.dump_generator_probs) {
                probs_dump.col(rec_col) = o.gen_prob;
            }
            if (cfg.dump_downstream_spikes) {
                Eigen::VectorXd s_dump = o.gen_spikes;
                if (bundle.sorted_indices_1based.size() > 0) {
                    s_dump = apply_sorted_indices_1based(s_dump, bundle.sorted_indices_1based);
                }
                spikes_dump.col(rec_col) = s_dump;
            }
            if (cfg.dump_decoder_features) {
                feats_dump.col(rec_col) = o.decoder_ensemble;
            }
            if (hardware_trace_out) {
                std::vector<std::uint8_t> packed;
                pack_rlpp_hardware_frame_v1(o, t, static_cast<std::uint64_t>(rec_col), packed);
                hardware_trace_out->write(
                    reinterpret_cast<const char*>(packed.data()),
                    static_cast<std::streamsize>(packed.size())
                );
                if (!*hardware_trace_out) {
                    throw std::runtime_error("replay: write failed on hardware_trace_v1.bin");
                }
            }

            ++rec_col;
            if (rec_col >= out_cols) {
                break;
            }
        }

        ++col;
    }

    if (rec_col != out_cols) {
        throw std::runtime_error("replay produced " + std::to_string(rec_col) +
                                 " recorded cols, expected " + std::to_string(out_cols));
    }

    Eigen::MatrixXd ref_logits = bundle.decoder_logits_ref.block(
        0, cfg.valid_col_start, num_labels, out_cols);
    Eigen::MatrixXd diff = (out.decoder_logits - ref_logits).cwiseAbs();
    out.max_abs_err = diff.maxCoeff();
    out.mean_abs_err = diff.mean();

    int mism = 0;
    for (int i = 0; i < out.labels.size(); ++i) {
        if (out.labels(i) != bundle.labels_ref(cfg.valid_col_start + i)) mism++;
    }
    out.label_mismatches = mism;

    if (cfg.dump_generator_probs && cfg.dump_downstream_spikes) {
        print_spike_rate_calibration(probs_dump, spikes_dump);
    }
    {
        Eigen::VectorXi ref_labels_window(out_cols);
        for (int i = 0; i < out_cols; ++i) {
            ref_labels_window(i) = bundle.labels_ref(cfg.valid_col_start + i);
        }
        print_confusion_matrix_1based_3class(out.labels, ref_labels_window);
    }

    if (cfg.dump_decoder_logits) {
        save_csv_matrix(cfg.out_dir + "/decoder_logits_replay.csv", out.decoder_logits);
    }
    if (cfg.dump_labels) {
        save_csv_vector(cfg.out_dir + "/labels_replay.csv", vectorxi_to_vectord(out.labels));
    }
    if (cfg.dump_latency_us) {
        save_csv_vector(cfg.out_dir + "/latency_us_replay.csv", int64vec_to_vectord(latency_us));
        print_latency_summary_us(latency_us);
    }
    if (cfg.dump_generator_probs) {
        save_csv_matrix(cfg.out_dir + "/generator_probs_replay.csv", probs_dump);
    }
    if (cfg.dump_downstream_spikes) {
        save_csv_matrix(cfg.out_dir + "/downstream_spikes_replay.csv", spikes_dump);
    }
    if (cfg.dump_decoder_features) {
        save_csv_matrix(cfg.out_dir + "/decoder_features_replay.csv", feats_dump);
    }

    return out;
}
