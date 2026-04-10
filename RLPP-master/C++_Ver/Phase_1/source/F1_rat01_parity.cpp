#include "F1_rat01_parity.hpp"
#include "two_layer_mlp.hpp"
#include "exponential_history_encoder.hpp"
#include "decoder_history_buffer.hpp"
#include "decoding_model01.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>
#include <vector>

// MATLAB: Y(i,:) = X(sortedIndices(i),:) with 1-based sortedIndices — match row order to ref build.
static Eigen::MatrixXd apply_sorted_row_order_matlab(
    const Eigen::MatrixXd& spikes_Ny_by_T,
    const Eigen::VectorXi& sorted_indices_1based
) {
    const int Ny = static_cast<int>(spikes_Ny_by_T.rows());
    const int T  = static_cast<int>(spikes_Ny_by_T.cols());
    if (sorted_indices_1based.size() != Ny) {
        throw std::runtime_error(
            "sortedIndices length must equal Ny (got " + std::to_string(sorted_indices_1based.size()) +
            ", Ny=" + std::to_string(Ny) + ")");
    }
    Eigen::MatrixXd out(Ny, T);
    for (int i = 0; i < Ny; ++i) {
        const int src = sorted_indices_1based(i) - 1; // 1-based -> 0-based
        if (src < 0 || src >= Ny) {
            throw std::runtime_error(
                "sortedIndices(" + std::to_string(i) + ")=" + std::to_string(sorted_indices_1based(i)) +
                " out of range for Ny=" + std::to_string(Ny));
        }
        out.row(i) = spikes_Ny_by_T.row(src);
    }
    return out;
}

static Eigen::MatrixXd build_decoder_features_from_spikes(
    const Eigen::MatrixXd& spikes_Ny_by_T,
    int num_lags // = his + 1
) {
    const int Ny = static_cast<int>(spikes_Ny_by_T.rows());
    const int T  = static_cast<int>(spikes_Ny_by_T.cols());
    DecoderHistoryBuffer hist(Ny, num_lags);
    Eigen::MatrixXd feats(Ny * num_lags, T);
    for (int t = 0; t < T; ++t) {
        hist.push(spikes_Ny_by_T.col(t));
        feats.col(t) = hist.flatten_for_python_decoder(); // lag0 (newest) first
    }
    return feats;
}
static double max_abs_err(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    return (A - B).cwiseAbs().maxCoeff();
}



void run_encoder_parity_or_throw(const F1Bundle& bundle) {
    const int T = static_cast<int>(bundle.upstream_spikes.rows());
    const int Nx = static_cast<int>(bundle.upstream_spikes.cols());

    const int feature_dim = static_cast<int>(bundle.encoder_features_ref.rows());
    const int T_valid = static_cast<int>(bundle.encoder_features_ref.cols());

    std::cout << "  upstream_spikes min=" << bundle.upstream_spikes.minCoeff()
              << " max=" << bundle.upstream_spikes.maxCoeff() << "\n";

    const int start_ref = T - T_valid + 1;
    std::cout << "Encoder parity debug:\n";
    std::cout << "  T=" << T << " Nx=" << Nx << " feature_dim=" << feature_dim << "\n";
    std::cout << "  T_valid(ref)=" << T_valid << " => Start_ref=" << start_ref << "\n";

    if (T <= 0 || Nx <= 0) {
        throw std::runtime_error("Encoder parity: upstream_spikes is empty");
    }
    if (feature_dim <= 0 || T_valid <= 0) {
        throw std::runtime_error("Encoder parity: encoder_features_ref is empty");
    }
    if (feature_dim % Nx != 0) {
        throw std::runtime_error("Encoder parity: feature_dim not divisible by Nx");
    }

    const int H = feature_dim / Nx;
    const double tau_bins = 150.0; // must match your MATLAB export config

    ExponentialHistoryEncoder encoder(Nx, H, tau_bins);

    Eigen::MatrixXd encoder_cpp(feature_dim, T_valid);
    int col = 0;

    // --- Debug: compute Start from upstream_spikes directly ---
    std::vector<int> t_h(Nx, -1);
    std::vector<int> count(Nx, 0);

    for (int t_idx = 0; t_idx < T; ++t_idx) {
        const int t = t_idx + 1;
        for (int i = 0; i < Nx; ++i) {
            if (bundle.upstream_spikes(t_idx, i) != 0.0) {   // MUST match your observe_bin rule
                count[i] += 1;
                if (count[i] == H && t_h[i] < 0) {
                    t_h[i] = t;
                }
            }
        }
    }

    // compute Start_from_upstream = max_i(t_h[i] + 1)
    int start_from_upstream = -1;
    int argmax_neuron = -1;
    for (int i = 0; i < Nx; ++i) {
        if (t_h[i] < 0) {
            throw std::runtime_error("Debug: neuron " + std::to_string(i) +
                                    " never reached H spikes in upstream_spikes");
        }
        const int cand = t_h[i] + 1;
        if (cand > start_from_upstream) {
            start_from_upstream = cand;
            argmax_neuron = i;
        }
    }

    std::cout << "  Start_from_upstream(nonzero rule)=" << start_from_upstream
            << " (limiting neuron=" << argmax_neuron
            << ", t_H=" << t_h[argmax_neuron] << ")\n";

    for (int t_idx = 0; t_idx < T; ++t_idx) {
        const int t = t_idx + 1; // 1-based time like MATLAB/Python

        Eigen::VectorXd spike_bin = bundle.upstream_spikes.row(t_idx).transpose();
        encoder.observe_bin(spike_bin, t);

        if (!encoder.can_encode(t)) {
            continue;
        }

        if (col >= T_valid) {
            throw std::runtime_error("Encoder parity: produced more columns than reference");
        }

        encoder_cpp.col(col) = encoder.encode(t);
        ++col;
    }

    const int start_cpp = encoder.start_time_bin();
    std::cout << "  Produced columns=" << col << "\n";
    std::cout << "  Start_cpp(encoder.start_time_bin())=" << start_cpp << "\n";

    if (col != T_valid) {
        throw std::runtime_error(
            "Encoder parity: produced " + std::to_string(col) +
            " columns but reference has " + std::to_string(T_valid));
    }

    Eigen::MatrixXd diff = (encoder_cpp - bundle.encoder_features_ref).cwiseAbs();
    const double max_abs_err = diff.maxCoeff();
    const double mean_abs_err = diff.mean();

    std::cout << "Encoder parity: max_abs_err=" << max_abs_err
              << " mean_abs_err=" << mean_abs_err << "\n";
}



void run_generator_parity_or_throw(const F1Bundle& bundle) {
    TwoLayerMLP gen(bundle.generator_W1, bundle.generator_W2);

    // X: [feature_dim x T_valid] — same layout as MATLAB/Python batchInput
    const Eigen::MatrixXd& X = bundle.encoder_features_ref;

    if (X.rows() != gen.feature_dim()) {
        throw std::runtime_error("generator parity: encoder feature rows != MLP feature_dim");
    }

    Eigen::MatrixXd probs_cpp = gen.forward_batch(X);  // [Ny x T_valid]

    if (probs_cpp.rows() != bundle.generator_probs_ref.rows() ||
        probs_cpp.cols() != bundle.generator_probs_ref.cols()) {
        throw std::runtime_error("generator parity: output shape mismatch");
    }

    Eigen::MatrixXd diff = (probs_cpp - bundle.generator_probs_ref).cwiseAbs();
    const double max_abs = diff.maxCoeff();
    const double mean_abs = diff.mean();

    std::cout << "Generator parity: max_abs_err=" << max_abs
              << " mean_abs_err=" << mean_abs << "\n";

    const double tol = 1e-9;  // adjust if needed; often tight with double
    if (max_abs > tol) {
        // Optional: find first bad column
        for (int c = 0; c < probs_cpp.cols(); ++c) {
            const double col_max = (probs_cpp.col(c) - bundle.generator_probs_ref.col(c)).cwiseAbs().maxCoeff();
            if (col_max > tol) {
                std::cout << "  first bad column c=" << c << " (0-based) col_max=" << col_max << "\n";
                break;
            }
        }
        throw std::runtime_error("generator parity: max_abs_err exceeds tolerance");
    }
}



void run_decoder_feature_parity_or_throw(const F1Bundle& bundle) {
    const int Ny = static_cast<int>(bundle.downstream_spikes_ref.rows());
    const int T  = static_cast<int>(bundle.downstream_spikes_ref.cols());
    const int dec_rows = static_cast<int>(bundle.decoder_features_ref.rows());
    if (Ny <= 0 || dec_rows % Ny != 0) {
        throw std::runtime_error("decoder feature parity: cannot infer num_lags from decoder_features_ref");
    }
    const int num_lags = dec_rows / Ny;
    if (bundle.decoder_features_ref.cols() != T) {
        throw std::runtime_error("decoder feature parity: time dimension mismatch");
    }
    Eigen::MatrixXd spikes_ordered = bundle.downstream_spikes_ref;
    if (bundle.sorted_indices_1based.size() > 0) {
        spikes_ordered = apply_sorted_row_order_matlab(bundle.downstream_spikes_ref,
                                                       bundle.sorted_indices_1based);
        std::cout << "Decoder feature parity: applied sortedIndices.csv (MATLAB row order).\n";
    } else {
        std::cout << "Decoder feature parity: sortedIndices.csv not found; using raw downstream row order "
                     "(export sortedIndices.csv if ref used sorted spikes).\n";
    }
    Eigen::MatrixXd feats_cpp = build_decoder_features_from_spikes(spikes_ordered, num_lags);
    const double err = max_abs_err(feats_cpp, bundle.decoder_features_ref);
    std::cout << "Decoder feature parity: max_abs_err=" << err << "\n";
    const double tol = 0.0;
    if (err > tol) {
        throw std::runtime_error(
            "decoder feature parity failed: nonzero error (check sortedIndices.csv vs MATLAB spkOutPredict order)");
    }
}

void run_decoder_output_model01_parity_or_throw(const F1Bundle& bundle,
                                         const Eigen::VectorXd& xoffset,
                                         const Eigen::VectorXd& gain,
                                         double ymin) {
    // Use reference decoder features as input first (isolates the decoder model)
    const Eigen::MatrixXd& ensemble = bundle.decoder_features_ref;

    const Eigen::MatrixXd& IW1_1 = bundle.decoder_W1;
    const Eigen::VectorXd& b1    = bundle.decoder_b1;
    const Eigen::MatrixXd& LW2_1 = bundle.decoder_W2;
    const Eigen::VectorXd& b2    = bundle.decoder_b2;

    Eigen::MatrixXd y_cpp = decodingModel01_forward(ensemble, xoffset, gain, ymin, IW1_1, b1, LW2_1, b2);

    if (y_cpp.rows() != bundle.decoder_logits_ref.rows() ||
        y_cpp.cols() != bundle.decoder_logits_ref.cols()) {
        throw std::runtime_error("decoder model01 parity: output shape mismatch");
    }

    Eigen::MatrixXd diff = (y_cpp - bundle.decoder_logits_ref).cwiseAbs();
    const double max_abs = diff.maxCoeff();
    const double mean_abs = diff.mean();

    std::cout << "Decoder model01 parity: max_abs_err=" << max_abs
              << " mean_abs_err=" << mean_abs << "\n";

    const double tol = 1e-9;
    if (max_abs > tol) {
        throw std::runtime_error("decoder model01 parity failed");
    }

    // Labels parity (argmax along rows, per column; MATLAB labels are 1-based)
    Eigen::VectorXi labels_cpp(y_cpp.cols());
    for (int c = 0; c < y_cpp.cols(); ++c) {
        Eigen::Index idx = 0;
        y_cpp.col(c).maxCoeff(&idx);
        labels_cpp(c) = static_cast<int>(idx) + 1;
    }

    // compare to labels_ref
    int mism = 0;
    for (int c = 0; c < labels_cpp.size(); ++c) {
        if (labels_cpp(c) != bundle.labels_ref(c)) mism++;
    }
    std::cout << "Decoder labels mismatches: " << mism << " / " << labels_cpp.size() << "\n";
    if (mism != 0) {
        throw std::runtime_error("decoder label parity failed");
    }
}