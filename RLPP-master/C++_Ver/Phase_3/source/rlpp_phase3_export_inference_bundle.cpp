#include "rlpp_phase3_export_inference_bundle.hpp"
#include "csv_utils.hpp"
#include "rlpp_decoding_params.hpp"

#include <Eigen/Dense>

#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

static void copy_file_or_throw(const fs::path& src, const fs::path& dst) {
    std::error_code ec;
    fs::create_directories(dst.parent_path(), ec);
    fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        throw std::runtime_error("copy_file failed: " + src.string() + " -> " + dst.string() + " : " + ec.message());
    }
}

static void save_sorted_indices_csv(const std::string& path, const Eigen::VectorXi& v) {
    Eigen::VectorXd vd(v.size());
    for (int i = 0; i < v.size(); ++i) vd(i) = static_cast<double>(v(i));
    save_csv_vector(path, vd);
}

void export_phase3_inference_bundle(
    const std::string& out_dir,
    const Eigen::MatrixXd& W1,
    const Eigen::MatrixXd& W2,
    const Eigen::VectorXi& sorted_indices_1based,
    int feat_rows_input_ensemble,
    const std::string& decoder_prefix,
    int export_nx,
    int export_encoder_h,
    const std::string& phase3_model_name,
    const std::string& phase3_case_dir
) {
    if (out_dir.empty()) throw std::runtime_error("export: out_dir empty");
    fs::create_directories(fs::path(out_dir));

    const int feat = feat_rows_input_ensemble;
    if (feat <= 0) throw std::runtime_error("export: bad feat_rows_input_ensemble");
    if (W1.cols() != feat + 1) throw std::runtime_error("export: W1 cols mismatch vs feat");
    if (W1.rows() <= 0 || W2.rows() <= 0) throw std::runtime_error("export: bad W1/W2 shape");

    int Nx = export_nx;
    int H_enc = export_encoder_h;
    if (Nx <= 0) {
        Nx = feat;
        H_enc = 1;
    } else {
        if (H_enc <= 0) H_enc = 1;
    }
    if (Nx * H_enc != feat) {
        throw std::runtime_error(
            "export: need Nx*H_enc == inputEnsemble.rows(). Got feat=" + std::to_string(feat) +
            " Nx=" + std::to_string(Nx) + " H_enc=" + std::to_string(H_enc) +
            ". Set --export-inference-nx and --export-encoder-h so Nx*H equals feat."
        );
    }

    const int Ny = static_cast<int>(W2.rows());
    if (sorted_indices_1based.size() != Ny) {
        throw std::runtime_error("export: sorted_indices length != Ny");
    }

    rlpp::DecodingNnParams dec = rlpp::load_decoding_nn_params_csv(decoder_prefix);
    const int in_dim = static_cast<int>(dec.xoffset.size());
    if (dec.IW1_1.cols() != in_dim) {
        throw std::runtime_error("export: decoder IW1_1 cols vs xoffset");
    }
    if (in_dim % Ny != 0) {
        throw std::runtime_error(
            "export: decoder input_dim not divisible by trained Ny=" + std::to_string(Ny)
        );
    }
    const int num_lags = in_dim / Ny;
    const int num_labels = static_cast<int>(dec.LW2_1.rows());

    // ---- Core weights / indices ----
    save_csv_matrix((fs::path(out_dir) / "generator_W1.csv").string(), W1);
    save_csv_matrix((fs::path(out_dir) / "generator_W2.csv").string(), W2);
    save_sorted_indices_csv((fs::path(out_dir) / "sortedIndices.csv").string(), sorted_indices_1based);

    // ---- Decoder NN + mapminmax (F1 bundle names) ----
    const fs::path out(out_dir);
    copy_file_or_throw(fs::path(decoder_prefix + "_xoffset.csv"), out / "decoder_xoffset.csv");
    copy_file_or_throw(fs::path(decoder_prefix + "_gain.csv"), out / "decoder_gain.csv");
    copy_file_or_throw(fs::path(decoder_prefix + "_ymin.csv"), out / "decoder_ymin.csv");
    copy_file_or_throw(fs::path(decoder_prefix + "_IW1_1.csv"), out / "decoder_W1.csv");
    copy_file_or_throw(fs::path(decoder_prefix + "_b1.csv"), out / "decoder_b1.csv");
    copy_file_or_throw(fs::path(decoder_prefix + "_LW2_1.csv"), out / "decoder_W2.csv");
    copy_file_or_throw(fs::path(decoder_prefix + "_b2.csv"), out / "decoder_b2.csv");

    // ---- Synthetic reference tensors (shape-only; not used by realtime inference hot path) ----
    constexpr int T_valid = 256;
    constexpr int T_up = 2000;

    Eigen::MatrixXd upstream = Eigen::MatrixXd::Zero(T_up, Nx);
    save_csv_matrix((out / "upstream_spikes.csv").string(), upstream);

    Eigen::MatrixXd encoder_features_ref = Eigen::MatrixXd::Zero(feat, T_valid);
    save_csv_matrix((out / "encoder_features_ref.csv").string(), encoder_features_ref);

    Eigen::MatrixXd generator_probs_ref = Eigen::MatrixXd::Zero(Ny, T_valid);
    save_csv_matrix((out / "generator_probs_ref.csv").string(), generator_probs_ref);

    Eigen::MatrixXd downstream_spikes_ref = Eigen::MatrixXd::Zero(Ny, T_valid);
    save_csv_matrix((out / "downstream_spikes_ref.csv").string(), downstream_spikes_ref);

    Eigen::MatrixXd decoder_features_ref = Eigen::MatrixXd::Zero(in_dim, T_valid);
    save_csv_matrix((out / "decoder_features_ref.csv").string(), decoder_features_ref);

    Eigen::MatrixXd decoder_logits_ref = Eigen::MatrixXd::Zero(num_labels, T_valid);
    save_csv_matrix((out / "decoder_logits_ref.csv").string(), decoder_logits_ref);

    Eigen::MatrixXd labels_ref(T_valid, 1);
    for (int i = 0; i < T_valid; ++i) labels_ref(i, 0) = 1.0;
    save_csv_matrix((out / "labels_ref.csv").string(), labels_ref);

    // ---- Manifest ----
    {
        std::ofstream m(out / "phase3_export_manifest.txt");
        if (!m) throw std::runtime_error("export: cannot write manifest");
        m << "RLPP Phase 3 -> Phase 1 inference bundle export\n"
          << "====================================\n"
          << "This directory was produced by RLPP_Phase_3_Run --export-bundle-dir.\n"
          << "It is intended for RLPP_realtime_udp_infer / RLPP_Phase_1 load_f1_bundle().\n\n"
          << "phase3_case_dir: " << phase3_case_dir << "\n"
          << "phase3_model: " << phase3_model_name << "\n"
          << "decoder_prefix: " << decoder_prefix << "\n"
          << "feat_rows_input_ensemble: " << feat << "\n"
          << "phase3_export_inference_Nx: " << Nx << "\n"
          << "phase3_export_encoder_h: " << H_enc << "\n"
          << "Ny: " << Ny << "\n"
          << "decoder_num_lags: " << num_lags << "\n"
          << "num_labels: " << num_labels << "\n\n"
          << "NOTE: encoder_features_ref / *_ref.csv files are synthetic placeholders for shape checks.\n"
          << "Realtime inference uses trained generator weights + decoder weights + sortedIndices.\n";
    }
}
