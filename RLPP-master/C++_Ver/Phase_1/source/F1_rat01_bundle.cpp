#include "F1_rat01_bundle.hpp"
#include "csv_utils.hpp"

#include <stdexcept>
#include <filesystem>


namespace fs = std::filesystem;
static Eigen::VectorXd matrix_to_vector(const Eigen::MatrixXd& M, const std::string& name) {
    if (M.rows() == 0 || M.cols() == 0) {
        return Eigen::VectorXd(); // empty
    }
    if (M.cols() == 1) {
        // Column vector
        return M.col(0);
    }
    if (M.rows() == 1) {
        // Row vector
        return M.row(0).transpose();
    }
    throw std::runtime_error("Expected vector in " + name + ", but got "
                             + std::to_string(M.rows()) + "x" + std::to_string(M.cols()));
}
static Eigen::VectorXi matrix_to_ivector(const Eigen::MatrixXd& M, const std::string& name) {
    Eigen::VectorXd v = matrix_to_vector(M, name);
    Eigen::VectorXi vi(v.size());
    for (int i = 0; i < v.size(); ++i) {
        vi(i) = static_cast<int>(std::llround(v(i)));
    }
    return vi;
}
F1Bundle load_f1_bundle(const std::string& dir) {
    F1Bundle b;
    auto path = [&](const std::string& filename) {
        return (fs::path(dir) / filename).string();
    };
    // --- required files ---
    b.upstream_spikes       = load_csv_matrix(path("upstream_spikes.csv"));
    b.generator_W1          = load_csv_matrix(path("generator_W1.csv"));
    b.generator_W2          = load_csv_matrix(path("generator_W2.csv"));
    b.encoder_features_ref  = load_csv_matrix(path("encoder_features_ref.csv"));
    b.generator_probs_ref   = load_csv_matrix(path("generator_probs_ref.csv"));
    b.downstream_spikes_ref = load_csv_matrix(path("downstream_spikes_ref.csv"));
    b.decoder_features_ref  = load_csv_matrix(path("decoder_features_ref.csv"));
    b.decoder_logits_ref    = load_csv_matrix(path("decoder_logits_ref.csv"));
    {
        Eigen::MatrixXd labels_mat = load_csv_matrix(path("labels_ref.csv"));
        b.labels_ref = matrix_to_ivector(labels_mat, "labels_ref.csv");
        Eigen::MatrixXd sorted_indices_mat = load_csv_matrix(path("sortedIndices.csv"));
        b.sorted_indices_1based = matrix_to_ivector(sorted_indices_mat, "sortedIndices.csv");
    }
    // --- optional decoder weights ---
    try {
        b.decoder_W1 = load_csv_matrix(path("decoder_W1.csv"));
    } catch (...) {
        b.decoder_W1.resize(0, 0);
    }
    try {
        Eigen::MatrixXd db1 = load_csv_matrix(path("decoder_b1.csv"));
        b.decoder_b1 = matrix_to_vector(db1, "decoder_b1.csv");
    } catch (...) {
        b.decoder_b1.resize(0);
    }
    try {
        b.decoder_W2 = load_csv_matrix(path("decoder_W2.csv"));
    } catch (...) {
        b.decoder_W2.resize(0, 0);
    }
    try {
        Eigen::MatrixXd db2 = load_csv_matrix(path("decoder_b2.csv"));
        b.decoder_b2 = matrix_to_vector(db2, "decoder_b2.csv");
    } catch (...) {
        b.decoder_b2.resize(0);
    }
    // --- quick sanity checks ---
    if (b.upstream_spikes.rows() == 0 || b.upstream_spikes.cols() == 0) {
        throw std::runtime_error("upstream_spikes.csv is empty or not loaded");
    }
    if (b.generator_W1.rows() == 0 || b.generator_W1.cols() == 0) {
        throw std::runtime_error("generator_W1.csv not loaded");
    }
    if (b.generator_W2.rows() == 0 || b.generator_W2.cols() == 0) {
        throw std::runtime_error("generator_W2.csv not loaded");
    }
    return b;
}