#include "F1_rat01_bundle.hpp"
#include "F1_rat01_validate.hpp"

#include <stdexcept>
#include <string>
#include <sstream>

static std::string shape_str(const Eigen::MatrixXd& M) {
    std::ostringstream oss;
    oss << M.rows() << "x" << M.cols();
    return oss.str();
}
static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error("F1 validation failed: " + msg);
}
static void require_matrix_nonempty(const Eigen::MatrixXd& M, const std::string& name) {
    require(M.rows() > 0 && M.cols() > 0, name + " is empty (" + shape_str(M) + ")");
}
static void require_vector_nonempty(const Eigen::VectorXi& v, const std::string& name) {
    require(v.size() > 0, name + " is empty (size=" + std::to_string(v.size()) + ")");
}
// If you know his from metadata, pass it in. Otherwise infer num_lags from decoder_features_ref.
void validate_f1_bundle_or_throw(const F1Bundle& b, int his /* pass -1 if unknown */) {
    // --- required matrices exist ---
    require_matrix_nonempty(b.upstream_spikes, "upstream_spikes");
    require_matrix_nonempty(b.generator_W1, "generator_W1");
    require_matrix_nonempty(b.generator_W2, "generator_W2");
    require_matrix_nonempty(b.encoder_features_ref, "encoder_features_ref");
    require_matrix_nonempty(b.generator_probs_ref, "generator_probs_ref");
    require_matrix_nonempty(b.downstream_spikes_ref, "downstream_spikes_ref");
    require_matrix_nonempty(b.decoder_features_ref, "decoder_features_ref");
    require_matrix_nonempty(b.decoder_logits_ref, "decoder_logits_ref");
    require_vector_nonempty(b.labels_ref, "labels_ref");
    // --- upstream spikes basic sanity ---
    const int T_up = static_cast<int>(b.upstream_spikes.rows());
    const int Nx   = static_cast<int>(b.upstream_spikes.cols());
    require(T_up > 0, "upstream_spikes must have T>0 rows");
    require(Nx  > 0, "upstream_spikes must have Nx>0 cols");
    // --- reference time axis agreement ---
    const int T_valid = static_cast<int>(b.encoder_features_ref.cols());
    require(T_valid > 0, "encoder_features_ref must have T_valid>0 columns");
    require(b.generator_probs_ref.cols() == T_valid,
            "generator_probs_ref time dimension mismatch: expected cols=" + std::to_string(T_valid) +
            " got " + shape_str(b.generator_probs_ref));
    require(b.downstream_spikes_ref.cols() == T_valid,
            "downstream_spikes_ref time dimension mismatch: expected cols=" + std::to_string(T_valid) +
            " got " + shape_str(b.downstream_spikes_ref));
    require(b.decoder_features_ref.cols() == T_valid,
            "decoder_features_ref time dimension mismatch: expected cols=" + std::to_string(T_valid) +
            " got " + shape_str(b.decoder_features_ref));
    require(b.decoder_logits_ref.cols() == T_valid,
            "decoder_logits_ref time dimension mismatch: expected cols=" + std::to_string(T_valid) +
            " got " + shape_str(b.decoder_logits_ref));
    require(b.labels_ref.size() == T_valid,
            "labels_ref length mismatch: expected " + std::to_string(T_valid) +
            " got " + std::to_string(b.labels_ref.size()));
    // --- feature dim from encoder ref ---
    const int feature_dim = static_cast<int>(b.encoder_features_ref.rows());
    require(feature_dim > 0, "encoder_features_ref feature_dim must be >0");
    // --- generator weight shape checks (bias-in-columns convention) ---
    const int hidden_dim = static_cast<int>(b.generator_W1.rows());
    require(hidden_dim > 0, "generator_W1 hidden_dim must be >0");
    require(b.generator_W1.cols() == feature_dim + 1,
            "generator_W1 cols must equal feature_dim+1 (bias column). "
            "feature_dim=" + std::to_string(feature_dim) +
            " expected cols=" + std::to_string(feature_dim + 1) +
            " got " + shape_str(b.generator_W1));
    const int Ny = static_cast<int>(b.generator_W2.rows());
    require(Ny > 0, "generator_W2 Ny must be >0");
    require(b.generator_W2.cols() == hidden_dim + 1,
            "generator_W2 cols must equal hidden_dim+1 (bias column). "
            "hidden_dim=" + std::to_string(hidden_dim) +
            " expected cols=" + std::to_string(hidden_dim + 1) +
            " got " + shape_str(b.generator_W2));
    // generator reference outputs must match Ny
    require(b.generator_probs_ref.rows() == Ny,
            "generator_probs_ref rows must equal Ny. Ny=" + std::to_string(Ny) +
            " got " + shape_str(b.generator_probs_ref));
    require(b.downstream_spikes_ref.rows() == Ny,
            "downstream_spikes_ref rows must equal Ny. Ny=" + std::to_string(Ny) +
            " got " + shape_str(b.downstream_spikes_ref));
    // --- decoder features shape sanity ---
    const int dec_feat_rows = static_cast<int>(b.decoder_features_ref.rows());
    require(dec_feat_rows % Ny == 0,
            "decoder_features_ref rows must be divisible by Ny=" + std::to_string(Ny) +
            " got rows=" + std::to_string(dec_feat_rows));
    const int num_lags = dec_feat_rows / Ny; // should equal his+1
    if (his >= 0) {
        require(num_lags == his + 1,
                "decoder_features_ref implies num_lags=" + std::to_string(num_lags) +
                " but expected his+1=" + std::to_string(his + 1));
    }
    // --- optional: decoder NN weight checks if present ---
    const bool has_decoder_nn =
        (b.decoder_W1.rows() > 0 && b.decoder_W1.cols() > 0 &&
         b.decoder_W2.rows() > 0 && b.decoder_W2.cols() > 0 &&
         b.decoder_b1.size() > 0 && b.decoder_b2.size() > 0);
    if (has_decoder_nn) {
        // If you’re using decodingModel_01/_02, their shapes differ from your generator MLP
        // (they’re not your TwoLayerMLP), so only do basic consistency checks here.
        require(b.decoder_W1.cols() == dec_feat_rows,
                "decoder_W1 cols should match decoder feature rows. expected " +
                std::to_string(dec_feat_rows) + " got " + shape_str(b.decoder_W1));
        require(b.decoder_b1.size() == b.decoder_W1.rows(),
                "decoder_b1 size must match decoder_W1 rows");
        require(b.decoder_W2.cols() == b.decoder_W1.rows(),
                "decoder_W2 cols must match decoder_W1 rows");
        require(b.decoder_b2.size() == b.decoder_W2.rows(),
                "decoder_b2 size must match decoder_W2 rows");
    }
}