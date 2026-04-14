#include "RLPP_inference.hpp"
#include "math_functions.hpp"

#include <stdexcept>

RLPPInference::RLPPInference(
    int Nx, int H, int Ny, int num_lags,
    const Eigen::VectorXi& sorted_indices_1based,
    const Eigen::MatrixXd& gen_W1,
    const Eigen::MatrixXd& gen_W2,
    const Eigen::VectorXd& dec_xoffset,
    const Eigen::VectorXd& dec_gain,
    double dec_ymin,
    const Eigen::MatrixXd& dec_W1,
    const Eigen::VectorXd& dec_b1,
    const Eigen::MatrixXd& dec_W2,
    const Eigen::VectorXd& dec_b2,
    const RLPPInferenceConfig& cfg
)
    : cfg_(cfg),
      Nx_(Nx), H_(H), Ny_(Ny), num_lags_(num_lags),
      sorted_indices_1based_(sorted_indices_1based),
      encoder_(Nx, H, cfg.tau_bins),
      generator_(gen_W1, gen_W2),
      dec_hist_(Ny, num_lags),
      dec_xoffset_(dec_xoffset),
      dec_gain_(dec_gain),
      dec_ymin_(dec_ymin),
      dec_W1_(dec_W1),
      dec_b1_(dec_b1),
      dec_W2_(dec_W2),
      dec_b2_(dec_b2),
      rng_(cfg.rng_seed)
{
    if (generator_.output_dim() != Ny_) throw std::runtime_error("generator Ny mismatch");
    if (sorted_indices_1based_.size() != 0 && sorted_indices_1based_.size() != Ny_) {
        throw std::runtime_error("sorted_indices_1based length != Ny");
    }
    if (dec_xoffset_.size() != Ny_ * num_lags_ || dec_gain_.size() != Ny_ * num_lags_) {
        throw std::runtime_error("decoder mapminmax length != Ny*(his+1)");
    }
    if (dec_W1_.cols() != Ny_ * num_lags_) throw std::runtime_error("decoder_W1 input dim mismatch");
}

Eigen::VectorXd RLPPInference::reorder_m1_if_needed(const Eigen::VectorXd& v) const {
    return apply_sorted_indices_1based(v, sorted_indices_1based_);
}

RLPPInferenceStepOutput RLPPInference::step(const Eigen::VectorXd& upstream_bin, int t) {
    RLPPInferenceStepOutput out;
    encoder_.observe_bin(upstream_bin, t);

    if (!encoder_.can_encode(t)) {
        out.valid = false;
        return out;
    }

    out.valid = true;

    out.encoder_x = encoder_.encode(t);         // [Nx*H]
    out.gen_prob  = generator_.forward(out.encoder_x); // [Ny]

    // faithful path: sample spikes like MATLAB applynets
    if (cfg_.spike_mode == SpikeDriveMode::SampledBernoulli) {
        out.gen_spikes = sample_Bernoulli(out.gen_prob, rng_);
    } else {
        out.gen_spikes = out.gen_prob;
    }

    // reorder to emulator_real M1 sorted order before lagging
    Eigen::VectorXd spikes_sorted = reorder_m1_if_needed(out.gen_spikes);

    dec_hist_.push(spikes_sorted);
    out.decoder_ensemble = dec_hist_.flatten_for_python_decoder(); // [Ny*num_lags]

    // decoder forward expects [input_dim x Q]; use Q=1
    Eigen::MatrixXd E(out.decoder_ensemble.size(), 1);
    E.col(0) = out.decoder_ensemble;

    Eigen::MatrixXd Y = decodingModel01_forward(
        E,
        dec_xoffset_, dec_gain_, dec_ymin_,
        dec_W1_, dec_b1_,
        dec_W2_, dec_b2_
    );

    out.decoder_y = Y.col(0);
    Eigen::Index arg = 0;
    out.decoder_y.maxCoeff(&arg);
    out.label_1based = static_cast<int>(arg) + 1;

    return out;
}

RLPPInferenceStepOutput RLPPInference::complete_step_with_provided_downstream(
    const Eigen::VectorXd& downstream_spikes_unsorted,
    int t
) {
    if (downstream_spikes_unsorted.size() != Ny_) {
        throw std::runtime_error("complete_step_with_provided_downstream: downstream size != Ny");
    }

    RLPPInferenceStepOutput out;
    out.valid = true;

    out.encoder_x = encoder_.encode(t);
    out.gen_prob  = generator_.forward(out.encoder_x);

    out.gen_spikes = downstream_spikes_unsorted;

    Eigen::VectorXd spikes_sorted = reorder_m1_if_needed(out.gen_spikes);
    dec_hist_.push(spikes_sorted);

    out.decoder_ensemble = dec_hist_.flatten_for_python_decoder();

    Eigen::MatrixXd E(out.decoder_ensemble.size(), 1);
    E.col(0) = out.decoder_ensemble;

    Eigen::MatrixXd Y = decodingModel01_forward(
        E,
        dec_xoffset_, dec_gain_, dec_ymin_,
        dec_W1_, dec_b1_,
        dec_W2_, dec_b2_
    );

    out.decoder_y = Y.col(0);
    Eigen::Index arg = 0;
    out.decoder_y.maxCoeff(&arg);
    out.label_1based = static_cast<int>(arg) + 1;

    return out;
}

RLPPInferenceStepOutput RLPPInference::step_with_downstream_spikes(
    const Eigen::VectorXd& upstream_bin,
    const Eigen::VectorXd& downstream_spikes_unsorted,
    int t
) {
    encoder_.observe_bin(upstream_bin, t);

    if (!encoder_.can_encode(t)) {
        RLPPInferenceStepOutput out;
        out.valid = false;
        return out;
    }

    return complete_step_with_provided_downstream(downstream_spikes_unsorted, t);
}

RLPPInferenceStepOutput RLPPInference::step_with_downstream_spikes(
    const Eigen::VectorXd& upstream_bin,
    const Eigen::MatrixXd& downstream_spikes_ref,
    int col,
    int t
) {
    encoder_.observe_bin(upstream_bin, t);

    if (!encoder_.can_encode(t)) {
        RLPPInferenceStepOutput out;
        out.valid = false;
        return out;
    }

    if (col < 0 || col >= downstream_spikes_ref.cols()) {
        throw std::runtime_error("step_with_downstream_spikes: col out of range for downstream_spikes_ref");
    }

    return complete_step_with_provided_downstream(downstream_spikes_ref.col(col), t);
}