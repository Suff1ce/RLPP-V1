#pragma once

#include <Eigen/Dense>
#include <deque>
#include <vector>

class ExponentialHistoryEncoder {
public:
    ExponentialHistoryEncoder(int num_inputs, int history_length, double tau_bins);

    // Record one spike for one neuron at a given 1-based time bin.
    // Time must be nondecreasing for each neuron.
    void observe_spike(int neuron_index, int time_bin);

    // Record spikes for a whole bin.
    // spikes(i) != 0.0 means neuron i spiked in this bin.
    void observe_bin(const Eigen::VectorXd& spikes, int time_bin);

    // Returns the Matlab-style global Start time:
    // max over neurons of (H-th spike time + 1).
    // Returns -1 if Start is not yet defined because some neuron
    // has fewer than H observed spikes.
    int start_time_bin() const;

    // True iff Matlab-style history is valid for this time bin.
    bool can_encode(int current_time_bin) const;

    // Encode current state at a given 1-based time bin.
    // Requires can_encode(current_time_bin) == true.
    // Feature layout:
    // [neuron0 oldest->newest H entries, neuron1 oldest->newest H entries, ...]
    Eigen::VectorXd encode(int current_time_bin) const;

    int feature_size() const;

private:
    int Nx;
    int H;
    double tau;

    // In strict online mode we only need a bounded suffix.
    // We keep up to H spikes per neuron:
    // - the most recent H for encoding
    // - one extra so Start can be determined once available
    std::vector<std::deque<int>> spike_times_per_neuron;

    // Stores the time of the H-th spike for each neuron once known, else -1.
    std::vector<int> h_spike_time;

    // Enforce whole-stream nondecreasing time for observe_bin / encode.
    int last_observed_time_bin;
    mutable int last_encoded_time_bin;
};