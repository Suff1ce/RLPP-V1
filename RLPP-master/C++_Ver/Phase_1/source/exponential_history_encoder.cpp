#include "exponential_history_encoder.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

ExponentialHistoryEncoder::ExponentialHistoryEncoder(int num_inputs,
                                                     int history_length,
                                                     double tau_bins)
    : Nx(num_inputs),
      H(history_length),
      tau(tau_bins),
      spike_times_per_neuron(num_inputs),
      h_spike_time(num_inputs, -1),
      last_observed_time_bin(0),
      last_encoded_time_bin(0) {
    if (Nx <= 0) {
        throw std::runtime_error("ExponentialHistoryEncoder: num_inputs must be positive");
    }
    if (H <= 0) {
        throw std::runtime_error("ExponentialHistoryEncoder: history_length must be positive");
    }
    if (tau <= 0.0) {
        throw std::runtime_error("ExponentialHistoryEncoder: tau_bins must be positive");
    }
}

void ExponentialHistoryEncoder::observe_spike(int neuron_index, int time_bin) {
    if (neuron_index < 0 || neuron_index >= Nx) {
        throw std::runtime_error("observe_spike: neuron_index out of range");
    }
    if (time_bin <= 0) {
        throw std::runtime_error("observe_spike: time_bin must be 1-based and positive");
    }

    std::deque<int>& spikes = spike_times_per_neuron[neuron_index];

    if (!spikes.empty() && time_bin < spikes.back()) {
        throw std::runtime_error(
            "observe_spike: spike times must be nondecreasing for each neuron");
    }

    spikes.push_back(time_bin);

    // Record the time of the H-th spike once, matching Matlab Start logic.
    if (h_spike_time[neuron_index] < 0 &&
        static_cast<int>(spikes.size()) == H) {
        h_spike_time[neuron_index] = time_bin;
    }

    // Keep only a bounded suffix for strict online operation.
    while (static_cast<int>(spikes.size()) > H) {
        spikes.pop_front();
    }
}

void ExponentialHistoryEncoder::observe_bin(const Eigen::VectorXd& spikes, int time_bin) {
    if (spikes.size() != Nx) {
        throw std::runtime_error("observe_bin: spikes vector size mismatch");
    }
    if (time_bin <= 0) {
        throw std::runtime_error("observe_bin: time_bin must be 1-based and positive");
    }
    if (time_bin < last_observed_time_bin) {
        throw std::runtime_error("observe_bin: time bins must be globally nondecreasing");
    }

    for (int i = 0; i < Nx; ++i) {
        if (spikes(i) != 0.0) {
            observe_spike(i, time_bin);
        }
    }

    last_observed_time_bin = time_bin;
}

int ExponentialHistoryEncoder::start_time_bin() const {
    int start = -1;

    for (int i = 0; i < Nx; ++i) {
        if (h_spike_time[i] < 0) {
            return -1;
        }
        int candidate = h_spike_time[i] + 1;
        if (candidate > start) {
            start = candidate;
        }
    }

    return start;
}

bool ExponentialHistoryEncoder::can_encode(int current_time_bin) const {
    if (current_time_bin <= 0) {
        return false;
    }

    int start = start_time_bin();
    if (start < 0) {
        return false;
    }

    return current_time_bin >= start;
}

Eigen::VectorXd ExponentialHistoryEncoder::encode(int current_time_bin) const {
    if (current_time_bin <= 0) {
        throw std::runtime_error("encode: current_time_bin must be 1-based and positive");
    }
    if (current_time_bin < last_encoded_time_bin) {
        throw std::runtime_error("encode: strict online mode requires nondecreasing encode times");
    }
    if (!can_encode(current_time_bin)) {
        throw std::runtime_error(
            "encode: current_time_bin is earlier than Python-compatible Start "
            "or Start is not yet available");
    }

    Eigen::VectorXd features = Eigen::VectorXd::Zero(Nx * H);

    for (int neuron = 0; neuron < Nx; ++neuron) {
        const std::deque<int>& spikes = spike_times_per_neuron[neuron];

        // In strict online mode, after Start is available and time advances online,
        // all retained spikes are <= current_time_bin as long as observe/encode are used
        // in nondecreasing time order.
        int n = static_cast<int>(spikes.size());
        int num_relevant = (n < H) ? n : H;
        int start_idx = n - num_relevant;
        int offset = neuron * H;

        for (int j = 0; j < num_relevant; ++j) {
            int spike_time = spikes[start_idx + j];
            int delta_t = current_time_bin - spike_time;

            if (delta_t < 0) {
                throw std::runtime_error(
                    "encode: encountered future spike; observe/encode must be used online");
            }

            features(offset + j) =
                std::exp(-static_cast<double>(delta_t) / tau);
        }
    }

    last_encoded_time_bin = current_time_bin;
    return features;
}

int ExponentialHistoryEncoder::feature_size() const {
    return Nx * H;
}