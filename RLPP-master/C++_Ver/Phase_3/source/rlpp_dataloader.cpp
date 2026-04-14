#include "rlpp_dataloader.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace rlpp {

static int wrap_index_numpy(int idx, int T) {
    // numpy semantics for negative indices: -1 -> T-1, etc.
    if (T <= 0) {
        throw std::runtime_error("wrap_index_numpy: T must be positive");
    }
    int out = idx;
    if (out < 0) {
        out = T + (out % T);
        if (out == T) out = 0;
    }
    // Python code does not clamp positive out-of-range, but with offsets [-L..0] it shouldn't happen.
    if (out < 0 || out >= T) {
        throw std::runtime_error("wrap_index_numpy: index out of range after wrap");
    }
    return out;
}

static std::vector<int> unique_sorted(std::vector<int> v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

DataLoaderBatch dataloader_forward_exact(
    const Eigen::MatrixXd& input_ensemble,
    const Eigen::MatrixXd& m1_truth,
    const Eigen::VectorXi& actions,
    const Eigen::VectorXi& trials,
    DataLoaderOpt& opt,
    std::mt19937& rng
) {
    const int T = static_cast<int>(input_ensemble.cols());
    if (m1_truth.cols() != T || actions.size() != T || trials.size() != T) {
        throw std::runtime_error("dataloader_forward_exact: time dimension mismatch");
    }
    if (T <= 0) {
        throw std::runtime_error("dataloader_forward_exact: empty time axis");
    }

    std::vector<int> trial_indexes;
    if (opt.mode == DataLoaderMode::Train) {
        if (opt.shuffle_when_cursor_is_one && opt.data_loader_cursor == 1) {
            // shuffle trainTrials in-place
            std::shuffle(opt.train_trials.begin(), opt.train_trials.end(), rng);
        }
        const int N = opt.number_of_train_trials > 0 ? opt.number_of_train_trials
                                                     : static_cast<int>(opt.train_trials.size());
        if (N <= 0) {
            throw std::runtime_error("dataloader_forward_exact: NumberOfTrainTrials <= 0");
        }
        const int start = opt.data_loader_cursor; // 1-based
        const int stop = std::min(N, opt.data_loader_cursor + opt.batch_size - 1); // inclusive, 1-based

        // advance cursor like Python: (stop % N) + 1
        opt.data_loader_cursor = (stop % N) + 1;

        if (static_cast<int>(opt.train_trials.size()) < N) {
            throw std::runtime_error("dataloader_forward_exact: train_trials shorter than NumberOfTrainTrials");
        }
        for (int i = start - 1; i < stop; ++i) {
            trial_indexes.push_back(opt.train_trials[static_cast<std::size_t>(i)]);
        }
    } else if (opt.mode == DataLoaderMode::Test) {
        trial_indexes = opt.test_trials;
    } else if (opt.mode == DataLoaderMode::All) {
        const int N = opt.number_of_train_trials > 0 ? opt.number_of_train_trials
                                                     : static_cast<int>(opt.train_trials.size());
        trial_indexes.reserve(N);
        for (int t = 1; t <= N; ++t) {
            trial_indexes.push_back(t);
        }
    } else {
        throw std::runtime_error("dataloader_forward_exact: unknown mode");
    }

    // time_indexes = concat(where(Trials == t) for each trial)
    std::vector<int> time_idx;
    for (int t : trial_indexes) {
        for (int i = 0; i < trials.size(); ++i) {
            if (trials(i) == t) {
                time_idx.push_back(i);
            }
        }
    }

    // offsets = [-discountLength..0], then unique(time_indexes[:,None] + offsets)
    std::vector<int> expanded;
    expanded.reserve(time_idx.size() * static_cast<std::size_t>(opt.discount_length + 1));
    for (int base : time_idx) {
        for (int off = -opt.discount_length; off <= 0; ++off) {
            expanded.push_back(base + off);
        }
    }
    expanded = unique_sorted(std::move(expanded));

    // Gather with numpy negative indexing
    const int n_time = static_cast<int>(expanded.size());
    DataLoaderBatch out;
    out.batch_input.resize(input_ensemble.rows(), n_time);
    out.batch_m1_truth.resize(m1_truth.rows(), n_time);
    out.batch_actions.resize(n_time);

    for (int j = 0; j < n_time; ++j) {
        const int idx = wrap_index_numpy(expanded[static_cast<std::size_t>(j)], T);
        out.batch_input.col(j) = input_ensemble.col(idx);
        out.batch_m1_truth.col(j) = m1_truth.col(idx);
        out.batch_actions(j) = actions(idx);
    }

    return out;
}

} // namespace rlpp

