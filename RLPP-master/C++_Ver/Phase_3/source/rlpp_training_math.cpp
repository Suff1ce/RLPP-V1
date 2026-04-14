#include "rlpp_training_math.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace rlpp {

static Eigen::MatrixXd sigmoid_mat(const Eigen::MatrixXd& z) {
    return z.unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
}

static Eigen::VectorXd row_std_ddof0(const Eigen::MatrixXd& m) {
    const int r = static_cast<int>(m.rows());
    Eigen::VectorXd out(r);
    for (int i = 0; i < r; ++i) {
        Eigen::RowVectorXd row = m.row(i);
        const double mean = row.mean();
        double var = 0.0;
        for (int j = 0; j < row.size(); ++j) {
            const double d = row(j) - mean;
            var += d * d;
        }
        var /= static_cast<double>(row.size());
        out(i) = std::sqrt(var);
    }
    return out;
}

static Eigen::MatrixXd sample_bernoulli_mat(const Eigen::MatrixXd& p, std::mt19937& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    Eigen::MatrixXd s(p.rows(), p.cols());
    for (int i = 0; i < p.rows(); ++i) {
        for (int j = 0; j < p.cols(); ++j) {
            s(i, j) = (u(rng) <= p(i, j)) ? 1.0 : 0.0;
        }
    }
    return s;
}

static Eigen::MatrixXd sample_bernoulli_mat_from_uniforms(const Eigen::MatrixXd& p, const Eigen::MatrixXd& u01) {
    if (u01.rows() != p.rows() || u01.cols() != p.cols()) {
        throw std::runtime_error("uniform_01 shape mismatch");
    }
    Eigen::MatrixXd s(p.rows(), p.cols());
    for (int i = 0; i < p.rows(); ++i) {
        for (int j = 0; j < p.cols(); ++j) {
            s(i, j) = (u01(i, j) <= p(i, j)) ? 1.0 : 0.0;
        }
    }
    return s;
}

ApplyNetsPrioriBatch applynets_priori_forward(
    const Eigen::MatrixXd& input_unit,
    const Eigen::MatrixXd& W1,
    const Eigen::MatrixXd& W2,
    int episode,
    double priori_m,
    double priori_n,
    std::mt19937& rng
) {
    const int n = static_cast<int>(input_unit.cols());
    const int hidden = static_cast<int>(W1.rows());
    const int ny = static_cast<int>(W2.rows());

    Eigen::MatrixXd hidden_pre = W1 * input_unit;
    Eigen::MatrixXd hidden_sig = sigmoid_mat(hidden_pre);
    Eigen::MatrixXd hidden_unit(hidden + 1, n);
    hidden_unit.topRows(hidden) = hidden_sig;
    hidden_unit.row(hidden).setOnes();

    Eigen::MatrixXd output_unit = W2 * hidden_unit;
    Eigen::MatrixXd p_output = sigmoid_mat(output_unit);

    Eigen::VectorXd std_per_row = row_std_ddof0(p_output);
    Eigen::VectorXd temp = (static_cast<double>(episode) + 1.0) * std_per_row;
    for (int i = 0; i < temp.size(); ++i) {
        if (temp(i) < 1e-3) {
            temp(i) = 1e-3;
        }
    }
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < n; ++j) {
            const double t = temp(i);
            p_output(i, j) = (t * p_output(i, j) + priori_m) / (t + priori_n);
        }
    }

    Eigen::MatrixXd spk_out = sample_bernoulli_mat(p_output, rng);

    ApplyNetsPrioriBatch out;
    out.p_output = std::move(p_output);
    out.hidden_unit = std::move(hidden_unit);
    out.spk_out = std::move(spk_out);
    return out;
}

ApplyNetsPrioriBatch applynets_priori_forward_with_uniforms(
    const Eigen::MatrixXd& input_unit,
    const Eigen::MatrixXd& W1,
    const Eigen::MatrixXd& W2,
    int episode,
    double priori_m,
    double priori_n,
    const Eigen::MatrixXd& uniform_01
) {
    const int n = static_cast<int>(input_unit.cols());
    const int hidden = static_cast<int>(W1.rows());
    const int ny = static_cast<int>(W2.rows());

    Eigen::MatrixXd hidden_pre = W1 * input_unit;
    Eigen::MatrixXd hidden_sig = sigmoid_mat(hidden_pre);
    Eigen::MatrixXd hidden_unit(hidden + 1, n);
    hidden_unit.topRows(hidden) = hidden_sig;
    hidden_unit.row(hidden).setOnes();

    Eigen::MatrixXd output_unit = W2 * hidden_unit;
    Eigen::MatrixXd p_output = sigmoid_mat(output_unit);

    Eigen::VectorXd std_per_row = row_std_ddof0(p_output);
    Eigen::VectorXd temp = (static_cast<double>(episode) + 1.0) * std_per_row;
    for (int i = 0; i < temp.size(); ++i) {
        if (temp(i) < 1e-3) {
            temp(i) = 1e-3;
        }
    }
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < n; ++j) {
            const double t = temp(i);
            p_output(i, j) = (t * p_output(i, j) + priori_m) / (t + priori_n);
        }
    }

    Eigen::MatrixXd spk_out = sample_bernoulli_mat_from_uniforms(p_output, uniform_01);

    ApplyNetsPrioriBatch out;
    out.p_output = std::move(p_output);
    out.hidden_unit = std::move(hidden_unit);
    out.spk_out = std::move(spk_out);
    return out;
}

void getgradient_rl(
    const Eigen::MatrixXd& reward_per_sample,
    const Eigen::MatrixXd& p_output,
    const Eigen::MatrixXd& spk_out_predict,
    const Eigen::MatrixXd& hidden_unit,
    const Eigen::MatrixXd& input_unit,
    const Eigen::MatrixXd& weight_hidden_output,
    int num_samples,
    Eigen::MatrixXd& weight_delta1,
    Eigen::MatrixXd& weight_delta2
) {
    const Eigen::MatrixXd delta = reward_per_sample.cwiseProduct(spk_out_predict - p_output);
    weight_delta1 = (delta * hidden_unit.transpose()) / static_cast<double>(num_samples);

    const Eigen::MatrixXd wtd = weight_hidden_output.transpose() * delta;
    const Eigen::MatrixXd gated = hidden_unit.cwiseProduct(
        (Eigen::MatrixXd::Ones(hidden_unit.rows(), hidden_unit.cols()) - hidden_unit).cwiseProduct(wtd)
    );
    weight_delta2 = (gated * input_unit.transpose()) / static_cast<double>(num_samples);
    if (weight_delta2.rows() > 0) {
        weight_delta2.conservativeResize(weight_delta2.rows() - 1, weight_delta2.cols());
    }
}

void getgradient_rl_broadcast(
    const Eigen::RowVectorXd& smoothed_reward_n,
    const Eigen::MatrixXd& p_output,
    const Eigen::MatrixXd& spk_out_predict,
    const Eigen::MatrixXd& hidden_unit,
    const Eigen::MatrixXd& input_unit,
    const Eigen::MatrixXd& weight_hidden_output,
    int num_samples,
    Eigen::MatrixXd& weight_delta1,
    Eigen::MatrixXd& weight_delta2
) {
    const int ny = static_cast<int>(p_output.rows());
    const int n = static_cast<int>(p_output.cols());
    if (smoothed_reward_n.size() != n) {
        throw std::runtime_error("getgradient_rl_broadcast: reward length");
    }
    Eigen::MatrixXd reward_mat(ny, n);
    for (int i = 0; i < ny; ++i) {
        reward_mat.row(i) = smoothed_reward_n;
    }
    getgradient_rl(
        reward_mat,
        p_output,
        spk_out_predict,
        hidden_unit,
        input_unit,
        weight_hidden_output,
        num_samples,
        weight_delta1,
        weight_delta2
    );
}

double learning_rate_rl_simulations(int episode, int max_episode) {
    return 0.1 * (1.0 - static_cast<double>(episode) / static_cast<double>(max_episode)) + 0.5;
}

double learning_rate_rl_real(int episode, int max_episode) {
    return 0.7 * (1.0 - static_cast<double>(episode) / static_cast<double>(max_episode)) + 0.5;
}

Eigen::RowVectorXd compute_smoothed_reward_rlpp(
    const Eigen::RowVectorXd& success,
    const Eigen::VectorXi& motor_perform,
    double epsilon,
    int episode,
    int max_episode,
    double discount_factor,
    int discount_length
) {
    const int n = static_cast<int>(success.size());
    if (motor_perform.size() != n) {
        throw std::runtime_error("compute_smoothed_reward_rlpp: size");
    }

    long long c1 = 0, c2 = 0, c3 = 0;
    for (int i = 0; i < n; ++i) {
        const int m = motor_perform(i);
        if (m == 1) {
            ++c1;
        } else if (m == 2) {
            ++c2;
        } else if (m == 3) {
            ++c3;
        }
    }
    const double n_motor1 = static_cast<double>(c1) + 1.0;
    const double n_motor2 = static_cast<double>(c2) + 1.0;
    const double n_motor3 = static_cast<double>(c3) + 1.0;
    const double n_max = std::max({n_motor1, n_motor2, n_motor3});

    Eigen::RowVectorXd inner_reward(1, n);
    for (int i = 0; i < n; ++i) {
        const int m = motor_perform(i);
        double ir = 0.0;
        if (m == 1) {
            ir = n_max / n_motor1 - 1.0;
        } else if (m == 2) {
            ir = n_max / n_motor2 - 1.0;
        } else if (m == 3) {
            ir = n_max / n_motor3 - 1.0;
        }
        inner_reward(i) = ir;
    }

    const double eps_scale = epsilon * (1.0 - static_cast<double>(episode) / static_cast<double>(max_episode));
    Eigen::RowVectorXd reward(1, n);
    for (int i = 0; i < n; ++i) {
        const double s = success(i);
        if (std::isnan(s)) {
            reward(i) = std::numeric_limits<double>::quiet_NaN();
        } else {
            reward(i) = s + eps_scale * inner_reward(i);
        }
    }

    Eigen::RowVectorXd temp = reward;
    for (int i = 0; i < n; ++i) {
        if (std::isnan(temp(i))) {
            temp(i) = 0.0;
        }
    }

    std::vector<double> filt(static_cast<std::size_t>(discount_length));
    for (int k = 0; k < discount_length; ++k) {
        const int power = discount_length - 1 - k;
        filt[static_cast<std::size_t>(k)] =
            std::pow(discount_factor, static_cast<double>(power)) / static_cast<double>(discount_length);
    }

    const int L = discount_length;
    const int conv_len = n + L - 1;
    Eigen::RowVectorXd full_conv(1, conv_len);
    full_conv.setZero();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < L; ++j) {
            full_conv(i + j) += temp(i) * filt[static_cast<std::size_t>(j)];
        }
    }

    Eigen::RowVectorXd smoothed(1, n);
    for (int i = 0; i < n; ++i) {
        smoothed(i) = full_conv(conv_len - n + i);
    }

    double sum = 0.0;
    int cnt = 0;
    for (int i = 0; i < n; ++i) {
        if (!std::isnan(reward(i))) {
            sum += smoothed(i);
            ++cnt;
        }
    }
    if (cnt > 0) {
        const double mean = sum / static_cast<double>(cnt);
        double var = 0.0;
        for (int i = 0; i < n; ++i) {
            if (!std::isnan(reward(i))) {
                const double d = smoothed(i) - mean;
                var += d * d;
            }
        }
        var /= static_cast<double>(cnt);
        const double std_dev = std::sqrt(var);
        if (std_dev > 1e-15) {
            for (int i = 0; i < n; ++i) {
                if (!std::isnan(reward(i))) {
                    smoothed(i) = (smoothed(i) - mean) / std_dev;
                }
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        if (std::isnan(reward(i))) {
            smoothed(i) = 0.0;
        }
    }

    return smoothed;
}

} // namespace rlpp
