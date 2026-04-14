#include "rlpp_emulator.hpp"

#include "decoding_model01.hpp"
#include "rlpp_decoding_params.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace rlpp {

static Eigen::VectorXi argsort_indices(const Eigen::VectorXi& v) {
    std::vector<int> idx(static_cast<std::size_t>(v.size()));
    for (int i = 0; i < v.size(); ++i) idx[static_cast<std::size_t>(i)] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        if (v(a) != v(b)) return v(a) < v(b);
        return a < b;
    });
    Eigen::VectorXi out(v.size());
    for (int i = 0; i < v.size(); ++i) out(i) = idx[static_cast<std::size_t>(i)];
    return out;
}

static Eigen::MatrixXd apply_row_permutation(const Eigen::MatrixXd& M, const Eigen::VectorXi& perm) {
    if (perm.size() != M.rows()) {
        throw std::runtime_error("apply_row_permutation: perm size mismatch");
    }
    Eigen::MatrixXd out(M.rows(), M.cols());
    for (int r = 0; r < M.rows(); ++r) {
        out.row(r) = M.row(perm(r));
    }
    return out;
}

static Eigen::VectorXi decodingModel_simulation_exact(const Eigen::MatrixXd& spk2_by_T, int his) {
    if (spk2_by_T.rows() != 2) {
        throw std::runtime_error("decodingModel_simulation: expected 2 neurons");
    }
    if (his <= 0) {
        throw std::runtime_error("decodingModel_simulation: his must be positive");
    }
    const int T = static_cast<int>(spk2_by_T.cols());

    // uniform_filter1d size=his, mode='nearest', origin=0:
    // window indices: [i - floor(his/2) .. i + ceil(his/2) - 1] with edge clamp.
    const int left = his / 2;              // floor
    const int right = his - left - 1;      // so total = left + right + 1 = his

    auto filt = [&](const Eigen::RowVectorXd& x) -> Eigen::RowVectorXd {
        Eigen::RowVectorXd y(1, T);
        for (int i = 0; i < T; ++i) {
            double s = 0.0;
            for (int k = -left; k <= right; ++k) {
                int j = i + k;
                if (j < 0) j = 0;
                if (j >= T) j = T - 1;
                s += x(j);
            }
            y(i) = s / static_cast<double>(his);
        }
        return y;
    };

    Eigen::RowVectorXd x1 = spk2_by_T.row(0).cast<double>();
    Eigen::RowVectorXd x2 = spk2_by_T.row(1).cast<double>();
    Eigen::RowVectorXd m1_1_mean = filt(x1);
    Eigen::RowVectorXd m1_2_mean = filt(x2);

    Eigen::VectorXi motor(T);
    for (int i = 0; i < T; ++i) {
        const bool a = (m1_2_mean(i) < 0.25) && (m1_1_mean(i) < 0.25);
        const bool b = (m1_2_mean(i) >= 0.25) && (m1_1_mean(i) <= 0.25);
        const bool c = (m1_2_mean(i) >= 0.25) && (m1_1_mean(i) > 0.25);
        motor(i) = (a ? 1 : 0) + (b ? 2 : 0) + (c ? 3 : 0);
    }
    return motor;
}

static Eigen::MatrixXd decodingModel_manual_exact(const Eigen::MatrixXd& M1ensemble) {
    // Python: M1_fr = np.array([mean(M1ensemble[i::4,:], axis=0) for i in range(4)])
    const int rows = static_cast<int>(M1ensemble.rows());
    const int T = static_cast<int>(M1ensemble.cols());
    if (rows % 4 != 0) {
        throw std::runtime_error("decodingModel_manual: expected rows divisible by 4");
    }
    Eigen::MatrixXd fr(4, T);
    fr.setZero();
    for (int g = 0; g < 4; ++g) {
        int count = 0;
        for (int r = g; r < rows; r += 4) {
            fr.row(g) += M1ensemble.row(r);
            ++count;
        }
        fr.row(g) /= static_cast<double>(count);
    }

    Eigen::MatrixXd decoded(3, T);
    decoded.row(0).setConstant(0.3);
    decoded.row(1) = fr.row(0) - fr.row(2);
    decoded.row(2) = fr.row(1) - fr.row(3);
    return decoded;
}

static EmulatorResult finish_reward(
    const Eigen::VectorXi& motor_perform,
    const Eigen::VectorXi& motor_expect
) {
    const int T = motor_expect.size();
    EmulatorResult out;
    out.motor_perform = motor_perform;
    out.success.resize(1, T);

    double sum = 0.0;
    int cnt = 0;
    for (int i = 0; i < T; ++i) {
        if (motor_expect(i) == 0) {
            out.success(i) = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        const double s = (motor_perform(i) == motor_expect(i)) ? 1.0 : 0.0;
        out.success(i) = s;
        sum += s;
        cnt += 1;
    }
    if (cnt == 0) {
        out.rate = std::numeric_limits<double>::quiet_NaN();
    } else {
        out.rate = sum / static_cast<double>(cnt);
    }
    return out;
}

EmulatorResult emulator_simu_exact(
    const Eigen::MatrixXd& spikes,
    const Eigen::VectorXi& motor_expect,
    const Eigen::VectorXi& indexes,
    int his,
    const std::string& model_name
) {
    if (spikes.cols() != motor_expect.size()) {
        throw std::runtime_error("emulator_simu: spikes time mismatch");
    }
    if (indexes.size() != spikes.rows()) {
        throw std::runtime_error("emulator_simu: indexes length mismatch");
    }
    if (model_name != "decodingModel_simulation") {
        throw std::runtime_error("emulator_simu: unknown model name: " + model_name);
    }

    // sortedIndices = argsort(indexes); spikes = spikes[sortedIndices,:]
    Eigen::VectorXi perm = argsort_indices(indexes);
    Eigen::MatrixXd spikes_sorted = apply_row_permutation(spikes, perm);

    Eigen::VectorXi motor = decodingModel_simulation_exact(spikes_sorted, his);

    EmulatorResult out = finish_reward(motor, motor_expect);
    out.ensemble.resize(0, 0);
    return out;
}

EmulatorResult emulator_real_manual_exact(
    const Eigen::MatrixXd& spikes,
    const Eigen::VectorXi& motor_expect,
    const Eigen::VectorXi& indexes,
    int his
) {
    if (spikes.cols() != motor_expect.size()) {
        throw std::runtime_error("emulator_real_manual: spikes time mismatch");
    }
    if (indexes.size() != spikes.rows()) {
        throw std::runtime_error("emulator_real_manual: indexes length mismatch");
    }
    if (his < 0) {
        throw std::runtime_error("emulator_real_manual: his must be >= 0");
    }

    Eigen::VectorXi perm = argsort_indices(indexes);
    Eigen::MatrixXd spikes_sorted = apply_row_permutation(spikes, perm);

    const int Ny = static_cast<int>(spikes_sorted.rows());
    const int T = static_cast<int>(spikes_sorted.cols());
    Eigen::MatrixXd ensemble((his + 1) * Ny, T);
    ensemble.setZero();

    // Python:
    // for i in range(his+1):
    //   block = hstack([zeros(Ny,i), spikes[:,:-i] if i>0 else spikes])
    for (int i = 0; i < his + 1; ++i) {
        if (i == 0) {
            ensemble.block(0, 0, Ny, T) = spikes_sorted;
        } else {
            // zeros Ny x i already
            const int cols = T - i;
            if (cols > 0) {
                ensemble.block(i * Ny, i, Ny, cols) = spikes_sorted.block(0, 0, Ny, cols);
            }
        }
    }

    Eigen::MatrixXd y = decodingModel_manual_exact(ensemble);
    Eigen::VectorXi motor(T);
    for (int t = 0; t < T; ++t) {
        Eigen::Index idx = 0;
        y.col(t).maxCoeff(&idx);
        motor(t) = static_cast<int>(idx) + 1;
    }

    EmulatorResult out = finish_reward(motor, motor_expect);
    out.ensemble = std::move(ensemble);
    return out;
}

EmulatorResult emulator_real_trained_nn_exact(
    const Eigen::MatrixXd& spikes,
    const Eigen::VectorXi& motor_expect,
    const Eigen::VectorXi& indexes,
    int his,
    const std::string& params_csv_prefix
) {
    if (spikes.cols() != motor_expect.size()) {
        throw std::runtime_error("emulator_real_trained_nn: spikes time mismatch");
    }
    if (indexes.size() != spikes.rows()) {
        throw std::runtime_error("emulator_real_trained_nn: indexes length mismatch");
    }
    if (his < 0) {
        throw std::runtime_error("emulator_real_trained_nn: his must be >= 0");
    }

    const rlpp::DecodingNnParams params = rlpp::load_decoding_nn_params_csv(params_csv_prefix);

    Eigen::VectorXi perm = argsort_indices(indexes);
    Eigen::MatrixXd spikes_sorted = apply_row_permutation(spikes, perm);

    const int Ny = static_cast<int>(spikes_sorted.rows());
    const int T = static_cast<int>(spikes_sorted.cols());
    Eigen::MatrixXd ensemble((his + 1) * Ny, T);
    ensemble.setZero();

    for (int i = 0; i < his + 1; ++i) {
        if (i == 0) {
            ensemble.block(0, 0, Ny, T) = spikes_sorted;
        } else {
            const int cols = T - i;
            if (cols > 0) {
                ensemble.block(i * Ny, i, Ny, cols) = spikes_sorted.block(0, 0, Ny, cols);
            }
        }
    }

    if (ensemble.rows() != params.xoffset.size()) {
        throw std::runtime_error("emulator_real_trained_nn: ensemble rows != decoder input dim (check his/Ny vs CSV)");
    }

    Eigen::MatrixXd y = decodingModel01_forward(
        ensemble,
        params.xoffset,
        params.gain,
        params.ymin,
        params.IW1_1,
        params.b1,
        params.LW2_1,
        params.b2
    );

    Eigen::VectorXi motor(T);
    for (int t = 0; t < T; ++t) {
        Eigen::Index idx = 0;
        y.col(t).maxCoeff(&idx);
        motor(t) = static_cast<int>(idx) + 1;
    }

    EmulatorResult out = finish_reward(motor, motor_expect);
    out.ensemble = std::move(ensemble);
    return out;
}

} // namespace rlpp

