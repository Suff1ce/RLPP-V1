#include "rlpp_decoding_params.hpp"

#include "csv_utils.hpp"

#include <stdexcept>

namespace rlpp {

DecodingNnParams load_decoding_nn_params_csv(const std::string& path_prefix) {
    DecodingNnParams p;
    p.xoffset = load_csv_vector(path_prefix + "_xoffset.csv");
    p.gain = load_csv_vector(path_prefix + "_gain.csv");
    const Eigen::MatrixXd ymin_m = load_csv_matrix(path_prefix + "_ymin.csv");
    if (ymin_m.size() != 1) {
        throw std::runtime_error("decoding params: ymin must be a single value");
    }
    p.ymin = ymin_m(0, 0);
    p.b1 = load_csv_vector(path_prefix + "_b1.csv");
    p.IW1_1 = load_csv_matrix(path_prefix + "_IW1_1.csv");
    p.b2 = load_csv_vector(path_prefix + "_b2.csv");
    p.LW2_1 = load_csv_matrix(path_prefix + "_LW2_1.csv");

    const int in_dim = static_cast<int>(p.xoffset.size());
    if (p.gain.size() != in_dim) {
        throw std::runtime_error("decoding params: gain size mismatch");
    }
    if (p.IW1_1.cols() != in_dim) {
        throw std::runtime_error("decoding params: IW1_1 cols vs xoffset");
    }
    if (p.b1.size() != p.IW1_1.rows()) {
        throw std::runtime_error("decoding params: b1 vs IW1_1 rows");
    }
    if (p.LW2_1.cols() != p.IW1_1.rows()) {
        throw std::runtime_error("decoding params: LW2_1 cols vs hidden");
    }
    if (p.b2.size() != p.LW2_1.rows()) {
        throw std::runtime_error("decoding params: b2 vs LW2_1 rows");
    }
    return p;
}

} // namespace rlpp
