#pragma once

#include <Eigen/Dense>
#include <string>

namespace rlpp {

/// Weights + mapminmax settings matching Python decodingModel_01 / decodingModel_02 (from exported CSV).
struct DecodingNnParams {
    Eigen::VectorXd xoffset;
    Eigen::VectorXd gain;
    double ymin = 0.0;
    Eigen::VectorXd b1;
    Eigen::MatrixXd IW1_1;
    Eigen::VectorXd b2;
    Eigen::MatrixXd LW2_1;
};

/// Load `{prefix}_xoffset.csv`, `{prefix}_gain.csv`, `{prefix}_ymin.csv`, `{prefix}_b1.csv`,
/// `{prefix}_IW1_1.csv`, `{prefix}_b2.csv`, `{prefix}_LW2_1.csv` (prefix has no trailing underscore).
DecodingNnParams load_decoding_nn_params_csv(const std::string& path_prefix);

} // namespace rlpp
