#pragma once

#include <string>
#include <Eigen/Dense>

Eigen::MatrixXd load_csv_matrix(const std::string& path);
Eigen::VectorXd load_csv_vector(const std::string& path);

void save_csv_matrix(const std::string& path, const Eigen::MatrixXd& matrix);
void save_csv_vector(const std::string& path, const Eigen::VectorXd& vector);