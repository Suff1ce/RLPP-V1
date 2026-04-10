#pragma once

#include <Eigen/Dense>
#include <random>

double sigmoid_scalar(double x);
Eigen::VectorXd sigmoid_vector(const Eigen::VectorXd& v);
Eigen::MatrixXd sigmoid_matrix(const Eigen::MatrixXd& M);
int argmax_index(const Eigen::VectorXd& v);
Eigen::VectorXd sample_Bernoulli(const Eigen::VectorXd& probabilities, std::mt19937& rng);