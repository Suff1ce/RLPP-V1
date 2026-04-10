#pragma once

#include <Eigen/Dense>

Eigen::MatrixXd decodingModel01_forward(const Eigen::MatrixXd& ensemble,
                                        const Eigen::VectorXd& xoffset,
                                        const Eigen::VectorXd& gain,
                                        double ymin,
                                        const Eigen::MatrixXd& IW1_1,
                                        const Eigen::VectorXd& b1,
                                        const Eigen::MatrixXd& LW2_1,
                                        const Eigen::VectorXd& b2);