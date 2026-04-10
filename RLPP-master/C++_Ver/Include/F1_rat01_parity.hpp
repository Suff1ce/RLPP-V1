#pragma once

#include "F1_rat01_bundle.hpp"

void run_encoder_parity_or_throw(const F1Bundle& bundle);

void run_generator_parity_or_throw(const F1Bundle& bundle);

void run_decoder_feature_parity_or_throw(const F1Bundle& bundle);

void run_decoder_output_model01_parity_or_throw(const F1Bundle& bundle,
                                                const Eigen::VectorXd& xoffset,
                                                const Eigen::VectorXd& gain,
                                                double ymin);