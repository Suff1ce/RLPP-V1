#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "RLPP_inference.hpp"
#include "hardware_frame_v1.hpp"

/// Fixed-size binary snapshot of one inference step for host/device handoff (Phase 2.5 contract).
/// Not a full stimulation waveform; map `label_1based` / logits to your stim stack downstream.
/// Magic/version: see hardware_frame_v1.hpp (unpack for device-side parsing).

void write_rlpp_hardware_io_contract_text(const std::string& path, int num_labels);

/// Append one v1 frame (little-endian). When `step.valid` is false, logits are written as zeros.
void pack_rlpp_hardware_frame_v1(
    const RLPPInferenceStepOutput& step,
    int time_bin_1based,
    std::uint64_t sequence,
    std::vector<std::uint8_t>& out
);
