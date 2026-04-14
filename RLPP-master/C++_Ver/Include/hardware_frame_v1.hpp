#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/// Binary layout must match pack_rlpp_hardware_frame_v1() in rlpp_hardware_output.cpp
constexpr std::uint32_t kRlppHardwareFrameMagicLe = 0x31504C52u;
constexpr std::uint16_t kRlppHardwareFrameVersion = 1;

struct RlppHardwareFrameV1Unpacked {
    std::uint64_t sequence = 0;
    std::int32_t time_bin_1based = 0;
    std::int32_t valid = 0;
    std::int32_t label_1based = 0;
    std::int32_t num_labels = 0;
    std::vector<double> logits;
};

/// Device-side parse: one frame from buffer. Returns false if truncated or invalid (bad magic/version/K).
bool unpack_rlpp_hardware_frame_v1(
    const std::uint8_t* data,
    std::size_t len,
    std::size_t* bytes_consumed,
    RlppHardwareFrameV1Unpacked& out
);

/// Count frames in a concatenated trace (e.g. hardware_trace_v1.bin).
std::size_t count_rlpp_hardware_frames_v1(const std::uint8_t* data, std::size_t len);
