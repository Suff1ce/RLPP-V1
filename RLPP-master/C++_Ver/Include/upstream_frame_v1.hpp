#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// Upstream DAQ -> model UDP payload (v1).
// This is intentionally small and easy to parse. One datagram = one time bin.
//
// Endianness: little-endian for all multi-byte fields.
// Floating point: IEEE-754 binary64 for values.
//
// Layout:
//   0   4  magic u32  'UPV1' LE (0x31565055)
//   4   2  version u16 1
//   6   2  flags u16   reserved = 0
//   8   8  sequence u64 (monotonic per datagram)
//   16  4  time_bin_1based i32
//   20  4  Nx i32 (number of upstream channels/features)
//   24  8*Nx  upstream_bin[0..Nx-1] double
//
// Values represent the upstream "spike count" (or rate) for that bin; caller-defined.
constexpr std::uint32_t kUpstreamFrameV1MagicLe = 0x31565055u; // 'UPV1' little-endian
constexpr std::uint16_t kUpstreamFrameV1Version = 1;

struct UpstreamFrameV1 {
    std::uint64_t sequence = 0;
    std::int32_t time_bin_1based = 0;
    std::int32_t Nx = 0;
    std::vector<double> upstream;
};

bool unpack_upstream_frame_v1(
    const std::uint8_t* data,
    std::size_t len,
    UpstreamFrameV1& out
);

// Helper to build a datagram.
std::vector<std::uint8_t> pack_upstream_frame_v1(
    std::uint64_t sequence,
    std::int32_t time_bin_1based,
    const std::vector<double>& upstream
);

