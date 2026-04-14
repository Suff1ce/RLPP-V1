#include "hardware_frame_v1.hpp"

#include <cstring>

static std::uint32_t read_le_u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

static std::uint16_t read_le_u16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0]) | (static_cast<std::uint16_t>(p[1]) << 8);
}

static std::uint64_t read_le_u64(const std::uint8_t* p) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    }
    return v;
}

static std::int32_t read_le_i32(const std::uint8_t* p) {
    std::uint32_t u = read_le_u32(p);
    return static_cast<std::int32_t>(u);
}

static double read_le_f64(const std::uint8_t* p) {
    static_assert(sizeof(double) == 8, "expected IEEE-754 binary64");
    double x;
    std::memcpy(&x, p, 8);
    return x;
}

bool unpack_rlpp_hardware_frame_v1(
    const std::uint8_t* data,
    std::size_t len,
    std::size_t* bytes_consumed,
    RlppHardwareFrameV1Unpacked& out
) {
    if (len < 32 || bytes_consumed == nullptr) {
        return false;
    }

    const std::uint32_t magic = read_le_u32(data + 0);
    if (magic != kRlppHardwareFrameMagicLe) {
        return false;
    }
    const std::uint16_t ver = read_le_u16(data + 4);
    if (ver != kRlppHardwareFrameVersion) {
        return false;
    }
    /* std::uint16_t flags = read_le_u16(data + 6); */
    out.sequence = read_le_u64(data + 8);
    out.time_bin_1based = read_le_i32(data + 16);
    out.valid = read_le_i32(data + 20);
    out.label_1based = read_le_i32(data + 24);
    out.num_labels = read_le_i32(data + 28);

    const int K = out.num_labels;
    if (K <= 0 || K > 65536) {
        return false;
    }
    const std::size_t need = 32 + static_cast<std::size_t>(K) * 8u;
    if (len < need) {
        return false;
    }

    out.logits.resize(static_cast<std::size_t>(K));
    for (int i = 0; i < K; ++i) {
        out.logits[static_cast<std::size_t>(i)] = read_le_f64(data + 32 + i * 8);
    }

    *bytes_consumed = need;
    return true;
}

std::size_t count_rlpp_hardware_frames_v1(const std::uint8_t* data, std::size_t len) {
    std::size_t off = 0;
    std::size_t n = 0;
    while (off < len) {
        RlppHardwareFrameV1Unpacked u;
        std::size_t consumed = 0;
        if (!unpack_rlpp_hardware_frame_v1(data + off, len - off, &consumed, u)) {
            break;
        }
        off += consumed;
        ++n;
    }
    return n;
}
