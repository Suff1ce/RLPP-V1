#include "upstream_frame_v1.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

static std::uint16_t read_le_u16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (static_cast<std::uint16_t>(p[1]) << 8));
}
static std::uint32_t read_le_u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0] |
                                      (static_cast<std::uint32_t>(p[1]) << 8) |
                                      (static_cast<std::uint32_t>(p[2]) << 16) |
                                      (static_cast<std::uint32_t>(p[3]) << 24));
}
static std::uint64_t read_le_u64(const std::uint8_t* p) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= (static_cast<std::uint64_t>(p[i]) << (8 * i));
    return v;
}
static std::int32_t read_le_i32(const std::uint8_t* p) {
    return static_cast<std::int32_t>(read_le_u32(p));
}

static void append_le_u16(std::vector<std::uint8_t>& o, std::uint16_t v) {
    o.push_back(static_cast<std::uint8_t>(v & 0xff));
    o.push_back(static_cast<std::uint8_t>((v >> 8) & 0xff));
}
static void append_le_u32(std::vector<std::uint8_t>& o, std::uint32_t v) {
    o.push_back(static_cast<std::uint8_t>(v & 0xff));
    o.push_back(static_cast<std::uint8_t>((v >> 8) & 0xff));
    o.push_back(static_cast<std::uint8_t>((v >> 16) & 0xff));
    o.push_back(static_cast<std::uint8_t>((v >> 24) & 0xff));
}
static void append_le_u64(std::vector<std::uint8_t>& o, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) o.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xff));
}
static void append_le_i32(std::vector<std::uint8_t>& o, std::int32_t v) {
    append_le_u32(o, static_cast<std::uint32_t>(v));
}
static void append_le_f64(std::vector<std::uint8_t>& o, double x) {
    static_assert(sizeof(double) == 8, "expected IEEE-754 binary64");
    std::uint8_t buf[8];
    std::memcpy(buf, &x, 8);
    o.insert(o.end(), buf, buf + 8);
}

bool unpack_upstream_frame_v1(const std::uint8_t* data, std::size_t len, UpstreamFrameV1& out) {
    if (!data || len < 24) return false;
    const std::uint32_t magic = read_le_u32(data + 0);
    const std::uint16_t ver = read_le_u16(data + 4);
    if (magic != kUpstreamFrameV1MagicLe || ver != kUpstreamFrameV1Version) return false;

    const std::uint64_t seq = read_le_u64(data + 8);
    const std::int32_t tbin = read_le_i32(data + 16);
    const std::int32_t Nx = read_le_i32(data + 20);
    if (Nx <= 0) return false;

    const std::size_t need = 24 + 8ull * static_cast<std::size_t>(Nx);
    if (len != need) return false; // enforce one-datagram == one-frame exact length

    out.sequence = seq;
    out.time_bin_1based = tbin;
    out.Nx = Nx;
    out.upstream.resize(static_cast<std::size_t>(Nx));
    const std::uint8_t* p = data + 24;
    for (int i = 0; i < Nx; ++i) {
        double x = 0.0;
        std::memcpy(&x, p + 8ull * static_cast<std::size_t>(i), 8);
        out.upstream[static_cast<std::size_t>(i)] = x;
    }
    return true;
}

std::vector<std::uint8_t> pack_upstream_frame_v1(
    std::uint64_t sequence,
    std::int32_t time_bin_1based,
    const std::vector<double>& upstream
) {
    if (upstream.empty()) {
        throw std::runtime_error("pack_upstream_frame_v1: upstream empty");
    }
    if (upstream.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("pack_upstream_frame_v1: upstream too large");
    }
    const std::int32_t Nx = static_cast<std::int32_t>(upstream.size());

    std::vector<std::uint8_t> o;
    o.reserve(24 + 8ull * upstream.size());
    append_le_u32(o, kUpstreamFrameV1MagicLe);
    append_le_u16(o, kUpstreamFrameV1Version);
    append_le_u16(o, 0);
    append_le_u64(o, sequence);
    append_le_i32(o, time_bin_1based);
    append_le_i32(o, Nx);
    for (double x : upstream) append_le_f64(o, x);
    return o;
}

