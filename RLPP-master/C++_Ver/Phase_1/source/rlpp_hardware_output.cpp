#include "rlpp_hardware_output.hpp"
#include "hardware_frame_v1.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

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
    for (int i = 0; i < 8; ++i) {
        o.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xff));
    }
}

static void append_le_i32(std::vector<std::uint8_t>& o, std::int32_t v) {
    append_le_u32(o, static_cast<std::uint32_t>(v));
}

static void append_le_f64(std::vector<std::uint8_t>& o, double x) {
    static_assert(sizeof(double) == 8, "expected IEEE-754 binary64");
    unsigned char buf[8];
    std::memcpy(buf, &x, 8);
    for (int i = 0; i < 8; ++i) {
        o.push_back(buf[i]);
    }
}

void write_rlpp_hardware_io_contract_text(const std::string& path, int num_labels) {
    std::ofstream f(path);
    if (!f) {
        throw std::runtime_error("write_rlpp_hardware_io_contract_text: cannot open " + path);
    }
    f << "RLPP Phase 2.5 - hardware I/O contract (inference snapshot only)\n"
         "================================================================\n"
         "This describes the binary layout produced by pack_rlpp_hardware_frame_v1().\n"
         "Purpose: fixed-size message for logging, replay verification, or a thin device API.\n"
         "It is NOT a multi-channel stimulation waveform; downstream firmware maps labels/logits to stim.\n\n"
         "Endianness: little-endian for all multi-byte fields (typical x86/ARM).\n"
         "Floating point: IEEE-754 binary64 (double), native host byte order (LE on LE hosts).\n\n"
         "Frame v1 layout (byte offset, size, field):\n"
         "  0   4  magic u32  0x31504C52 ('RLP1' LE)\n"
         "  4   2  version u16  " << kRlppHardwareFrameVersion << "\n"
         "  6   2  flags u16    reserved, set 0\n"
         "  8   8  sequence u64 monotonic index of valid output bins (caller-defined)\n"
         "  16  4  time_bin_1based i32  encoder time t\n"
         "  20  4  valid i32      1 if encoder produced a step this bin, else 0\n"
         "  24  4  label_1based i32  argmax class 1..K when valid, else 0\n"
         "  28  4  num_labels i32  K (must match this export: " << num_labels << ")\n"
         "  32  8*K  decoder_y[0..K-1] double softmax probabilities; zeros if invalid\n\n"
         "Total size: 32 + 8*K bytes.\n\n"
         "Concatenated trace file (replay_runner when dump_hardware_trace_v1 is true):\n"
         "  Filename: hardware_trace_v1.bin in the replay out_dir.\n"
         "  Content: frames back-to-back in recorded time order (one frame per recorded valid bin).\n"
         "  sequence field: 0-based index within the replay window (matches CSV column order).\n\n"
         "UDP transport test (optional):\n"
         "  C++: RLPP_hardware_udp_loopback --trace hardware_trace_v1.bin [--port N]\n"
         "  Python: hardware_trace_udp_loopback.py --trace ... [--port N]\n"
         "  Sends one datagram per frame to 127.0.0.1 and parses with the same v1 unpack (device-side path).\n";
}

void pack_rlpp_hardware_frame_v1(
    const RLPPInferenceStepOutput& step,
    int time_bin_1based,
    std::uint64_t sequence,
    std::vector<std::uint8_t>& out
) {
    const int K = static_cast<int>(step.decoder_y.size());
    if (K <= 0) {
        throw std::runtime_error("pack_rlpp_hardware_frame_v1: empty decoder_y");
    }

    out.clear();
    out.reserve(32 + 8 * K);

    append_le_u32(out, kRlppHardwareFrameMagicLe);
    append_le_u16(out, kRlppHardwareFrameVersion);
    append_le_u16(out, 0); // flags
    append_le_u64(out, sequence);
    append_le_i32(out, static_cast<std::int32_t>(time_bin_1based));
    append_le_i32(out, step.valid ? 1 : 0);
    append_le_i32(out, step.valid ? static_cast<std::int32_t>(step.label_1based) : 0);
    append_le_i32(out, static_cast<std::int32_t>(K));

    for (int i = 0; i < K; ++i) {
        double v = 0.0;
        if (step.valid && i < step.decoder_y.size()) {
            v = step.decoder_y(i);
        }
        append_le_f64(out, v);
    }
}
