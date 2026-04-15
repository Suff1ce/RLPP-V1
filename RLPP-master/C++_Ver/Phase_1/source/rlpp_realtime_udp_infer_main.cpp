// Real-time concurrency scaffold:
// - DAQ thread: recv UDP upstream frames -> SPSC ring buffer
// - Model thread: fixed-period tick -> RLPPInference step() -> send packed hardware_frame_v1 over UDP
// - Logging: per-tick latency stats and overflow counters

#include "F1_rat01_bundle.hpp"
#include "RLPP_inference.hpp"
#include "csv_utils.hpp"
#include "rlpp_hardware_output.hpp"
#include "spsc_ring.hpp"
#include "upstream_frame_v1.hpp"

#include <Eigen/Dense>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using sock_t = SOCKET;
static constexpr sock_t kSockInvalid = INVALID_SOCKET;
static void sock_close(sock_t s) { closesocket(s); }
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using sock_t = int;
static constexpr sock_t kSockInvalid = -1;
static void sock_close(sock_t s) { close(s); }
#endif

namespace fs = std::filesystem;
using clock_rt = std::chrono::steady_clock;

struct UpstreamMsg {
    std::uint64_t seq = 0;
    std::int32_t tbin = 0;
    std::uint64_t recv_ns = 0; // steady_clock nanoseconds since epoch (process-local)
    // Fixed maximum to avoid allocations in the queue.
    static constexpr int kMaxNx = 512;
    int Nx = 0;
    double x[kMaxNx]{};
};

static std::uint64_t steady_now_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock_rt::now().time_since_epoch()).count()
    );
}

static void print_latency_summary_us(const std::vector<long long>& us) {
    if (us.empty()) {
        std::cout << "Latency: no samples\n";
        return;
    }
    std::vector<long long> s = us;
    std::sort(s.begin(), s.end());
    auto pct = [&](double p) -> long long {
        const double idx = p * (s.size() - 1);
        const std::size_t i = static_cast<std::size_t>(idx);
        return s[i];
    };
    const long long minv = s.front();
    const long long maxv = s.back();
    const double mean = std::accumulate(s.begin(), s.end(), 0.0) / static_cast<double>(s.size());
    std::cout << "Tick latency (us):"
              << " n=" << s.size()
              << " min=" << minv
              << " mean=" << mean
              << " p50=" << pct(0.50)
              << " p95=" << pct(0.95)
              << " p99=" << pct(0.99)
              << " p999=" << pct(0.999)
              << " max=" << maxv
              << "\n";
}

static void usage() {
    std::cout
        << "Usage: RLPP_realtime_udp_infer --bundle-dir DIR [options]\n"
        << "  DAQ recv UDP -> SPSC queue -> fixed-period inference tick -> send hardware_frame_v1 over UDP.\n"
        << "\n"
        << "Required:\n"
        << "  --bundle-dir DIR        F1 bundle directory (CSV weights + refs)\n"
        << "\n"
        << "Options:\n"
        << "  --udp-in-port N         upstream UDP listen port (default 46000)\n"
        << "  --udp-out-port N        downstream UDP dest port (default 45123)\n"
        << "  --udp-out-host HOST     downstream host (default 127.0.0.1)\n"
        << "  --bin-ms N              model tick period in ms (default 5)\n"
        << "  --ticks N               number of ticks to run (default 2000)\n"
        << "  --tau-bins X            encoder tau in bins (default 150)\n"
        << "  --rng-seed N            generator sampling seed (default 0)\n"
        << "  --log-dir DIR           write latency + counters (default: <bundle-dir>/realtime_logs)\n"
        << "\n"
        << "Upstream UDP payload format:\n"
        << "  upstream_frame_v1 (see Include/upstream_frame_v1.hpp). One datagram = one time bin.\n";
}

static bool parse_u16(const std::string& s, std::uint16_t& out) {
    try {
        const int v = std::stoi(s);
        if (v < 1 || v > 65535) return false;
        out = static_cast<std::uint16_t>(v);
        return true;
    } catch (...) {
        return false;
    }
}

static void load_decoder_mapminmax_from_bundle_dir(
    const std::string& bundle_dir,
    Eigen::VectorXd& xoffset,
    Eigen::VectorXd& gain,
    double& ymin
) {
    auto file = [&](const char* name) { return (fs::path(bundle_dir) / name).string(); };
    Eigen::MatrixXd xoffset_m = load_csv_matrix(file("decoder_xoffset.csv"));
    Eigen::MatrixXd gain_m = load_csv_matrix(file("decoder_gain.csv"));
    Eigen::MatrixXd ymin_m = load_csv_matrix(file("decoder_ymin.csv"));

    auto to_vector = [](const Eigen::MatrixXd& M) -> Eigen::VectorXd {
        if (M.cols() == 1) return M.col(0);
        if (M.rows() == 1) return M.row(0).transpose();
        throw std::runtime_error("Expected a vector CSV (Nx1 or 1xN).");
    };

    xoffset = to_vector(xoffset_m);
    gain = to_vector(gain_m);
    if (ymin_m.size() != 1) throw std::runtime_error("decoder_ymin.csv must be scalar");
    ymin = ymin_m(0, 0);
}

int main(int argc, char** argv) {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        std::cerr << "WSAStartup failed\n";
        return 1;
    }
#endif

    std::string bundle_dir;
    std::uint16_t udp_in_port = 46000;
    std::uint16_t udp_out_port = 45123;
    std::string udp_out_host = "127.0.0.1";
    int bin_ms = 5;
    int ticks = 2000;
    double tau_bins = 150.0;
    unsigned int rng_seed = 0;
    std::string log_dir;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value after " + flag);
            return argv[++i];
        };
        if (a == "--help" || a == "-h") {
            usage();
#ifdef _WIN32
            WSACleanup();
#endif
            return 0;
        }
        if (a == "--bundle-dir") { bundle_dir = need(a); continue; }
        if (a == "--udp-in-port") {
            if (!parse_u16(need(a), udp_in_port)) throw std::runtime_error("bad --udp-in-port");
            continue;
        }
        if (a == "--udp-out-port") {
            if (!parse_u16(need(a), udp_out_port)) throw std::runtime_error("bad --udp-out-port");
            continue;
        }
        if (a == "--udp-out-host") { udp_out_host = need(a); continue; }
        if (a == "--bin-ms") { bin_ms = std::stoi(need(a)); continue; }
        if (a == "--ticks") { ticks = std::stoi(need(a)); continue; }
        if (a == "--tau-bins") { tau_bins = std::stod(need(a)); continue; }
        if (a == "--rng-seed") { rng_seed = static_cast<unsigned int>(std::stoul(need(a))); continue; }
        if (a == "--log-dir") { log_dir = need(a); continue; }
        throw std::runtime_error("Unknown arg: " + a);
    }

    if (bundle_dir.empty()) {
        std::cerr << "Required: --bundle-dir\n";
#ifdef _WIN32
        WSACleanup();
#endif
        return 1;
    }
    if (!fs::is_directory(bundle_dir)) {
        std::cerr << "bundle_dir is not a directory: " << bundle_dir << "\n";
#ifdef _WIN32
        WSACleanup();
#endif
        return 1;
    }
    if (log_dir.empty()) {
        log_dir = (fs::path(bundle_dir) / "realtime_logs").string();
    }
    fs::create_directories(fs::path(log_dir));

    // ---- Load model from bundle ----
    F1Bundle bundle = load_f1_bundle(bundle_dir);
    Eigen::VectorXd dec_xoffset, dec_gain;
    double dec_ymin = 0.0;
    load_decoder_mapminmax_from_bundle_dir(bundle_dir, dec_xoffset, dec_gain, dec_ymin);

    const int Nx = static_cast<int>(bundle.upstream_spikes.cols());
    const int feature_dim = static_cast<int>(bundle.encoder_features_ref.rows());
    if (feature_dim % Nx != 0) throw std::runtime_error("bundle feature_dim not divisible by Nx");
    const int H = feature_dim / Nx;
    const int Ny = TwoLayerMLP(bundle.generator_W1, bundle.generator_W2).output_dim();
    if (bundle.decoder_features_ref.rows() % Ny != 0) throw std::runtime_error("decoder_features_ref rows not divisible by Ny");
    const int num_lags = static_cast<int>(bundle.decoder_features_ref.rows() / Ny);

    RLPPInferenceConfig icfg;
    icfg.tau_bins = tau_bins;
    icfg.spike_mode = SpikeDriveMode::SampledBernoulli;
    icfg.rng_seed = rng_seed;

    RLPPInference inf(
        Nx, H, Ny, num_lags,
        bundle.sorted_indices_1based,
        bundle.generator_W1, bundle.generator_W2,
        dec_xoffset, dec_gain, dec_ymin,
        bundle.decoder_W1, bundle.decoder_b1,
        bundle.decoder_W2, bundle.decoder_b2,
        icfg
    );

    const int num_labels = static_cast<int>(bundle.decoder_W2.rows());
    if (num_labels <= 0) {
        throw std::runtime_error("bundle missing decoder_W2.csv (required for realtime UDP inference)");
    }

    // ---- UDP sockets ----
    sock_t sock_in = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_in == kSockInvalid) throw std::runtime_error("socket() failed (in)");
    int yes = 1;
#ifdef _WIN32
    setsockopt(sock_in, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
#else
    setsockopt(sock_in, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#endif
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(udp_in_port);
    addr.sin_addr.s_addr = htonl(0x7F000001u); // 127.0.0.1
    if (bind(sock_in, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        sock_close(sock_in);
        throw std::runtime_error("bind failed (udp-in-port in use?)");
    }

    sock_t sock_out = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_out == kSockInvalid) {
        sock_close(sock_in);
        throw std::runtime_error("socket() failed (out)");
    }
    sockaddr_in dest{};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(udp_out_port);
    if (inet_pton(AF_INET, udp_out_host.c_str(), &dest.sin_addr) != 1) {
        sock_close(sock_out);
        sock_close(sock_in);
        throw std::runtime_error("bad --udp-out-host");
    }

    // ---- SPSC queue ----
    // NOTE: keep this off the stack (UpstreamMsg is large) to avoid stack overflow on Windows.
    static constexpr std::size_t kQueueCap = 1024;
    auto q = std::make_unique<SpscRing<UpstreamMsg, kQueueCap>>();
    std::atomic<bool> stop{false};
    std::atomic<long long> dropped_full{0};
    std::atomic<long long> dropped_bad{0};
    std::atomic<long long> dropped_oversize{0};
    std::atomic<long long> received_ok{0};

    // ---- DAQ thread ----
    std::thread daq([&]() {
        std::vector<std::uint8_t> buf(65536);
        while (!stop.load(std::memory_order_relaxed)) {
#ifdef _WIN32
            sockaddr_in peer{};
            int peerlen = sizeof(peer);
            const int n = recvfrom(
                sock_in,
                reinterpret_cast<char*>(buf.data()),
                static_cast<int>(buf.size()),
                0,
                reinterpret_cast<sockaddr*>(&peer),
                &peerlen
            );
            if (n <= 0) continue;
            const std::size_t len = static_cast<std::size_t>(n);
#else
            sockaddr_in peer{};
            socklen_t peerlen = sizeof(peer);
            const ssize_t n = recvfrom(sock_in, buf.data(), buf.size(), 0, reinterpret_cast<sockaddr*>(&peer), &peerlen);
            if (n <= 0) continue;
            const std::size_t len = static_cast<std::size_t>(n);
#endif

            UpstreamFrameV1 fr;
            if (!unpack_upstream_frame_v1(buf.data(), len, fr)) {
                dropped_bad.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (fr.Nx != Nx) {
                dropped_bad.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (fr.Nx > UpstreamMsg::kMaxNx) {
                dropped_oversize.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            UpstreamMsg m;
            m.seq = fr.sequence;
            m.tbin = fr.time_bin_1based;
            m.recv_ns = steady_now_ns();
            m.Nx = fr.Nx;
            for (int i = 0; i < fr.Nx; ++i) m.x[i] = fr.upstream[static_cast<std::size_t>(i)];

            if (!q->push(m)) {
                dropped_full.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            received_ok.fetch_add(1, std::memory_order_relaxed);
        }
    });

    // ---- Model thread (fixed-period tick) ----
    std::vector<long long> tick_latency_us;
    tick_latency_us.reserve(static_cast<std::size_t>(ticks));

    std::uint64_t out_seq = 0;
    std::int32_t tbin = 0;

    // stash latest message per bin
    UpstreamMsg latest{};
    bool have_latest = false;

    const auto t_start = clock_rt::now();
    auto next_tick = t_start;

    std::vector<std::uint8_t> packed;
    for (int i = 0; i < ticks; ++i) {
        next_tick += std::chrono::milliseconds(bin_ms);
        std::this_thread::sleep_until(next_tick);

        // Drain queue; keep the newest message whose tbin <= current.
        ++tbin;
        UpstreamMsg m{};
        while (q->pop(m)) {
            if (m.tbin <= tbin) {
                latest = m;
                have_latest = true;
            } else {
                // Future message (clock skew / sender ahead). Keep it by placing into latest only when caught up.
                // For simplicity, we drop it now; sender should align tbin with binning clock.
            }
        }

        // Build upstream vector; if missing, fill zeros (valid=false will happen naturally until encoder warmup).
        Eigen::VectorXd u(Nx);
        if (have_latest && latest.tbin == tbin && latest.Nx == Nx) {
            for (int k = 0; k < Nx; ++k) u(k) = latest.x[k];
        } else {
            u.setZero();
        }

        const auto t0 = clock_rt::now();
        RLPPInferenceStepOutput o = inf.step(u, tbin);
        if (!o.valid && o.decoder_y.size() == 0) {
            o.decoder_y = Eigen::VectorXd::Zero(num_labels);
        }
        const auto t1 = clock_rt::now();

        const long long us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        tick_latency_us.push_back(us);

        // Pack + send hardware frame (even if invalid, packer will zero logits).
        pack_rlpp_hardware_frame_v1(o, tbin, out_seq++, packed);
#ifdef _WIN32
        const int sent = sendto(
            sock_out,
            reinterpret_cast<const char*>(packed.data()),
            static_cast<int>(packed.size()),
            0,
            reinterpret_cast<sockaddr*>(&dest),
            static_cast<int>(sizeof(dest))
        );
        (void)sent;
#else
        sendto(sock_out, packed.data(), packed.size(), 0, reinterpret_cast<sockaddr*>(&dest), sizeof(dest));
#endif
    }

    stop.store(true, std::memory_order_relaxed);
    sock_close(sock_in); // unblock recv
    daq.join();
    sock_close(sock_out);

    // ---- Write logs ----
    {
        std::ofstream f(fs::path(log_dir) / "latency_us_tick.csv");
        for (long long v : tick_latency_us) f << v << "\n";
    }
    {
        std::ofstream f(fs::path(log_dir) / "counters.csv");
        f << "received_ok," << received_ok.load() << "\n";
        f << "dropped_full," << dropped_full.load() << "\n";
        f << "dropped_bad," << dropped_bad.load() << "\n";
        f << "dropped_oversize," << dropped_oversize.load() << "\n";
    }

    print_latency_summary_us(tick_latency_us);
    std::cout << "Logs: " << log_dir << "\n";

#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}

