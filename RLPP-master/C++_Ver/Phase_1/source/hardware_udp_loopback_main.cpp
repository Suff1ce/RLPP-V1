// UDP loopback transport test: send each hardware_trace_v1 frame as one datagram to 127.0.0.1
// and receive/parse with unpack_rlpp_hardware_frame_v1 (device-side parser path).

#include "hardware_frame_v1.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
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
static void sock_close(sock_t s) {
    closesocket(s);
}
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
using sock_t = int;
static constexpr sock_t kSockInvalid = -1;
static void sock_close(sock_t s) {
    close(s);
}
#endif

static bool read_file_all(const std::string& path, std::vector<std::uint8_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return false;
    }
    const auto sz = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    out.resize(sz);
    if (sz > 0) {
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::ptrdiff_t>(sz));
    }
    return static_cast<bool>(f);
}

static bool send_trace_udp(
    const std::vector<std::uint8_t>& trace,
    std::uint16_t port,
    std::atomic<int>* send_errors
) {
    sock_t s = socket(AF_INET, SOCK_DGRAM, 0);
    if (s == kSockInvalid) {
        std::cerr << "socket() failed\n";
        return false;
    }

    sockaddr_in dest{};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(port);
    dest.sin_addr.s_addr = htonl(0x7F000001u); // 127.0.0.1

    std::size_t off = 0;
    int frame_idx = 0;
    while (off < trace.size()) {
        [[maybe_unused]] RlppHardwareFrameV1Unpacked u;
        std::size_t consumed = 0;
        if (!unpack_rlpp_hardware_frame_v1(trace.data() + off, trace.size() - off, &consumed, u)) {
            std::cerr << "send: bad frame at offset " << off << "\n";
            (*send_errors)++;
            break;
        }
#ifdef _WIN32
        const int r = sendto(
            s,
            reinterpret_cast<const char*>(trace.data() + off),
            static_cast<int>(consumed),
            0,
            reinterpret_cast<sockaddr*>(&dest),
            static_cast<int>(sizeof(dest))
        );
        if (r != static_cast<int>(consumed)) {
            std::cerr << "sendto failed at frame " << frame_idx << "\n";
            (*send_errors)++;
            sock_close(s);
            return false;
        }
#else
        const ssize_t r = sendto(
            s,
            trace.data() + off,
            consumed,
            0,
            reinterpret_cast<sockaddr*>(&dest),
            sizeof(dest)
        );
        if (r != static_cast<ssize_t>(consumed)) {
            std::cerr << "sendto failed at frame " << frame_idx << "\n";
            (*send_errors)++;
            sock_close(s);
            return false;
        }
#endif
        off += consumed;
        ++frame_idx;
    }

    sock_close(s);
    return true;
}

static void recv_loop_udp(std::uint16_t port, int expected_frames, int first_timeout_sec, std::atomic<int>* ok_count) {
    sock_t r = socket(AF_INET, SOCK_DGRAM, 0);
    if (r == kSockInvalid) {
        return;
    }

    int yes = 1;
#ifdef _WIN32
    setsockopt(r, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
#else
    setsockopt(r, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#endif

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(0x7F000001u);

    if (bind(r, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::cerr << "bind failed (is port in use?)\n";
        sock_close(r);
        return;
    }

    std::vector<std::uint8_t> buf(65536);
    int got = 0;
    while (got < expected_frames) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(r, &rfds);
        timeval tv{};
        tv.tv_sec = (got == 0) ? first_timeout_sec : 5;
        tv.tv_usec = 0;
#ifdef _WIN32
        const int sel = select(0, &rfds, nullptr, nullptr, &tv);
#else
        const int sel = select(static_cast<int>(r) + 1, &rfds, nullptr, nullptr, &tv);
#endif
        if (sel <= 0) {
            std::cerr << "recv timeout or error (got " << got << " / " << expected_frames << ")\n";
            break;
        }

        sockaddr_in peer{};
#ifdef _WIN32
        int peerlen = sizeof(peer);
        const int n = recvfrom(
            r,
            reinterpret_cast<char*>(buf.data()),
            static_cast<int>(buf.size()),
            0,
            reinterpret_cast<sockaddr*>(&peer),
            &peerlen
        );
#else
        socklen_t peerlen = sizeof(peer);
        const ssize_t n = recvfrom(
            r,
            buf.data(),
            buf.size(),
            0,
            reinterpret_cast<sockaddr*>(&peer),
            &peerlen
        );
#endif
        if (n <= 0) {
            break;
        }

        RlppHardwareFrameV1Unpacked u;
        std::size_t consumed = 0;
        if (!unpack_rlpp_hardware_frame_v1(buf.data(), static_cast<std::size_t>(n), &consumed, u)) {
            std::cerr << "recv: unpack failed (len=" << n << ")\n";
            break;
        }
        if (consumed != static_cast<std::size_t>(n)) {
            std::cerr << "recv: datagram had extra bytes\n";
            break;
        }
        ++got;
    }

    *ok_count = got;
    sock_close(r);
}

int main(int argc, char** argv) {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        std::cerr << "WSAStartup failed\n";
        return 1;
    }
#endif

    std::string trace_path;
    std::uint16_t port = 45123;
    int first_timeout_sec = 30;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            std::cout << "Usage: RLPP_hardware_udp_loopback --trace PATH [--port N] [--first-timeout-sec N]\n"
                         "  Sends each frame from hardware_trace_v1.bin as one UDP datagram to 127.0.0.1\n"
                         "  and receives them on the same port (loopback transport + device parser test).\n";
#ifdef _WIN32
            WSACleanup();
#endif
            return 0;
        }
        if (a == "--trace" && i + 1 < argc) {
            trace_path = argv[++i];
            continue;
        }
        if (a == "--port" && i + 1 < argc) {
            port = static_cast<std::uint16_t>(std::stoi(argv[++i]));
            continue;
        }
        if (a == "--first-timeout-sec" && i + 1 < argc) {
            first_timeout_sec = std::stoi(argv[++i]);
            continue;
        }
        std::cerr << "Unknown arg: " << a << " (try --help)\n";
#ifdef _WIN32
        WSACleanup();
#endif
        return 1;
    }

    if (trace_path.empty()) {
        std::cerr << "Required: --trace PATH\n";
#ifdef _WIN32
        WSACleanup();
#endif
        return 1;
    }

    std::vector<std::uint8_t> trace;
    if (!read_file_all(trace_path, trace)) {
        std::cerr << "Cannot read trace file: " << trace_path << "\n";
#ifdef _WIN32
        WSACleanup();
#endif
        return 1;
    }

    const std::size_t nframes = count_rlpp_hardware_frames_v1(trace.data(), trace.size());
    if (nframes == 0 && !trace.empty()) {
        std::cerr << "No valid frames in file (or corrupt)\n";
#ifdef _WIN32
        WSACleanup();
#endif
        return 1;
    }

    std::cout << "frames=" << nframes << " bytes=" << trace.size() << " port=" << port << "\n";

    std::atomic<int> ok_recv{0};
    std::atomic<int> send_err{0};

    std::thread recv_thr([&]() {
        recv_loop_udp(port, static_cast<int>(nframes), first_timeout_sec, &ok_recv);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    if (!send_trace_udp(trace, port, &send_err)) {
        recv_thr.join();
#ifdef _WIN32
        WSACleanup();
#endif
        return 1;
    }

    recv_thr.join();

#ifdef _WIN32
    WSACleanup();
#endif

    if (send_err.load() != 0) {
        std::cerr << "UDP_LOOPBACK FAIL (send)\n";
        return 1;
    }
    if (static_cast<std::size_t>(ok_recv.load()) != nframes) {
        std::cerr << "UDP_LOOPBACK FAIL: received " << ok_recv.load() << " expected " << nframes << "\n";
        return 1;
    }

    std::cout << "UDP_LOOPBACK OK (transport + unpack)\n";
    return 0;
}
