#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <type_traits>

// Simple lock-free single-producer/single-consumer ring buffer.
// - Fixed capacity (compile-time).
// - Non-blocking: push/pop return false on full/empty.
// - Intended for real-time hot paths (no allocations, no locks).
//
// NOTE: Capacity must be >= 2.
template <class T, std::size_t Capacity>
class SpscRing {
    static_assert(Capacity >= 2, "SpscRing capacity must be >= 2");
    static_assert(std::is_trivially_copyable_v<T>,
                  "SpscRing requires trivially copyable T (to avoid lifetime complexity)");

public:
    SpscRing() = default;

    SpscRing(const SpscRing&) = delete;
    SpscRing& operator=(const SpscRing&) = delete;

    bool push(const T& v) noexcept {
        const std::size_t h = head_.load(std::memory_order_relaxed);
        const std::size_t next = inc(h);
        if (next == tail_.load(std::memory_order_acquire)) {
            return false; // full
        }
        buf_[h] = v;
        head_.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T& out) noexcept {
        const std::size_t t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire)) {
            return false; // empty
        }
        out = buf_[t];
        tail_.store(inc(t), std::memory_order_release);
        return true;
    }

    // Consumer-only convenience: drain up to max_n items.
    template <class Fn>
    std::size_t drain(Fn&& fn, std::size_t max_n = static_cast<std::size_t>(-1)) noexcept {
        std::size_t n = 0;
        T tmp{};
        while (n < max_n && pop(tmp)) {
            fn(tmp);
            ++n;
        }
        return n;
    }

private:
    static constexpr std::size_t inc(std::size_t i) noexcept { return (i + 1) % Capacity; }

    alignas(64) std::array<T, Capacity> buf_{};
    alignas(64) std::atomic<std::size_t> head_{0}; // producer writes
    alignas(64) std::atomic<std::size_t> tail_{0}; // consumer writes
};

