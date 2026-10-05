// multithreaded streaming bandwidth probe
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>

static double now_s() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    const size_t N = (argc > 1) ? (size_t) atoll(argv[1]) : (size_t) 256 * 1024 * 1024; // elements (uint64)
    const int T = (argc > 2) ? atoi(argv[2]) : 6;
    const int iters = (argc > 3) ? atoi(argv[3]) : 10;
    std::vector<uint64_t> a(N), b(N);
    for (size_t i = 0; i < N; i++) { a[i] = i * 2654435761u; b[i] = i + 12345; }

    // copy bandwidth
    std::atomic<size_t> total{0};
    double t0 = now_s();
    for (int it = 0; it < iters; ++it) {
        std::vector<std::thread> th;
        std::atomic<uint64_t> sink{0};
        for (int t = 0; t < T; ++t) {
            th.emplace_back([&, t]() {
                size_t lo = N * t / T, hi = N * (t + 1) / T;
                uint64_t s = 0;
                for (size_t i = lo; i < hi; i++) { a[i] = b[i] + 1; s += a[i]; }
                sink.fetch_add(s, std::memory_order_relaxed);
            });
        }
        for (auto & x : th) x.join();
        total += sink.load();
    }
    double t1 = now_s();
    double bytes = (double) N * 8 * 2 * iters; // read b + write a
    printf("copy: N=%zu elems (%.0f MB array) T=%d -> %.2f GB/s  (sink=%zu)\n",
           N, N * 8.0 / 1e6, T, bytes / (t1 - t0) / 1e9, total.load());

    // read-only bandwidth (sum reduction)
    t0 = now_s();
    for (int it = 0; it < iters; ++it) {
        std::vector<std::thread> th;
        std::atomic<uint64_t> sink{0};
        for (int t = 0; t < T; ++t) {
            th.emplace_back([&, t]() {
                size_t lo = N * t / T, hi = N * (t + 1) / T;
                uint64_t s = 0;
                for (size_t i = lo; i < hi; i++) s += a[i];
                sink.fetch_add(s, std::memory_order_relaxed);
            });
        }
        for (auto & x : th) x.join();
        total += sink.load();
    }
    t1 = now_s();
    bytes = (double) N * 8 * iters;
    printf("read: T=%d -> %.2f GB/s\n", T, bytes / (t1 - t0) / 1e9);
    return 0;
}
