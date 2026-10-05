// multithreaded streaming bandwidth probe (vectorized, multiple accumulators)
#include <immintrin.h>
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

static std::atomic<long long> g_sink;

static void read_thread(const int * a, size_t lo, size_t hi) {
    __m256i s0 = _mm256_setzero_si256(), s1 = _mm256_setzero_si256();
    __m256i s2 = _mm256_setzero_si256(), s3 = _mm256_setzero_si256();
    size_t i = lo;
    for (; i + 32 <= hi; i += 32) {
        s0 = _mm256_add_epi32(s0, _mm256_loadu_si256((const __m256i*)(a+i+0)));
        s1 = _mm256_add_epi32(s1, _mm256_loadu_si256((const __m256i*)(a+i+8)));
        s2 = _mm256_add_epi32(s2, _mm256_loadu_si256((const __m256i*)(a+i+16)));
        s3 = _mm256_add_epi32(s3, _mm256_loadu_si256((const __m256i*)(a+i+24)));
    }
    s0 = _mm256_add_epi32(_mm256_add_epi32(s0,s1), _mm256_add_epi32(s2,s3));
    __m128i x = _mm_add_epi32(_mm256_castsi256_si128(s0), _mm256_extracti128_si256(s0,1));
    g_sink += _mm_extract_epi32(x,0);
}

static void copy_thread(const int * __restrict a, int * __restrict b, size_t lo, size_t hi) {
    for (size_t i = lo; i + 8 <= hi; i += 8) {
        _mm256_storeu_si256((__m256i*)(b+i), _mm256_loadu_si256((const __m256i*)(a+i)));
    }
}

int main(int argc, char ** argv) {
    const size_t N = (argc > 1) ? (size_t) atoll(argv[1]) : (size_t) 128 * 1024 * 1024; // ints
    const int T = (argc > 2) ? atoi(argv[2]) : 6;
    const int iters = (argc > 3) ? atoi(argv[3]) : 10;
    std::vector<int> a(N), b(N);
    for (size_t i = 0; i < N; i++) { a[i] = (int)i; }

    auto run = [&](const char * name, void (*fn)(const int*, int*, size_t, size_t)) {
        // warm
        { std::vector<std::thread> th; for (int t=0;t<T;t++) th.emplace_back([&,t]{ fn(a.data(), b.data(), N*t/T, N*(t+1)/T); }); for (auto&x:th)x.join(); }
        double t0 = now_s();
        for (int it = 0; it < iters; ++it) {
            std::vector<std::thread> th;
            for (int t = 0; t < T; ++t) th.emplace_back([&,t]{ fn(a.data(), b.data(), N*t/T, N*(t+1)/T); });
            for (auto & x : th) x.join();
        }
        double t1 = now_s();
        double bytes = (double) N * 4 * iters;
        printf("%-8s T=%d -> %.2f GB/s\n", name, T, bytes / (t1-t0) / 1e9);
    };
    run("read", [](const int*a,int*,size_t lo,size_t hi){ read_thread(a,lo,hi); });
    run("copy", [](const int*a,int*b,size_t lo,size_t hi){ copy_thread(a,b,lo,hi); });
    return 0;
}
