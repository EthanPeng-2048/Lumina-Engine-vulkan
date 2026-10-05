// Standalone microbenchmark for the Q1_0 x Q8_0 CPU dot kernel.
// Compile: cl /O2 /arch:AVX2 /EHsc bench_q1.cpp
#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif
typedef uint16_t ggml_half;
typedef int8_t   q8_t;

#define QK1_0 128
typedef struct {
    ggml_half d;
    uint8_t   qs[QK1_0 / 8];
} block_q1_0;

#define QK8_0 32
typedef struct {
    ggml_half d;
    int8_t    qs[QK8_0];
} block_q8_0;

static inline float fp16_to_fp32(ggml_half h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h)));
}
static inline ggml_half fp32_to_fp16(float f) {
    return (ggml_half) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), 0), 0);
}

// ---------------------------------------------------------------- current AVX2 kernel
NOINLINE
void vec_dot_q1_0_q8_0_avx2(int n, float * s, const void * vx, const void * vy) {
    const int qk = QK1_0;
    const int nb = n / qk;
    const block_q1_0 * x = (const block_q1_0 *) vx;
    const block_q8_0 * y = (const block_q8_0 *) vy;

    const __m256i ones_8 = _mm256_set1_epi8(1);
    const __m256i ones_16 = _mm256_set1_epi16(1);
    const __m256i byte_shuf = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bit_masks = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, (char) -128, 1, 2, 4, 8, 16, 32, 64, (char) -128,
            1, 2, 4, 8, 16, 32, 64, (char) -128, 1, 2, 4, 8, 16, 32, 64, (char) -128);
    const __m256i zero = _mm256_setzero_si256();
    __m256 acc = _mm256_setzero_ps();

    for (int ib = 0; ib < nb; ++ib) {
        const float d0 = fp16_to_fp32(x[ib].d);
        const uint32_t * qs32 = (const uint32_t *) x[ib].qs;
        const block_q8_0 * y_ptr = &y[ib * 4];

        __m256 acc_block;
        {
            const __m256i qy = _mm256_loadu_si256((const __m256i *) y_ptr[0].qs);
            const __m256i sm = _mm256_cmpeq_epi8(
                    _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) qs32[0]), byte_shuf), bit_masks), zero);
            const __m256i sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
            const __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
            acc_block = _mm256_mul_ps(_mm256_set1_ps(fp16_to_fp32(y_ptr[0].d)), _mm256_cvtepi32_ps(s32));
        }
        for (int K = 1; K < 4; ++K) {
            const __m256i qy = _mm256_loadu_si256((const __m256i *) y_ptr[K].qs);
            const __m256i sm = _mm256_cmpeq_epi8(
                    _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) qs32[K]), byte_shuf), bit_masks), zero);
            const __m256i sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
            const __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
            acc_block = _mm256_fmadd_ps(_mm256_set1_ps(fp16_to_fp32(y_ptr[K].d)), _mm256_cvtepi32_ps(s32), acc_block);
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d0), acc_block, acc);
    }

    // hsum
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    *s = _mm_cvtss_f32(lo);
}

// ---------------------------------------------------------------- candidate: process 2 rows at once
// Dot four 32-bit weight words (128 weights) against the same y; returns int32 sums.
static inline __m256i q1_block_sums(const block_q1_0 * xb, const block_q8_0 * yb,
                                    const __m256i ones_8, const __m256i ones_16,
                                    const __m256i byte_shuf, const __m256i bit_masks, const __m256i zero) {
    const uint32_t * qs32 = (const uint32_t *) xb->qs;
    __m256i acc = _mm256_setzero_si256();
    for (int K = 0; K < 4; ++K) {
        const __m256i qy = _mm256_loadu_si256((const __m256i *) yb[K].qs);
        const __m256i sm = _mm256_cmpeq_epi8(
                _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) qs32[K]), byte_shuf), bit_masks), zero);
        const __m256i sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
        // widen each 32-bit lane sum with scale
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16));
    }
    return acc;
}

NOINLINE
void vec_dot_q1_0_q8_0_avx2_intacc(int n, float * s, const void * vx, const void * vy) {
    const int qk = QK1_0;
    const int nb = n / qk;
    const block_q1_0 * x = (const block_q1_0 *) vx;
    const block_q8_0 * y = (const block_q8_0 *) vy;

    const __m256i ones_8 = _mm256_set1_epi8(1);
    const __m256i ones_16 = _mm256_set1_epi16(1);
    const __m256i byte_shuf = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bit_masks = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, (char) -128, 1, 2, 4, 8, 16, 32, 64, (char) -128,
            1, 2, 4, 8, 16, 32, 64, (char) -128, 1, 2, 4, 8, 16, 32, 64, (char) -128);
    const __m256i zero = _mm256_setzero_si256();

    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ++ib) {
        const float d0 = fp16_to_fp32(x[ib].d);
        const __m256i isum = q1_block_sums(&x[ib], &y[ib * 4], ones_8, ones_16, byte_shuf, bit_masks, zero);
        const __m256 d1 = _mm256_set_ps(fp16_to_fp32(y[ib*4+3].d), fp16_to_fp32(y[ib*4+2].d),
                                        fp16_to_fp32(y[ib*4+1].d), fp16_to_fp32(y[ib*4+0].d),
                                        fp16_to_fp32(y[ib*4+3].d), fp16_to_fp32(y[ib*4+2].d),
                                        fp16_to_fp32(y[ib*4+1].d), fp16_to_fp32(y[ib*4+0].d));
        // isum lanes: [b0, b1, b2, b3, b4, b5, b6, b7] each sums 4 values (128/32)
        __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(isum), d1);
        // horizontal add
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        sumf += d0 * _mm_cvtss_f32(lo);
    }
    *s = sumf;
}

// ---------------------------------------------------------------- generic reference
void vec_dot_q1_0_q8_0_ref(int n, float * s, const void * vx, const void * vy) {
    const int nb = n / QK1_0;
    const block_q1_0 * x = (const block_q1_0 *) vx;
    const block_q8_0 * y = (const block_q8_0 *) vy;
    float sumf = 0;
    for (int i = 0; i < nb; i++) {
        const float d0 = fp16_to_fp32(x[i].d);
        float sumi = 0;
        for (int k = 0; k < 4; k++) {
            const block_q8_0 * yb = &y[i*4+k];
            const float d1 = fp16_to_fp32(yb->d);
            int sumi_block = 0;
            const uint8_t * bits = &x[i].qs[k*4];
            const int8_t * qy = yb->qs;
            for (int b = 0; b < 4; ++b, qy += 8) {
                const unsigned mask = bits[b];
                sumi_block += ((mask & 0x01) ? qy[0] : -qy[0])
                           +  ((mask & 0x02) ? qy[1] : -qy[1])
                           +  ((mask & 0x04) ? qy[2] : -qy[2])
                           +  ((mask & 0x08) ? qy[3] : -qy[3])
                           +  ((mask & 0x10) ? qy[4] : -qy[4])
                           +  ((mask & 0x20) ? qy[5] : -qy[5])
                           +  ((mask & 0x40) ? qy[6] : -qy[6])
                           +  ((mask & 0x80) ? qy[7] : -qy[7]);
            }
            sumi += d1 * sumi_block;
        }
        sumf += d0 * sumi;
    }
    *s = sumf;
}

static double now_s() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    const int K = (argc > 1) ? atoi(argv[1]) : 4096;
    const int ROWS = (argc > 2) ? atoi(argv[2]) : 4096;
    const int iters = (argc > 3) ? atoi(argv[3]) : 200;

    const int nb = K / QK1_0;
    block_q1_0 * x = (block_q1_0 *) _mm_malloc((size_t) ROWS * nb * sizeof(block_q1_0), 64);
    block_q8_0 * y = (block_q8_0 *) _mm_malloc((size_t) nb * 4 * sizeof(block_q8_0), 64);

    srand(1);
    for (size_t i = 0; i < (size_t) ROWS * nb; ++i) {
        x[i].d = fp32_to_fp16(0.001f + 0.001f * (rand() % 100));
        for (int j = 0; j < 16; ++j) x[i].qs[j] = rand() & 0xff;
    }
    for (int i = 0; i < nb * 4; ++i) {
        y[i].d = fp32_to_fp16(0.001f + 0.001f * (rand() % 100));
        for (int j = 0; j < 32; ++j) y[i].qs[j] = (int8_t) (rand() % 255 - 127);
    }

    // correctness
    {
        float r1, r2, r3;
        vec_dot_q1_0_q8_0_ref(K, &r1, x, y);
        vec_dot_q1_0_q8_0_avx2(K, &r2, x, y);
        vec_dot_q1_0_q8_0_avx2_intacc(K, &r3, x, y);
        printf("ref=%.6f avx2=%.6f intacc=%.6f  diff=(%.2e, %.2e)\n",
               r1, r2, r3, fabs(r1-r2), fabs(r1-r3));
    }

    const double bytes = (double) ROWS * nb * sizeof(block_q1_0);

    auto bench = [&](const char * name, void (*fn)(int, float *, const void *, const void *)) {
        float sink = 0;
        // warmup
        for (int r = 0; r < ROWS; ++r) fn(K, &sink, x + (size_t) r * nb, y);
        double t0 = now_s();
        for (int it = 0; it < iters; ++it)
            for (int r = 0; r < ROWS; ++r) fn(K, &sink, x + (size_t) r * nb, y);
        double t1 = now_s();
        double dt = t1 - t0;
        double total_bytes = bytes * iters;
        printf("%-24s %8.3f ms   %7.2f GB/s  %8.2f GMAC/s  (sink=%g)\n", name, dt * 1e3,
               total_bytes / dt / 1e9, (double) ROWS * K * iters / dt / 1e9, sink);
    };

    printf("K=%d ROWS=%d iters=%d  weight bytes=%.2f MB\n", K, ROWS, iters, bytes / 1e6);
    bench("ref-scalar", vec_dot_q1_0_q8_0_ref);
    bench("avx2-current", vec_dot_q1_0_q8_0_avx2);
    bench("avx2-intacc", vec_dot_q1_0_q8_0_avx2_intacc);
    return 0;
}
