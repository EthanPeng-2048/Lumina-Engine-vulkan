#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "traits.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <vector>

#include "q1t.h"

#define UNUSED GGML_UNUSED

// rows per tile and columns per tile of the register blocking
#define Q1T_ROWS 2
#define Q1T_COLS 4

// rows per group for the single-column (token generation) path. Q1T_GEMV_ROWS_MAX bounds the
// register arrays; the effective value defaults to Q1T_GEMV_ROWS_DEFAULT and can be tuned at
// runtime with the GGML_Q1_GEMV_ROWS environment variable (1, 2 or 4).
#define Q1T_GEMV_ROWS_MAX     4
#define Q1T_GEMV_ROWS_DEFAULT 1

// smallest src1 batch for which the tiled path pays for itself
#define Q1T_MIN_BATCH 4

// activation block size used by the tiled path: one fp16 scale per 128 values.
// A whole src0 row-block (4 Q1_0 sub-blocks) then shares a single activation scale, so the
// integer dot products of the four sub-blocks can be summed before the (expensive) float
// conversion. Q8_0 uses one scale per 32 values, which forces a float conversion per sub-block.
#define QK8_128 128
typedef struct {
    ggml_half d;
    int8_t    qs[QK8_128];
} block_q8_128;
static_assert(sizeof(block_q8_128) == sizeof(ggml_half) + QK8_128, "wrong q8_128 block size/padding");

#if defined(_MSC_VER)
#define Q1T_NOINLINE __declspec(noinline)
#else
#define Q1T_NOINLINE __attribute__((noinline))
#endif

// VNNI: a single vpdpbusd replaces the vpmaddubsw + vpmaddwd pair.
#if defined(__AVX512VNNI__)
#  define Q1T_VNNI 1
#  define Q1T_DPBUSD(acc, a, b) _mm256_dpbusd_epi32(acc, a, b)
#elif defined(__AVXVNNI__)
#  define Q1T_VNNI 1
#  define Q1T_DPBUSD(acc, a, b) _mm256_dpbusd_avx_epi32(acc, a, b)
#endif

#if defined(__AVX2__)

static inline float q1t_hsum_8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

static const __m256i q1t_byte_shuf = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
static const __m256i q1t_bit_masks = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, (char) -128, 1, 2, 4, 8, 16, 32, 64, (char) -128,
        1, 2, 4, 8, 16, 32, 64, (char) -128, 1, 2, 4, 8, 16, 32, 64, (char) -128);

// 0x01 for a set sign bit (adds the activation), 0xFF for a clear one (negates it)
static inline __m256i q1t_expand(uint32_t bits) {
    const __m256i bc = _mm256_set1_epi32((int) bits);
    const __m256i sh = _mm256_shuffle_epi8(bc, q1t_byte_shuf);
    return _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(sh, q1t_bit_masks), _mm256_setzero_si256()), _mm256_set1_epi8(1));
}

// quantize one 128-value block of floats to the per-128 activation format
#if defined(__AVX2__)
static void q1t_quantize_block(const float * GGML_RESTRICT x, block_q8_128 * GGML_RESTRICT y) {
    __m256 v[4][4];
    for (int g = 0; g < 4; ++g) {
        for (int q = 0; q < 4; ++q) {
            v[g][q] = _mm256_loadu_ps(x + g * 32 + q * 8);
        }
    }

    const __m256 sign_bit = _mm256_set1_ps(-0.0f);
    __m256 max_abs = _mm256_andnot_ps(sign_bit, v[0][0]);
    for (int g = 0; g < 4; ++g) {
        for (int q = 0; q < 4; ++q) {
            max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v[g][q]));
        }
    }

    __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(max_abs, 1), _mm256_castps256_ps128(max_abs));
    max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
    const float amax = _mm_cvtss_f32(max4);

    const float d  = amax / 127.0f;
    const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
    y->d = GGML_FP32_TO_FP16(d);

    const __m256 mul = _mm256_set1_ps(id);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    for (int g = 0; g < 4; ++g) {
        __m256i i0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v[g][0], mul), _MM_ROUND_NEAREST));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v[g][1], mul), _MM_ROUND_NEAREST));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v[g][2], mul), _MM_ROUND_NEAREST));
        __m256i i3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v[g][3], mul), _MM_ROUND_NEAREST));

        i0 = _mm256_packs_epi32(i0, i1);
        i2 = _mm256_packs_epi32(i2, i3);
        i0 = _mm256_packs_epi16(i0, i2);
        i0 = _mm256_permutevar8x32_epi32(i0, perm);

        _mm256_storeu_si256((__m256i *) (y->qs + g * 32), i0);
    }
}
#else
static void q1t_quantize_block(const float * GGML_RESTRICT x, block_q8_128 * GGML_RESTRICT y) {
    float amax = 0.0f;
    for (int j = 0; j < QK8_128; ++j) {
        const float a = fabsf(x[j]);
        if (a > amax) {
            amax = a;
        }
    }

    const float d  = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    y->d = GGML_FP32_TO_FP16(d);

    for (int j = 0; j < QK8_128; ++j) {
        int q = (int) roundf(x[j] * id);
        q = q < -128 ? -128 : (q > 127 ? 127 : q);
        y->qs[j] = (int8_t) q;
    }
}
#endif

// R src0 rows x T src1 columns; the R*T results are written to out[r*T + t].
// The sign vectors of the R rows are expanded once per K-block and reused for all T columns.
// The four sub-blocks of a Q1_0 block share one activation scale, so their integer dot
// products are accumulated in int32 and converted to float only once per block.
template <int R, int T>
Q1T_NOINLINE void q1t_dot(int n, float * GGML_RESTRICT out,
                    const block_q1_0 * const * x,
                    const block_q8_128 * const * y) {
    const int nb = n / QK1_0;

    const __m256i ones = _mm256_set1_epi8(1);
#if !defined(Q1T_VNNI)
    const __m256i ones16 = _mm256_set1_epi16(1);
#endif

    __m256 acc[R][T];
    for (int r = 0; r < R; ++r) {
        for (int t = 0; t < T; ++t) {
            acc[r][t] = _mm256_setzero_ps();
        }
    }

    for (int ib = 0; ib < nb; ++ib) {
        __m256 d0v[R];
        for (int r = 0; r < R; ++r) {
            d0v[r] = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[r][ib].d));
        }

        __m256i t32[R][T];
        for (int r = 0; r < R; ++r) {
            for (int t = 0; t < T; ++t) {
                t32[r][t] = _mm256_setzero_si256();
            }
        }

        for (int k = 0; k < 4; ++k) {
            __m256i sg[R];
            for (int r = 0; r < R; ++r) {
                sg[r] = q1t_expand(((const uint32_t *) x[r][ib].qs)[k]);
            }

            for (int t = 0; t < T; ++t) {
                const __m256i qy = _mm256_loadu_si256((const __m256i *) (y[t][ib].qs + k * 32));
                for (int r = 0; r < R; ++r) {
                    const __m256i sy = _mm256_sign_epi8(qy, sg[r]);
#if defined(Q1T_VNNI)
                    t32[r][t] = Q1T_DPBUSD(t32[r][t], ones, sy);
#else
                    t32[r][t] = _mm256_add_epi32(t32[r][t], _mm256_madd_epi16(_mm256_maddubs_epi16(ones, sy), ones16));
#endif
                }
            }
        }

        for (int t = 0; t < T; ++t) {
            const __m256 d1v = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[t][ib].d));
            for (int r = 0; r < R; ++r) {
                const __m256 dscale = _mm256_mul_ps(d0v[r], d1v);
                acc[r][t] = _mm256_fmadd_ps(dscale, _mm256_cvtepi32_ps(t32[r][t]), acc[r][t]);
            }
        }
    }

    for (int r = 0; r < R; ++r) {
        for (int t = 0; t < T; ++t) {
            out[r * T + t] = q1t_hsum_8(acc[r][t]);
        }
    }
}

// Single-column (token generation) kernel. R src0 rows are processed together so the
// activation vector is loaded once and, thanks to the per-128 activation scale, the four
// sub-block integer dot products are summed before the single float conversion per row.
template <int R>
Q1T_NOINLINE void q1t_gemv(int n, float * GGML_RESTRICT out,
                     const block_q1_0 * const * x,
                     const block_q8_128 * GGML_RESTRICT y) {
    const int nb = n / QK1_0;

    const __m256i ones = _mm256_set1_epi8(1);
#if !defined(Q1T_VNNI)
    const __m256i ones16 = _mm256_set1_epi16(1);
#endif

    __m256 acc[R];
    for (int r = 0; r < R; ++r) {
        acc[r] = _mm256_setzero_ps();
    }

    for (int ib = 0; ib < nb; ++ib) {
        __m256 d0v[R];
        __m256i t32[R];
        for (int r = 0; r < R; ++r) {
            d0v[r] = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[r][ib].d));
            t32[r] = _mm256_setzero_si256();
        }

        for (int k = 0; k < 4; ++k) {
            const __m256i qy = _mm256_loadu_si256((const __m256i *) (y[ib].qs + k * 32));
            for (int r = 0; r < R; ++r) {
                const __m256i sg = q1t_expand(((const uint32_t *) x[r][ib].qs)[k]);
                const __m256i sy = _mm256_sign_epi8(qy, sg);
#if defined(Q1T_VNNI)
                t32[r] = Q1T_DPBUSD(t32[r], ones, sy);
#else
                t32[r] = _mm256_add_epi32(t32[r], _mm256_madd_epi16(_mm256_maddubs_epi16(ones, sy), ones16));
#endif
            }
        }

        const __m256 d1v = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[ib].d));
        for (int r = 0; r < R; ++r) {
            acc[r] = _mm256_fmadd_ps(_mm256_mul_ps(d0v[r], d1v), _mm256_cvtepi32_ps(t32[r]), acc[r]);
        }
    }

    for (int r = 0; r < R; ++r) {
        out[r] = q1t_hsum_8(acc[r]);
    }
}

#endif // __AVX2__

// ---------------------------------------------------------------------------
// 2-value (binary) activation lookup table for token generation.
//
// A group of 4 consecutive Q1_0 weights only has 2^4 = 16 sign patterns, so for one
// activation vector we precompute, for every group of 4 activations, a 16-entry table
//
//     T[c] = sum_{i=0..3} ( bit_i(c) ? +a[i] : -a[i] )
//
// and the partial dot product of the group is a single vpshufb lookup indexed by the packed
// weight code. The table is built once per activation vector (per token) and reused for all
// output rows, which is what makes a table pay for a 2-value weight alphabet. The table is
// 8 bytes per weight: it fits the 32 KiB L1d for K<=4096 but spills to L2 for larger K. The
// table is built once per activation (token) and reused for every output row, and for prefill
// it is built once per activation column chunk and reused across all 32-row weight groups, so
// the lookup codes stay in cache and the kernel is compute-bound. On this machine (AVX2, no
// VNNI) that makes it faster than the packed kernels for both token generation and prefill;
// see wiki/Optimizations.md. The table is emitted as 16 low bytes followed by 16 high bytes so
// one lookup is two vpshufb.
//
// The 1-bit weights are permuted once (lazily, on first use of a tensor) into the layout the
// kernel wants: per group of 32 output rows and group of 4 K positions, 16 bytes hold the 32
// four-bit codes, two codes per byte. This is a pure permutation, same footprint as the
// packed weights, and is cached for the lifetime of the process. Nothing here is copied from
// any third-party kernel; it is built from the Q1_0 format and AVX2 primitives.
// ---------------------------------------------------------------------------

#define Q1LUT_ROWS      32
#define Q1LUT_GROUP     4
#define Q1LUT_PP_COLS   32 // prefill: number of activation columns whose tables are built per chunk
#define Q1LUT_PP_MIN    8  // smallest prefill batch for which the per-column LUT beats the tiled gemm
#define Q1LUT_ENTRIES   (1 << Q1LUT_GROUP)
#define Q1LUT_TABLE_LEN (Q1LUT_ENTRIES * 2) // low bytes then high bytes

// parse a boolean environment variable: unset/empty -> `def`; "0"/"false"/"off"/"no"
// (case-insensitive) -> false; anything else -> true.
static bool q1t_env_bool(const char * name, bool def) {
    const char * e = getenv(name);
    if (e == nullptr || e[0] == '\0') {
        return def;
    }
    if (e[0] == '0' && e[1] == '\0') {
        return false;
    }
    if ((e[0] == 'f' || e[0] == 'F') && (e[1] == 'a' || e[1] == 'A') && (e[2] == 'l' || e[2] == 'L') &&
        (e[3] == 's' || e[3] == 'S') && (e[4] == 'e' || e[4] == 'E') && e[5] == '\0') {
        return false;
    }
    if ((e[0] == 'o' || e[0] == 'O') && (e[1] == 'f' || e[1] == 'F') && (e[2] == 'f' || e[2] == 'F') && e[3] == '\0') {
        return false;
    }
    if ((e[0] == 'n' || e[0] == 'N') && (e[1] == 'o' || e[1] == 'O') && e[2] == '\0') {
        return false;
    }
    return true;
}

// The 2-value activation LUT is the fastest token-generation kernel, so it is enabled by
// default. Set GGML_Q1_LUT=0 to fall back to the packed Q1_0 GEMV (A/B, debugging, or to
// save the ~1 GiB of per-tensor lookup codes).
static bool q1t_lut_enabled(void) {
    static const bool enabled = q1t_env_bool("GGML_Q1_LUT", true);
    return enabled;
}

// Batched prefill LUT. On AVX2 CPUs it beats the 2x4 register-tiled GEMM for every batch
// size >= Q1LUT_PP_MIN (measured 8), and the margin grows with the batch. It is enabled by
// default for those batches; GGML_Q1_LUT_PP=0 forces the tiled GEMM instead.
static bool q1t_lut_pp_enabled(void) {
    static const bool enabled = q1t_env_bool("GGML_Q1_LUT_PP", true);
    return enabled;
}

// K-block size for the LUT path (experimental tuning knob). The table slice for one block is
// (kb/QK1_0)*1 KiB. Default 0 means "whole K, one block", which measured fastest: the table is
// shared and read from L2, and small blocks only add accumulator read-modify-write traffic.
static int64_t q1t_lut_kb(void) {
    static const int64_t kb = []() -> int64_t {
        int64_t v = 0;
        if (const char * e = getenv("GGML_Q1_LUT_KB")) {
            const long long t = atoll(e);
            v = t > 0 ? (int64_t) t : 0;
        }
        if (v > 0) {
            v = (v / QK1_0) * QK1_0; // align to whole Q1_0 blocks
            if (v < QK1_0) {
                v = QK1_0;
            }
        }
        return v;
    }();
    return kb;
}

// lazily built, process-lifetime cache of the permuted Q1_0 codes. Entries are allocated once
// and then filled cooperatively by all worker threads (row groups partitioned by thread id),
// so the one-time cost is spread over the thread pool.
struct q1lut_codes {
    const void *     data;
    int64_t          ne0;
    int64_t          ne1;
    size_t           nb01;
    int64_t          nb;
    int64_t          ngroups;
    size_t           per_group;
    uint8_t *        codes;
    ggml_half *      d0; // [ngroups][nb][Q1LUT_ROWS], same fp16 scale as block_q1_0
    std::atomic<bool> built;
};

static std::mutex                 q1lut_mutex;
static std::vector<q1lut_codes *> q1lut_cache; // entries are stable; leaked for process lifetime

static q1lut_codes * q1lut_get_or_alloc(const struct ggml_tensor * src0) {
    std::lock_guard<std::mutex> lock(q1lut_mutex);
    for (q1lut_codes * e : q1lut_cache) {
        if (e->data == src0->data && e->ne0 == src0->ne[0] && e->ne1 == src0->ne[1] && e->nb01 == src0->nb[1]) {
            return e;
        }
    }

    const int64_t K  = src0->ne[0];
    const int64_t M  = src0->ne[1];
    const int64_t nb = K / QK1_0;
    const size_t  nb01 = src0->nb[1];
    const size_t  per_group = (size_t) (K / Q1LUT_GROUP) * 16;
    const int64_t ngroups   = (M + Q1LUT_ROWS - 1) / Q1LUT_ROWS;

    uint8_t *   codes = (uint8_t *)   malloc((size_t) ngroups * per_group);
    ggml_half * d0    = (ggml_half *) malloc((size_t) ngroups * nb * Q1LUT_ROWS * sizeof(ggml_half));
    if (codes == nullptr || d0 == nullptr) {
        free(codes);
        free(d0);
        return nullptr;
    }
    memset(codes, 0, (size_t) ngroups * per_group);

    if (q1t_env_bool("GGML_Q1_LUT_DEBUG", false)) {
        fprintf(stderr, "q1lut alloc: data=%p ne0=%lld ne1=%lld bytes=%zu cache=%zu\n",
                src0->data, (long long) K, (long long) M, (size_t) ngroups * per_group, q1lut_cache.size());
    }

    q1lut_codes * e = new q1lut_codes{ src0->data, K, M, nb01, nb, ngroups, per_group, codes, d0, false };
    q1lut_cache.push_back(e);
    return e;
}

// fill one 32-row group
static void q1lut_fill_group(q1lut_codes * e, int64_t r0) {
    const int64_t M  = e->ne1;
    const int64_t nb = e->nb;
    const size_t  nb01 = e->nb01;
    const int64_t rg = r0 / Q1LUT_ROWS;
    uint8_t * g = e->codes + (size_t) rg * e->per_group;
    const uint8_t * data = (const uint8_t *) e->data;

    for (int64_t ib = 0; ib < nb; ++ib) {
        const size_t boff = (size_t) ib * sizeof(block_q1_0) + 2; // skip the fp16 scale
        ggml_half * gd = e->d0 + ((size_t) rg * nb + ib) * Q1LUT_ROWS;
        for (int r = 0; r < Q1LUT_ROWS; ++r) {
            const int64_t ri = r0 + r;
            // copy the fp16 scale verbatim: keeps tg weight traffic equal to the packed format
            gd[r] = ri < M
                ? ((const block_q1_0 *) (data + (size_t) ri * nb01 + ib * sizeof(block_q1_0)))->d
                : (ggml_half) 0;
        }
        for (int b = 0; b < 16; ++b) {
            const size_t kg0 = (size_t) (ib * 32 + 2 * b);
            for (int j = 0; j < Q1LUT_ROWS / 2; ++j) {
                const int64_t r0i = r0 + 2 * j;
                const int64_t r1i = r0i + 1;
                const uint8_t v0 = r0i < M ? data[(size_t) r0i * nb01 + boff + b] : 0;
                const uint8_t v1 = r1i < M ? data[(size_t) r1i * nb01 + boff + b] : 0;
                g[kg0 * 16 + j]       = (uint8_t) ((v0 & 0x0f) | ((v1 & 0x0f) << 4));
                g[(kg0 + 1) * 16 + j] = (uint8_t) (((v0 >> 4) & 0x0f) | ((v1 >> 4) << 4));
            }
        }
    }
}

// build the 16-entry (int16) table for one group of 4 int8 activations.
// True 2-value symmetry: complementing all four weight bits flips every sign, so
// T[c ^ 0xf] == -T[c]. Only the lower half (c = 0..7) has to be summed.
static inline void q1lut_build_group(const int8_t * a, uint8_t * t) {
    for (int c = 0; c < Q1LUT_ENTRIES / 2; ++c) {
        int v = 0;
        for (int i = 0; i < Q1LUT_GROUP; ++i) {
            v += ((c >> i) & 1) ? a[i] : -a[i];
        }
        const int nc = (Q1LUT_ENTRIES - 1) - c;
        t[c]                  = (uint8_t) (v & 0xff);
        t[nc]                 = (uint8_t) ((-v) & 0xff);
        t[Q1LUT_ENTRIES + c]  = (uint8_t) ((v >> 8) & 0xff);
        t[Q1LUT_ENTRIES + nc] = (uint8_t) (((-v) >> 8) & 0xff);
    }
}

#if defined(__AVX2__)
// load Q1LUT_ROWS fp16 scales into floats (F16C)
static inline void q1lut_load_d0(const ggml_half * h, float * f) {
    for (int i = 0; i < Q1LUT_ROWS; i += 8) {
        _mm256_storeu_ps(f + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (h + i))));
    }
}

// one 128-weight block for 32 output rows: 32 table lookups, int16 accumulation
static inline void q1lut_acc128(const uint8_t * codes, const uint8_t * tbl, size_t g0,
                                const ggml_half * d0, const block_q8_128 * yq8, float * fdot) {
    const __m128i m4 = _mm_set1_epi8(0x0f);
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    for (int kg = 0; kg < 32; ++kg) {
        const size_t g = g0 + kg;
        const __m128i tlo = _mm_loadu_si128((const __m128i *) (tbl + Q1LUT_TABLE_LEN * g));
        const __m128i thi = _mm_loadu_si128((const __m128i *) (tbl + Q1LUT_TABLE_LEN * g + Q1LUT_ENTRIES));
        const __m128i w   = _mm_loadu_si128((const __m128i *) (codes + 16 * g));
        const __m128i ce  = _mm_and_si128(w, m4);
        const __m128i co  = _mm_and_si128(_mm_srli_epi16(w, 4), m4);
        const __m256i C   = _mm256_set_m128i(co, ce);
        const __m256i L   = _mm256_broadcastsi128_si256(tlo);
        const __m256i H   = _mm256_broadcastsi128_si256(thi);
        const __m256i LE  = _mm256_shuffle_epi8(L, C);
        const __m256i HE  = _mm256_shuffle_epi8(H, C);
        acc0 = _mm256_add_epi16(acc0, _mm256_unpacklo_epi8(LE, HE));
        acc1 = _mm256_add_epi16(acc1, _mm256_unpackhi_epi8(LE, HE));
    }

    int32_t tmp[Q1LUT_ROWS];
    _mm256_storeu_si256((__m256i *) (tmp + 0),  _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc0)));
    _mm256_storeu_si256((__m256i *) (tmp + 8),  _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc0, 1)));
    _mm256_storeu_si256((__m256i *) (tmp + 16), _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc1)));
    _mm256_storeu_si256((__m256i *) (tmp + 24), _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc1, 1)));

    float d0f[Q1LUT_ROWS];
    q1lut_load_d0(d0, d0f);
    const float d1 = GGML_CPU_FP16_TO_FP32(yq8->d);
    for (int j = 0; j < 8; ++j) {
        fdot[2 * j]      += (float) tmp[j]      * (d1 * d0f[2 * j]);
        fdot[2 * j + 1]  += (float) tmp[8 + j]  * (d1 * d0f[2 * j + 1]);
        fdot[16 + 2 * j] += (float) tmp[16 + j] * (d1 * d0f[16 + 2 * j]);
        fdot[17 + 2 * j] += (float) tmp[24 + j] * (d1 * d0f[17 + 2 * j]);
    }
}

// 32 output rows, full K, one activation LUT. `out` is the first row's float destination.
static void q1lut_gemv32(int64_t K, char * out, size_t nb0,
                         const uint8_t * codes, const uint8_t * tbl,
                         const ggml_half * d0buf, const block_q8_128 * yq8) {
    const int64_t nb = K / QK1_0;
    float fdot[Q1LUT_ROWS] = {0};
    for (int64_t ib = 0; ib < nb; ++ib) {
        q1lut_acc128(codes, tbl, (size_t) ib * 32, d0buf + (size_t) ib * Q1LUT_ROWS, yq8 + ib, fdot);
    }
    for (int r = 0; r < Q1LUT_ROWS; ++r) {
        *(float *) (out + (size_t) r * nb0) = fdot[r];
    }
}

// Accumulate a K-block [ib0, ib1) of 128-value groups into 32 output rows. The caller sweeps
// every output row for one K-block before moving on, so the live table slice (ib1-ib0)*1 KiB
// stays in L1 and is only materialised once per token instead of once per 32-row group.
static void q1lut_gemv32_add(int64_t ib0, int64_t ib1, char * out, size_t nb0,
                             const uint8_t * codes, const uint8_t * tbl,
                             const ggml_half * d0base, const block_q8_128 * yq8) {
    float fdot[Q1LUT_ROWS] = {0};
    for (int64_t ib = ib0; ib < ib1; ++ib) {
        q1lut_acc128(codes, tbl, (size_t) ib * 32, d0base + (size_t) ib * Q1LUT_ROWS, yq8 + ib, fdot);
    }
    for (int r = 0; r < Q1LUT_ROWS; ++r) {
        *(float *) (out + (size_t) r * nb0) += fdot[r];
    }
}
#endif

// effective row group size for the single-column GEMV (tunable via GGML_Q1_GEMV_ROWS)
static int q1t_gemv_rows(void) {
    static const int rows = []() {
        int r = Q1T_GEMV_ROWS_DEFAULT;
        if (const char * e = getenv("GGML_Q1_GEMV_ROWS")) {
            const int v = atoi(e);
            if (v == 1 || v == 2 || v == 4) {
                r = v;
            }
        }
        return r;
    }();
    return rows;
}

bool ggml_cpu_q1t_supports_mul_mat(const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (src0->type != GGML_TYPE_Q1_0) {
        return false;
    }

    if (ggml_get_type_traits_cpu(src0->type)->vec_dot_type != GGML_TYPE_Q8_0) {
        return false;
    }

#if !defined(__AVX2__)
    return false;
#else
    if (!ggml_cpu_has_avx2()) {
        return false;
    }

    static const bool disabled = getenv("GGML_NO_Q1_TILE") != nullptr;
    if (disabled) {
        return false;
    }

    if (src1->type != GGML_TYPE_F32) {
        return false;
    }

    if (src0->ne[0] % QK1_0 != 0 || src0->ne[0] % QK8_128 != 0) {
        return false;
    }

    // need at least one full row group
    if (src0->ne[1] < Q1T_ROWS) {
        return false;
    }

    // single column: token-generation GEMV; small batches: tiled GEMM
    if (src1->ne[1] == 1) {
        static const bool no_gemv = getenv("GGML_NO_Q1_GEMV") != nullptr;
        if (no_gemv) {
            return false;
        }
    } else if (src1->ne[1] < Q1T_MIN_BATCH) {
        return false;
    }

    // plain 2D weight matmuls only (src1 may still be batched over ne12)
    if (src0->ne[2] != 1 || src0->ne[3] != 1) {
        return false;
    }

    if (src1->ne[3] != 1) {
        return false;
    }

    if (!ggml_is_contiguous(src0)) {
        return false;
    }

    if (dst->type != GGML_TYPE_F32 || dst->nb[0] != sizeof(float)) {
        return false;
    }

    return true;
#endif
}

// shared scratch: the per-128 activation for all src1 rows, plus the optional LUT tables
size_t ggml_cpu_q1t_scratch_size(const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const size_t qrow = (size_t) (src1->ne[0] / QK8_128) * sizeof(block_q8_128);
    size_t size = GGML_PAD(qrow * src1->ne[1] * src1->ne[2], 64);
    if (q1t_lut_enabled() && src1->ne[1] == 1) {
        const size_t lut_row = (size_t) (src0->ne[0] / Q1LUT_GROUP) * Q1LUT_TABLE_LEN;
        size += GGML_PAD(lut_row * src1->ne[2], 64);
    } else if (q1t_lut_enabled() && q1t_lut_pp_enabled() && src1->ne[1] >= Q1LUT_PP_MIN) {
        const size_t lut_row = (size_t) (src0->ne[0] / Q1LUT_GROUP) * Q1LUT_TABLE_LEN;
        const int64_t ntbl = (int64_t) MIN((int64_t) src1->ne[1], (int64_t) Q1LUT_PP_COLS) * src1->ne[2];
        size += GGML_PAD(lut_row * (size_t) ntbl, 64);
    }
    return size;
}

void ggml_compute_forward_mul_mat_q1t(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const size_t nbw1 = ggml_row_size(GGML_TYPE_Q8_0, ne10);
    const size_t nbw2 = nbw1 * ne11;

    const size_t scratch_size   = ggml_cpu_q1t_scratch_size(dst);
    const size_t scratch_offset = GGML_PAD(nbw2 * ne12, 64);

    GGML_ASSERT(scratch_offset + scratch_size <= params->wsize);

    char * q8 = (char *) params->wdata + scratch_offset;
    const size_t qrow_size = (size_t) (ne10 / QK8_128) * sizeof(block_q8_128);

    const int64_t ngroups = ne01 / Q1T_ROWS;

#if defined(__AVX2__)
    // quantize the F32 activations to the per-128 format (shared, parallel over blocks)
    {
        const int64_t nblocks_per_row = ne10 / QK8_128;
        const int64_t total_blocks    = nblocks_per_row * ne11 * ne12;

        for (int64_t blk = ith; blk < total_blocks; blk += nth) {
            const int64_t row = blk / nblocks_per_row;
            const int64_t bi  = blk % nblocks_per_row;

            const int64_t i12 = row / ne11;
            const int64_t i11 = row % ne11;

            const float * src = (const float *) ((const char *) src1->data + i12 * nb12 + i11 * nb11) + bi * QK8_128;
            block_q8_128 * qrow = (block_q8_128 *) (q8 + row * qrow_size) + bi;

            q1t_quantize_block(src, qrow);
        }
    }

    ggml_barrier(params->threadpool);

    if (ne11 == 1) {
        // ---- optional 2-value activation LUT path (32 rows per group) ----
        if (q1t_lut_enabled() && ne01 >= Q1LUT_ROWS && ne00 % QK1_0 == 0) {
            q1lut_codes * lut = q1lut_get_or_alloc(src0);
            if (lut != nullptr) {
                // one-time cooperative permutation of this tensor
                if (!lut->built.load(std::memory_order_acquire)) {
                    for (int64_t g = ith; g < lut->ngroups; g += nth) {
                        q1lut_fill_group(lut, g * Q1LUT_ROWS);
                    }
                    ggml_barrier(params->threadpool);
                    if (ith == 0) {
                        lut->built.store(true, std::memory_order_release);
                    }
                    ggml_barrier(params->threadpool);
                }

                const uint8_t * codes = lut->codes;
                const int64_t   nb    = ne00 / QK1_0;
                const size_t lut_row = (size_t) (ne00 / Q1LUT_GROUP) * Q1LUT_TABLE_LEN;
                uint8_t * lut_base = (uint8_t *) q8 + GGML_PAD((size_t) qrow_size * ne11 * ne12, 64);
                const int64_t nkg = ne00 / Q1LUT_GROUP;

                // build the activation tables (parallel over groups), then sync
                for (int64_t i12 = 0; i12 < ne12; ++i12) {
                    const block_q8_128 * yrow = (const block_q8_128 *) (q8 + i12 * qrow_size);
                    uint8_t * tbl = lut_base + i12 * lut_row;
                    for (int64_t kg = ith; kg < nkg; kg += nth) {
                        const int64_t ib = kg / 32; // 32 groups of 4 per 128-value block
                        const int64_t kk = kg % 32;
                        q1lut_build_group(yrow[ib].qs + Q1LUT_GROUP * kk, tbl + Q1LUT_TABLE_LEN * kg);
                    }
                }
                ggml_barrier(params->threadpool);

                const int64_t ngroups = ne01 / Q1LUT_ROWS;
                const size_t  per_group = (size_t) nkg * 16;
                const int64_t kb_req = q1t_lut_kb();

                if (kb_req <= 0) {
                    // whole-K direct write (default; fastest and no extra passes over dst)
                    const int64_t chunks_per_thread = ggml_is_numa() ? 1 : 4;
                    const int64_t groups_per_chunk  = MAX(1, (ngroups + nth * chunks_per_thread - 1) / (nth * chunks_per_thread));
                    const int64_t nchunk            = (ngroups + groups_per_chunk - 1) / groups_per_chunk;

                    int chunk = ith;
                    while (chunk < nchunk) {
                        const int64_t g0 = chunk * groups_per_chunk;
                        const int64_t g1 = MIN(g0 + groups_per_chunk, ngroups);
                        for (int64_t g = g0; g < g1; ++g) {
                            const int64_t r = g * Q1LUT_ROWS;
                            const uint8_t *    gcodes = codes + (size_t) g * per_group;
                            const ggml_half *  gd0    = lut->d0 + (size_t) g * nb * Q1LUT_ROWS;
                            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                                const uint8_t * tbl = lut_base + i12 * lut_row;
                                const block_q8_128 * yq8 = (const block_q8_128 *) (q8 + i12 * qrow_size);
                                char * dst_ptr = (char *) dst->data + i12 * nb2 + (size_t) r * nb0;
                                q1lut_gemv32(ne00, dst_ptr, nb0, gcodes, tbl, gd0, yq8);
                            }
                        }
                        chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
                    }
                } else {
                    // experimental K-blocked accumulate path; keeps the live table slice small
                    for (int64_t idx = ith; idx < ne01 * ne12; idx += nth) {
                        const int64_t i12 = idx / ne01;
                        const int64_t rr  = idx % ne01;
                        *(float *) ((char *) dst->data + i12 * nb2 + (size_t) rr * nb0) = 0.0f;
                    }
                    ggml_barrier(params->threadpool);

                    const int64_t nbb = MAX((int64_t) 1, kb_req / QK1_0);
                    for (int64_t ib0 = 0; ib0 < nb; ib0 += nbb) {
                        const int64_t ib1 = MIN(ib0 + nbb, nb);
                        for (int64_t g = ith; g < ngroups; g += nth) {
                            const uint8_t *    gcodes = codes + (size_t) g * per_group;
                            const ggml_half *  gd0    = lut->d0 + (size_t) g * nb * Q1LUT_ROWS;
                            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                                const uint8_t * tbl = lut_base + i12 * lut_row;
                                const block_q8_128 * yq8 = (const block_q8_128 *) (q8 + i12 * qrow_size);
                                char * dst_ptr = (char *) dst->data + i12 * nb2 + (size_t) (g * Q1LUT_ROWS) * nb0;
                                q1lut_gemv32_add(ib0, ib1, dst_ptr, nb0, gcodes, tbl, gd0, yq8);
                            }
                        }
                    }
                }

                // trailing rows that did not fill a full group
                const int64_t r_tail = ngroups * Q1LUT_ROWS;
                for (int64_t r = r_tail + ith; r < ne01; r += nth) {
                    const block_q1_0 * xp[1] = { (const block_q1_0 *) ((const char *) src0->data + r * nb01) };
                    for (int64_t i12 = 0; i12 < ne12; ++i12) {
                        const block_q8_128 * y = (const block_q8_128 *) (q8 + i12 * qrow_size);
                        char * dst_ptr = (char *) dst->data + i12 * nb2;
                        float out;
                        q1t_gemv<1>(ne00, &out, xp, y);
                        *(float *) (dst_ptr + r * nb0) = out;
                    }
                }

                UNUSED(scratch_size);
                UNUSED(src1);
                return;
            }
        }

        // ---- single-column (token generation) GEMV ----
        const int     gr    = q1t_gemv_rows();
        const int64_t ngemv = ne01 / gr;

        const int64_t gemv_chunks_per_thread = ggml_is_numa() ? 1 : 4;
        const int64_t gemv_groups_per_chunk  = MAX(1, (ngemv + nth * gemv_chunks_per_thread - 1) / (nth * gemv_chunks_per_thread));
        const int64_t gemv_nchunk            = (ngemv + gemv_groups_per_chunk - 1) / gemv_groups_per_chunk;

        int gemv_chunk = ith;

        while (gemv_chunk < gemv_nchunk) {
            const int64_t g0 = gemv_chunk * gemv_groups_per_chunk;
            const int64_t g1 = MIN(g0 + gemv_groups_per_chunk, ngemv);

            for (int64_t g = g0; g < g1; ++g) {
                const int64_t r = g * gr;

                const block_q1_0 * x[Q1T_GEMV_ROWS_MAX];
                for (int rr = 0; rr < gr; ++rr) {
                    x[rr] = (const block_q1_0 *) ((const char *) src0->data + (r + rr) * nb01);
                }

                for (int64_t i12 = 0; i12 < ne12; ++i12) {
                    const block_q8_128 * y = (const block_q8_128 *) (q8 + i12 * qrow_size);
                    char * dst_ptr = (char *) dst->data + i12 * nb2;

                    float out[Q1T_GEMV_ROWS_MAX];
                    switch (gr) {
                        case 1:  q1t_gemv<1>(ne00, out, x, y); break;
                        case 2:  q1t_gemv<2>(ne00, out, x, y); break;
                        default: q1t_gemv<4>(ne00, out, x, y); break;
                    }

                    for (int rr = 0; rr < gr; ++rr) {
                        *(float *) (dst_ptr + (r + rr) * nb0) = out[rr];
                    }
                }
            }

            gemv_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }

        // trailing rows that did not fill a full group
        const int64_t gemv_tail = ngemv * gr;
        for (int64_t r = gemv_tail + ith; r < ne01; r += nth) {
            const block_q1_0 * xp[1] = { (const block_q1_0 *) ((const char *) src0->data + r * nb01) };

            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                const block_q8_128 * y = (const block_q8_128 *) (q8 + i12 * qrow_size);
                char * dst_ptr = (char *) dst->data + i12 * nb2;

                float out;
                q1t_gemv<1>(ne00, &out, xp, y);
                *(float *) (dst_ptr + r * nb0) = out;
            }
        }

        UNUSED(scratch_size);
        UNUSED(src1);
        return;
    }

    // ---- optional 2-value LUT path for batched prefill (table shared across 32-row groups) ----
    if (q1t_lut_enabled() && q1t_lut_pp_enabled() && ne11 >= Q1LUT_PP_MIN && ne01 >= Q1LUT_ROWS && ne00 % QK1_0 == 0) {
        q1lut_codes * lut = q1lut_get_or_alloc(src0);
        if (lut != nullptr) {
            if (!lut->built.load(std::memory_order_acquire)) {
                for (int64_t g = ith; g < lut->ngroups; g += nth) {
                    q1lut_fill_group(lut, g * Q1LUT_ROWS);
                }
                ggml_barrier(params->threadpool);
                if (ith == 0) {
                    lut->built.store(true, std::memory_order_release);
                }
                ggml_barrier(params->threadpool);
            }

            const uint8_t * codes      = lut->codes;
            const int64_t   nb         = ne00 / QK1_0;
            const size_t    lut_row    = (size_t) (ne00 / Q1LUT_GROUP) * Q1LUT_TABLE_LEN;
            uint8_t *       lut_base   = (uint8_t *) q8 + GGML_PAD((size_t) qrow_size * ne11 * ne12, 64);
            const int64_t   nkg        = ne00 / Q1LUT_GROUP;
            const int64_t   lut_ngroups = ne01 / Q1LUT_ROWS;
            const size_t    per_group  = (size_t) nkg * 16;
            const int64_t   ncols      = MIN(ne11, (int64_t) Q1LUT_PP_COLS);

            for (int64_t c0 = 0; c0 < ne11; c0 += ncols) {
                const int64_t nc = MIN(ncols, ne11 - c0);

                // build nc tables per batch element (parallel over K groups)
                for (int64_t i12 = 0; i12 < ne12; ++i12) {
                    for (int64_t cl = 0; cl < nc; ++cl) {
                        const block_q8_128 * yrow = (const block_q8_128 *)
                            (q8 + (i12 * ne11 + c0 + cl) * qrow_size);
                        uint8_t * tbl = lut_base + (size_t) (i12 * nc + cl) * lut_row;
                        for (int64_t kg = ith; kg < nkg; kg += nth) {
                            const int64_t ib = kg / 32;
                            const int64_t kk = kg % 32;
                            q1lut_build_group(yrow[ib].qs + Q1LUT_GROUP * kk, tbl + Q1LUT_TABLE_LEN * kg);
                        }
                    }
                }
                ggml_barrier(params->threadpool);

                // static partition: the shared work-stealing counter cannot be reused across
                // chunks because it is only reset once per op
                const int64_t total = lut_ngroups * ne12;
                for (int64_t a = ith; a < total; a += nth) {
                    const int64_t i12 = a / lut_ngroups;
                    const int64_t g   = a % lut_ngroups;
                    const uint8_t *   gcodes = codes + (size_t) g * per_group;
                    const ggml_half * gd0    = lut->d0 + (size_t) g * nb * Q1LUT_ROWS;
                    for (int64_t cl = 0; cl < nc; ++cl) {
                        const uint8_t * tbl = lut_base + (size_t) (i12 * nc + cl) * lut_row;
                        const block_q8_128 * yq8 = (const block_q8_128 *)
                            (q8 + (i12 * ne11 + c0 + cl) * qrow_size);
                        char * dst_ptr = (char *) dst->data + i12 * nb2
                            + (c0 + cl) * nb1 + (size_t) (g * Q1LUT_ROWS) * nb0;
                        q1lut_gemv32(ne00, dst_ptr, nb0, gcodes, tbl, gd0, yq8);
                    }
                }
                ggml_barrier(params->threadpool); // next chunk reuses lut_base
            }

            // trailing rows (ne01 % 32) via vec_dot
            const int64_t lut_r_tail = lut_ngroups * Q1LUT_ROWS;
            if (lut_r_tail < ne01) {
                ggml_vec_dot_t vec_dot = ggml_get_type_traits_cpu(GGML_TYPE_Q1_0)->vec_dot;
                for (int64_t r = lut_r_tail + ith; r < ne01; r += nth) {
                    const char * x_row = (const char *) src0->data + r * nb01;
                    for (int64_t i12 = 0; i12 < ne12; ++i12) {
                        const char * src1_ptr = (const char *) params->wdata + i12 * nbw2;
                        char *       dst_ptr  = (char *) dst->data + i12 * nb2;
                        for (int64_t c = 0; c < ne11; ++c) {
                            float val;
                            vec_dot(ne00, &val, 0, x_row, 0, src1_ptr + c * nbw1, 0, 1);
                            *(float *) (dst_ptr + c * nb1 + r * nb0) = val;
                        }
                    }
                }
            }

            UNUSED(scratch_size);
            UNUSED(src1);
            return;
        }
    }

    ggml_vec_dot_t vec_dot = ggml_get_type_traits_cpu(GGML_TYPE_Q1_0)->vec_dot;

    const int64_t chunks_per_thread = ggml_is_numa() ? 1 : 4;
    const int64_t groups_per_chunk  = MAX(1, (ngroups + nth * chunks_per_thread - 1) / (nth * chunks_per_thread));
    const int64_t nchunk            = (ngroups + groups_per_chunk - 1) / groups_per_chunk;

    int current_chunk = ith;

    while (current_chunk < nchunk) {
        const int64_t g0 = current_chunk * groups_per_chunk;
        const int64_t g1 = MIN(g0 + groups_per_chunk, ngroups);

        for (int64_t g = g0; g < g1; ++g) {
            const int64_t r = g * Q1T_ROWS;

            const block_q1_0 * x[Q1T_ROWS];
            for (int rr = 0; rr < Q1T_ROWS; ++rr) {
                x[rr] = (const block_q1_0 *) ((const char *) src0->data + (r + rr) * nb01);
            }

            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                char *       dst_ptr  = (char *) dst->data + i12 * nb2;

                int64_t c = 0;
                for (; c + Q1T_COLS <= ne11; c += Q1T_COLS) {
                    const block_q8_128 * y[Q1T_COLS];
                    for (int t = 0; t < Q1T_COLS; ++t) {
                        y[t] = (const block_q8_128 *) (q8 + ((i12 * ne11) + c + t) * qrow_size);
                    }

                    float out[Q1T_ROWS * Q1T_COLS];
                    q1t_dot<Q1T_ROWS, Q1T_COLS>(ne00, out, x, y);

                    for (int rr = 0; rr < Q1T_ROWS; ++rr) {
                        for (int t = 0; t < Q1T_COLS; ++t) {
                            *(float *) (dst_ptr + (c + t) * nb1 + (r + rr) * nb0) = out[rr * Q1T_COLS + t];
                        }
                    }
                }

                // trailing columns, plain vec_dot on the Q8_0 work buffer
                for (; c < ne11; ++c) {
                    const char * src1_ptr = (const char *) params->wdata + i12 * nbw2;
                    for (int rr = 0; rr < Q1T_ROWS; ++rr) {
                        float val;
                        vec_dot(ne00, &val, 0, x[rr], 0, src1_ptr + c * nbw1, 0, 1);
                        *(float *) (dst_ptr + c * nb1 + (r + rr) * nb0) = val;
                    }
                }
            }
        }

        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
#else
    UNUSED(ngroups);
    UNUSED(q8);
    UNUSED(qrow_size);
#endif

    // trailing rows that did not fill a full group
    const int64_t r_tail = ngroups * Q1T_ROWS;
    if (r_tail < ne01) {
        ggml_vec_dot_t vec_dot = ggml_get_type_traits_cpu(GGML_TYPE_Q1_0)->vec_dot;

        for (int64_t r = r_tail + ith; r < ne01; r += nth) {
            const char * x_row = (const char *) src0->data + r * nb01;

            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                const char * src1_ptr = (const char *) params->wdata + i12 * nbw2;
                char *       dst_ptr  = (char *) dst->data + i12 * nb2;

                for (int64_t c = 0; c < ne11; ++c) {
                    float val;
                    vec_dot(ne00, &val, 0, x_row, 0, src1_ptr + c * nbw1, 0, 1);
                    *(float *) (dst_ptr + c * nb1 + r * nb0) = val;
                }
            }
        }
    }

    UNUSED(scratch_size);
    UNUSED(src1);
}
