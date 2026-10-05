// Standalone microbenchmark comparing Q1_0 dot approaches including the panel scheme.
#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>

typedef uint16_t ggml_half;
#define QK1_0 128
typedef struct { ggml_half d; uint8_t qs[QK1_0/8]; } block_q1_0;
#define QK8_0 32
typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;

static inline float f2f(ggml_half h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h))); }
static inline ggml_half t2h(float f) { return (ggml_half)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f),0),0); }

static const __m256i BYTE_SHUF = _mm256_setr_epi8(
    0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
static const __m256i BIT_MASKS = _mm256_setr_epi8(
    1,2,4,8,16,32,64,(char)-128, 1,2,4,8,16,32,64,(char)-128,
    1,2,4,8,16,32,64,(char)-128, 1,2,4,8,16,32,64,(char)-128);

// decode to +1/-1 signs
static void decode_signs(const block_q1_0 * x, int nb, int8_t * out) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi8(1);
    for (int ib = 0; ib < nb; ++ib) {
        const uint32_t * q32 = (const uint32_t *) x[ib].qs;
        for (int k = 0; k < 4; ++k) {
            __m256i bc = _mm256_set1_epi32((int)q32[k]);
            __m256i sh = _mm256_shuffle_epi8(bc, BYTE_SHUF);
            __m256i m  = _mm256_cmpeq_epi8(_mm256_and_si256(sh, BIT_MASKS), zero);
            _mm256_storeu_si256((__m256i*)(out + ib*128 + k*32), _mm256_or_si256(m, ones));
        }
    }
}
// decode to 0x00 / 0xFF masks
static void decode_masks(const block_q1_0 * x, int nb, int8_t * out) {
    const __m256i zero = _mm256_setzero_si256();
    for (int ib = 0; ib < nb; ++ib) {
        const uint32_t * q32 = (const uint32_t *) x[ib].qs;
        for (int k = 0; k < 4; ++k) {
            __m256i bc = _mm256_set1_epi32((int)q32[k]);
            __m256i sh = _mm256_shuffle_epi8(bc, BYTE_SHUF);
            __m256i m  = _mm256_cmpeq_epi8(_mm256_and_si256(sh, BIT_MASKS), zero);
            _mm256_storeu_si256((__m256i*)(out + ib*128 + k*32), m);
        }
    }
}

static inline float hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v,1);
    lo = _mm_add_ps(lo,hi); lo = _mm_hadd_ps(lo,lo); lo = _mm_hadd_ps(lo,lo);
    return _mm_cvtss_f32(lo);
}

// panel dot with sign_epi8
template<int R>
static void dot_signs(int n, float * s, size_t bs, const block_q1_0 * const * x, const int8_t * const * sg, const block_q8_0 * y) {
    const int nb = n/QK1_0;
    const __m256i ones8 = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1);
    __m256 acc[R];
    for (int r=0;r<R;r++) acc[r]=_mm256_setzero_ps();
    for (int ib=0;ib<nb;ib++) {
        float d0[R]; for(int r=0;r<R;r++) d0[r]=f2f(x[r][ib].d);
        const block_q8_0 * yb=&y[ib*4];
        for (int k=0;k<4;k++) {
            __m256i qy=_mm256_loadu_si256((const __m256i*)yb[k].qs);
            float d1=f2f(yb[k].d);
            for(int r=0;r<R;r++){
                __m256i sm=_mm256_loadu_si256((const __m256i*)(sg[r]+ib*128+k*32));
                __m256i sy=_mm256_sign_epi8(qy,sm);
                __m256i t=_mm256_madd_epi16(_mm256_maddubs_epi16(ones8,sy),ones16);
                acc[r]=_mm256_fmadd_ps(_mm256_set1_ps(d0[r]*d1),_mm256_cvtepi32_ps(t),acc[r]);
            }
        }
    }
    for(int r=0;r<R;r++) s[r*bs]=hsum8(acc[r]);
}
// panel dot with xor/sub masks
template<int R>
static void dot_masks(int n, float * s, size_t bs, const block_q1_0 * const * x, const int8_t * const * sg, const block_q8_0 * y) {
    const int nb = n/QK1_0;
    const __m256i ones8 = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1);
    __m256 acc[R];
    for (int r=0;r<R;r++) acc[r]=_mm256_setzero_ps();
    for (int ib=0;ib<nb;ib++) {
        float d0[R]; for(int r=0;r<R;r++) d0[r]=f2f(x[r][ib].d);
        const block_q8_0 * yb=&y[ib*4];
        for (int k=0;k<4;k++) {
            __m256i qy=_mm256_loadu_si256((const __m256i*)yb[k].qs);
            float d1=f2f(yb[k].d);
            for(int r=0;r<R;r++){
                __m256i m=_mm256_loadu_si256((const __m256i*)(sg[r]+ib*128+k*32));
                __m256i sy=_mm256_sub_epi8(_mm256_xor_si256(qy,m),m);
                __m256i t=_mm256_madd_epi16(_mm256_maddubs_epi16(ones8,sy),ones16);
                acc[r]=_mm256_fmadd_ps(_mm256_set1_ps(d0[r]*d1),_mm256_cvtepi32_ps(t),acc[r]);
            }
        }
    }
    for(int r=0;r<R;r++) s[r*bs]=hsum8(acc[r]);
}

// current per-(row,col) kernel for reference
static void vec_dot_cur(int n, float * s, const void * vx, const void * vy) {
    const int nb=n/QK1_0;
    const block_q1_0 * x=(const block_q1_0*)vx; const block_q8_0 * y=(const block_q8_0*)vy;
    const __m256i ones8=_mm256_set1_epi8(1), ones16=_mm256_set1_epi16(1), zero=_mm256_setzero_si256();
    __m256 acc=_mm256_setzero_ps();
    for(int ib=0;ib<nb;ib++){
        float d0=f2f(x[ib].d);
        const uint32_t *q32=(const uint32_t*)x[ib].qs;
        const block_q8_0 *yb=&y[ib*4];
        __m256 ab;
        {   __m256i qy=_mm256_loadu_si256((const __m256i*)yb[0].qs);
            __m256i sm=_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)q32[0]),BYTE_SHUF),BIT_MASKS),zero);
            __m256i sy=_mm256_sub_epi8(_mm256_xor_si256(qy,sm),sm);
            __m256i t=_mm256_madd_epi16(_mm256_maddubs_epi16(ones8,sy),ones16);
            ab=_mm256_mul_ps(_mm256_set1_ps(f2f(yb[0].d)),_mm256_cvtepi32_ps(t)); }
        for(int k=1;k<4;k++){
            __m256i qy=_mm256_loadu_si256((const __m256i*)yb[k].qs);
            __m256i sm=_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)q32[k]),BYTE_SHUF),BIT_MASKS),zero);
            __m256i sy=_mm256_sub_epi8(_mm256_xor_si256(qy,sm),sm);
            __m256i t=_mm256_madd_epi16(_mm256_maddubs_epi16(ones8,sy),ones16);
            ab=_mm256_fmadd_ps(_mm256_set1_ps(f2f(yb[k].d)),_mm256_cvtepi32_ps(t),ab);
        }
        acc=_mm256_fmadd_ps(_mm256_set1_ps(d0),ab,acc);
    }
    *s=hsum8(acc);
}

static constexpr int R = 4;

static double now_s(){using C=std::chrono::high_resolution_clock;return std::chrono::duration<double>(C::now().time_since_epoch()).count();}

int main(int argc,char**argv){
    const int K=(argc>1)?atoi(argv[1]):4096;
    const int M=(argc>2)?atoi(argv[2]):4096;
    const int N=(argc>3)?atoi(argv[3]):32;
    const int iters=(argc>4)?atoi(argv[4]):50;
    const int nb=K/QK1_0;
    block_q1_0 * x=(block_q1_0*)_mm_malloc((size_t)M*nb*sizeof(block_q1_0),64);
    block_q8_0 * y=(block_q8_0*)_mm_malloc((size_t)N*nb*4*sizeof(block_q8_0),64);
    srand(1);
    for(size_t i=0;i<(size_t)M*nb;i++){x[i].d=t2h(0.001f+0.001f*(rand()%100));for(int j=0;j<16;j++)x[i].qs[j]=rand()&0xff;}
    for(int i=0;i<N*nb*4;i++){y[i].d=t2h(0.001f+0.001f*(rand()%100));for(int j=0;j<32;j++)y[i].qs[j]=(int8_t)(rand()%255-127);}

    
    int8_t * panel=(int8_t*)_mm_malloc((size_t)R*K,64);

    auto bench=[&](const char*name, auto fn){
        float sink=0; double t0=now_s();
        for(int it=0;it<iters;it++) fn(sink);
        double t1=now_s();
        double macs=(double)M*K*N*iters;
        printf("%-16s %8.3f ms  %8.2f GMAC/s  (sink=%g)\n",name,(t1-t0)*1e3,macs/(t1-t0)/1e9,sink);
    };

    printf("K=%d M=%d N=%d iters=%d\n",K,M,N,iters);

    bench("current-tiled", [&](float&sink){
        for(int iir1=0;iir1<N;iir1+=16){
            for(int iir0=0;iir0<M;iir0+=16){
                for(int ir1=iir1; ir1<iir1+16 && ir1<N; ir1++){
                    const block_q8_0 * ycol=y+(size_t)ir1*nb*4;
                    for(int ir0=iir0; ir0<iir0+16 && ir0<M; ir0++){
                        float v; vec_dot_cur(K,&v,x+(size_t)ir0*nb,ycol); sink+=v;
                    }
                }
            }
        }
    });
    bench("current", [&](float&sink){
        for(int r=0;r<M;r++) for(int c=0;c<N;c++){ float v; vec_dot_cur(K,&v,x+(size_t)r*nb,y+(size_t)c*nb*4); sink+=v; }
    });
    bench("panel-signs", [&](float&sink){
        for(int r=0;r<M;r+=R){
            const block_q1_0 *xp[R]; const int8_t *sp[R];
            for(int rr=0;rr<R;rr++){xp[rr]=x+(size_t)(r+rr)*nb;sp[rr]=panel+(size_t)rr*K;decode_signs(xp[rr],nb,panel+(size_t)rr*K);}
            for(int c=0;c<N;c++){ float tmp[R]; dot_signs<R>(K,tmp,1,(const block_q1_0 * const *)xp,(const int8_t * const *)sp,y+(size_t)c*nb*4); for(int rr=0;rr<R;rr++)sink+=tmp[rr]; }
        }
    });
    bench("panel-masks", [&](float&sink){
        for(int r=0;r<M;r+=R){
            const block_q1_0 *xp[R]; const int8_t *sp[R];
            for(int rr=0;rr<R;rr++){xp[rr]=x+(size_t)(r+rr)*nb;sp[rr]=panel+(size_t)rr*K;decode_masks(xp[rr],nb,panel+(size_t)rr*K);}
            for(int c=0;c<N;c++){ float tmp[R]; dot_masks<R>(K,tmp,1,(const block_q1_0 * const *)xp,(const int8_t * const *)sp,y+(size_t)c*nb*4); for(int rr=0;rr<R;rr++)sink+=tmp[rr]; }
        }
    });
    // verify masks vs current on first group
    {
        const block_q1_0 *xp[R]; const int8_t *sp[R];
        for(int rr=0;rr<R;rr++){xp[rr]=x+(size_t)rr*nb;sp[rr]=panel+(size_t)rr*K;}
        for(int rr=0;rr<R;rr++)decode_masks(xp[rr],nb,panel+(size_t)rr*K);
        float a[R],b[R];
        dot_masks<R>(K,a,1,(const block_q1_0 * const *)xp,(const int8_t * const *)sp,y);
        for(int rr=0;rr<R;rr++) vec_dot_cur(K,&b[rr],xp[rr],y);
        printf("check masks diff: %.3e %.3e %.3e %.3e\n",fabs(a[0]-b[0]),fabs(a[1]-b[1]),fabs(a[2]-b[2]),fabs(a[3]-b[3]));
    }
    return 0;
}
