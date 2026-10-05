// Q1_0 panel variants: mask (xor/sub), and the 0/2 "bias" trick using vpmaddubsw directly.
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

static inline float f2f(ggml_half h){return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));}
static inline ggml_half t2h(float f){return (ggml_half)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f),0),0);}

static const __m256i BYTE_SHUF=_mm256_setr_epi8(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3);
static const __m256i BIT_MASKS=_mm256_setr_epi8(
    1,2,4,8,16,32,64,(char)-128,1,2,4,8,16,32,64,(char)-128,
    1,2,4,8,16,32,64,(char)-128,1,2,4,8,16,32,64,(char)-128);

static inline float hsum8(__m256 v){__m128 lo=_mm256_castps256_ps128(v),hi=_mm256_extractf128_ps(v,1);lo=_mm_add_ps(lo,hi);lo=_mm_hadd_ps(lo,lo);lo=_mm_hadd_ps(lo,lo);return _mm_cvtss_f32(lo);}

// decode to 0x00/0xFF masks
static void decode_masks(const block_q1_0 *x,int nb,int8_t*out){
    const __m256i zero=_mm256_setzero_si256();
    for(int ib=0;ib<nb;ib++){const uint32_t*q=(const uint32_t*)x[ib].qs;
        for(int k=0;k<4;k++){__m256i bc=_mm256_set1_epi32((int)q[k]);__m256i sh=_mm256_shuffle_epi8(bc,BYTE_SHUF);
            __m256i m=_mm256_cmpeq_epi8(_mm256_and_si256(sh,BIT_MASKS),zero);
            _mm256_storeu_si256((__m256i*)(out+ib*128+k*32),m);}}
}
// decode to 0x00/0x02 (panel = 2 for bit set, 0 for bit clear)
static void decode_bias(const block_q1_0 *x,int nb,int8_t*out){
    const __m256i zero=_mm256_setzero_si256(); const __m256i twos=_mm256_set1_epi8(2);
    for(int ib=0;ib<nb;ib++){const uint32_t*q=(const uint32_t*)x[ib].qs;
        for(int k=0;k<4;k++){__m256i bc=_mm256_set1_epi32((int)q[k]);__m256i sh=_mm256_shuffle_epi8(bc,BYTE_SHUF);
            __m256i m=_mm256_cmpeq_epi8(_mm256_and_si256(sh,BIT_MASKS),zero); // 0xFF where bit==0
            _mm256_storeu_si256((__m256i*)(out+ib*128+k*32),_mm256_andnot_si256(m,twos));}}
}
static inline int qsum_block(const block_q8_0*yb){
    const __m256i ones8=_mm256_set1_epi8(1),ones16=_mm256_set1_epi16(1);
    __m256i qy=_mm256_loadu_si256((const __m256i*)yb->qs);
    __m256i t=_mm256_madd_epi16(_mm256_maddubs_epi16(ones8,qy),ones16);
    __m128i x=_mm_add_epi32(_mm256_castsi256_si128(t),_mm256_extracti128_si256(t,1));
    x=_mm_hadd_epi32(x,x); x=_mm_hadd_epi32(x,x);
    return _mm_cvtsi128_si32(x);
}

// panel dot, mask variant
template<int R>
static void dot_masks(int n,float*s,const block_q1_0*const*x,const int8_t*const*sg,const block_q8_0*y){
    const int nb=n/QK1_0; const __m256i ones8=_mm256_set1_epi8(1),ones16=_mm256_set1_epi16(1);
    __m256 acc[R]; for(int r=0;r<R;r++)acc[r]=_mm256_setzero_ps();
    for(int ib=0;ib<nb;ib++){float d0[R];for(int r=0;r<R;r++)d0[r]=f2f(x[r][ib].d);
        const block_q8_0*yb=&y[ib*4];
        for(int k=0;k<4;k++){__m256i qy=_mm256_loadu_si256((const __m256i*)yb[k].qs);float d1=f2f(yb[k].d);
            for(int r=0;r<R;r++){__m256i m=_mm256_loadu_si256((const __m256i*)(sg[r]+ib*128+k*32));
                __m256i sy=_mm256_sub_epi8(_mm256_xor_si256(qy,m),m);
                __m256i t=_mm256_madd_epi16(_mm256_maddubs_epi16(ones8,sy),ones16);
                acc[r]=_mm256_fmadd_ps(_mm256_set1_ps(d0[r]*d1),_mm256_cvtepi32_ps(t),acc[r]);}}}
    for(int r=0;r<R;r++)s[r]=hsum8(acc[r]);
}

// panel dot, bias variant; C holds per block the precomputed sum_k d1_k*qsum_k
template<int R>
static void dot_bias(int n,float*s,const block_q1_0*const*x,const int8_t*const*sg,const block_q8_0*y,const float*C){
    const int nb=n/QK1_0; const __m256i ones8=_mm256_set1_epi8(1),ones16=_mm256_set1_epi16(1);
    __m256 acc[R]; for(int r=0;r<R;r++)acc[r]=_mm256_setzero_ps();
    for(int ib=0;ib<nb;ib++){float d0[R];for(int r=0;r<R;r++)d0[r]=f2f(x[r][ib].d);
        const block_q8_0*yb=&y[ib*4];
        for(int k=0;k<4;k++){__m256i qy=_mm256_loadu_si256((const __m256i*)yb[k].qs);float d1=f2f(yb[k].d);
            for(int r=0;r<R;r++){__m256i p=_mm256_loadu_si256((const __m256i*)(sg[r]+ib*128+k*32));
                __m256i t=_mm256_madd_epi16(_mm256_maddubs_epi16(p,qy),ones16);
                acc[r]=_mm256_fmadd_ps(_mm256_set1_ps(d0[r]*d1),_mm256_cvtepi32_ps(t),acc[r]);}}
        const float c=C[ib];
        for(int r=0;r<R;r++) acc[r]=_mm256_fnmadd_ps(_mm256_set1_ps(d0[r]*c),_mm256_set1_ps(1.0f),acc[r]);
    }
    for(int r=0;r<R;r++)s[r]=hsum8(acc[r]);
}

static constexpr int R = 4;
static double now_s(){using C=std::chrono::high_resolution_clock;return std::chrono::duration<double>(C::now().time_since_epoch()).count();}

int main(int argc,char**argv){
    const int K=(argc>1)?atoi(argv[1]):4096;
    const int M=(argc>2)?atoi(argv[2]):4096;
    const int N=(argc>3)?atoi(argv[3]):32;
    const int iters=(argc>4)?atoi(argv[4]):10;

    const int nb=K/QK1_0;
    block_q1_0*x=(block_q1_0*)_mm_malloc((size_t)M*nb*sizeof(block_q1_0),64);
    block_q8_0*y=(block_q8_0*)_mm_malloc((size_t)N*nb*4*sizeof(block_q8_0),64);
    srand(1);
    for(size_t i=0;i<(size_t)M*nb;i++){x[i].d=t2h(0.001f+0.001f*(rand()%100));for(int j=0;j<16;j++)x[i].qs[j]=rand()&0xff;}
    for(int i=0;i<N*nb*4;i++){y[i].d=t2h(0.001f+0.001f*(rand()%100));for(int j=0;j<32;j++)y[i].qs[j]=(int8_t)(rand()%255-127);}
    int8_t*panel=(int8_t*)_mm_malloc((size_t)R*K,64);
    float*C=(float*)_mm_malloc((size_t)N*nb*sizeof(float),64);
    // precompute C per column
    for(int c=0;c<N;c++)for(int ib=0;ib<nb;ib++){float s=0;for(int k=0;k<4;k++){const block_q8_0*yb=&y[((size_t)c*nb+ib)*4+k];s+=f2f(yb->d)*qsum_block(yb);}C[(size_t)c*nb+ib]=s;}

    auto bench=[&](const char*name,auto fn){float sink=0;double t0=now_s();for(int it=0;it<iters;it++)fn(sink);double t1=now_s();printf("%-12s %8.2f GMAC/s (sink=%g)\n",name,(double)M*K*N*iters/(t1-t0)/1e9,sink);};
    printf("K=%d M=%d N=%d R=%d iters=%d\n",K,M,N,R,iters);

    bench("panel-mask",[&](float&sink){for(int r=0;r<M;r+=R){const block_q1_0*xp[R];const int8_t*sp[R];
        for(int rr=0;rr<R;rr++){xp[rr]=x+(size_t)(r+rr)*nb;sp[rr]=panel+(size_t)rr*K;decode_masks(xp[rr],nb,panel+(size_t)rr*K);}
        for(int c=0;c<N;c++){float tmp[R];dot_masks<R>(K,tmp,(const block_q1_0*const*)xp,(const int8_t*const*)sp,y+(size_t)c*nb*4);for(int rr=0;rr<R;rr++)sink+=tmp[rr];}}});
    bench("panel-bias",[&](float&sink){for(int r=0;r<M;r+=R){const block_q1_0*xp[R];const int8_t*sp[R];
        for(int rr=0;rr<R;rr++){xp[rr]=x+(size_t)(r+rr)*nb;sp[rr]=panel+(size_t)rr*K;decode_bias(xp[rr],nb,panel+(size_t)rr*K);}
        for(int c=0;c<N;c++){float tmp[R];dot_bias<R>(K,tmp,(const block_q1_0*const*)xp,(const int8_t*const*)sp,y+(size_t)c*nb*4,C+(size_t)c*nb);for(int rr=0;rr<R;rr++)sink+=tmp[rr];}}});
    // correctness: compare bias vs mask on one group
    {
        const block_q1_0 *xp[R]; const int8_t *sp[R];
        for(int rr=0;rr<R;rr++){xp[rr]=x+(size_t)rr*nb;sp[rr]=panel+(size_t)rr*K;}
        for(int rr=0;rr<R;rr++) decode_masks(xp[rr],nb,panel+(size_t)rr*K);
        int8_t *panel_b=(int8_t*)_mm_malloc((size_t)R*K,64);
        for(int rr=0;rr<R;rr++) decode_bias(xp[rr],nb,panel_b+(size_t)rr*K);
        const int8_t *spb[R]; for(int rr=0;rr<R;rr++) spb[rr]=panel_b+(size_t)rr*K;
        float a[R],b[R];
        dot_masks<R>(K,a,(const block_q1_0*const*)xp,(const int8_t*const*)sp,y);
        dot_bias<R>(K,b,(const block_q1_0*const*)xp,(const int8_t*const*)spb,y,C);
        printf("check diff: %.3e %.3e %.3e %.3e\n",fabs(a[0]-b[0]),fabs(a[1]-b[1]),fabs(a[2]-b[2]),fabs(a[3]-b[3]));
    }

    return 0;
}
