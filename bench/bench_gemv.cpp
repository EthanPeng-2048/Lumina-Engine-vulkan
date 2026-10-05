// GEMV (n=1) benchmark: per-row vec_dot vs multi-row GEMV sharing the activation.
#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
static const __m256i BIT_MASKS=_mm256_setr_epi8(1,2,4,8,16,32,64,(char)-128,1,2,4,8,16,32,64,(char)-128,1,2,4,8,16,32,64,(char)-128,1,2,4,8,16,32,64,(char)-128);
static inline float hsum8(__m256 v){__m128 lo=_mm256_castps256_ps128(v),hi=_mm256_extractf128_ps(v,1);lo=_mm_add_ps(lo,hi);lo=_mm_hadd_ps(lo,lo);lo=_mm_hadd_ps(lo,lo);return _mm_cvtss_f32(lo);}
static inline __m256i expand1(uint32_t bits){
    __m256i bc=_mm256_set1_epi32((int)bits);
    __m256i sh=_mm256_shuffle_epi8(bc,BYTE_SHUF);
    return _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(sh,BIT_MASKS),_mm256_setzero_si256()),_mm256_set1_epi8(1));
}

// current generic signb vec_dot (one row, one column)
__declspec(noinline) void dot1(int n,float*s,const block_q1_0*x,const block_q8_0*y){
    const int nb=n/QK1_0;
    const __m256i ones8=_mm256_set1_epi8(1),ones16=_mm256_set1_epi16(1),zero=_mm256_setzero_si256();
    __m256 acc=_mm256_setzero_ps();
    for(int ib=0;ib<nb;ib++){float d0=f2f(x[ib].d);const uint32_t*q32=(const uint32_t*)x[ib].qs;const block_q8_0*yb=&y[ib*4];
        __m256 ab=_mm256_setzero_ps();
        for(int k=0;k<4;k++){
            __m256i qy=_mm256_loadu_si256((const __m256i*)yb[k].qs);
            __m256i sg=_mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)q32[k]),BYTE_SHUF),BIT_MASKS),zero),ones8);
            __m256i t=_mm256_madd_epi16(_mm256_maddubs_epi16(ones8,_mm256_sign_epi8(qy,sg)),ones16);
            ab=_mm256_fmadd_ps(_mm256_set1_ps(f2f(yb[k].d)),_mm256_cvtepi32_ps(t),ab);
        }
        acc=_mm256_fmadd_ps(_mm256_set1_ps(d0),ab,acc);
    }
    *s=hsum8(acc);
}

// multi-row GEMV: R rows share the q8 activation load and d1 conversion
template<int R>
__declspec(noinline) void gemv(int n,float*out,const block_q1_0*const*x,const block_q8_0*y){
    const int nb=n/QK1_0;
    const __m256i ones8=_mm256_set1_epi8(1),ones16=_mm256_set1_epi16(1),zero=_mm256_setzero_si256();
    __m256 acc[R];
    for(int r=0;r<R;r++)acc[r]=_mm256_setzero_ps();
    for(int ib=0;ib<nb;ib++){
        float d0[R]; for(int r=0;r<R;r++)d0[r]=f2f(x[r][ib].d);
        const block_q8_0*yb=&y[ib*4];
        for(int k=0;k<4;k++){
            const __m256i qy=_mm256_loadu_si256((const __m256i*)yb[k].qs);
            const __m256 d1v=_mm256_set1_ps(f2f(yb[k].d));
            for(int r=0;r<R;r++){
                const __m256i sg=expand1(((const uint32_t*)x[r][ib].qs)[k]);
                const __m256i t=_mm256_madd_epi16(_mm256_maddubs_epi16(ones8,_mm256_sign_epi8(qy,sg)),ones16);
                acc[r]=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(d0[r]),d1v),_mm256_cvtepi32_ps(t),acc[r]);
            }
        }
    }
    for(int r=0;r<R;r++)out[r]=hsum8(acc[r]);
}

static double now_s(){using C=std::chrono::high_resolution_clock;return std::chrono::duration<double>(C::now().time_since_epoch()).count();}

int main(int argc,char**argv){
    const int K=(argc>1)?atoi(argv[1]):4096;
    const int M=(argc>2)?atoi(argv[2]):200000;
    const int iters=(argc>3)?atoi(argv[3]):8;
    const int nb=K/QK1_0;
    block_q1_0*x=(block_q1_0*)_mm_malloc((size_t)M*nb*sizeof(block_q1_0),64);
    block_q8_0*y=(block_q8_0*)_mm_malloc((size_t)nb*4*sizeof(block_q8_0),64);
    srand(1);
    for(size_t i=0;i<(size_t)M*nb;i++){x[i].d=t2h(0.001f+0.001f*(rand()%100));for(int j=0;j<16;j++)x[i].qs[j]=rand()&0xff;}
    for(int i=0;i<nb*4;i++){y[i].d=t2h(0.001f+0.001f*(rand()%100));for(int j=0;j<32;j++)y[i].qs[j]=(int8_t)(rand()%255-127);}
    const double wbytes=(double)M*nb*sizeof(block_q1_0);

    auto bench=[&](const char*name,auto fn){float sink=0;double t0=now_s();for(int it=0;it<iters;it++)fn(sink);double t1=now_s();double dt=t1-t0;printf("%-12s %8.2f GMAC/s  %6.2f GB/s (sink=%g)\n",name,(double)M*K*iters/dt/1e9,wbytes*iters/dt/1e9,sink);};

    bench("ref-1row",[&](float&sink){for(int r=0;r<M;r++){float v;dot1(K,&v,x+(size_t)r*nb,y);sink+=v;}});
    bench("gemv<2>",[&](float&sink){float o[2];for(int r=0;r+2<=M;r+=2){const block_q1_0*xp[2]={x+(size_t)r*nb,x+(size_t)(r+1)*nb};gemv<2>(K,o,(const block_q1_0*const*)xp,y);sink+=o[0]+o[1];}});
    bench("gemv<4>",[&](float&sink){float o[4];for(int r=0;r+4<=M;r+=4){const block_q1_0*xp[4]={x+(size_t)r*nb,x+(size_t)(r+1)*nb,x+(size_t)(r+2)*nb,x+(size_t)(r+3)*nb};gemv<4>(K,o,(const block_q1_0*const*)xp,y);sink+=o[0]+o[1]+o[2]+o[3];}});
    bench("gemv<6>",[&](float&sink){float o[6];for(int r=0;r+6<=M;r+=6){const block_q1_0*xp[6]={x+(size_t)(r+0)*nb,x+(size_t)(r+1)*nb,x+(size_t)(r+2)*nb,x+(size_t)(r+3)*nb,x+(size_t)(r+4)*nb,x+(size_t)(r+5)*nb};gemv<6>(K,o,(const block_q1_0*const*)xp,y);for(int i=0;i<6;i++)sink+=o[i];}});
    {float a,b;dot1(K,&a,x,y);float o[4];const block_q1_0*xp[4]={x,x+nb,x+2*nb,x+3*nb};gemv<4>(K,o,(const block_q1_0*const*)xp,y);printf("check: ref=%.6f gemv=%.6f\n",a,o[0]);}
    return 0;
}
