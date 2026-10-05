// Q1_0 x Q8_0 AVX2 kernel variants
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

static const __m256i BYTE_SHUF = _mm256_setr_epi8(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3);
static const __m256i BIT_MASKS = _mm256_setr_epi8(
    1,2,4,8,16,32,64,(char)-128,1,2,4,8,16,32,64,(char)-128,
    1,2,4,8,16,32,64,(char)-128,1,2,4,8,16,32,64,(char)-128);

static inline float hsum8(__m256 v) {
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); lo=_mm_hadd_ps(lo,lo); lo=_mm_hadd_ps(lo,lo);
    return _mm_cvtss_f32(lo);
}

// A: current
__declspec(noinline) void vA(int n,float*s,const void*vx,const void*vy){
    const int nb=n/QK1_0; const block_q1_0*x=(const block_q1_0*)vx; const block_q8_0*y=(const block_q8_0*)vy;
    const __m256i ones8=_mm256_set1_epi8(1),ones16=_mm256_set1_epi16(1),zero=_mm256_setzero_si256();
    __m256 acc=_mm256_setzero_ps();
    for(int ib=0;ib<nb;ib++){
        float d0=f2f(x[ib].d); const uint32_t*q32=(const uint32_t*)x[ib].qs; const block_q8_0*yb=&y[ib*4];
        __m256 ab;
        { __m256i qy=_mm256_loadu_si256((const __m256i*)yb[0].qs);
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

// B: sign_epi8 with branchless +1/-1
__declspec(noinline) void vB(int n,float*s,const void*vx,const void*vy){
    const int nb=n/QK1_0; const block_q1_0*x=(const block_q1_0*)vx; const block_q8_0*y=(const block_q8_0*)vy;
    const __m256i ones8=_mm256_set1_epi8(1),ones16=_mm256_set1_epi16(1),zero=_mm256_setzero_si256();
    __m256 acc=_mm256_setzero_ps();
    for(int ib=0;ib<nb;ib++){
        float d0=f2f(x[ib].d); const uint32_t*q32=(const uint32_t*)x[ib].qs; const block_q8_0*yb=&y[ib*4];
        __m256 ab=_mm256_setzero_ps();
        for(int k=0;k<4;k++){
            __m256i qy=_mm256_loadu_si256((const __m256i*)yb[k].qs);
            __m256i m=_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)q32[k]),BYTE_SHUF),BIT_MASKS);
            __m256i sg=_mm256_or_si256(_mm256_cmpeq_epi8(m,zero),ones8);
            __m256i sy=_mm256_sign_epi8(qy,sg);
            __m256i t=_mm256_madd_epi16(_mm256_maddubs_epi16(ones8,sy),ones16);
            ab=_mm256_fmadd_ps(_mm256_set1_ps(f2f(yb[k].d)),_mm256_cvtepi32_ps(t),ab);
        }
        acc=_mm256_fmadd_ps(_mm256_set1_ps(d0),ab,acc);
    }
    *s=hsum8(acc);
}

// C: current, 2 blocks unrolled
__declspec(noinline) void vC(int n,float*s,const void*vx,const void*vy){
    const int nb=n/QK1_0; const block_q1_0*x=(const block_q1_0*)vx; const block_q8_0*y=(const block_q8_0*)vy;
    const __m256i ones8=_mm256_set1_epi8(1),ones16=_mm256_set1_epi16(1),zero=_mm256_setzero_si256();
    __m256 acc=_mm256_setzero_ps();
    int ib=0;
    for(;ib+2<=nb;ib+=2){
        __m256 ab0=_mm256_setzero_ps(), ab1=_mm256_setzero_ps();
        const uint32_t*q0=(const uint32_t*)x[ib].qs, *q1=(const uint32_t*)x[ib+1].qs;
        const block_q8_0*yb0=&y[ib*4], *yb1=&y[(ib+1)*4];
        for(int k=0;k<4;k++){
            __m256i a0=_mm256_loadu_si256((const __m256i*)yb0[k].qs);
            __m256i a1=_mm256_loadu_si256((const __m256i*)yb1[k].qs);
            __m256i s0=_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)q0[k]),BYTE_SHUF),BIT_MASKS),zero);
            __m256i s1=_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)q1[k]),BYTE_SHUF),BIT_MASKS),zero);
            __m256i z0=_mm256_sub_epi8(_mm256_xor_si256(a0,s0),s0);
            __m256i z1=_mm256_sub_epi8(_mm256_xor_si256(a1,s1),s1);
            __m256i t0=_mm256_madd_epi16(_mm256_maddubs_epi16(ones8,z0),ones16);
            __m256i t1=_mm256_madd_epi16(_mm256_maddubs_epi16(ones8,z1),ones16);
            ab0=_mm256_fmadd_ps(_mm256_set1_ps(f2f(yb0[k].d)),_mm256_cvtepi32_ps(t0),ab0);
            ab1=_mm256_fmadd_ps(_mm256_set1_ps(f2f(yb1[k].d)),_mm256_cvtepi32_ps(t1),ab1);
        }
        acc=_mm256_fmadd_ps(_mm256_set1_ps(f2f(x[ib].d)),ab0,acc);
        acc=_mm256_fmadd_ps(_mm256_set1_ps(f2f(x[ib+1].d)),ab1,acc);
    }
    for(;ib<nb;ib++){
        float d0=f2f(x[ib].d); const uint32_t*q32=(const uint32_t*)x[ib].qs; const block_q8_0*yb=&y[ib*4];
        __m256 ab=_mm256_setzero_ps();
        for(int k=0;k<4;k++){
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

// D: current with software prefetch
__declspec(noinline) void vD(int n,float*s,const void*vx,const void*vy){
    const int nb=n/QK1_0; const block_q1_0*x=(const block_q1_0*)vx; const block_q8_0*y=(const block_q8_0*)vy;
    const __m256i ones8=_mm256_set1_epi8(1),ones16=_mm256_set1_epi16(1),zero=_mm256_setzero_si256();
    __m256 acc=_mm256_setzero_ps();
    for(int ib=0;ib<nb;ib++){
        if(ib+4<nb){ _mm_prefetch((const char*)&x[ib+4],_MM_HINT_T0); _mm_prefetch((const char*)&y[(ib+4)*4],_MM_HINT_T0); }
        float d0=f2f(x[ib].d); const uint32_t*q32=(const uint32_t*)x[ib].qs; const block_q8_0*yb=&y[ib*4];
        __m256 ab;
        { __m256i qy=_mm256_loadu_si256((const __m256i*)yb[0].qs);
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


// E: signb + unroll2
__declspec(noinline) void vE(int n,float*s,const void*vx,const void*vy){
    const int nb=n/QK1_0; const block_q1_0*x=(const block_q1_0*)vx; const block_q8_0*y=(const block_q8_0*)vy;
    const __m256i ones8=_mm256_set1_epi8(1),ones16=_mm256_set1_epi16(1),zero=_mm256_setzero_si256();
    __m256 acc=_mm256_setzero_ps();
    int ib=0;
    for(;ib+2<=nb;ib+=2){
        __m256 ab0=_mm256_setzero_ps(),ab1=_mm256_setzero_ps();
        const uint32_t*q0=(const uint32_t*)x[ib].qs,*q1=(const uint32_t*)x[ib+1].qs;
        const block_q8_0*yb0=&y[ib*4],*yb1=&y[(ib+1)*4];
        for(int k=0;k<4;k++){
            __m256i a0=_mm256_loadu_si256((const __m256i*)yb0[k].qs);
            __m256i a1=_mm256_loadu_si256((const __m256i*)yb1[k].qs);
            __m256i g0=_mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)q0[k]),BYTE_SHUF),BIT_MASKS),zero),ones8);
            __m256i g1=_mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)q1[k]),BYTE_SHUF),BIT_MASKS),zero),ones8);
            __m256i z0=_mm256_sign_epi8(a0,g0), z1=_mm256_sign_epi8(a1,g1);
            __m256i t0=_mm256_madd_epi16(_mm256_maddubs_epi16(ones8,z0),ones16);
            __m256i t1=_mm256_madd_epi16(_mm256_maddubs_epi16(ones8,z1),ones16);
            ab0=_mm256_fmadd_ps(_mm256_set1_ps(f2f(yb0[k].d)),_mm256_cvtepi32_ps(t0),ab0);
            ab1=_mm256_fmadd_ps(_mm256_set1_ps(f2f(yb1[k].d)),_mm256_cvtepi32_ps(t1),ab1);
        }
        acc=_mm256_fmadd_ps(_mm256_set1_ps(f2f(x[ib].d)),ab0,acc);
        acc=_mm256_fmadd_ps(_mm256_set1_ps(f2f(x[ib+1].d)),ab1,acc);
    }
    for(;ib<nb;ib++){
        const uint32_t*q32=(const uint32_t*)x[ib].qs; const block_q8_0*yb=&y[ib*4];
        __m256 ab=_mm256_setzero_ps();
        for(int k=0;k<4;k++){
            __m256i qy=_mm256_loadu_si256((const __m256i*)yb[k].qs);
            __m256i sg=_mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)q32[k]),BYTE_SHUF),BIT_MASKS),zero),ones8);
            __m256i t=_mm256_madd_epi16(_mm256_maddubs_epi16(ones8,_mm256_sign_epi8(qy,sg)),ones16);
            ab=_mm256_fmadd_ps(_mm256_set1_ps(f2f(yb[k].d)),_mm256_cvtepi32_ps(t),ab);
        }
        acc=_mm256_fmadd_ps(_mm256_set1_ps(f2f(x[ib].d)),ab,acc);
    }
    *s=hsum8(acc);
}

// F: signb + prefetch
__declspec(noinline) void vF(int n,float*s,const void*vx,const void*vy){
    const int nb=n/QK1_0; const block_q1_0*x=(const block_q1_0*)vx; const block_q8_0*y=(const block_q8_0*)vy;
    const __m256i ones8=_mm256_set1_epi8(1),ones16=_mm256_set1_epi16(1),zero=_mm256_setzero_si256();
    __m256 acc=_mm256_setzero_ps();
    for(int ib=0;ib<nb;ib++){
        if(ib+3<nb){ _mm_prefetch((const char*)&x[ib+3],_MM_HINT_T0); }
        float d0=f2f(x[ib].d); const uint32_t*q32=(const uint32_t*)x[ib].qs; const block_q8_0*yb=&y[ib*4];
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

static double now_s(){using C=std::chrono::high_resolution_clock;return std::chrono::duration<double>(C::now().time_since_epoch()).count();}

int main(int argc,char**argv){
    const int K=(argc>1)?atoi(argv[1]):4096;
    const int M=(argc>2)?atoi(argv[2]):200000;
    const int iters=(argc>3)?atoi(argv[3]):8;
    const int nb=K/QK1_0;
    block_q1_0*x=(block_q1_0*)_mm_malloc((size_t)M*nb*sizeof(block_q1_0),64);
    int nby=nb*4;
    block_q8_0*y=(block_q8_0*)_mm_malloc((size_t)nby*sizeof(block_q8_0),64);
    srand(1);
    for(size_t i=0;i<(size_t)M*nb;i++){x[i].d=t2h(0.001f+0.001f*(rand()%100));for(int j=0;j<16;j++)x[i].qs[j]=rand()&0xff;}
    for(int i=0;i<nby;i++){y[i].d=t2h(0.001f+0.001f*(rand()%100));for(int j=0;j<32;j++)y[i].qs[j]=(int8_t)(rand()%255-127);}

    const char* names[]={"A-current","B-signb","C-unroll2","D-prefetch","E-signb-u2","F-signb-pf"};
    void (*fns[])(int,float*,const void*,const void*)={vA,vB,vC,vD,vE,vF};
    const int NF=6;
    // verify
    {
        float r0[4]; for(int i=0;i<4;i++) fns[i](K,&r0[i],x,y);
        printf("check: A=%.6f B=%.6f C=%.6f D=%.6f\n",r0[0],r0[1],r0[2],r0[3]);
    }
    for(int f=0;f<NF;f++){
        float sink=0;
        for(int r=0;r<M;r++) fns[f](K,&sink,x+(size_t)r*nb,y);
        double t0=now_s();
        for(int it=0;it<iters;it++) for(int r=0;r<M;r++) fns[f](K,&sink,x+(size_t)r*nb,y);
        double t1=now_s();
        printf("%-12s %8.2f GMAC/s  (sink=%g)\n",names[f],(double)M*K*iters/(t1-t0)/1e9,sink);
    }
    return 0;
}
