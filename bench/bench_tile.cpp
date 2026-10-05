// Register-tiled Q1_0 x Q8_0 kernel: expand sign bits once per (row, k-block),
// reuse across T src1 columns held in registers.
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
static inline __m256i expand1(uint32_t qs, int k){
    __m256i bc=_mm256_set1_epi32((int)qs);
    __m256i sh=_mm256_shuffle_epi8(bc,BYTE_SHUF);
    return _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(sh,BIT_MASKS),_mm256_setzero_si256()),_mm256_set1_epi8(1));
}

// reference signb kernel (current)
__declspec(noinline) void vRef(int n,float*s,const void*vx,const void*vy){
    const int nb=n/QK1_0; const block_q1_0*x=(const block_q1_0*)vx; const block_q8_0*y=(const block_q8_0*)vy;
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

// register-tiled: R rows x T columns, out[r*T+t]
template<int R,int T>
__declspec(noinline) void dot_tile(int n,float*out,const block_q1_0*const*x,const block_q8_0*const*y){
    const int nb=n/QK1_0;
    const __m256i ones8=_mm256_set1_epi8(1),ones16=_mm256_set1_epi16(1),zero=_mm256_setzero_si256();
    __m256 acc[R][T];
    for(int r=0;r<R;r++)for(int t=0;t<T;t++)acc[r][t]=_mm256_setzero_ps();
    for(int ib=0;ib<nb;ib++){
        float d0[R]; for(int r=0;r<R;r++)d0[r]=f2f(x[r][ib].d);
        const block_q8_0 *yb[T]; for(int t=0;t<T;t++)yb[t]=&y[t][ib*4];
        for(int k=0;k<4;k++){
            __m256i sg[R];
            for(int r=0;r<R;r++) sg[r]=expand1(((const uint32_t*)x[r][ib].qs)[k],k);
            for(int t=0;t<T;t++){
                __m256i qy=_mm256_loadu_si256((const __m256i*)yb[t][k].qs);
                float d1=f2f(yb[t][k].d);
                for(int r=0;r<R;r++){
                    __m256i sy=_mm256_sign_epi8(qy,sg[r]);
                    __m256i t32=_mm256_madd_epi16(_mm256_maddubs_epi16(ones8,sy),ones16);
                    acc[r][t]=_mm256_fmadd_ps(_mm256_set1_ps(d0[r]*d1),_mm256_cvtepi32_ps(t32),acc[r][t]);
                }
            }
        }
    }
    for(int r=0;r<R;r++)for(int t=0;t<T;t++)out[r*T+t]=hsum8(acc[r][t]);
}

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

    auto bench=[&](const char*name,auto fn){float sink=0;double t0=now_s();for(int it=0;it<iters;it++)fn(sink);double t1=now_s();printf("%-14s %8.2f GMAC/s (sink=%g)\n",name,(double)M*K*N*iters/(t1-t0)/1e9,sink);};

    bench("ref-signb",[&](float&sink){for(int r=0;r<M;r++)for(int c=0;c<N;c++){float v;vRef(K,&v,x+(size_t)r*nb,y+(size_t)c*nb*4);sink+=v;}});
    bench("tile<2,4>",[&](float&sink){float o[8];for(int r=0;r+2<=M;r+=2)for(int c=0;c+4<=N;c+=4){
        const block_q1_0*xp[2]={x+(size_t)r*nb,x+(size_t)(r+1)*nb};const block_q8_0*yp[4]={y+(size_t)c*nb*4,y+(size_t)(c+1)*nb*4,y+(size_t)(c+2)*nb*4,y+(size_t)(c+3)*nb*4};
        dot_tile<2,4>(K,o,(const block_q1_0*const*)xp,(const block_q8_0*const*)yp);for(int i=0;i<8;i++)sink+=o[i];}});
    bench("tile<4,2>",[&](float&sink){float o[8];for(int r=0;r+4<=M;r+=4)for(int c=0;c+2<=N;c+=2){
        const block_q1_0*xp[4]={x+(size_t)r*nb,x+(size_t)(r+1)*nb,x+(size_t)(r+2)*nb,x+(size_t)(r+3)*nb};const block_q8_0*yp[2]={y+(size_t)c*nb*4,y+(size_t)(c+1)*nb*4};
        dot_tile<4,2>(K,o,(const block_q1_0*const*)xp,(const block_q8_0*const*)yp);for(int i=0;i<8;i++)sink+=o[i];}});
    bench("tile<2,8>",[&](float&sink){float o[16];for(int r=0;r+2<=M;r+=2)for(int c=0;c+8<=N;c+=8){
        const block_q1_0*xp[2]={x+(size_t)r*nb,x+(size_t)(r+1)*nb};const block_q8_0*yp[8]={y+(size_t)(c+0)*nb*4,y+(size_t)(c+1)*nb*4,y+(size_t)(c+2)*nb*4,y+(size_t)(c+3)*nb*4,y+(size_t)(c+4)*nb*4,y+(size_t)(c+5)*nb*4,y+(size_t)(c+6)*nb*4,y+(size_t)(c+7)*nb*4};
        dot_tile<2,8>(K,o,(const block_q1_0*const*)xp,(const block_q8_0*const*)yp);for(int i=0;i<16;i++)sink+=o[i];}});
    bench("tile<4,4>",[&](float&sink){float o[16];for(int r=0;r+4<=M;r+=4)for(int c=0;c+4<=N;c+=4){
        const block_q1_0*xp[4]={x+(size_t)r*nb,x+(size_t)(r+1)*nb,x+(size_t)(r+2)*nb,x+(size_t)(r+3)*nb};const block_q8_0*yp[4]={y+(size_t)(c+0)*nb*4,y+(size_t)(c+1)*nb*4,y+(size_t)(c+2)*nb*4,y+(size_t)(c+3)*nb*4};
        dot_tile<4,4>(K,o,(const block_q1_0*const*)xp,(const block_q8_0*const*)yp);for(int i=0;i<16;i++)sink+=o[i];}});

    // correctness: compare tile<2,4> with ref on first 2 rows x 4 cols
    {
        float o[8];
        const block_q1_0*xp[2]={x,x+nb};const block_q8_0*yp[4]={y,y+nb*4,y+2*nb*4,y+3*nb*4};
        dot_tile<2,4>(K,o,(const block_q1_0*const*)xp,(const block_q8_0*const*)yp);
        float maxd=0;
        for(int r=0;r<2;r++)for(int c=0;c<4;c++){float v;vRef(K,&v,x+(size_t)r*nb,y+(size_t)c*nb*4);maxd=fmax(maxd,fabs(v-o[r*4+c]));}
        printf("correctness max diff = %.3e\n",maxd);
    }
    return 0;
}
