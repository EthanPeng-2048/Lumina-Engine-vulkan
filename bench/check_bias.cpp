// Minimal check of the "masked sum" formulation:
//   sign_j in {+1,-1}, panel_j in {0,2} (2 iff sign=+1)
//   sum sign*q = sum panel*q - sum q
#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

typedef uint16_t ggml_half;
#define QK1_0 128
typedef struct { ggml_half d; uint8_t qs[QK1_0/8]; } block_q1_0;
#define QK8_0 32
typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;

static inline float f2f(ggml_half h){return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));}
static inline ggml_half t2h(float f){return (ggml_half)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f),0),0);}

int main(){
    srand(1234);
    block_q1_0 x;
    x.d=t2h(0.013f);
    for(int j=0;j<16;j++) x.qs[j]=rand()&0xff;
    block_q8_0 y;
    y.d=t2h(0.021f);
    for(int j=0;j<32;j++) y.qs[j]=(int8_t)(rand()%255-127);

    const __m256i BYTE_SHUF=_mm256_setr_epi8(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3);
    const __m256i BIT_MASKS=_mm256_setr_epi8(1,2,4,8,16,32,64,(char)-128,1,2,4,8,16,32,64,(char)-128,
                                             1,2,4,8,16,32,64,(char)-128,1,2,4,8,16,32,64,(char)-128);

    // build panel = 0 or 2 for one 32-wide sub-block (k=0)
    uint32_t q32=((const uint32_t*)x.qs)[0];
    __m256i bc=_mm256_set1_epi32((int)q32);
    __m256i sh=_mm256_shuffle_epi8(bc,BYTE_SHUF);
    __m256i m =_mm256_cmpeq_epi8(_mm256_and_si256(sh,BIT_MASKS),_mm256_setzero_si256()); // 0xFF where bit==0
    __m256i panel=_mm256_andnot_si256(m,_mm256_set1_epi8(2)); // 2 where bit==1

    __m256i qy=_mm256_loadu_si256((const __m256i*)y.qs);
    __m256i ones8=_mm256_set1_epi8(1),ones16=_mm256_set1_epi16(1);
    __m256i raw16=_mm256_maddubs_epi16(panel,qy);
    __m256i raw32=_mm256_madd_epi16(raw16,ones16);
    __m128i r=_mm_add_epi32(_mm256_castsi256_si128(raw32),_mm256_extracti128_si256(raw32,1));
    r=_mm_hadd_epi32(r,r);r=_mm_hadd_epi32(r,r);
    int raw=_mm_cvtsi128_si32(r);

    __m256i q16=_mm256_maddubs_epi16(ones8,qy);
    __m256i q32v=_mm256_madd_epi16(q16,ones16);
    __m128i qq=_mm_add_epi32(_mm256_castsi256_si128(q32v),_mm256_extracti128_si256(q32v,1));
    qq=_mm_hadd_epi32(qq,qq);qq=_mm_hadd_epi32(qq,qq);
    int qsum=_mm_cvtsi128_si32(qq);

    int64_t scalar=0;
    for(int j=0;j<32;j++){int bit=(q32>>j)&1;scalar+=(bit?y.qs[j]:-y.qs[j]);}
    printf("raw=%d qsum=%d raw-qsum=%d scalar=%lld  %s\n",raw,qsum,raw-qsum,(long long)scalar,(raw-qsum==scalar)?"OK":"MISMATCH");
    // full 128
    float ref=f2f(x.d)*0.0f;
    for(int k=0;k<4;k++){
        __m256i bck=_mm256_set1_epi32((int)((const uint32_t*)x.qs)[k]);
        __m256i shk=_mm256_shuffle_epi8(bck,BYTE_SHUF);
        __m256i mk=_mm256_cmpeq_epi8(_mm256_and_si256(shk,BIT_MASKS),_mm256_setzero_si256());
        __m256i pk=_mm256_andnot_si256(mk,_mm256_set1_epi8(2));
        __m256i qyk=_mm256_loadu_si256((const __m256i*)y.qs);
        __m256i rk=_mm256_madd_epi16(_mm256_maddubs_epi16(pk,qyk),ones16);
        __m128i rr=_mm_add_epi32(_mm256_castsi256_si128(rk),_mm256_extracti128_si256(rk,1));
        rr=_mm_hadd_epi32(rr,rr);rr=_mm_hadd_epi32(rr,rr);
        int rawi=_mm_cvtsi128_si32(rr);
        __m256i qk=_mm256_madd_epi16(_mm256_maddubs_epi16(ones8,qyk),ones16);
        __m128i qx=_mm_add_epi32(_mm256_castsi256_si128(qk),_mm256_extracti128_si256(qk,1));
        qx=_mm_hadd_epi32(qx,qx);qx=_mm_hadd_epi32(qx,qx);
        int qsi=_mm_cvtsi128_si32(qx);
        // NOTE: in the real kernel the 4 sub-blocks use 4 different q8 blocks; here we reuse y for the demo
        (void)rawi;(void)qsi;
    }
    (void)ref;
    return 0;
}
