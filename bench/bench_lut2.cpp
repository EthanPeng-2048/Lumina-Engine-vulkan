// Independent 2-value (binary) activation LUT GEMV benchmark.
//
// Idea (inspired only conceptually by table-lookup quantised GEMM, no third-party code):
//   A group of 4 consecutive 1-bit weights has 2^4 = 16 possible sign patterns. For one
//   activation vector we precompute, for every group of 4 activations, a 16-entry table
//       T[c] = sum_{i=0..3} ( bit_i(c) ? +a[i] : -a[i] )
//   so the partial dot product of the group is a single table lookup indexed by the packed
//   weight code.  The table is built once per activation vector and reused for every output
//   row, which is exactly what makes a lookup table pay off for a 2-value weight alphabet.
//
// The table for 4096 activations is (4096/4)*16*2 = 32 KiB of int16, i.e. it is the L1
// working set.  We block K so the live table stays <= 16 KiB (half of the 32 KiB L1d) and
// keep the table resident while sweeping 32 output rows.
//
// Weight layout used by the LUT path ("repacked"): for each group of 32 output rows and each
// group of 4 K positions, 16 bytes hold the 32 four-bit codes (two per byte).  This is a pure
// permutation of the original 1-bit weights, same footprint, produced once at load time.
#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>

typedef uint16_t ggml_half;
#define QK1_0  128
#define QK8_128 128
typedef struct { ggml_half d; uint8_t qs[QK1_0/8]; } block_q1_0;
typedef struct { float d; int8_t qs[QK8_128]; } block_q8_128f; // fp32 scale for the bench

static inline float f2f(ggml_half h){ return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h))); }
static inline ggml_half t2h(float f){ return (ggml_half)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f),0),0); }
static inline float hsum8(__m256 v){__m128 lo=_mm256_castps256_ps128(v),hi=_mm256_extractf128_ps(v,1);lo=_mm_add_ps(lo,hi);lo=_mm_hadd_ps(lo,lo);lo=_mm_hadd_ps(lo,lo);return _mm_cvtss_f32(lo);}

static const __m256i BYTE_SHUF=_mm256_setr_epi8(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3);
static const __m256i BIT_MASKS=_mm256_setr_epi8(1,2,4,8,16,32,64,(char)-128,1,2,4,8,16,32,64,(char)-128,1,2,4,8,16,32,64,(char)-128,1,2,4,8,16,32,64,(char)-128);
static inline __m256i expand1(uint32_t bits){
    __m256i bc=_mm256_set1_epi32((int)bits);
    __m256i sh=_mm256_shuffle_epi8(bc,BYTE_SHUF);
    return _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(sh,BIT_MASKS),_mm256_setzero_si256()),_mm256_set1_epi8(1));
}
static inline __m256i dot4(__m256i acc,__m256i ones8,__m256i ones16,__m256i a,__m256i b){
#if defined(__AVXVNNI__) || defined(__AVX512VNNI__)
    (void)ones16;
#  if defined(__AVX512VNNI__)
    return _mm256_dpbusd_epi32(acc,a,b);
#  else
    return _mm256_dpbusd_avx_epi32(acc,a,b);
#  endif
#else
    return _mm256_add_epi32(acc,_mm256_madd_epi16(_mm256_maddubs_epi16(a,b),ones16));
#endif
}

// ---- baseline: current single-row GEMV (per-128 activation scale) ----
__declspec(noinline) void base_gemv(int n,float*s,const block_q1_0*x,const block_q8_128f*y){
    const int nb=n/QK1_0;
    const __m256i ones8=_mm256_set1_epi8(1),ones16=_mm256_set1_epi16(1);
    __m256 acc=_mm256_setzero_ps();
    for(int ib=0;ib<nb;ib++){
        __m256 d0v=_mm256_set1_ps(f2f(x[ib].d));
        __m256i t=_mm256_setzero_si256();
        for(int k=0;k<4;k++){
            const __m256i qy=_mm256_loadu_si256((const __m256i*)(y[ib].qs+k*32));
            const __m256i sg=expand1(((const uint32_t*)x[ib].qs)[k]);
            t=dot4(t,ones8,ones16,ones8,_mm256_sign_epi8(qy,sg));
        }
        acc=_mm256_fmadd_ps(_mm256_mul_ps(d0v,_mm256_set1_ps(y[ib].d)),_mm256_cvtepi32_ps(t),acc);
    }
    *s=hsum8(acc);
}

// ---- 2-value activation LUT ----
// Build one group: 16 int16 entries, stored as 16 low bytes followed by 16 high bytes so
// that the whole table is two vpshufb lookups.
static inline void lut_build_group(const int8_t*a,uint8_t*t){
    // true binary symmetry: T[c ^ 0xf] == -T[c], so only c = 0..7 needs summing
    for(int c=0;c<8;c++){
        int v=0;
        for(int i=0;i<4;i++) v += ((c>>i)&1)? a[i] : -a[i];
        const int nc=15-c;
        t[c]     =(uint8_t)(v&0xff);
        t[nc]    =(uint8_t)((-v)&0xff);
        t[16+c]  =(uint8_t)((v>>8)&0xff);
        t[16+nc] =(uint8_t)(((-v)>>8)&0xff);
    }
}
// Build the whole table for K activations (one batch element). Layout: per group 32 bytes.
static void lut_build(int K,const int8_t*a,uint8_t*tbl){
    for(int kg=0;kg<K/4;kg++) lut_build_group(a+4*kg,tbl+32*kg);
}

// Repack 1-bit weights (row-major, K/8 bytes/row) into LUT codes.
// out[rg][kg][16] : 32 rows x 4-bit codes, two codes per byte (even row = low nibble).
static void lut_pack(int M,int K,const uint8_t*w,uint8_t*out){
    const int kg_total=K/4;
    for(int r=0;r<M;r+=32){
        for(int kg=0;kg<kg_total;kg++){
            for(int j=0;j<16;j++){
                const int r0=r+2*j, r1=r0+1;
                uint8_t c0=0,c1=0;
                for(int i=0;i<4;i++){
                    const int bit=4*kg+i;
                    const uint8_t b0=(w[(size_t)r0*(K/8)+bit/8]>>(bit%8))&1;
                    const uint8_t b1=(w[(size_t)r1*(K/8)+bit/8]>>(bit%8))&1;
                    c0|=(uint8_t)(b0<<i);
                    c1|=(uint8_t)(b1<<i);
                }
                out[(size_t)(r/32)*kg_total*16 + (size_t)kg*16 + j] = (uint8_t)(c0|(c1<<4));
            }
        }
    }
}

// One K-block (BK multiple of 128) for 32 rows. acc[2] hold, per 128-bit lane, the integer
// accumulators for the even (lane 0) and odd (lane 1) output rows inside each row pair group.
__declspec(noinline) void lut_gemv32_block(int BK,const uint8_t*tbl,const uint8_t*codes,__m256i acc[2]){
    const int kg_total=BK/4;
    const __m128i m4=_mm_set1_epi8(0x0f);
    for(int kg=0;kg<kg_total;kg++){
        const __m128i tlo=_mm_loadu_si128((const __m128i*)(tbl+32*kg));
        const __m128i thi=_mm_loadu_si128((const __m128i*)(tbl+32*kg+16));
        const __m128i w=_mm_loadu_si128((const __m128i*)(codes+16*kg));
        const __m128i ce=_mm_and_si128(w,m4);
        const __m128i co=_mm_and_si128(_mm_srli_epi16(w,4),m4);
        const __m256i C=_mm256_set_m128i(co,ce);
        const __m256i L=_mm256_broadcastsi128_si256(tlo);
        const __m256i H=_mm256_broadcastsi128_si256(thi);
        const __m256i LE=_mm256_shuffle_epi8(L,C);
        const __m256i HE=_mm256_shuffle_epi8(H,C);
        acc[0]=_mm256_add_epi16(acc[0],_mm256_unpacklo_epi8(LE,HE));
        acc[1]=_mm256_add_epi16(acc[1],_mm256_unpackhi_epi8(LE,HE));
    }
}

// Full GEMV over K for 32 rows: float out[32]. d1 is the per-128 activation scale, d0 the
// per-row weight scale.
__declspec(noinline) void lut_gemv32(int K,float*out,const uint8_t*codes,const uint8_t*tbl,const float*d0,const float*d1){
    float fdot[32];
    for(int r=0;r<32;r++) fdot[r]=0.0f;
    const int blocks=K/QK1_0;
    for(int ib=0;ib<blocks;ib++){
        __m256i acc[2]={_mm256_setzero_si256(),_mm256_setzero_si256()};
        lut_gemv32_block(QK1_0, tbl+32*(QK1_0/4)*ib, codes+16*(QK1_0/4)*ib, acc);
        // acc[0] lane0 = even rows 0,2,..14 ; lane1 = odd rows 1,3,..15
        // acc[1] lane0 = even rows 16,..30   ; lane1 = odd rows 17,..31
        __m256i a0lo=_mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc[0]));
        __m256i a0hi=_mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc[0],1));
        __m256i a1lo=_mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc[1]));
        __m256i a1hi=_mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc[1],1));
        int32_t tmp[32];
        _mm256_storeu_si256((__m256i*)(tmp+0 ),a0lo);
        _mm256_storeu_si256((__m256i*)(tmp+8 ),a0hi);
        _mm256_storeu_si256((__m256i*)(tmp+16),a1lo);
        _mm256_storeu_si256((__m256i*)(tmp+24),a1hi);
        const float s=d1[ib];
        for(int j=0;j<8;j++){ fdot[2*j]     += (float)tmp[j]     * s; fdot[2*j+1]   += (float)tmp[8+j]  * s; }
        for(int j=0;j<8;j++){ fdot[16+2*j]  += (float)tmp[16+j]  * s; fdot[17+2*j]  += (float)tmp[24+j] * s; }
    }
    for(int r=0;r<32;r++) out[r]=fdot[r]*d0[r];
}

// L1-resident variant: K is processed in blocks of KB weights. The table slice for one
// K-block is KB/4*32 = KB*8 bytes, which fits the 32 KiB L1d, and it is swept over every
// output row before moving to the next K-block. The table is therefore read from memory
// once per token instead of once per 32-row group. This mirrors the K-blocked TL layout
// where the live LUT is kept hot while the weights stream through.
__declspec(noinline) void lut_gemv_blocked(int K,int M,float*out,const uint8_t*codes,const uint8_t*tbl,const float*d0,const float*d1,int KB){
    const int blocks=K/QK1_0;
    const int bpkb=KB/QK1_0;
    const int kg_total=K/4;
    memset(out,0,(size_t)M*sizeof(float));
    for(int ib0=0;ib0<blocks;ib0+=bpkb){
        for(int rg=0;rg+32<=M;rg+=32){
            float fdot[32];
            for(int r=0;r<32;r++) fdot[r]=0.0f;
            const uint8_t*crg=codes+(size_t)(rg/32)*kg_total*16;
            for(int ib=ib0;ib<ib0+bpkb;ib++){
                __m256i acc[2]={_mm256_setzero_si256(),_mm256_setzero_si256()};
                lut_gemv32_block(QK1_0,tbl+32*(QK1_0/4)*ib,crg+16*(QK1_0/4)*ib,acc);
                __m256i a0lo=_mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc[0]));
                __m256i a0hi=_mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc[0],1));
                __m256i a1lo=_mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc[1]));
                __m256i a1hi=_mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc[1],1));
                int32_t tmp[32];
                _mm256_storeu_si256((__m256i*)(tmp+0 ),a0lo);
                _mm256_storeu_si256((__m256i*)(tmp+8 ),a0hi);
                _mm256_storeu_si256((__m256i*)(tmp+16),a1lo);
                _mm256_storeu_si256((__m256i*)(tmp+24),a1hi);
                const float s=d1[ib];
                for(int j=0;j<8;j++){ fdot[2*j]     += (float)tmp[j]     * s; fdot[2*j+1]   += (float)tmp[8+j]  * s; }
                for(int j=0;j<8;j++){ fdot[16+2*j]  += (float)tmp[16+j]  * s; fdot[17+2*j]  += (float)tmp[24+j] * s; }
            }
            for(int r=0;r<32;r++) out[rg+r]+=fdot[r]*d0[rg+r];
        }
    }
}

static double now_s(){using C=std::chrono::high_resolution_clock;return std::chrono::duration<double>(C::now().time_since_epoch()).count();}

int main(int argc,char**argv){
    const int K=(argc>1)?atoi(argv[1]):4096;
    const int M=(argc>2)?atoi(argv[2]):65536;
    const int iters=(argc>3)?atoi(argv[3]):6;
    const int KB=(argc>4)?atoi(argv[4]):1024;
    const int nb=K/QK1_0;
    srand(1);
    uint8_t*w=(uint8_t*)_mm_malloc((size_t)M*(K/8),64);
    for(size_t i=0;i<(size_t)M*(K/8);i++) w[i]=(uint8_t)(rand()&0xff);
    block_q1_0*x=(block_q1_0*)_mm_malloc((size_t)M*nb*sizeof(block_q1_0),64);
    for(int r=0;r<M;r++) for(int ib=0;ib<nb;ib++){ x[(size_t)r*nb+ib].d=t2h(0.001f); memcpy(x[(size_t)r*nb+ib].qs,w+(size_t)r*(K/8)+ib*16,16); }
    // one activation vector (batch = 1) with per-128 scale
    block_q8_128f*y=(block_q8_128f*)_mm_malloc(nb*sizeof(block_q8_128f),64);
    for(int ib=0;ib<nb;ib++){ y[ib].d=0.01f; for(int j=0;j<128;j++) y[ib].qs[j]=(int8_t)(rand()%255-127); }

    // LUT table for the single activation vector
    uint8_t*tbl=(uint8_t*)_mm_malloc((size_t)(K/4)*32,64);
    // build from the full activation vector
    {
        int8_t*act=(int8_t*)_mm_malloc(K,64);
        for(int ib=0;ib<nb;ib++) memcpy(act+ib*128,y[ib].qs,128);
        lut_build(K,act,tbl);
        _mm_free(act);
    }
    uint8_t*codes=(uint8_t*)_mm_malloc((size_t)(M/32)*(K/4)*16,64);
    lut_pack(M,K,w,codes);
    float*d0=(float*)_mm_malloc((size_t)M*sizeof(float),64);
    for(int r=0;r<M;r++) d0[r]=f2f(x[(size_t)r*nb].d);

    const double wbytes=(double)M*(K/8);

    // correctness: compare LUT result for rows 0..31 against scalar
    {
        float out[32];
        float *d1s=(float*)_mm_malloc(nb*sizeof(float),64);
        for(int ib=0;ib<nb;ib++) d1s[ib]=y[ib].d;
        lut_gemv32(K,out,codes,tbl,d0,d1s);
        double maxerr=0;
        for(int r=0;r<32;r++){
            double ref=0;
            for(int ib=0;ib<nb;ib++){
                int dot=0;
                for(int j=0;j<128;j++){
                    const int bit=(x[(size_t)r*nb+ib].qs[j/8]>>(j%8))&1;
                    dot += bit? y[ib].qs[j] : -y[ib].qs[j];
                }
                ref += (double)dot*d1s[ib]*d0[r];
            }
            double e=fabs(ref-out[r]);
            if(e>maxerr) maxerr=e;
        }
        printf("lut correctness max abs err = %.6g\n",maxerr);
        _mm_free((void*)d1s);
    }

    // correctness: blocked variant, rows 0..31, versus the same scalar reference
    {
        float *out=(float*)_mm_malloc((size_t)M*sizeof(float),64);
        float *d1s=(float*)_mm_malloc(nb*sizeof(float),64);
        for(int ib=0;ib<nb;ib++) d1s[ib]=y[ib].d;
        lut_gemv_blocked(K,M,out,codes,tbl,d0,d1s,KB);
        double maxerr=0;
        for(int r=0;r<32;r++){
            double ref=0;
            for(int ib=0;ib<nb;ib++){
                int dot=0;
                for(int j=0;j<128;j++){
                    const int bit=(x[(size_t)r*nb+ib].qs[j/8]>>(j%8))&1;
                    dot += bit? y[ib].qs[j] : -y[ib].qs[j];
                }
                ref += (double)dot*d1s[ib]*d0[r];
            }
            double e=fabs(ref-out[r]);
            if(e>maxerr) maxerr=e;
        }
        printf("blocked correctness (KB=%d) max abs err = %.6g\n",KB,maxerr);
        _mm_free((void*)out);
        _mm_free((void*)d1s);
    }

    auto bench=[&](const char*name,auto fn){float sink=0;double t0=now_s();for(int it=0;it<iters;it++)fn(sink);double t1=now_s();double dt=t1-t0;printf("%-14s %8.2f GMAC/s  %6.2f GB/s (sink=%g)\n",name,(double)M*K*iters/dt/1e9,wbytes*iters/dt/1e9,sink);};

    bench("base-1row",[&](float&sink){for(int r=0;r<M;r++){float v;base_gemv(K,&v,x+(size_t)r*nb,y);sink+=v;}});
    bench("lut-32row",[&](float&sink){
        float *d1s=(float*)_mm_malloc(nb*sizeof(float),64);
        for(int ib=0;ib<nb;ib++) d1s[ib]=0.01f; // stable for timing
        float out[32];
        for(int r=0;r+32<=M;r+=32){ lut_gemv32(K,out,codes+(size_t)(r/32)*(K/4)*16,tbl,d0+r,d1s); for(int i=0;i<32;i++)sink+=out[i]; }
        _mm_free((void*)d1s);
    });
    bench("lut-blocked",[&](float&sink){
        float *d1s=(float*)_mm_malloc(nb*sizeof(float),64);
        for(int ib=0;ib<nb;ib++) d1s[ib]=0.01f;
        float *out=(float*)_mm_malloc((size_t)M*sizeof(float),64);
        lut_gemv_blocked(K,M,out,codes,tbl,d0,d1s,KB);
        for(int r=0;r<M;r++) sink+=out[r];
        _mm_free((void*)out);
        _mm_free((void*)d1s);
    });
    return 0;
}
