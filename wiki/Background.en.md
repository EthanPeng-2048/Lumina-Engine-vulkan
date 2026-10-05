# Background

> 🌐 [中文](Background.md) | **English**

## 1. The Bonsai 1-bit model

Bonsai is a family of end-to-end 1-bit language models from PrismML, deployable on CPU/GPU/mobile.
Using the `Bonsai-8B.gguf` shipped with this repo as an example (read from GGUF metadata):

| Item | Value |
|------|----|
| Architecture | Qwen3-8B dense |
| Layers | 36 |
| Hidden size | 4096 |
| FFN size | 12288 |
| Attention | GQA: 32 query heads / 8 KV heads, head_dim = 128 |
| RoPE | YaRN, factor 4.0, original_ctx 16384, freq_base 1e6 |
| RMS epsilon | 1e-6 |
| Context | 65536 |
| Vocab | 151936 (LM head tensor is `[4096, 151669]`) |
| Weight format | GGUF **Q1_0** |
| Weight tensors | 254 × Q1_0 + 145 × F32 (norms only) |

Key point: **embedding, Q/K/V/O projections, FFN gate/up/down and the LM head are all Q1_0**;
only the per-layer RMSNorm is F32. Almost every matmul in inference therefore goes through a
Q1_0 kernel.

## 2. The Q1_0 quantization format

```c
#define QK1_0 128
typedef struct {
    ggml_half d;           // 2-byte scale
    uint8_t   qs[QK1_0/8]; // 16 sign bytes
} block_q1_0;              // 18 bytes / 128 weights
```

- One bit per weight: `1 → +d`, `0 → −d` (`d` is a positive fp16 scale).
- One shared `d` per 128 weights.
- Effective width **1.125 bit/weight** (1 sign bit + 16-bit scale / 128).
- 8.19B params ≈ **1.15 GB**, **14.2×** smaller than FP16 (16.38 GB).

Dot product (`ggml_vec_dot_q1_0_q8_0`):

```
result = Σ_blocks d0 · Σ_{k=0..3} d1_k · Σ_j  sign_j · q_j
```

where `q` is the int8-quantized activation, `sign_j = ±1`, `d0` is the weight-block scale and
`d1_k` the activation sub-block scale.

## 3. GGUF and the runtime

GGUF is llama.cpp's model container: tensor metadata, types, tokenizer and chat template. After
loading:

- Weights stay in their quantized format (Q1_0); no FP16/FP32 materialization.
- Activations (`src1`) are quantized to `vec_dot_type` before each `mul_mat` (for Q1_0 the default
  is Q8_0: one scale per 32 values).
- The `vec_dot` kernel `ggml_vec_dot_q1_0_q8_0` is then called.

## 4. Why 1-bit needs special care on CPU

### 4.1 Token generation (tg) — memory-bandwidth bound

Autoregressive generation must read **all weights** for every token:

```
weight traffic per token ≈ 1.06 GB
```

On single-channel DDR4-2666 the measured streaming read bandwidth is ~**14–17 GB/s**, so the
theoretical tg ceiling is `16.7 / 1.06 ≈ 15.7 tok/s`. With the binary lookup table (Q1LUT, default)
the machine reaches ~13 tok/s, close to that ceiling; the generic kernel reaches ~9.5–10 tok/s.

> ⚠️ **The test platform is single-channel memory**: the bandwidth and tg ceiling below are set by
> single-channel; dual-channel / higher-bandwidth platforms will show higher tg and a gain ratio
> closer to the kernel-level value.

**Bottom line**: tg is primarily memory-bandwidth bound. Kernel work is only worth it up to the
bandwidth limit.

### 4.2 Prompt processing (pp) — compute bound

One forward pass can process N tokens; weights are read once, but dot products run over N columns:

```
MACs = Σ weights × N
```

So pp is **compute bound**. The default `ggml_compute_forward_mul_mat` uses 16×16 tiling and calls
`vec_dot` once per (row, column), and `vec_dot` **re-expands the 1-bit sign bits for every column**.
For 1-bit weights that bit-expansion is a large fraction of the cost, and the main optimization target.

### 4.3 Instruction cost of the 1-bit kernel

The AVX2 dot product per 32 weights (one Q8_0 sub-block) is roughly:

```
expand signs: broadcast + shuffle + and + cmpeq + or   (5)
apply signs:  vpsignb                                   (1)
dot product:  vpmaddubsw + vpmaddwd                     (2)
float:        cvtdq2ps + fmadd(+scale)                  (2~3)
```

≈ **10–11 instructions / 32 weights**. Directions:
- fewer instructions per weight (reuse the expansion, fuse with VNNI);
- fewer float conversions per 128 weights (per-128 activation scale);
- for binary weights, switch to an **activation-side lookup table** (Q1LUT): a single `vpshufb`
  covers 4 weights.

See [Optimizations](Optimizations.en.md).

## 5. References

- 1-bit Bonsai whitepaper: `docx/1-bit-bonsai-8b-whitepaper.pdf`
- Model README: `docx/README.md`
- Upstream llama.cpp: <https://github.com/ggml-org/llama.cpp>
