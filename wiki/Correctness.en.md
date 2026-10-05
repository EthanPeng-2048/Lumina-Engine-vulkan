# Correctness Verification

> 🌐 [中文](Correctness.md) | **English**

## 1. Numerical equivalence

### 1-bit bitwise ops: exact

Expanding and applying the weight signs is pure integer arithmetic:

- `q1t_expand` expands a 1-bit value into `+1 / -1` (0x01 / 0xFF).
- `_mm256_sign_epi8` negates bytes by sign.
- `vpmaddubsw + vpmaddwd` or `vpdpbusd` accumulate into int32.

None of these involve floating point, so results are **bit-identical** to the reference. A standalone
program compares against the original `xor+sub` kernel:

```
check: xor/sub ✓  vpsignb ✓   diff = 0
```

### Float summation order: may differ, without quality loss

The tiled kernel accumulates the four sub-blocks' int32 dot products first, then multiplies by `d0·d1`;
the generic kernel multiplies per sub-block. The float summation order differs, which in theory
produces ~1e-6 differences. In practice:

- **No perplexity loss** (see below).
- Greedy decoding may **diverge** on some long texts, but both continuations are coherent and of
  comparable quality. This is common to all optimized GEMM paths (including upstream repack/GEMM) and
  is normal numerical behavior.

---

## 2. Backend operator tests

```bat
build-release\bin\test-backend-ops.exe test -b CPU -o MUL_MAT -p "type_a=q1_0"
```

Result:

```
47/47 tests passed
1/1 backends passed
```

Covering `m=16..4096`, `n=1..512`, and various `bs/nr/per` combinations.

> This project's changes do not affect `iq1_s/iq1_m`, etc. The upstream CPU-vs-CPU tolerance warnings
> on those types are a **pre-existing** phenomenon (all pass with `GGML_NO_IQ_PANEL=1`) and unrelated
> to these changes.

---

## 3. Perplexity

```bat
Lumina-Engine-perplexity.exe -m Bonsai-8B.gguf -f docx/whitepaper.txt -c 512 -t 6 --chunks 4
```

| Config | PPL |
|------|----:|
| Generic (Q8_0 activations, pp+tg) | 5.7160 ± 0.507 |
| **Optimized (pp tiled + tg single-column GEMV, both Q8_128)** | **5.7173 ± 0.507** |
| **Binary LUT (Q1LUT, enabled by default)** | **5.7049** (difference from float summation order) |

Conclusion: changing activations from per-32 to per-128 (both pp and tg) **causes no accuracy loss**
(the difference is within noise). This is consistent with llama.cpp K-quant's use of per-256 activation
scales.

---

## 4. Greedy decoding consistency

Comparing the tiled path with the generic path at `--temp 0`:

```bat
Lumina-Engine-cli.exe -m Bonsai-8B.gguf -p "..." -n 128 --temp 0 -t 6 -st
set GGML_NO_Q1_TILE=1
Lumina-Engine-cli.exe -m Bonsai-8B.gguf -p "..." -n 128 --temp 0 -t 6 -st
```

For most prompts the output is **byte-identical**; on some long texts it diverges, but both
continuations are coherent (see the note in §1).

Short generations (≤64 tokens) and non-multiples-of-4 batches are consistent:

| batch | Result |
|------:|:----:|
| 3 / 5 / 7 / 13 / 17 | consistent with the generic path |

---

## 5. Sampling parameters

The tiled kernel only changes the last bits of the logits, not the sampling logic. Verified:

```bat
Lumina-Engine-cli.exe -m Bonsai-8B.gguf -p "Write a haiku about autumn." ^
  -n 32 --temp 0.7 --top-k 20 --top-p 0.9 --seed 5 -t 6 -st
```

Example output:

```
Crimson leaves descend,
whispers of cold winds through the trees—
harvest moon rises.
```

Different `--seed` values produce different but sensible output; `--temp/--top-k/--top-p` all work.

---

## 6. Edge cases

The tiled path's `ggml_cpu_q1t_supports_mul_mat()` enables only when all of the following hold:

- `src0->type == GGML_TYPE_Q1_0`
- `vec_dot_type == Q8_0`
- `ggml_cpu_has_avx2()`
- `src1->type == GGML_TYPE_F32`
- `ne00 % 128 == 0`
- `ne01 >= Q1T_ROWS (2)`
- `ne11 >= Q1T_MIN_BATCH (4)`
- `src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[3] == 1`
- `src0` contiguous
- `dst->type == F32 && dst->nb[0] == 4`

Otherwise it falls back to the generic kernel.

The binary lookup table (Q1LUT, enabled by default) has its own conditions:
`src0->type == GGML_TYPE_Q1_0`, `src1->type == GGML_TYPE_F32`, AVX2, and `ne11 == 1` (tg) or
`ne11 >= 8` (pp); `ne11 = 2..7` still uses the 2×4 tiled GEMM. Therefore:

- **tg (ne11=1)** uses Q1LUT by default (AVX2); with `GGML_Q1_LUT=0` or no AVX2 it falls back to the
  single-column GEMV / generic kernel.
- **pp (ne11 >= 8)** uses Q1LUT by default; `ne11 = 2..7` uses the 2×4 tiled GEMM.
- **Non-AVX2 CPUs** use the generic kernel.
- **Other models / other quantization types** are completely unaffected.
- **Row tails** (`ne01 % 2`) and **column tails** (`ne11 % 4`) fall back to the generic `vec_dot`,
  ensuring full coverage.

---

## 7. How to verify yourself

```bat
REM 1. Backend tests (requires a tree configured with -DLLAMA_BUILD_TESTS=ON)
build-release\bin\test-backend-ops.exe test -b CPU -o MUL_MAT -p "type_a=q1_0"

REM 2. Perplexity comparison
Lumina-Engine-perplexity.exe -m Bonsai-8B.gguf -f docx\whitepaper.txt -c 512 -t 6 --chunks 4
set GGML_NO_Q1_TILE=1
Lumina-Engine-perplexity.exe -m Bonsai-8B.gguf -f docx\whitepaper.txt -c 512 -t 6 --chunks 4
set GGML_NO_Q1_TILE=

REM 3. Greedy output comparison
Lumina-Engine-cli.exe -m Bonsai-8B.gguf -p "The capital of France is" -n 16 --temp 0 -t 6 -st
```
