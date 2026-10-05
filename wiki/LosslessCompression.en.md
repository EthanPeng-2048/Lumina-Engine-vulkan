# Lossless model compression (Build 8)

> 🌐 [中文](LosslessCompression.md) | **English**
>
> Just want "how to pack and how to use"? See the user guide: [ModelCompression](ModelCompression.en.md).

## Goal

Shrink a GGUF model file **without quantization and without pruning**: every weight value,
tensor count and tensor type stays exactly the same.  Only the *storage encoding* changes,
and the file is reconstructed byte-for-byte at load time.

## Why generic compression is not enough

`SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf` is already Q4_K / Q6_K.  Its per-byte entropy is
about 7.8–7.9 bit/byte, so a plain `zstd -19` over the whole file only reaches ~**96.9%**.
The reason is that a k-quant block interleaves two very different kinds of data:

| field | share of block | entropy |
|-------|---------------:|--------:|
| `d` / `dmin` (fp16 scales) | 4 / 144 | ~5.8 bit/byte |
| `scales` (6-bit sub-scales / mins) | 12 / 144 | ~7.1 bit/byte |
| 4-bit weights (`qs`) | 128 / 144 | ~7.7 bit/byte |

Feeding the high-entropy weight stream and the low-entropy scale streams to a compressor
interleaved wastes almost all of the modelling budget on the scales.

## Approach: field split + context sort + DEFLATE (container v2)

`tools/lumina_gguf_pack.py` applies a reversible **field split** to every tensor; container v2
adds a few more reversible transforms on top:

- **Q4_K**: split each block into byte-planes for the fp16 `d` / `dmin`, the 8 six-bit `scale`
  and 8 six-bit `min` fields, and the 4-bit quants.  The quants are grouped by sub-block and then
  **stably sorted by the context key** `(scale << 6 | min)`, which places sub-blocks with similar
  distributions next to each other and lets DEFLATE model each sub-distribution efficiently.
- **Q6_K**: byte-plane for `d`, even/odd split for `scales`, and the usual low/high nibble + `qh`
  split.
- **F32**: whole block.

All streams are concatenated across the tensor and then DEFLATEd.  At load time each tensor is
inflated, un-sorted and its original block bytes rebuilt.  The container layout (v2) is documented
at the top of `tools/lumina_gguf_pack.py`; the C++ decoder still reads v1.

### Result

| file | size | ratio |
|------|-----:|------:|
| original `...Q4_K_M.gguf` | 2,630,213,408 B | 100% |
| packed (Build 8 delivery, container v2) | 2,479,779,960 B | **94.28%** |

**143.5 MiB saved** (25.6 MiB more than container v1's 117.9 MiB).  The reconstructed file
matches the original SHA-256 exactly:

```
fb584d0b0f79f9e9944f91ac061e24726a8d014b6702213bfd5585729db7ab61
```

### How much headroom is left

99.7% of the model is Q4_K / Q6_K 4/6-bit quants.  For this model the zero-order entropy of the
Q4_K quants is 3.87 bit / 4 bit (a ~96.7% ceiling) and the conditional entropy given
`(scale, min)` is 3.82 bit (~95.6%); v2's context sort already reaches 95.5% of that stream, i.e.
**essentially the lossless limit**.  The Q6_K quants are almost uniformly distributed (close to
full 6-bit entropy), so a generic compressor cannot help there.  In other words, without
quantization or pruning, 94.3% is close to the theoretical floor; an order-of-magnitude gain is
only possible with lossy methods.

## Engine side

- `src/lumina-gguf.cpp` / `.h`: detects the `LGGC` container magic and streams the file to
  `%TEMP%\lumina-<hash>.gguf`, after which the normal GGUF loader takes over.  The decoder reads
  both container v1 and v2.
- **Extension convention**: the packer writes `A.gguf` to `A.gguf.lumina` by default (the original
  file is kept).  The engine keys off the `LGGC` magic and **not the extension**, so both `.gguf`
  and `.gguf.lumina` load directly; the server's model-directory listing and the CLI completion
  also recognise `.gguf.lumina` as a model.
- `src/llama.cpp`: `llama_model_load_from_file()` calls `lumina_gguf_resolve_path()`; plain
  GGUF files are returned unchanged, so nothing else is affected.
- `src/miniz/`: embedded single-file DEFLATE (MIT / public domain), no external dependency.
- No quantization and no pruning: every tensor is restored byte-for-byte.  The temporary file
  is cached/reused and can be cleaned up by the OS.

## Usage

```bat
REM pack (offline, once): by default A.gguf -> A.gguf.lumina, original kept
python tools\lumina_gguf_pack.py model.gguf

REM byte-exact check
python tools\lumina_gguf_unpack.py verify model.gguf.lumina model.gguf

REM load directly (the engine detects and restores it; .gguf or .gguf.lumina both work)
Lumina-Engine-cli.exe -m model.gguf.lumina -p "..." -n 16
```
