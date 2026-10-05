# 正确性验证

> 🌐 **中文** | [English](Correctness.en.md)

## 1. 数值等价性

### 1-bit 位运算：逐位精确

权重符号的展开与应用是纯整数运算：

- `q1t_expand` 把 1-bit 展开为 `+1 / -1`（0x01 / 0xFF）。
- `_mm256_sign_epi8` 按符号逐字节取反。
- `vpmaddubsw + vpmaddwd` 或 `vpdpbusd` 累加到 int32。

这些操作不涉及浮点，结果与参考实现**逐位一致**。用独立小程序对照原 `xor+sub` 内核：

```
check: xor/sub ✓  vpsignb ✓   diff = 0
```

### 浮点求和顺序：可能不同，但无质量损失

分块内核把 4 个子块的 int32 点积先累加，再统一乘以 `d0·d1`；通用内核是每子块各乘一次。
浮点求和顺序不同，理论上会产生 1e-6 量级的差异。实际上：

- **困惑度无损失**（见下）。
- 贪婪解码在个别长文本上可能出现**分叉**，但两条续写都语义连贯、质量相当。这是所有
  优化 GEMM 路径（包括上游的 repack/GEMM）共有的现象，属正常数值行为。

---

## 2. 后端算子测试

```bat
build\bin\test-backend-ops.exe test -b CPU -o MUL_MAT -p "type_a=q1_0"
```

结果：

```
47/47 tests passed
1/1 backends passed
```

覆盖 `m=16..4096`、`n=1..512`、多种 `bs/nr/per` 组合。

> 注：本项目修改不影响 `iq1_s/iq1_m` 等其它类型。上游 IQP 路径在这些类型上的
> CPU-vs-CPU 容差告警是**既有现象**（`GGML_NO_IQ_PANEL=1` 后全部通过），与本次改动无关。

---

## 3. 困惑度

```bat
Lumina-Engine-perplexity.exe -m Bonsai-8B.gguf -f docx/whitepaper.txt -c 512 -t 6 --chunks 4
```

| 配置 | PPL |
|------|----:|
| 通用路径（Q8_0 激活，pp+tg） | 5.7160 ± 0.507 |
| **优化后（pp 分块 + tg 单列 GEMV，均 Q8_128 激活）** | **5.7173 ± 0.507** |
| **二值查表（Q1LUT，默认启用）** | **5.7049**（差异来自浮点求和顺序） |

结论：激活从 per-32 改为 per-128（pp 与 tg 均如此）**没有造成精度损失**（差异在噪声范围内）。
这与 llama.cpp K-quant 使用 per-256 激活尺度的工程实践一致。

---

## 4. 贪婪解码一致性

`--temp 0` 下对比分块路径与通用路径：

```bat
Lumina-Engine-cli.exe -m Bonsai-8B.gguf -p "..." -n 128 --temp 0 -t 6 -st
set GGML_NO_Q1_TILE=1
Lumina-Engine-cli.exe -m Bonsai-8B.gguf -p "..." -n 128 --temp 0 -t 6 -st
```

在多数 prompt 上**逐字一致**；在个别长文本上分叉，但两条续写均连贯（见第 1 节说明）。

短生成（≤64 token）与非 4 倍数 batch 均一致：

| batch | 结果 |
|------:|:----:|
| 3 / 5 / 7 / 13 / 17 | 与通用路径一致 |

---

## 5. 采样参数

分块内核只改变 logits 的末位，不改变采样逻辑。已验证：

```bat
Lumina-Engine-cli.exe -m Bonsai-8B.gguf -p "Write a haiku about autumn." ^
  -n 32 --temp 0.7 --top-k 20 --top-p 0.9 --seed 5 -t 6 -st
```

输出（示例）：

```
Crimson leaves descend,
whispers of cold winds through the trees—
harvest moon rises.
```

不同 `--seed` 产生不同但合理的输出；`--temp/--top-k/--top-p` 均正常工作。

---

## 6. 边界情况

分块路径的 `ggml_cpu_q1t_supports_mul_mat()` 只在以下条件全部满足时启用：

- `src0->type == GGML_TYPE_Q1_0`
- `vec_dot_type == Q8_0`
- `ggml_cpu_has_avx2()`
- `src1->type == GGML_TYPE_F32`
- `ne00 % 128 == 0`
- `ne01 >= Q1T_ROWS (2)`
- `ne11 >= Q1T_MIN_BATCH (4)`
- `src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[3] == 1`
- `src0` 连续
- `dst->type == F32 && dst->nb[0] == 4`

不满足时自动回退到通用内核。

二值查表（Q1LUT，默认启用）另有独立的启用条件：`src0->type == GGML_TYPE_Q1_0`、
`src1->type == GGML_TYPE_F32`、AVX2，且 `ne11 == 1`（tg）或 `ne11 >= 8`（pp）；
`ne11 = 2..7` 仍走 2×4 分块 GEMM。因此：

- **tg（ne11=1）** 默认走 Q1LUT 查表（AVX2）；`GGML_Q1_LUT=0` 或非 AVX2 时回退到单列 GEMV / 通用内核。
- **pp（ne11 >= 8）** 默认走 Q1LUT；`ne11 = 2..7` 走 2×4 分块 GEMM。
- **非 AVX2 CPU** 走通用内核。
- **其它模型 / 其它量化类型** 完全不受影响。
- **行尾部**（`ne01 % 2`）与**列尾部**（`ne11 % 4`）分别用通用 `vec_dot` 处理，保证覆盖完整。

---

## 7. 如何自行验证

```bat
REM 1. 后端测试（需使用开启了 -DLLAMA_BUILD_TESTS=ON 的构建树）
build-release\bin\test-backend-ops.exe test -b CPU -o MUL_MAT -p "type_a=q1_0"

REM 2. 困惑度对照
Lumina-Engine-perplexity.exe -m Bonsai-8B.gguf -f docx\whitepaper.txt -c 512 -t 6 --chunks 4
set GGML_NO_Q1_TILE=1
Lumina-Engine-perplexity.exe -m Bonsai-8B.gguf -f docx\whitepaper.txt -c 512 -t 6 --chunks 4
set GGML_NO_Q1_TILE=

REM 3. 贪婪输出对照
Lumina-Engine-cli.exe -m Bonsai-8B.gguf -p "The capital of France is" -n 16 --temp 0 -t 6 -st
```
