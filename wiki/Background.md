# 背景

> 🌐 **中文** | [English](Background.en.md)

## 1. Bonsai 1-bit 模型

Bonsai 是 PrismML 发布的端到端 1-bit 语言模型家族，CPU/GPU/移动端均可部署。以本仓库使用的
`Bonsai-8B.gguf` 为例（读取自 GGUF 元数据）：

| 项目 | 值 |
|------|----|
| 架构 | Qwen3-8B dense |
| 层数 | 36 |
| 隐藏维度 | 4096 |
| FFN 维度 | 12288 |
| 注意力 | GQA：32 query heads / 8 KV heads，head_dim = 128 |
| RoPE | YaRN，factor 4.0，original_ctx 16384，freq_base 1e6 |
| RMS epsilon | 1e-6 |
| 上下文 | 65536 |
| 词表 | 151936（LM head 张量为 `[4096, 151669]`） |
| 权重格式 | GGUF **Q1_0** |
| 权重张量 | 254 个 Q1_0 + 145 个 F32（仅 norm） |

关键点：**embedding、Q/K/V/O 投影、FFN gate/up/down、LM head 全部是 Q1_0**，只有各层的
RMSNorm 是 F32。也就是说，推理中几乎所有矩阵乘都走 Q1_0 内核。

## 2. Q1_0 量化格式

```c
#define QK1_0 128
typedef struct {
    ggml_half d;           // 2 字节尺度
    uint8_t   qs[QK1_0/8]; // 16 字节符号位
} block_q1_0;              // 共 18 字节 / 128 权重
```

- 每个权重 1 bit：`1 → +d`，`0 → −d`（`d` 为正的 fp16 尺度）。
- 每 128 个权重共享一个 `d`。
- 有效位宽 **1.125 bit/权重**（1 sign bit + 16-bit scale / 128）。
- 8.19B 参数 ≈ **1.15 GB**，相比 FP16（16.38 GB）缩小 **14.2×**。

点积定义（`ggml_vec_dot_q1_0_q8_0`）：

```
result = Σ_blocks d0 · Σ_{k=0..3} d1_k · Σ_j  sign_j · q_j
```

其中 `q` 是 int8 量化后的激活，`sign_j = ±1`，`d0` 是权重块尺度，`d1_k` 是激活子块尺度。

## 3. GGUF 与运行时

GGUF 是 llama.cpp 的模型容器：包含张量元数据、类型、以及 tokenizer/chat template。加载后：

- 权重保持在量化格式（Q1_0）中，不做 FP16/FP32 物化。
- 激活（`src1`）在每个 `mul_mat` 前被量化为 `vec_dot_type`（Q1_0 默认是 Q8_0，每 32 个值一个尺度）。
- 然后调用 `vec_dot` 内核：`ggml_vec_dot_q1_0_q8_0`。

## 4. 为什么 1-bit 在 CPU 上需要特别优化

### 4.1 Token 生成（tg）—— 内存带宽瓶颈

自回归生成每产生一个 token 都需要读取**全部权重**：

```
每 token 权重流量 ≈ 1.06 GB
```

在单通道 DDR4-2666 上实测流式读带宽约 **14–17 GB/s**，因此 tg 理论上限约
`16.7 / 1.06 ≈ 15.7 tok/s`。启用二值查表（Q1LUT，默认）后本机达到约 13 tok/s，
更接近该上限；通用内核约 9.5–10 tok/s。

> ⚠️ **测试平台为单通道内存**：下面的带宽与 tg 上限均由单通道决定；
> 双通道/更高带宽平台上 tg 会更高，提升比例更接近内核层面。

**结论**：tg 主要受内存带宽约束，内核优化的空间有限，但把内核做到接近带宽上限仍有收益。

### 4.2 Prompt 处理（pp）—— 计算瓶颈

一次前向可处理 N 个 token，权重只读一遍，但要对 N 列激活做点积：

```
MACs = Σ 权重数 × N
```

因此 pp 是**计算受限**。默认的 `ggml_compute_forward_mul_mat` 采用 16×16 分块，
对每个 (行, 列) 调用一次 `vec_dot`，而 `vec_dot` 会对每一列都**重新展开 1-bit 符号位**。
对于 1-bit 权重，这个“位展开”成本占比很高，是主要的优化对象。

### 4.3 1-bit 内核的指令成本

每 32 个权重（一个 Q8_0 子块）的 AVX2 点积大致为：

```
展开符号: broadcast + shuffle + and + cmpeq + or   (5)
应用符号: vpsignb                                   (1)
点积:     vpmaddubsw + vpmaddwd                     (2)
浮点:     cvtdq2ps + fmadd(+scale)                  (2~3)
```

共约 **10–11 条指令 / 32 权重**。优化方向：
- 降低**每权重指令数**（展开复用、VNNI 合并）。
- 降低**每 128 权重的浮点转换次数**（per-128 激活尺度）。
- 对二值权重改用**激活侧查表**（Q1LUT）：用 `vpshufb` 一次处理 4 个权重。

详见 [Optimizations](Optimizations.md)。

## 5. 参考

- 1-bit Bonsai 白皮书：`docx/1-bit-bonsai-8b-whitepaper.pdf`
- 模型 README：`docx/README.md`
- 上游 llama.cpp：<https://github.com/ggml-org/llama.cpp>
