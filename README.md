# Lumina-Engine — Bonsai 1-bit LLM CPU 推理优化（llama.cpp 分支）

本项目名为 **Lumina-Engine**，编译产物统一以 `Lumina-Engine` 命名（如 `Lumina-Engine-cli.exe`、`Lumina-Engine.dll`）。

本仓库基于 `llama.cpp`（b10991）为 **PrismML Bonsai 系列 1-bit 量化模型（GGUF Q1_0）** 增加并优化了一条高性能的 **纯 CPU 推理路径**。在完整保留 llama.cpp 原有通用推理能力（所有既有量化类型、所有后端、所有模型）的前提下，只在 `GGML_TYPE_Q1_0` 上启用专门的优化内核。

> 目标平台：搭载 SSE4.2 / AVX / AVX2 / AVX-512 / VNNI / AVX-VNNI / BF16 / AMX 的 **Windows 10 / 11 x86-64** 设备。

---

## 亮点

- **Prompt 处理（read / pp）提升 29%–43%**（相对通用路径，6 线程实测）。
- **Token 生成（tg）提升约 13%**（相对旧版，12 线程实测）：新增单列 GEMV，同样使用 per-128 激活，把每个 128 权重块的浮点转换从 4 次降到 1 次。
- 新增 **per-128 激活量化（Q8_128）**，同时用于寄存器分块 GEMM（pp）与单列 GEMV（tg）。
- 利用 **VNNI（`vpdpbusd`）** 把 `vpmaddubsw + vpmaddwd` 两条指令合并为一条。
- 通用内核使用 **`vpsignb`** 实现“+1 保留 / -1 取反”，缩短依赖链。
- **14 个 CPU 微架构变体**，覆盖几乎所有 Windows x86-64 CPU。
- **数值无损**：Q1_0 位运算完全精确；激活量化改为 per-128 后困惑度不升反略降（5.7111 vs 5.7160）。
- **2 值查表（Q1LUT，默认启用）**：按 BitNet TL 的思路为二值权重实现激活侧查表 GEMV。利用真正的二值对称性 `T[c^0xf]=-T[c]`（只算一半表项，尺度用 fp16 存储）。本机（AVX2、无 VNNI）受控 A/B：**tg 约 +30%**（9.8→13.0 t/s，已达单通道带宽上限）；**pp 从 `ne11>=8` 起全面快于 2×4 分块 GEMM**，batch 越大优势越大（pp512 +26%、pp2048 +57%）。因此默认 **tg 走查表、pp 按实际 token 数选择**（`ne11==1` 与 `ne11>=8` 查表，`2..7` 走分块 GEMM）。代价：首次约 0.7 s 生成查表码、额外约 0.95 GiB 内存；`GGML_Q1_LUT=0` 可关闭。详见 [Optimizations](wiki/Optimizations.md#优化-7二值查表q1lut默认启用)。
- 改动**严格隔离**：只在 Q1_0 + AVX2 路径生效，其它模型/量化类型零影响。

---

## 快速开始

### 直接使用预编译产物

`Lumina-Engine-Bin/` 目录已经包含编译好的完整运行时（14 个 CPU 变体 + 全部工具 + 运行库）：

```bat
cd D:\Lumina-Engine\Lumina-Engine-Bin

REM 交互式推理
Lumina-Engine-cli.exe -m ..\Bonsai-8B.gguf -p "Explain quantum computing simply." -n 256 -t 6

REM 基准测试
Lumina-Engine-bench.exe -m ..\Bonsai-8B.gguf -p 512 -n 128 -t 6

REM 困惑度
Lumina-Engine-perplexity.exe -m ..\Bonsai-8B.gguf -f ..\docx\whitepaper.txt -c 512 -t 6
```

推荐采样参数（与模型发布方一致）：

```
temperature = 0.5   (0.5 ~ 0.7)
top-k       = 20    (20 ~ 40)
top-p       = 0.9   (0.85 ~ 0.95)
repetition penalty = 1.0
```

> **线程数默认 = 物理核心数（不启用超线程）**：默认路径由 `common_cpu_get_num_math()` 决定，
> 在 Windows / Linux 上均返回物理核。只有显式传 `-t <逻辑核数>`（或 `-t 0`）才会启用超线程。
> 这符合本仓库的 A/B 结论：1-bit GEMV 在超线程下共享 L1 与内存带宽，通常更慢。

### 从源码编译

见 [wiki/Building.md](wiki/Building.md)。简述：

```bat
cmake -S llama.cpp-b10991 -B build-release -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ^
  -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON ^
  -DGGML_NATIVE=OFF -DGGML_RPC=ON ^
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build build-release -j
```

> **交付产物默认组装到 `D:\Lumina-Engine\Lumina-Engine-Bin`**。直接运行 `D:\Lumina-Engine\build-bin.bat` 会依次编译
> MSVC（`build-release`）与 Clang（`build-clang`）两棵树，再调用 `bench\assemble_bin.bat`
> 组装（clang 产物为基底 + 9 个 MSVC 变体覆盖），无需手动拷贝。

---

## 性能（本机实测）

**测试机**：Intel Core i5-10400（Comet Lake，6 核 12 线程，2.90 GHz）
**内存**：8 GB 单通道 DDR4-2666（实测流式读带宽 ~14–17 GB/s）
**系统**：Windows 10 19045
**模型**：Bonsai-8B.gguf（Qwen3-8B 架构，全权重 Q1_0，1.07 GiB）
**参数**：`-t 6 -r 3`，tokens/s，均值 ± 标准差

> ⚠️ **测试机为单通道内存**（8 GB DDR4-2666，实测流式读带宽 ~14–17 GB/s）：
> tg（token 生成）受 ~15.7 tok/s 的带宽上限压制，双通道/更高带宽平台上 tg 提升更接近内核层面的 ~30%；
> pp 为计算受限，受影响较小。请以同场 A/B 的**比值**而非绝对值为准。

| 测试 | 通用路径 | **优化后** | 提升 |
|------|---------:|-----------:|-----:|
| pp32  | 13.18 | **18.78** | **+42.5%** |
| pp64  | 13.14 | **18.46** | **+40.5%** |
| pp128 | 13.11 | **17.94** | **+36.8%** |
| pp256 | 12.98 | **17.31** | **+33.4%** |
| pp512 | 12.83 | **16.49** | **+28.5%** |

> 通用路径 = `GGML_NO_Q1_TILE=1`（仍含 `vpsignb` 与 VNNI 优化）。

### Token 生成（tg）

旧版 tg 走通用 `vec_dot`（Q8_0 激活，每 32 权重一次浮点转换）。新版增加**单列 GEMV**：
激活改为 per-128（Q8_128），每个 128 权重块只做一次浮点转换；量化也向量化并跨线程分块。

同场交替 A/B（`-p 64 -n 32,64,128 -t 12 -r 3`，两轮均值）：

| 测试 | 旧版 | **新版** | 提升 |
|------|-----:|---------:|-----:|
| tg32  | 11.27 | **12.67** | **+12.4%** |
| tg64  | 11.22 | **12.83** | **+14.3%** |
| tg128 | 11.17 | **12.69** | **+13.6%** |

> 6 线程约 +4–10%（本机单通道带宽是主要上限）；内存带宽越高的平台，收益越接近内核层面的 ~25–30%。
> 行分组大小可用 `GGML_Q1_GEMV_ROWS`（1/2/4，默认 1）微调。

**困惑度（whitepaper.txt，4 chunks，c=512）**

| 配置 | PPL |
|------|----:|
| 通用路径（Q8_0 激活，pp+tg） | 5.7160 |
| **优化后（pp 分块 + tg 单列 GEMV，均 Q8_128）** | **5.7173** |

**正确性**

| 检查 | 结果 |
|------|------|
| `test-backend-ops` Q1_0 MUL_MAT | 47 / 47 通过 |
| 贪婪解码输出 | 与通用路径基本一致（浮点求和顺序不同，1e-6 量级） |
| `--temp/--top-k/--top-p/--seed` 采样 | 正常，输出合理 |

完整测试数据与方法见 [wiki/Benchmarks.md](wiki/Benchmarks.md)。

---

## 优化做了什么（概览）

1. **激活 per-128 量化（Q8_128）** — 一个 Q1_0 块（128 权重）内只保留 1 个激活尺度，使 4 个子块的 int32 点积可以先累加，浮点转换从每 32 权重一次降到每 128 权重一次。
2. **单列 GEMV（tg）** — token 生成复用 per-128 激活，每个 128 权重块只做一次浮点转换；激活量化向量化并按块跨线程并行。
3. **寄存器分块 GEMM（2 行 × 4 列，pp）** — 符号位展开一次，供 4 列复用，且不产生额外内存流量。
4. **VNNI** — 支持 VNNI 的 CPU 用 `vpdpbusd` 直接累加 int32。
5. **`vpsignb`** — 通用内核用单条指令完成“+1 保留 / −1 取反”。
6. **多微架构构建** — 同一份源码编译出 14 个 CPU 后端，运行时自动选择最佳。

详细原理见 [wiki/Optimizations.md](wiki/Optimizations.md)。

---

## 兼容性

`Lumina-Engine-Bin/` 中带有 14 个 CPU 后端 DLL，运行时会根据 CPU 特性自动选择：

```
x64  sse42  sandybridge  ivybridge  piledriver  haswell  skylakex
cannonlake  cascadelake  icelake  cooperlake  alderlake  zen4  sapphirerapids
```

- 支持 AVX2 的 CPU 使用寄存器分块 GEMM。
- 支持 VNNI 的 CPU 额外使用 `vpdpbusd`。
- 更老的 CPU 回退到原有的通用 Q1_0 内核（仍然正确）。
- 所有 14 个变体均由 MSVC（9 个常用）与 Clang 20（5 个 clang 专属）编译，并已验证可混用。

细节见 [wiki/Compatibility.md](wiki/Compatibility.md)。

---

## 目录结构

```
D:\Lumina-Engine\
├─ Bonsai-8B.gguf            # 模型（示例）
├─ docx\                     # Bonsai 白皮书与 README
├─ Lumina-Engine-Bin\        # ★ 预编译交付产物（14 变体 + 工具 + 运行库；默认编译输出目录）
├─ llama.cpp-b10991\         # ★ 修改后的源码
│   ├─ ggml\src\ggml-cpu\
│   │   ├─ q1t.cpp / q1t.h   # 新增：Q1_0 分块 GEMM + 单列 GEMV + 二值查表(Q1LUT)
├─ bench\                    # 内核微基准（含 bench_lut2.cpp 查表 A/B）
│   │   ├─ ggml-cpu.c        # 接入 hook + work-size
│   │   └─ arch\x86\quants.c # vpsignb + VNNI
│   ├─ build-release\        # MSVC 构建目录
│   └─ build-clang\          # Clang 20 构建目录
├─ README.md                 # 本文件
└─ wiki\                     # 详细文档
```

---

## 模型无损压缩（Build 8 优化）

在**不量化、不剪枝**的前提下，`SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf` 可被无损压缩为
一个更小的容器，权重逐字节不变：

- 原理：Q4_K / Q6_K 的每个 256 权重块内混有低熵的 `d/dmin/scales` 和近似均匀的 4/6-bit
  权重流。打包器把这些字段**分开排列**（field split），并对 Q4_K 的量化值按
  `(scale, min)` 上下文做**稳定排序**，再 DEFLATE，避免高熵权重流淹没有效统计；
  F32 张量（norm / router）单独 DEFLATE。
- 效果（本模型）：`2,630,213,408 B → 2,479,779,960 B`，**94.28%（节省 143.5 MiB）**；
  解压后 SHA-256 与原文件一致（容器 v1 为 95.30% / 117.9 MiB）。
- 用法：打包器默认把 `A.gguf` 输出为 `A.gguf.lumina`（原文件保留），直接用文件路径即可；
  引擎按魔数 `LGGC` 自动识别并还原，`.gguf` 与 `.gguf.lumina` 均可加载：

```bat
python tools\lumina_gguf_pack.py   model.gguf                 REM -> model.gguf.lumina
python tools\lumina_gguf_unpack.py verify model.gguf.lumina  model.gguf
Lumina-Engine-cli.exe -m model.gguf.lumina -p "The capital of France is" -n 16
```

首次加载会解压到系统临时目录（`%TEMP%\lumina-<hash>.gguf`）并复用；后续加载直接复用。

> 完整的压缩 / 加载步骤、FAQ 见用户指南 **[wiki/ModelCompression.md](wiki/ModelCompression.md)**（[EN](wiki/ModelCompression.en.md)）。

---

## Vulkan 后端与自动设备策略

交付包含 `ggml-vulkan.dll`（本机 Intel Iris Xe 已实测）。**一套预编译版本，程序自动决定 CPU/GPU**，
用户无需选择（显式 `-ngl` 始终优先）：

| 机器/模型 | 自动策略 |
|---|---|
| 无 GPU | 纯 CPU |
| 集成显卡（IGPU）+ dense 模型 | 层留 CPU + Vulkan 算子 offload（pp ≈2.4×） |
| 集成显卡 + **MoE 模型** | 纯 CPU（MoE 走算子 offload 反而慢 ~30%） |
| 独立显卡（dGPU） | 按显存 offload 整层 |

- 想强制某一行为：`-ngl 99`（整层 offload）、`-ngl 0`（层留 CPU）、`--no-op-offload`（纯 CPU，逐位确定）。
- 构建时若存在 `D:\VulkanSDK\sdk` 则自动启用 Vulkan；未安装 SDK 时自动回退 CPU-only。
- **训练默认强制纯 CPU**（反向算子仅 CPU 实现）；`LUMINA_TRAIN_ALLOW_GPU=1` 为实验开关，不推荐。
- 完整 A/B 数据见 **[OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md)**。

---

## 训练引擎（QLoRA 微调量化模型）

新增纯 C++ 训练工具 **`Lumina-Engine-train`**，用于微调 **Q4_K / Q8_0 / Q1_0 等量化模型**
以及 `.gguf.lumina` 压缩容器。量化权重不可直接训练，因此采用 **QLoRA**：基座权重保持冻结
（仍以量化形式存放），只训练低秩适配器 A/B。

```bat
REM 量化基座 QLoRA 微调，输出标准 LoRA 适配器（单一文本 -f/-p）
Lumina-Engine-train.exe -m model.Q4_K_M.gguf -f train.txt ^
  -c 512 -b 512 -ub 128 -t 6 -lr 1e-4 --epochs 3 ^
  -lora-rank 8 -lora-alpha 16 -o model.lora.gguf

REM 使用数据集（.jsonl：text / messages / conversations 三种格式自动识别）
Lumina-Engine-train.exe -m model.Q4_K_M.gguf ^
  --dataset data/stage2_sft/train.jsonl --dataset-val data/stage2_sft/val.jsonl ^
  -c 512 -b 512 -ub 128 --cpu-cores 8 -lr 5e-5 --epochs 2 ^
  -lora-rank 8 -lora-alpha 16 ^
  --save-steps 500 --save-every-epochs 1 --keep-last 3 ^
  -o model.lora.gguf

REM 用现有推理工具加载适配器
Lumina-Engine-cli.exe -m model.Q4_K_M.gguf --lora model.lora.gguf -p "..." -n 128 -t 6
```

- **数据集支持**（`--dataset` / `--dataset-val`）：每行一个 JSON 对象，自动识别三种常见格式——
  `{"text": "..."}`（继续预训练）、`{"messages": [{"role","content"}, ...]}`（OpenAI 风格）、
  `{"conversations": [{"from","value"}, ...]}`（ShareGPT）。对话记录使用**模型自带 chat template**
  渲染；默认 `--mask-history` 只对 assistant 回复计算 loss（`--no-mask-history` 可关闭）。
- **训练集 / 验证集分离**：`--dataset-val` 指定独立验证集；不指定时用 `--val-split` 从训练集切分。
- **CPU 核心数**：`--cpu-cores N`（默认 `0` = 使用全部可用核心；本机可显式限制，如 `--cpu-cores 8`）。
- **检查点**：`--save-steps N` 每 N 个优化步保存一次（**默认 500**）、`--save-every-epochs N` 每个 epoch
  保存、`--keep-last N` 仅保留最近 N 个检查点、`--resume` 从最新检查点续训。检查点写入
  `<out>.ckpt-<step>.gguf` + 同名 `.json` 元数据（`--checkpoint-dir` 可自定义目录）。
- 输出为**标准 LoRA GGUF**（`adapter.type=lora`、`<tensor>.lora_a/.lora_b`），与上游工具互通。
- 支持**稠密模型与 MoE 混合专家模型**：3D 专家权重 `[n_in, n_out, n_expert]` 会生成 3D 适配器
  （为此补齐了 ggml 的 `MUL_MAT_ID` 反向、`REGLU`/`SIGMOID`/`TANH`/`CLAMP` 反向、broadcast `DIV`
  反向与批量 `GET_ROWS_BACK`）。router 与专家均可训练。
- 训练时自动关闭 repack（反向传播的 `OUT_PROD` 需要原生量化 buffer）。
- 详细用法、参数、实现与验证见 **[wiki/Training.md](wiki/Training.md)**（[EN](wiki/Training.en.md)）。

---

## 主要更改

| 文件 | 类型 | 说明 |
|------|------|------|
| `ggml/src/ggml-cpu/q1t.cpp` | 新增 | Q1_0 per-128 激活：pp 的寄存器分块 GEMM + tg 的单列 GEMV + 二值激活查表（Q1LUT，默认启用；pp 按 batch 选择） |
| `ggml/src/ggml-cpu/q1t.h` | 新增 | 对外接口 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | 修改 | mul_mat hook、work buffer 大小 |
| `ggml/src/ggml-cpu/CMakeLists.txt` | 修改 | 编译新源文件 |
| `ggml/src/ggml-cpu/arch/x86/quants.c` | 修改 | 通用 Q1_0 内核：`vpsignb` + VNNI |
| `src/lumina-gguf.cpp` / `.h` | 新增 | 无损压缩 GGUF 容器解码器，加载时透明还原为原模型（不量化、不剪枝） |
| `src/miniz/miniz.{c,h}` | 新增 | 内置 DEFLATE 实现（MIT/公有领域），供解压使用 |
| `src/llama.cpp` | 修改 | `llama_model_load_from_file()` 接入压缩容器 |
| `src/CMakeLists.txt` | 修改 | 编译新源文件 |
| `tools/lumina_gguf_pack.py` | 新增 | 打包器：按 k-quant 字段拆分 + 上下文排序后 DEFLATE（容器 v2） |
| `tools/lumina_gguf_unpack.py` | 新增 | 参考解包器 / 逐字节校验 |
| `common/build-info.cpp.in` | 修改 | 版本号升至 **Build 8** |
| `common/preset.cpp` | 修改 | 模型目录列表识别 `.gguf.lumina` |
| `common/arg.cpp` | 修改 | `-m` 补全同时匹配 `.gguf` 与 `.gguf.lumina` |
| `include/llama.h` | 修改 | 新增 `llama_adapter_lora_init_trainable()` / `llama_adapter_lora_save()` / `llama_adapter_lora_copy_weights()` |
| `src/llama-adapter.cpp` | 修改 | 可训练 LoRA 适配器的创建（A/B 分配与初始化）、GGUF 保存与权重拷贝（续训） |
| `src/llama-context.cpp` / `.h` | 修改 | `opt_init` 额外注册 LoRA 参数；训练期图节点预算翻倍；repack 禁用下的诊断；负标签实现 loss 掩码 |
| `common/common.h` / `common/arg.cpp` | 修改 | 新增 `-lora-rank` / `-lora-alpha` / `-lora-targets`；`--dataset` / `--dataset-val` / `--mask-history` / `--cpu-cores` / `--save-steps` / `--keep-last` / `--resume` 等训练参数 |
| `tools/lumina-train/` | 新增 | **`Lumina-Engine-train`：量化模型 / `.gguf.lumina` 的 QLoRA 微调工具（支持数据集与检查点）** |
| `ggml/src/ggml.c` | 修改 | 新增 MoE `MUL_MAT_ID` 反向、`REGLU`/`SIGMOID`/`TANH`/`CLAMP` 反向、broadcast `DIV` 反向、批量 `GET_ROWS_BACK` |
| `ggml/src/ggml-cpu/ggml-cpu.c` / `ops.cpp` | 修改 | 新反向算子的 CPU 内核（含量化专家权重的梯度） |
| `ggml/include/ggml.h` / `ggml-rpc.h` | 修改 | 新增 `GGML_OP_MUL_MAT_ID_BACK_AS/_B` 算子与 RPC 版本 |
| `common/build-info.cpp.in` / `CMakeLists.txt` | 修改 | 版本号升至 **Build 16** |
| `configure-release.bat` | 修改 | 检测 `D:\VulkanSDK\sdk` 并启用 `GGML_VULKAN`（无 SDK 自动回退） |
| `common/common.h` / `common/arg.cpp` | 修改 | `-ngl` 默认 `auto` → `0`（层留 CPU + Vulkan 算子 offload）；移除旧断言 |
| `common/common.cpp` | 修改 | **机器+模型感知的自动设备策略**：IGPU→层留 CPU、dGPU→offload、MoE→关闭算子 offload |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | 修改 | 修复 `COUNT_EQUAL` 的 I32/I64 类型不匹配（GPU 训练 acc 损坏） |
| `common/build-info.*` / `tools/cli` / `tools/server` | 修改 | 控制台横幅显示 Build 号 + 项目链接（GitHub / 爱发电 / Bilibili） |
| `tools/lumina-train/` | 修改 | 训练强制纯 CPU（`no_op_offload`），修复 Vulkan 污染训练指标 |

完整改动说明见 [wiki/Changes.md](wiki/Changes.md)。

---

## 环境变量

| 变量 | 作用 |
|------|------|
| `GGML_NO_Q1_TILE=1` | 关闭分块/GEMV 路径，回退到通用 Q1_0 内核（用于 A/B 或排障） |
| `GGML_NO_Q1_GEMV=1` | 只关闭单列 GEMV（tg）路径 |
| `GGML_Q1_GEMV_ROWS=1\|2\|4` | 单列 GEMV 的行分组大小（默认 1） |
| `GGML_Q1_LUT=0` | **关闭**整个二值查表路径（默认启用）。关闭后 tg/pp 回到打包内核，可省约 0.95 GiB 内存 |
| `GGML_Q1_LUT_PP=0` | 只关闭批处理（pp）查表，pp 回退到 2×4 分块 GEMM（默认启用，`ne11>=8` 时生效） |
| `GGML_Q1_LUT_KB=N` | 查表路径的 K 分块大小（0=整表，默认且最快；>0 为实验性 L1 分块，实测更慢） |
| `GGML_Q1_LUT_DEBUG=1` | 打印查表缓存分配/填充信息（排障；`=0` 关闭） |
| `LUMINA_Q1_VK_GEMV=0\|1` | Vulkan Q1_0 DP4A GEMV：`0` 强制关、`1` 强制开；**默认 auto（独显开 / 集显 UMA 关）** |
| `LUMINA_Q1_VK_CACHE_MB=N` | Vulkan Q1_0 权重重排缓存上限（MB，默认 2048，`0`=不限） |
| `LUMINA_Q1_VK_INT8=1` | 实验性 int8 协同矩阵 prefill 路径（默认关；在本机 Turing 上慢于原生） |

---

## 许可与致谢

- 基于 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)（MIT License）。
- Q1_0 格式与 Bonsai 模型来自 [PrismML](https://prismml.com)。
- 本仓库的 CPU 优化仅针对 Q1_0，遵循上游许可。

---

## 文档索引（wiki）

- [Home](wiki/Home.md) — 文档总览
- [Background](wiki/Background.md) — Bonsai / Q1_0 / GGUF 背景
- [Optimizations](wiki/Optimizations.md) — 优化原理详解
- [Building](wiki/Building.md) — 编译指南（含多变体）
- [Delivery](wiki/Delivery.md) — **交付产物目录 `Lumina-Engine-Bin` 与生成配置**
- [Benchmarks](wiki/Benchmarks.md) — 完整测试结果与方法
- [Correctness](wiki/Correctness.md) — 正确性验证
- [Compatibility](wiki/Compatibility.md) — CPU 变体与分发
- [Changes](wiki/Changes.md) — 主要代码更改
- [ModelCompression](wiki/ModelCompression.md) — **用户指南：压缩模型 & 使用压缩模型**
- [LosslessCompression](wiki/LosslessCompression.md) — 无损压缩容器原理与格式
- [Training](wiki/Training.md) — **训练引擎：对量化模型 / `.gguf.lumina` 做 QLoRA 微调**
- [Troubleshooting](wiki/Troubleshooting.md) — 常见问题

> 每个 wiki 页面均提供英文版（`wiki/*.en.md`），例如 [Delivery (EN)](wiki/Delivery.en.md)。
