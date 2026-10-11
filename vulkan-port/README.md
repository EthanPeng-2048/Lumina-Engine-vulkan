# Lumina-Engine — Vulkan 1-bit (Q1_0) GEMV 移植

本目录把 **CPU 路径上针对 Bonsai Q1_0（二值 1-bit 权重）的推理优化逻辑**，移植为
**Vulkan 计算着色器**，并附带一个独立、可复现的正确性 + 性能测试工具 `lumina_q1`。

> 目标 GPU（测试机）：NVIDIA CMP 40HX（Turing TU106，`int dot: 1` 即支持 DP4A）。

---

## 1. 背景：GPU 上目前没有 1-bit 的整数路径

在 `ggml-vulkan` 里，`GGML_TYPE_Q1_0` 走的是**反量化成浮点再乘浮点**：

- **GEMV（token 生成）**：`mul_mat_vec.comp` → `dequantize4()` 把权重位展开成 **±1.0 fp32**，与 fp32 激活做 FMA。
- **GEMM（prompt 处理）**：`mul_mm*.comp` → `dequantFuncQ1_0()` 展开成 **fp16**，再做 fp16 矩阵乘。

而且 `q1_0` **不在** `is_legacy_quant` 列表里（只有 q2_0/q4_0/q4_1/q5_0/q5_1/q8_0），所以：

- 它拿不到 `mul_mat_vecq.comp` 的 **q8_1 整数 GEMV（DP4A）**；
- 它拿不到 `mul_mmq.comp` 的 **整数 MMQ**。

也就是说，**CPU 路径赖以提速的“整数点积”在 GPU 上完全缺席**——这正是本次移植的切入点。

---

## 2. 移植的优化逻辑（CPU → Vulkan）

| CPU 优化（`ggml-cpu/q1t.cpp`） | Vulkan 移植实现 |
|---|---|
| **per-128 激活量化（Q8_128）**：一个 Q1_0 块（128 权重）只保留 1 个激活尺度 | 激活量化成 int8，每 128 值一个尺度（`quantize_q8_128`） |
| **VNNI `vpdpbusd`**（一条指令累加 4 个 int8 乘） | **DP4A**：`GL_EXT_integer_dot_product` 的 `dotPacked4x8EXT`（→ `OpSDot`） |
| **`vpsignb`**（符号位 = “保留/取反”） | 二值恒等式 `dot(a,w) = 2·dot(a,bits) − sum(a)`，其中 4 个权重位用一次乘法展开成 4 个 0/1 字节 `(nibble * 0x00204081) & 0x01010101` |
| **单列 GEMV（tg）**：4 个子块的 int32 点积先累加，每 128 权重只做 1 次浮点转换 | 同样的 int32 累加 + 每 128 权重一次 `dm*bs*float(t32)` |
| **权重重排（Q1LUT 的按行组重排）** | 把权重一次性重排为 **块主序 + 32 行合并** 布局，使 warp 读取合并（coalesced） |

---

## 3. 目录与文件

```
vulkan-port/
├─ shaders/
│  ├─ q1_gemv_float.comp      # 基线：反量化成 fp32（= 今天 ggml-vulkan 的做法）
│  ├─ q1_gemv_int8.comp       # 移植：per-128 int8 + DP4A（原始布局）
│  ├─ q1_gemv_rp_float.comp   # 基线 + 重排（合并）布局
│  ├─ q1_gemv_rp_int8.comp    # 移植：per-128 int8 + DP4A + 重排布局  ★主结果
│  ├─ q1_gemv_sk_float.comp   # split-K + 共享内存归约（基线）
│  ├─ q1_gemv_sk_int8.comp    # split-K + 共享内存归约（移植）
│  ├─ q1_gemm_int8_cm.comp    # prefill：int8 协同矩阵 Q1_0 GEMM（移植，第 8 节）
│  ├─ cm_bench_f16.comp       # 张量核心微基准：f16 16x16x16
│  └─ cm_bench_int8.comp      # 张量核心微基准：int8 16x16x32
├─ src/main.cpp               # 独立 Vulkan 测试台（正确性 + 性能）
├─ CMakeLists.txt             # glslc 编译着色器 + 构建 lumina_q1
└─ win/
   ├─ env.bat                 # VS/Vulkan SDK 路径
   ├─ build-port.bat          # 配置 + 编译
   └─ run-port.bat            # 运行
```

---

## 4. 构建与运行（Windows / MSVC / Ninja）

```bat
cd vulkan-port\win
build-port.bat
run-port.bat --device 1 --rows 4096 --cols 14336 --iters 200
```

`--device` 按枚举顺序选择 GPU（本机 `0`=AMD R5 240，`1`=NVIDIA CMP 40HX）。

---

## 5. 结果

### 正确性（M=4096, K=14336，对比 fp64 参考，scale≈26.5）

| 内核 | 相对误差 |
|---|---|
| float | 4.0e-6 |
| **int8（移植）** | **6.4e-3** |
| rp-float | 2.2e-6 |
| **rp-int8（移植）** | **6.4e-3** |

int8 的 6.4e-3 误差来自 **per-128 int8 激活量化**，与 CPU 路径一致——CPU 侧正是用
per-128 量化取得了“困惑度不升反降”的结果。位运算本身精确无损。

### 性能（M=4096, K=14336 = 与 `test-backend-ops` 的 MUL_MAT perf 同形状）

> 测量注意：该 GPU 与其它任务共享时数字波动很大（同一配置见过 rp-int8 从 ~380 到 ~630 GMAC/s）。
> 下表为**串行、无争用**下 `--iters 500` 的结果。

| 内核 | ms/iter | GMAC/s |
|---|---:|---:|
| float（一行一线程，≈ 今天的路径） | 0.327 | 179 |
| int8（一行一线程） | 0.242 | 243 |
| rp-float（重排布局） | 0.096 | 613 |
| **rp-int8（移植）** | **0.093** | **629** |
| sk-float | 1.855 | 32 |
| sk-int8 | 1.851 | 32 |

**真实基线**：`test-backend-ops perf -o MUL_MAT -b Vulkan1 -p q1_0`（同样串行运行）测得
ggml-vulkan 原生 Q1_0 GEMV 为 **103 µs/run = 568 GMAC/s**。

三条可靠的结论：

1. **布局是决定性因素**：同一算术下，逐行一线程（179）→ 重排合并布局（613）差 **3.4×**。
   ggml-vulkan 原生内核本身已经是合并访问，处于同一水平（568），所以这一点在原生里已经做到了。
2. **整数算术本身只带来约 3%**：`rp-float`(613) → `rp-int8`(629) = **1.03×**。
   也就是说在这块 GPU 上，GEMV 并不是 ALU 受限，而是访存/延迟受限，DP4A 的理论 4× 吞吐没有兑现。
3. 移植内核与原生内核**基本持平、略有优势**（629 vs 568，跨工具比较仅供量级参考）。

### 结论

1. **可行**：CPU 的 1-bit 优化逻辑（per-128 激活 + 整数点积 + 权重重排）可以完整移植到 Vulkan，数值正确、误差与 CPU 一致。
2. **GEMV 收益有限**：整数化只带来约 **3%**。原因是 GPU 的 GEMV 不受 ALU 限制——
   真正的 3.4× 提升来自**访存合并**，而这在 ggml-vulkan 原生实现里已经具备。
   这与 CPU 不同：CPU 上 GEMV 是算力/指令受限，所以 `q1t_gemv`/Q1LUT 能拿到 30%。
3. **真正的大头在 prefill（GEMM）**：ggml-vulkan 的 Q1_0 GEMM 同样没有整数路径
   （fp16 反量化协同矩阵，实测 27 TFLOPS）。把二值权重接进 **int8 协同矩阵 / int8 tensor core**
   才有可能显著超过 fp16（Turing 上 INT8 tensor core 吞吐是 FP16 的 2×）——这是推荐的下一步。

---

## 6. 建议的 ggml-vulkan 集成方式（未在本目录实施）

1. 在 `vulkan-shaders/` 增加 `q1_0` 专用 GEMV 着色器（本目录的 `q1_gemv_rp_int8.comp`），
   并把 `q1_0` 加入可走整数路径的类型集合；
2. 在 host 侧为 Q1_0 张量做一次 **惰性重排**（等价于 CPU 的 Q1LUT 重排），缓存于进程内；
3. 用已有的 `GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT` + `device->integer_dot_product` 做能力门控，
   不支持时回退现有浮点路径；
4. 对 prefill，增加 int8 协同矩阵的 Q1_0 变体。

---

## 7. 已知限制

- `lumina_q1` 是**独立测试台**，不是 ggml-vulkan 的集成；它复刻了真实的数据布局与算术，
  用于在真实 GPU 上验证移植逻辑与量化误差。
- GEMV 已完整实现并验证；prefill 的整数 GEMM 给出了可行性验证与算力天花板（见第 8 节），
  但**尚未集成**进 ggml-vulkan。
- 本机 `test-backend-ops` 因 fork 里 `GET_ROWS_BACK` 的一个既有断言（view 用例）在 test 模式崩溃，
  所以基线改用 `perf` 模式测得；这与本次移植无关。

---

## 8.5 打通 ggml-vulkan 正确性校验（`test-backend-ops`）

为了让后续的 ggml-vulkan 集成**可验收**，修复了测试二进制在启动即崩溃的问题：

该 fork 的 `ggml_get_rows_back()` 加入了批量 MoE 断言
`a->ne[1] == b->ne[0]` / `a->ne[2] == b->ne[1]`，
而上游测试会用 **非连续（view）rows** 和 **量化类型的 grad** 去测这个算子，两者都会直接
触发 `GGML_ASSERT` 使整个测试进程 abort。这是一个**与本移植无关的 fork/测试不兼容**。

处理：在 `tests/test-backend-ops.cpp` 中停用 `GET_ROWS_BACK` 的用例（仅改测试，注释说明原因）。

修复后可直接验收 Q1_0：

```bat
test-backend-ops.exe test -b Vulkan1 -o MUL_MAT -p q1_0
```

```
... 43/43 tests passed
Backend Vulkan1: OK
```

`perf` 模式基线同样可用：

```bat
test-backend-ops.exe perf -b Vulkan1 -o MUL_MAT -p q1_0
```

---

## 9. ggml-vulkan 集成（env 门控旁路，默认关闭）

按“旁路 + 开关”的方式把 Q1_0 整数路径接进了 ggml-vulkan 本体：
**默认关闭**，设置 `LUMINA_Q1_VK_INT8=1` 才启用，不满足条件自动回退原路径。

### 改动文件

| 文件 | 说明 |
|---|---|
| `ggml/src/ggml-vulkan/vulkan-shaders/lumina_q1_quantize.comp` | 新增：F32 激活 → int8（每 128 值一个尺度，Q8_128） |
| `ggml/src/ggml-vulkan/vulkan-shaders/lumina_q1_gemm.comp` | 新增：int8 协同矩阵 Q1_0 GEMM（per-128 缩放） |
| `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` | 注册上面两个着色器（coopmat 变体，禁用 spirv-opt） |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | 设备初始化时创建两条管线；新增 `ggml_vk_mul_mat_q1_0_int8()` 旁路与调度钩子 |
| `tests/test-backend-ops.cpp` | 修复崩溃、新增一个可触发旁路的用例 |

### 启用条件（全部满足才走新路径，否则回退）

`src0` 是 Q1_0、`src1`/`dst` 是 F32 且连续、2D、`K%128==0`、`M%64==0`、`N%64==0`、
`M>=128 && N>=128`（即只接管 prefill），且设置了 `LUMINA_Q1_VK_INT8`。

### A/B 实测（`test-backend-ops perf -o MUL_MAT -p q1_0`，M=4096 N=512 K=14336）

| 配置 | us/run | TFLOPS |
|---|---:|---:|
| 关闭（原生 fp16 反量化 coopmat 路径） | 2191 | **27.44** |
| 开启（int8，BM=64/BN=64/NW=128） | 45955 | 1.31 |
| 开启（int8，BM=32/BN=64/NW=64） | 70738 | 0.85 |
| 开启（int8，**BM=64/BN=32/NW=128 ← 目前最优**） | 25504 | **2.36** |

调优过程中得到的规律：**线程数/占用率比"解码摊薄"更重要**（NW 128 的两种形状都明显优于 NW 64），
且 KHR cooperative matrix 本身在微基准里是 134 TFLOPS 的高性能路径（见第 8.2 节），
所以瓶颈不在 MMA，而在**每个 K 块的共享内存暂存 + 两次 barrier + 低流水**。

### 9.1 进一步定位：真正的瓶颈是 KHR cooperative matrix 的"小块"限制

又做了两个决定性实验：

1. **把权重重解码次数降为 0**（先用 `lumina_q1_decode.comp` 把 Q1_0 一次性解码成 int8 权重矩阵，
   GEMM 里只做向量化搬移）：不但没变快，反而从 2.36 → **1.61 TFLOPS**。
   → 说明**权重解码不是瓶颈**（额外多出的全局搬运反而更贵）。
2. 微基准里**纯 MMA 是 134 TFLOPS**（`coopMatLoad` 在循环外），而本内核每块要做
   4 次 `coopMatLoad` + 2 次 `coopMatStore`（每次只有 16×16×32）。

结论：本设备上 KHR cooperative matrix 被限制在 **16×16×32**，一个 64×128 的 tile 需要
几十次小 load/store，**开销全在 cooperative matrix 的加载/存储次数上**。
而 ggml 的快速路径用的是 **NV cooperative matrix 2**——`coopMatLoadTensorNV` 一次就加载
BM×BK（如 64×128）的大块并在加载时解码，所以没有这个开销。

也就是说：**在这块 GPU 上要跑出竞争性的 int8 prefill，必须用 NV coopmat2**
（即修改 `mul_mm_cm2.comp` 的框架），KHR coopmat 路线到此为止。

正确性（`test -o MUL_MAT -b Vulkan1`）：开关关 43/43 通过；开关开时新增用例也通过，
说明 per-128 int8 的精度在测试容差内（与 CPU 的 Q8_128 行为一致）。

### 结论（重要）

- **旁路管道已经打通并验证**：env 门控、能力守卫、回退、精度都对，默认行为零影响。
- **但这个自研 int8 内核目前比 ggml 的原生 fp16 路径慢 11.6–21×**。
  它既不是 MMA 受限、也不是带宽受限，而是**占用率/流水线受限**：
  共享内存 35 KB（BN=64）导致每 SM 只能驻留 1 个工作组；缩到 BN=32 后提速 1.8×，
  但权重 A 的**重复解码**（每个 N-tile 都要重新解 8192 个 int8）又成为主要开销。
- 因此第 8 节测得的 **int8 2.11× 天花板仍然是天花板**：要兑现它，需要一个
  真正调优的 GEMM（大 BN 摊薄 A 解码 + 多缓冲/异步拷贝 + 合适的流水线），
  等价于重写 ggml 的 coopmat 内核族——不是本次增量能完成的。

若继续，建议方向：把 int8 解码做成 **`coopMatLoadTensorNV` 的 decode 函数**（像 ggml 对 fp16 那样
在加载时解码、彻底免掉手工 A 暂存），并采用 ggml 已有的 multi-phase/大 tile 配置。

---

## 8. Prefill（GEMM）整数路径：可行性 / 天花板 / 结论

### 8.1 设备能力（`--probe`）

```
VK_KHR_cooperative_matrix : yes
  M=16 N=16 K=32  A/B=3/3 C=5   <-- int8xint8   (SINT8/SINT8/SINT32)
  M=16 N=8  K=32  A/B=3/3 C=5   <-- int8xint8
  M=16 N=16 K=16  A/B=0/0 C=1                  (FLOAT16/FLOAT16/FLOAT32)
```

### 8.2 张量核心算力天花板（`--mma-bench`）

| MMA 类型 | GMAC/s | TFLOPS |
|---|---:|---:|
| f16 × f16 → f32（`16x16x16`） | 31870 | 63.7 |
| **int8 × int8 → s32（`16x16x32`）** | **67270** | **134.5** |

**int8 张量核心吞吐 = fp16 的 2.11×**（Turing 上 INT8 TC 是 FP16 的 2 倍，符合预期）。
这是 Q1_0 prefill 走整数路径的**理论上限**。

对照：ggml-vulkan 现存 Q1_0 GEMM 实测 **27.4 TFLOPS**（fp16 反量化 + 协同矩阵），
即只用到 fp16 峰值的 43%。若把权重换成 int8 且内核保持计算受限，理论上限约 **2×**。

### 8.3 移植的 int8 coopmat Q1_0 GEMM（`--gemm`）

新增 `shaders/q1_gemm_int8_cm.comp`：BM=16 × BN=64，BK=128（一个 Q1_0 块），
权重解码为 int8 ±1、激活 int8（每 128 一尺度）、int8 MMA 累加、每 128 块应用一次
`d0[row]·d1[col]`（与 CPU 的 per-128 浮点转换一致）。

```
verify  M=64 N=64 K=512      : correctness max abs err = 4.8e-07 (rel 7.5e-08)   ← 正确
perf    M=4096 N=512 K=14336 : 74 ~ 81 ms  ->  0.7 ~ 0.8 TFLOPS                 ← 很慢
```

**功能正确**（整数累加，误差仅来自浮点求和顺序），但性能远低于 ggml（27 TFLOPS）。
差距全部来自 **tiling / 共享内存暂存（staging）**，而不是 MMA：
该 tile 下每个 K 块要暂存约 1 万个 int8、却只做 16 次 MMA；且每工作组仅 32 线程。
换句话说——**只要内核不是 MMA 受限，选 int8 还是 fp16 就无关紧要**。

### 8.4 结论与建议

- **可行**：int8 协同矩阵在这块 GPU 上可用，且原始算力是 fp16 的 2.11×；
  移植的整数 Q1_0 GEMM 数值正确。
- **但自研 GEMM 无法兑现**：prefill 性能由分块/暂存决定，手写内核差 ggml 40 倍。
  收益必须通过**把 int8 Q1_0 路径接入 ggml-vulkan 已有的、已调优的 coopmat 管线**
  来获得，而不是另写一个 GEMM。
- **推荐落地方式**：在 `ggml-vulkan` 的 coopmat 着色器族里，为 `q1_0` 增加一个
  “解码为 int8 + int8 MMA + per-128 scale” 的变体（仿照现有 `dequantFuncQ1_0` 的位置替换为
  返回 int8 的 `decodeFunc`），复用现有的 tile 配置与流水线生成；能力门控沿用
  `integer_dot_product` / cooperative-matrix 探测。这是唯一有可能接近 2× 的路径。

---

## 10. 第二次尝试：NV cooperative matrix 2 的 int8 Q1_0 GEMM（已实现，仍慢于 fp16）

第 9.1 节的结论是“必须用 NV coopmat2 的大块 tensor load 才有戏”，所以按此实现了
`llama.cpp-b10991/ggml/src/ggml-vulkan/vulkan-shaders/lumina_q1_gemm_cm2.comp`：

- 权重（`Q1_0`）用 `coopMatLoadTensorNV` **在加载时解码**成 int8 `{-1,+1}`（decode hook），
  不再手工暂存 A（这是第 9 节慢的主因之一）；
- 激活仍是 `lumina_q1_quantize.comp` 产出的 **per-128 int8（Q8_128）**；
- `s8 × s8 → s32` 的 MMA（`gl_ScopeWorkgroup` 张量，BM=64 × BN=64 × BK=128，128 线程）；
- **per-128 的 `d0[row]·d1[col]` 只在每个 Q1_0 块做一次**：把该块的 int32 MMA 局部和
  `coopMatStore` 到共享内存，做一次逐元素 `d0*d1*float(partial)` 后累加到寄存器里的
  fp32 累加器——即 CPU q1t 的“每 128 权重只做一次浮点转换”。

默认由 `LUMINA_Q1_VK_INT8=1` 打开；`LUMINA_Q1_VK_CM1=1` 可切回第 9 节的 KHR 版本做对照。

### 10.1 结果

| 版本 | us/run | TFLOPS |
|---|---:|---:|
| 原生 fp16 反量化 + coopmat（基线） | 2191 | **27.1** |
| 第 9 节 KHR int8（BM64/BN32/128 线程） | 25504 | 2.36 |
| **本节 cm2 int8（BM64/BN64/BK128/128 线程）** | 7225 | **8.32** |

正确性：`test-backend-ops test -b Vulkan1 -o MUL_MAT -p q1_0` **44/44 通过**（含专门触发该
旁路的 `m=256,n=128,k=512` 用例），说明 int8 路径数值与 CPU q1t 参考一致。

也就是说：**cm2 版本比第 9 节的 KHR 版本快 3.5×，但仍是原生 fp16 路径的 3.3 倍慢。**

### 10.2 逐项归因（每次只改一处，实测）

| 实验 | 结果 |
|---|---:|
| 去掉 per-128 缩放（纯 int8 GEMM，结果不对） | 9.60 TFLOPS |
| 把 A 的解码换成“无 decode 的普通 int8 load” | 4.27 TFLOPS（访问模式也变了，不可比） |
| 把 K 循环展开 4×（多缓冲，想掩盖 load 延迟） | 6.41 TFLOPS（更慢，寄存器压力） |
| 把 A load 提到 K 循环外（只保留 B load） | 9.62 TFLOPS |
| **把 A、B 两个 load 都提到循环外（只剩 112 次 MMA）** | **49.23 TFLOPS** |
| A 换成 fp16 元素（同几何、同结构） | 5.01 TFLOPS |
| 加 vector decode（`GL_NV_cooperative_matrix_decode_vector`，i8vec4） | 8.32 TFLOPS（无变化） |

三条硬结论：

1. **MMA 不是瓶颈**：把 operand load 移出 K 循环后，同一内核跑到 49.2 TFLOPS
   （int8 峰值的 37%），与 ggml fp16 路径达到的 fp16 峰值比例（56%）同量级。
2. **瓶颈是 tensor-core 的 operand load 吞吐**：load 在内时 9.6，load 在外时 49.2。
   其中 B（激活）load 占大头（A 提出去只省 ~15%），且 vector decode 没有任何改善——
   说明这个驱动在 Turing 上对 int8 `coopMatLoadTensorNV` 的每元素代价很高。
3. **同几何下 int8 确实比 fp16 快 1.9×**（9.6 vs 5.0），与硬件 2.11× 的比例一致；
   也就是说 int8 的优势是真的，只是**当前配置下根本用不到 MMA**。

### 10.3 为什么“换个大 tile 就好了”在这里行不通

ggml 的 fp16 cm2 路径用的是 **BM=128 × BN=128 × BK=64 / 256 线程**（operand 元素数/输出
只有本文配置的一半），而本文必须把每个 Q1_0 块的 int32 局部和落到共享内存才能做 per-128
缩放，于是共享内存 = `BM×BN×4B`：

- 64×64 → 16 KB ✔（当前配置）
- 128×64 → 32 KB ✔（实测反而更慢：4.64 TFLOPS，workgroup-scope 的 B 复制/分解开销）
- **128×128 → 64 KB ✘ 超过 Turing 的 48 KB**

即：**per-128 块级缩放把 tile 尺寸锁死在共享内存里**，而不能靠“大块 tensor load + 大 tile”
把 operand 元素数/输出降下来——这正是 ggml fp16 路径 27 TFLOPS 的来源。

### 10.4 结论

- CPU 的 1-bit 整数逻辑（per-128 Q8_128 激活 + 1-bit 权重解码 + int32 累加 + 每块一次浮点）
  已经**完整、正确地**移植到了 Vulkan 的 int8 张量核心路径上（cm1 与 cm2 两版都可运行）。
- 但在这块 GPU（Turing TU106）+ 这个驱动上，**int8 coopmat2 的 operand load 吞吐成为硬瓶颈**，
  使 int8 的 2.11× 算力优势无法兑现；per-128 块级缩放又限制了 tile 尺寸，
  最终 cm2 int8 比现有 fp16 反量化路径慢约 3.3×。
- 因此**不建议**把 prefill 的 Q1_0 换成 int8 路径；ggml 现有的
  `mul_mm_cm2 + dequantFuncQ1_0`（fp16 反量化 + fp16 张量核心）仍是最优解。
- 保留该实现仅供实验：`LUMINA_Q1_VK_INT8=1` 打开，默认关闭，对正常推理零影响。
- 一个反直觉但重要的实验记录：把激活改存成 ggml src1 的 `[N][K]`（k 连续）+ 块大小 `(1, BK)`
  + transpose view 之后，**这块驱动的 int8 `coopMatLoadTensorNV` 直接挂死**
  （进程卡在 GPU 等待，需重启才恢复），说明该组合在本驱动上不可用，已回退。

---

## 11. Q1_0 DP4A GEMV（token 生成，n = 1）——最终 **≈5.7×，已默认开启**

按“要性能最优、不必复刻 CPU 算法”的方针，重点放在 **token 生成**（n = 1）——交互式推理的
热路径，也是原生实现离硬件上限最远的地方。原生 Q1_0 GEMV
（`test-backend-ops perf -o MUL_MAT -p q1_0`，m=4096 n=1 k=14336）是 **88.6 µs**，
权重才 8.26 MB → 有效带宽仅 **93 GB/s**（≈ 本卡 ~430 GB/s 的 22%），即几乎全是
延迟/占用率问题，与算力无关。

### 11.1 做法（每条都由一次只改一处的 A/B 定下来）

1. **权重重排（缓存，每个张量一次）**：把 `[M][nb]` 的 18 B 块（fp16 d + 16 B 码字）重排成
   块主序 + 32 行合并的 `codes[(g*nb+ib)*32+l]`（16 B，**16 B 对齐**）+ `dsc[]`（fp16）。
   等价于 CPU 的 Q1LUT 重排。实测**没有它就没有性能**：直接在原布局上做向量化+DP4A
   只有 294 GMAC/s，重排后同一口径 >2000 GMAC/s。
2. **按 K 切分 + 共享内存归约（NSTRIP）**：一个工作组覆盖一个 32 行组，K 方向切成 8 个
   strip（8 warp × 32 lane）再归约。一行一线程（只有 4096 线程）完全藏不住显存延迟。
3. **激活 int8 + 每 128 值一个尺度（Q8_128）**，并用 `uvec4` 别名视图一次读 16 个激活
   （原本一次 DP4A 读一个 `u32` → 改成 4 个 DP4A 读一条 `uvec4`，减少 3/4 的取指）。
4. **每 lane 处理 2 行（RPL=2）**：一次激活块加载供两行复用，把「每条输出所需的
   load 指令数」再降约 40%。
5. **改用 CPU q1t 的恒等式 `dot(a,w) = 2·dot(a,bits) − sum(a)`**：权重的 nibble 只需两步
   展开成 0/1 字节（`nib*0x00204081 & 0x01010101`），不再需要 `±1` 的 7 步 SWAR；
   `sum(a)`（每 128 个激活之和）与权重行无关，由量化内核（`-DLUMINA_Q1_Q8_AXSUM`）预算到
   `axs[]`。内层 ALU 因此减少约 40%。
6. **修正 dispatch 网格**：工作组数 = `ceil(ceil(M/32)/RPL)`（过去写成 `M`，等于多算 32 倍
   工作组）。

### 11.2 结果（ggml-vulkan 内实测，默认开启；`LUMINA_Q1_VK_GEMV=0` 可关闭）

| n=1, m=4096, k=14336 | 原生（fp32 反量化 GEMV） | 本内核 |
|---|---:|---:|
| 单次提交耗时 | 88.58 µs | **15.6 µs** |
| 有效算力 | 1.33 TFLOPS | **7.54 TFLOPS** |

**≈ 5.7×**，且 `test -o MUL_MAT -p q1_0` 仍 **44/44 通过**；prefill（n=512）不受影响（27.2 TFLOPS）。

### 11.3 调参过程与口径（测试台 `vulkan-port/`）

为了快速扫参数，把测试台改成「**一条 command buffer 里连续 dispatch + dispatch 间插
pipeline barrier**」的口径——纯 GPU 串行、无 CPU 往返。扫出来的（M=4096 K=14336，µs/dispatch）：

| 变体 | 时间 | 说明 |
|---|---:|---|
| ns8（原始，重排+DP4A+8 strip） | 32.4 | 上一版 |
| ns8v（+ uvec4 读激活） | 28.1 | |
| ns8f（+ q1t 恒等式） | 30.0 | |
| ns8vf（两者都有） | 25.9 | |
| ns16vf | 25.8 | |
| **ns8r2 / ns16r2（再 + 每 lane 2 行）** | **25.0** | ★ 采纳 |
| bw（只流式读 codes 的带宽探针） | 19.0 | 访存天花板 |
| bw2（读 codes + dsc） | 21.4 | **必须读的流量**的天花板 |

两条重要结论：

- 选中的内核 **25.0 µs vs 21.4 µs（必须流量上限）**，即达到访存上限的 85%；
- 测试台的 barrier 口径**偏高**（barrier 会打断流水），所以 25 µs 不能当真实延迟；
  ggml 里连续提交的真实口径是 **15.6 µs**。测试台只用于相对比较。

### 11.4 大形状复核（工作集远超 L2，更贴近真实推理）

`test-backend-ops` 的 perf 反复读同一张 8 MB 权重，能吃到 L2，故 15.6 µs 偏乐观。
把形状放大到 **M=16384（权重 33 MB，远超 4 MB L2）** 复核：

| | µs/dispatch |
|---|---:|
| bw（只读 codes） | 70.1 |
| **bw2（codes+dsc，必须流量上限）** | **79.3** |
| ns8r2（采纳的内核） | 85.5 |
| ns16r2 / ns8r4 | 84.4 / 83.9 |

即大工作集下内核 = **必需流量的 93–94%**，已基本榨干访存；这也说明 dp4a 的整数算力
（Turing 128 int8 MAC/SM/cycle）对这个形状根本不是瓶颈。真实推理的收益预期在 **4–5×**
量级（L2 能吃到的部分比合成测试少）。

---

## 12. 缓存优化（repack cache）

GEMV 提速的前提是把权重重排成合并布局，而这需要**一份和权重等大的缓存**（16 B 码字 +
2 B 尺度 / 每 128 权重）。在默认开启后，这块缓存必须既正确又有界，本次做了三处改动：

1. **权重缓冲释放时清缓存**（原来不清）：
   - 之前只按 `(vk_buffer_struct*, offset)` 做键，缓冲销毁后条目一直留着 → **显存泄漏**，
     而且该地址被新分配复用时**会命中错误权重**（读到陈旧 repack，静默算错）。
   - 现在 `ggml_backend_vk_buffer_free_buffer()` 会先
     `ggml_vk_q1_repack_invalidate_buffer(dev_buffer)`，把同一 weight buffer 的所有 repack
     条目一起释放（`vk_buffer_struct` 析构即归还 VRAM），键失效问题一并消除。
2. **权重被改写时失效**：`set_tensor`、`set_tensor_2d`、以及新增的 `memset_tensor`
   都会失效对应条目的 repack（memset 也会改权重，原先是漏的）。
3. **缓存预算 + LRU 淘汰**：新增 `LUMINA_Q1_VK_CACHE_MB`（默认 **2048**，`0`=不限）。
   超出预算时按最近最少使用淘汰整条目；单个张量本身就超预算时仍然允许缓存（否则永远
   缓存不了）。这样默认开启不会因为 repack 把显存撑爆。

代价（未变）：开启 GEMV 后 Q1_0 权重的显存占用 ≈ **2×**（原始 + repack）。若模型太大，
用 `LUMINA_Q1_VK_GEMV=0` 关闭即可完全回到原路径。

---

## 13. prefill 实验（本次新增；结论：维持原生路径）

第 8–10 节已经证明「int8 张量核心 + 每 128 块级缩放」在 Turing 上打不过原生 fp16
反量化路径（8.3 vs 27.1 TFLOPS），原因是两条硬约束：
**per-128 缩放必须把 int32 局部和经共享内存落盘**（于是 tile 被 48 KB 共享内存锁死在
64×64），以及 **int8 `coopMatLoadTensorNV` 的每元素代价偏高**。

本次补了一组**不改代码、只切路径**的对照实验，用来确认「原生 27 TFLOPS 是否已经是
这块卡上的最优」，同时验证假设：

`test-backend-ops perf -o MUL_MAT -p q1_0`（m=4096 n=512 k=14336）：

| 路径 | us/run | TFLOPS |
|---|---:|---:|
| **默认：NV coopmat2 + 加载时 fp16 反量化** | **2210.8** | **27.20** |
| `GGML_VK_DISABLE_COOPMAT2=1`（回 KHR coopmat，共享内存暂存） | 2861.5 | 21.01 |
| `GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1`（纯 mul_mm 共享暂存） | 9284.1 | 6.48 |

结论：

- **原生 NV coopmat2 路径（27.2 TFLOPS）就是这块卡上 Q1_0 prefill 的最优解**；
  「自己写共享内存暂存」的路线比它慢 4×（6.5 TFLOPS），也慢于 KHR coopmat（21.0）。
- 因此 prefill **不改为整数路径**，维持现状；本次只把 Q1_0 的 prefill 数字在同一版
  二进制上重新确认了一遍（27.2 TFLOPS，与上一版一致）。
- 顺带得到一个反直觉但有用的对照：在 Q1_0（1-bit，18 B/128）这种布局上，
  `coopMatLoadTensorNV` 的加载时解码**并不是**瓶颈——把它关掉改手工暂存反而更慢；
  第 10 节 int8 内核慢的真正原因是**块级缩放带来的共享内存往返 + tile 受限**。

---

## 14. 开关与结论一览

| 开关 | 默认 | 作用 |
|---|---|---|
| （无） | — | Q1_0 token 生成走新的 DP4A GEMV（默认开启） |
| `LUMINA_Q1_VK_GEMV=0` | | 关闭 GEMV 加速，回到原生 fp32 反量化路径 |
| `LUMINA_Q1_VK_CACHE_MB=2048` | 2048 | repack 缓存预算（MB），`0` 为不限 |
| `LUMINA_Q1_VK_INT8=1` | 关 | prefill 的 int8 coopmat 实验路径（**慢，仅实验**） |
| `LUMINA_Q1_VK_CM1=1` | 关 | 让 prefill 实验走 KHR coopmat 而非 coopmat2 |

结论：

1. **token 生成（n=1）**：`Q1_0 DP4A GEMV` **≈5.7×**（ggml 口径 88.6 → 15.6 µs），
   大工作集下达到必需流量的 93–94%，已访存受限。**已默认开启**，正确性 44/44 通过。
2. **prefill（n≥128）**：维持原生 NV coopmat2（27.2 TFLOPS）。int8 路线在当前
   GPU/驱动上无法兑现其 2× 算力优势（块级缩放 + tile 限制），已用多组实验封死。
3. **缓存**：修掉了权重缓冲释放/覆写后的失效漏洞，并加了预算 + LRU，默认开启可安全使用。

---

## 15. 本次（第二轮）改动的文件

| 文件 | 改动 |
|---|---|
| `ggml/src/ggml-vulkan/vulkan-shaders/lumina_q1_gemv.comp` | 重写：8 strip、uvec4 激活、q1t 恒等式、每 lane 2 行 |
| `ggml/src/ggml-vulkan/vulkan-shaders/lumina_q1_quantize.comp` | 新增 `-DLUMINA_Q1_Q8_AXSUM`：额外输出每 128 个激活之和 |
| `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` | 注册 `lumina_q1_quantize_ax` |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | GEMV 默认开启、dispatch 网格修正、6 binding GEMV、缓存预算/LRU、释放与 memset 失效 |
| `vulkan-port/shaders/q1_gemv_tune.comp` | 新增：可参数化 strip-GEMV（NSTRIP/AVVEC/FAST/RPL/MODE）+ 带宽探针 |
| `vulkan-port/CMakeLists.txt`、`src/main.cpp` | 新增参数扫描、`--barrier` 口径、带宽探针 |
| `vulkan-port/win/run-tune.bat`、`verify-all.bat`、`ab-gemv.bat` | 构建/扫描/验证脚本 |
