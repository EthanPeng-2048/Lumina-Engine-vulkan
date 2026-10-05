# 模型无损压缩（Build 8）

> 🌐 **中文** | [English](LosslessCompression.en.md)
>
> 只想看“怎么压、怎么用”？请直接看用户指南：[ModelCompression](ModelCompression.md)。

## 目标

在**不量化、不剪枝**的前提下减小 GGUF 模型文件体积：保持每一个权重值、张量数量与
张量类型完全不变，仅改变文件的**存储编码**，并在加载时逐字节还原。

## 为什么不能只靠通用压缩

`SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf` 的权重已经是 Q4_K / Q6_K。逐字节熵约
7.8–7.9 bit/byte，直接 `zstd -19` 全文件压缩仅约 **96.9%**。原因是 k-quant 块把两种
性质完全不同的数据交织在一起：

| 字段 | 占块比例 | 熵 |
|------|---------:|----:|
| `d` / `dmin`（fp16 尺度） | 4 / 144 | ~5.8 bit/byte |
| `scales`（6-bit 子尺度 / min） | 12 / 144 | ~7.1 bit/byte |
| 4-bit 权重（`qs`） | 128 / 144 | ~7.7 bit/byte |

把高熵权重流和低熵尺度流混在一起送进压缩器，会让尺度流几乎得不到有效建模。

## 方案：字段拆分 + 上下文排序 + DEFLATE（容器 v2）

`tools/lumina_gguf_pack.py` 对每个张量做可逆的**字段拆分（field split）**；容器 v2 在此之上
又加入了若干保持可逆的变换：

- **Q4_K**：把每块拆成 `d` / `dmin` 的 fp16 字节平面、8 个 6-bit `scale` 与 8 个 `min`
  字段，以及 4-bit 量化值。量化值按子块分组后，再按上下文键 `(scale << 6 | min)` 做**稳定
  排序**——分布相近的子块被排到一起，DEFLATE 对每个子分布的建模效率明显提高。
- **Q6_K**：`d` 的 fp16 字节平面、`scales` 的奇偶拆分，以及 `ql` 低/高半字节 + `qh` 拆分。
- **F32**：整块。

所有流在整个张量上连续排列，再经 zlib/DEFLATE 压缩。加载时按张量逐个解压、反排序并重建
原始块字节。容器格式（v2）见 `tools/lumina_gguf_pack.py` 顶部注释；C++ 解码器同时兼容 v1。

### 实测

| 文件 | 大小 | 比例 |
|------|-----:|-----:|
| 原始 `...Q4_K_M.gguf` | 2,630,213,408 B | 100% |
| 压缩后（Build 8 交付，容器 v2） | 2,479,779,960 B | **94.28%** |

节省 **143.5 MiB**（比容器 v1 的 117.9 MiB 再多省 **25.6 MiB**）。解压结果与原文件 SHA-256 完全一致：

```
fb584d0b0f79f9e9944f91ac061e24726a8d014b6702213bfd5585729db7ab61
```

### 还剩多少空间

模型的 99.7% 数据是 Q4_K / Q6_K 的 4/6-bit 量化值：本模型 Q4_K 量化值的零阶熵为
3.87 bit / 4 bit（约 96.7% 上限），条件熵（已知 `scale, min`）为 3.82 bit（约 95.6%），
v2 的上下文排序已经把这个流压到 95.5%，**基本触及无损上限**。Q6_K 量化值几乎是均匀分布
（接近 6 bit 满熵），因此通用压缩几乎无能为力。换句话说，在不量化、不剪枝的前提下，
94.3% 已接近理论极限；想再小一个数量级只能靠有损手段。

## 引擎侧实现

- `src/lumina-gguf.cpp` / `.h`：识别容器魔数 `LGGC`，把压缩文件流式解压到系统临时目录
  （`%TEMP%\lumina-<hash>.gguf`），之后走原有 GGUF 加载路径；解码器同时支持容器 v1 与 v2。
- **扩展名约定**：打包器默认把 `A.gguf` 输出为 `A.gguf.lumina`（原文件保留）。引擎**只认文件头
  的 `LGGC` 魔数，不依赖扩展名**，因此 `.gguf` 与 `.gguf.lumina` 均可直接加载；服务端模型目录
  列表与命令行补全也会把 `.gguf.lumina` 识别为模型。
- `src/llama.cpp`：`llama_model_load_from_file()` 调用
  `lumina_gguf_resolve_path()`；非压缩文件原样返回，零影响。
- `src/miniz/`：内置单文件 DEFLATE（MIT / 公有领域），不引入外部依赖。
- 不量化、不剪枝：所有张量逐字节还原；临时文件缓存复用，模型释放后可由系统清理。

## 使用

```bat
REM 打包（离线，一次）：默认 A.gguf -> A.gguf.lumina，原文件保留
python tools\lumina_gguf_pack.py model.gguf

REM 逐字节校验
python tools\lumina_gguf_unpack.py verify model.gguf.lumina model.gguf

REM 直接加载（引擎自动识别并还原；.gguf 与 .gguf.lumina 均可）
Lumina-Engine-cli.exe -m model.gguf.lumina -p "..." -n 16
```
