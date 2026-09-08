# NATTEN 性能优化:逼近 memory roofline 了吗?(B70 实测)

承接 [`NATTEN_SYCL.zh.md`](NATTEN_SYCL.zh.md)(朴素 baseline)。本文把 1D NATTEN 从
0.05–0.12 TFLOPS 优化到 0.4–0.7 TFLOPS(**4.5–8.5x**),并用 roofline 模型解释**为什么
SIMT 到不了理论上限、以及要到上限必须上 DPAS**。所有数字为 Arc Pro B70 实测。

## 1. Roofline 框架(B70:160 TFLOPS 算力 / 608 GB/s 带宽)

`perf = min(160, AI × 608/1000)`,拐点 AI ≈ 263 FLOP/byte。NATTEN 是**局部窗口**注意力,
算术强度被窗口大小卡死:理想 SLM 复用下 `AI ≈ window_keys / bytes_per_elem`。bf16 下:

| 1D 窗口 | keys | AI | memory 天花板 |
|--------|-----:|----:|--------------:|
| w=8 | 17 | 8.5 | **5.16 TFLOPS** |
| w=16 | 33 | 16.5 | **9.99 TFLOPS** |
| w=32 | 65 | 32.5 | **19.68 TFLOPS** |

即 NATTEN 的理论上限是**这几 TFLOPS 的 memory roof**,不是 160(要到 160 需 ~526 宽窗口≈
全注意力,失去 NATTEN 意义)。

## 2. 优化历程(naive → D-split + SLM)

**朴素版**(`natten1d_sycl.cpp`):一个 work-item 一个 query,窗口×D 全串行,K/V 每个 query
都从 global 重读 → 实测 0.05–0.12 TFLOPS。

**优化版**(`natten1d_opt.cpp`)三招:
1. **D 拆到子组**:一个 **subgroup(G 条 lane)负责一个 query**,每 lane 只管 `DS=D/G` 个
   维度 → 私有累加器从 128 float 降到 2–4 float,**消除寄存器 spill、拉高 occupancy**;
   QK 分数 = 每 lane 部分点积 + `sycl::reduce_over_group` 广播。
2. **SLM tiling + K/V 复用**:work-group 处理 `TQ` 个连续 query,把窗口并集按 `TK` 分块
   载入 shared local memory,组内复用 → 干掉朴素版 `(2w+1)×` 的 global 重读。
3. **在线 softmax + 带状掩码**:running max/rescale 单趟完成,跳过窗口外的 key-tile。
   `--dtype bf16` 把字节减半。

## 3. 实测结果(runners/test_natten_sycl.py)

| shape (B,H,S,D,w) | naive | opt(bf16) | 加速 | memory 天花板 | **opt % roof** |
|-------------------|------:|----------:|-----:|--------------:|---------------:|
| 2,8,2048,64,8 | 0.092 | 0.415 | 4.5x | 5.16 | **8.0%** |
| 4,16,4096,64,8 | 0.086 | 0.399 | 4.6x | 5.16 | **7.7%** |
| 2,8,2048,128,16 | 0.075 | 0.691 | 9.2x | 9.99 | **6.9%** |
| 4,16,4096,128,32 | 0.081 | 0.686 | 8.5x | 19.68 | **3.5%** |

正确性:naive 与 opt 全部对 fp32 参考 `Disposition: Passed`。

## 4. 关键发现:卡在 ~4–8% roof,因为**不是 bandwidth-bound,是 issue-bound**

证据:
- **bf16 只在小形状涨(2x),大形状/D=128 反而持平或变慢**。若真贴着 memory roof,字节减半
  应稳定 ~2x。它没涨 → 瓶颈不在带宽。
- 手动扫 `qtile×kvtile`(见下)最好也只到 ~0.4–0.7 TFLOPS,再大窗口 % roof 反而**降**
  (3.5%)——说明限制来自**指令发射**,不是数据搬运。

根因:D-split 每个 key 都要一次 `reduce_over_group`(32 lane ≈ 5 次 shuffle),窗口 17 个 key
就是 ~85 次 shuffle/query,加上标量 FMA,**发射带宽先于访存带宽饱和**。这正是"memory-bound
应用如何多用计算"的**反面教材**:当 kernel 连 memory roof 都没碰到(处于 issue-bound),
提 AI 的常规手段(tiling/降精度)收益有限,**必须换更宽的计算原语(DPAS)把每次发射的 FLOP
数拉上去**。

tile 参数敏感性(B4H16S4096D64w8,bf16):

| qtile\kvtile | 16 | 32 | 64 |
|---|---:|---:|---:|
| 4 | 0.322 | 0.356 | 0.355 |
| 8 | 0.309 | 0.384 | 0.386 |
| 16 | 0.258 | 0.398 | 0.401 |
| 32 | 0.187 | 0.254 | 0.402 |

大 `kvtile` 更好(每次 barrier 复用更多 key、barrier 更少),但整体封顶 ~0.4。

## 5. 要到 roofline,得靠 DPAS(Path B)

- FA/GEMM 在 B70 能到 90–114 TFLOPS,靠的是 **DPAS 张量指令**(一条做 8×16×k 的 MAC),
  把 matmul 的指令数砍一到两个数量级 → 才能真正把访存管线喂满、贴上 roof。
- NATTEN = "带状掩码的 FA"。最省力的 DPAS 路线是 **fork sycl-tla 的 FMHA mainloop**,把
  `xe_fmha_fwd_mainloop.hpp:699` 的 causal 三角掩码换成带状 `|col−row|>w`,并把 K-tile 循环
  限制到窗口内。预期可逼近第 1 节的 memory 天花板(数~十几 TFLOPS)。
- 代价:改外部 CuTe kernel,复杂度和风险都高(故计划里定为"B later if needed")。

## 6. 复现

```bash
source /opt/intel/oneapi/setvars.sh
export AIBENCH_SYCL_TARGET=bmg-g31 ONEAPI_DEVICE_SELECTOR=level_zero:gpu
uv run python runners/test_natten_sycl.py            # naive vs opt + % roofline
uv run python runners/test_natten_sycl.py --quick
```

## 7. 结论

- **已达成**:1D NATTEN 从 0.05–0.12 → 0.4–0.7 TFLOPS(**4.5–9x**),正确性全过。
- **未达成理论上限**:SIMT 封顶 ~4–8% memory roof,因为**issue-bound**;继续 tiling/降精度
  边际收益低。
- **下一步(需拍板)**:上 **DPAS(Path B,fork FMHA 带状掩码)** 才能真正逼近 roofline。
