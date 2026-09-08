# Intel SYCL 版 NATTEN(邻域注意力)—— Xe-Forge 手写 DPC++ 实现(B70 实测)

本文记录在 Xe-Forge 里**从零实现一个 Intel SYCL(DPC++)版本的 Neighborhood Attention
(NATTEN)** 的过程。范围是 **Tier 1:自包含的 standalone kernel**(1D + 2D),
**正确性优先 + 基准测试**,通过 `SyclExecutor.execute_raw` 在 Arc Pro B70 上编译、校验、跑分。

> 背景:仓库原本**完全不支持 NATTEN**——`KernelType` 枚举、`tile_search` 策略、sycl-tla
> 里都没有邻域/带状注意力;sycl-tla 的 FMHA 只有一个编译期 `CausalMask_` 布尔、三角掩码写死在
> mainloop 里,无法直接复用。所以这是一次真正的新增实现。

---

## 1. 设计:复用 execute_raw 契约,零改动 SyclExecutor

Xe-Forge 的 `SyclExecutor.execute_raw(kernel_path=…, args={…})` 会:
- 用 `icpx -fsycl … -device bmg-g31` 编译**任意** `.cpp`(额外的 CUTLASS include/define 对
  纯 DPC++ kernel 无害);
- 把 `args` 拼成 `--key=value` 传给二进制;
- 从 stdout 解析 `Disposition: Passed|Failed` 和 `… TFlop/s … ms`。

因此 NATTEN kernel 写成**自包含程序**:host 端生成 fp32 输入 → 跑 SYCL kernel → host 端算
朴素 fp32 参考 → 比对打印 `Disposition` → 计时打印 `Performance`。**不依赖 CUTLASS/sycl-tla,
不改 `SyclExecutor`**。

新增文件:
- `examples/natten/natten1d_sycl.cpp` —— 1D 邻域注意力
- `examples/natten/natten2d_sycl.cpp` —— 2D 邻域注意力
- `runners/test_natten_sycl.py` —— 手动跑分脚本(需 XPU,仿 `runners/test_kb_examples.py`)

## 2. 算法(correctness-first)

**1D**:布局 `[B,H,S,D]`,半窗 `w`,query `i` 只看 `j∈[max(0,i-w), min(S-1,i+w)]`。
**2D**:布局 `[B,H,Hi,Wi,D]`,窗口 `kh×kw`,query `(r,c)` 看 `(r+dr,c+dc)`(`|dr|≤kh/2,
|dc|≤kw/2`,越界裁剪)。

两者同构:**一个 work-item 负责一个 query 位置**,对窗口做**两趟 softmax**
(先求 max,再 `exp` 累加分母与 `Σ pⱼ·Vⱼ`),`scale=1/√D`,fp32 累加,`v` 存 fp32。
窗口分数累加进**私有数组 `acc[MAX_D]`**,最后一次性写回全局。

## 3. 踩到的两个坑(都靠 execute_raw 复现并修掉)

1. **并发写 `dO` 的数据竞争**:最初计时循环 `for it: q.parallel_for(kernel);` 不等待,
   SYCL 队列会**并发执行**这些写同一 `dO` 的 kernel → 结果被破坏(表现为 `max_abs_err≈0.44`,
   且 iteration 越多越明显)。修复:计时循环里**每次 `.wait()`**(`for it: q.parallel_for(...).wait()`)。
2. **全局读-改-写太慢**:早期把累加直接写在全局 `Oi[d]+=e*Vj[d]`,每个 work-item 做上千次
   全局 RMW,慢到 0.016 TFLOPS。改成**私有 `acc[MAX_D]` 数组、末尾一次写回**,并加 `D≤MAX_D` 守卫。

## 4. 结果(B70,`Intel(R) Graphics [0xe223]`)

### 正确性(verify=1,对拍朴素 fp32 参考)—— 全部 PASS

| kernel | shape | window | 结果 |
|--------|-------|--------|------|
| 1D | B1 H2 S64 D32 | w=4 | PASS |
| 1D | B1 H4 S128 D64 | w=8 | PASS |
| 1D | B2 H8 S512 D64 | w=8 | PASS |
| 1D | B1 H8 S1024 D128 | w=16 | PASS |
| 2D | B1 H2 16×16 D32 | 3×3 | PASS |
| 2D | B1 H4 24×24 D64 | 5×5 | PASS |
| 2D | B1 H4 32×32 D64 | 7×7 | PASS |
| 2D | B1 H8 48×48 D128 | 3×3 | PASS |

### 基准(verify=0,30 iters)

1D `[B,H,S,D]`:

| shape | window | TFLOPS | ms |
|-------|--------|-------:|-----:|
| B2 H8 S1024 D64 | w=8 | 0.052 | 1.359 |
| B2 H8 S2048 D64 | w=8 | 0.086 | 1.664 |
| B4 H16 S4096 D64 | w=8 | 0.087 | 13.083 |
| B2 H8 S2048 D128 | w=16 | 0.075 | 7.351 |

2D `[B,H,Hi,Wi,D]`:

| shape | window | TFLOPS | ms |
|-------|--------|-------:|-----:|
| B1 H8 64×64 D64 | 7×7 | 0.122 | 3.192 |
| B1 H8 96×96 D64 | 7×7 | 0.115 | 7.727 |
| B2 H8 64×64 D128 | 5×5 | 0.080 | 10.040 |

## 5. 关于性能的诚实说明

这是**朴素 baseline**(0.05–0.12 TFLOPS),不是性能优化版:
- **瓶颈是 occupancy**:work-item 数 = `B·H·S`(或 `B·H·Hi·Wi`),每个 work-item 内部对
  window×D 串行。S=512→2048 时 work-item 翻 4 倍、TFLOPS 也大致翻 3–4 倍,说明是并行度不足而非
  算力不足。
- 想逼近 FA 级吞吐需要 **work-group 切分 D + 子组归约 + DPAS 分块 matmul**(即计划里的 Tier 3),
  本次按你选的"correctness-first"明确**不做**。
- 当前用 fp32 I/O;bf16 存储、causal+邻域混合掩码也是后续项。

## 6. 复现

```bash
# 环境同 SYCL_WORKFLOW.zh.md §2(source oneAPI;AIBENCH_SYCL_TARGET=bmg-g31 等)
source /opt/intel/oneapi/setvars.sh
export AIBENCH_SYCL_TARGET=bmg-g31 ONEAPI_DEVICE_SELECTOR=level_zero:gpu
uv run python runners/test_natten_sycl.py            # 全量
uv run python runners/test_natten_sycl.py --quick    # 小形状
uv run python runners/test_natten_sycl.py --dim 2d   # 只跑 2D
```

## 7. 后续可选(Tier 2/3,未做)

- **Tier 2**:把 NATTEN 接成 tile-tune 策略(新增 `KernelType.NATTEN` + Jinja 模板 +
  `validate_natten_tile` + `NATTENStrategy` + CLI),让 LLM 搜索 window/tile 参数。
- **Tier 3**:work-group + DPAS 的高性能 kernel,或 fork sycl-tla FMHA mainloop 把 causal 掩码
  换成带状掩码。
