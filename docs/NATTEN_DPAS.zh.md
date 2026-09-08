# NATTEN Path B:DPAS(改 sycl-tla FMHA 加带状掩码)逼近 roofline 了吗?(B70 实测)

承接 [`NATTEN_OPTIMIZATION.zh.md`](NATTEN_OPTIMIZATION.zh.md)(SIMT 优化封顶 ~4–8% roof、
且 issue-bound)。本文按用户选择走 **Path B:fork sycl-tla 的 FMHA DPAS kernel,把 causal
三角掩码换成邻域带状掩码**,用真正的张量指令跑 NATTEN,并给出**为什么连 DPAS 也只到 ~11–12%
roof** 的定论。所有数字 B70 实测。

## 1. 给 sycl-tla 打的补丁(已存为 `examples/natten/sycl-tla-natten.patch`)

sycl-tla 的 FMHA 只有编译期 `CausalMask_` 布尔、三角掩码写死在 mainloop。改动(2 文件 45 行,
全部以 `natten_window>0` 为开关,**不影响 FA 原路径**):

1. **`xe_fmha_fwd_kernel.hpp`**:`KernelArguments` 末尾加 `int natten_window=0`(经
   `KernelParams=KernelArguments` 自动流到 `params.kernel.natten_window`)。
2. **kernel `operator()`**:当 `natten_window>0`,按 Q-tile 计算带状 K-tile 区间
   `[blk_k0, blk_k1)`(= 该 tile 内所有 query 窗口的并 `[q0-w, q0+BLQ-1+w]`),传给 mainloop。
   **关键**:mainloop 早已支持任意 `[blk_k0,blk_k1)`(用 `k_start` 偏移 prefetch 指针),
   所以"只算窗口内的 K-tile"几乎零成本复用。
3. **mainloop `operator()`**:新增 `int natten_window=0` 形参;在 causal 掩码块后加**带状掩码块**
   ——对每个 S 元素取全局 `(row,col)`,若 `|col-row|>w` 置 `-INFINITY`。

Xe-Forge 侧:`examples/natten/natten1d_fmha.cpp` 用 `FMHAConfigGenWithTileShape`(非 causal、
非 persistent)实例化 FMHA,并在 `KernelArguments` 末尾传入 `window`;自带 fp32 邻域参考校验。

## 2. 验证补丁有效

- **正确性**:多形状 `Disposition: Passed`(含 S=2048/4096、w=8/16/32)。
- **带状 K-loop 确实生效**:固定形状下,时间随窗口增长——w=8→3.8ms、w=128→5.8ms、
  w=512→14ms、w=2048(≈全注意力)→38ms。小窗口时间被**下限**卡住(见 §4)。

## 3. 结果:SIMT vs DPAS vs memory roofline(B4H16S4096D128,bf16)

| window | SIMT-opt | **DPAS(FMHA)** | memory 天花板 | DPAS % roof |
|-------:|---------:|---------------:|--------------:|------------:|
| w=8 | 0.630 | 0.601 | 5.16 | **11.6%** |
| w=32 | 0.686 | **2.291** | 19.68 | **11.6%** |
| w=64 | 0.772 | **3.878** | 38.91 | **10.0%** |

- **w≥32 时 DPAS 完胜 SIMT**(3.3x / 5x),因为它把 matmul 交给张量指令、且窗口越大每
  band-tile 的有用比例越高。
- 但 **DPAS 也封顶在 ~10–12% roof**,没有逼近理论上限。

## 4. 为什么 DPAS 也只到 ~11%(未调优时)

> 注:下表 DPAS 数字用的是**未调优**的 `WgTileQ=256`。经 §7 的 tile-tune 调优后
> 可到 **7.35 TFLOPS(37% roof)**。本节解释的是"为什么默认配置低效"。


1. **稠密矩形 vs 薄带的错配**:FMHA 对一个 Q-tile 算的是 `[BLQ × (BLQ+2w)]` 的**稠密矩形**,
   而 NATTEN 只需要 `2w+1` 的**对角带**。有效比例 ≈ `(2w+1)/(BLQ+2w)`。w=8、BLQ=256 时仅 ~6%,
   大部分 DPAS 算力浪费在带外矩形上(且我对每个 tile 都跑掩码,带外元素越多、掩码越贵)。
2. **小窗口被固定开销托底**:w≤32 时间恒为 ~3.8ms——被 per-work-group 的 prologue/epilogue +
   `BLQ` 跨度下限(BLQ=256 → 至少 8 个 K-tile,与 w 无关)吃掉。扫 `WgTileQ∈{32,64,128,256}`
   几乎无差别(都 ~3.8ms),因为省下的带外矩形被"更多 Q-tile 的固定开销"抵消。
3. **该 FMHA 配置本身在 D=128 MHA 下就不高**:即便 w=2048(全注意力)也只有 ~10.8 TF,
   远低于 FA tile-tune 在 GQA 配置下的 90 TF——tile 与 head_dim 不匹配 + 我的逐 tile 掩码开销。

**结论**:NATTEN 的薄对角带与 FA 的稠密矩形 tiling 是**架构性错配**。SIMT 受限于发射
(issue-bound),DPAS-FMHA 受限于带外浪费 + 固定开销托底,**两者都到不了 memory roof**。
要真正逼近 5–40 TF 的天花板,需要一个**沿对角带 tiling 的专用 kernel**(每个 work-group 只处理
窄带、K/V 载入一次、精确 2w+1 次点积),这已超出"fork FMHA"的范畴。

## 5. 复现

```bash
# 1) 给 sycl-tla 打补丁(仓库已存 patch;clone 在 v0.9.2)
cd /home/lm/sycl-tla && git apply /home/lm/Xe-Forge/examples/natten/sycl-tla-natten.patch
# 2) 环境同 SYCL_WORKFLOW.zh.md §2(bmg-g31 等)
# 3) 跑对比
uv run python runners/test_natten_sycl.py            # 含 SIMT vs DPAS vs % roof
```

## 6. Tier-2:接入 tile-tune 搜索(把 DPAS NATTEN 变成可自动调优)

把 NATTEN 接成 Xe-Forge 的 tile-tune 策略(`--tune-config` 的 `mode: natten`),让 LLM 搜索
FMHA tile 配置。因为 NATTEN 复用 FMHA,**直接复用 FA 的 config 空间与 `validate_fa_tile`**,
只加一个 workload 级 `window` 参数:

- `KernelType.NATTEN`(include 目录同 FA);`templates/natten.cpp.j2` + `generate_natten_source`
  (在 `KernelArguments` 末尾传 `window`);`NATTENStrategy`(继承 `FAStrategy`,重写
  source-gen / run-args / 硬件提示 / seed);CLI `mode: natten` 映射;`tune_natten.yaml`。

**关键调优点**:默认 FA seed 是 `WgTileQ=256`(对薄带很差)。给 `NATTENStrategy` 换了
**小 Q-tile seed(`qk_m=32, sg_q=16` → SgTileQ=2)** 和明确的硬件提示("薄带要用小 `qk_m` +
大 `sg_q`"),3 轮搜索稳定收敛到最优:

| 方案 | 配置 | TFLOPS | % memory roof |
|------|------|-------:|--------------:|
| naive | — | 0.08 | 0.4% |
| SIMT-opt | D-split+SLM | 0.69 | 3.5% |
| DPAS 未调优 | WgTileQ=256 | 2.29 | 11.6% |
| **DPAS + tile-tune** | **WgTileQ=32, SgTileQ=2** | **7.35** | **37.3%** |

即 tile-tune 把 DPAS NATTEN 从 11.6% 提到 **37% roof**(再 ~3.2x),验证了"搜索 FMHA tile
配置"这条路能显著改善薄带 NATTEN。搜索测了 11 个 config,最优 `qk_m=32, sg_q=16, pv_k=32`。

复现:`uv run python -m xe_forge.cli --dsl sycl --tune-config examples/tile_search/tune_natten.yaml`
(需先按 §5 打 sycl-tla 补丁)。

## 7. 交付物

- `examples/natten/sycl-tla-natten.patch` —— sycl-tla FMHA 的带状掩码补丁(可 `git apply`)。
- `examples/natten/natten1d_fmha.cpp` —— 用打补丁的 FMHA 跑 1D NATTEN(DPAS)。
- `src/xe_forge/core/tile_search/templates/natten.cpp.j2` + `natten.py` —— NATTEN tile-tune 模板。
- `src/xe_forge/core/tile_search/agent.py:NATTENStrategy` —— tile-tune 策略(复用 FA config)。
- `examples/tile_search/tune_natten.yaml` —— NATTEN 调优配置。
- `runners/test_natten_sycl.py` —— naive / SIMT-opt / DPAS 三方对比 + % roofline。

## 8. 结论(修正版)

- DPAS(FMHA fork)+ tile-tune 调优 → **7.35 TFLOPS,37% memory roof**,较未调优 DPAS 再 3.2x、
  较 SIMT 10.7x、较 naive ~90x。
- 仍未到 60–80% roof:受限于稠密矩形 vs 薄带的架构错配(§4)。再往上需**对角带专用 kernel**。
