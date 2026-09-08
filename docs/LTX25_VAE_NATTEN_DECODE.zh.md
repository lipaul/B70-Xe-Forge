# LTX-2.5 VAE Decode × 我们的 SYCL DPAS NATTEN —— 可行性与代理测试报告(B70)

结论先行:**LTX-2.5 的 VAE 解码用的是 3D 邻域注意力(NeighborhoodAttention3D),而我们
自研的 DPAS NATTEN 目前是 1D(2D 只有 SIMT 版)——维度对不上;加之本机没有 LTX-2.5 权重、
也没有 build 它的 venv,所以无法端到端跑真实 decode。** 本报告给出架构证据、我们在 LTX 的
head_dim=64 上跑的**代理性能**,以及把 DPAS NATTEN 真正接入 VAE decode 的具体路径。

## 1. 我们的 SYCL DPAS NATTEN 代码在哪

| 文件 | 作用 |
|------|------|
| `examples/natten/sycl-tla-natten.patch` | 给 sycl-tla FMHA 打补丁:加 `natten_window` 走带状掩码 + 把 K-loop 限制到窗口 |
| `examples/natten/natten1d_fmha.cpp` | **独立可编译的 1D DPAS NATTEN**(直接实例化打了补丁的 FMHA) |
| `src/xe_forge/core/tile_search/templates/natten.cpp.j2` | **DPAS NATTEN 的代码生成模板**(tile 参数化) |
| `src/xe_forge/core/tile_search/templates/natten.py:generate_natten_source` | 模板渲染入口 |
| `src/xe_forge/core/tile_search/agent.py:NATTENStrategy` | tile-tune 策略(复用 FA config + 带状语义) |
| `examples/tile_search/tune_natten.yaml` | 调优配置;`outputs/natten_tune.json` |

> SIMT(非 DPAS)版另见 `natten1d_opt.cpp`、`natten2d_opt.cpp`。DPAS 的 = 上面基于 FMHA 的那几件,**且只有 1D**。

## 2. LTX-2.5 VAE 解码的注意力到底是什么(读源码)

- 模块:`packages/ltx-core/.../video_vae/transformer/attention.py:NeighborhoodAttention3D`,
  输入布局 `(B, T, H, W, heads, head_dim)`,**默认 `head_dim=64`**,`num_heads = dim/head_dim`,
  通过 `natten.na3d(...)` 执行;边界用 **NATTEN 的 inward-shift**(和我们发现的一致)。
- LTX 自带 NVIDIA 专用 3D-NA CuTe DSL kernel(`ltx-kernels/.../vae/*dsl*`,`natten.na3d` 的
  drop-in,Blackwell),以及 `natten` extra——但那些是 CUDA,**B70 上用不了**。
- 解码器 `diffusion_video_decoder.py` 的分阶段配置(即 decode 各 stage 的 NA 规模):

| stage | channels | heads(d=64) | 3D kernel (kt,kh,kw) | depth | 平面分辨率 HxW |
|------:|---------:|------------:|:--------------------:|------:|:--------------:|
| 0 | 1024 | 16 | (3,7,7) | 4 | 32×48 |
| 1 | 512  | 8  | (3,7,7) | 6 | 64×96 |
| 2 | 256  | 4  | (3,5,5) | 4 | 64×96 |
| 3 | 256  | 4  | (3,5,5) | 2 | 128×192 |
| 4 | 128  | 2  | (3,3,3) | 2 | 256×384 |
| diff-5 | (config) | — | (3,7,7) | 8 | — |

(平面分辨率按脚手架默认输出 1536×1024、121 帧推算,latent 网格 `F'=1+(F-1)/8=16`,`H'=H/32=32`,`W'=W/32=48`,逐级上采样。)

**要点:每层的 NA 是 3D 窗口 `(kt,kh,kw)=(3,7,7)…(3,3,3)`,head_dim=64。**

## 3. 为什么不能直接"用我们的 DPAS NATTEN 跑 decode"

1. **维度不匹配**:我们的 DPAS NATTEN 是 **1D**(沿单条序列的带状窗口)。VAE 要的是 **3D 空间-时间窗口**。1D 带 ≠ 把 `(T,H,W)` 展平后的带状——语义完全不同。
2. **2D 只有 SIMT**:能覆盖 3D 分解里的"单平面空间 NA"(LTX 的 DSL kernel 也是这么分解:每平面 2D + 时间 1D),但那是 SIMT,不是 DPAS,且在小网格上 overhead-bound。
3. **缺权重/环境**:`~/paul/models` 为空、`vendor/LTX-2` 未 build(setup.sh 未跑),无法端到端解码;真实 decode 需要下载权重 + 用官方补丁好的 LTX venv。

## 4. 代理性能测试(在 LTX 的真实 head_dim=64 与阶段形状上,跑我们现有的 kernel)

### 4.1 1D DPAS NATTEN @ head_dim=64(作为"时间轴/展平"吞吐参考,非等价 3D)

| L(tokens) | w=8 | TFLOPS | % roof(5.2) |
|----------:|----:|-------:|------------:|
| 1024 | ✓ | 0.35 | 6.8% |
| 4096 | ✓ | 0.69 | 13.3% |
| 16384 | ✓ | 1.37 | 26.5% |

→ 我们的 DPAS **支持 head_dim=64**,且随序列长度 occupancy 提升逼近 roof(和之前规律一致)。但这仍是 1D。

### 4.2 2D SIMT NATTEN @ LTX 空间平面(3D-NA 的 per-plane 分量,head_dim=64,bf16)

| 阶段 | planes×(HxW)×win | 我们的2D-SIMT | % roof |
|------|------------------|--------------:|-------:|
| stage0 | 256×(32×48)×7×7 | 0.30 TF | 2.0% |
| stage1 | 128×(64×96)×7×7 | 0.30 TF | 2.0% |
| stage2 | 128×(128×192)×5×5 | 0.26 TF | 3.4% |

→ 空间平面 NA 能算对,但小网格下被 barrier/`reduce_over_group` 固定开销拖住(~2–3% roof)。

### 4.3 参照:官方 NATTEN flex(Triton)在 B70(我们已测)
同一 head_dim 下,NATTEN flex 编译版在更大 1D 形状最高 ~2.9 TF(14.7% roof)。**我们的 1D DPAS
在同规模下(7.35 TF/37%)显著更快**——但那优势目前只在 1D;3D 我们还没 kernel。

## 5. 要把 DPAS NATTEN 真正接入 LTX-2.5 VAE decode,需要

1. **写一个 3D SYCL DPAS NATTEN**:把 FMHA 带状掩码推广到 `(kt,kh,kw)` 三轴(LTX 的 DSL 也是
   "单平面 2D 空间 NA + 时间 1D NA" 的分解,可复用我们已有 2D/1D 思路,但要 DPAS 化 + 支持
   inward-shift 边界以匹配 `natten.na3d` 语义)。
2. **封成 torch custom op(xpu)**,替换 LTX `NeighborhoodAttention3D` 的 backend(它本就支持注入
   `attention_function`),即 `configure_natten_backend` 那条路。
3. **搭真实 decode**:`bash setup.sh` 建 `vendor/LTX-2/.venv` + 下载 LTX-2.5 VAE 权重,只实例化
   decoder、喂一个随机/真实 latent 跑 decode,对比 我们的3D-DPAS vs natten.na3d(flex) vs CPU。

## 6. 现状结论
- 代码位置见 §1;DPAS NATTEN = **1D**,head_dim 支持到 64。
- LTX-2.5 VAE decode = **3D NA**,本机**无权重/无 LTX venv** → 端到端 decode 测试**未能执行**。
- 已给**代理数据**证明:我们 1D DPAS 能跑 LTX 的 head_dim=64 且比官方 flex 快;3D 接入的**关键缺口是缺一个 3D DPAS kernel**(和把 1D/2D 组合成 3D 的 reduction/边界语义)。

> 下一步待定:是否要我 **实现 3D SYCL DPAS NATTEN**(§5.1,真正让"DPAS NATTEN 服务 VAE decode"成立),以及**是否下载 LTX-2.5 VAE 权重 + build venv** 跑端到端。
