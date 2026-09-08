# 用官方 SHI-Labs/NATTEN 在 B70 上测性能(结论:我们的 SYCL 版更快)

目标:参考 **官方 NATTEN** 在 Intel Arc Pro B70 上测邻域注意力性能,和我们自研的 SYCL 版对比。
结论先行:**官方 NATTEN 的快 kernel 在 B70 上根本跑不了;唯一能在 Intel GPU 上跑的是它的
flex(Triton)后端,而且需要绕过它自己的设备门禁。绕过后它能在 B70 跑通、数值与 CPU 一致,
但性能 ~2.9 TFLOPS —— 我们的 DPAS 版 7.35 TFLOPS,快约 2.5x。**

## 1. 官方 NATTEN 的后端支持矩阵(读源码得出)

| 后端 | 实现 | 能在 B70(XPU)跑? |
|------|------|:---:|
| `cutlass-fmha/fna`、`hopper-*`、`blackwell-*` | CUDA + CUTLASS(`_libnatten`,需 NVIDIA SM) | ❌ 仅 CUDA |
| `reference` | **编译型 CUDA 参考 kernel**(`reference_na1d_forward`,带 `@custom_fwd(device_type="cuda")`) | ❌ 仅 CUDA |
| `flex` | 纯 PyTorch + `torch.nn.attention.flex_attention`(torch.compile→Triton) | ⚠️ 被 NATTEN 自己的门禁拒绝 |

关键点:
- `import natten` **不需要编译 CUDA 扩展**(`_libnatten` 会 fallback 到 stubs,`HAS_LIBNATTEN=False`)。
- 但 `can_run_flex_attention` 里写死:设备不是 CUDA/ROCm/CPU 就 `return False` → **XPU 被拒**。
- 所以官方 NATTEN 在 Intel GPU 上**开箱即用地什么都跑不了**;唯一出路是 flex 后端 + 绕过门禁。

## 2. 怎么在 B70 上跑起来(隔离环境)

- 单独 venv(`/tmp/natten-venv`),通过 `.pth` 复用 Xe-Forge 的 torch-xpu + 把 NATTEN 源码挂到
  `PYTHONPATH`(不重装 torch、不改动 Xe-Forge 环境)。
- 在探针脚本里 **monkeypatch** `can_run_flex_attention`(两处:`checks` 和 `flex` 模块)放行 XPU,
  并 `natten.allow_flex_compile()` 打开编译路径。
- 用 `natten.functional.na1d(..., backend="flex-fna", torch_compile=True)`。
- 完整脚本:`runners/natten_official_probe.py`(含 setup 注释)。

## 3. 结果

### 3.1 能跑 + 数值正确
- NATTEN flex 在 B70 上 **成功编译并运行**(Triton autotune 出 `triton_flex_attention` kernel)。
- **XPU vs CPU 数值一致**:max_abs_err = **4.8e-7**(fp32,L=256)。→ 官方 NATTEN 在 Intel GPU 上算得对。

### 3.2 性能对比(B70,bf16,K=kernel_size=65≈w32)

| 实现 | B4H16 L4096 D128 | TFLOPS | ms | % memory roof |
|------|------------------|-------:|----:|-------------:|
| 我们的 naive | | 0.08 | — | 0.4% |
| 我们的 SIMT(D-split+SLM) | | 0.69 | — | 3.5% |
| **我们的 DPAS(FMHA+tile-tune)** | | **7.35** | 1.18 | **37%** |
| **官方 NATTEN flex(Triton)** | | **2.90** | 3.01 | **14.7%** |

NATTEN flex 各形状(B70):

| shape | TFLOPS | ms |
|-------|-------:|----:|
| B2H8 L1024 D64 K65 | 0.097 | 2.81 |
| B4H16 L2048 D64 K65 | 0.751 | 2.90 |
| B2H8 L2048 D128 K65 | 0.390 | 2.80 |
| B4H16 L4096 D128 K65 | 2.899 | 3.01 |

**结论:在 Intel B70 上,我们自研的 SYCL DPAS NATTEN(7.35 TF)比官方 NATTEN 唯一能跑的
flex/Triton 后端(2.90 TF)快约 2.5x**,且更接近 memory roofline(37% vs 14.7%)。

## 4. 一个重要语义差异(边界处理)

对比官方 NATTEN 与我们的 kernel 时发现:**只有边界若干 query 的输出不同,内部完全一致**。
原因:
- **官方 NATTEN(flex)在边界用"平移满窗"**:把窗口中心 clamp,使每个 query 始终看满 `kernel_size`
  个 key(边缘 query 的窗口整体内移)。
- **我们的 SYCL kernel 用"裁剪窗"**:query i 只看 `[i-w, i+w] ∩ [0, L)`(边缘 key 数变少)。

两者都是合法的邻域注意力变体,但**边界语义不同**。这也意味着 NATTEN flex 每 query 固定算满 K 个
key(边缘多算),我们裁剪窗边缘少算——大 L 下差异可忽略,perf 对比仍公平。若要严格对齐官方语义,
需把我们的 kernel 边界改成平移满窗。

## 5. 注意事项 / 局限

- 官方 NATTEN 的**快 kernel(CUTLASS/Hopper/Blackwell)在 B70 上无法测**——它们是 CUDA-only,
  本机无 NVIDIA GPU。所以拿不到官方"speed-of-light"数字,只能比 flex。
- flex 的 `torch_compile` 走 `max-autotune`(6 个 Triton 配置),**编译很慢**(单形状数十秒)。
- 部分大形状(D=128 且 L 较大)在 triton-xpu 上编译会**静默崩溃**(segfault,无 Python 异常);
  上面能跑通的形状是稳定子集。
- 绕过 `can_run_flex_attention` 是**测试用 monkeypatch**,非 fork NATTEN。

## 6. 复现

```bash
git clone --depth 1 https://github.com/SHI-Labs/NATTEN /tmp/natten-src
uv venv /tmp/natten-venv --python 3.12
SP=/home/lm/Xe-Forge/.venv/lib/python3.12/site-packages
printf '%s\n%s\n' "$SP" "/tmp/natten-src/src" > /tmp/natten-venv/lib/python3.12/site-packages/reuse.pth
source /opt/intel/oneapi/setvars.sh; export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
/tmp/natten-venv/bin/python runners/natten_official_probe.py
```
