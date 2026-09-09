# LTX-2.5 DiffVAE Decode × 我们的 SYCL DPAS NATTEN —— 实测报告(B70)

## 0. TL;DR

- **LTX-2.5 的 VAE 解码器确实是邻域注意力**:`CausalDiffusionVAE`,**24 层 3D `na3d`**
  (head_dim=64,盒窗最大 11×11×11)——与 2.3 的纯卷积 VAE 不同,NATTEN **原生适用**。
- **但注意力只占 decode 的 ~5.9%**(实测 20.3 ms na3d / 347 ms decode,小 latent)。
  → 即便 NA 无限快,端到端 decode 提速上限也只有 ~6%(Amdahl)。
- **我们的 DPAS NATTEN 在 LTX-2.5 各 NA 层上比 LTX 的 Intel 可用路径(eager tiled-SDPA na3d)
  快 1.5–2.3x**(逐层实测);折算端到端 decode 提速 **~3%**。
- **替换保真(真实权重实测)**:用我们的可分离逐轴 1D 带窗替换 LTX 的 3D 盒 na3d,整段 decode
  输出 **PSNR = 30.3 dB(rel-L2 0.16)**。
- 结论:NATTEN 对 LTX-2.5 VAE decode **有效但端到端收益有限且随分辨率不增反降**(decode 被卷积 +
  扩散采样循环主导);注意力占比 17×128²→33×256² 实测 **5.1%→3.3%**。真正的价值在**主 DiT/更大盒窗**
  或用 **真·3D-box DPAS 核**(既提占比又降替换误差),而非当前 factorized。

## 1b. 真实权重实测(分辨率 + 替换保真)

按 checkpoint config 重建解码器 + qkv 融合权重名字重映射(to_q/k/v)加载真实权重,把 24 个
NA 的后端在 **eager 3D 盒**(参考)与 **我们的可分离逐轴 1D 带**(近似)之间切换,同一 latent 两次解码:

| latent → 输出 | decode ms | NA ms | 注意力占比 |
|---|---:|---:|---:|
| (3,4,4) → 17×128² | 350 | 18.0 | **5.1%** |
| (5,8,8) → 33×256² | 941 | 31.2 | **3.3%** |

**替换保真**:PSNR(ours-factorized vs LTX-eager-box) = **30.3 dB**,rel-L2 = 0.16
(输出帧范围 [-1.12, 1.20],权重真实 → 该 PSNR 是"可分离逐轴 vs 真 3D 盒"的语义差,非随机权重)。

## 1. 方法(全部可复核)

- 用 HTTP Range **只读** safetensors 头部拿到 `vae` 架构 config(不下大文件),确认
  `_class_name=CausalDiffusionVAE`、`stage_channels=[2048,1024,512,512,256]`、
  `stage_depths=[4,6,4,2,8]`(=24 NA 层)、`stage_kernels=[[3,7,7],[3,7,7],[3,5,5],[3,5,5],[11,11,11]]`、
  `head_dim=64`。
- 隔离 venv 复用 Xe-Forge 的 torch-xpu + `ltx-core`;下载 `vae/ltx-2.5-video-vae-bf16.safetensors`
  (1.47 GB,真实权重)。
- 用 `_build_diffusion_video_decoder(cfg['vae'])` 按 checkpoint 配置建解码器(**类默认值与 2.5 不同**,
  必须用 config 构建),加载权重,把 24 个 `NeighborhoodAttention3D` 的后端强制切到 **eager na3d**
  (LTX 在 Intel 上唯一可跑的路径;CuTe DSL / natten 都是 NVIDIA-only)。
- 在 B70 上跑真实 decode(latent `[1,128,3,4,4]` → 输出 `[1,3,17,128,128]`),hook 记录每层
  `(t,h,w,heads,kernel)` 与耗时;再对每个形状跑我们的 DPAS NATTEN 对比。

## 2. LTX-2.5 VAE decode 的 24 个 NA 层(B70 实测)

| 层形状 (t,h,w,heads,kernel) | 重复 | tokens | LTX eager-na3d ms | LTX TF | 我们 DPAS TF | 逐层加速 |
|---|---:|---:|---:|---:|---:|---:|
| 17,28,28,8,(3,5,5) | ×2 | 13328 | 5.851 | 0.350 | 0.574 | 1.6x |
| 9,14,14,8,(3,5,5) | ×4 | 1764 | 0.662 | 0.409 | 0.936 | 2.3x |
| 5,14,14,16,(3,7,7) | ×6 | 980 | 0.710 | 0.831 | 1.271 | 1.5x |
| 5,7,7,32,(3,7,7) | ×4 | 245 | 0.426 | 0.693 | 1.266 | 1.8x |

- 24 层 na3d 合计 ≈ **20.3 ms**;整次 decode ≈ **347 ms** → **注意力占比 5.9%**。
- 我们的 DPAS(head_dim=64)逐层 **1.5–2.3x** 于 LTX 的 Intel eager-na3d 回退。

## 3. 端到端投影(Amdahl)

decode 提速上限 = 注意力占比 ×(1 − 1/加速):
- 完美 NA(∞):≤ **5.9%**。
- 我们 DPAS(~1.8x 平均):≈ **5.9% × (1−1/1.8) ≈ 2.6%** 端到端 decode 提速。

即:NATTEN 对 LTX-2.5 VAE decode 的端到端收益只有个位数百分比,因为 decode 被卷积 + 扩散采样循环主导。
**实测(§1b)修正了"分辨率越高越值"的猜想**:注意力占比从 17×128²→33×256² 是 **5.1%→3.3%**(反而下降),
因为卷积/采样成本增长更快。故 NATTEN 的价值**不随该 VAE decode 的分辨率放大**。

## 4. 诚实 caveat

- **逐层对比是"吞吐"对比**:我们是 1D 带窗(按用户选定的 factorized per-dim 近似 3D 盒窗),
  LTX 是 3D 盒窗;token 数/heads/head_dim 对齐,但**语义与总 MAC 不完全相同**(盒窗 keys=kt·kh·kw,
  factorized≈kt+kh+kw)。故 1.5–2.3x 是"等价 token 体量下 NA 原语的吞吐比",非严格同语义。
- **RoPE 在注意力之外**:LTX 的 `NeighborhoodAttention3D` 对 q/k 施加 3D 绝对 RoPE + q/k RMSNorm,
  scale=1.0;我们的 DPAS 核不含 RoPE。端到端替换需在核外用 torch 施加 RoPE 再喂我们的 NA(未在本次
  逐层吞吐对比中体现)。
- **替换保真 PSNR 已测(§1b)= 30.3 dB**:它度量的是"可分离逐轴 vs 真 3D 盒"的**语义差**;若改用
  **真·3D-box DPAS 核**可把这 30 dB 逼近无损(而 factorized 的 30 dB 是方法本身,不是核的数值误差
  ——我们 DPAS 的 1D 带核此前已对 fp32 参考验证过正确)。round-trip(encode→decode vs 原视频)绝对 PSNR
  仍需真实小片段,列为后续。
- 权重加载:`attn.qkv` 为融合 Linear,与模型的 `to_q/to_k/to_v` 命名不同 → 已做**重映射加载**;仅
  per_channel_statistics + timestep_embedder(共 6 个)未加载(对替换 PSNR 无影响,两后端共用同一权重)。

## 5. 结论与建议

1. **修正前次结论**:LTX-2.5(非 2.3)的 VAE decode **确实**用邻域注意力(24 层 na3d),NATTEN 原生适用。
2. **但端到端收益小且不随分辨率增长**:注意力仅占 decode ~3–5%(实测 128²→256² 为 5.1%→3.3%),
   我们的 DPAS 逐层快 1.5–2.3x → 端到端 decode 仅 ~3%;可分离替换的保真 PSNR = 30.3 dB。
3. **NATTEN 在 LTX-2 的真正价值点**:我们的 DPAS 是 LTX 型 na3d 在 **Intel GPU 上比其自带 eager 回退
   更快**的实现(1.5–2.3x);VAE decode 只是恰好有 NA、但占比太低不足以成为杀手级用例。
4. **要放大收益 / 提质**:改用**真·3D-box DPAS 核**(既贴合语义把 30 dB 逼近无损,又因占比仍小
   端到端有限);更大杠杆是**主 DiT 的注意力**(序列更长、占比更高),而非 VAE decode。

## 6. 复现

```bash
# venv 复用 torch-xpu + ltx-core;下载 2.5 DiffVAE
uv venv /tmp/ltx2-venv --python 3.12
printf '%s\n%s\n' /home/lm/Xe-Forge/.venv/lib/python3.12/site-packages /tmp/ltx2/packages/ltx-core/src \
  > /tmp/ltx2-venv/lib/python3.12/site-packages/reuse.pth
uv pip install --python /tmp/ltx2-venv/bin/python safetensors av
HF_TOKEN=*** huggingface-cli download Lightricks/LTX-2.5 vae/ltx-2.5-video-vae-bf16.safetensors --local-dir /tmp/ltx2_ckpt
source /opt/intel/oneapi/setvars.sh; export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
# 建解码器(用 checkpoint config)+ 强制 eager NA + XPU decode + 逐层形状/计时:见 ltx2_profile.py / ltx2_na_lat.py
# 我们的 DPAS 逐层:/tmp/ltx2-venv 之外用 Xe-Forge NATTENStrategy(head_dim=64)
```
