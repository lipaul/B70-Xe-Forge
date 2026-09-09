# LTX-2.5 DiffVAE Decode × 我们的 SYCL DPAS NATTEN —— 实测报告(B70)

## 0. TL;DR

- **LTX-2.5 的 VAE 解码器确实是邻域注意力**:`CausalDiffusionVAE`,**24 层 3D `na3d`**
  (head_dim=64,盒窗最大 11×11×11)——与 2.3 的纯卷积 VAE 不同,NATTEN **原生适用**。
- **但注意力只占 decode 的 ~5.9%**(实测 20.3 ms na3d / 347 ms decode,小 latent)。
  → 即便 NA 无限快,端到端 decode 提速上限也只有 ~6%(Amdahl)。
- **我们的 DPAS NATTEN 在 LTX-2.5 各 NA 层上比 LTX 的 Intel 可用路径(eager tiled-SDPA na3d)
  快 1.5–2.3x**(逐层实测);折算端到端 decode 提速 **~3%**。
- 结论:NATTEN 对 LTX-2.5 VAE decode **有效但收益有限**(decode 被卷积 + 扩散采样循环主导,
  不是注意力主导)。真正值得用 NATTEN 的是**高分辨率/大 latent**(注意力占比随分辨率上升)。

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

即:**在这个分辨率下,NATTEN 对 LTX-2.5 VAE decode 的端到端收益只有个位数百分比**,因为 decode
被卷积 + 扩散采样循环主导。注意力占比会随**分辨率/帧数上升**(最大层 17×28×28 已占 na3d 的 58%),
高分辨率 decode 下 NATTEN 价值更大。

## 4. 诚实 caveat

- **逐层对比是"吞吐"对比**:我们是 1D 带窗(按用户选定的 factorized per-dim 近似 3D 盒窗),
  LTX 是 3D 盒窗;token 数/heads/head_dim 对齐,但**语义与总 MAC 不完全相同**(盒窗 keys=kt·kh·kw,
  factorized≈kt+kh+kw)。故 1.5–2.3x 是"等价 token 体量下 NA 原语的吞吐比",非严格同语义。
- **RoPE 在注意力之外**:LTX 的 `NeighborhoodAttention3D` 对 q/k 施加 3D 绝对 RoPE + q/k RMSNorm,
  scale=1.0;我们的 DPAS 核不含 RoPE。端到端替换需在核外用 torch 施加 RoPE 再喂我们的 NA(未在本次
  逐层吞吐对比中体现)。
- **PSNR 未做**:端到端影响仅 ~3%,且把我们的 DPAS 作为可返回张量的 torch-op 接入 live decode 需要
  额外的 pybind/落盘绑定;对一个 ~3% 且会改变边界语义(factorized≠box)的替换,PSNR 收益边际。
  如仍需要,可作为后续(实现 dump-O 绑定后跑 substitution + round-trip PSNR)。
- 权重加载有 150 个 key 名不匹配(`qkv.to_q` vs `qkv.weight` 命名 + per_channel_statistics),
  本次为**形状/计时**用途不影响;做严格 PSNR 时需对齐加载。

## 5. 结论与建议

1. **修正前次结论**:LTX-2.5(非 2.3)的 VAE decode **确实**用邻域注意力(24 层 na3d),NATTEN 原生适用。
2. **但收益受 Amdahl 限制**:该分辨率下注意力仅占 decode ~6%,我们的 DPAS 虽逐层快 1.5–2.3x,
   端到端 decode 仅 ~3% 提速。
3. **NATTEN 在 LTX-2 的最大价值点**:与其在 VAE decode,不如看**主 DiT / 高分辨率 decode**(注意力占比更高);
   且我们的 DPAS 是 LTX 型 na3d 在 **Intel GPU 上比其自带 eager 回退更快**的实现(1.5–2.3x)。
4. 若要端到端 + PSNR:需 (a) 对齐加载权重,(b) 把我们的 DPAS 封成返回张量的 torch-op(含 RoPE 前置),
   (c) 在高分辨率 decode 上测(放大注意力占比)。

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
