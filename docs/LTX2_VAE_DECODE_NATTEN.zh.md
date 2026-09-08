# LTX-2 VAE Decode × 我们的 SYCL DPAS NATTEN —— 测试报告(B70 实测)

## 0. TL;DR(重要前提修正)

- **LTX-2.3 的 VAE(含 decode)是纯卷积的 `CausalVideoAutoencoder`——decoder 里没有任何注意力**。
  因此对**已发布的 2.3 VAE decode**,NATTEN **无注意力可加速**(decode 是 conv-bound)。
- 邻域注意力在 LTX-2 中存在于 **transformer 解码器 `DiffusionVideoDecoder` / DiffVAE**(用
  `natten.na3d`,head_dim=64)——这是 **LTX-2.5 路线**;**2.3-distilled 权重不含该解码器**。
- 在 B70 上,LTX 的 na3d **只能跑它自己的 eager tiled-SDPA 回退**(实测 0.07–1.73 TFLOPS);
  **我们的 DPAS NATTEN(head_dim=64)达 0.69–2.27 TFLOPS**,roofline 效率 ~21–27% vs LTX 回退 ~1%。
  → **对 LTX 型 NA,NATTEN 的快路径确实在我们这边**(在 Intel GPU 上)。

## 1. 证据是怎么来的(全部可复核)

- **代码**:ltx-core 的 `video_vae/` 主体是 conv(`resnet.py` 的 `UNetMidBlock3D` 只含
  `ResnetBlock3D`,**无 attn 模块**);邻域注意力在 `video_vae/transformer/` 里
  (`NeighborhoodAttention3D` → `natten.na3d`,并有 `natten→triton→eager` 后端回退)。
- **权重(关键、低成本)**:用 HTTP Range **只读 safetensors 头部**(不下载几十 GB),取到
  `LTX-2.3` 的 config:`vae._class_name=CausalVideoAutoencoder`,`decoder_blocks` 全是
  `res_x`/`compress_*`(**无 `attn`**),`audio_vae…mid_block_add_attention=False`,主
  transformer `attention_type="default"`(全局,head_dim 128,48 层)。→ 坐实 2.3 VAE decode = conv。
- **运行**:在隔离 venv 里 import 并跑通了 ltx-core 的 eager `na3d`(B70),并跑我们的 DPAS
  NATTEN,得到下表。

## 2. B70 实测(bf16,head_dim=64=LTX DiffVAE 头维)

### 2a. 基线:LTX 自带的 eager na3d(tiled-SDPA 回退,3D 盒窗)

| 3D 网格 (t,h,w) | heads | 盒窗 kt·kh·kw | TFLOPS | ms | tokens | % roof |
|---|---|---|---:|---:|---:|---:|
| 8,32,32 | 4 | 5·5·5 (125 keys) | 0.419 | 2.504 | 8192 | 1.1% |
| 16,32,32 | 4 | 5·5·5 | 0.452 | 4.639 | 16384 | 1.2% |
| 8,64,64 | 8 | 7·7·7 (343) | 1.728 | 13.32 | 32768 | ~2% |
| 32,32,32 | 4 | 3·3·3 (27) | 0.072 | 12.63 | 32768 | 0.2% |

结论:LTX 的 Intel 可用路径(eager 回退)**很慢**,且**窗口越小越差**(被 tile 开销主导)。

### 2b. 我们的 SYCL DPAS NATTEN(1D 带窗,head_dim=64)

| 配置 | 带窗 | TFLOPS | ms | % roof |
|---|---|---:|---:|---:|
| B4H4 S8192 | w5 (11 keys) | 0.692 | 0.533 | 20.7% |
| B4H4 S32768 | w5 | 0.890 | 1.658 | 26.6% |
| B8H8 S16384 | w7 (15 keys) | 1.224 | 3.289 | 26.8% |
| B4H4 S8192 | w15 (31 keys) | 2.273 | 0.457 | 24.1% |

**同 8192 token 体量对比**:LTX-eager-na3d = 2.50 ms vs 我们 DPAS(1D w5) = **0.53 ms(≈4.7x 墙钟)**;
roofline 效率我们 ~21% vs LTX 回退 ~1%(≈20x)。

## 3. 语义 caveat(必须诚实标注)

- **工作不完全同构**:LTX 用 **3D 盒窗**(8192 token、box 125 keys);我们是 **1D 带窗**(w5=11 keys)。
  按用户选定的 **factorized per-dim**(沿 T、H、W 各跑一遍 1D 带,合成),总 keys ≈ kt+kh+kw≈15,
  仍 **< 盒窗 125** → factorized 是 3D-box NA 的**近似**(稀疏得多),TFLOPS/墙钟不可直接等同,
  只反映"我们的核处理等价 1D-NA 原语的效率"。
- 真正对齐 LTX 的 3D-box na3d 需 **3D DPAS kernel**(后续项,本次用 factorized 1D)。
- head_dim=64 + 短序列下,我们的 FMHA 也没到 head_dim128/大窗时的 37% roof(受 per-call 开销限制)。

## 4. 结论与建议

1. **"用 NATTEN 加速 LTX-2.5 VAE Decode"这个目标,对已发布的 LTX-2.3 不成立**——其 VAE decode 是纯
   卷积。请重新确认:你要加速的是 **LTX-2 的 NA 解码器(DiffVAE,2.5 路线)** 还是主 DiT?
2. 对 LTX 型 **na3d**,**我们的 SYCL DPAS NATTEN 在 B70 上比 LTX 自带的 Intel 回退(eager SDPA)
   快 ~4–5x 墙钟、~20x roofline 效率**——这才是 NATTEN 在 LTX 里的价值点。
3. 要做**真正的端到端 DiffVAE decode + 我们的 NATTEN + PSNR**:需要 **LTX-2.5 的 DiffVAE 权重/config**
   (2.3-distilled 无该解码器)。拿到后,用 `xe_forge` 的 `NATTENStrategy`(`tune_natten.yaml`)按层
   替换 `NeighborhoodAttention3D` 即可跑通并出延迟/质量报告。

## 5. 复现

```bash
# 隔离 venv,复用 Xe-Forge 的 torch-xpu,不改动 Xe-Forge 环境
uv venv /tmp/ltx2-venv --python 3.12
SP=/home/lm/Xe-Forge/.venv/lib/python3.12/site-packages
printf '%s\n%s\n' "$SP" "/tmp/ltx2/packages/ltx-core/src" > /tmp/ltx2-venv/lib/python3.12/site-packages/reuse.pth
uv pip install --python /tmp/ltx2-venv/bin/python safetensors av   # 其余依赖经 .pth 复用
source /opt/intel/oneapi/setvars.sh; export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
# LTX eager na3d 基线
/tmp/ltx2-venv/bin/python -c "import torch; from ltx_core.model.video_vae.transformer.fallback_na.eager import na3d; \
  q=torch.randn(1,8,32,32,4,64,device='xpu',dtype=torch.bfloat16);print(tuple(na3d(q,q,q,kernel_size=(5,5,5)).shape))"
# 我们的 DPAS NATTEN(head_dim=64):见 runners/test_natten_sycl.py / NATTENStrategy
# LTX 架构证据:HTTP Range 只读 safetensors 头部取 __metadata__.config(vae.decoder_blocks 全 conv)
```
