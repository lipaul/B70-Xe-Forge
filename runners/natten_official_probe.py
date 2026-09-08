#!/usr/bin/env python3
"""
Probe + benchmark the OFFICIAL SHI-Labs/NATTEN on Intel XPU (B70).

NATTEN's fast kernels (cutlass/hopper/blackwell fmha/fna) and even its "reference"
backend are CUDA-only and cannot run on Intel GPUs. The ONE device-agnostic backend
is "flex" (torch.nn.attention.flex_attention -> torch.compile/Triton), but NATTEN's
own gate `can_run_flex_attention` refuses any device that isn't CUDA/ROCm/CPU.

This script monkeypatches that gate to allow XPU, then:
  1. verifies NATTEN-flex on XPU matches NATTEN-flex on CPU (numerics),
  2. benchmarks NATTEN-flex (compiled) on B70 for shapes comparable to our SYCL NATTEN.

Setup (isolated venv that reuses Xe-Forge's torch-xpu, NATTEN on PYTHONPATH):
    git clone --depth 1 https://github.com/SHI-Labs/NATTEN /tmp/natten-src
    uv venv /tmp/natten-venv --python 3.12
    echo "/home/lm/Xe-Forge/.venv/lib/python3.12/site-packages" > \
         /tmp/natten-venv/lib/python3.12/site-packages/reuse.pth
    echo "/tmp/natten-src/src" >> \
         /tmp/natten-venv/lib/python3.12/site-packages/reuse.pth
    source /opt/intel/oneapi/setvars.sh
    /tmp/natten-venv/bin/python runners/natten_official_probe.py

Requires: torch.xpu available, NATTEN importable (HAS_LIBNATTEN=False is fine).
"""

import argparse
import time

import natten
import natten.backends.configs.checks as checks
import natten.backends.flex as flex
import torch
from natten.functional import na1d


def _allow_flex_xpu(*args, **kwargs):
    # Bypass NATTEN's CUDA/ROCm/CPU-only gate so the flex backend can target XPU.
    return True


def _patch_gate():
    checks.can_run_flex_attention = _allow_flex_xpu
    flex.can_run_flex_attention = _allow_flex_xpu


def _band_ref(q, k, v, w):
    """Clipped-window neighborhood attention (our SYCL semantics), fp32."""
    _, L, _, D = q.shape
    scale = D**-0.5
    out = torch.zeros_like(q)
    for i in range(L):
        lo, hi = max(0, i - w), min(L - 1, i + w)
        s = torch.einsum("bhd,bkhd->bhk", q[:, i], k[:, lo : hi + 1]) * scale
        p = s.softmax(-1)
        out[:, i] = torch.einsum("bhk,bkhd->bhd", p, v[:, lo : hi + 1])
    return out


def agreement():
    torch.manual_seed(0)
    B, H, L, D, K = 1, 4, 256, 64, 17
    q = torch.randn(B, L, H, D)
    k = torch.randn(B, L, H, D)
    v = torch.randn(B, L, H, D)
    oc = na1d(q, k, v, kernel_size=K, backend="flex-fna", torch_compile=False)
    ox = na1d(
        q.to("xpu"),
        k.to("xpu"),
        v.to("xpu"),
        kernel_size=K,
        backend="flex-fna",
        torch_compile=True,
    )
    err = (oc - ox.cpu()).abs().max().item()
    print(f"[agree] NATTEN flex CPU vs XPU max_abs_err = {err:.3e}")
    return err


def bench(B, H, L, D, K, iters=20):
    torch.manual_seed(0)
    q = torch.randn(B, L, H, D, device="xpu", dtype=torch.bfloat16)
    k = torch.randn(B, L, H, D, device="xpu", dtype=torch.bfloat16)
    v = torch.randn(B, L, H, D, device="xpu", dtype=torch.bfloat16)
    for _ in range(3):  # warmup + compile/autotune
        na1d(q, k, v, kernel_size=K, backend="flex-fna", torch_compile=True)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        na1d(q, k, v, kernel_size=K, backend="flex-fna", torch_compile=True)
    torch.xpu.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / iters
    flops = 4.0 * B * H * L * K * D  # NATTEN shifted windows: full K per query
    return ms, flops * 1e-12 / (ms * 1e-3)


def main() -> int:
    ap = argparse.ArgumentParser(description="Official NATTEN flex probe/bench on XPU")
    ap.add_argument("--shapes", default="2 8 1024 64 65,4 16 2048 64 65,4 16 4096 128 65")
    args = ap.parse_args()

    if not torch.xpu.is_available():
        print("ERROR: no Intel XPU device")
        return 1
    print(
        f"natten {natten.__version__}  torch {torch.__version__}  "
        f"device {torch.xpu.get_device_name(0)}"
    )
    _patch_gate()
    agreement()
    print("\n=== NATTEN flex (compiled/Triton) perf on B70 ===")
    for spec in args.shapes.split(","):
        B, H, L, D, K = map(int, spec.split())
        try:
            ms, tf = bench(B, H, L, D, K)
            print(f"  B{B} H{H} L{L} D{D} K{K}: {tf:.3f} TFLOPS  {ms:.3f} ms")
        except Exception as e:
            print(f"  B{B} H{H} L{L} D{D} K{K}: FAIL {type(e).__name__}: {str(e)[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
