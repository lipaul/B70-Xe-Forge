#!/usr/bin/env python3
"""
Manual runner for the Intel SYCL (DPC++) NATTEN kernels in examples/natten/.

Requires real Intel GPU hardware (torch.xpu.is_available()). Not a pytest suite.

Usage:
    python runners/test_natten_sycl.py            # correctness + naive-vs-opt sweep
    python runners/test_natten_sycl.py --quick    # smaller shapes only
    python runners/test_natten_sycl.py --dim 1d   # only 1D
    python runners/test_natten_sycl.py --dim 2d   # only 2D

Kernels are compiled with icpx (AOT -device from AIBENCH_SYCL_TARGET) and run via
SyclExecutor.execute_raw, which parses "Disposition:" (correctness) and
"Performance: ... TFlop/s ... ms" (benchmark). The memory-roofline ceiling for
each shape is computed here (Arc Pro B70: 608 GB/s) and reported as % of roof.
"""

import argparse
import sys
from pathlib import Path

import torch

from xe_forge.core.sycl_executor import KernelType, SyclExecutor

_here = Path(__file__).resolve().parent
_NATTEN_DIR = _here.parent / "examples" / "natten"
_K1D = str(_NATTEN_DIR / "natten1d_sycl.cpp")  # naive
_K1D_OPT = str(_NATTEN_DIR / "natten1d_opt.cpp")  # D-split + SLM (SIMT)
_K1D_DPAS = str(_NATTEN_DIR / "natten1d_fmha.cpp")  # DPAS via patched sycl-tla FMHA
_K2D = str(_NATTEN_DIR / "natten2d_sycl.cpp")  # naive
_K2D_OPT = str(_NATTEN_DIR / "natten2d_opt.cpp")  # D-split + SLM (SIMT)

PEAK_BW_GBPS = 608.0  # Arc Pro B70 (matches scripts/roofline.py preset)


def _ceiling_1d(S: int, D: int, W: int, nbytes: int) -> float:
    """Memory-roofline TFLOPS ceiling for 1D NATTEN with ideal K/V reuse."""
    pairs = sum(min(S - 1, i + W) - max(0, i - W) + 1 for i in range(S))
    flops = 4.0 * pairs * D  # per (b,h)
    membytes = 4.0 * S * D * nbytes  # Q,K,V,O once each (ideal reuse)
    ai = flops / membytes
    return ai * PEAK_BW_GBPS / 1000.0


def _ceiling_2d(Hi: int, Wi: int, D: int, KH: int, KW: int, nbytes: int) -> float:
    """Memory-roofline TFLOPS ceiling for 2D NATTEN with ideal K/V reuse."""
    pairs = 0.0
    rh, rw = KH // 2, KW // 2
    for r in range(Hi):
        for c in range(Wi):
            rlo, rhi = max(0, r - rh), min(Hi - 1, r + rh)
            clo, chi = max(0, c - rw), min(Wi - 1, c + rw)
            pairs += (rhi - rlo + 1) * (chi - clo + 1)
    flops = 4.0 * pairs * D
    membytes = 4.0 * Hi * Wi * D * nbytes
    return flops / membytes * PEAK_BW_GBPS / 1000.0


def _bench(ex: SyclExecutor, kernel: str, args: dict):
    res = ex.execute_raw(kernel_path=kernel, output_name="n", args=args, timeout=300)
    return res


def _correctness(ex: SyclExecutor, quick: bool, dim: str) -> bool:
    print("\n=== Correctness (verify=1, fp32 reference) ===")
    allok = True
    if dim in ("1d", "both"):
        shapes = [(1, 2, 64, 32, 4), (1, 4, 128, 64, 8)]
        if not quick:
            shapes += [(2, 8, 512, 64, 8), (1, 8, 1024, 128, 16)]
        for B, H, S, D, W in shapes:
            for kern, tag in ((_K1D, "1d-naive"), (_K1D_OPT, "1d-opt")):
                r = _bench(
                    ex,
                    kern,
                    {
                        "batch": B,
                        "heads": H,
                        "seq": S,
                        "dim": D,
                        "window": W,
                        "qtile": 8,
                        "kvtile": 32,
                        "dtype": "bf16",
                        "iterations": 5,
                        "warmup": 2,
                        "verify": 1,
                    },
                )
                ok = bool(r.success and r.output_correct is not False)
                allok &= ok
                print(
                    f"  [{'PASS' if ok else 'FAIL'}] {tag:<9} "
                    f"B{B}H{H}S{S}D{D}w{W} correct={r.output_correct}"
                )
                if not r.success and r.error_message:
                    print("         err: " + r.error_message.strip().splitlines()[-1][:120])
    if dim in ("2d", "both"):
        shapes = [(1, 2, 16, 16, 32, 3, 3), (1, 4, 24, 24, 64, 5, 5)]
        if not quick:
            shapes += [(1, 4, 32, 32, 64, 7, 7), (1, 8, 48, 48, 128, 3, 3)]
        for B, H, HI, WI, D, KH, KW in shapes:
            for kern, tag in ((_K2D, "2d-naive"), (_K2D_OPT, "2d-opt")):
                r = _bench(
                    ex,
                    kern,
                    {
                        "batch": B,
                        "heads": H,
                        "himg": HI,
                        "wimg": WI,
                        "dim": D,
                        "kh": KH,
                        "kw": KW,
                        "qtile": 8,
                        "kvtile": 32,
                        "dtype": "bf16",
                        "iterations": 5,
                        "warmup": 2,
                        "verify": 1,
                    },
                )
                ok = bool(r.success and r.output_correct is not False)
                allok &= ok
                print(
                    f"  [{'PASS' if ok else 'FAIL'}] {tag}  "
                    f"B{B}H{H}{HI}x{WI}D{D}{KH}x{KW} correct={r.output_correct}"
                )
                if not r.success and r.error_message:
                    print("         err: " + r.error_message.strip().splitlines()[-1][:120])
    return allok


def _benchmark(ex: SyclExecutor, quick: bool, dim: str) -> None:
    if dim in ("1d", "both"):
        print("\n=== 1D benchmark: naive vs optimized (bf16), % of memory roofline ===")
        print("  shape                 naive    opt    ceiling   opt%roof")
        rows = [(2, 8, 2048, 64, 8), (4, 16, 4096, 64, 8)]
        if not quick:
            rows += [(2, 8, 2048, 128, 16), (4, 16, 4096, 128, 32)]
        for B, H, S, D, W in rows:
            base = {
                "batch": B,
                "heads": H,
                "seq": S,
                "dim": D,
                "window": W,
                "iterations": 30,
                "warmup": 5,
                "verify": 0,
            }
            rn = _bench(ex, _K1D, base)  # naive (fp32)
            ro = _bench(ex, _K1D_OPT, {**base, "qtile": 16, "kvtile": 64, "dtype": "bf16"})
            ceil = _ceiling_1d(S, D, W, nbytes=2)  # bf16 ceiling
            n = rn.tflops or 0.0
            o = ro.tflops or 0.0
            pct = 100.0 * o / ceil if ceil else 0.0
            print(f"  B{B}H{H}S{S}D{D}w{W:<3}  {n:6.3f}  {o:6.3f}  {ceil:7.2f}   {pct:5.1f}%")
    if dim in ("2d", "both"):
        print("\n=== 2D benchmark: naive vs optimized (bf16), % of memory roofline ===")
        print("  shape                    naive    opt    ceiling  opt%roof")
        rows = [(2, 8, 64, 64, 64, 7, 7), (2, 8, 64, 64, 128, 5, 5)]
        if not quick:
            rows += [(4, 8, 128, 128, 64, 3, 3)]
        for B, H, HI, WI, D, KH, KW in rows:
            base = {
                "batch": B,
                "heads": H,
                "himg": HI,
                "wimg": WI,
                "dim": D,
                "kh": KH,
                "kw": KW,
                "iterations": 30,
                "warmup": 5,
                "verify": 0,
            }
            rn = _bench(ex, _K2D, base)
            ro = _bench(ex, _K2D_OPT, {**base, "qtile": 8, "kvtile": 32, "dtype": "bf16"})
            ceil = _ceiling_2d(HI, WI, D, KH, KW, nbytes=2)
            n = rn.tflops or 0.0
            o = ro.tflops or 0.0
            pct = 100.0 * o / ceil if ceil else 0.0
            print(f"  B{B}H{H}{HI}x{WI}D{D}{KH}x{KW}  {n:6.3f}  {o:6.3f}  {ceil:7.2f}  {pct:5.1f}%")


def _benchmark_dpas(ex: SyclExecutor, quick: bool, dim: str) -> None:
    if dim not in ("1d", "both"):
        return
    from xe_forge.core.tile_search import NATTENStrategy

    ex_n = SyclExecutor(kernel_type=KernelType.NATTEN, verify=False)  # FMHA includes
    st = NATTENStrategy()
    print("\n=== 1D DPAS (patched FMHA, tuned tile) bf16 vs fp8, % of roof ===")
    print("  shape                 SIMT   DPAS-bf16  DPAS-fp8  bf16ceil  bf16%roof")
    rows = [(4, 16, 4096, 128, 32)]
    if not quick:
        rows += [(4, 16, 4096, 128, 8), (4, 16, 4096, 128, 64)]
    for B, H, S, D, W in rows:
        wl = {
            "head_dim": D,
            "batch": B,
            "num_heads_q": H,
            "num_heads_kv": H,
            "seq_qo": S,
            "seq_kv": S,
            "window": W,
        }
        simt = _bench(
            ex,
            _K1D_OPT,
            {
                "batch": B,
                "heads": H,
                "seq": S,
                "dim": D,
                "window": W,
                "qtile": 16,
                "kvtile": 64,
                "dtype": "bf16",
                "iterations": 30,
                "warmup": 5,
                "verify": 0,
            },
        )

        def _dpas(dtype, wl):
            cfg = st.enrich_config(
                {"qk_m": 32, "qk_n": 32, "qk_k": 32, "pv_n": 32, "pv_k": 32, "sg_q": 16}, wl
            )
            src = st.generate_source(cfg, dtype)
            return ex_n.execute_raw(
                kernel_code=src,
                output_name=st.output_name(cfg) + "_" + dtype,
                args=st.build_run_args(cfg, wl),
                timeout=300,
            )

        b16 = _dpas("bf16", wl)
        f8 = _dpas("fp8_e4m3", wl)
        ceil = _ceiling_1d(S, D, W, nbytes=2)
        sv = simt.tflops or 0.0
        bv = b16.tflops or 0.0
        fv = f8.tflops or 0.0
        pct = 100.0 * bv / ceil if ceil else 0.0
        print(
            f"  B{B}H{H}S{S}D{D}w{W:<3}  {sv:5.3f}  {bv:7.3f}  {fv:7.3f}  {ceil:7.2f}   {pct:6.1f}%"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Intel SYCL NATTEN kernels on XPU")
    ap.add_argument("--quick", action="store_true", help="smaller shapes only")
    ap.add_argument("--dim", choices=["1d", "2d", "both"], default="both")
    args = ap.parse_args()

    if not torch.xpu.is_available():
        print("ERROR: no Intel XPU device (torch.xpu.is_available() is False)")
        return 1
    print(f"XPU device: {torch.xpu.get_device_name(0)}")

    ex = SyclExecutor(kernel_type=KernelType.GEMM, verify=True)
    allok = _correctness(ex, args.quick, args.dim)
    _benchmark(ex, args.quick, args.dim)
    _benchmark_dpas(ex, args.quick, args.dim)
    print("\n=== RESULT:", "ALL PASS" if allok else "SOME FAILED", "===")
    return 0 if allok else 2


if __name__ == "__main__":
    sys.exit(main())
