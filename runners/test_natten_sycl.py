#!/usr/bin/env python3
"""
Manual runner for the Intel SYCL (DPC++) NATTEN kernels in examples/natten/.

Requires real Intel GPU hardware (torch.xpu.is_available()). Not a pytest suite.

Usage:
    python runners/test_natten_sycl.py            # correctness + benchmark sweep
    python runners/test_natten_sycl.py --quick    # smaller shapes only
    python runners/test_natten_sycl.py --dim 1d   # only 1D
    python runners/test_natten_sycl.py --dim 2d   # only 2D

Each kernel is compiled with icpx (AOT -device from AIBENCH_SYCL_TARGET) and run
via SyclExecutor.execute_raw, which parses "Disposition:" (correctness) and
"Performance: ... TFlop/s ... ms" (benchmark) from the binary's stdout.
"""

import argparse
import sys
from pathlib import Path

import torch

from xe_forge.core.sycl_executor import KernelType, SyclExecutor

_here = Path(__file__).resolve().parent
_NATTEN_DIR = _here.parent / "examples" / "natten"
_KERNEL_1D = str(_NATTEN_DIR / "natten1d_sycl.cpp")
_KERNEL_2D = str(_NATTEN_DIR / "natten2d_sycl.cpp")


def _run_one(ex: SyclExecutor, kernel: str, name: str, args: dict) -> bool:
    res = ex.execute_raw(kernel_path=kernel, output_name=name, args=args, timeout=300)
    # verify=0 -> output_correct is None (not checked); success alone is enough.
    ok = bool(res.success and res.output_correct is not False)
    tf = f"{res.tflops:.4f}" if res.tflops is not None else "--"
    ms = f"{res.execution_time_ms:.4f}" if res.execution_time_ms is not None else "--"
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {name:<9} "
        f"tflops={tf:<8} ms={ms:<8} correct={res.output_correct}"
    )
    if not res.success and res.error_message:
        print("         err: " + res.error_message.strip().splitlines()[-1][:120])
    return ok


def _correctness(ex: SyclExecutor, quick: bool, dim: str) -> bool:
    print("\n=== Correctness (verify=1, fp32 reference) ===")
    allok = True
    if dim in ("1d", "both"):
        shapes = [(1, 2, 64, 32, 4), (1, 4, 128, 64, 8)]
        if not quick:
            shapes += [(2, 8, 512, 64, 8), (1, 8, 1024, 128, 16)]
        for B, H, S, D, W in shapes:
            allok &= _run_one(
                ex,
                _KERNEL_1D,
                "natten1d",
                {
                    "batch": B,
                    "heads": H,
                    "seq": S,
                    "dim": D,
                    "window": W,
                    "iterations": 5,
                    "warmup": 2,
                    "verify": 1,
                },
            )
    if dim in ("2d", "both"):
        shapes = [(1, 2, 16, 16, 32, 3, 3), (1, 4, 24, 24, 64, 5, 5)]
        if not quick:
            shapes += [(1, 4, 32, 32, 64, 7, 7), (1, 8, 48, 48, 128, 3, 3)]
        for B, H, HI, WI, D, KH, KW in shapes:
            allok &= _run_one(
                ex,
                _KERNEL_2D,
                "natten2d",
                {
                    "batch": B,
                    "heads": H,
                    "himg": HI,
                    "wimg": WI,
                    "dim": D,
                    "kh": KH,
                    "kw": KW,
                    "iterations": 5,
                    "warmup": 2,
                    "verify": 1,
                },
            )
    return allok


def _benchmark(ex: SyclExecutor, quick: bool, dim: str) -> None:
    print("\n=== Benchmark (verify=0, 30 iters) ===")
    if dim in ("1d", "both"):
        print("  1D neighborhood attention [B,H,S,D], window w:")
        rows = [(2, 8, 1024, 64, 8), (2, 8, 2048, 64, 8)]
        if not quick:
            rows += [(4, 16, 4096, 64, 8), (2, 8, 2048, 128, 16)]
        for B, H, S, D, W in rows:
            _run_one(
                ex,
                _KERNEL_1D,
                "natten1d",
                {
                    "batch": B,
                    "heads": H,
                    "seq": S,
                    "dim": D,
                    "window": W,
                    "iterations": 30,
                    "warmup": 5,
                    "verify": 0,
                },
            )
    if dim in ("2d", "both"):
        print("  2D neighborhood attention [B,H,Hi,Wi,D], window kh x kw:")
        rows = [(1, 8, 64, 64, 64, 7, 7)]
        if not quick:
            rows += [(1, 8, 96, 96, 64, 7, 7), (2, 8, 64, 64, 128, 5, 5)]
        for B, H, HI, WI, D, KH, KW in rows:
            _run_one(
                ex,
                _KERNEL_2D,
                "natten2d",
                {
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
                },
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
    print("\n=== RESULT:", "ALL PASS" if allok else "SOME FAILED", "===")
    return 0 if allok else 2


if __name__ == "__main__":
    sys.exit(main())
