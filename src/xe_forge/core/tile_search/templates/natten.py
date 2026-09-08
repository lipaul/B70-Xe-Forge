"""
C++ template generator for NATTEN (neighborhood attention) tile tuning.

NATTEN reuses the sycl-tla FMHA DPAS kernel (FMHAConfigGenWithTileShape,
non-causal, non-persistent) with KernelArguments::natten_window set to the
half-window w, which activates the band-mask patch (see
examples/natten/sycl-tla-natten.patch). The tile-config space is identical to
FA (WgTileQ, WgTileK, WgTileV, SgTileQ, SgTileK, HeadDimQK, HeadDimV); the
window is a workload parameter, not a tuned tile.

Output format matches _parse_raw_output:
  - "Disposition: Passed" / "Disposition: Failed"
  - "Performance:   X.XXX  GB/s,    Y.YYY  TFlop/s,   Z.ZZZZ  ms"
"""

from __future__ import annotations

from xe_forge.core.tile_search.templates.fa_v2 import _DTYPE_MAP


def generate_natten_source(
    wg_tile_q: int,
    wg_tile_k: int,
    wg_tile_v: int,
    sg_tile_q: int,
    sg_tile_k: int,
    head_dim_qk: int,
    head_dim_v: int,
    window: int = 8,
    dtype: str = "bf16",
    mode: str = "prefill",
    iterations: int = 50,
) -> str:
    """Generate a complete NATTEN C++ source (FMHA + band-mask patch)."""
    from xe_forge.core.tile_search.templates import render

    element_type = _DTYPE_MAP.get(dtype, _DTYPE_MAP["bf16"])
    mode_cpp = "FMHAMode::Decode" if mode == "decode" else "FMHAMode::Prefill"

    return render(
        "natten.cpp.j2",
        wg_tile_q=wg_tile_q,
        wg_tile_k=wg_tile_k,
        wg_tile_v=wg_tile_v,
        sg_tile_q=sg_tile_q,
        sg_tile_k=sg_tile_k,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        element_type=element_type,
        mode=mode_cpp,
        window_default=window,
        iterations_default=iterations,
    )
