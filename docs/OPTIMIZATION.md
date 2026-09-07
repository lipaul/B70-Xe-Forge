# Xe-Forge Optimization Strategy — General Framework and a Worked Analysis

Two parts: **A** describes the generalized optimization strategy (grounded in the code),
**B** analyzes the concrete `14_Gemm_Divide_Sum_Scaling` run on an Intel Arc Pro B70.
A companion step-by-step log narrative lives in [`WORKFLOW.md`](WORKFLOW.md).

---

## Part A — The generalized optimization strategy

### A.1 The control loop

Xe-Forge is not a rule-based pass manager; it is an **LLM that proposes code, validated on
real hardware**. `pipeline.py:XeForgePipeline.optimize` runs:

```
measure baseline
  → ANALYSIS      (LLM lists scored issues)
  → PLANNING      (map issues → ordered subset of stages)
  → for each active stage:
        OPTIMIZE  (LLM rewrites the kernel for that stage's issues)
        CoVeR     (compile → run → compare → retry)
        RE-ANALYZE (later stages see the updated code)
  → final full-warmup measurement vs baseline
```

The defining property is the **CoVeR gate** (compile→run→compare→retry, `agents/cover.py`):
every proposed kernel must (a) match the baseline output within tolerance and (b) be faster
than the current best, or it is discarded and the agent retries with the error fed back
(up to `AGENT_MAX_ITERATIONS`). Worst case the stage fails and the **original code is kept** —
the pipeline is safe-by-default and never ships a wrong or slower kernel.

### A.2 The stage catalog (the "strategy space")

Stages are the fixed vocabulary of optimizations. Each is backed by a family of
`IssueType`s the analyzer can emit (`models.py`), so analysis and optimization share a
taxonomy. Canonical order is `planner.DEFAULT_STAGE_ORDER` (ANALYSIS always runs first):

| Stage | Fixes (issue types) |
|-------|---------------------|
| `algorithmic` | redundant_computation, suboptimal_algorithm, associativity_reorder, common_subexpression, algebraic_simplification, **cacheable_intermediate**, loop_invariant_code, unnecessary_materialization, gemm_simplification, reduction_tree_suboptimal |
| `discovery` | open_ended (novel, high-value ideas with a required before/after sketch) |
| `dtype_fix` | dtype_float64, dtype_precision, dtype_input_conversion |
| `fusion` | unfused_kernels, unfused_elementwise, unfused_reduction, fusion_register_pressure, fusion_replaces_vendor, fusion_noop |
| `memory_access` | manual_pointer_arithmetic, missing_boundary_check, transpose_in_loop, missing_tma, uncoalesced_access, device_host_sync, non_contiguous_input, cache_eviction_risk, long_liveness, high_register_pressure |
| `block_pointers` | missing_block_pointers, block_ptr_boundary_wrong, block_ptr_multiple_of_misuse |
| `persistent_kernel` | missing_persistent, persistent_num_progs_hardcoded |
| `device_specific` | suboptimal_tile_size, suboptimal_warps, missing_grf_mode, no_swizzling, repack_in_forward, missing_packed_transpose, serialized_n_tiles, autotune_duplicate_params, sigmoid_slow_exp |
| `autotuning` | missing_autotune, suboptimal_autotune_configs, autotune_key_missing |

**Ordering is not arbitrary.** `planner._HARD_DEPENDENCIES` enforces, regardless of what the
LLM returns, e.g.: `algorithmic`/`discovery` before `dtype_fix`/`fusion`; `dtype_fix` before
`fusion`; `memory_access` before `block_pointers`; `fusion`/`block_pointers` before
`device_specific`; `device_specific` before `autotuning`. The logic: fix the math first (it
changes what there is to fuse/tune), then data movement, then hardware-specific knobs, and
autotune **last** so the search space reflects the final code.

### A.3 Stages are gated by DSL

`dsl_registry.DSL_SUPPORTED_STAGES` restricts which stages apply per DSL — the planner can
only pick from the active DSL's set:

| DSL | block_pointers | persistent | autotuning | notes |
|-----|:---:|:---:|:---:|-------|
| triton | ✓ | ✓ | ✓ | reference path, full set |
| gluon | ✗ | ✗ | ✓ | |
| sycl | ✗ | ✗ | ✗ | C++/CUTLASS; no Triton-style autotune/block-ptr |
| cuda | ✗ | ✓ | ✓ | |

So "the same strategy" is not literally the same code path across DSLs; check the registry
before assuming a stage will run.

### A.4 The knowledge base steers both ends

`knowledge/loader.py` loads YAML in priority order `common/ → <dsl>/ → <dsl>/<device>/`.
It feeds the **analyzer** hard constraints (device placement, dtype contracts, `grf_mode`
usage) so it flags real violations, and the **optimizer** before/after pattern pairs and
reference kernels per stage. It is guidance, not law — the compiler/backend is the
authority (see B.4). Disabled by default (`KNOWLEDGE_BASE_ENABLED=false`).

### A.5 Correctness contract and tolerances

The harness executes the input file's `Model(torch.nn.Module)` and compares original vs
optimized outputs. Tolerance precedence: **CLI `--rtol/--atol` > spec variant > env >
defaults (rtol 0.01, atol 1e-5)**. Critically, the optimized `Model` must keep a
**compatible constructor signature** (init args declared in the spec's `inits`) so the
harness can copy identical weights into both models — breaking this is the most common way a
"fast" proposal gets rejected (see B.3).

---

## Part B — Analysis of the concrete example

**Kernel:** `examples/gemm/14_Gemm_Divide_Sum_Scaling.py` — computes
`sum(x @ W.T, dim=1) * scale` as two Triton kernels (`sum_weight_kernel` reduces the weight
to a per-column sum `w_colsum`; `dot_row_kernel` dots each row of `x` against it).
**Shape:** `X:(1024,8192)` fp16, `W:(8192,8192)`, ~137 GFLOP. **Device:** Arc Pro B70.
**Result:** 0.989 ms → 0.726 ms (**1.36x**), 138.9 → 189.3 TFLOPS, correctness PASSED.

### B.1 Analysis output (Part A.2 taxonomy in action)

The analyzer emitted 5 issues with severities — all real and correctly categorized:

| Sev | Issue type | Stage it routes to |
|----|------------|--------------------|
| 5 | `cacheable_intermediate` (`w_colsum` recomputed every forward) | algorithmic |
| 5 | `suboptimal_algorithm` (two kernels + intermediate HBM traffic) | algorithmic |
| 4 | `unfused_kernels` | fusion |
| 3 | `missing_autotune` (hardcoded BLOCK_M/N/K, no warp sweep) | autotuning |
| 2 | `dtype_precision` (fp32 output) | dtype_fix |

### B.2 Plan (Part A.2/A.3 selection + skipping)

Planner produced `algorithmic → dtype_fix → fusion → autotuning` and **skipped**
memory_access, block_pointers, persistent_kernel, device_specific, discovery (no matching
issues). This is the strategy operating as designed: only the stages with evidence run.

### B.3 Per-stage outcomes — where the CoVeR gate mattered

- **algorithmic — rejected by the gate.** The agent cached `w_colsum`; first timing was
  **408 µs (~2.4x!)** but correctness **failed** (`max_diff=4.31`,
  `Parameter count mismatch: original=1, optimized=2` → weights could not be copied). The
  rewrite changed the `Model` signature, violating the A.5 contract. Retries fell back to a
  compatible version at 1.00x, so the stage failed and the original was kept. **The single
  biggest theoretical win was left on the table — on purpose** — because it could not be
  made correct within the contract.
- **dtype_fix / fusion — no-ops.** Re-analysis concluded "No changes needed": the KB dtype
  contract forbids changing the output dtype away from the original fp32, and fusion is moot
  once caching was rejected. Correctness preserved, code unchanged.
- **autotuning — accepted, the actual win.** Added `@triton.autotune` to both kernels
  (108 configs for the sum kernel sweeping BLOCK_M/BLOCK_N × warps × stages; 60 for the dot
  kernel sweeping BLOCK_K × warps × stages), `key=['N','K']` / `key=['M','K']`. First attempt
  followed the KB advice to sweep `grf_mode` inside `triton.Config`, which the installed
  Triton rejected; CoVeR fed the error back and the retry dropped it → **1.57x** (631 µs).

### B.4 The net diff — what Xe-Forge actually changed

The only accepted change is autotuning; the kernel math, dtypes, and `Model` are byte-for-byte
unchanged. Concretely (`outputs/14_gemm_optimized.py`):

1. `@triton.autotune(...)` + config builders on both kernels.
2. Removed hardcoded `BLOCK_M=256, BLOCK_N=128, BLOCK_K=256`.
3. **Static grid → dynamic `META` grid** — a correctness-preserving *companion* change, not
   an independent optimization:
   ```python
   - grid0 = (triton.cdiv(K, BLOCK_N),)                 # fixed BLOCK_N=128
   + def grid_sum_weight(META):
   +     return (triton.cdiv(K, META['BLOCK_N']),)      # adapts to the chosen tile
   ```
   With autotune varying `BLOCK_N`, a fixed grid would under/over-cover the output.

This is the honest picture of what the pipeline did here: **it searched launch
configuration, it did not restructure the algorithm** — because the one restructuring idea it
had was correctly vetoed by the verification gate.

### B.5 Takeaways (example → general)

- **The gate is the product.** Analysis found a genuinely better algorithm (caching), but
  CoVeR's correctness check — not the LLM's confidence — decided what shipped. Trust the
  verified final number (1.36x), not the tempting 2.4x that failed validation.
- **Headroom is bounded by the starting point.** Baseline was already 138 TFLOPS on a ~160
  TFLOPS peak, so autotune-only could yield ~1.4–1.5x, not an order of magnitude. The
  strategy shines on *untuned* kernels, not near-roofline ones.
- **Contract compatibility is a first-class constraint.** Any optimization that changes the
  `Model` signature or output dtype must also update the spec (`inits`, dtype) or it will be
  rejected — the algorithmic stage is the case study.
- **KB is advisory, backend is authority.** The `grf_mode` rejection is the general rule:
  generated Triton must compile under the pinned `pytorch-triton-xpu`; the knowledge base
  suggests, the compiler decides.
