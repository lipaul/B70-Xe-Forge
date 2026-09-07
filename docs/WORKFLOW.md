# Optimization Workflow (worked example on Intel Arc Pro B70)

This document walks through one **real** Xe-Forge run end-to-end and uses it to answer
two questions: *how does the pipeline actually behave*, and *when is this kind of
LLM-driven optimization worth running*. Everything below is taken from an actual run on
an Intel Arc Pro B70 (`0xe223`), not a hypothetical.

## The run we are documenting

| Item | Value |
|------|-------|
| Kernel | `examples/gemm/14_Gemm_Divide_Sum_Scaling.py` (computes `sum(x @ W.T, dim=1) * scale`) |
| Variant | `bench-gpu` → `X:(1024, 8192)` fp16, `W:(8192, 8192)`, ~137 GFLOP |
| Device | Intel Arc Pro B70, `torch 2.11+xpu`, `pytorch-triton-xpu` (triton 3.7) |
| Engine | `dspy` (default), CoVeR agent, `AGENT_MAX_ITERATIONS=3` |
| LLM | `qwen3.8-flash` via an OpenAI-compatible endpoint (`/v1/chat/completions`) |
| Knowledge base | enabled (`KNOWLEDGE_BASE_ENABLED=true`) |

Headline result: **0.989 ms → 0.726 ms (1.36x), 138.9 → 189.3 TFLOPS, correctness PASSED.**

## Stage-by-stage, what actually happened

### 0. Baseline
The executor imports `Model`, moves it to XPU, and times the original:
`Original: 138.92 TFLOPS, 0.99 ms`. This number is the "true baseline" every later
stage is compared against.

### 1. ANALYSIS — LLM reads the kernel, emits scored issues
The analyzer (with KB constraints injected) returned 5 issues, each with a severity 1–5:

- `[5] cacheable_intermediate` — `w_colsum` (column-sum of the frozen weight) is
  recomputed by a kernel on **every** forward; it is constant and should be cached.
- `[5] suboptimal_algorithm` — two-kernel split + an intermediate HBM write/read for `w_colsum`.
- `[4] unfused_kernels` — `sum_weight_kernel` + `dot_row_kernel` could collapse to one if `w_colsum` is cached.
- `[3] missing_autotune` — hardcoded `BLOCK_M/N/K`, no `num_warps` sweep.
- `[2] dtype_precision` — output allocated fp32.

These are real, specific, and correctly prioritized — this is the part that is hard to
get from a static linter.

### 2. PLANNING — map issues to stages, skip the rest
The planner produced an ordered plan and **skipped stages with no matching issue**:

```
+ algorithmic  [#1]: suboptimal_algorithm, cacheable_intermediate
+ dtype_fix    [#2]: dtype_precision
+ fusion       [#3]: unfused_kernels
+ autotuning   [#4]: missing_autotune
- memory_access / block_pointers / persistent_kernel / device_specific / discovery: skipped
```

### 3. Per-stage optimization with CoVeR (compile → run → compare → retry)
Each stage's optimizer proposes a kernel, then CoVeR compiles it, runs it on the GPU,
and compares output + timing against the baseline. A failed check feeds the error back
and the agent retries (up to `AGENT_MAX_ITERATIONS`).

**algorithmic — proposed a great idea, CoVeR rejected the implementation.**
The agent cached `w_colsum` and the first timing was `408 µs` (~2.4x!). But correctness
**failed**: `max_diff=4.31`, with `Parameter count mismatch: original=1, optimized=2` →
`Could not copy weights - using seed-based initialization`. The rewrite changed the
`Model` constructor signature, so the harness could not copy identical weights into both
models and compared different random weights. CoVeR correctly refused it; later retries
fell back to a signature-compatible version at `1.00x`, so the stage was marked
`failed: no valid result in budget` and the original code was kept.

> Lesson: the *idea* (cache the weight-derived constant) is valid and fast; the
> *implementation* broke the `Model` contract (see "Rules" in the README — init args must
> stay compatible and be declared in the spec's `inits`). CoVeR is what turns a plausible
> but wrong kernel into a no-op instead of a silent correctness bug.

**dtype_fix / fusion — no-ops.** After re-analysis the agent decided "No changes needed"
(dtype must match the original fp32 output per the KB dtype contract; fusion is moot once
caching is rejected). Correctness preserved, code unchanged.

**autotuning — the win, after one CoVeR retry.** The agent added `@triton.autotune` to
both kernels, sweeping `BLOCK_M/BLOCK_N` (sum kernel) and `BLOCK_K` (dot kernel) with
`num_warps`/`num_stages`, and switched to dynamic `grid=lambda META: ...`. Its first
attempt followed the KB advice to sweep `grf_mode` inside `triton.Config(...)`, which the
installed Triton rejected; CoVeR fed the error back and the retry dropped `grf_mode`.
Result: `631 µs` → **1.57x**, kept as best (a further iteration was 1.42x, so it stopped).

### 4. Final measurement
The full optimized kernel is re-benchmarked with warmup: **1.36x, 189.3 TFLOPS**. An
independent re-check with the standalone skill agreed:

```
xe-forge-skill benchmark <orig> outputs/14_gemm_optimized.py --spec <yaml> --variant bench-gpu --triton-baseline
→ Correctness: PASSED, 989.25 µs → 647.57 µs, speedup=1.53x
```

Note the spread (1.36x final vs 1.53–1.57x stage/standalone): autotune selection and
GPU timing vary run-to-run. **Trust the verified final number, not the best single stage.**

## What actually changed in the code

The only accepted change was autotuning. Before: fixed `BLOCK_M=256, BLOCK_N=128,
BLOCK_K=256`, `num_warps` hardcoded, static grid. After (`outputs/14_gemm_optimized.py`):

```python
@triton.autotune(configs=_sum_weight_configs(), key=[...])   # sweeps BLOCK_M/BLOCK_N × warps × stages
def sum_weight_kernel(...): ...
@triton.autotune(configs=_dot_row_configs(), key=[...])       # sweeps BLOCK_K × warps × stages
def dot_row_kernel(...): ...
def grid_sum_weight(META): return (triton.cdiv(K, META['BLOCK_N']),)   # grid adapts to chosen tile
```

## When is this workflow worth running?

**Good fit — run it when:**
- The kernel is **functionally correct but untuned** (hand-written or LLM-generated
  Triton/Gluon), i.e. there is headroom above the current tile/warp/dtype choices.
- The inefficiency is **pattern-shaped** — dtype, fusion, autotune, caching, coalescing,
  block-pointers — because those are exactly the stages the pipeline encodes.
- You have a **reference or a stable input contract** so CoVeR can verify correctness
  (a spec + a `Model` whose signature you will not break).
- Shapes are **large enough that tuning matters** (here 8192³-class GEMM). The gains come
  from search, which only pays off when there is room to search.
- You want a **safe, verified** improvement: CoVeR guarantees you never ship a kernel that
  is wrong or slower than baseline (worst case it returns the original).

**Poor fit — skip or temper expectations when:**
- The kernel is **already near roofline**. We started at 138 TFLOPS on a ~160 TFLOPS peak,
  so even a good autotune sweep only yielded 1.36x. Near the ceiling, LLM search has
  little to grab.
- **Shapes are tiny / launch-bound**: measurement noise (we saw 408 µs vs 1000 µs swings)
  and per-launch overhead dominate; speedup numbers become unreliable.
- The right fix **changes the `Model` contract** (new init args, different output dtype).
  CoVeR will reject it unless you also update the spec `inits` / dtype — the algorithmic
  stage above is the cautionary example.
- There is **no correctness reference** and you run `--no-correctness`: you then get speed
  without a safety net, which defeats the main value of the loop.
- You are tuning **CUTLASS SYCL GEMM/FA tiles** — that is a different subsystem
  (`--tile-tune` / `--tune-config`, see `docs/TILE.md`), not this pipeline.

**Cost/latency note:** each stage is one or more LLM calls plus on-GPU compile+run. With a
reasoning model on big prompts this is minutes per call; for this run we disabled thinking
(`LLM_DISABLE_THINKING=true`) to keep the whole 4-stage pipeline to ~15 min. Budget time
accordingly, and use `--stages` to run only the subset you care about.

## Relationship to `intel/intel-xpu-backend-for-triton`

Xe-Forge does **not** replace Intel's Triton backend — it sits on top of it and depends on it.

- **It is the compiler.** The `intel` extra installs `pytorch-triton-xpu`
  (`pyproject.toml`), which *is* the Triton distribution built by
  `intel-xpu-backend-for-triton`. Xe-Forge generates Triton Python; that backend lowers it
  to Xe GPU code (DPAS, `grf_mode`, etc.) and runs it on the GPU. No backend, nothing to
  benchmark.
- **It is the source of the example kernels.** `docs/EXAMPLES.md` and the README state the
  curated examples come from KernelBench L2 **and**
  `intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark`. The `examples/` tree
  is a categorized view of those.
- **It defines the tuning knobs the KB teaches.** The XPU knowledge base
  (`knowledge_base/triton/xpu/*.yaml`) encodes that backend's specific levers —
  `grf_mode`, 256×256 tiles, `GROUP_SIZE_M` swizzling, `num_warps=32`. The analyzer/optimizer
  are pointed at these because they are what the backend actually responds to.
- **The friction we hit is the interface between the two.** The KB told the agent to sweep
  `grf_mode` inside `triton.Config`; the installed Triton from that backend rejected it, and
  CoVeR corrected the code. That is the general pattern: **the KB is guidance, the
  backend/compiler is the authority** — anything the generated kernel does must compile
  under the pinned `pytorch-triton-xpu`.

**Division of labor, in one line:** `intel-xpu-backend-for-triton` = the Triton *compiler*
for Intel GPUs plus a set of *hand-tuned* reference kernels; Xe-Forge = an LLM layer that
*automatically* optimizes arbitrary Triton kernels *for* that backend, verifying every
proposal on real hardware. Complementary, not competing: Xe-Forge's output is only as good
as, and only runs because of, the backend underneath it.

## Reproduce this run

```bash
uv sync --extra intel --dev
cp .env.example .env   # then set the four lines below

# .env (OpenAI-compatible reasoning endpoint, e.g. b.ai):
#   OPENAI_API_BASE=https://api.b.ai/v1
#   OPENAI_API_KEY=...
#   LLM_MODEL=openai/qwen3.8-flash
#   LLM_MODEL_TYPE=chat            # endpoint has no /v1/responses
#   LLM_DISABLE_THINKING=true      # keep big-prompt calls under the client timeout
#   KNOWLEDGE_BASE_ENABLED=true

uv run xe-forge -i examples/gemm/14_Gemm_Divide_Sum_Scaling.py \
    -s examples/gemm/14_Gemm_Divide_Sum_Scaling.yaml \
    -o outputs/14_gemm_optimized.py --variant bench-gpu

# independent re-check:
uv run xe-forge-skill benchmark examples/gemm/14_Gemm_Divide_Sum_Scaling.py \
    outputs/14_gemm_optimized.py -s examples/gemm/14_Gemm_Divide_Sum_Scaling.yaml \
    --variant bench-gpu --triton-baseline
```

`LLM_MODEL_TYPE` and `LLM_DISABLE_THINKING` are env-gated additions made to
`cli.py`/`pipeline.py` so non-OpenAI reasoning endpoints work; defaults preserve the
original `responses` behavior.
