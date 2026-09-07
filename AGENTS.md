# AGENTS.md

Xe-Forge: multi-stage LLM-driven optimization of GPU kernels (Triton/Gluon/SYCL/CUDA),
targeting Intel XPU and NVIDIA CUDA. Active development — interfaces are unstable.

## Toolchain

- Python **>=3.11**, package manager is **uv** (not pip/venv). Source is a `src/` layout:
  the importable package is `xe_forge` at `src/xe_forge/`.
- Install one backend only — `intel` and `nvidia` extras are mutually exclusive
  (declared in `[tool.uv] conflicts`):
  - `uv sync --extra intel` (XPU) or `uv sync --extra nvidia` (CUDA)
  - add dev tools with `uv sync --dev`
- `ai-bench` is a pinned git dependency; network access is required to resolve it.

## Verify before pushing

- Format is the **only** thing CI enforces: `.github/workflows/ruff.yaml` runs
  `ruff format --check .` — it does **not** run `ruff check`. So passing CI ≠ lint-clean.
- Run the full local gate (pre-commit does check+fix then format):
  `uv run pre-commit run --all-files`
- Unit tests need **no GPU**: `uv run pytest`. Single test:
  `uv run pytest tests/test_spec_loader.py::<name>`. Tests live in `tests/` and stub the
  LLM/executor — do not add GPU or live-API dependencies there.
- `runners/test_kb_examples.py` is NOT a pytest suite; it requires real XPU hardware
  (`torch.xpu.is_available()`) and is run manually: `python runners/test_kb_examples.py [id]`.

## Ruff scope

`ruff` excludes `test_kernels/`, `knowledge_base/**/examples`, `outputs/kernels`,
`examples/`, `templates/`. Line length 100, double quotes, target py311. Don't be surprised
when generated/example kernels aren't formatted.

## Entry points & config

- Two console scripts: `xe-forge` (`xe_forge.cli:main`) and `xe-forge-skill`
  (`xe_forge.skills:main`). Also runnable as `python -m xe_forge.cli`.
- **All runtime config is env-driven** via a `.env` file (copy `.env.example`). Requires
  `OPENAI_API_KEY`, `OPENAI_API_BASE`, `LLM_MODEL` (litellm id, e.g. `openai/...`).
  Config is assembled in `src/xe_forge/config.py` (`ConfigManager`); CLI flags override env.
- Knowledge base is **off by default** (`KNOWLEDGE_BASE_ENABLED=false`); enable with
  `KNOWLEDGE_BASE_ENABLED=true` and it loads from `KNOWLEDGE_DIR` (default `./knowledge_base`).

## Architecture (read the wiring, not just filenames)

- `pipeline.py` (`XeForgePipeline.optimize`) is the orchestrator; `cli.py` only loads config,
  resolves the spec/variant, and dispatches to an engine.
- `engines/` selects the path: `dspy` (default, automated) vs `claude` (generates a workspace).
  Factory: `engines/__init__.py:create_engine`.
- `agents/` holds the LLM agents (analyzer, optimizer, coordinator, CoVeR/ReAct). CoVeR =
  compile→run→compare→retry loop, the core correctness/speedup gate.
- DSL-awareness is central and end-to-end. `models.py` defines the `DSL` enum;
  `dsl_registry.py` maps each DSL to its supported `OptimizationStage`s; `core/` branches on
  DSL (e.g. `executor.py` = KernelBench for Python DSLs, `sycl_executor.py` for SYCL).
  When changing stages/behavior, check `dsl_registry.py` and the DSL branches, not just one file.
- `core/tile_search/` is a separate subsystem (SYCL/CUTLASS GEMM+FA tile tuning), reached via
  `--tile-tune` / `--tune-config`, not the optimization pipeline.
- Knowledge base loads in priority order `common/ → <dsl>/ → <dsl>/<device>/`
  (`knowledge/loader.py`). Add patterns as YAML under the right dsl/device subdir.

## Kernel input conventions

- Every input kernel file must define a `Model(torch.nn.Module)` with `forward()` doing all
  Triton launches; optional `get_example_inputs()` for non-random inputs.
- Spec is a YAML file (`inputs`, optional `inits`, `bench-gpu`/`bench-gpu-N`/`ci` variants with
  `dims` and a `flop` string expression). PyTorch reference is auto-discovered as
  `<kernel>_pytorch.py` next to the kernel.
- Tolerance precedence: CLI `--rtol/--atol` > spec variant > env > defaults (rtol 0.01, atol 1e-5).

## Docs to consult for specific tasks

`docs/DSL.md` (adding a new DSL — touches models, registry, executor, prompts, KB),
`docs/TILE.md` (tile tuning), `docs/VTUNE.md` (profiling), `docs/EXAMPLES.md` (sample kernels).
`scripts/` are standalone PEP 723 scripts run via `uv run scripts/...` and do **not** import
`xe_forge` (roofline plotting from result CSVs).
