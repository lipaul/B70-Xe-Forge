# Xe-Forge 优化策略 —— 通用框架与一个实例分析

本文分两部分:**A** 描述通用化优化策略(依据代码核实),**B** 分析在 Intel Arc Pro B70
上对 `14_Gemm_Divide_Sum_Scaling` 的一次真实运行。配套的逐步日志叙述见
[`WORKFLOW.md`](WORKFLOW.md);英文原版见 [`OPTIMIZATION.md`](OPTIMIZATION.md)。

---

## A 部分 —— 通用化优化策略

### A.1 控制循环

Xe-Forge 不是基于规则的 pass 管理器,而是**由 LLM 提出代码、在真实硬件上验证**。
`pipeline.py:XeForgePipeline.optimize` 的流程为:

```
测量 baseline
  → ANALYSIS   (LLM 列出带严重度的 issues)
  → PLANNING   (把 issues 映射成一组有序、且被裁剪过的 stage)
  → 对每个激活的 stage:
        OPTIMIZE (LLM 针对该 stage 的 issues 重写 kernel)
        CoVeR    (编译 → 运行 → 比对 → 重试)
        RE-ANALYZE (后续 stage 看到的是更新后的代码)
  → 与 baseline 做完整 warmup 的最终测量
```

核心是 **CoVeR 闸门**(compile→run→compare→retry,`agents/cover.py`):每个被提出的
kernel 必须 (a) 在容差内与 baseline 输出一致,且 (b) 比当前最优更快;否则丢弃并把错误
回灌给 agent 重试(最多 `AGENT_MAX_ITERATIONS` 次)。最坏情况该 stage 失败并**保留原始
代码**——流水线默认安全,绝不会交付错误或更慢的 kernel。

### A.2 stage 目录(“策略空间”)

stage 是固定的优化词汇表。每个 stage 背后都有一族分析器可发出的 `IssueType`
(`models.py`),因此分析与优化共享同一套分类。规范顺序取自
`planner.DEFAULT_STAGE_ORDER`(ANALYSIS 永远最先执行):

| stage | 负责修复的 issue 类型 |
|-------|---------------------|
| `algorithmic` | redundant_computation, suboptimal_algorithm, associativity_reorder, common_subexpression, algebraic_simplification, **cacheable_intermediate**, loop_invariant_code, unnecessary_materialization, gemm_simplification, reduction_tree_suboptimal |
| `discovery` | open_ended(新颖、高价值想法,必须给出 before/after 草图) |
| `dtype_fix` | dtype_float64, dtype_precision, dtype_input_conversion |
| `fusion` | unfused_kernels, unfused_elementwise, unfused_reduction, fusion_register_pressure, fusion_replaces_vendor, fusion_noop |
| `memory_access` | manual_pointer_arithmetic, missing_boundary_check, transpose_in_loop, missing_tma, uncoalesced_access, device_host_sync, non_contiguous_input, cache_eviction_risk, long_liveness, high_register_pressure |
| `block_pointers` | missing_block_pointers, block_ptr_boundary_wrong, block_ptr_multiple_of_misuse |
| `persistent_kernel` | missing_persistent, persistent_num_progs_hardcoded |
| `device_specific` | suboptimal_tile_size, suboptimal_warps, missing_grf_mode, no_swizzling, repack_in_forward, missing_packed_transpose, serialized_n_tiles, autotune_duplicate_params, sigmoid_slow_exp |
| `autotuning` | missing_autotune, suboptimal_autotune_configs, autotune_key_missing |

**顺序并非随意。** `planner._HARD_DEPENDENCIES` 无论 LLM 返回什么都会强制,例如:
`algorithmic`/`discovery` 先于 `dtype_fix`/`fusion`;`dtype_fix` 先于 `fusion`;
`memory_access` 先于 `block_pointers`;`fusion`/`block_pointers` 先于 `device_specific`;
`device_specific` 先于 `autotuning`。其逻辑是:先修数学(它决定了后续可融合/可调优的对象),
再改数据搬运,再上硬件相关旋钮,最后做 autotune,让搜索空间反映最终代码。

### A.3 stage 按 DSL 裁剪

`dsl_registry.DSL_SUPPORTED_STAGES` 限制每个 DSL 可用的 stage——planner 只能从当前 DSL 的
集合里挑选。因此“同一套策略”在不同 DSL 上并非同一条代码路径:

| DSL | block_pointers | persistent | autotuning | 备注 |
|-----|:---:|:---:|:---:|------|
| triton | ✓ | ✓ | ✓ | 参考路径,集合最全 |
| gluon | ✗ | ✗ | ✓ | |
| sycl | ✗ | ✗ | ✗ | C++/CUTLASS;无 Triton 式 autotune/block-ptr |
| cuda | ✗ | ✓ | ✓ | |

在假设某 stage 会执行前,先查这个注册表。

### A.4 知识库从两端施加影响

`knowledge/loader.py` 按优先级 `common/ → <dsl>/ → <dsl>/<device>/` 加载 YAML。它给
**分析器**提供硬约束(device 放置、dtype 契约、`grf_mode` 用法),使其能识别真实违规;给
**优化器**按 stage 提供 before/after 模式对与参考 kernel。它是“建议”而非“法律”——编译器/
后端才是权威(见 B.4)。默认关闭(`KNOWLEDGE_BASE_ENABLED=false`)。

### A.5 正确性契约与容差

harness 会执行输入文件里的 `Model(torch.nn.Module)`,并比对原始与优化后的输出。容差优先级:
**CLI `--rtol/--atol` > spec 变体 > 环境变量 > 默认(rtol 0.01,atol 1e-5)**。关键约束:
优化后的 `Model` 必须保持**兼容的构造函数签名**(init 参数需在 spec 的 `inits` 中声明),
这样 harness 才能把相同权重拷进两个模型——破坏这一点,是“很快”的提案被拒的最常见原因
(见 B.3)。

---

## B 部分 —— 具体实例分析

**kernel:** `examples/gemm/14_Gemm_Divide_Sum_Scaling.py` —— 用两个 Triton kernel 计算
`sum(x @ W.T, dim=1) * scale`(`sum_weight_kernel` 把权重按列归约成 `w_colsum`;
`dot_row_kernel` 把 `x` 的每一行与之点积)。
**形状:** `X:(1024,8192)` fp16,`W:(8192,8192)`,约 137 GFLOP。**设备:** Arc Pro B70。
**结果:** 0.989 ms → 0.726 ms(**1.36x**),138.9 → 189.3 TFLOPS,正确性 PASSED。

### B.1 分析输出(A.2 分类法落地)

分析器给出 5 个带严重度的 issue,全部真实且归类正确:

| 严重度 | issue 类型 | 路由到的 stage |
|----|------------|--------------------|
| 5 | `cacheable_intermediate`(`w_colsum` 每次 forward 都重算) | algorithmic |
| 5 | `suboptimal_algorithm`(两 kernel + 中间 HBM 读写) | algorithmic |
| 4 | `unfused_kernels` | fusion |
| 3 | `missing_autotune`(BLOCK_M/N/K 写死,无 warp 扫描) | autotuning |
| 2 | `dtype_precision`(fp32 输出) | dtype_fix |

### B.2 规划(A.2/A.3 的选择与跳过)

planner 产出 `algorithmic → dtype_fix → fusion → autotuning`,并**跳过** memory_access、
block_pointers、persistent_kernel、device_specific、discovery(无匹配 issue)。这正是策略按
设计工作:只有拿到证据的 stage 才运行。

### B.3 逐阶段结果 —— CoVeR 闸门起作用之处

- **algorithmic —— 被闸门否决。** agent 缓存了 `w_colsum`,首次计时 **408 µs(约 2.4x!)**,
  但正确性**失败**(`max_diff=4.31`,`Parameter count mismatch: original=1, optimized=2` →
  权重无法拷贝)。该重写改动了 `Model` 签名,违反 A.5 的契约。重试退回到签名兼容版本为
  1.00x,于是该 stage 判失败、保留原始代码。**理论上最大的那一步收益被有意留在了门外**——
  因为它无法在契约内做到正确。
- **dtype_fix / fusion —— 空操作。** 重新分析后判定“无需改动”:KB 的 dtype 契约禁止把输出
  dtype 从原始 fp32 改掉,且缓存被拒后 fusion 也无意义。正确性保持,代码不变。
- **autotuning —— 被接受,真正的收益。** 给两个 kernel 加了 `@triton.autotune`(sum kernel
  扫 BLOCK_M/BLOCK_N × warps × stages 共 108 个 config;dot kernel 扫 BLOCK_K × warps ×
  stages 共 60 个),`key=['N','K']` / `key=['M','K']`。首次尝试按 KB 建议在 `triton.Config`
  里扫 `grf_mode`,被所装 Triton 拒绝;CoVeR 回灌错误后重试去掉它 → **1.57x**(631 µs)。

### B.4 净 diff —— Xe-Forge 实际改了什么

唯一被接受的改动是 autotuning;kernel 数学、dtype、`Model` 逐字未变。具体
(`outputs/14_gemm_optimized.py`):

1. 给两个 kernel 加 `@triton.autotune(...)` 及 config 构造函数。
2. 删除写死的 `BLOCK_M=256, BLOCK_N=128, BLOCK_K=256`。
3. **静态 grid → 动态 `META` grid** —— 这是为保正确性的*配套*改动,而非独立优化:
   ```python
   - grid0 = (triton.cdiv(K, BLOCK_N),)                 # 固定 BLOCK_N=128
   + def grid_sum_weight(META):
   +     return (triton.cdiv(K, META['BLOCK_N']),)      # 随被选中的 tile 变化
   ```
   autotune 会改变 `BLOCK_N`,固定 grid 会漏算/越界。

这就是本次流水线做了什么的诚实画像:**它搜索了 launch 配置,而没有重构算法**——因为它唯一
的重构想法被验证闸门正确地否决了。

### B.5 结论(由实例回到通用)

- **闸门才是产品。** 分析找到了真正更优的算法(缓存),但决定交付什么的是 CoVeR 的正确性
  校验——而非 LLM 的自信。请信任经过验证的最终数字(1.36x),而不是那个校验失败的诱人 2.4x。
- **收益受起点限制。** baseline 已在约 160 TFLOPS 峰值上跑到 138 TFLOPS,所以仅 autotune 只能
  带来约 1.4–1.5x,而非一个数量级。该策略在*未调优*的 kernel 上最出彩,而非近 roofline 者。
- **契约兼容是一等约束。** 任何改动 `Model` 签名或输出 dtype 的优化,必须同步更新 spec
  (`inits`、dtype),否则会被拒——algorithmic 阶段就是活教材。
- **KB 是建议,后端是权威。** `grf_mode` 被拒即是通则:生成的 Triton 必须在所钉的
  `pytorch-triton-xpu` 下能编译;知识库负责提示,编译器说了算。
