# Xe-Forge SYCL 复杂示例:Flash Attention V2 tile-tune(Intel Arc Pro B70 实测)

本文是 [`SYCL_WORKFLOW.zh.md`](SYCL_WORKFLOW.zh.md)(GEMM)的进阶篇,演示一个**明显更复杂**
的算子:**Flash Attention V2(prefill)**。所有数字来自 B70 真实运行。

相比 GEMM,FA 的复杂度体现在:
- 每个注意力头内部是**两次 GEMM + 在线 softmax**(QKᵀ → softmax → PV),tile 参数从 3 个
  (M,N,K) 变成 **7 个**(`qk_m, qk_n, qk_k, pv_n, pv_k, sg_q, pipeline_stages`);
- 问题维度从 (M,N,K) 变成 `(head_dim, batch, num_heads_q, num_heads_kv, seq_qo, seq_kv)`;
- 还有 `causal / prefill|decode / persistent` 三个模式开关。

> 说明:仓库**不支持 NATTEN**(邻域注意力)——`tile_search` 只有 GEMM/FA/MoE/Grouped-GEMM
> 四种策略,`KernelType` 枚举里也没有 NATTEN。因此"更复杂的注意力示例"这里选 FA。

---

## 1. 发现并修复了两个真实 bug(FA 模板 ↔ sycl-tla v0.9.2 版本漂移)

冒烟测试第一次编译 FA 就失败,暴露出 Xe-Forge 的 `fa_v2.cpp.j2` 模板是针对**更老版本**的
sycl-tla 写的,与仓库文档要求 clone 的 **v0.9.2** 不兼容。两处修复(`src/xe_forge/core/`
`tile_search/templates/fa_v2.cpp.j2`):

**Bug 1 — `FMHAConfigGenWithTileShape` 模板参数错位。** v0.9.2 的签名在 4 个 layout 之后要求
`ElementScale` + **6 个 bool**(`Causal, VarLen, CachedKV, PagedKV, Persistent, BlockScale`),
而模板只给了 **5 个 bool、且缺 `ElementScale`**,导致其后所有实参整体错位、模板实例化失败,
表现为 `use of undeclared identifier 'FMHAKernel'` 的连锁报错。修复:

```diff
-    <% causal %>, false, false, false, <% persistent %>,
+    float, <% causal %>, false, false, false, <% persistent %>, false,
```

**Bug 2 — 非 persistent 的 `KernelArguments` 初始化少字段。** v0.9.2 的 `KernelArguments` 在
`dO` 之后、`K_cache` 之前,插入了 `scaleQ, dScaleQ, scaleK, dScaleK, scaleV, dScaleV,
scale_k, scale_v, scale_q, group_size` 共 10 个字段(均有默认值)。模板直接把
`block_K_cache.get()`(bf16*)喂给了 `scaleQ`(float*),报
`cannot initialize ... 'const float *' with ... 'bfloat16_t *'`。修复(用 Jinja 条件只对
非 persistent 路径补齐,保持 persistent 的短初始化不变):

```diff
          block_V.get(), stride_V, block_O.get(), stride_O,
+         <%% if persistent != "true" %%>
+         nullptr, {}, nullptr, {}, nullptr, {},
+         1.f, 1.f, 1.f, 32,
+         <%% endif %%>
          block_K_cache.get(), stride_K_cache,
```

修复后两种模式都能正确渲染出 **23 个** config-gen 实参(与 v0.9.2 签名一致)。

## 2. 冒烟测试(证明工具链能编能跑,且结果正确)

用 `KNOWN_FA_CONFIGS[128]` 的已知 good tile(`qk_m=256, qk_n=32, qk_k=32, pv_n=32, pv_k=32,
sg_q=16`)直接编译+运行,`verify=1`:

```
validate: True
target: bmg-g31 | includes: 6
elapsed 41.2s
success: True | correct: True | tflops: 39.747 | ms: 6.9157
```

`Disposition: Passed` → FA 的 C++ 端 host 参考校验通过。注意 FA 需要额外 include 目录
(`applications`、`examples/06_bmg_flash_attention`、`benchmarks/flash_attention`),
`_include_dirs` 已按 `KernelType.FA` 自动加上。

## 3. tile-tune 端到端

配置 `examples/tile_search/tune_fa2_llama3_b70.yaml`(Llama-3 8B prefill,head_dim=128,
32 Q 头 / 8 KV 头,seq=4096,bf16,非 causal、非 persistent,3 轮):

```bash
python -m xe_forge.cli --dsl sycl --tune-config examples/tile_search/tune_fa2_llama3_b70.yaml
```

循环同 GEMM:**LLM 依据 head_dim/seq/heads + DPAS 约束 + 历史 → 提 3–5 个 FA tile →
`validate_fa_tile` 过滤 → icpx 编译 → B70 测 TFLOPS → 回灌**。共测 **12 个 config**。

逐轮进展:

| 轮 | 事件 | 当前最优 |
|----|------|----------|
| Seed | 注入 `KNOWN_FA_CONFIGS[128]` | `[256,32,32] sg16` → **39.67 TFLOPS** |
| Round 1 | 提案未超过 seed | 39.67 |
| Round 2 | 发现 `qk_k=16` 大幅更优 | `[256,32,16] sg16` → **90.14 TFLOPS** |
| Round 3 | 无提升 | 90.14 |

12 个 config 成绩(TFLOPS 降序;`wg=(qk_m,qk_n,qk_k)`,`sg_q`=子组数,`pv=(pv_n,pv_k)`):

| rank | wg (qk_m,qk_n,qk_k) | sg_q | pv_n,pv_k | TFLOPS | ms |
|----|---------------------|-----:|-----------|-------:|-----:|
| 1 | **[256, 32, 16]** | 16 | 32,32 | **90.14** | 3.050 |
| 2 | [128, 32, 16] | 8 | 32,32 | 80.27 | 3.424 |
| 3 | [256, 32, 16] | 32 | 32,32 | 46.44 | 5.919 |
| 4 | [256, 32, 32] | 32 | 32,32 | 46.24 | 5.944 |
| 5 | [256, 32, 32] | 16 | 32,32 | 39.67 | 6.929 |
| 6 | [256, 64, 32] | 16 | 32,64 | 13.95 | 19.698 |
| 7 | [256, 32, 64] | 16 | 32,32 | 12.72 | 21.619 |
| 8 | [128, 64, 32] | 8 | 32,64 | 8.50 | 32.351 |
| 9 | [128, 32, 64] | 8 | 32,32 | 7.96 | 34.544 |
| 10 | [256, 32, 32] | 8 | 32,32 | 7.80 | 35.245 |
| 11 | [128, 32, 32] | 4 | 32,32 | 5.55 | 49.540 |
| 12 | [256, 32, **8**] | 16 | 32,32 | — | — | ← 被 validator 拒绝:`qk_k (8) should be 16,32,64` |

## 4. 结果与规律

- **seed `[256,32,32] sg16`:39.67 TFLOPS(6.929 ms)** → **最优 `[256,32,16] sg16`:
  90.14 TFLOPS(3.050 ms)**,**净加速 ≈ 2.27x**,11/12 通过、1 个被约束验证器提前拦截。
- **`qk_k`(QK matmul 的 head-dim tile)是主导变量**:`16 ≫ 32 ≫ 64`
  (同 `[256,32,*] sg16`:16→90.1、32→39.7、64→12.7)——与 GEMM 篇里 `wg_K=16` 最优的
  规律**惊人一致**,说明 B70 上"K 方向小 tile + 更多次迭代"更契合 DPAS 流水。
- **`sg_q` 与 `qk_k` 存在耦合**:`qk_k=32` 时 `sg_q=32`(46.2)> `16`(39.7)> `8`(7.8);
  而 `qk_k=16` 时 `sg_q=16`(90.1)> `32`(46.4)。搜索把这两个维度一起调对了。
- **validator 有效**:rank 12 的 `qk_k=8` 违反 DPAS atom 约束,未进编译即被过滤,省掉一次
  无谓的 40s 编译。

## 5. 复现

```bash
# 环境同 SYCL_WORKFLOW.zh.md §2(bmg-g31 target 等)
python -m xe_forge.cli --dsl sycl --tune-config examples/tile_search/tune_fa2_llama3_b70.yaml
# 结果写入 outputs/fa2_llama3.json,其 tile_shapes 可直接喂 sycl-tla 的
# SYCL_TLA_ADDITIONAL_TILE_SHAPES
```

## 6. 注意事项 / 已知限制

- **FA tile-tune 默认 `verify=0`**:`FAStrategy.build_run_args` 硬编码 `verify:0`(host 参考
  是 O(seq²) CPU 计算,seq=4096 太慢),故搜索阶段**不逐 config 校验正确性**;正确性由本文
  §2 的冒烟测试单独证明。GEMM 路径则是 `verify=1`。
- **persistent prefill 未跑通**:v0.9.2 的 persistent FMHA 基准全是 decode 形态;对 prefill
  的已知 tile,persistent 分支 `can_implement` 会拒绝。本文走的是默认非 persistent 路径。
- **模板已修但仅覆盖 bf16 非 block-scale**:scale 字段以 `nullptr/1.f/32` 占位;fp8/mxfp 的
  block-scale FA 需要真正传入 scale 张量,当前模板不支持。
- **后台运行仍需 `setsid`**:FA 单个编译约 40s、整轮约 10 分钟,须脱离会话运行(见 GEMM 篇 §8)。
