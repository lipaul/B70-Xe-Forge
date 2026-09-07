# Xe-Forge SYCL 端到端优化流程(CUTLASS GEMM tile-tune,Intel Arc Pro B70 实测)

本文用一次**真实运行**演示 Xe-Forge 在 **SYCL DSL** 上的端到端优化流程。与 Triton 路径
(见 [`OPTIMIZATION.zh.md`](OPTIMIZATION.zh.md))不同,SYCL 走的是 **CUTLASS tile 搜索**这条
独立子系统(`--tile-tune`),本文所有数字均来自 B70 上的实测,非虚构。

配套英文 Triton 版流程见 [`WORKFLOW.md`](WORKFLOW.md)。

---

## 1. 为什么 SYCL 用 tile-tune 而不是分阶段流水线

- Xe-Forge 有两条 SYCL 路径:
  1. **分阶段优化流水线**(`--dsl sycl`):对**手写的 CUTLASS C++ kernel** 做
     analyze→plan→逐阶段 CoVeR。但 `dsl_registry` 里 SYCL 只支持
     analysis/algorithmic/dtype_fix/fusion/memory_access/device_specific/discovery
     (**没有 autotuning/block_pointers/persistent**),且仓库内**没有任何 SYCL `.cpp` 示例**。
  2. **tile-tune 子系统**(`--tile-tune` / `--tune-config`):LLM 驱动的 **propose→validate→
     compile→benchmark** 循环,针对 CUTLASS SYCL 的 GEMM/FA/MoE/Grouped-GEMM 搜索**编译期 tile
     形状**。这是本仓库最成熟、可自包含运行的 SYCL 路径——C++ 源码由 Jinja2 模板生成,**无需
     手写输入 kernel**。
- 因此本文演示路径 2(GEMM),它最能干净地展示"优化过程"。

## 2. 环境搭建(一次性)

关键依赖是 **`intel/sycl-tla`**(前身 CUTLASS-SYCL,header-only 的 CUTLASS→Intel GPU 移植)。
Xe-Forge 的 `SyclExecutor` 只借用它的头文件,自身不编译库。

```bash
git clone --depth 1 --branch v0.9.2 --recurse-submodules https://github.com/intel/sycl-tla /home/lm/sycl-tla
source /opt/intel/oneapi/setvars.sh          # 提供 icpx/icx(本机 2026.0)

export SYCL_TLA_DIR=/home/lm/sycl-tla
export AIBENCH_SYCL_TARGET=bmg-g31           # ← B70 的正确 AOT target,见下
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export DSL=sycl DEVICE_TYPE=xpu
# LLM 复用 .env:openai/qwen3.8-flash + LLM_MODEL_TYPE=chat + LLM_DISABLE_THINKING=true
```

Xe-Forge 从 `SYCL_TLA_DIR` 派生 include 路径(`sycl_executor.py:_include_dirs`),GEMM 需要:
`include/`、`tools/util/include`、`examples/common`(`helper.h`、`sycl_common.hpp`)——均已确认存在。

**B70 的坑(重要):** `sycl_executor.py` 的 `_DEVICE_NAME_TO_TARGET` 表**不含 B70**
(torch 报的设备名是 `Intel(R) Graphics [0xe223]`),自动探测会打印
`Unknown XPU device ... skipping AOT target` 并返回空。而 `ai_bench` 的 `SYCLCompiler` 会把
`AIBENCH_SYCL_TARGET` 拼成 `icpx ... -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen
"-device bmg-g31"`。**没有 target 就不启用 SPIRV 扩展、编译失败**,所以必须手动
`export AIBENCH_SYCL_TARGET=bmg-g31`(B70 属 Battlemage **G31** die;实测该值可编可跑)。

## 3. 冒烟测试:先证明工具链能编能跑(不花 LLM)

用模板生成一个已知 good tile `[256,128,64]` 的完整 CUTLASS GEMM 程序,直接编译+运行:

```
generated 8889 bytes of C++ for tile (256, 128, 64)
target_device: bmg-g31 | compiler: icpx
elapsed 38.9s
success: True | correct: True | tflops: 22.773 | ms: 6.035
```

icpx 对着 sycl-tla 编译约 38s,B70 上跑通、`Disposition: Passed`。这一步把"能否编译/运行/
target 是否正确"全部 de-risk,再进入 LLM 搜索。

## 4. tile-tune 端到端(真正的优化过程)

```bash
python -m xe_forge.cli --dsl sycl --tile-tune \
    --m 4096 --gemm-n 4096 --k 4096 --max-rounds 3 --gemm-dtype bf16 \
    --tune-output outputs/gemm_4k.json
```

每轮循环(`tile_search/agent.py:TileTuningAgent.tune`):
**LLM 依据 shape+硬件约束+历史 → 提 3–5 个 tile → DPAS validator 过滤非法 → icpx 编译 →
B70 基准测 TFLOPS → 结果回灌下一轮**。本次 3 轮共测 **14 个 config,全部通过正确性**。

逐轮进展(取自日志):

| 轮 | LLM 学到的东西 | 本轮最优 |
|----|----------------|----------|
| Round 1 | "大方阵 GEMM 偏好大 tile 以提高算术强度" | `[256,256,32]` → **59.59 TFLOPS** |
| Round 2 | 对比历史,试 `wg_K` 更小的变体 | `[256,256,16]` → **113.97 TFLOPS** |
| Round 3 | "`wg_K=16` 是该硬件/形状下的最优 K 维" | 无提升,保持 113.97 |

14 个 config 的完整成绩(TFLOPS 降序):

| rank | wg (M,N,K) | TFLOPS | ms |
|----|------------|-------:|-----:|
| 1 | **[256, 256, 16]** | **113.97** | 1.206 |
| 2 | [128, 256, 16] | 107.21 | 1.282 |
| 3 | [256, 128, 16] | 101.33 | 1.356 |
| 4 | [512, 128, 16] | 72.53 | 1.895 |
| 5 | [256, 256, 32] | 59.59 | 2.306 |
| 6 | [128, 128, 16] | 59.15 | 2.324 |
| 7 | [256, 128, 32] | 41.97 | 3.275 |
| 8 | [512, 128, 32] | 28.49 | 4.825 |
| 9 | [128, 256, 64] | 27.95 | 4.917 |
| 10 | [128, 128, 64] | 27.55 | 4.988 |
| 11 | [256, 128, 64] | 22.78 | 6.033 |
| 12 | [128, 512, 32] | 21.50 | 6.394 |
| 13 | [256, 256, 64] | 16.29 | 8.436 |
| 14 | [512, 256, 32] | 5.60 | 24.534 |

最优解写回 `outputs/gemm_4k.json`,其 `tile_shapes` 字段(`wg=[256,256,16], sg=[8,4,1]`)
可直接喂给 sycl-tla 的 `SYCL_TLA_ADDITIONAL_TILE_SHAPES`。

## 5. 结果与加速

- **默认/朴素 tile `[256,128,64]`:22.78 TFLOPS(6.033 ms)**(即冒烟测试那个)。
- **搜索最优 `[256,256,16]`:113.97 TFLOPS(1.206 ms)**。
- **净加速 ≈ 5.0x**(6.033→1.206 ms),且**正确性全部通过**(每个 config `verify=1`,
  C++ 端 `Disposition: Passed`)。
- 规律非常清晰:**`wg_K=16` ≫ 32 ≫ 64**(同一 256×256 tile:16→114、32→59.6、64→16.3)。
  离群点 `[512,256,32]=5.6 TFLOPS` 是过大 tile 导致占用率/寄存器崩塌的典型反例——搜索把它
  测出来并淘汰了。

## 6. 机制解读:Xe-Forge ↔ sycl-tla 的耦合

- **sycl-tla 是"被编译的对象"**:Xe-Forge 用 Jinja2 生成 CUTLASS C++,`ai_bench.SYCLCompiler`
  调 `icpx -fsycl ... -device <target>` 编成 SPIR-V,在 B70 上跑。Xe-Forge 本身不含 CUTLASS。
- **target/SPIRV 是硬接口**:`-device bmg-g31` 决定启用哪些 Xe2 指令与 SPIRV 扩展
  (`SPV_INTEL_subgroup_matrix_multiply_accumulate` 等)。target 选错 → 编译或运行失败。这正是
  B70 需要手动指定 target 的原因。
- **DPAS 约束是验证器的依据**:`validators/gemm.py` 按 Intel Xe DPAS atom(M=8、N=16、
  K 随 dtype)过滤 LLM 提案(`wg_M%8==0, wg_N%16==0, wg_K%atom_K==0`,SLM 128KB、subgroup≤32),
  非法 tile 根本不进编译。
- **与 Triton 路径的分工**:Triton 路径优化的是"LLM 改写 kernel 源码";SYCL tile-tune 优化的
  是"在固定 CUTLASS 模板下搜索编译期 tile 形状"。两者共享同一 CoVeR 式思想(提出→在真实硬件
  上验证→回灌),但作用对象不同。

## 7. 复现

```bash
# 见 §2 环境;然后:
python -m xe_forge.cli --dsl sycl --tile-tune \
    --m 4096 --gemm-n 4096 --k 4096 --max-rounds 3 --gemm-dtype bf16 \
    --tune-output outputs/gemm_4k.json
# 多形状批量:python -m xe_forge.cli --dsl sycl --tune-config examples/tile_search/tune_gemm_basic.yaml
```

## 8. 注意事项 / 踩过的坑

- **后台运行要用 `setsid`**:tile-tune 每个 CUTLASS 编译约 30–90s、整轮数分钟;若用普通
  `nohup &` 在交互式 shell 里跑,命令超时会被进程组连坐杀掉。`setsid nohup ... & disown`
  让其独立于会话存活。
- **B70 不在自动 target 表**:务必 `export AIBENCH_SYCL_TARGET=bmg-g31`;g31 不行再退 `bmg-g21`
  或 `bmg`。
- **reasoning 模型要关思考**:沿用 `LLM_DISABLE_THINKING=true`,否则大 prompt 下单次提案调用会
  很慢甚至超时。
- **正确性由 C++ runner 自校验**:`build_run_args` 传 `verify=1`,二进制打印 `Disposition`,
  `_parse_raw_output` 解析;`passed=false` 的 config 不会被记为 best。
