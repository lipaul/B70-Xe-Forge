// NATTEN 1D via the (patched) sycl-tla FMHA DPAS kernel.
// Instantiates FMHAConfigGenWithTileShape (non-causal, non-persistent) and sets
// KernelArguments::natten_window = w, so the kernel computes neighborhood
// attention (band [i-w, i+w]) with the K-loop restricted to the band.
// Verifies against a simple fp32 host band-attention reference.

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "fmha_configuration.hpp"
#include "sycl_common.hpp"

#include <cute/tensor.hpp>
#include <random>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace cute;
using namespace cutlass::flash_attention;

using FAConfig = FMHAConfigGenWithTileShape<
    FMHAMode::Prefill,
    bfloat16_t, bfloat16_t, bfloat16_t, float,
    cutlass::layout::RowMajor, cutlass::layout::ColumnMajor,
    cutlass::layout::RowMajor, cutlass::layout::RowMajor,
    float, false, false, false, false, false, false,
    256, 32, 32,
    16, 32,
    32, 128
>::type;

using FMHAKernel = typename FAConfig::FMHAKernel;
using ProblemShapeType = typename FAConfig::ProblemShapeType;
using ElementQ = typename FMHAKernel::ElementQ;
using ElementK = typename FMHAKernel::ElementK;
using ElementV = typename FMHAKernel::ElementV;
using ElementO = typename FMHAKernel::ElementO;
using StrideQ = typename FMHAKernel::StrideQ;
using StrideK = typename FMHAKernel::StrideK;
using StrideV = typename FMHAKernel::StrideV;
using StrideO = typename FMHAKernel::StrideO;

static void launch_kernel(typename FMHAKernel::Params params) {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    dim3 const block = FMHAKernel::get_block_shape();
    dim3 const grid = FMHAKernel::get_grid_shape(params);
    int smem_size = FMHAKernel::SharedStorageSize;
    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);
    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(smem_size)};
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>,
#if (SYCL_INTEL_TARGET == 35)
        intelex::grf_size<512>
#else
        intelex::grf_size<256>
#endif
    };
    compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
    auto event = compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel>(policy, params);
    EventManager::getInstance().addEvent(event);
}

int main(int argc, const char** argv) {
    cutlass::CommandLine cmd(argc, argv);
    int batch = 1, num_heads_q = 4, num_heads_kv = 4;
    int seq_len_qo = 512, seq_len_kv = 512;
    int head_size_qk = 128, head_size_vo = 128;
    int window = 8;
    int iterations = 50, warmup = 10, do_verify = 1;
    cmd.get_cmd_line_argument("batch", batch, batch);
    cmd.get_cmd_line_argument("num_heads_q", num_heads_q, num_heads_q);
    cmd.get_cmd_line_argument("num_heads_kv", num_heads_kv, num_heads_q);
    cmd.get_cmd_line_argument("seq_len_qo", seq_len_qo, seq_len_kv);
    cmd.get_cmd_line_argument("seq_len_kv", seq_len_kv, seq_len_kv);
    cmd.get_cmd_line_argument("head_size_qk", head_size_qk, head_size_qk);
    cmd.get_cmd_line_argument("head_size_vo", head_size_vo, head_size_vo);
    cmd.get_cmd_line_argument("window", window, window);
    cmd.get_cmd_line_argument("iterations", iterations, iterations);
    cmd.get_cmd_line_argument("warmup", warmup, warmup);
    cmd.get_cmd_line_argument("verify", do_verify, do_verify);

    const float scale = 1.0f / std::sqrt((float)head_size_qk);
    ProblemShapeType shape;
    shape.batch = batch; shape.num_heads_q = num_heads_q; shape.num_heads_kv = num_heads_kv;
    shape.seq_len_qo = seq_len_qo; shape.seq_len_kv = seq_len_kv; shape.seq_len_kv_cache = 0;
    shape.head_size_qk = head_size_qk; shape.head_size_vo = head_size_vo;

    auto shape_Q = make_shape(seq_len_qo, head_size_qk, num_heads_q, batch);
    auto shape_K = make_shape(seq_len_kv, head_size_qk, num_heads_kv, batch);
    auto shape_V = make_shape(head_size_vo, seq_len_kv, num_heads_kv, batch);
    auto shape_O = make_shape(seq_len_qo, head_size_vo, num_heads_q, batch);
    auto stride_Q = cutlass::make_cute_packed_stride(StrideQ{}, shape_Q);
    auto stride_K = cutlass::make_cute_packed_stride(StrideK{}, shape_K);
    auto stride_V = cutlass::make_cute_packed_stride(StrideV{}, shape_V);
    auto stride_O = cutlass::make_cute_packed_stride(StrideO{}, shape_O);

    size_t total_Q = (size_t)batch * num_heads_q * seq_len_qo * head_size_qk;
    size_t total_K = (size_t)batch * num_heads_kv * seq_len_kv * head_size_qk;
    size_t total_V = (size_t)batch * num_heads_kv * seq_len_kv * head_size_vo;
    size_t total_O = (size_t)batch * num_heads_q * seq_len_qo * head_size_vo;

    cutlass::DeviceAllocation<ElementQ> block_Q(total_Q);
    cutlass::DeviceAllocation<ElementK> block_K(total_K);
    cutlass::DeviceAllocation<ElementV> block_V(total_V);
    cutlass::DeviceAllocation<ElementO> block_O(total_O);
    cutlass::DeviceAllocation<ElementK> block_K_cache;
    cutlass::DeviceAllocation<ElementV> block_V_cache;
    StrideK stride_K_cache{}; StrideV stride_V_cache{};
    compat::memset(block_O.get(), 0, total_O * sizeof(ElementO));
    initialize_block(block_Q, 2023);
    initialize_block(block_K, 2022);
    initialize_block(block_V, 2021);

    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    typename FMHAKernel::Arguments arguments{
        {shape, block_Q.get(), stride_Q, block_K.get(), stride_K,
         block_V.get(), stride_V, block_O.get(), stride_O,
         nullptr, {}, nullptr, {}, nullptr, {},
         1.f, 1.f, 1.f, 32,
         block_K_cache.get(), stride_K_cache,
         block_V_cache.get(), stride_V_cache,
         window},
        {scale, nullptr, 0, nullptr},
        {},
        hw_info
    };

    size_t ws = FMHAKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(ws);
    if (!FMHAKernel::can_implement(arguments)) {
        std::cout << "Disposition: Failed (can_implement)" << std::endl;
        return -1;
    }
    FMHAKernel::initialize_workspace(arguments, workspace.get());
    auto params = FMHAKernel::to_underlying_arguments(arguments, workspace.get());

    for (int i = 0; i < warmup; ++i) launch_kernel(params);
    compat::wait();

    int rc = 0;
    if (do_verify) {
        std::vector<ElementQ> hQ(total_Q); block_Q.copy_to_host(hQ.data(), total_Q);
        std::vector<ElementK> hK(total_K); block_K.copy_to_host(hK.data(), total_K);
        std::vector<ElementV> hV(total_V); block_V.copy_to_host(hV.data(), total_V);
        std::vector<ElementO> hO(total_O); block_O.copy_to_host(hO.data(), total_O);
        const int S = seq_len_qo, D = head_size_qk, DV = head_size_vo, H = num_heads_q;
        double maxerr = 0.0;
        std::vector<float> acc(DV);
        for (int b = 0; b < batch; ++b)
            for (int h = 0; h < H; ++h)
                for (int i = 0; i < S; ++i) {
                    int lo = i - window < 0 ? 0 : i - window;
                    int hi = i + window > S - 1 ? S - 1 : i + window;
                    float m = -1e30f, lsum = 0.f;
                    for (int d = 0; d < DV; ++d) acc[d] = 0.f;
                    for (int j = lo; j <= hi; ++j) {
                        float s = 0.f;
                        for (int d = 0; d < D; ++d)
                            s += float(hQ[((size_t)b*H+h)*S*D + (size_t)i*D+d]) *
                                 float(hK[((size_t)b*H+h)*S*D + (size_t)j*D+d]);
                        s *= scale;
                        float mn = s > m ? s : m;
                        float corr = std::exp(m - mn), p = std::exp(s - mn);
                        lsum = lsum * corr + p;
                        for (int d = 0; d < DV; ++d)
                            acc[d] = acc[d]*corr + p*float(hV[((size_t)b*H+h)*S*DV + (size_t)j*DV+d]);
                        m = mn;
                    }
                    float inv = 1.f / lsum;
                    for (int d = 0; d < DV; ++d) {
                        float ref = acc[d] * inv;
                        float got = float(hO[((size_t)b*H+h)*S*DV + (size_t)i*DV+d]);
                        double e = std::fabs((double)ref - (double)got);
                        if (e > maxerr) maxerr = e;
                    }
                }
        bool ok = maxerr < 3e-2;
        printf("max_abs_err: %.3e\n", maxerr);
        std::cout << "Disposition: " << (ok ? "Passed" : "Failed") << std::endl;
        if (!ok) rc = -1;
    }

    if (iterations > 0) {
        GPU_Clock timer; timer.start();
        for (int i = 0; i < iterations; ++i) launch_kernel(params);
        compat::wait();
        double t = timer.seconds() / iterations;
        double pairs = 0;
        for (int i = 0; i < seq_len_qo; ++i) {
            int lo = i - window < 0 ? 0 : i - window;
            int hi = i + window > seq_len_kv - 1 ? seq_len_kv - 1 : i + window;
            pairs += (hi - lo + 1);
        }
        double flops = 4.0 * batch * num_heads_q * pairs * head_size_qk;
        double tflops = (flops * 1e-12) / t;
        double bytes = (double)(total_Q + total_K + total_V + total_O) * sizeof(ElementQ);
        double gbps = (bytes * 1e-9) / t;
        double ai = flops / bytes;
        double ceil = ai * 608.0 / 1000.0;
        printf("AI=%.2f mem_ceiling=%.2f TFLOPS pct_of_roofline=%.1f%%\n",
                    ai, ceil, 100.0 * tflops / ceil);
        printf("Performance:   %.3f  GB/s,    %.3f  TFlop/s,   %.4f  ms\n", gbps, tflops, t*1000);
    }
    return rc;
}
