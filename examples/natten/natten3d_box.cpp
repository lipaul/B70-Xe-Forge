// True 3D-box neighborhood attention (na3d) via the patched sycl-tla FMHA (DPAS).
//
// Same FMHA kernel as natten1d_fmha.cpp but with 3D-box mask mode enabled:
// Q/K/V laid out [B,H,T*H*W,D] (volume flattened row-major), each query attends
// the (kt,kh,kw) box under NATTEN shift-clamp (window stays kernel-wide, clamped
// at volume edges). Self-verifies vs a shift-clamp box fp32 reference + benchmarks.
// Reuse for the LTX-2.5 DiffVAE na3d primitive (head_dim=64).

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

// head_dim-64 tile (KNOWN_FA_CONFIGS[64]): qk 128x64x32, pv 32/64, sg_q 8.
using FAConfig = FMHAConfigGenWithTileShape<
    FMHAMode::Prefill,
    bfloat16_t, bfloat16_t, bfloat16_t, float,
    cutlass::layout::RowMajor, cutlass::layout::ColumnMajor,
    cutlass::layout::RowMajor, cutlass::layout::RowMajor,
    float, false, false, false, false, false, false,
    128, 64, 32,
    8, 64,
    32, 64
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
    dim3 block = FMHAKernel::get_block_shape(), grid = FMHAKernel::get_grid_shape(params);
    int smem = FMHAKernel::SharedStorageSize;
    auto sb = compat::dim3(block.x, block.y, block.z), sg = compat::dim3(grid.x, grid.y, grid.z);
    compat::experimental::launch_properties lp{syclex::work_group_scratch_size(smem)};
    compat::experimental::kernel_properties kp{syclex::sub_group_size<cute::intel::sg_size>,
#if (SYCL_INTEL_TARGET == 35)
        intelex::grf_size<512>
#else
        intelex::grf_size<256>
#endif
    };
    compat::experimental::launch_policy policy{sg, sb, lp, kp};
    auto e = compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel>(policy, params);
    EventManager::getInstance().addEvent(e);
}

static int clampstart(int q, int k, int len) {
    int s = q - (k - 1) / 2;
    if (s < 0) s = 0;
    if (s > len - k) s = len - k;
    return s;
}

int main(int argc, const char** argv) {
    cutlass::CommandLine cmd(argc, argv);
    int B = 1, H = 8, T = 8, Hh = 16, W = 16, D = 64;
    int kt = 3, kh = 5, kw = 5, iterations = 50, warmup = 10, do_verify = 1;
    cmd.get_cmd_line_argument("batch", B, B);
    cmd.get_cmd_line_argument("num_heads_q", H, H);
    cmd.get_cmd_line_argument("t", T, T);
    cmd.get_cmd_line_argument("h", Hh, Hh);
    cmd.get_cmd_line_argument("w", W, W);
    cmd.get_cmd_line_argument("head", D, D);
    cmd.get_cmd_line_argument("kt", kt, kt);
    cmd.get_cmd_line_argument("kh", kh, kh);
    cmd.get_cmd_line_argument("kw", kw, kw);
    cmd.get_cmd_line_argument("iterations", iterations, iterations);
    cmd.get_cmd_line_argument("warmup", warmup, warmup);
    cmd.get_cmd_line_argument("verify", do_verify, do_verify);

    int S = T * Hh * W;  // flattened volume
    float scale = 1.0f / std::sqrt((float)D);

    ProblemShapeType shape;
    shape.batch = B; shape.num_heads_q = H; shape.num_heads_kv = H;
    shape.seq_len_qo = S; shape.seq_len_kv = S; shape.seq_len_kv_cache = 0;
    shape.head_size_qk = D; shape.head_size_vo = D;

    size_t N = (size_t)B * H * S * D;
    std::vector<float> fQ(N), fK(N), fV(N);
    std::mt19937 rng(7); std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (size_t i = 0; i < N; ++i) { fQ[i] = dist(rng); fK[i] = dist(rng); fV[i] = dist(rng); }
    // round to bf16 so host ref sees identical inputs as the kernel
    for (size_t i = 0; i < N; ++i) { fQ[i] = float(cutlass::bfloat16_t(fQ[i]));
        fK[i] = float(cutlass::bfloat16_t(fK[i])); fV[i] = float(cutlass::bfloat16_t(fV[i])); }

    cutlass::DeviceAllocation<ElementQ> bQ(N); cutlass::DeviceAllocation<ElementK> bK(N);
    cutlass::DeviceAllocation<ElementV> bV(N); cutlass::DeviceAllocation<ElementO> bO(N);
    std::vector<ElementQ> hQ(N), hK(N), hV(N);
    for (size_t i = 0; i < N; ++i) { hQ[i] = ElementQ(fQ[i]); hK[i] = ElementK(fK[i]); hV[i] = ElementV(fV[i]); }
    bQ.copy_from_host(hQ.data()); bK.copy_from_host(hK.data()); bV.copy_from_host(hV.data());
    compat::memset(bO.get(), 0, N * sizeof(ElementO));

    auto shQ = make_shape(S, D, H, B), shK = make_shape(S, D, H, B), shV = make_shape(D, S, H, B), shO = make_shape(S, D, H, B);
    auto sQ = cutlass::make_cute_packed_stride(StrideQ{}, shQ);
    auto sK = cutlass::make_cute_packed_stride(StrideK{}, shK);
    auto sV = cutlass::make_cute_packed_stride(StrideV{}, shV);
    auto sO = cutlass::make_cute_packed_stride(StrideO{}, shO);
    cutlass::DeviceAllocation<ElementK> bKc; cutlass::DeviceAllocation<ElementV> bVc;
    StrideK sKc{}; StrideV sVc{};
    cutlass::KernelHardwareInfo hw;
    hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw.device_id);

    typename FMHAKernel::Arguments args{
        {shape, bQ.get(), sQ, bK.get(), sK, bV.get(), sV, bO.get(), sO,
         nullptr, {}, nullptr, {}, nullptr, {}, 1.f, 1.f, 1.f, 32,
         bKc.get(), sKc, bVc.get(), sVc,
         0, T, Hh, W, kt, kh, kw},  // natten_window=0; 3D-box (T,H,W,kt,kh,kw)
        {scale, nullptr, 0, nullptr}, {}, hw};

    size_t ws = FMHAKernel::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> wk(ws);
    if (!FMHAKernel::can_implement(args)) { std::cout << "Disposition: Failed (can_implement)\n"; return -1; }
    FMHAKernel::initialize_workspace(args, wk.get());
    auto params = FMHAKernel::to_underlying_arguments(args, wk.get());
    for (int i = 0; i < warmup; ++i) launch_kernel(params);
    compat::wait();

    int rc = 0;
    if (do_verify) {
        std::vector<ElementO> hO(N); bO.copy_to_host(hO.data());
        double maxerr = 0.0;
        std::vector<float> num(D), den(D);
        for (int b = 0; b < B; ++b) for (int hh = 0; hh < H; ++hh)
            for (int qt = 0; qt < T; ++qt) for (int qh = 0; qh < Hh; ++qh) for (int qw = 0; qw < W; ++qw) {
                int st = clampstart(qt, kt, T), sh = clampstart(qh, kh, Hh), sw = clampstart(qw, kw, W);
                size_t qoff = (((size_t)b*H+hh)*S + (size_t)qt*Hh*W + (size_t)qh*W + qw) * D;
                float m = -1e30f, l = 0.f; for (int d=0;d<D;++d) num[d]=0.f;
                for (int dt=0; dt<kt; ++dt) for (int dh=0; dh<kh; ++dh) for (int dw=0; dw<kw; ++dw) {
                    int ct=st+dt, ch=sh+dh, cw=sw+dw;
                    size_t koff = (((size_t)b*H+hh)*S + (size_t)ct*Hh*W + (size_t)ch*W + cw) * D;
                    float s = 0.f; for (int d=0;d<D;++d) s += fQ[qoff+d]*fK[koff+d]; s *= scale;
                    float mn = s>m?s:m; float corr=std::exp(m-mn); float e=std::exp(s-mn);
                    l = l*corr+e;
                    for (int d=0;d<D;++d) num[d] = num[d]*corr + e*fV[koff+d];
                    m=mn;
                }
                for (int d=0;d<D;++d) { float ref = num[d]/l; double err = std::fabs((double)ref-(double)float(hO[qoff+d])); if(err>maxerr)maxerr=err; }
            }
        bool ok = maxerr < 3e-2;
        printf("max_abs_err: %.3e\n", maxerr);
        std::cout << "Disposition: " << (ok ? "Passed" : "Failed") << "\n";
        if (!ok) rc = -1;
    }

    if (iterations > 0) {
        GPU_Clock timer; timer.start();
        for (int i = 0; i < iterations; ++i) launch_kernel(params);
        compat::wait();
        double t = timer.seconds() / iterations;
        long keys = (long)kt*kh*kw;
        double flops = 4.0 * B * H * (double)S * keys * D;
        double tflops = flops * 1e-12 / t;
        double bytes = (double)N * 2.0 * 4.0;  // bf16 QKVO once (ideal)
        double ai = flops / bytes;
        double ceil = ai * 608.0 / 1000.0;
        printf("AI=%.1f mem_ceiling=%.2f pct_roofline=%.1f\n", ai, ceil, 100.0*tflops/ceil);
        printf("Performance:   %.3f  GB/s,    %.3f  TFlop/s,   %.4f  ms\n", bytes*1e-9/t, tflops, t*1000);
    }
    return rc;
}
