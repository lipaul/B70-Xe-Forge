// Intel SYCL (DPC++) Neighborhood Attention (NATTEN) - 1D, correctness-first.
//
// Self-contained program: generates fp32 inputs on host, runs a tiled SYCL
// kernel where each query attends to a local window [i-w, i+w], computes a
// naive fp32 host reference, self-verifies, and benchmarks.
//
// Contract with xe_forge.core.sycl_executor.SyclExecutor.execute_raw:
//   - args passed as --key=value
//   - prints "Disposition: Passed|Failed" (only when --verify=1)
//   - prints "Performance: <g> GB/s, <t> TFlop/s, <m> ms"
//   - exit 0 on success, non-zero on verification failure
//
// Layout: Q/K/V/O as [B, H, S, D] row-major, fp32.

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static int get_int(int argc, char** argv, const char* key, int def) {
    std::string p = std::string("--") + key + "=";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind(p, 0) == 0) return std::atoi(a.c_str() + p.size());
    }
    return def;
}

// Max head_dim supported by the private accumulator array (correctness-first
// baseline; head_dim 64/128 covers typical transformer configs).
constexpr int MAX_D = 128;

int main(int argc, char** argv) {
    const int B = get_int(argc, argv, "batch", 1);
    const int H = get_int(argc, argv, "heads", 4);
    const int S = get_int(argc, argv, "seq", 512);
    const int D = get_int(argc, argv, "dim", 64);
    const int W = get_int(argc, argv, "window", 8);
    const int iterations = get_int(argc, argv, "iterations", 50);
    const int warmup = get_int(argc, argv, "warmup", 10);
    const int verify = get_int(argc, argv, "verify", 1);
    const float scale = 1.0f / std::sqrt((float)D);

    if (D > MAX_D) {
        std::fprintf(stderr, "dim=%d exceeds MAX_D=%d\n", D, MAX_D);
        return 2;
    }

    const size_t N = (size_t)B * H * S * D;
    std::vector<float> hQ(N), hK(N), hV(N), hO(N, 0.0f);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < N; ++i) {
        hQ[i] = dist(rng);
        hK[i] = dist(rng);
        hV[i] = dist(rng);
    }

    sycl::queue q{sycl::gpu_selector_v};
    std::printf("Device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("Config: B=%d H=%d S=%d D=%d window=%d (kernel_size=%d)\n",
                B, H, S, D, W, 2 * W + 1);

    float* dQ = sycl::malloc_device<float>(N, q);
    float* dK = sycl::malloc_device<float>(N, q);
    float* dV = sycl::malloc_device<float>(N, q);
    float* dO = sycl::malloc_device<float>(N, q);
    q.memcpy(dQ, hQ.data(), N * sizeof(float));
    q.memcpy(dK, hK.data(), N * sizeof(float));
    q.memcpy(dV, hV.data(), N * sizeof(float));
    q.wait();

    // One work-item per (b, h, query_position i). Two-pass softmax over the
    // local window; accumulates into a fixed-size private array (written to
    // global once) to avoid per-key global read-modify-writes.
    auto kernel = [=](sycl::id<1> idx) {
        const int i = idx[0] % S;
        const int bh = idx[0] / S;
        const int h = bh % H;
        const int b = bh / H;
        const size_t base = ((size_t)b * H + h) * S * D;
        int jlo = i - W;
        if (jlo < 0) jlo = 0;
        int jhi = i + W;
        if (jhi > S - 1) jhi = S - 1;

        const float* Qi = dQ + base + (size_t)i * D;

        float maxs = -1e30f;
        for (int j = jlo; j <= jhi; ++j) {
            const float* Kj = dK + base + (size_t)j * D;
            float s = 0.0f;
            for (int d = 0; d < D; ++d) s += Qi[d] * Kj[d];
            s *= scale;
            if (s > maxs) maxs = s;
        }
        float acc[MAX_D];
        for (int d = 0; d < D; ++d) acc[d] = 0.0f;
        float denom = 0.0f;
        for (int j = jlo; j <= jhi; ++j) {
            const float* Kj = dK + base + (size_t)j * D;
            const float* Vj = dV + base + (size_t)j * D;
            float s = 0.0f;
            for (int d = 0; d < D; ++d) s += Qi[d] * Kj[d];
            const float e = sycl::exp(s * scale - maxs);
            denom += e;
            for (int d = 0; d < D; ++d) acc[d] += e * Vj[d];
        }
        const float inv = 1.0f / denom;
        float* Oi = dO + base + (size_t)i * D;
        for (int d = 0; d < D; ++d) Oi[d] = acc[d] * inv;
    };

    const sycl::range<1> grng((size_t)B * H * S);
    for (int i = 0; i < warmup; ++i) q.parallel_for(grng, kernel).wait();
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it) q.parallel_for(grng, kernel).wait();
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;

    q.memcpy(hO.data(), dO, N * sizeof(float));
    q.wait();

    // Effective (query,key) pairs summed over positions (edge-clipped window).
    double pairs = 0.0;
    for (int i = 0; i < S; ++i) {
        int lo = i - W < 0 ? 0 : i - W;
        int hi = i + W > S - 1 ? S - 1 : i + W;
        pairs += (hi - lo + 1);
    }
    const double flops = 4.0 * (double)B * H * pairs * D;  // QK(2) + PV(2)
    const double tflops = (flops * 1e-12) / (ms * 1e-3);
    const double bytes = (double)N * sizeof(float) * 4.0;  // Q,K,V read + O write
    const double gbps = (bytes * 1e-9) / (ms * 1e-3);

    int rc = 0;
    if (verify) {
        std::vector<float> ref(N, 0.0f);
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < H; ++h) {
                const size_t base = ((size_t)b * H + h) * S * D;
                for (int i = 0; i < S; ++i) {
                    int lo = i - W < 0 ? 0 : i - W;
                    int hi = i + W > S - 1 ? S - 1 : i + W;
                    float maxs = -1e30f;
                    for (int j = lo; j <= hi; ++j) {
                        float s = 0.0f;
                        for (int d = 0; d < D; ++d)
                            s += hQ[base + (size_t)i * D + d] *
                                 hK[base + (size_t)j * D + d];
                        s *= scale;
                        if (s > maxs) maxs = s;
                    }
                    std::vector<float> acc(D, 0.0f);
                    float denom = 0.0f;
                    for (int j = lo; j <= hi; ++j) {
                        float s = 0.0f;
                        for (int d = 0; d < D; ++d)
                            s += hQ[base + (size_t)i * D + d] *
                                 hK[base + (size_t)j * D + d];
                        const float e = std::exp(s * scale - maxs);
                        denom += e;
                        for (int d = 0; d < D; ++d)
                            acc[d] += e * hV[base + (size_t)j * D + d];
                    }
                    for (int d = 0; d < D; ++d)
                        ref[base + (size_t)i * D + d] = acc[d] / denom;
                }
            }
        double maxerr = 0.0;
        for (size_t k = 0; k < N; ++k) {
            double e = std::fabs((double)hO[k] - (double)ref[k]);
            if (e > maxerr) maxerr = e;
        }
        const bool ok = maxerr < 1e-2;
        std::printf("max_abs_err: %.3e\n", maxerr);
        std::printf("Disposition: %s\n", ok ? "Passed" : "Failed");
        if (!ok) rc = -1;
    }

    std::printf("Performance:   %.3f  GB/s,    %.3f  TFlop/s,   %.4f  ms\n",
                gbps, tflops, ms);

    sycl::free(dQ, q);
    sycl::free(dK, q);
    sycl::free(dV, q);
    sycl::free(dO, q);
    return rc;
}
