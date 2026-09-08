// Intel SYCL (DPC++) Neighborhood Attention (NATTEN) - 2D, correctness-first.
//
// Each query at spatial position (r, c) attends to a kh x kw local window
// centered on it (clipped at the borders). Same self-contained contract as the
// 1D kernel: host-generated fp32 inputs, naive fp32 reference, self-verify,
// benchmark, print "Disposition:" + "Performance: ... TFlop/s ... ms".
//
// Layout: Q/K/V/O as [B, H, Hi, Wi, D] row-major, fp32.

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

constexpr int MAX_D = 128;

int main(int argc, char** argv) {
    const int B = get_int(argc, argv, "batch", 1);
    const int H = get_int(argc, argv, "heads", 4);
    const int Hi = get_int(argc, argv, "himg", 32);
    const int Wi = get_int(argc, argv, "wimg", 32);
    const int D = get_int(argc, argv, "dim", 64);
    const int KH = get_int(argc, argv, "kh", 7);
    const int KW = get_int(argc, argv, "kw", 7);
    const int iterations = get_int(argc, argv, "iterations", 50);
    const int warmup = get_int(argc, argv, "warmup", 10);
    const int verify = get_int(argc, argv, "verify", 1);
    const float scale = 1.0f / std::sqrt((float)D);
    const int RH = KH / 2;
    const int RW = KW / 2;

    if (D > MAX_D) {
        std::fprintf(stderr, "dim=%d exceeds MAX_D=%d\n", D, MAX_D);
        return 2;
    }

    const size_t SP = (size_t)Hi * Wi;  // spatial positions per (b,h)
    const size_t N = (size_t)B * H * SP * D;
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
    std::printf("Config: B=%d H=%d Hi=%d Wi=%d D=%d window=%dx%d\n",
                B, H, Hi, Wi, D, KH, KW);

    float* dQ = sycl::malloc_device<float>(N, q);
    float* dK = sycl::malloc_device<float>(N, q);
    float* dV = sycl::malloc_device<float>(N, q);
    float* dO = sycl::malloc_device<float>(N, q);
    q.memcpy(dQ, hQ.data(), N * sizeof(float));
    q.memcpy(dK, hK.data(), N * sizeof(float));
    q.memcpy(dV, hV.data(), N * sizeof(float));
    q.wait();

    // One work-item per (b, h, r, c). Two-pass softmax over the kh x kw
    // neighborhood, accumulated into a private array (written to global once).
    auto kernel = [=](sycl::id<1> idx) {
        const int pos = idx[0] % SP;
        const int bh = idx[0] / SP;
        const int h = bh % H;
        const int b = bh / H;
        const int r = pos / Wi;
        const int c = pos % Wi;
        const size_t base = ((size_t)b * H + h) * SP * D;

        const float* Qrow = dQ + base + (size_t)pos * D;

        int rlo = r - RH < 0 ? 0 : r - RH;
        int rhi = r + RH > Hi - 1 ? Hi - 1 : r + RH;
        int clo = c - RW < 0 ? 0 : c - RW;
        int chi = c + RW > Wi - 1 ? Wi - 1 : c + RW;

        float maxs = -1e30f;
        for (int rr = rlo; rr <= rhi; ++rr)
            for (int cc = clo; cc <= chi; ++cc) {
                const float* Krow = dK + base + ((size_t)rr * Wi + cc) * D;
                float s = 0.0f;
                for (int d = 0; d < D; ++d) s += Qrow[d] * Krow[d];
                s *= scale;
                if (s > maxs) maxs = s;
            }
        float acc[MAX_D];
        for (int d = 0; d < D; ++d) acc[d] = 0.0f;
        float denom = 0.0f;
        for (int rr = rlo; rr <= rhi; ++rr)
            for (int cc = clo; cc <= chi; ++cc) {
                const size_t kp = (size_t)rr * Wi + cc;
                const float* Krow = dK + base + kp * D;
                const float* Vrow = dV + base + kp * D;
                float s = 0.0f;
                for (int d = 0; d < D; ++d) s += Qrow[d] * Krow[d];
                const float e = sycl::exp(s * scale - maxs);
                denom += e;
                for (int d = 0; d < D; ++d) acc[d] += e * Vrow[d];
            }
        const float inv = 1.0f / denom;
        float* Orow = dO + base + (size_t)pos * D;
        for (int d = 0; d < D; ++d) Orow[d] = acc[d] * inv;
    };

    const sycl::range<1> grng((size_t)B * H * SP);
    for (int i = 0; i < warmup; ++i) q.parallel_for(grng, kernel).wait();
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it) q.parallel_for(grng, kernel).wait();
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;

    q.memcpy(hO.data(), dO, N * sizeof(float));
    q.wait();

    double pairs = 0.0;
    for (int r = 0; r < Hi; ++r)
        for (int c = 0; c < Wi; ++c) {
            int rlo = r - RH < 0 ? 0 : r - RH;
            int rhi = r + RH > Hi - 1 ? Hi - 1 : r + RH;
            int clo = c - RW < 0 ? 0 : c - RW;
            int chi = c + RW > Wi - 1 ? Wi - 1 : c + RW;
            pairs += (double)(rhi - rlo + 1) * (chi - clo + 1);
        }
    const double flops = 4.0 * (double)B * H * pairs * D;
    const double tflops = (flops * 1e-12) / (ms * 1e-3);
    const double bytes = (double)N * sizeof(float) * 4.0;
    const double gbps = (bytes * 1e-9) / (ms * 1e-3);

    int rc = 0;
    if (verify) {
        std::vector<float> ref(N, 0.0f);
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < H; ++h) {
                const size_t base = ((size_t)b * H + h) * SP * D;
                for (int r = 0; r < Hi; ++r)
                    for (int c = 0; c < Wi; ++c) {
                        const size_t qp = (size_t)r * Wi + c;
                        int rlo = r - RH < 0 ? 0 : r - RH;
                        int rhi = r + RH > Hi - 1 ? Hi - 1 : r + RH;
                        int clo = c - RW < 0 ? 0 : c - RW;
                        int chi = c + RW > Wi - 1 ? Wi - 1 : c + RW;
                        float maxs = -1e30f;
                        for (int rr = rlo; rr <= rhi; ++rr)
                            for (int cc = clo; cc <= chi; ++cc) {
                                const size_t kp = (size_t)rr * Wi + cc;
                                float s = 0.0f;
                                for (int d = 0; d < D; ++d)
                                    s += hQ[base + qp * D + d] *
                                         hK[base + kp * D + d];
                                s *= scale;
                                if (s > maxs) maxs = s;
                            }
                        std::vector<float> acc(D, 0.0f);
                        float denom = 0.0f;
                        for (int rr = rlo; rr <= rhi; ++rr)
                            for (int cc = clo; cc <= chi; ++cc) {
                                const size_t kp = (size_t)rr * Wi + cc;
                                float s = 0.0f;
                                for (int d = 0; d < D; ++d)
                                    s += hQ[base + qp * D + d] *
                                         hK[base + kp * D + d];
                                const float e = std::exp(s * scale - maxs);
                                denom += e;
                                for (int d = 0; d < D; ++d)
                                    acc[d] += e * hV[base + kp * D + d];
                            }
                        for (int d = 0; d < D; ++d)
                            ref[base + qp * D + d] = acc[d] / denom;
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
