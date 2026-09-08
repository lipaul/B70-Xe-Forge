// Intel SYCL (DPC++) NATTEN 1D - OPTIMIZED (D-split across subgroup + SLM tiling).
//
// Band-restricted, SLM-tiled flash-attention for 1D neighborhood attention.
//   * One SUBGROUP (G lanes) per query; each lane owns a D-slice (DS=D/G) ->
//     high occupancy, no big private accumulator, vectorizable dots.
//   * QK score = per-lane partial dot + sycl::reduce_over_group (broadcast).
//   * Online (running max + rescale) softmax; PV accumulated per-lane.
//   * K/V streamed through SLM in key-tiles, reused across the TQ queries.
//   * Band mask: skip out-of-window keys.
//   * --dtype f32|bf16 (bf16 halves bytes -> ~2x the memory-roofline ceiling).
// Self-verifies vs an fp32 reference computed from the same (rounded) inputs.

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using bf16 = sycl::ext::oneapi::bfloat16;

static int get_int(int argc, char** argv, const char* key, int def) {
    std::string p = std::string("--") + key + "=";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind(p, 0) == 0) return std::atoi(a.c_str() + p.size());
    }
    return def;
}
static std::string get_str(int argc, char** argv, const char* key,
                           const std::string& def) {
    std::string p = std::string("--") + key + "=";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind(p, 0) == 0) return a.substr(p.size());
    }
    return def;
}

constexpr int MAX_DS = 8;
constexpr double PEAK_BW_GBPS = 608.0;

template <typename T>
static inline float to_f(T x);
template <>
inline float to_f<float>(float x) {
    return x;
}
template <>
inline float to_f<bf16>(bf16 x) {
    return float(x);
}

struct Params {
    int B, H, S, D, W, TQ, nqt, tk, G, DS;
    float scale;
};

template <typename T>
auto make_kernel(const Params& p, T* dQ, T* dK, T* dV, T* dO,
                 sycl::local_accessor<T, 1>& kslm, sycl::local_accessor<T, 1>& vslm) {
    const int B = p.B, H = p.H, S = p.S, D = p.D, W = p.W, TQ = p.TQ, nqt = p.nqt;
    const int tk = p.tk, G = p.G, DS = p.DS;
    const float scale = p.scale;
    const size_t local = (size_t)TQ * G;
    return [=](sycl::nd_item<1> it) {
        auto sg = it.get_sub_group();
        const int lid = it.get_local_id(0);
        const int lane = lid % G;
        const int lq = lid / G;
        const size_t grp = it.get_global_id(0) / ((size_t)TQ * G);
        const int qtile = grp % nqt;
        const int bh = grp / nqt;
        const int hh = bh % H;
        const int bb = bh / H;
        const int i = qtile * TQ + lq;
        const size_t base = ((size_t)bb * H + hh) * S * D;
        const int d0 = lane * DS;

        float qq[MAX_DS], acc[MAX_DS];
        for (int k = 0; k < DS; ++k) {
            qq[k] = (i < S) ? to_f<T>(dQ[base + (size_t)i * D + d0 + k]) : 0.0f;
            acc[k] = 0.0f;
        }
        float m = -1e30f, lsum = 0.0f;
        int jlo = i - W < 0 ? 0 : i - W;
        int jhi = i + W > S - 1 ? S - 1 : i + W;
        int glo = qtile * TQ - W;
        if (glo < 0) glo = 0;
        int ghi = qtile * TQ + TQ - 1 + W;
        if (ghi > S - 1) ghi = S - 1;

        for (int kt = glo; kt <= ghi; kt += tk) {
            int kend = kt + tk;
            if (kend > ghi + 1) kend = ghi + 1;
            const int cnt = kend - kt;
            for (int e = lid; e < cnt * D; e += local) {
                const int jr = e / D;
                const int dd = e % D;
                const size_t gk = (size_t)(kt + jr) * D + dd;
                kslm[e] = dK[base + gk];
                vslm[e] = dV[base + gk];
            }
            it.barrier(sycl::access::fence_space::local_space);

            if (i < S) {
                int a = jlo > kt ? jlo : kt;
                int bnd = (jhi < kend - 1) ? jhi : kend - 1;
                for (int j = a; j <= bnd; ++j) {
                    const T* Kj = &kslm[(size_t)(j - kt) * D];
                    float part = 0.0f;
                    for (int k = 0; k < DS; ++k) part += qq[k] * to_f<T>(Kj[d0 + k]);
                    const float s =
                        sycl::reduce_over_group(sg, part, sycl::plus<float>()) * scale;
                    const float mn = s > m ? s : m;
                    const float corr = sycl::exp(m - mn);
                    const float pexp = sycl::exp(s - mn);
                    lsum = lsum * corr + pexp;
                    const T* Vj = &vslm[(size_t)(j - kt) * D];
                    for (int k = 0; k < DS; ++k)
                        acc[k] = acc[k] * corr + pexp * to_f<T>(Vj[d0 + k]);
                    m = mn;
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
        }
        if (i < S) {
            const float inv = 1.0f / lsum;
            T* Oi = dO + base + (size_t)i * D;
            for (int k = 0; k < DS; ++k) Oi[d0 + k] = T(acc[k] * inv);
        }
    };
}

int main(int argc, char** argv) {
    const int B = get_int(argc, argv, "batch", 1);
    const int H = get_int(argc, argv, "heads", 4);
    const int S = get_int(argc, argv, "seq", 512);
    const int D = get_int(argc, argv, "dim", 64);
    const int W = get_int(argc, argv, "window", 8);
    const int TQ = get_int(argc, argv, "qtile", 8);
    const int TK = get_int(argc, argv, "kvtile", 32);
    const int iterations = get_int(argc, argv, "iterations", 50);
    const int warmup = get_int(argc, argv, "warmup", 10);
    const int verify = get_int(argc, argv, "verify", 1);
    const std::string dtype = get_str(argc, argv, "dtype", "bf16");
    const bool is_bf16 = (dtype == "bf16" || dtype == "bf16" || dtype == "bf");
    const float scale = 1.0f / std::sqrt((float)D);
    const int bytes = is_bf16 ? 2 : 4;

    sycl::queue q{sycl::gpu_selector_v};
    int G = 0;
    for (int sz : q.get_device().get_info<sycl::info::device::sub_group_sizes>())
        if (D % sz == 0 && (G == 0 || sz > G)) G = sz;
    if (G == 0) {
        std::fprintf(stderr, "no sub_group_size divides D=%d\n", D);
        return 2;
    }
    const int DS = D / G;
    if (DS > MAX_DS) {
        std::fprintf(stderr, "D/G=%d exceeds MAX_DS=%d\n", DS, MAX_DS);
        return 2;
    }
    int tk = TK;
    while ((size_t)2 * tk * D * bytes > 128 * 1024 && tk > 8) tk /= 2;
    const int nqt = (S + TQ - 1) / TQ;

    const size_t N = (size_t)B * H * S * D;
    std::vector<float> fQ(N), fK(N), fV(N), hO(N, 0.0f);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < N; ++i) {
        fQ[i] = dist(rng);
        fK[i] = dist(rng);
        fV[i] = dist(rng);
    }
    // Round inputs to the storage dtype so the reference sees identical inputs.
    if (is_bf16)
        for (size_t i = 0; i < N; ++i) {
            fQ[i] = float(bf16(fQ[i]));
            fK[i] = float(bf16(fK[i]));
            fV[i] = float(bf16(fV[i]));
        }

    std::printf("Device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("Config: B=%d H=%d S=%d D=%d w=%d dtype=%s qtile=%d kvtile=%d(->%d) G=%d DS=%d\n",
                B, H, S, D, W, dtype.c_str(), TQ, TK, tk, G, DS);

    const size_t local = (size_t)TQ * G;
    const size_t global = (size_t)B * H * nqt * TQ * G;
    const size_t slm = (size_t)tk * D;
    Params p{B, H, S, D, W, TQ, nqt, tk, G, DS, scale};

    auto run_all = [&](auto type_tag) {
        using T = decltype(type_tag);
        T *dQ, *dK, *dV, *dO;
        dQ = sycl::malloc_device<T>(N, q);
        dK = sycl::malloc_device<T>(N, q);
        dV = sycl::malloc_device<T>(N, q);
        dO = sycl::malloc_device<T>(N, q);
        std::vector<T> hQ(N), hK(N), hV(N);
        for (size_t i = 0; i < N; ++i) {
            hQ[i] = T(fQ[i]);
            hK[i] = T(fK[i]);
            hV[i] = T(fV[i]);
        }
        q.memcpy(dQ, hQ.data(), N * sizeof(T));
        q.memcpy(dK, hK.data(), N * sizeof(T));
        q.memcpy(dV, hV.data(), N * sizeof(T));
        q.wait();
        auto submit = [&]() {
            q.submit([&](sycl::handler& h) {
                sycl::local_accessor<T, 1> kslm(sycl::range<1>(slm), h);
                sycl::local_accessor<T, 1> vslm(sycl::range<1>(slm), h);
                h.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
                    make_kernel<T>(p, dQ, dK, dV, dO, kslm, vslm));
            });
        };
        for (int i = 0; i < warmup; ++i) submit(), q.wait();
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iterations; ++it) submit(), q.wait();
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
        std::vector<T> hOut(N);
        q.memcpy(hOut.data(), dO, N * sizeof(T));
        q.wait();
        for (size_t i = 0; i < N; ++i) hO[i] = to_f<T>(hOut[i]);
        sycl::free(dQ, q);
        sycl::free(dK, q);
        sycl::free(dV, q);
        sycl::free(dO, q);
        return ms;
    };

    double ms = is_bf16 ? run_all(bf16{}) : run_all(float{});

    double pairs = 0.0;
    for (int i = 0; i < S; ++i) {
        int lo = i - W < 0 ? 0 : i - W;
        int hi = i + W > S - 1 ? S - 1 : i + W;
        pairs += (hi - lo + 1);
    }
    const double flops = 4.0 * (double)B * H * pairs * D;
    const double tflops = (flops * 1e-12) / (ms * 1e-3);
    const double membytes = (double)N * bytes * 4.0;
    const double gbps = (membytes * 1e-9) / (ms * 1e-3);
    const double ai = flops / membytes;
    const double ceiling = ai * PEAK_BW_GBPS / 1000.0;
    std::printf("AI=%.2f FLOP/B  mem_ceiling=%.2f TFLOPS  pct_of_roofline=%.1f%%\n",
                ai, ceiling, 100.0 * tflops / ceiling);

    int rc = 0;
    if (verify) {
        std::vector<float> ref(N, 0.0f);
        for (int b = 0; b < B; ++b)
            for (int hh = 0; hh < H; ++hh) {
                const size_t base = ((size_t)b * H + hh) * S * D;
                for (int i = 0; i < S; ++i) {
                    int lo = i - W < 0 ? 0 : i - W;
                    int hi = i + W > S - 1 ? S - 1 : i + W;
                    float maxs = -1e30f;
                    for (int j = lo; j <= hi; ++j) {
                        float s = 0.0f;
                        for (int d = 0; d < D; ++d)
                            s += fQ[base + (size_t)i * D + d] *
                                 fK[base + (size_t)j * D + d];
                        s *= scale;
                        if (s > maxs) maxs = s;
                    }
                    std::vector<float> acc(D, 0.0f);
                    float denom = 0.0f;
                    for (int j = lo; j <= hi; ++j) {
                        float s = 0.0f;
                        for (int d = 0; d < D; ++d)
                            s += fQ[base + (size_t)i * D + d] *
                                 fK[base + (size_t)j * D + d];
                        const float e = std::exp(s * scale - maxs);
                        denom += e;
                        for (int d = 0; d < D; ++d)
                            acc[d] += e * fV[base + (size_t)j * D + d];
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
        const double tol = is_bf16 ? 3e-2 : 1e-2;
        const bool ok = maxerr < tol;
        std::printf("max_abs_err: %.3e (tol %.0e)\n", maxerr, tol);
        std::printf("Disposition: %s\n", ok ? "Passed" : "Failed");
        if (!ok) rc = -1;
    }

    std::printf("Performance:   %.3f  GB/s,    %.3f  TFlop/s,   %.4f  ms\n",
                gbps, tflops, ms);
    return rc;
}
