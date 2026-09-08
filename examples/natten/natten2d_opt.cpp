// Intel SYCL (DPC++) NATTEN 2D - OPTIMIZED (D-split across subgroup + SLM tiling).
//
// 2D neighborhood attention: query (r,c) attends a kh x kw window. Same
// optimizations as natten1d_opt.cpp:
//   * One SUBGROUP (G lanes) per query; each lane owns a D-slice (DS=D/G).
//   * A work-group handles a horizontal strip of TQ queries (same r, c in
//     [c0,c0+TQ)); their shared neighborhood block (rows [r-rh,r+rh], cols
//     [c0-rw,c0+TQ-1+rw]) is streamed through SLM and reused across the strip.
//   * Online softmax; band mask reduces to |cc-c|<=rw (rows are always in-range
//     for a row-strip). fp32 reference self-verify. --dtype f32|bf16.
//
// Layout: Q/K/V/O as [B, H, Hi, Wi, D] row-major.

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
    int B, H, Hi, Wi, D, RH, RW, TQ, nqt, tk, G, DS;
    float scale;
};

template <typename T>
auto make_kernel(const Params& p, T* dQ, T* dK, T* dV, T* dO,
                 sycl::local_accessor<T, 1>& kslm, sycl::local_accessor<T, 1>& vslm,
                 sycl::local_accessor<int, 1>& kcol) {
    const int B = p.B, H = p.H, Hi = p.Hi, Wi = p.Wi, D = p.D, RH = p.RH, RW = p.RW;
    const int TQ = p.TQ, nqt = p.nqt, tk = p.tk, G = p.G, DS = p.DS;
    const float scale = p.scale;
    const size_t SP = (size_t)Hi * Wi;
    const size_t local = (size_t)TQ * G;
    return [=](sycl::nd_item<1> it) {
        auto sg = it.get_sub_group();
        const int lid = it.get_local_id(0);
        const int lane = lid % G;
        const int lq = lid / G;
        const size_t grp = it.get_global_id(0) / ((size_t)TQ * G);
        const int qt = grp % nqt;
        const int bhr = grp / nqt;
        const int r = bhr % Hi;
        const int bh = bhr / Hi;
        const int hh = bh % H;
        const int bb = bh / H;
        const int c0 = qt * TQ;
        const int c = c0 + lq;
        const size_t base = ((size_t)bb * H + hh) * SP * D;
        const int d0 = lane * DS;

        float qq[MAX_DS], acc[MAX_DS];
        for (int k = 0; k < DS; ++k) {
            qq[k] = (c < Wi) ? to_f<T>(dQ[base + ((size_t)r * Wi + c) * D + d0 + k]) : 0.0f;
            acc[k] = 0.0f;
        }
        float m = -1e30f, lsum = 0.0f;

        int rlo = r - RH < 0 ? 0 : r - RH;
        int rhi = r + RH > Hi - 1 ? Hi - 1 : r + RH;
        int clo = c0 - RW < 0 ? 0 : c0 - RW;
        int chi = c0 + TQ - 1 + RW > Wi - 1 ? Wi - 1 : c0 + TQ - 1 + RW;
        const int bc = chi - clo + 1;
        const int nkeys = (rhi - rlo + 1) * bc;

        for (int kt = 0; kt < nkeys; kt += tk) {
            int kend = kt + tk;
            if (kend > nkeys) kend = nkeys;
            const int cnt = kend - kt;
            for (int e = lid; e < cnt * D; e += local) {
                const int krel = e / D;
                const int dd = e % D;
                const int ki = kt + krel;
                const int rr = rlo + ki / bc;
                const int cc = clo + ki % bc;
                if (dd == 0) kcol[krel] = cc;
                const size_t gk = ((size_t)rr * Wi + cc) * D + dd;
                kslm[e] = dK[base + gk];
                vslm[e] = dV[base + gk];
            }
            it.barrier(sycl::access::fence_space::local_space);

            if (c < Wi) {
                for (int krel = 0; krel < cnt; ++krel) {
                    const int cc = kcol[krel];
                    if (cc > c + RW || cc < c - RW) continue;  // band mask (cols)
                    const T* Kj = &kslm[(size_t)krel * D];
                    float part = 0.0f;
                    for (int k = 0; k < DS; ++k) part += qq[k] * to_f<T>(Kj[d0 + k]);
                    const float s =
                        sycl::reduce_over_group(sg, part, sycl::plus<float>()) * scale;
                    const float mn = s > m ? s : m;
                    const float corr = sycl::exp(m - mn);
                    const float pe = sycl::exp(s - mn);
                    lsum = lsum * corr + pe;
                    const T* Vj = &vslm[(size_t)krel * D];
                    for (int k = 0; k < DS; ++k)
                        acc[k] = acc[k] * corr + pe * to_f<T>(Vj[d0 + k]);
                    m = mn;
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
        }
        if (c < Wi) {
            const float inv = 1.0f / lsum;
            T* Oi = dO + base + ((size_t)r * Wi + c) * D;
            for (int k = 0; k < DS; ++k) Oi[d0 + k] = T(acc[k] * inv);
        }
    };
}

int main(int argc, char** argv) {
    const int B = get_int(argc, argv, "batch", 1);
    const int H = get_int(argc, argv, "heads", 4);
    const int Hi = get_int(argc, argv, "himg", 32);
    const int Wi = get_int(argc, argv, "wimg", 32);
    const int D = get_int(argc, argv, "dim", 64);
    const int KH = get_int(argc, argv, "kh", 7);
    const int KW = get_int(argc, argv, "kw", 7);
    const int TQ = get_int(argc, argv, "qtile", 8);
    const int TK = get_int(argc, argv, "kvtile", 32);
    const int iterations = get_int(argc, argv, "iterations", 50);
    const int warmup = get_int(argc, argv, "warmup", 10);
    const int verify = get_int(argc, argv, "verify", 1);
    const std::string dtype = get_str(argc, argv, "dtype", "bf16");
    const bool is_bf16 = (dtype == "bf16" || dtype == "bf");
    const float scale = 1.0f / std::sqrt((float)D);
    const int RH = KH / 2, RW = KW / 2;
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
    const int nqt = (Wi + TQ - 1) / TQ;

    const size_t SP = (size_t)Hi * Wi;
    const size_t N = (size_t)B * H * SP * D;
    std::vector<float> fQ(N), fK(N), fV(N), hO(N, 0.0f);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < N; ++i) {
        fQ[i] = dist(rng);
        fK[i] = dist(rng);
        fV[i] = dist(rng);
    }
    if (is_bf16)
        for (size_t i = 0; i < N; ++i) {
            fQ[i] = float(bf16(fQ[i]));
            fK[i] = float(bf16(fK[i]));
            fV[i] = float(bf16(fV[i]));
        }

    std::printf("Device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("Config: B=%d H=%d Hi=%d Wi=%d D=%d window=%dx%d dtype=%s qtile=%d kvtile=%d(->%d) G=%d DS=%d\n",
                B, H, Hi, Wi, D, KH, KW, dtype.c_str(), TQ, TK, tk, G, DS);

    const size_t local = (size_t)TQ * G;
    const size_t global = (size_t)B * H * Hi * nqt * TQ * G;
    const size_t slm = (size_t)tk * D;
    Params p{B, H, Hi, Wi, D, RH, RW, TQ, nqt, tk, G, DS, scale};

    auto run_all = [&](auto type_tag) {
        using T = decltype(type_tag);
        T *dQ = sycl::malloc_device<T>(N, q);
        T *dK = sycl::malloc_device<T>(N, q);
        T *dV = sycl::malloc_device<T>(N, q);
        T *dO = sycl::malloc_device<T>(N, q);
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
                sycl::local_accessor<int, 1> kcol(sycl::range<1>(tk), h);
                h.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
                    make_kernel<T>(p, dQ, dK, dV, dO, kslm, vslm, kcol));
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
    for (int r = 0; r < Hi; ++r)
        for (int c = 0; c < Wi; ++c) {
            int rlo = r - RH < 0 ? 0 : r - RH, rhi = r + RH > Hi - 1 ? Hi - 1 : r + RH;
            int clo = c - RW < 0 ? 0 : c - RW, chi = c + RW > Wi - 1 ? Wi - 1 : c + RW;
            pairs += (double)(rhi - rlo + 1) * (chi - clo + 1);
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
                const size_t base = ((size_t)b * H + hh) * SP * D;
                for (int r = 0; r < Hi; ++r)
                    for (int c = 0; c < Wi; ++c) {
                        const size_t qp = ((size_t)r * Wi + c) * D;
                        int rlo = r - RH < 0 ? 0 : r - RH, rhi = r + RH > Hi - 1 ? Hi - 1 : r + RH;
                        int clo = c - RW < 0 ? 0 : c - RW, chi = c + RW > Wi - 1 ? Wi - 1 : c + RW;
                        float maxs = -1e30f;
                        for (int rr = rlo; rr <= rhi; ++rr)
                            for (int cc = clo; cc <= chi; ++cc) {
                                const size_t kp = ((size_t)rr * Wi + cc) * D;
                                float s = 0.0f;
                                for (int d = 0; d < D; ++d) s += fQ[base + qp + d] * fK[base + kp + d];
                                s *= scale;
                                if (s > maxs) maxs = s;
                            }
                        std::vector<float> acc(D, 0.0f);
                        float denom = 0.0f;
                        for (int rr = rlo; rr <= rhi; ++rr)
                            for (int cc = clo; cc <= chi; ++cc) {
                                const size_t kp = ((size_t)rr * Wi + cc) * D;
                                float s = 0.0f;
                                for (int d = 0; d < D; ++d) s += fQ[base + qp + d] * fK[base + kp + d];
                                const float e = std::exp(s * scale - maxs);
                                denom += e;
                                for (int d = 0; d < D; ++d) acc[d] += e * fV[base + kp + d];
                            }
                        for (int d = 0; d < D; ++d) ref[base + qp + d] = acc[d] / denom;
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
