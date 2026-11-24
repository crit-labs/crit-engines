// CPU expert tier for GGUF experts (IQ3_S gate|up, IQ4_XS or Q6_K down), joined to the device inside a captured
// decode step through a mailbox in mapped pinned memory.
//
// One slot per routed layer. A decode step's experts call posts the layer's host routes into its slot (post_kernel: the
// token rows as FP32, each host route's token and host expert index, ready = 1), runs the device routes on the device,
// then waits for the slot (wait_kernel: one thread polls done, then resets it) and sums each token's routes in route
// order, a host route's row read from the slot (combine_kernel). A host service thread finds raised slots, computes
// their routes on a worker pool (gate|up rows, silu(gate) * up, down rows) straight from the stored GGUF blocks, with
// the token rows quantized to the formats' int8 units of 32 as the device's kernels quantize them (AVX2 integer dot
// products, FP32 scales), and answers done = 1. A slot with no host route is answered by the device itself (done = 1 in
// the post), so every captured wait completes. Quality mode speed: the CPU's rows round like the device's but sum in
// their own order.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <functional>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <chrono>
#include <thread>
#include <vector>

#include "iq3s_grid_host.h"

namespace {

constexpr int MAX_TOKENS = 8;
constexpr int KIND_IQ4_XS = 2, KIND_IQ3_S = 3, KIND_Q6_K = 5;

struct SlotHeader {          // 64 bytes at the slot's start
    volatile int32_t ready;
    volatile int32_t done;
    int32_t routes;
    int32_t tokens;
    int32_t pad[12];
};

struct Layout {
    int hidden, top, max_routes;
    size_t tok_off, idx_off, x_off, out_off, bytes;
    explicit Layout(int hidden_ = 0, int top_ = 8) : hidden(hidden_), top(top_), max_routes(MAX_TOKENS * top_) {
        tok_off = sizeof(SlotHeader);
        idx_off = tok_off + sizeof(int32_t) * max_routes;
        x_off = (idx_off + sizeof(int32_t) * max_routes + 255) / 256 * 256;
        out_off = x_off + sizeof(float) * (size_t)MAX_TOKENS * hidden;
        bytes = (out_off + sizeof(float) * (size_t)max_routes * hidden + 255) / 256 * 256;
    }
};

// ---- device kernels ------------------------------------------------------------------------------------------------
__global__ void post_kernel(uint8_t* slot, const int64_t* __restrict__ ids, const int32_t* __restrict__ codes,
                            const __nv_bfloat16* __restrict__ x, int T, int top, int hidden, int32_t* __restrict__ pos,
                            size_t tok_off, size_t idx_off, size_t x_off) {
    SlotHeader* head = reinterpret_cast<SlotHeader*>(slot);
    float* xs = reinterpret_cast<float*>(slot + x_off);
    for (int i = threadIdx.x; i < T * hidden; i += blockDim.x) xs[i] = __bfloat162float(x[i]);
    __shared__ int count;
    if (threadIdx.x == 0) {
        int32_t* tok = reinterpret_cast<int32_t*>(slot + tok_off);
        int32_t* idx = reinterpret_cast<int32_t*>(slot + idx_off);
        int n = 0;
        for (int r = 0; r < T * top; ++r) {
            const int64_t e = ids[r];
            const int c = e >= 0 ? codes[e] : -1;
            if (c >= 0) {
                tok[n] = r / top;
                idx[n] = c;
                pos[r] = n++;
            } else {
                pos[r] = -1;
            }
        }
        head->routes = n;
        head->tokens = T;
        count = n;
    }
    __syncthreads();
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        if (count == 0) head->done = 1;      // nothing for the host: answered here
        else head->ready = 1;
        __threadfence_system();
    }
}

__global__ void wait_kernel(uint8_t* slot) {
    SlotHeader* head = reinterpret_cast<SlotHeader*>(slot);
    // __ldcv: fetched again every time (never served from a cache line an earlier read left in L1 or L2)
    while (__ldcv(const_cast<const int*>(reinterpret_cast<volatile int*>(&head->done))) == 0) __nanosleep(256);
    head->done = 0;
    __threadfence_system();
}

// out[t][h] = sum over k in order of w[t][k] * row(t, k)[h]: a device route's row from y [T * top][hidden], a host
// route's from the slot's outputs (pos >= 0), routes of id < 0 skipped
__global__ void combine_kernel(const float* __restrict__ y, const uint8_t* __restrict__ slot, size_t out_off,
                               const int32_t* __restrict__ pos, const int64_t* __restrict__ ids,
                               const float* __restrict__ w, int top, int hidden, float* __restrict__ out) {
    const int t = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    const float* host = reinterpret_cast<const float*>(slot + out_off);
    float s = 0.f;
    for (int k = 0; k < top; ++k) {
        const int r = t * top + k;
        if (ids[r] < 0) continue;
        // a host row read past every cache (the CPU rewrote these bytes since the last step read them)
        const float v = pos[r] >= 0 ? __ldcv(host + (size_t)pos[r] * hidden + h) : y[(size_t)r * hidden + h];
        s += w[r] * v;
    }
    out[(size_t)t * hidden + h] = s;
}

// ---- host arithmetic -----------------------------------------------------------------------------------------------
inline float half_to_float(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {                               // subnormal: normalize
            int e = -1;
            do { ++e; man <<= 1; } while ((man & 0x400u) == 0);
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((man & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline float hsum_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

inline int hsum_epi32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

// int32 sums of 32 signed int8 products, per 32-bit lane (8 lanes of 4 products)
inline __m256i dot_lanes(__m256i w, __m256i x) {
    const __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
    return _mm256_madd_epi16(p, _mm256_set1_epi16(1));
}


#define CRIT_HD inline
inline float crit_rint(float v) { return std::nearbyint(v); }
inline float crit_fma(float a, float b, float c) { return std::fmaf(a, b, c); }
inline float crit_bits_float(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
inline float crit_mul(float a, float b) { return a * b; }
inline float crit_add(float a, float b) { return a + b; }
inline float crit_div(float a, float b) { return a / b; }

// exp(x) from IEEE operations only (no library exp, whose last bit differs between CUDA and a host libm): x = n ln2 + r,
// |r| <= ln2 / 2 (n rounded to nearest, r by two fmas with ln2 split in a high and a low part), e^r by a degree-7
// Taylor polynomial in Horner form with fmaf, times 2^n set in the exponent bits. Relative error ~1e-7.
CRIT_HD inline float crit_exp(float x) {
    x = x < -87.0f ? -87.0f : (x > 88.0f ? 88.0f : x);
    const float n = crit_rint(crit_mul(x, 1.44269504088896341f));
    float r = crit_fma(n, -0.693145751953125f, x);
    r = crit_fma(n, -1.428606765330187045e-06f, r);
    float p = 1.98412698412698413e-04f;
    p = crit_fma(p, r, 1.38888888888888889e-03f);
    p = crit_fma(p, r, 8.33333333333333333e-03f);
    p = crit_fma(p, r, 4.16666666666666667e-02f);
    p = crit_fma(p, r, 1.66666666666666667e-01f);
    p = crit_fma(p, r, 0.5f);
    p = crit_fma(p, r, 1.0f);
    p = crit_fma(p, r, 1.0f);
    const int e = (int)n + 127;
    return crit_mul(p, crit_bits_float((uint32_t)e << 23));
}

// silu(g) * u in FP32: g / (1 + exp(-g)) times u
CRIT_HD inline float crit_swiglu(float g, float u) {
    return crit_mul(crit_div(g, crit_add(1.0f, crit_exp(-g))), u);
}

#undef CRIT_HD

// quantize n values to int8 units of 32 (d = max |v| / 127, q = round(v / d)), as the device's quantize_kernel
void quantize_units(const float* v, int n, int8_t* q, float* d) {
    for (int u = 0; u < n / 32; ++u) {
        float m = 0.f;
        for (int i = 0; i < 32; ++i) m = std::max(m, std::fabs(v[u * 32 + i]));
        const float s = m / 127.f;
        d[u] = s;
        for (int i = 0; i < 32; ++i) {
            const int c = m > 0.f ? (int)std::nearbyint(v[u * 32 + i] / s) : 0;
            q[u * 32 + i] = (int8_t)std::max(-127, std::min(127, c));
        }
    }
}

// ---- a superblock (8 units of 32 values) of a row: each unit's 8 partial int32 dot products (4 products each, already
// times the unit's integer sub-scales for Q6_K) with the activation units, and each unit's weight scale exactly as the
// device's unit_isum computes it (gguf_gemv.cu)
struct Consts {
    __m256i bits, spread, one8, low4_256, two, bias, ones16, shifts, one32;
    __m128i table, low4;
    Consts() {
        bits = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
        spread = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                  2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
        one8 = _mm256_set1_epi8(1);
        low4_256 = _mm256_set1_epi8(15);
        two = _mm256_set1_epi8(3);
        bias = _mm256_set1_epi8(32);
        ones16 = _mm256_set1_epi16(1);
        shifts = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        one32 = _mm256_set1_epi32(1);
        table = _mm_setr_epi8(-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);
        low4 = _mm_set1_epi8(15);
    }
};
static const Consts C;

inline __m256i dot8(__m256i w, __m256i x) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w)), C.ones16);
}

template <int KIND> struct Block;

template <> struct Block<KIND_IQ3_S> {
    static constexpr int BYTES = 110;
    static void run(const uint8_t* b, const int8_t* xq, __m256i (&v)[8], float (&sc)[8]) {
        const float d = half_to_float(b);
        for (int j = 0; j < 8; ++j) {
            const int nib = (b[106 + (j >> 1)] >> (4 * (j & 1))) & 15;
            sc[j] = d * (float)(1 + 2 * nib);
            const uint8_t* qs = b + 2 + j * 8;
            const int qh = b[66 + j];
            // eight grid words by plain loads (Zen 2's gathers measured slower: 6.0 against 9.6 GB/s of experts)
            const __m256i mag = _mm256_setr_epi32(
                (int)IQ3S_GRID_HOST[qs[0] | ((qh << 8) & 256)], (int)IQ3S_GRID_HOST[qs[1] | ((qh << 7) & 256)],
                (int)IQ3S_GRID_HOST[qs[2] | ((qh << 6) & 256)], (int)IQ3S_GRID_HOST[qs[3] | ((qh << 5) & 256)],
                (int)IQ3S_GRID_HOST[qs[4] | ((qh << 4) & 256)], (int)IQ3S_GRID_HOST[qs[5] | ((qh << 3) & 256)],
                (int)IQ3S_GRID_HOST[qs[6] | ((qh << 2) & 256)], (int)IQ3S_GRID_HOST[qs[7] | ((qh << 1) & 256)]);
            uint32_t s32;
            std::memcpy(&s32, b + 74 + j * 4, 4);
            const __m256i sb8 = _mm256_shuffle_epi8(_mm256_set1_epi32((int)s32), C.spread);
            const __m256i neg = _mm256_cmpeq_epi8(_mm256_and_si256(sb8, C.bits), C.bits);
            const __m256i w = _mm256_sign_epi8(mag, _mm256_or_si256(neg, C.one8));
            v[j] = dot8(w, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + j * 32)));
        }
    }
};

template <> struct Block<KIND_IQ4_XS> {
    static constexpr int BYTES = 136;
    static void run(const uint8_t* b, const int8_t* xq, __m256i (&v)[8], float (&sc)[8]) {
        const float d = half_to_float(b);
        const uint32_t scales_h = (uint32_t)b[2] | ((uint32_t)b[3] << 8);
        for (int j = 0; j < 8; ++j) {
            const int low = (b[4 + (j >> 1)] >> (4 * (j & 1))) & 15;
            const int ls = low | (((scales_h >> (2 * j)) & 3) << 4);
            sc[j] = d * (float)(ls - 32);
            const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + 8 + j * 16));
            const __m128i lo = _mm_shuffle_epi8(C.table, _mm_and_si128(q, C.low4));
            const __m128i hi = _mm_shuffle_epi8(C.table, _mm_and_si128(_mm_srli_epi16(q, 4), C.low4));
            v[j] = dot8(_mm256_set_m128i(hi, lo), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + j * 32)));
        }
    }
};

template <> struct Block<KIND_Q6_K> {
    static constexpr int BYTES = 210;
    static void run(const uint8_t* b, const int8_t* xq, __m256i (&v)[8], float (&sc)[8]) {
        const float d = half_to_float(b + 208);
        for (int j = 0; j < 8; ++j) {
            const int n = j >> 2, m = j & 3;
            sc[j] = d;
            const __m256i ql = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 64 * n + (m & 1) * 32));
            const __m256i qh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 128 + 32 * n));
            const __m256i nib = (m >> 1) ? _mm256_and_si256(_mm256_srli_epi16(ql, 4), C.low4_256)
                                         : _mm256_and_si256(ql, C.low4_256);
            const __m256i h2 = _mm256_and_si256(_mm256_srli_epi16(qh, 2 * m), C.two);
            const __m256i w = _mm256_sub_epi8(_mm256_or_si256(nib, _mm256_slli_epi16(h2, 4)), C.bias);
            const int sc0 = (int)(int8_t)b[192 + 8 * n + 2 * m], sc1 = (int)(int8_t)b[193 + 8 * n + 2 * m];
            v[j] = _mm256_mullo_epi32(dot8(w, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + j * 32))),
                                      _mm256_setr_epi32(sc0, sc0, sc0, sc0, sc1, sc1, sc1, sc1));
        }
    }
};

// the int32 totals of eight units' partial vectors, in unit order (integer sums: exact in any order)
inline __m256i unit_totals(const __m256i (&v)[8]) {
    const __m256i h1 = _mm256_hadd_epi32(v[0], v[1]), h2 = _mm256_hadd_epi32(v[2], v[3]);
    const __m256i h3 = _mm256_hadd_epi32(v[4], v[5]), h4 = _mm256_hadd_epi32(v[6], v[7]);
    const __m256i g1 = _mm256_hadd_epi32(h1, h2), g2 = _mm256_hadd_epi32(h3, h4);
    return _mm256_add_epi32(_mm256_permute2x128_si256(g1, g2, 0x20), _mm256_permute2x128_si256(g1, g2, 0x31));
}

// the device's warp butterfly (v += shfl_xor(v, off), off 16, 8, 4, 2, 1) over 32 lanes held as four vectors; lane 0
inline float butterfly(__m256 a0, __m256 a1, __m256 a2, __m256 a3) {
    const __m256 b0 = _mm256_add_ps(a0, a2), b1 = _mm256_add_ps(a1, a3);              // off 16 (lanes 0..15)
    const __m256 c = _mm256_add_ps(b0, b1);                                            // off 8 (lanes 0..7)
    const __m256 d = _mm256_add_ps(c, _mm256_permute2f128_ps(c, c, 1));                // off 4
    const __m256 e = _mm256_add_ps(d, _mm256_shuffle_ps(d, d, _MM_SHUFFLE(1, 0, 3, 2)));   // off 2
    const __m256 f = _mm256_add_ps(e, _mm256_shuffle_ps(e, e, _MM_SHUFFLE(2, 3, 0, 1)));   // off 1
    return _mm256_cvtss_f32(f);
}

// One row of a matrix (row `n` of it) times one activation (K values, int8 units xq, scales xd), bit for bit as the
// device's gathered GEMV (gemvq_kernel) computes it: rows of more than 32 units (G = 1) lane l takes units l, l + 32,
// ... with fmaf((float)isum, wscale * xd, acc); shorter rows run in strips of 4 rows (G = 4), unit uu of the strip on
// lane uu % 32 added as (float)isum * (wscale * xd); then the butterfly. A superblock (8 units) at a time: the units'
// integer totals by hadd, the eight lanes' FMAs (or products) as one vector operation, which rounds as eight scalar ones.
template <int KIND>
float row_dot_t(const uint8_t* row, int K, const int8_t* xq, const float* xd, int n) {
    const int units = K / 32;
    __m256i v[8];
    alignas(32) float sc[8];
    if (units > 32) {
        __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (int u0 = 0; u0 < units; u0 += 8) {
            Block<KIND>::run(row + (u0 >> 3) * Block<KIND>::BYTES, xq + u0 * 32, v, sc);
            const __m256 s = _mm256_cvtepi32_ps(unit_totals(v));
            const __m256 p = _mm256_mul_ps(_mm256_load_ps(sc), _mm256_loadu_ps(xd + u0));
            const int a = (u0 >> 3) & 3;
            acc[a] = _mm256_fmadd_ps(s, p, acc[a]);
        }
        return butterfly(acc[0], acc[1], acc[2], acc[3]);
    }
    alignas(32) float lanes[32] = {};
    const int g = n & 3;
    for (int u0 = 0; u0 < units; u0 += 8) {
        Block<KIND>::run(row + (u0 >> 3) * Block<KIND>::BYTES, xq + u0 * 32, v, sc);
        alignas(32) float prod[8];
        const __m256 p = _mm256_mul_ps(_mm256_load_ps(sc), _mm256_loadu_ps(xd + u0));
        _mm256_store_ps(prod, _mm256_mul_ps(_mm256_cvtepi32_ps(unit_totals(v)), p));
        for (int i = 0; i < 8; ++i) {
            const int lane = (g * units + u0 + i) & 31;
            lanes[lane] = lanes[lane] + prod[i];
        }
    }
    return butterfly(_mm256_load_ps(lanes), _mm256_load_ps(lanes + 8), _mm256_load_ps(lanes + 16),
                     _mm256_load_ps(lanes + 24));
}

float row_dot(int kind, const uint8_t* row, int K, const int8_t* xq, const float* xd, int n) {
    if (kind == KIND_IQ3_S) return row_dot_t<KIND_IQ3_S>(row, K, xq, xd, n);
    if (kind == KIND_IQ4_XS) return row_dot_t<KIND_IQ4_XS>(row, K, xq, xd, n);
    return row_dot_t<KIND_Q6_K>(row, K, xq, xd, n);
}

struct LayerTable {
    const uint8_t* gu = nullptr;     // host gate|up rows [host experts][2 * inter][gu_rb]
    const uint8_t* dn = nullptr;     // host down rows [host experts][hidden][dn_rb]
    int64_t gu_rb = 0, dn_rb = 0;
    int gu_kind = 0, dn_kind = 0;
};

class Pool {
public:
    explicit Pool(int workers) {
        for (int i = 0; i < workers; ++i) threads_.emplace_back([this] { work(); });
    }
    ~Pool() {
        stop_.store(true);
        for (auto& t : threads_) t.join();
    }
    // run fn(i) for i in [0, n) on the pool and the caller, chunk at a time
    template <typename F>
    void run(int n, int chunk, F&& fn) {
        task_ = [&fn](int i) { fn(i); };
        n_ = n;
        chunk_ = chunk;
        next_.store(0);
        left_.store((int)threads_.size());
        epoch_.fetch_add(1, std::memory_order_release);
        drain();
        while (left_.load(std::memory_order_acquire) > 0) _mm_pause();
    }
    int size() const { return (int)threads_.size() + 1; }

private:
    void drain() {
        for (;;) {
            const int a = next_.fetch_add(chunk_);
            if (a >= n_) break;
            const int b = std::min(n_, a + chunk_);
            for (int i = a; i < b; ++i) task_(i);
        }
    }
    void work() {
        uint64_t seen = 0;          // the epoch before any job (a worker that starts late still runs the first)
        int idle = 0;
        while (!stop_.load(std::memory_order_relaxed)) {
            const uint64_t now = epoch_.load(std::memory_order_acquire);
            if (now != seen) {
                seen = now;
                drain();
                left_.fetch_sub(1, std::memory_order_acq_rel);
                idle = 0;
                continue;
            }
            if (++idle < (1 << 22)) _mm_pause();
            else std::this_thread::yield();
        }
    }
    std::vector<std::thread> threads_;
    std::function<void(int)> task_;
    std::atomic<uint64_t> epoch_{0};
    std::atomic<int> next_{0}, left_{0};
    std::atomic<bool> stop_{false};
    int n_ = 0, chunk_ = 1;
};

class Tier {
public:
    Tier(int layers, int hidden, int inter, int top, int workers)
        : layout_(hidden, top), layers_(layers), hidden_(hidden), inter_(inter), tables_(layers) {
        void* p = nullptr;
        C10_CUDA_CHECK(cudaHostAlloc(&p, layout_.bytes * layers, cudaHostAllocMapped | cudaHostAllocPortable));
        std::memset(p, 0, layout_.bytes * layers);
        host_ = reinterpret_cast<uint8_t*>(p);
        void* d = nullptr;
        C10_CUDA_CHECK(cudaHostGetDevicePointer(&d, p, 0));
        device_ = reinterpret_cast<uint8_t*>(d);
        pool_ = std::make_unique<Pool>(std::max(0, workers - 1));
        xq_.resize((size_t)MAX_TOKENS * hidden);
        xd_.resize((size_t)MAX_TOKENS * hidden / 32);
        h_.resize((size_t)layout_.max_routes * 2 * inter);
        aq_.resize((size_t)layout_.max_routes * inter);
        ad_.resize((size_t)layout_.max_routes * inter / 32);
        service_ = std::thread([this] { serve(); });
    }
    ~Tier() { close(); }
    // the service thread and the worker pool stopped and joined, the mailbox freed: at once, not when Python
    // collects the object (idle threads spin, and a closed engine's would take the next one's cores)
    void close() {
        stop_.store(true);
        if (service_.joinable()) service_.join();
        pool_.reset();
        if (host_ != nullptr) {
            cudaFreeHost(host_);
            host_ = nullptr;
            device_ = nullptr;
        }
    }
    void set_layer(int layer, int64_t gu, int64_t dn, int64_t gu_rb, int64_t dn_rb, int gu_kind, int dn_kind) {
        TORCH_CHECK(layer >= 0 && layer < layers_);
        TORCH_CHECK(gu_kind == KIND_IQ3_S || gu_kind == KIND_IQ4_XS || gu_kind == KIND_Q6_K);
        TORCH_CHECK(dn_kind == KIND_IQ3_S || dn_kind == KIND_IQ4_XS || dn_kind == KIND_Q6_K);
        tables_[layer] = {reinterpret_cast<const uint8_t*>(gu), reinterpret_cast<const uint8_t*>(dn), gu_rb, dn_rb,
                          gu_kind, dn_kind};
    }
    int64_t slot_device(int layer) const { return reinterpret_cast<int64_t>(device_ + layout_.bytes * layer); }
    const Layout& layout() const { return layout_; }
    std::vector<int64_t> counters() const { return {served_.load(), routes_.load(), compute_ns_.load()}; }

private:
    void serve() {
        int idle = 0;
        while (!stop_.load(std::memory_order_relaxed)) {
            bool any = false;
            for (int l = 0; l < layers_; ++l) {
                SlotHeader* head = reinterpret_cast<SlotHeader*>(host_ + layout_.bytes * l);
                if (head->ready == 1) {
                    std::atomic_thread_fence(std::memory_order_acquire);
                    head->ready = 0;
                    const auto t0 = std::chrono::steady_clock::now();
                    compute(l);
                    compute_ns_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - t0).count());
                    std::atomic_thread_fence(std::memory_order_release);
                    _mm_sfence();
                    head->done = 1;
                    any = true;
                }
            }
            if (any) {
                idle = 0;
            } else if (++idle < (1 << 22)) {
                _mm_pause();
            } else {
                std::this_thread::yield();
            }
        }
    }

    void compute(int l) {
        uint8_t* slot = host_ + layout_.bytes * l;
        const SlotHeader* head = reinterpret_cast<const SlotHeader*>(slot);
        const int R = head->routes, T = head->tokens;
        const int32_t* tok = reinterpret_cast<const int32_t*>(slot + layout_.tok_off);
        const int32_t* idx = reinterpret_cast<const int32_t*>(slot + layout_.idx_off);
        const float* xs = reinterpret_cast<const float*>(slot + layout_.x_off);
        float* out = reinterpret_cast<float*>(slot + layout_.out_off);
        const LayerTable& L = tables_[l];
        const int H = hidden_, I = inter_;
        for (int t = 0; t < T; ++t) quantize_units(xs + (size_t)t * H, H, xq_.data() + (size_t)t * H,
                                                   xd_.data() + (size_t)t * H / 32);
        // gate|up rows of every route
        pool_->run(R * 2 * I, 16, [&](int i) {
            const int r = i / (2 * I), row = i - r * 2 * I;
            const uint8_t* w = L.gu + ((int64_t)idx[r] * 2 * I + row) * L.gu_rb;
            h_[(size_t)r * 2 * I + row] = row_dot(L.gu_kind, w, H, xq_.data() + (size_t)tok[r] * H,
                                                  xd_.data() + (size_t)tok[r] * H / 32, row);
        });
        for (int r = 0; r < R; ++r) {
            float act[4096];
            const float* g = h_.data() + (size_t)r * 2 * I;
            for (int i = 0; i < I; ++i) act[i] = crit_swiglu(g[i], g[I + i]);   // the device's swiglu_kernel
            // the activation rounds to BF16 first, as the device's act tensor does
            for (int i = 0; i < I; ++i) {
                uint32_t bitsv;
                std::memcpy(&bitsv, &act[i], 4);
                bitsv = (bitsv + 0x7FFF + ((bitsv >> 16) & 1)) & 0xFFFF0000u;
                std::memcpy(&act[i], &bitsv, 4);
            }
            quantize_units(act, I, aq_.data() + (size_t)r * I, ad_.data() + (size_t)r * I / 32);
        }
        pool_->run(R * H, 32, [&](int i) {
            const int r = i / H, row = i - r * H;
            const uint8_t* w = L.dn + ((int64_t)idx[r] * H + row) * L.dn_rb;
            out[(size_t)r * H + row] = row_dot(L.dn_kind, w, I, aq_.data() + (size_t)r * I,
                                               ad_.data() + (size_t)r * I / 32, row);
        });
        served_.fetch_add(1);
        routes_.fetch_add(R);
    }

    Layout layout_;
    int layers_, hidden_, inter_;
    std::vector<LayerTable> tables_;
    uint8_t* host_ = nullptr;
    uint8_t* device_ = nullptr;
    std::unique_ptr<Pool> pool_;
    std::vector<int8_t> xq_, aq_;
    std::vector<float> xd_, h_, ad_;
    std::thread service_;
    std::atomic<bool> stop_{false};
    std::atomic<int64_t> served_{0}, routes_{0}, compute_ns_{0};
};

}  // namespace

// post: the layer's host routes and token rows into its slot; pos [T * top] int32 (device) out
void post(int64_t slot, torch::Tensor ids, torch::Tensor codes, torch::Tensor x, torch::Tensor pos, int64_t top,
          int64_t tok_off, int64_t idx_off, int64_t x_off) {
    TORCH_CHECK(ids.is_cuda() && codes.is_cuda() && x.is_cuda() && pos.is_cuda());
    TORCH_CHECK(ids.scalar_type() == torch::kInt64 && codes.scalar_type() == torch::kInt32 &&
                x.scalar_type() == torch::kBFloat16 && pos.scalar_type() == torch::kInt32);
    TORCH_CHECK(x.is_contiguous() && ids.is_contiguous() && x.size(0) <= MAX_TOKENS);
    const int T = (int)x.size(0), H = (int)x.size(1);
    post_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<uint8_t*>(slot), ids.data_ptr<int64_t>(), codes.data_ptr<int32_t>(),
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), T, (int)top, H, pos.data_ptr<int32_t>(),
        (size_t)tok_off, (size_t)idx_off, (size_t)x_off);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wait(int64_t slot) {
    wait_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<uint8_t*>(slot));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor combine(torch::Tensor y, int64_t slot, int64_t out_off, torch::Tensor pos, torch::Tensor ids,
                      torch::Tensor w, int64_t top) {
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32 && y.is_contiguous());
    TORCH_CHECK(w.scalar_type() == torch::kFloat32 && w.is_contiguous() && ids.is_contiguous() && pos.is_contiguous());
    const int H = (int)y.size(1), T = (int)(ids.numel() / top);
    auto out = torch::empty({T, H}, y.options());
    if (T == 0) return out;
    combine_kernel<<<dim3(T, (H + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        y.data_ptr<float>(), reinterpret_cast<const uint8_t*>(slot), (size_t)out_off, pos.data_ptr<int32_t>(),
        ids.data_ptr<int64_t>(), w.data_ptr<float>(), (int)top, H, out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// the host dot product of one row with one activation (tests): kind 2 IQ4_XS, 3 IQ3_S, 5 Q6_K
double host_row(torch::Tensor row, int64_t kind, torch::Tensor x) {
    TORCH_CHECK(row.scalar_type() == torch::kUInt8 && x.scalar_type() == torch::kFloat32 && !row.is_cuda());
    const int K = (int)x.numel();
    std::vector<int8_t> q(K);
    std::vector<float> d(K / 32);
    quantize_units(x.contiguous().data_ptr<float>(), K, q.data(), d.data());
    return row_dot((int)kind, row.contiguous().data_ptr<uint8_t>(), K, q.data(), d.data(), 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Tier>(m, "Tier")
        .def(py::init<int, int, int, int, int>(), py::arg("layers"), py::arg("hidden"), py::arg("inter"),
             py::arg("top"), py::arg("workers"))
        .def("set_layer", &Tier::set_layer)
        .def("slot_device", &Tier::slot_device)
        .def("offsets", [](const Tier& t) {
            const Layout& l = t.layout();
            return std::vector<int64_t>{(int64_t)l.tok_off, (int64_t)l.idx_off, (int64_t)l.x_off, (int64_t)l.out_off,
                                        (int64_t)l.bytes};
        })
        .def("counters", &Tier::counters)
        .def("close", &Tier::close);
    m.def("post", &post, "post a layer's host routes into its mailbox slot (device)");
    m.def("wait", &wait, "wait for the host's answer on a slot and reset it (device)");
    m.def("combine", &combine, "each token's routes in route order, host routes from the slot");
    m.def("host_row", &host_row, "one host row dot product (tests)");
}
