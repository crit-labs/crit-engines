// NVFP4 W4A4 GEMV on tensor cores for decode and verify (1-8 columns): out[m][n] = alpha * sum_k x[m][k] * w[n][k],
// x and w E2M1 codes (two per byte, low nibble first) times e4m3 group-16 scales in cuBLASLt's 128x4 tile order.
// Weights are decoded in registers to fp16 (a code times a scale: at most 6 significant bits, exact) and multiplied by
// the columns (decoded once to fp16, exact likewise) with mma.m16n8k16 into FP32: every product is exact. A warp
// takes 16 weight rows; a lane of quad position t streams the 32 values [32t, 32t + 32) of each 128-value slice of
// its two rows, which gives the k slots of the MMA fragments an order the columns' fragments follow, so the sum is the
// dot product. The instruction stream does not depend on the column count: a column gets the same bits alone or in a
// verify. K is split in a fixed order into SPLIT parts reduced without atomics.
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace {

constexpr int WARPS = 2;
constexpr int COLS = 8;

__device__ __forceinline__ int scale_offset(int row, int group, int n_cb) {
    int rb = row >> 7, rr = row & 127, cb = group >> 2, cc = group & 3;
    return (rb * n_cb + cb) * 512 + (rr & 31) * 16 + (rr >> 5) * 4 + cc;
}

__device__ __forceinline__ float e2m1(unsigned c) {
    unsigned e = (c >> 1) & 3u, f = c & 1u;
    unsigned bits = e ? ((e + 126u) << 23) | (f << 22) : (f ? (126u << 23) : 0u);
    return __uint_as_float(bits | ((c & 8u) << 28));
}

__device__ __forceinline__ __half e4m3h(unsigned char b) {
    __nv_fp8_e4m3 v;
    v.__x = b;
    return __float2half(float(v));
}

// One byte (two codes) times its group scale as an fp16 pair (low nibble first): the hardware E2M1 conversion
// (cvt.rn.f16x2.e2m1x2, sm_120a) then one half2 multiply.
__device__ __forceinline__ unsigned pair(unsigned byte, __half s) {
    unsigned v;
    asm("{ .reg .b8 t0, t1, t2, t3; mov.b32 {t0, t1, t2, t3}, %1; cvt.rn.f16x2.e2m1x2 %0, t0; }"
        : "=r"(v) : "r"(byte));
    __half2 h = *reinterpret_cast<__half2*>(&v);
    h = __hmul2(h, __half2half2(s));
    return *reinterpret_cast<unsigned*>(&h);
}

__global__ void decode_kernel(const unsigned char* __restrict__ x, const unsigned char* __restrict__ xs,
                              __half* __restrict__ xv, int K, int M) {
    const int m = blockIdx.y;
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= K / 2) return;
    __half2 v = __floats2half2_rn(0.f, 0.f);
    if (m < M) {
        unsigned char byte = x[(size_t)m * (K / 2) + b];
        __nv_fp8_e4m3 sc;
        sc.__x = xs[scale_offset(m, (2 * b) >> 4, K / 64)];
        float s = float(sc);
        v = __floats2half2_rn(e2m1(byte & 15) * s, e2m1(byte >> 4) * s);
    }
    reinterpret_cast<__half2*>(xv)[(size_t)m * (K / 2) + b] = v;
}

__global__ void __launch_bounds__(WARPS * 32) mma_kernel(const __half* __restrict__ xv, const unsigned char* __restrict__ w,
                                                         const unsigned char* __restrict__ ws, float* __restrict__ part,
                                                         int N, int K, int split) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gid = lane >> 2, tig = lane & 3;
    const int r0 = (blockIdx.x * WARPS + warp) * 16;
    const int ra = min(r0 + gid, N - 1), rb = min(r0 + gid + 8, N - 1);
    const int bytes = K / 2, n_cb = K / 64;
    const int per = K / split, k_lo = blockIdx.y * per, k_hi = k_lo + per;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
    for (int k0 = k_lo; k0 < k_hi; k0 += 128) {
        const int kb = k0 + 32 * tig;                 // this lane's 32 values
        uint4 wa = __ldg(reinterpret_cast<const uint4*>(w + (size_t)ra * bytes + kb / 2));
        uint4 wb = __ldg(reinterpret_cast<const uint4*>(w + (size_t)rb * bytes + kb / 2));
        const int g = kb >> 4;
        __half sa0 = e4m3h(ws[scale_offset(ra, g, n_cb)]), sa1 = e4m3h(ws[scale_offset(ra, g + 1, n_cb)]);
        __half sb0 = e4m3h(ws[scale_offset(rb, g, n_cb)]), sb1 = e4m3h(ws[scale_offset(rb, g + 1, n_cb)]);
        const uint4* xp = reinterpret_cast<const uint4*>(xv + (size_t)gid * K + kb);
        uint4 x0 = __ldg(xp), x1 = __ldg(xp + 1), x2 = __ldg(xp + 2), x3 = __ldg(xp + 3);
        const unsigned* pa = reinterpret_cast<const unsigned*>(&wa);
        const unsigned* pb = reinterpret_cast<const unsigned*>(&wb);
        const unsigned xw[16] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w,
                                 x2.x, x2.y, x2.z, x2.w, x3.x, x3.y, x3.z, x3.w};
#pragma unroll
        for (int s = 0; s < 8; ++s) {                 // 8 MMAs of 4 values per lane and row
            const unsigned wordA = pa[s >> 1], wordB = pb[s >> 1];
            const int sh = (s & 1) * 16;
            const __half sA = s < 4 ? sa0 : sa1, sB = s < 4 ? sb0 : sb1;
            unsigned a0 = pair((wordA >> sh) & 0xffu, sA);
            unsigned a1 = pair((wordB >> sh) & 0xffu, sB);
            unsigned a2 = pair((wordA >> (sh + 8)) & 0xffu, sA);
            unsigned a3 = pair((wordB >> (sh + 8)) & 0xffu, sB);
            unsigned b0 = xw[2 * s], b1 = xw[2 * s + 1];
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    // c0, c1: row gid, columns 2 tig, 2 tig + 1; c2, c3: row gid + 8
    float* p = part + (size_t)blockIdx.y * COLS * N;
    const int col = 2 * tig;
    if (r0 + gid < N) {
        p[(size_t)col * N + r0 + gid] = c[0];
        p[(size_t)(col + 1) * N + r0 + gid] = c[1];
    }
    if (r0 + gid + 8 < N) {
        p[(size_t)col * N + r0 + gid + 8] = c[2];
        p[(size_t)(col + 1) * N + r0 + gid + 8] = c[3];
    }
}


// Wide variant: a lane streams 128 contiguous values (64 bytes) of each of its two rows per step, so each row is read
// in 256-byte runs; same exact products and fixed order.
__global__ void __launch_bounds__(WARPS * 32) mma_wide_kernel(const __half* __restrict__ xv,
                                                              const unsigned char* __restrict__ w,
                                                              const unsigned char* __restrict__ ws, float* __restrict__ part,
                                                              int N, int K, int split) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gid = lane >> 2, tig = lane & 3;
    const int r0 = (blockIdx.x * WARPS + warp) * 16;
    const int ra = min(r0 + gid, N - 1), rb = min(r0 + gid + 8, N - 1);
    const int bytes = K / 2, n_cb = K / 64;
    const int per = K / split, k_lo = blockIdx.y * per, k_hi = k_lo + per;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k0 = k_lo; k0 < k_hi; k0 += 512) {
        const int kb = k0 + 128 * tig;
        const uint4* wpa = reinterpret_cast<const uint4*>(w + (size_t)ra * bytes + kb / 2);
        const uint4* wpb = reinterpret_cast<const uint4*>(w + (size_t)rb * bytes + kb / 2);
        uint4 wa[4], wb[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { wa[i] = __ldg(wpa + i); wb[i] = __ldg(wpb + i); }
        const int g = kb >> 4;   // 8 groups: two quads of the tile order
        unsigned qa0 = *reinterpret_cast<const unsigned*>(ws + scale_offset(ra, g, n_cb));
        unsigned qa1 = *reinterpret_cast<const unsigned*>(ws + scale_offset(ra, g + 4, n_cb));
        unsigned qb0 = *reinterpret_cast<const unsigned*>(ws + scale_offset(rb, g, n_cb));
        unsigned qb1 = *reinterpret_cast<const unsigned*>(ws + scale_offset(rb, g + 4, n_cb));
        const uint4* xp = reinterpret_cast<const uint4*>(xv + (size_t)gid * K + kb);
#pragma unroll
        for (int q = 0; q < 4; ++q) {          // 16 bytes = 32 values = 2 groups per row
            uint4 x0 = __ldg(xp + 4 * q), x1 = __ldg(xp + 4 * q + 1), x2 = __ldg(xp + 4 * q + 2), x3 = __ldg(xp + 4 * q + 3);
            const unsigned xw[16] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w,
                                     x2.x, x2.y, x2.z, x2.w, x3.x, x3.y, x3.z, x3.w};
            const unsigned* pa = reinterpret_cast<const unsigned*>(&wa[q]);
            const unsigned* pb = reinterpret_cast<const unsigned*>(&wb[q]);
            const unsigned qa = q < 2 ? qa0 : qa1, qb = q < 2 ? qb0 : qb1;
            const int gi = (q & 1) * 2;        // group index within the quad
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const int sg = gi + (s >> 2);
                const __half sA = e4m3h((qa >> (8 * sg)) & 0xffu), sB = e4m3h((qb >> (8 * sg)) & 0xffu);
                const unsigned wordA = pa[s >> 1], wordB = pb[s >> 1];
                const int sh = (s & 1) * 16;
                unsigned a0 = pair((wordA >> sh) & 0xffu, sA);
                unsigned a1 = pair((wordB >> sh) & 0xffu, sB);
                unsigned a2 = pair((wordA >> (sh + 8)) & 0xffu, sA);
                unsigned a3 = pair((wordB >> (sh + 8)) & 0xffu, sB);
                unsigned b0 = xw[2 * s], b1 = xw[2 * s + 1];
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
    }
    float* p = part + (size_t)blockIdx.y * COLS * N;
    const int col = 2 * tig;
    if (r0 + gid < N) {
        p[(size_t)col * N + r0 + gid] = c[0];
        p[(size_t)(col + 1) * N + r0 + gid] = c[1];
    }
    if (r0 + gid + 8 < N) {
        p[(size_t)col * N + r0 + gid + 8] = c[2];
        p[(size_t)(col + 1) * N + r0 + gid + 8] = c[3];
    }
}

__global__ void reduce_kernel(const float* __restrict__ part, const float* __restrict__ alpha,
                              __nv_bfloat16* __restrict__ out, int N, int M, int split, int per_row) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;
    float v = 0.f;
    for (int s = 0; s < split; ++s) v += part[((size_t)s * COLS + m) * N + n];
    out[(size_t)m * N + n] = __float2bfloat16(v * alpha[per_row ? m : 0]);
}

}  // namespace

torch::Tensor nvfp4_mma(torch::Tensor x, torch::Tensor xs, torch::Tensor w, torch::Tensor ws, torch::Tensor alpha,
                        int64_t split, bool wide, bool per_row) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == torch::kUInt8, "x: packed uint8 [M, K/2]");
    const int M = x.size(0), K = x.size(1) * 2, N = w.size(0);
    TORCH_CHECK(M >= 1 && M <= COLS, "1-8 columns");
    TORCH_CHECK(w.size(1) * 2 == K && K % ((wide ? 512 : 128) * split) == 0, "K must split into whole steps");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto xv = torch::empty({COLS, K}, x.options().dtype(torch::kHalf));
    decode_kernel<<<dim3((K / 2 + 255) / 256, COLS), 256, 0, stream>>>(x.data_ptr<unsigned char>(),
        reinterpret_cast<unsigned char*>(xs.data_ptr()), reinterpret_cast<__half*>(xv.data_ptr()), K, M);
    auto part = torch::empty({split, COLS, N}, x.options().dtype(torch::kFloat));
    dim3 grid((N + WARPS * 16 - 1) / (WARPS * 16), split);
    auto kernel = wide ? mma_wide_kernel : mma_kernel;
    kernel<<<grid, WARPS * 32, 0, stream>>>(reinterpret_cast<const __half*>(xv.data_ptr()), w.data_ptr<unsigned char>(),
        reinterpret_cast<unsigned char*>(ws.data_ptr()), part.data_ptr<float>(), N, K, split);
    auto out = torch::empty({M, N}, x.options().dtype(torch::kBFloat16));
    reduce_kernel<<<dim3((N + 255) / 256, M), 256, 0, stream>>>(part.data_ptr<float>(), alpha.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), N, M, split, per_row ? 1 : 0);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("nvfp4_mma", &nvfp4_mma); }
