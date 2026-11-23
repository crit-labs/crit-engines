// Decode GEMVs straight from GGUF blocks (ggml block layouts, ggml-common.h, MIT License, Copyright (c) 2023-2026
// The ggml authors): y[r][n] = sum_k W[n][k] x[r][k] for up to 8 activation rows, W as stored (Q8_0, IQ4_NL, IQ4_XS,
// IQ3_S), each weight decoded to the FP32 value gguf-py's dequantize gives and multiplied in FP32. One warp owns one
// output row: lane l sums the 32-element units l, l + 32, ... in order, then a fixed butterfly reduction, so every
// activation row's result is its own whatever rows share the call. gathered_gemv does the same for (route, expert)
// pairs: each pair's row of x times its expert's matrix.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <algorithm>

#include "iq3s_grid.h"

namespace {

constexpr int MAX_ROWS = 8;
constexpr int WARPS = 8;
constexpr int STAGED_ROW_BYTES = 1024;   // a row up to this size is staged in shared memory before it is decoded

__constant__ int8_t KVALUES[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

__device__ __forceinline__ float half_at(const uint8_t* p) {
    uint16_t bits = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    return __half2float(__ushort_as_half(bits));
}

// The 32 FP32 values of unit u of a row (a Q8_0 / IQ4_NL block, or a 32-wide sub-block of an IQ4_XS / IQ3_S
// superblock), multiplied with x rows and accumulated: acc[r] += sum_i w_i x[r][u * 32 + i].
template <int KIND>
__device__ __forceinline__ void unit_values(const uint8_t* row, int u, float* w, const uint32_t* grid,
                                            const float* kv) {
    if (KIND == 0) {            // Q8_0: d, 32 x int8
        const uint8_t* b = row + u * 34;
        float d = half_at(b);
        #pragma unroll
        for (int i = 0; i < 32; ++i) w[i] = d * (float)(int8_t)b[2 + i];
    } else if (KIND == 1) {     // IQ4_NL: d, 16 bytes of nibbles (low: 0..15, high: 16..31)
        const uint8_t* b = row + u * 18;
        float d = half_at(b);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t q = b[2 + i];
            w[i] = d * kv[q & 15];
            w[i + 16] = d * kv[q >> 4];
        }
    } else if (KIND == 2) {     // IQ4_XS: d, scales_h (u16), scales_l[4], qs[128]; unit j of superblock s
        const uint8_t* b = row + (u >> 3) * 136;
        int j = u & 7;
        float d = half_at(b);
        uint32_t scales_h = (uint32_t)b[2] | ((uint32_t)b[3] << 8);
        int low = (b[4 + (j >> 1)] >> (4 * (j & 1))) & 15;
        int ls = low | (((scales_h >> (2 * j)) & 3) << 4);
        float dl = d * (float)(ls - 32);
        const uint8_t* qs = b + 8 + j * 16;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t q = qs[i];
            w[i] = dl * kv[q & 15];
            w[i + 16] = dl * kv[q >> 4];
        }
    } else {                    // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]; unit j of superblock s
        const uint8_t* b = row + (u >> 3) * 110;
        int j = u & 7;
        float d = half_at(b);
        int nib = (b[106 + (j >> 1)] >> (4 * (j & 1))) & 15;
        float db = d * (float)(1 + 2 * nib);
        uint8_t qh = b[66 + j];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint32_t index = (uint32_t)b[2 + j * 8 + k] | (((uint32_t)(qh >> k) & 1u) << 8);
            uint32_t g = grid[index];
            uint8_t sign = b[74 + j * 4 + (k >> 1)];
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                int e = k * 4 + c;                      // element of the unit
                float mag = (float)((g >> (8 * c)) & 255u);
                bool neg = (sign >> (e & 7)) & 1;
                w[e] = db * (neg ? -mag : mag);
            }
        }
    }
}

template <int KIND>
__device__ __forceinline__ void unit_dot(const uint8_t* row, int u, const __nv_bfloat16* x, int K, int R,
                                         float* acc, const uint32_t* grid, const float* kv) {
    float w[32];
    unit_values<KIND>(row, u, w, grid, kv);
    const int base = u * 32;
    for (int r = 0; r < R; ++r) {
        // the unit's 32 activations as four 16-byte loads (a unit starts 64 bytes into a row of a multiple of 32)
        const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)r * K + base);
        float s = 0.f;
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            uint4 packed = __ldg(xr + v);
            const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f = __bfloat1622float2(pairs[j]);
                s = fmaf(w[v * 8 + 2 * j], f.x, s);
                s = fmaf(w[v * 8 + 2 * j + 1], f.y, s);
            }
        }
        acc[r] += s;
    }
}

template <int KIND>
__global__ void gemv_kernel(const uint8_t* __restrict__ raw, int64_t row_bytes, const __nv_bfloat16* __restrict__ x,
                            int K, int R, int N, float* __restrict__ y, int64_t expert_stride,
                            const int32_t* __restrict__ experts, const int32_t* __restrict__ starts,
                            const int32_t* __restrict__ counts, int x_div) {
    __shared__ uint32_t grid[512];
    __shared__ float kv[16];
    if (KIND == 3) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) grid[i] = IQ3S_GRID[i];
    }
    if (KIND == 1 || KIND == 2) {
        if (threadIdx.x < 16) kv[threadIdx.x] = (float)KVALUES[threadIdx.x];
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blockIdx.x * WARPS + warp;
    const int pair = blockIdx.y;               // gathered: one activation row per (route) pair
    if (n >= N) return;
    const uint8_t* base = raw;
    const __nv_bfloat16* xs = x;
    int rows = R;
    int first = pair;                          // the first x row (and y row) of this block's rows
    if (experts != nullptr) {
        base += (int64_t)experts[pair] * expert_stride;
        if (starts != nullptr) {               // a group: up to MAX_ROWS consecutive rows routed to one expert
            first = starts[pair];
            rows = counts[pair];
            xs += (size_t)first * K;
        } else {
            rows = 1;
            xs += (size_t)(pair / x_div) * K;
        }
    }
    const uint8_t* row = base + (int64_t)n * row_bytes;
    // The row's blocks pass through shared memory a chunk at a time (whole blocks, at most STAGE bytes), copied with
    // the warp's 16-byte loads, all issued before any is stored (one memory round trip a chunk): the aligned 16-byte
    // words covering the chunk, so a few bytes either side of it come along (they stay inside the weight's own
    // allocation, whose size is rounded up to at least 16 bytes) and the chunk starts `skew` bytes into the copy. The
    // decode's scattered byte reads then hit shared memory instead of each waiting on its own global load.
    constexpr int STAGE = KIND == 0 ? 1024 : 2048;
    constexpr int STAGE_WORDS = STAGE / 16 + 2;
    constexpr int PER_LANE = (STAGE_WORDS + 31) / 32;
    __shared__ uint4 staged[WARPS][STAGE_WORDS];
    constexpr int BLOCK_BYTES = KIND == 0 ? 34 : KIND == 1 ? 18 : KIND == 2 ? 136 : 110;
    constexpr int BLOCK_UNITS = (KIND == 0 || KIND == 1) ? 1 : 8;
    constexpr int CHUNK_BLOCKS = STAGE / BLOCK_BYTES;
    float acc[MAX_ROWS];
    #pragma unroll
    for (int r = 0; r < MAX_ROWS; ++r) acc[r] = 0.f;
    const int units = K / 32;
    const int blocks = units / BLOCK_UNITS;
    for (int b0 = 0; b0 < blocks; b0 += CHUNK_BLOCKS) {
        const int nb = min(CHUNK_BLOCKS, blocks - b0);
        const uint8_t* start = row + (int64_t)b0 * BLOCK_BYTES;
        const int skew = (int)(reinterpret_cast<uintptr_t>(start) & 15);
        const uint4* src = reinterpret_cast<const uint4*>(start - skew);
        const int words = (skew + nb * BLOCK_BYTES + 15) >> 4;
        uint4 held[PER_LANE];
        #pragma unroll
        for (int j = 0; j < PER_LANE; ++j) {
            const int i = lane + 32 * j;
            if (i < words) held[j] = __ldg(src + i);
        }
        __syncwarp();
        #pragma unroll
        for (int j = 0; j < PER_LANE; ++j) {
            const int i = lane + 32 * j;
            if (i < words) staged[warp][i] = held[j];
        }
        __syncwarp();
        // unit_values indexes from the row's start: point it before the chunk so unit u lands in the staged copy
        const uint8_t* chunk = reinterpret_cast<const uint8_t*>(staged[warp]) + skew - (int64_t)b0 * BLOCK_BYTES;
        const int u_end = (b0 + nb) * BLOCK_UNITS;
        for (int u = b0 * BLOCK_UNITS + lane; u < u_end; u += 32) unit_dot<KIND>(chunk, u, xs, K, rows, acc, grid, kv);
    }
    for (int r = 0; r < rows; ++r) {
        float v = acc[r];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0) {
            if (experts != nullptr) y[(size_t)(first + r) * N + n] = v;
            else y[(size_t)r * N + n] = v;
        }
    }
}

// ---- stored-format activations: the GGUF formats' own int8 activation blocks ---------------------------------------
// An activation row is quantized in units of 32 values to int8 with one FP32 scale (d = max |x| / 127, q = round(x / d)),
// the activation form ggml multiplies these block formats with; a unit's weight codes then meet it in integer dot
// products (dp4a, exact int32 sums) and the unit's sum is scaled once: acc += isum * d_w * d_x. Each activation row is
// quantized alone, so a row's result is still its own whatever rows share the call.

// realigned 32-bit read of shared bytes at any address
__device__ __forceinline__ uint32_t ld32u(const uint8_t* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* w = reinterpret_cast<const uint32_t*>(a & ~(uintptr_t)3);
    return __funnelshift_r(w[0], w[1], (uint32_t)(a & 3) * 8);
}

// IQ4 code values as int8, four to a word, for byte permutes (KVALUES in order)
constexpr uint32_t KV0 = 0xBFAD9881u;   // -127 -104 -83 -65
constexpr uint32_t KV1 = 0xF6EADDCFu;   //  -49  -35 -22 -10
constexpr uint32_t KV2 = 0x26190D01u;   //    1   13  25  38
constexpr uint32_t KV3 = 0x71594535u;   //   53   69  89 113

// the IQ4 values of a word of nibble pairs: .x the four low nibbles' values (bytes 0..3), .y the high nibbles'
__device__ __forceinline__ int2 iq4_values(uint32_t q) {
    uint32_t r[2];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint32_t s = q >> (16 * h);
        const uint32_t lo = __byte_perm(KV0, KV1, s), hi = __byte_perm(KV2, KV3, s);
        r[h] = __byte_perm(lo, hi, 0x3210u | ((s & 0x8888u) >> 1));
    }
    return make_int2((int)__byte_perm(r[0], r[1], 0x6420u), (int)__byte_perm(r[0], r[1], 0x7531u));
}

__device__ __forceinline__ float half_u(const uint8_t* p) {
    return __half2float(__ushort_as_half((uint16_t)(ld32u(p) & 0xffffu)));
}

// the int32 dot product of unit u's weight codes with an activation unit (8 words of int8), and the weights' scale
template <int KIND>
__device__ __forceinline__ int unit_isum(const uint8_t* row, int u, const int* xq, float& scale, const uint32_t* grid) {
    int s = 0;
    if (KIND == 0) {                                    // Q8_0
        const uint8_t* b = row + u * 34;
        scale = half_u(b);
        #pragma unroll
        for (int i = 0; i < 8; ++i) s = __dp4a((int)ld32u(b + 2 + 4 * i), xq[i], s);
    } else if (KIND == 1 || KIND == 2) {                // IQ4_NL, IQ4_XS
        const uint8_t* qs;
        if (KIND == 1) {
            const uint8_t* b = row + u * 18;
            scale = half_u(b);
            qs = b + 2;
        } else {
            const uint8_t* b = row + (u >> 3) * 136;
            const int j = u & 7;
            const uint32_t head = ld32u(b);
            const uint32_t scales_h = head >> 16;
            const int low = (b[4 + (j >> 1)] >> (4 * (j & 1))) & 15;
            const int ls = low | (((scales_h >> (2 * j)) & 3) << 4);
            scale = __half2float(__ushort_as_half((uint16_t)(head & 0xffffu))) * (float)(ls - 32);
            qs = b + 8 + j * 16;
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int2 v = iq4_values(ld32u(qs + 4 * i));
            s = __dp4a(v.x, xq[i], s);
            s = __dp4a(v.y, xq[4 + i], s);
        }
    } else {                                            // IQ3_S
        const uint8_t* b = row + (u >> 3) * 110;
        const int j = u & 7;
        const int nib = (b[106 + (j >> 1)] >> (4 * (j & 1))) & 15;
        scale = half_u(b) * (float)(1 + 2 * nib);
        const uint32_t qh = b[66 + j];
        const uint32_t q0 = ld32u(b + 2 + j * 8), q1 = ld32u(b + 6 + j * 8);
        const uint32_t signs = ld32u(b + 74 + j * 4);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint32_t qb = ((k < 4 ? q0 : q1) >> (8 * (k & 3))) & 255u;
            const uint32_t g = grid[qb | (((qh >> k) & 1u) << 8)];
            const uint32_t spread = (((signs >> (4 * k)) & 15u) * 0x204081u) & 0x01010101u;   // bit c -> byte c
            const uint32_t w = (g ^ (spread * 255u)) + spread;                                 // negated where set
            s = __dp4a((int)w, xq[k], s);
        }
    }
    return s;
}

// xq [rows][K] int8 and xd [rows][K / 32] FP32 of x [rows][K] BF16: one warp a unit
__global__ void quantize_kernel(const __nv_bfloat16* __restrict__ x, int64_t units, int8_t* __restrict__ xq,
                                float* __restrict__ xd) {
    const int64_t unit = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (unit >= units) return;
    const int lane = threadIdx.x & 31;
    const float v = __bfloat162float(x[unit * 32 + lane]);
    float m = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    const float d = m / 127.f;
    const int q = m > 0.f ? max(-127, min(127, __float2int_rn(v / d))) : 0;
    xq[unit * 32 + lane] = (int8_t)q;
    if (lane == 0) xd[unit] = d;
}

// One warp a sequence of output rows (n0, n0 + stride, ...): each row's blocks pass through the warp's shared buffer a
// chunk at a time, and the next chunk's (or next row's) 16-byte loads are issued before the current chunk is decoded,
// so a warp keeps a load in flight while it computes; the launch sizes the grid to one resident wave.
template <int KIND>
struct Stage {
    static constexpr int STAGE = KIND == 0 ? 1024 : 2048;
    static constexpr int WORDS = STAGE / 16 + 2;
    static constexpr int PER_LANE = (WORDS + 31) / 32;
    static constexpr int BLOCK_BYTES = KIND == 0 ? 34 : KIND == 1 ? 18 : KIND == 2 ? 136 : 110;
    static constexpr int BLOCK_UNITS = (KIND == 0 || KIND == 1) ? 1 : 8;
    static constexpr int CHUNK_BLOCKS = STAGE / BLOCK_BYTES;
};

template <int KIND>
__device__ __forceinline__ int issue_chunk(const uint8_t* row, int c, int blocks, int lane,
                                           uint4 (&held)[Stage<KIND>::PER_LANE]) {
    using S = Stage<KIND>;
    const int nb = min(S::CHUNK_BLOCKS, blocks - c * S::CHUNK_BLOCKS);
    const uint8_t* start = row + (int64_t)c * S::CHUNK_BLOCKS * S::BLOCK_BYTES;
    const int skew = (int)(reinterpret_cast<uintptr_t>(start) & 15);
    const uint4* src = reinterpret_cast<const uint4*>(start - skew);
    const int words = (skew + nb * S::BLOCK_BYTES + 15) >> 4;
    #pragma unroll
    for (int j = 0; j < S::PER_LANE; ++j) {
        const int i = lane + 32 * j;
        if (i < words) held[j] = __ldg(src + i);
    }
    return skew;
}

// G > 1 (one activation row only): a warp's item is G consecutive output rows, contiguous in memory, decoded as one
// strip of G * K / 32 units (short rows would leave lanes idle); unit u of the strip belongs to row u / (K / 32).
template <int KIND, int G>
__global__ void __launch_bounds__(256) gemvq_kernel(
        const uint8_t* __restrict__ raw, int64_t row_bytes, const int8_t* __restrict__ xq,
        const float* __restrict__ xd, int K, int R, int N, float* __restrict__ y, int64_t expert_stride,
        const int32_t* __restrict__ experts, const int32_t* __restrict__ starts, const int32_t* __restrict__ counts,
        int x_div, int rows_per_warp) {
    using S = Stage<KIND>;
    __shared__ uint32_t grid[KIND == 3 ? 512 : 1];
    if (KIND == 3) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) grid[i] = IQ3S_GRID[i];
        __syncthreads();
    }
    __shared__ uint4 staged[WARPS][S::WORDS];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n0 = blockIdx.x * WARPS + warp, stride = gridDim.x * WARPS;   // in items of G rows
    const int pair = blockIdx.y;
    const int items = (N + G - 1) / G;
    if (n0 >= items) return;
    const uint8_t* base = raw;
    int rows = R, first = pair, xrow = 0;
    if (experts != nullptr) {
        base += (int64_t)experts[pair] * expert_stride;
        if (starts != nullptr) {
            first = starts[pair];
            rows = counts[pair];
            xrow = first;
        } else {
            rows = 1;
            xrow = pair / x_div;
        }
    }
    const int units = K / 32;
    const int8_t* xs = xq + (size_t)xrow * K;
    const float* ds = xd + (size_t)xrow * units;
    const int blocks = G * units / S::BLOCK_UNITS;              // an item's blocks (the last item may hold fewer rows)
    const int chunks = (blocks + S::CHUNK_BLOCKS - 1) / S::CHUNK_BLOCKS;
    const int my_rows = min(rows_per_warp, (items - 1 - n0) / stride + 1);
    const int total = my_rows * chunks;
    const int64_t item_bytes = (int64_t)G * row_bytes;
    uint4 held[S::PER_LANE];
    int skew = issue_chunk<KIND>(base + (int64_t)n0 * item_bytes, 0, blocks, lane, held);
    float acc[MAX_ROWS];
    for (int q = 0; q < total; ++q) {
        const int i = q / chunks, c = q - i * chunks;
        const int n = n0 + i * stride;
        const int nb = min(S::CHUNK_BLOCKS, blocks - c * S::CHUNK_BLOCKS);
        const int words = (skew + nb * S::BLOCK_BYTES + 15) >> 4;
        __syncwarp();
        #pragma unroll
        for (int j = 0; j < S::PER_LANE; ++j) {
            const int w = lane + 32 * j;
            if (w < words) staged[warp][w] = held[j];
        }
        __syncwarp();
        const int here = skew;
        if (q + 1 < total) {
            const int i2 = (q + 1) / chunks, c2 = (q + 1) - i2 * chunks;
            skew = issue_chunk<KIND>(base + (int64_t)(n0 + i2 * stride) * item_bytes, c2, blocks, lane, held);
        }
        if (c == 0) {
            #pragma unroll
            for (int r = 0; r < MAX_ROWS; ++r) acc[r] = 0.f;
        }
        const int b0 = c * S::CHUNK_BLOCKS;
        const uint8_t* chunk = reinterpret_cast<const uint8_t*>(staged[warp]) + here - (int64_t)b0 * S::BLOCK_BYTES;
        const int u_end = (b0 + nb) * S::BLOCK_UNITS;
        if (G > 1) {
            const int g_rows = min(G, N - n * G);
            for (int u = b0 * S::BLOCK_UNITS + lane; u < u_end; u += 32) {
                const int g = u / units, uu = u - g * units;
                if (g >= g_rows) break;
                const uint4* xv = reinterpret_cast<const uint4*>(xs + uu * 32);
                const uint4 a = __ldg(xv), b = __ldg(xv + 1);
                const int xw[8] = {(int)a.x, (int)a.y, (int)a.z, (int)a.w, (int)b.x, (int)b.y, (int)b.z, (int)b.w};
                float wscale;
                const int s = unit_isum<KIND>(chunk, u, xw, wscale, grid);
                const float v = (float)s * (wscale * __ldg(ds + uu));
                #pragma unroll
                for (int k = 0; k < G; ++k) acc[k] += (g == k) ? v : 0.f;
            }
        } else
        for (int u = b0 * S::BLOCK_UNITS + lane; u < u_end; u += 32) {
            for (int r = 0; r < rows; ++r) {
                const uint4* xv = reinterpret_cast<const uint4*>(xs + (size_t)r * K + u * 32);
                const uint4 a = __ldg(xv), b = __ldg(xv + 1);
                const int xw[8] = {(int)a.x, (int)a.y, (int)a.z, (int)a.w, (int)b.x, (int)b.y, (int)b.z, (int)b.w};
                float wscale;
                const int s = unit_isum<KIND>(chunk, u, xw, wscale, grid);
                acc[r] = fmaf((float)s, wscale * __ldg(ds + (size_t)r * units + u), acc[r]);
            }
        }
        if (c == chunks - 1) {
            if (G > 1) {
                const int g_rows = min(G, N - n * G);
                #pragma unroll
                for (int k = 0; k < G; ++k) {
                    float v = acc[k];
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
                    if (lane == 0 && k < g_rows) y[(size_t)first * N + n * G + k] = v;
                }
            } else
            for (int r = 0; r < rows; ++r) {
                float v = acc[r];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
                if (lane == 0) {
                    if (experts != nullptr) y[(size_t)(first + r) * N + n] = v;
                    else y[(size_t)r * N + n] = v;
                }
            }
        }
    }
}

// ---- Q8_0 in the aligned layout (kind Q8_0R): a matrix's int8 codes [N][K] first, then its FP16 scales [N][K/32] ----
// The same bytes as the stored blocks in another order (values unchanged): a unit's 32 codes are 32 aligned bytes.
__global__ void repack_q8_kernel(const uint8_t* __restrict__ src, int64_t row_bytes, int K, int64_t N,
                                 uint8_t* __restrict__ dst) {
    const int units = K / 32;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;   // (row, unit)
    if (i >= N * units) return;
    const int64_t n = i / units;
    const int u = (int)(i - n * units);
    const uint8_t* b = src + n * row_bytes + (int64_t)u * 34;
    uint8_t* codes = dst + n * K + (int64_t)u * 32;
    #pragma unroll
    for (int j = 0; j < 32; ++j) codes[j] = b[2 + j];
    uint8_t* scale = dst + N * K + (n * units + u) * 2;
    scale[0] = b[0];
    scale[1] = b[1];
}

// decode rows: one warp a row at a time (rows n0, n0 + stride, ...), lane l the units l, l + 32, ... with 16-byte code
// loads straight into dp4a; each unit's int32 sum scaled once (acc += isum * d_w * d_x), as gemvq_kernel sums them
__global__ void __launch_bounds__(256) gemv_q8r_kernel(const uint8_t* __restrict__ raw, const int8_t* __restrict__ xq,
                                                       const float* __restrict__ xd, int K, int R, int N,
                                                       float* __restrict__ y, int rows_per_warp) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n0 = blockIdx.x * WARPS + warp, stride = gridDim.x * WARPS;
    const int units = K / 32;
    const __half* scales = reinterpret_cast<const __half*>(raw + (int64_t)N * K);
    for (int i = 0; i < rows_per_warp; ++i) {
        const int n = n0 + i * stride;
        if (n >= N) return;
        const uint4* codes = reinterpret_cast<const uint4*>(raw + (int64_t)n * K);
        const __half* sc = scales + (int64_t)n * units;
        float acc[MAX_ROWS];
        #pragma unroll
        for (int r = 0; r < MAX_ROWS; ++r) acc[r] = 0.f;
        #pragma unroll 3
        for (int u = lane; u < units; u += 32) {
            const uint4 c0 = __ldg(codes + 2 * u), c1 = __ldg(codes + 2 * u + 1);
            const float sw = __half2float(sc[u]);
            for (int r = 0; r < R; ++r) {
                const uint4* xv = reinterpret_cast<const uint4*>(xq + (size_t)r * K) + 2 * u;
                const uint4 a0 = __ldg(xv), a1 = __ldg(xv + 1);
                int s = 0;
                s = __dp4a((int)c0.x, (int)a0.x, s);
                s = __dp4a((int)c0.y, (int)a0.y, s);
                s = __dp4a((int)c0.z, (int)a0.z, s);
                s = __dp4a((int)c0.w, (int)a0.w, s);
                s = __dp4a((int)c1.x, (int)a1.x, s);
                s = __dp4a((int)c1.y, (int)a1.y, s);
                s = __dp4a((int)c1.z, (int)a1.z, s);
                s = __dp4a((int)c1.w, (int)a1.w, s);
                acc[r] = fmaf((float)s, sw * __ldg(xd + (size_t)r * units + u), acc[r]);
            }
        }
        for (int r = 0; r < R; ++r) {
            float v = acc[r];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
            if (lane == 0) y[(size_t)r * N + n] = v;
        }
    }
}

// BF16 [N][K] of a Q8_0R matrix
__global__ void dequant_q8r_kernel(const uint8_t* __restrict__ raw, int K, int64_t N, __nv_bfloat16* __restrict__ out) {
    const int units = K / 32;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * units) return;
    const int64_t n = i / units;
    const int u = (int)(i - n * units);
    const float d = __half2float(reinterpret_cast<const __half*>(raw + N * K)[n * units + u]);
    const int8_t* c = reinterpret_cast<const int8_t*>(raw + n * K + (int64_t)u * 32);
    __nv_bfloat16* o = out + n * K + (int64_t)u * 32;
    #pragma unroll
    for (int j = 0; j < 32; ++j) o[j] = __float2bfloat16(d * (float)c[j]);
}

std::pair<torch::Tensor, torch::Tensor> quantize_rows_q8(const torch::Tensor& x) {
    const int64_t R = x.size(0), K = x.size(1);
    auto xq = torch::empty({R, K}, x.options().dtype(torch::kInt8));
    auto xd = torch::empty({R, K / 32}, x.options().dtype(torch::kFloat32));
    const int64_t units = R * (K / 32);
    if (units > 0) {
        quantize_kernel<<<(unsigned)((units + 7) / 8), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), units, xq.data_ptr<int8_t>(), xd.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {xq, xd};
}

template <int KIND, int G>
void launch_q_kind(const uint8_t* raw, int64_t row_bytes, const int8_t* xq, const float* xd, int K, int R, int N,
                   float* y, int64_t expert_stride, const int32_t* experts, int pairs, const int32_t* starts,
                   const int32_t* counts, int x_div) {
    // one resident wave: the blocks an SM holds of this kernel, times the SMs, shared out over the rows
    static int resident = 0;
    if (resident == 0) {
        int device = 0, sms = 0, per_sm = 0;
        C10_CUDA_CHECK(cudaGetDevice(&device));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gemvq_kernel<KIND, G>, 32 * WARPS, 0));
        resident = std::max(1, sms * std::max(1, per_sm));
    }
    const int64_t row_groups = ((N + G - 1) / G + WARPS - 1) / WARPS;   // blocks if every warp took one item
    const int64_t want_x = std::max<int64_t>(1, resident / std::max(1, pairs));
    const int rows_per_warp = (int)std::max<int64_t>(1, (row_groups + want_x - 1) / want_x);
    const int blocks_x = (int)((row_groups + rows_per_warp - 1) / rows_per_warp);
    dim3 grid_dim(blocks_x, pairs);
    gemvq_kernel<KIND, G><<<grid_dim, 32 * WARPS, 0, at::cuda::getCurrentCUDAStream()>>>(
        raw, row_bytes, xq, xd, K, R, N, y, expert_stride, experts, starts, counts, x_div, rows_per_warp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_q(int kind, const uint8_t* raw, int64_t row_bytes, const int8_t* xq, const float* xd, int K, int R, int N,
              float* y, int64_t expert_stride, const int32_t* experts, int pairs, const int32_t* starts = nullptr,
              const int32_t* counts = nullptr, int x_div = 1) {
    TORCH_CHECK(kind != 4, "gguf_gemv: Q8_0R decode rows run through gemv_q8's own kernel");
    // short rows with one activation row a pair: strips of 4 rows
    // (gathered pairs only: a plain GEMV's rows keep one summation order whatever rows share the call)
    const bool strips = (K / 32) <= 32 && R == 1 && experts != nullptr && starts == nullptr;
    switch (kind * 2 + (strips ? 1 : 0)) {
        case 0: launch_q_kind<0, 1>(raw, row_bytes, xq, xd, K, R, N, y, expert_stride, experts, pairs, starts, counts, x_div); break;
        case 1: launch_q_kind<0, 4>(raw, row_bytes, xq, xd, K, R, N, y, expert_stride, experts, pairs, starts, counts, x_div); break;
        case 2: launch_q_kind<1, 1>(raw, row_bytes, xq, xd, K, R, N, y, expert_stride, experts, pairs, starts, counts, x_div); break;
        case 3: launch_q_kind<1, 4>(raw, row_bytes, xq, xd, K, R, N, y, expert_stride, experts, pairs, starts, counts, x_div); break;
        case 4: launch_q_kind<2, 1>(raw, row_bytes, xq, xd, K, R, N, y, expert_stride, experts, pairs, starts, counts, x_div); break;
        case 5: launch_q_kind<2, 4>(raw, row_bytes, xq, xd, K, R, N, y, expert_stride, experts, pairs, starts, counts, x_div); break;
        case 6: launch_q_kind<3, 1>(raw, row_bytes, xq, xd, K, R, N, y, expert_stride, experts, pairs, starts, counts, x_div); break;
        default: launch_q_kind<3, 4>(raw, row_bytes, xq, xd, K, R, N, y, expert_stride, experts, pairs, starts, counts, x_div); break;
    }
}

// BF16 values of rows: out [rows][K], unit by unit (one thread a 32-element unit), the same FP32 values the GEMV
// multiplies, rounded once. rows of expert e = experts[i] when experts is given (raw [E][N][row_bytes]).
template <int KIND>
__global__ void dequant_kernel(const uint8_t* __restrict__ raw, int64_t row_bytes, int K, int64_t rows,
                               __nv_bfloat16* __restrict__ out, int N, int64_t expert_stride,
                               const int32_t* __restrict__ experts) {
    __shared__ uint32_t grid[512];
    __shared__ float kv[16];
    if (KIND == 3) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) grid[i] = IQ3S_GRID[i];
    }
    if (KIND == 1 || KIND == 2) {
        if (threadIdx.x < 16) kv[threadIdx.x] = (float)KVALUES[threadIdx.x];
    }
    __syncthreads();
    const int units = K / 32;
    const int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * units) return;
    const int64_t r = index / units;
    const int u = (int)(index % units);
    const uint8_t* row;
    if (experts != nullptr) row = raw + (int64_t)experts[r / N] * expert_stride + (r % N) * row_bytes;
    else row = raw + r * row_bytes;
    float w[32];
    unit_values<KIND>(row, u, w, grid, kv);
    __nv_bfloat16* o = out + r * K + (int64_t)u * 32;
    #pragma unroll
    for (int i = 0; i < 32; ++i) o[i] = __float2bfloat16(w[i]);
}

int kind_of(const std::string& name) {
    if (name == "Q8_0") return 0;
    if (name == "IQ4_NL") return 1;
    if (name == "IQ4_XS") return 2;
    if (name == "IQ3_S") return 3;
    if (name == "Q8_0R") return 4;
    TORCH_CHECK(false, "gguf_gemv: no kernel for ", name);
}

void launch(int kind, const uint8_t* raw, int64_t row_bytes, const __nv_bfloat16* x, int K, int R, int N, float* y,
            int64_t expert_stride, const int32_t* experts, int pairs, const int32_t* starts = nullptr,
            const int32_t* counts = nullptr, int x_div = 1) {
    TORCH_CHECK(kind != 4, "gguf_gemv: the Q8_0R layout runs through gemv_q8, mmq and dequant only");
    dim3 grid_dim((N + WARPS - 1) / WARPS, pairs);
    dim3 block(32 * WARPS);
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (kind) {
        case 0: gemv_kernel<0><<<grid_dim, block, 0, stream>>>(raw, row_bytes, x, K, R, N, y, expert_stride, experts, starts, counts, x_div); break;
        case 1: gemv_kernel<1><<<grid_dim, block, 0, stream>>>(raw, row_bytes, x, K, R, N, y, expert_stride, experts, starts, counts, x_div); break;
        case 2: gemv_kernel<2><<<grid_dim, block, 0, stream>>>(raw, row_bytes, x, K, R, N, y, expert_stride, experts, starts, counts, x_div); break;
        default: gemv_kernel<3><<<grid_dim, block, 0, stream>>>(raw, row_bytes, x, K, R, N, y, expert_stride, experts, starts, counts, x_div); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// y [R][N] FP32 = x [R][K] (BF16, R <= 8) times raw rows [N][row_bytes] of `kind`.
torch::Tensor gemv(torch::Tensor x, torch::Tensor raw, const std::string& kind, int64_t K) {
    TORCH_CHECK(x.is_cuda() && raw.is_cuda() && x.scalar_type() == torch::kBFloat16 && raw.scalar_type() == torch::kUInt8);
    TORCH_CHECK(x.dim() == 2 && x.size(0) <= MAX_ROWS && x.size(1) == K && K % 32 == 0);
    x = x.contiguous();
    TORCH_CHECK(raw.is_contiguous() && raw.dim() == 2);
    const int N = (int)raw.size(0), R = (int)x.size(0);
    auto y = torch::empty({R, N}, x.options().dtype(torch::kFloat32));
    launch(kind_of(kind), raw.data_ptr<uint8_t>(), raw.size(1), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
           (int)K, R, N, y.data_ptr<float>(), 0, nullptr, 1);
    return y;
}

// gemv with the stored format's activations: x's rows quantized to int8 units, integer dot products
torch::Tensor gemv_q8(torch::Tensor x, torch::Tensor raw, const std::string& kind, int64_t K) {
    TORCH_CHECK(x.is_cuda() && raw.is_cuda() && x.scalar_type() == torch::kBFloat16 && raw.scalar_type() == torch::kUInt8);
    TORCH_CHECK(x.dim() == 2 && x.size(0) <= MAX_ROWS && x.size(1) == K && K % 32 == 0);
    x = x.contiguous();
    TORCH_CHECK(raw.is_contiguous() && raw.dim() == 2);
    const int N = (int)raw.size(0), R = (int)x.size(0);
    auto y = torch::empty({R, N}, x.options().dtype(torch::kFloat32));
    if (R == 0) return y;
    auto q = quantize_rows_q8(x);
    if (kind_of(kind) == 4) {
        static int resident = 0;
        if (resident == 0) {
            int device = 0, sms = 0, per_sm = 0;
            C10_CUDA_CHECK(cudaGetDevice(&device));
            C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
            C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gemv_q8r_kernel, 32 * WARPS, 0));
            resident = std::max(1, sms * std::max(1, per_sm));
        }
        const int64_t groups = (N + WARPS - 1) / WARPS;
        const int rpw = (int)std::max<int64_t>(1, (groups + resident - 1) / resident);
        const int blocks = (int)((groups + rpw - 1) / rpw);
        gemv_q8r_kernel<<<blocks, 32 * WARPS, 0, at::cuda::getCurrentCUDAStream()>>>(
            raw.data_ptr<uint8_t>(), q.first.data_ptr<int8_t>(), q.second.data_ptr<float>(), (int)K, R, N,
            y.data_ptr<float>(), rpw);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return y;
    }
    launch_q(kind_of(kind), raw.data_ptr<uint8_t>(), raw.size(1), q.first.data_ptr<int8_t>(), q.second.data_ptr<float>(),
             (int)K, R, N, y.data_ptr<float>(), 0, nullptr, 1);
    return y;
}

// gathered_gemv with the stored format's activations
torch::Tensor gathered_gemv_q8(torch::Tensor x, torch::Tensor raw, torch::Tensor experts, const std::string& kind,
                               int64_t K, int64_t x_div) {
    TORCH_CHECK(x.is_cuda() && experts.is_cuda() && raw.is_cuda());
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && raw.scalar_type() == torch::kUInt8 &&
                experts.scalar_type() == torch::kInt32);
    TORCH_CHECK(raw.dim() == 3 && raw.is_contiguous() && x.dim() == 2 && x.size(1) == K && K % 32 == 0);
    x = x.contiguous();
    experts = experts.contiguous();
    TORCH_CHECK(x_div >= 1 && x.size(0) * x_div >= experts.size(0));
    const int P = (int)experts.size(0), N = (int)raw.size(1);
    auto y = torch::empty({P, N}, x.options().dtype(torch::kFloat32));
    if (P == 0) return y;
    auto q = quantize_rows_q8(x);
    launch_q(kind_of(kind), raw.data_ptr<uint8_t>(), raw.size(2), q.first.data_ptr<int8_t>(), q.second.data_ptr<float>(),
             (int)K, 1, N, y.data_ptr<float>(), raw.size(1) * raw.size(2), experts.data_ptr<int32_t>(), P, nullptr,
             nullptr, (int)x_div);
    return y;
}

// y [P][N] FP32: pair p's row of x [P][K] (BF16) times expert experts[p]'s raw rows, raw [E][N][row_bytes].
torch::Tensor gathered_gemv(torch::Tensor x, torch::Tensor raw, torch::Tensor experts, const std::string& kind,
                            int64_t K, int64_t x_div) {
    TORCH_CHECK(x.is_cuda() && experts.is_cuda());
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && raw.scalar_type() == torch::kUInt8 &&
                experts.scalar_type() == torch::kInt32);
    TORCH_CHECK(raw.dim() == 3 && raw.is_contiguous() && x.dim() == 2 && x.size(1) == K && K % 32 == 0);
    // raw on the device, or in pinned host memory read in place over the link (its mapped device address)
    const uint8_t* rows = raw.data_ptr<uint8_t>();
    if (!raw.is_cuda()) {
        TORCH_CHECK(raw.is_pinned(), "gathered_gemv: host rows must be pinned");
        void* mapped = nullptr;
        C10_CUDA_CHECK(cudaHostGetDevicePointer(&mapped, raw.data_ptr(), 0));
        rows = reinterpret_cast<const uint8_t*>(mapped);
    }
    x = x.contiguous();
    experts = experts.contiguous();
    TORCH_CHECK(x_div >= 1 && x.size(0) * x_div >= experts.size(0));
    const int P = (int)experts.size(0), N = (int)raw.size(1);
    auto y = torch::empty({P, N}, x.options().dtype(torch::kFloat32));
    if (P == 0) return y;
    launch(kind_of(kind), rows, raw.size(2), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
           (int)K, 1, N, y.data_ptr<float>(), raw.size(1) * raw.size(2), experts.data_ptr<int32_t>(), P, nullptr,
           nullptr, (int)x_div);
    return y;
}

void launch_dequant(int kind, const uint8_t* raw, int64_t row_bytes, int K, int64_t rows, __nv_bfloat16* out, int N,
                    int64_t expert_stride, const int32_t* experts) {
    const int64_t work = rows * (K / 32);
    if (work == 0) return;
    const int threads = 256;
    dim3 blocks((unsigned)((work + threads - 1) / threads));
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (kind) {
        case 0: dequant_kernel<0><<<blocks, threads, 0, stream>>>(raw, row_bytes, K, rows, out, N, expert_stride, experts); break;
        case 1: dequant_kernel<1><<<blocks, threads, 0, stream>>>(raw, row_bytes, K, rows, out, N, expert_stride, experts); break;
        case 2: dequant_kernel<2><<<blocks, threads, 0, stream>>>(raw, row_bytes, K, rows, out, N, expert_stride, experts); break;
        default: dequant_kernel<3><<<blocks, threads, 0, stream>>>(raw, row_bytes, K, rows, out, N, expert_stride, experts); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// BF16 [rows][K] of raw rows [rows][row_bytes].
torch::Tensor dequant(torch::Tensor raw, const std::string& kind, int64_t K) {
    TORCH_CHECK(raw.is_cuda() && raw.scalar_type() == torch::kUInt8 && raw.dim() == 2 && raw.is_contiguous());
    TORCH_CHECK(K % 32 == 0);
    auto out = torch::empty({raw.size(0), K}, raw.options().dtype(torch::kBFloat16));
    if (kind_of(kind) == 4) {
        const int64_t work = raw.size(0) * (K / 32);
        if (work > 0) {
            dequant_q8r_kernel<<<(unsigned)((work + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                raw.data_ptr<uint8_t>(), (int)K, raw.size(0), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return out;
    }
    launch_dequant(kind_of(kind), raw.data_ptr<uint8_t>(), raw.size(1), (int)K, raw.size(0),
                   reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), 1, 0, nullptr);
    return out;
}

// BF16 [len(experts)][N][K] of the chosen experts of raw [E][N][row_bytes].
torch::Tensor dequant_experts(torch::Tensor raw, torch::Tensor experts, const std::string& kind, int64_t K) {
    TORCH_CHECK(raw.is_cuda() && raw.scalar_type() == torch::kUInt8 && raw.dim() == 3 && raw.is_contiguous());
    TORCH_CHECK(experts.is_cuda() && experts.scalar_type() == torch::kInt32 && K % 32 == 0);
    experts = experts.contiguous();
    const int N = (int)raw.size(1);
    auto out = torch::empty({experts.size(0), N, K}, raw.options().dtype(torch::kBFloat16));
    launch_dequant(kind_of(kind), raw.data_ptr<uint8_t>(), raw.size(2), (int)K, experts.size(0) * N,
                   reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), N, raw.size(1) * raw.size(2),
                   experts.data_ptr<int32_t>());
    return out;
}

// y [P][N] FP32 for x [P][K] (BF16) whose rows come in groups: group g is rows starts[g] .. starts[g] + counts[g] - 1
// (counts <= 8), all routed to expert experts[g] of raw [E][N][row_bytes] (device or pinned host memory). Each group
// decodes its expert's weights once for all its rows.
torch::Tensor grouped_gemv(torch::Tensor x, torch::Tensor raw, torch::Tensor experts, torch::Tensor starts,
                           torch::Tensor counts, const std::string& kind, int64_t K) {
    TORCH_CHECK(x.is_cuda() && experts.is_cuda() && starts.is_cuda() && counts.is_cuda());
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && raw.scalar_type() == torch::kUInt8 &&
                experts.scalar_type() == torch::kInt32 && starts.scalar_type() == torch::kInt32 &&
                counts.scalar_type() == torch::kInt32);
    TORCH_CHECK(raw.dim() == 3 && raw.is_contiguous() && x.dim() == 2 && x.size(1) == K && K % 32 == 0);
    const uint8_t* rows = raw.data_ptr<uint8_t>();
    if (!raw.is_cuda()) {
        TORCH_CHECK(raw.is_pinned(), "grouped_gemv: host rows must be pinned");
        void* mapped = nullptr;
        C10_CUDA_CHECK(cudaHostGetDevicePointer(&mapped, raw.data_ptr(), 0));
        rows = reinterpret_cast<const uint8_t*>(mapped);
    }
    x = x.contiguous();
    experts = experts.contiguous();
    starts = starts.contiguous();
    counts = counts.contiguous();
    const int P = (int)x.size(0), N = (int)raw.size(1), G = (int)experts.size(0);
    auto y = torch::empty({P, N}, x.options().dtype(torch::kFloat32));
    if (G == 0) return y;
    launch(kind_of(kind), rows, raw.size(2), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int)K, 1, N,
           y.data_ptr<float>(), raw.size(1) * raw.size(2), experts.data_ptr<int32_t>(), G,
           starts.data_ptr<int32_t>(), counts.data_ptr<int32_t>());
    return y;
}

// dst[i] = src[ids[i]] for each of P items of S bytes (S a multiple of 16): routed experts' rows copied from pinned
// host memory (read through its mapped address with 16-byte loads, the link's efficient transfer size) or device
// memory into a device buffer, on the device, so a captured step can stage what it routes to.
__global__ void gather_kernel(const uint4* __restrict__ src, const int32_t* __restrict__ ids, uint4* __restrict__ dst,
                              int64_t words) {
    const int i = blockIdx.y;
    const uint4* from = src + (int64_t)ids[i] * words;
    uint4* to = dst + (int64_t)i * words;
    for (int64_t w = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; w < words; w += (int64_t)gridDim.x * blockDim.x)
        to[w] = from[w];
}

void gather_rows(torch::Tensor src, torch::Tensor ids, torch::Tensor dst) {
    TORCH_CHECK(ids.is_cuda() && dst.is_cuda() && ids.scalar_type() == torch::kInt32);
    TORCH_CHECK(src.scalar_type() == torch::kUInt8 && dst.scalar_type() == torch::kUInt8 && src.is_contiguous() &&
                dst.is_contiguous());
    const int64_t item = src.numel() / src.size(0);
    TORCH_CHECK(item % 16 == 0 && dst.numel() / dst.size(0) == item && dst.size(0) >= ids.size(0));
    const uint8_t* from = src.data_ptr<uint8_t>();
    if (!src.is_cuda()) {
        TORCH_CHECK(src.is_pinned(), "gather_rows: host rows must be pinned");
        void* mapped = nullptr;
        C10_CUDA_CHECK(cudaHostGetDevicePointer(&mapped, src.data_ptr(), 0));
        from = reinterpret_cast<const uint8_t*>(mapped);
    }
    const int P = (int)ids.size(0);
    if (P == 0) return;
    ids = ids.contiguous();
    dim3 grid(64, P);
    gather_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint4*>(from), ids.data_ptr<int32_t>(), reinterpret_cast<uint4*>(dst.data_ptr<uint8_t>()),
        item / 16);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// act[p][i] = gelu_tanh(gu[p][i]) * gu[p][inter + i], BF16 out
__global__ void geglu_kernel(const float* __restrict__ gu, int inter, __nv_bfloat16* __restrict__ act) {
    const int p = blockIdx.x;
    for (int i = threadIdx.x; i < inter; i += blockDim.x) {
        const float g = gu[(int64_t)p * 2 * inter + i], u = gu[(int64_t)p * 2 * inter + inter + i];
        const float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
        act[(int64_t)p * inter + i] = __float2bfloat16(0.5f * g * (1.f + tanhf(inner)) * u);
    }
}

// out[t][h] = sum over k in order of weights[t * top + k] * y[row(t * top + k)][h], routes with id < 0 skipped;
// row(r) = r, or rows[r] where the routes' outputs sit in another order (sorted by expert)
__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename TY>
__global__ void combine_kernel(const TY* __restrict__ y, const int64_t* __restrict__ ids,
                               const float* __restrict__ weights, int top, int H, float* __restrict__ out,
                               const int64_t* __restrict__ rows) {
    const int t = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h < H) {
        float s = 0.f;
        for (int k = 0; k < top; ++k) {
            const int64_t r = (int64_t)t * top + k;
            if (ids[r] >= 0) s += weights[r] * to_float(y[(rows != nullptr ? rows[r] : r) * H + h]);
        }
        out[(int64_t)t * H + h] = s;
    }
}

torch::Tensor geglu(torch::Tensor gu) {
    TORCH_CHECK(gu.is_cuda() && gu.scalar_type() == torch::kFloat32 && gu.dim() == 2 && gu.size(1) % 2 == 0);
    gu = gu.contiguous();
    const int P = (int)gu.size(0), inter = (int)(gu.size(1) / 2);
    auto act = torch::empty({P, inter}, gu.options().dtype(torch::kBFloat16));
    if (P == 0) return act;
    geglu_kernel<<<P, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(gu.data_ptr<float>(), inter,
                                                                   reinterpret_cast<__nv_bfloat16*>(act.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return act;
}

torch::Tensor combine(torch::Tensor y, torch::Tensor ids, torch::Tensor weights, int64_t top,
                      c10::optional<torch::Tensor> rows) {
    TORCH_CHECK(y.is_cuda() && (y.scalar_type() == torch::kFloat32 || y.scalar_type() == torch::kBFloat16) &&
                ids.scalar_type() == torch::kInt64 &&
                weights.scalar_type() == torch::kFloat32 && y.size(0) == ids.numel() && ids.numel() % top == 0);
    y = y.contiguous();
    ids = ids.contiguous();
    weights = weights.contiguous();
    const int T = (int)(ids.numel() / top), H = (int)y.size(1);
    auto out = torch::empty({T, H}, y.options().dtype(torch::kFloat32));
    if (T == 0) return out;
    torch::Tensor rows_c;
    if (rows.has_value()) {
        TORCH_CHECK(rows->scalar_type() == torch::kInt64 && rows->numel() == ids.numel());
        rows_c = rows->contiguous();
    }
    const int64_t* rows_p = rows.has_value() ? rows_c.data_ptr<int64_t>() : nullptr;
    if (y.scalar_type() == torch::kFloat32)
        combine_kernel<float><<<dim3(T, (H + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            y.data_ptr<float>(), ids.data_ptr<int64_t>(), weights.data_ptr<float>(), (int)top, H, out.data_ptr<float>(),
            rows_p);
    else
        combine_kernel<__nv_bfloat16><<<dim3(T, (H + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), ids.data_ptr<int64_t>(), weights.data_ptr<float>(),
            (int)top, H, out.data_ptr<float>(), rows_p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// ---- prompt rows on the tensor cores: int8 MMA straight from the GGUF blocks ----------------------------------------
// y[m][n] = sum_u dx[m][u] * dw[n][u] * sum_{i<32} qx[m][32u + i] * qw[n][32u + i]: activations as the stored formats'
// int8 units of 32 (quantize_rows_q8), weights' 32-element units decoded to int8 codes with their scale (Q8_0 codes and
// d; IQ4 values and d or the sub-block's scale; IQ3_S signed grid magnitudes and d * (1 + 2 * nibble)), each unit's
// int32 sum (mma.sync m16n8k32 s8, exact) scaled once into the FP32 sum, units in order. A block owns a tile of 32 rows
// of one segment (a run of rows sharing one matrix: the whole call for a dense matrix, one expert's routes for
// experts) and 64 output columns; its weights pass through shared memory 8 units (256 elements) at a time.
constexpr int MQ_BM = 32, MQ_BN = 128, MQ_KB = 4, MQ_THREADS = 256;   // measured best of 64x8x128, 128x4x256, 256x2x512
constexpr int MQ_STRIDE = MQ_KB * 32 + 16;   // bytes a shared row (padded: the fragments' rows hit distinct banks)

__device__ __forceinline__ uint32_t ld32g(const uint8_t* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* w = reinterpret_cast<const uint32_t*>(a & ~(uintptr_t)3);
    return __funnelshift_r(__ldg(w), __ldg(w + 1), (uint32_t)(a & 3) * 8);
}

__device__ __forceinline__ float half_g(const uint8_t* p) {   // p 2-byte aligned
    return __half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(p))));
}

// unit u of a row as 8 words of int8 codes (elements 4i .. 4i + 3 in word i) and the unit's scale
template <int KIND>
__device__ __forceinline__ float unit_codes(const uint8_t* row, int u, uint32_t (&w)[8], const uint32_t* grid) {
    if (KIND == 0) {
        const uint8_t* b = row + u * 34;
        const float d = half_g(b);
        const uintptr_t a = reinterpret_cast<uintptr_t>(b + 2);
        const uint32_t* q = reinterpret_cast<const uint32_t*>(a & ~(uintptr_t)3);
        const uint32_t sh = (uint32_t)(a & 3) * 8;
        uint32_t v[9];
        #pragma unroll
        for (int i = 0; i < 9; ++i) v[i] = __ldg(q + i);
        #pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = __funnelshift_r(v[i], v[i + 1], sh);
        return d;
    } else if (KIND == 1 || KIND == 2) {
        const uint8_t* qs;
        float d;
        if (KIND == 1) {
            const uint8_t* b = row + u * 18;
            d = half_g(b);
            qs = b + 2;
        } else {
            const uint8_t* b = row + (u >> 3) * 136;
            const int j = u & 7;
            const uint32_t head = ld32g(b);
            const int low = (__ldg(b + 4 + (j >> 1)) >> (4 * (j & 1))) & 15;
            const int ls = low | ((((head >> 16) >> (2 * j)) & 3) << 4);
            d = __half2float(__ushort_as_half((uint16_t)(head & 0xffffu))) * (float)(ls - 32);
            qs = b + 8 + j * 16;
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int2 v = iq4_values(ld32g(qs + 4 * i));
            w[i] = (uint32_t)v.x;
            w[4 + i] = (uint32_t)v.y;
        }
        return d;
    } else {
        const uint8_t* b = row + (u >> 3) * 110;
        const int j = u & 7;
        const int nib = (__ldg(b + 106 + (j >> 1)) >> (4 * (j & 1))) & 15;
        const float d = half_g(b) * (float)(1 + 2 * nib);
        const uint32_t qh = __ldg(b + 66 + j);
        const uint32_t q0 = ld32g(b + 2 + j * 8), q1 = ld32g(b + 6 + j * 8);
        const uint32_t signs = ld32g(b + 74 + j * 4);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint32_t qb = ((k < 4 ? q0 : q1) >> (8 * (k & 3))) & 255u;
            const uint32_t g = grid[qb | (((qh >> k) & 1u) << 8)];
            const uint32_t spread = (((signs >> (4 * k)) & 15u) * 0x204081u) & 0x01010101u;
            w[k] = (g ^ (spread * 255u)) + spread;
        }
        return d;
    }
}

__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// exact for |c| < 2^22 (a unit's sum is at most 32 * 127 * 127)
__device__ __forceinline__ float small_int_to_float(int c) {
    return __int_as_float(c + 0x4B400000) - 12582912.0f;
}

// segments: offsets [E + 1] (rows of segment e: offsets[e] .. offsets[e + 1] - 1), tile_off [E + 1] (its tiles'
// prefix); dense (offsets null): one segment of dense_rows rows. raw + e * expert_stride is segment e's matrix.
template <int KIND, int BM>
__global__ void __launch_bounds__(MQ_THREADS) mmq_kernel(
        const int8_t* __restrict__ xq, const float* __restrict__ xd, const uint8_t* __restrict__ raw,
        int64_t row_bytes, int64_t expert_stride, int K, int N, const int32_t* __restrict__ offsets,
        const int32_t* __restrict__ tile_off, int E, int dense_rows, float* __restrict__ y,
        __nv_bfloat16* __restrict__ ybf) {
    __shared__ __align__(16) uint8_t As[BM][MQ_STRIDE];
    __shared__ __align__(16) uint8_t Bs[MQ_BN][MQ_STRIDE];
    __shared__ float Sa[BM][MQ_KB];
    __shared__ float Sb[MQ_BN][MQ_KB];
    __shared__ uint32_t grid[KIND == 3 ? 512 : 1];
    const int tile = blockIdx.x, n_base = blockIdx.y * MQ_BN;
    int e = 0, m_first, m_end;
    if (offsets == nullptr) {
        m_first = tile * BM;
        m_end = dense_rows;
        if (m_first >= m_end) return;
    } else {
        if (tile >= tile_off[E]) return;
        int lo = 0, hi = E - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (tile_off[mid] <= tile) lo = mid; else hi = mid - 1;
        }
        e = lo;
        m_first = offsets[e] + (tile - tile_off[e]) * BM;
        m_end = offsets[e + 1];
    }
    if (KIND == 3) {
        for (int i = threadIdx.x; i < 512; i += MQ_THREADS) grid[i] = IQ3S_GRID[i];
    }
    const int m_count = min(BM, m_end - m_first);
    const uint8_t* W = raw + (int64_t)e * expert_stride;
    const int units = K / 32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    constexpr int MT = BM / 16;   // 16-row tiles a warp's columns meet
    float acc[MT][2][4];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            #pragma unroll
            for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;
    for (int kb0 = 0; kb0 < units; kb0 += MQ_KB) {
        const int nkb = min(MQ_KB, units - kb0);
        __syncthreads();
        for (int i = threadIdx.x; i < BM * nkb * 2; i += MQ_THREADS) {
            const int m = i / (nkb * 2), c = i - m * (nkb * 2);
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (m < m_count)
                v = __ldg(reinterpret_cast<const uint4*>(xq + (size_t)(m_first + m) * K + kb0 * 32) + c);
            *reinterpret_cast<uint4*>(&As[m][c * 16]) = v;
        }
        for (int i = threadIdx.x; i < BM * nkb; i += MQ_THREADS) {
            const int m = i / nkb, kb = i - m * nkb;
            Sa[m][kb] = m < m_count ? __ldg(xd + (size_t)(m_first + m) * units + kb0 + kb) : 0.f;
        }
        for (int i = threadIdx.x; i < MQ_BN * nkb; i += MQ_THREADS) {
            const int n = i / nkb, kb = i - n * nkb, nn = n_base + n;
            uint32_t w[8];
            float s = 0.f;
            if (nn < N && KIND == 4) {
                const uint4* c = reinterpret_cast<const uint4*>(W + (int64_t)nn * K + (int64_t)(kb0 + kb) * 32);
                const uint4 c0 = __ldg(c), c1 = __ldg(c + 1);
                w[0] = c0.x; w[1] = c0.y; w[2] = c0.z; w[3] = c0.w;
                w[4] = c1.x; w[5] = c1.y; w[6] = c1.z; w[7] = c1.w;
                s = __half2float(reinterpret_cast<const __half*>(W + (int64_t)N * K)[(int64_t)nn * units + kb0 + kb]);
            } else if (nn < N) {
                s = unit_codes<KIND == 4 ? 0 : KIND>(W + (int64_t)nn * row_bytes, kb0 + kb, w, grid);
            } else {
                #pragma unroll
                for (int q = 0; q < 8; ++q) w[q] = 0u;
            }
            uint4* dst = reinterpret_cast<uint4*>(&Bs[n][kb * 32]);
            dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
            dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
            Sb[n][kb] = s;
        }
        __syncthreads();
        for (int kb = 0; kb < nkb; ++kb) {
            uint32_t a[MT][4], b[2][2];
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const int r = mt * 16 + g;
                a[mt][0] = *reinterpret_cast<const uint32_t*>(&As[r][kb * 32 + t * 4]);
                a[mt][1] = *reinterpret_cast<const uint32_t*>(&As[r + 8][kb * 32 + t * 4]);
                a[mt][2] = *reinterpret_cast<const uint32_t*>(&As[r][kb * 32 + 16 + t * 4]);
                a[mt][3] = *reinterpret_cast<const uint32_t*>(&As[r + 8][kb * 32 + 16 + t * 4]);
            }
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int col = warp * 16 + nt * 8 + g;
                b[nt][0] = *reinterpret_cast<const uint32_t*>(&Bs[col][kb * 32 + t * 4]);
                b[nt][1] = *reinterpret_cast<const uint32_t*>(&Bs[col][kb * 32 + 16 + t * 4]);
            }
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const float sa0 = Sa[mt * 16 + g][kb], sa1 = Sa[mt * 16 + g + 8][kb];
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    int c[4] = {0, 0, 0, 0};
                    mma_s8(c, a[mt], b[nt]);
                    const int col = warp * 16 + nt * 8 + t * 2;
                    const float sb0 = Sb[col][kb], sb1 = Sb[col + 1][kb];
                    acc[mt][nt][0] = fmaf(small_int_to_float(c[0]), sa0 * sb0, acc[mt][nt][0]);
                    acc[mt][nt][1] = fmaf(small_int_to_float(c[1]), sa0 * sb1, acc[mt][nt][1]);
                    acc[mt][nt][2] = fmaf(small_int_to_float(c[2]), sa1 * sb0, acc[mt][nt][2]);
                    acc[mt][nt][3] = fmaf(small_int_to_float(c[3]), sa1 * sb1, acc[mt][nt][3]);
                }
            }
        }
    }
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int r0 = mt * 16 + g, r1 = r0 + 8;
            const int n = n_base + warp * 16 + nt * 8 + t * 2;
            if (ybf != nullptr) {
                if (r0 < m_count) {
                    if (n < N) ybf[(size_t)(m_first + r0) * N + n] = __float2bfloat16(acc[mt][nt][0]);
                    if (n + 1 < N) ybf[(size_t)(m_first + r0) * N + n + 1] = __float2bfloat16(acc[mt][nt][1]);
                }
                if (r1 < m_count) {
                    if (n < N) ybf[(size_t)(m_first + r1) * N + n] = __float2bfloat16(acc[mt][nt][2]);
                    if (n + 1 < N) ybf[(size_t)(m_first + r1) * N + n + 1] = __float2bfloat16(acc[mt][nt][3]);
                }
                continue;
            }
            if (r0 < m_count) {
                if (n < N) y[(size_t)(m_first + r0) * N + n] = acc[mt][nt][0];
                if (n + 1 < N) y[(size_t)(m_first + r0) * N + n + 1] = acc[mt][nt][1];
            }
            if (r1 < m_count) {
                if (n < N) y[(size_t)(m_first + r1) * N + n] = acc[mt][nt][2];
                if (n + 1 < N) y[(size_t)(m_first + r1) * N + n + 1] = acc[mt][nt][3];
            }
        }
    }
}

std::vector<torch::Tensor> quantize_q8(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.size(1) % 32 == 0);
    x = x.contiguous();
    auto q = quantize_rows_q8(x);
    return {q.first, q.second};
}

// y [P][N] FP32 of prompt rows xq [P][K] int8 / xd [P][K/32] and raw ([N][row_bytes] or [E][N][row_bytes]).
// offsets / tile_off (int32 [E + 1], device) name the segments; without them the call is one dense segment.
torch::Tensor mmq(torch::Tensor xq, torch::Tensor xd, torch::Tensor raw, const std::string& kind, int64_t K,
                  c10::optional<torch::Tensor> offsets, c10::optional<torch::Tensor> tile_off, int64_t max_tiles,
                  bool out_bf16) {
    TORCH_CHECK(xq.is_cuda() && xq.scalar_type() == torch::kInt8 && xq.dim() == 2 && xq.size(1) == K && K % 32 == 0);
    TORCH_CHECK(xd.is_cuda() && xd.scalar_type() == torch::kFloat32 && xd.size(0) == xq.size(0));
    TORCH_CHECK(raw.is_cuda() && raw.scalar_type() == torch::kUInt8 && raw.is_contiguous() &&
                (raw.dim() == 2 || raw.dim() == 3));
    xq = xq.contiguous();
    xd = xd.contiguous();
    const bool grouped = offsets.has_value();
    const int N = (int)raw.size(raw.dim() - 2);
    const int64_t row_bytes = raw.size(raw.dim() - 1);
    const int64_t stride = raw.dim() == 3 ? raw.size(1) * raw.size(2) : 0;
    const int P = (int)xq.size(0);
    auto y = torch::empty({P, N}, xq.options().dtype(out_bf16 ? torch::kBFloat16 : torch::kFloat32));
    if (P == 0) return y;
    int E = 1;
    const int32_t* op = nullptr;
    const int32_t* tp = nullptr;
    // dense calls of 64 rows or more: tiles of 64 rows (each weight block decoded once for twice the rows); expert
    // segments: tiles of 32 (gguf-experts' tile offsets count in 32s, MMQ_ROWS)
    const int bm = (!grouped && P >= 64) ? 64 : MQ_BM;
    int tiles = (P + bm - 1) / bm;
    if (grouped) {
        TORCH_CHECK(tile_off.has_value() && offsets->scalar_type() == torch::kInt32 &&
                    tile_off->scalar_type() == torch::kInt32 && offsets->numel() == tile_off->numel());
        E = (int)offsets->numel() - 1;
        TORCH_CHECK(raw.dim() == 3 && raw.size(0) >= E);
        op = offsets->data_ptr<int32_t>();
        tp = tile_off->data_ptr<int32_t>();
        tiles = (int)max_tiles;
    }
    if (tiles <= 0) return y;
    dim3 grid_dim(tiles, (N + MQ_BN - 1) / MQ_BN);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int8_t* xp = xq.data_ptr<int8_t>();
    const float* dp = xd.data_ptr<float>();
    const uint8_t* rp = raw.data_ptr<uint8_t>();
    float* yp = out_bf16 ? nullptr : y.data_ptr<float>();
    __nv_bfloat16* ybp = out_bf16 ? reinterpret_cast<__nv_bfloat16*>(y.data_ptr()) : nullptr;
    const int kind_id = kind_of(kind);
    TORCH_CHECK(kind_id != 4 || !grouped, "mmq: Q8_0R is a dense layout");
#define CRIT_MQ(KD, B) mmq_kernel<KD, B><<<grid_dim, MQ_THREADS, 0, stream>>>(xp, dp, rp, row_bytes, stride, (int)K, N, \
        op, tp, E, P, yp, ybp)
#define CRIT_MQ_K(KD) if (bm == 64) CRIT_MQ(KD, 64); else CRIT_MQ(KD, 32)
    switch (kind_id) {
        case 0: CRIT_MQ_K(0); break;
        case 1: CRIT_MQ_K(1); break;
        case 2: CRIT_MQ_K(2); break;
        case 3: CRIT_MQ_K(3); break;
        default: CRIT_MQ_K(4); break;
    }
#undef CRIT_MQ_K
#undef CRIT_MQ
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// Routing in two kernels: the router logits, a warp an (token, expert) with all its loads issued at once
// (logits[t][e] = r[t] . W[e], BF16 in, FP32 sums); then a warp a token: the softmax over all experts, the top experts
// by score (ties to the lower id; NaN ranks below every number, so a junk row still picks distinct experts, with
// weight 0), their scores renormalized (w / max(sum + eps, floor)) and scaled, times each chosen expert's own scale.
constexpr int ROUTE_MAX_EXPERTS = 256;
constexpr int ROUTE_MAX_TOP = 16;
constexpr int ROUTE_VECS = 16;           // hidden <= 16 * 256 = 4096 on the vector path
constexpr int ROUTE_WARPS = 8;

__device__ __forceinline__ bool route_better(float a, int ia, float b, int ib) {
    const bool na = isnan(a), nb = isnan(b);
    if (na != nb) return nb;
    if (!na && a != b) return a > b;
    return ia < ib;
}

__global__ void __launch_bounds__(ROUTE_WARPS * 32) route_logits_kernel(
        const __nv_bfloat16* __restrict__ r, const __nv_bfloat16* __restrict__ W, int H, int E,
        float* __restrict__ logits) {
    const int t = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * ROUTE_WARPS + warp;
    if (e >= E) return;
    const __nv_bfloat16* rr = r + (int64_t)t * H;
    const __nv_bfloat16* w = W + (int64_t)e * H;
    float s = 0.f;
    if ((H & 7) == 0 && H <= ROUTE_VECS * 256) {
        uint4 a[ROUTE_VECS], b[ROUTE_VECS];
        #pragma unroll
        for (int i = 0; i < ROUTE_VECS; ++i) {
            const int at = lane * 8 + 256 * i;
            if (at < H) {
                a[i] = *reinterpret_cast<const uint4*>(rr + at);
                b[i] = __ldg(reinterpret_cast<const uint4*>(w + at));
            }
        }
        #pragma unroll
        for (int i = 0; i < ROUTE_VECS; ++i) {
            if (lane * 8 + 256 * i < H) {
                const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&a[i]);
                const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&b[i]);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 fa = __bfloat1622float2(pa[j]), fb = __bfloat1622float2(pb[j]);
                    s = fmaf(fa.x, fb.x, s);
                    s = fmaf(fa.y, fb.y, s);
                }
            }
        }
    } else {
        for (int i = lane; i < H; i += 32) s = fmaf(__bfloat162float(rr[i]), __bfloat162float(w[i]), s);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
    if (lane == 0) logits[(int64_t)t * E + e] = s;
}

// Prompt rows of the router logits, ROUTE_TT tokens a block: the tokens' rows staged in shared memory, each warp an
// expert's weights read once into registers for all of them. Every logit is still one warp's sum, each lane's
// elements and FMAs in route_logits_kernel's order and the same shuffle tree, so the bits are the same as one token at
// a time (decode rows keep route_logits_kernel), with an eighth of the weight reads.
constexpr int ROUTE_TT = 8;

__global__ void __launch_bounds__(ROUTE_WARPS * 32) route_logits_tiled_kernel(
        const __nv_bfloat16* __restrict__ r, const __nv_bfloat16* __restrict__ W, int H, int E, int T,
        float* __restrict__ logits) {
    extern __shared__ __align__(16) unsigned char route_rows[];
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(route_rows);
    const int t0 = blockIdx.x * ROUTE_TT, nt = min(ROUTE_TT, T - t0);
    const int vecs = H / 8;
    for (int i = threadIdx.x; i < ROUTE_TT * vecs; i += blockDim.x) {
        const int tt = i / vecs, c = i - tt * vecs;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (tt < nt) v = *reinterpret_cast<const uint4*>(r + (int64_t)(t0 + tt) * H + c * 8);
        reinterpret_cast<uint4*>(xs + tt * H)[c] = v;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int e = warp; e < E; e += ROUTE_WARPS) {
        const __nv_bfloat16* w = W + (int64_t)e * H;
        uint4 b[ROUTE_VECS];
        #pragma unroll
        for (int i = 0; i < ROUTE_VECS; ++i) {
            const int at = lane * 8 + 256 * i;
            if (at < H) b[i] = __ldg(reinterpret_cast<const uint4*>(w + at));
        }
        float s[ROUTE_TT];
        #pragma unroll
        for (int tt = 0; tt < ROUTE_TT; ++tt) s[tt] = 0.f;
        #pragma unroll
        for (int i = 0; i < ROUTE_VECS; ++i) {
            const int at = lane * 8 + 256 * i;
            if (at < H) {
                const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&b[i]);
                #pragma unroll
                for (int tt = 0; tt < ROUTE_TT; ++tt) {
                    const uint4 av = *reinterpret_cast<const uint4*>(xs + tt * H + at);
                    const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&av);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 fa = __bfloat1622float2(pa[j]), fb = __bfloat1622float2(pb[j]);
                        s[tt] = fmaf(fa.x, fb.x, s[tt]);
                        s[tt] = fmaf(fa.y, fb.y, s[tt]);
                    }
                }
            }
        }
        #pragma unroll
        for (int tt = 0; tt < ROUTE_TT; ++tt) {
            float v = s[tt];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
            if (lane == 0 && tt < nt) logits[(int64_t)(t0 + tt) * E + e] = v;
        }
    }
}

__global__ void route_select_kernel(const float* __restrict__ logits, int E, int top,
                                    const float* __restrict__ expert_scale, float eps, float floor_, float scaling,
                                    int64_t* __restrict__ ids, float* __restrict__ weights) {
    constexpr int PER = ROUTE_MAX_EXPERTS / 32;
    __shared__ float chosen_w[ROUTE_MAX_TOP];
    __shared__ int chosen_id[ROUTE_MAX_TOP];
    const int t = blockIdx.x, lane = threadIdx.x;
    const float* lr = logits + (int64_t)t * E;
    float v[PER];
    #pragma unroll
    for (int j = 0; j < PER; ++j) v[j] = (lane + 32 * j < E) ? lr[lane + 32 * j] : NAN;
    // softmax over the experts (NaN logits stay NaN)
    float m = -INFINITY;
    #pragma unroll
    for (int j = 0; j < PER; ++j) if (lane + 32 * j < E) m = fmaxf(m, v[j]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    float total = 0.f;
    #pragma unroll
    for (int j = 0; j < PER; ++j) {
        if (lane + 32 * j < E) {
            v[j] = expf(v[j] - m);
            total += isnan(v[j]) ? 0.f : v[j];
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) total += __shfl_xor_sync(0xffffffffu, total, off);
    #pragma unroll
    for (int j = 0; j < PER; ++j) v[j] = v[j] / total;
    uint32_t taken = 0u;
    float sum = 0.f;
    for (int k = 0; k < top; ++k) {
        float best = NAN;
        int bi = 1 << 30;
        #pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int e = lane + 32 * j;
            if (e < E && !((taken >> j) & 1u) && (bi == (1 << 30) || route_better(v[j], e, best, bi))) {
                best = v[j];
                bi = e;
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (oi != (1 << 30) && (bi == (1 << 30) || route_better(ov, oi, best, bi))) {
                best = ov;
                bi = oi;
            }
        }
        if ((bi & 31) == lane) taken |= 1u << (bi >> 5);
        if (lane == 0) {
            chosen_w[k] = best;
            chosen_id[k] = bi;
        }
        sum += isnan(best) ? 0.f : best;
    }
    __syncwarp();
    if (lane < top) {
        const float denom = fmaxf(sum + eps, floor_);
        float w = chosen_w[lane] / denom * scaling;
        if (isnan(w)) w = 0.f;
        const int id = chosen_id[lane];
        ids[(int64_t)t * top + lane] = id;
        weights[(int64_t)t * top + lane] = w * expert_scale[id];
    }
}

std::vector<torch::Tensor> route(torch::Tensor r, torch::Tensor W, int64_t top, torch::Tensor expert_scale, double eps,
                                 double floor_, double scaling) {
    TORCH_CHECK(r.is_cuda() && W.is_cuda() && r.scalar_type() == torch::kBFloat16 && W.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(r.dim() == 2 && W.dim() == 2 && r.size(1) == W.size(1) && W.size(0) <= ROUTE_MAX_EXPERTS &&
                top >= 1 && top <= ROUTE_MAX_TOP && top <= W.size(0));
    TORCH_CHECK(expert_scale.is_cuda() && expert_scale.scalar_type() == torch::kFloat32 &&
                expert_scale.numel() == W.size(0));
    r = r.contiguous();
    W = W.contiguous();
    expert_scale = expert_scale.contiguous();
    const int T = (int)r.size(0), H = (int)r.size(1), E = (int)W.size(0);
    auto ids = torch::empty({T, top}, r.options().dtype(torch::kInt64));
    auto weights = torch::empty({T, top}, r.options().dtype(torch::kFloat32));
    if (T > 0) {
        auto logits = torch::empty({T, E}, r.options().dtype(torch::kFloat32));
        auto stream = at::cuda::getCurrentCUDAStream();
        const size_t rows_bytes = (size_t)ROUTE_TT * H * sizeof(__nv_bfloat16);
        if (T >= ROUTE_TT && (H & 7) == 0 && H <= ROUTE_VECS * 256 && rows_bytes <= 48 * 1024) {
            route_logits_tiled_kernel<<<(T + ROUTE_TT - 1) / ROUTE_TT, ROUTE_WARPS * 32, rows_bytes, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(r.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), H, E, T, logits.data_ptr<float>());
        } else {
            route_logits_kernel<<<dim3((E + ROUTE_WARPS - 1) / ROUTE_WARPS, T), ROUTE_WARPS * 32, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(r.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), H, E, logits.data_ptr<float>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        route_select_kernel<<<T, 32, 0, stream>>>(logits.data_ptr<float>(), E, (int)top, expert_scale.data_ptr<float>(),
                                                   (float)eps, (float)floor_, (float)scaling, ids.data_ptr<int64_t>(),
                                                   weights.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {ids, weights};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("repack_q8", [](torch::Tensor src, torch::Tensor dst, int64_t K) {
        TORCH_CHECK(src.is_cuda() && dst.is_cuda() && src.dim() == 2 && src.sizes() == dst.sizes() && K % 32 == 0);
        TORCH_CHECK(src.size(1) == K / 32 * 34 && src.is_contiguous() && dst.is_contiguous());
        const int64_t work = src.size(0) * (K / 32);
        if (work > 0) {
            repack_q8_kernel<<<(unsigned)((work + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                src.data_ptr<uint8_t>(), src.size(1), (int)K, src.size(0), dst.data_ptr<uint8_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }, "Q8_0 rows to the aligned layout (codes [N][K], then scales [N][K/32])");
    m.def("quantize", &quantize_q8, "BF16 rows to the stored formats' int8 units of 32 and their FP32 scales");
    m.def("mmq", &mmq, "prompt rows times GGUF blocks on the int8 tensor cores, dense or per segment",
          py::arg("xq"), py::arg("xd"), py::arg("raw"), py::arg("kind"), py::arg("K"), py::arg("offsets") = py::none(),
          py::arg("tile_off") = py::none(), py::arg("max_tiles") = 0, py::arg("out_bf16") = false);
    m.def("route", &route, "router logits, softmax, top experts, renormalized and scaled (two kernels)");
    m.def("geglu", &geglu, "tanh-GELU(gate) * up of FP32 gate|up rows, BF16 out");
    m.def("combine", &combine, "each token's weighted routes summed in route order", py::arg("y"), py::arg("ids"),
          py::arg("weights"), py::arg("top"), py::arg("rows") = py::none());
    m.def("gather_rows", &gather_rows, "copy chosen items (experts) of pinned host or device memory into a device buffer");
    m.def("grouped_gemv", &grouped_gemv, "GGUF block GEMV per group of rows routed to one expert, FP32 out");
    m.def("dequant", &dequant, "GGUF rows to BF16");
    m.def("dequant_experts", &dequant_experts, "chosen experts' GGUF rows to BF16");
    m.def("gemv", &gemv, "GGUF block GEMV for up to 8 BF16 rows, FP32 out");
    m.def("gemv_q8", &gemv_q8, "gemv with the stored format's int8 activation units (dp4a)");
    m.def("gathered_gemv_q8", &gathered_gemv_q8, "gathered_gemv with the stored format's int8 activation units",
          py::arg("x"), py::arg("raw"), py::arg("experts"), py::arg("kind"), py::arg("K"), py::arg("x_div") = 1);
    m.def("gathered_gemv", &gathered_gemv, "GGUF block GEMV per (row, expert) pair, FP32 out",
          py::arg("x"), py::arg("raw"), py::arg("experts"), py::arg("kind"), py::arg("K"), py::arg("x_div") = 1);
}
