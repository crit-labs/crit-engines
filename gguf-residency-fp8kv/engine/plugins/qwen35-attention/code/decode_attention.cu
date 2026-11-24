// Decode attention over gemma4-attention's pools, one query position per row, everything a launch reads in device
// buffers (so CUDA graphs capture it): rows' positions, ring bases or page tables. Keys and values are FP8 E4M3 (or
// BF16) rows [slots][kv_heads][dim]. A ring pool keeps position p of a row at base + p % ring and a query at pos
// reads (pos - window, pos]; a derived-key pool keeps values only, at table[p / page] * page + p % page, and each key
// is rope(weight * value) at its position (half-split pairs, inv_freq[i] for pair (i, i + dim/2), zero frequencies
// leaving a pair as it is).
//
// Work: grid (rows, kv_heads, splits): a row's visible positions shared out over the launch's splits in chunks of
// max(MIN_CHUNK, ceil(visible / splits)) positions (so a short row still fills the device and a long one keeps the
// partial buffers small; the chunking depends only on the row's own position), 4 warps a block; warp w owns query heads w, w + 4, ...
// of the kv head's group. Each warp walks its split's positions in order with an online softmax in FP32, lane l
// holding dims [l * DH, (l + 1) * DH) of each half. A split writes (max, sum, acc[dim]) per query head; merge()
// combines the splits in order. A row's result never depends on the other rows.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <algorithm>
#include <cfloat>

namespace {

constexpr int MIN_CHUNK = 16;

// positions a split of a row reads, from the row's visible count and the launch's split count
__device__ __forceinline__ int64_t chunk_of(int64_t visible, int splits) {
    return max((int64_t)MIN_CHUNK, (visible + splits - 1) / splits);
}
constexpr int WARPS = 4;
constexpr int MAX_QPW = 2;     // query heads a warp holds (group / WARPS rounded up)

#include <cuda_fp16.h>

// Element d of row `row` (slot * kv_heads + head) of a pool of D-wide rows. MODE 0: BF16; 1: FP8 E4M3; 2: INT4, two
// codes a byte (element 2j low nibble, 2j + 1 high), value (code - 8) * scale, one FP16 scale per 32 elements.
template <int MODE>
__device__ __forceinline__ float load(const void* base, const __half* scales, int64_t row, int D, int d) {
    if (MODE == 1) {
        __nv_fp8_e4m3 v = reinterpret_cast<const __nv_fp8_e4m3*>(base)[row * D + d];
        return float(v);
    }
    if (MODE == 2) {
        const uint8_t byte = reinterpret_cast<const uint8_t*>(base)[row * (D / 2) + (d >> 1)];
        const int code = (d & 1) ? (byte >> 4) : (byte & 15);
        return (float)(code - 8) * __half2float(scales[row * (D / 32) + (d >> 5)]);
    }
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(base)[row * D + d]);
}

// out[j] = element d0 + j of row `row`, j < DH: INT4 rows read as one DH / 2-byte word and one scale (d0 a multiple of
// DH, DH a power of two up to 32, so the lane's elements sit in one group of 32), others element by element
template <int DH, int MODE>
__device__ __forceinline__ void load_run(const void* base, const __half* scales, int64_t row, int D, int d0,
                                         float (&out)[DH]) {
    if (MODE == 2 && (DH == 4 || DH == 8)) {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(base) + row * (D / 2) + (d0 >> 1);
        const float s = __half2float(scales[row * (D / 32) + (d0 >> 5)]);
        uint32_t w;
        if (DH == 8) w = *reinterpret_cast<const uint32_t*>(bytes);
        else w = *reinterpret_cast<const uint16_t*>(bytes);
        #pragma unroll
        for (int j = 0; j < DH; ++j) out[j] = (float)((int)((w >> (4 * j)) & 15u) - 8) * s;
    } else {
        #pragma unroll
        for (int j = 0; j < DH; ++j) out[j] = load<MODE>(base, scales, row, D, d0 + j);
    }
}

// a lane's DH INT4 codes of one half as their raw word and scale (DH 4 or 8), and the values of such a word
template <int DH>
__device__ __forceinline__ void raw_run(const void* base, const __half* scales, int64_t row, int D, int d0,
                                        uint32_t& w, float& s) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(base) + row * (D / 2) + (d0 >> 1);
    s = __half2float(scales[row * (D / 32) + (d0 >> 5)]);
    if (DH == 8) w = *reinterpret_cast<const uint32_t*>(bytes);
    else w = *reinterpret_cast<const uint16_t*>(bytes);
}

template <int DH>
__device__ __forceinline__ void unpack_run(uint32_t w, float s, float (&out)[DH]) {
    #pragma unroll
    for (int j = 0; j < DH; ++j) out[j] = (float)((int)((w >> (4 * j)) & 15u) - 8) * s;
}

template <int DH, int MODE, bool DERIVED>
__global__ void split_kernel(const __nv_bfloat16* __restrict__ q, const void* __restrict__ keys,
                             const void* __restrict__ values, const __half* __restrict__ kscale,
                             const __half* __restrict__ vscale, const int64_t* __restrict__ pos,
                             const int64_t* __restrict__ base, const int32_t* __restrict__ table, int table_stride,
                             int page, int ring, int window, const float* __restrict__ weight,
                             const float* __restrict__ inv_freq, int heads, int kv_heads, float scale,
                             float* __restrict__ part_acc, float* __restrict__ part_ml, int splits) {
    constexpr int D = 64 * DH;
    const int r = blockIdx.x, h = blockIdx.y, s = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int group = heads / kv_heads;
    const int64_t p_end = pos[r] + 1;                                      // positions < p_end are visible
    const int64_t p_first = window > 0 ? max((int64_t)0, p_end - window) : 0;
    const int64_t chunk = chunk_of(p_end - p_first, splits);
    const int64_t a = p_first + (int64_t)s * chunk;
    const int64_t b = min(p_end, a + chunk);
    int qn = 0;
    int qh[MAX_QPW];
    for (int g = warp; g < group && qn < MAX_QPW; g += WARPS) qh[qn++] = h * group + g;
    // this warp's queries, lane dims [lane*DH, +DH) of each half
    float qv[MAX_QPW][2 * DH];
    #pragma unroll
    for (int i = 0; i < MAX_QPW; ++i) if (i < qn) {
        const __nv_bfloat16* qr = q + ((int64_t)r * heads + qh[i]) * D;
        #pragma unroll
        for (int j = 0; j < DH; ++j) {
            qv[i][j] = __bfloat162float(qr[lane * DH + j]);
            qv[i][DH + j] = __bfloat162float(qr[D / 2 + lane * DH + j]);
        }
    }
    float m[MAX_QPW], l[MAX_QPW], acc[MAX_QPW][2 * DH];
    for (int i = 0; i < MAX_QPW; ++i) {
        m[i] = -FLT_MAX; l[i] = 0.f;
        #pragma unroll
        for (int j = 0; j < 2 * DH; ++j) acc[i][j] = 0.f;
    }
    float w_lo[DH], w_hi[DH], f[DH];
    // derived keys' rotation at the split's position, then advanced a position at a time by the rotation of one step
    // (one accurate sincos a split and a pair instead of one at each position's large angle)
    float rc[DH], rs[DH], sc[DH], ss[DH];
    if (DERIVED) {
        #pragma unroll
        for (int j = 0; j < DH; ++j) {
            w_lo[j] = weight[lane * DH + j];
            w_hi[j] = weight[D / 2 + lane * DH + j];
            f[j] = inv_freq[lane * DH + j];
            sincosf((float)a * f[j], &rs[j], &rc[j]);
            sincosf(f[j], &ss[j], &sc[j]);
        }
    }
    if (qn > 0) {
        // one position's keys and values into the warp's queries: derived keys rotated at the running angle, the
        // scores, the online softmax and the weighted values, positions in order
        auto consume = [&](float (&v_lo)[DH], float (&v_hi)[DH], float (&k_lo)[DH], float (&k_hi)[DH]) {
            if (DERIVED) {
                #pragma unroll
                for (int j = 0; j < DH; ++j) {
                    float x1 = w_lo[j] * v_lo[j], x2 = w_hi[j] * v_hi[j];
                    if (f[j] != 0.f) {
                        k_lo[j] = x1 * rc[j] - x2 * rs[j];
                        k_hi[j] = x2 * rc[j] + x1 * rs[j];
                        const float c2 = rc[j] * sc[j] - rs[j] * ss[j], s2 = rs[j] * sc[j] + rc[j] * ss[j];
                        rc[j] = c2;
                        rs[j] = s2;
                    } else {
                        k_lo[j] = x1;
                        k_hi[j] = x2;
                    }
                }
            }
            #pragma unroll
    for (int i = 0; i < MAX_QPW; ++i) if (i < qn) {
                float d = 0.f;
                #pragma unroll
                for (int j = 0; j < DH; ++j) d = fmaf(qv[i][j], k_lo[j], fmaf(qv[i][DH + j], k_hi[j], d));
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
                const float score = d * scale;
                const float top = fmaxf(m[i], score);
                const float corr = __expf(m[i] - top);
                const float e = __expf(score - top);
                l[i] = l[i] * corr + e;
                #pragma unroll
                for (int j = 0; j < DH; ++j) {
                    acc[i][j] = acc[i][j] * corr + e * v_lo[j];
                    acc[i][DH + j] = acc[i][DH + j] * corr + e * v_hi[j];
                }
                m[i] = top;
            }
        };
        auto row_of = [&](int64_t p) -> int64_t {
            const int64_t slot = table != nullptr ? (int64_t)table[(int64_t)r * table_stride + p / page] * page + p % page
                                         : base[r] + p % ring;
            return slot * kv_heads + h;
        };
        constexpr bool FAST = MODE == 2 && (DH == 4 || DH == 8);
        if (FAST) {
            // INT4 rows: the next PF positions' raw words and scales loaded together, so their latencies overlap
            constexpr int PF = 4;
            for (int64_t p0 = a; p0 < b; p0 += PF) {
                uint32_t rv[PF][2], rk[PF][2];
                float sv[PF][2], sk[PF][2];
                #pragma unroll
                for (int u = 0; u < PF; ++u) {
                    if (p0 + u < b) {
                        const int64_t row = row_of(p0 + u);
                        raw_run<DH>(values, vscale, row, D, lane * DH, rv[u][0], sv[u][0]);
                        raw_run<DH>(values, vscale, row, D, D / 2 + lane * DH, rv[u][1], sv[u][1]);
                        if (!DERIVED) {
                            raw_run<DH>(keys, kscale, row, D, lane * DH, rk[u][0], sk[u][0]);
                            raw_run<DH>(keys, kscale, row, D, D / 2 + lane * DH, rk[u][1], sk[u][1]);
                        }
                    }
                }
                #pragma unroll
                for (int u = 0; u < PF; ++u) {
                    if (p0 + u < b) {
                        float v_lo[DH], v_hi[DH], k_lo[DH], k_hi[DH];
                        unpack_run<DH>(rv[u][0], sv[u][0], v_lo);
                        unpack_run<DH>(rv[u][1], sv[u][1], v_hi);
                        if (!DERIVED) {
                            unpack_run<DH>(rk[u][0], sk[u][0], k_lo);
                            unpack_run<DH>(rk[u][1], sk[u][1], k_hi);
                        }
                        consume(v_lo, v_hi, k_lo, k_hi);
                    }
                }
            }
        } else {
            for (int64_t p = a; p < b; ++p) {
                const int64_t row = row_of(p);
                float v_lo[DH], v_hi[DH], k_lo[DH], k_hi[DH];
                load_run<DH, MODE>(values, vscale, row, D, lane * DH, v_lo);
                load_run<DH, MODE>(values, vscale, row, D, D / 2 + lane * DH, v_hi);
                if (!DERIVED) {
                    load_run<DH, MODE>(keys, kscale, row, D, lane * DH, k_lo);
                    load_run<DH, MODE>(keys, kscale, row, D, D / 2 + lane * DH, k_hi);
                }
                consume(v_lo, v_hi, k_lo, k_hi);
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < MAX_QPW; ++i) if (i < qn) {
        const int64_t at = (((int64_t)r * heads + qh[i]) * splits + s);
        if (a < b) {          // an empty split writes its (max, sum) only; merge never reads its acc
            float* out = part_acc + at * D;
            #pragma unroll
            for (int j = 0; j < DH; ++j) {
                out[lane * DH + j] = acc[i][j];
                out[D / 2 + lane * DH + j] = acc[i][DH + j];
            }
        }
        if (lane == 0) {
            part_ml[at * 2] = (a < b) ? m[i] : -FLT_MAX;
            part_ml[at * 2 + 1] = l[i];
        }
    }
}

// out[r][head][:] = sum_s exp(m_s - M) acc_s / sum_s exp(m_s - M) l_s, splits in order.
__global__ void merge_kernel(const float* __restrict__ part_acc, const float* __restrict__ part_ml, int capacity_splits,
                             int D, int heads, __nv_bfloat16* __restrict__ out, const int64_t* __restrict__ pos,
                             int window) {
    // each split's weight exp(m_s - M) once into shared memory (splits in order), then every dim's weighted sum
    constexpr int MAX_MERGE = 1024;
    __shared__ float weight_of[MAX_MERGE];
    __shared__ float top_s, total_s;
    const int r = blockIdx.x, hq = blockIdx.y;
    const int64_t at = ((int64_t)r * heads + hq) * capacity_splits;
    const int64_t visible = window > 0 ? min((int64_t)window, pos[r] + 1) : pos[r] + 1;
    const int64_t chunk = chunk_of(visible, capacity_splits);
    const int splits = min(min(capacity_splits, MAX_MERGE), (int)((visible + chunk - 1) / chunk));
    if (threadIdx.x == 0) {
        float top = -FLT_MAX;
        for (int s = 0; s < splits; ++s) top = fmaxf(top, part_ml[(at + s) * 2]);
        top_s = top;
    }
    __syncthreads();
    for (int s = threadIdx.x; s < splits; s += blockDim.x) {
        const float ms = part_ml[(at + s) * 2];
        weight_of[s] = ms > -FLT_MAX ? __expf(ms - top_s) : 0.f;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.f;
        for (int s = 0; s < splits; ++s) total += weight_of[s] * part_ml[(at + s) * 2 + 1];
        total_s = total;
    }
    __syncthreads();
    const float total = total_s;
    for (int d = blockIdx.z * blockDim.x + threadIdx.x; d < D; d += blockDim.x * gridDim.z) {
        float sum = 0.f;
        for (int s = 0; s < splits; ++s) {
            const float w = weight_of[s];
            if (w != 0.f) sum += w * part_acc[(at + s) * D + d];
        }
        out[((int64_t)r * heads + hq) * D + d] = __float2bfloat16(total > 0.f ? sum / total : 0.f);
    }
}

template <int DH, int MODE, bool DERIVED>
void launch_split(const at::Tensor& q, const void* keys, const void* values, const __half* kscale,
                  const __half* vscale, const at::Tensor& pos,
                  const int64_t* base, const int32_t* table, int table_stride, int page, int ring, int window,
                  const float* weight, const float* inv_freq, int kv_heads, float scale, at::Tensor& acc,
                  at::Tensor& ml, int splits) {
    const int R = (int)q.size(0), heads = (int)q.size(1);
    dim3 grid(R, kv_heads, splits);
    split_kernel<DH, MODE, DERIVED><<<grid, 32 * WARPS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), keys, values, kscale, vscale, pos.data_ptr<int64_t>(),
        base, table,
        table_stride, page, ring, window, weight, inv_freq, heads, kv_heads, scale, acc.data_ptr<float>(),
        ml.data_ptr<float>(), splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// ---- prompt rows: fused attention on the BF16 tensor cores over an expanded span piece -------------------------------
// expand_kernel decodes positions p0 .. p0 + n - 1 of a sequence from the pool once into BF16 scratch rows
// [n][kv_heads][D] (ring keys as stored; derived keys rope(weight * value) at their positions, as the decode kernel
// derives them; values as stored). prefill_kernel: a block owns 16 rows of one kv head (row r: query r / group, query
// head kv_head * group + r % group) and walks the piece's keys its rows see, 32 a step: scores Q K^T from BF16
// operands with FP32 sums (mma.sync m16n8k16; each of the 4 warps sums a quarter of the dims, the quarters added in
// order), the online softmax in FP32 registers, O += P V (P rounded to BF16). A long span runs as several pieces with
// the rows' (max, sum, O) carried in FP32 between them, so memory never grows with the context.
constexpr int PF_TQ = 16, PF_TK = 32, PF_WARPS = 4;

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

template <int MODE, bool DERIVED>
__global__ void expand_kernel(const void* __restrict__ keys, const void* __restrict__ values,
                              const __half* __restrict__ kscale, const __half* __restrict__ vscale, int64_t base,
                              const int32_t* __restrict__ table, int page, int ring, const float* __restrict__ weight,
                              const float* __restrict__ inv_freq, int kv_heads, int D, int64_t p0, int n,
                              __nv_bfloat16* __restrict__ ko, __nv_bfloat16* __restrict__ vo) {
    const int half = D / 2;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;   // (position, head, pair)
    if (i >= (int64_t)n * kv_heads * half) return;
    const int d = (int)(i % half);
    const int h = (int)((i / half) % kv_heads);
    const int j = (int)(i / ((int64_t)half * kv_heads));
    const int64_t p = p0 + j;
    const int64_t slot = table != nullptr ? (int64_t)table[p / page] * page + p % page : base + p % ring;
    const int64_t row = slot * kv_heads + h;
    const float v1 = load<MODE>(values, vscale, row, D, d), v2 = load<MODE>(values, vscale, row, D, d + half);
    float k1, k2;
    if (DERIVED) {
        const float x1 = weight[d] * v1, x2 = weight[d + half] * v2, f = inv_freq[d];
        if (f != 0.f) {
            float sn, cs;
            sincosf((float)p * f, &sn, &cs);
            k1 = x1 * cs - x2 * sn;
            k2 = x2 * cs + x1 * sn;
        } else {
            k1 = x1;
            k2 = x2;
        }
    } else {
        k1 = load<MODE>(keys, kscale, row, D, d);
        k2 = load<MODE>(keys, kscale, row, D, d + half);
    }
    const int64_t at = ((int64_t)j * kv_heads + h) * D;
    ko[at + d] = __float2bfloat16(k1);
    ko[at + d + half] = __float2bfloat16(k2);
    vo[at + d] = __float2bfloat16(v1);
    vo[at + d + half] = __float2bfloat16(v2);
}

// state (pieces > 1): ml [rows_total][2] and acc [rows_total][D] FP32, rows_total = count * heads indexed like out
template <int D>
__global__ void __launch_bounds__(PF_WARPS * 32) prefill_kernel(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ ks, const __nv_bfloat16* __restrict__ vs,
        int64_t p0, int n, int window, int heads, int kv_heads, int64_t position, int count, int64_t first_key,
        float scale, bool first, bool last, float* __restrict__ ml, float* __restrict__ acc,
        __nv_bfloat16* __restrict__ out) {
    constexpr int DS = D / PF_WARPS, KS = DS / 16, NT = DS / 8, STR = D + 8, VEC = D / 8;
    __shared__ __align__(16) __nv_bfloat16 KV[PF_TK][STR];
    __shared__ float Sp[PF_WARPS][PF_TQ][PF_TK + 1];
    const int h = blockIdx.y, tile = blockIdx.x;
    const int group = heads / kv_heads, rows = count * group;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int r0 = tile * PF_TQ + g, r1 = r0 + 8;
    const int64_t qp0 = r0 < rows ? position + r0 / group : -1;
    const int64_t qp1 = r1 < rows ? position + r1 / group : -1;
    const int64_t o0 = r0 < rows ? ((int64_t)(r0 / group) * heads + h * group + r0 % group) : -1;
    const int64_t o1 = r1 < rows ? ((int64_t)(r1 / group) * heads + h * group + r1 % group) : -1;
    const __nv_bfloat16* q0 = o0 >= 0 ? q + o0 * D + warp * DS : nullptr;
    const __nv_bfloat16* q1 = o1 >= 0 ? q + o1 * D + warp * DS : nullptr;
    uint32_t qa[KS][4];
    #pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
        qa[kk][0] = q0 ? *reinterpret_cast<const uint32_t*>(q0 + kk * 16 + t * 2) : 0u;
        qa[kk][1] = q1 ? *reinterpret_cast<const uint32_t*>(q1 + kk * 16 + t * 2) : 0u;
        qa[kk][2] = q0 ? *reinterpret_cast<const uint32_t*>(q0 + kk * 16 + 8 + t * 2) : 0u;
        qa[kk][3] = q1 ? *reinterpret_cast<const uint32_t*>(q1 + kk * 16 + 8 + t * 2) : 0u;
    }
    float o[NT][4];
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    #pragma unroll
    for (int nn = 0; nn < NT; ++nn) o[nn][0] = o[nn][1] = o[nn][2] = o[nn][3] = 0.f;
    if (!first) {
        if (o0 >= 0) { m0 = ml[o0 * 2]; l0 = ml[o0 * 2 + 1]; }
        if (o1 >= 0) { m1 = ml[o1 * 2]; l1 = ml[o1 * 2 + 1]; }
        #pragma unroll
        for (int nn = 0; nn < NT; ++nn) {
            const int col = warp * DS + nn * 8 + t * 2;
            if (o0 >= 0) { o[nn][0] = acc[o0 * D + col]; o[nn][1] = acc[o0 * D + col + 1]; }
            if (o1 >= 0) { o[nn][2] = acc[o1 * D + col]; o[nn][3] = acc[o1 * D + col + 1]; }
        }
    }
    const int last_row = min(rows - 1, tile * PF_TQ + PF_TQ - 1);
    const int64_t qmin = position + (int64_t)(tile * PF_TQ) / group, qmax = position + last_row / group;
    int64_t k_lo = max(first_key, p0);
    if (window > 0) k_lo = max(k_lo, qmin - window + 1);
    const int64_t k_hi = min(qmax, p0 + n - 1);
    for (int64_t kb0 = k_lo; kb0 <= k_hi; kb0 += PF_TK) {
        __syncthreads();
        for (int i = threadIdx.x; i < PF_TK * VEC; i += PF_WARPS * 32) {
            const int j = i / VEC, c = i - j * VEC;
            const int64_t kp = kb0 + j;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (kp <= k_hi) v = __ldg(reinterpret_cast<const uint4*>(ks + ((kp - p0) * kv_heads + h) * D) + c);
            *reinterpret_cast<uint4*>(&KV[j][c * 8]) = v;
        }
        __syncthreads();
        float s[4][4];
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
            #pragma unroll
            for (int kk = 0; kk < KS; ++kk) {
                const __nv_bfloat16* kr = &KV[nt * 8 + g][warp * DS + kk * 16 + t * 2];
                mma_bf16(s[nt], qa[kk], *reinterpret_cast<const uint32_t*>(kr),
                         *reinterpret_cast<const uint32_t*>(kr + 8));
            }
            Sp[warp][g][nt * 8 + t * 2] = s[nt][0];
            Sp[warp][g][nt * 8 + t * 2 + 1] = s[nt][1];
            Sp[warp][g + 8][nt * 8 + t * 2] = s[nt][2];
            Sp[warp][g + 8][nt * 8 + t * 2 + 1] = s[nt][3];
        }
        __syncthreads();
        // the step's values replace its keys (every warp read its keys before the barrier); scores meanwhile
        for (int i = threadIdx.x; i < PF_TK * VEC; i += PF_WARPS * 32) {
            const int j = i / VEC, c = i - j * VEC;
            const int64_t kp = kb0 + j;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (kp <= k_hi) v = __ldg(reinterpret_cast<const uint4*>(vs + ((kp - p0) * kv_heads + h) * D) + c);
            *reinterpret_cast<uint4*>(&KV[j][c * 8]) = v;
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int col = nt * 8 + t * 2 + c;
                const int64_t kp = kb0 + col;
                const float a = ((Sp[0][g][col] + Sp[1][g][col]) + Sp[2][g][col]) + Sp[3][g][col];
                const float b = ((Sp[0][g + 8][col] + Sp[1][g + 8][col]) + Sp[2][g + 8][col]) + Sp[3][g + 8][col];
                const bool in = kp <= k_hi;
                const bool ok0 = in && kp <= qp0 && (window == 0 || qp0 - kp < window);
                const bool ok1 = in && kp <= qp1 && (window == 0 || qp1 - kp < window);
                s[nt][c] = ok0 ? a * scale : -INFINITY;
                s[nt][2 + c] = ok1 ? b * scale : -INFINITY;
                mx0 = fmaxf(mx0, s[nt][c]);
                mx1 = fmaxf(mx1, s[nt][2 + c]);
            }
        }
        #pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
        }
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float sf0 = n0 == -INFINITY ? 0.f : n0, sf1 = n1 == -INFINITY ? 0.f : n1;
        const float c0 = __expf(m0 - sf0), c1 = __expf(m1 - sf1);
        float ps0 = 0.f, ps1 = 0.f;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            s[nt][0] = __expf(s[nt][0] - sf0);
            s[nt][1] = __expf(s[nt][1] - sf0);
            s[nt][2] = __expf(s[nt][2] - sf1);
            s[nt][3] = __expf(s[nt][3] - sf1);
            ps0 += s[nt][0] + s[nt][1];
            ps1 += s[nt][2] + s[nt][3];
        }
        #pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            ps0 += __shfl_xor_sync(0xffffffffu, ps0, off);
            ps1 += __shfl_xor_sync(0xffffffffu, ps1, off);
        }
        l0 = l0 * c0 + ps0;
        l1 = l1 * c1 + ps1;
        m0 = n0;
        m1 = n1;
        #pragma unroll
        for (int nn = 0; nn < NT; ++nn) {
            o[nn][0] *= c0;
            o[nn][1] *= c0;
            o[nn][2] *= c1;
            o[nn][3] *= c1;
        }
        uint32_t pa[2][4];
        #pragma unroll
        for (int k2 = 0; k2 < 2; ++k2) {
            pa[k2][0] = pack_bf16(s[2 * k2][0], s[2 * k2][1]);
            pa[k2][1] = pack_bf16(s[2 * k2][2], s[2 * k2][3]);
            pa[k2][2] = pack_bf16(s[2 * k2 + 1][0], s[2 * k2 + 1][1]);
            pa[k2][3] = pack_bf16(s[2 * k2 + 1][2], s[2 * k2 + 1][3]);
        }
        __syncthreads();
        #pragma unroll
        for (int k2 = 0; k2 < 2; ++k2) {
            #pragma unroll
            for (int nn = 0; nn < NT; ++nn) {
                const uint32_t addr = (uint32_t)__cvta_generic_to_shared(&KV[k2 * 16 + (lane & 15)][warp * DS + nn * 8]);
                uint32_t b0, b1;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                             : "=r"(b0), "=r"(b1) : "r"(addr));
                mma_bf16(o[nn], pa[k2], b0, b1);
            }
        }
    }
    if (!last) {
        if (warp == 0 && t == 0) {
            if (o0 >= 0) { ml[o0 * 2] = m0; ml[o0 * 2 + 1] = l0; }
            if (o1 >= 0) { ml[o1 * 2] = m1; ml[o1 * 2 + 1] = l1; }
        }
        #pragma unroll
        for (int nn = 0; nn < NT; ++nn) {
            const int col = warp * DS + nn * 8 + t * 2;
            if (o0 >= 0) { acc[o0 * D + col] = o[nn][0]; acc[o0 * D + col + 1] = o[nn][1]; }
            if (o1 >= 0) { acc[o1 * D + col] = o[nn][2]; acc[o1 * D + col + 1] = o[nn][3]; }
        }
        return;
    }
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
    #pragma unroll
    for (int nn = 0; nn < NT; ++nn) {
        const int col = warp * DS + nn * 8 + t * 2;
        if (o0 >= 0) *reinterpret_cast<uint32_t*>(out + o0 * D + col) = pack_bf16(o[nn][0] * i0, o[nn][1] * i0);
        if (o1 >= 0) *reinterpret_cast<uint32_t*>(out + o1 * D + col) = pack_bf16(o[nn][2] * i1, o[nn][3] * i1);
    }
}


// ---- prompt rows, wide tiles: a block owns PW_TQ = 64 rows of one kv head (warp w rows 16w .. 16w + 15, the whole
// head dim each), Q staged in shared memory once and read by ldmatrix, the step's 32 keys and values loaded together;
// the rows' arithmetic per key tile as prefill_kernel's (FP32 scores from BF16 operands, online softmax in FP32, P
// rounded to BF16), the sum over the head dim one mma chain. Key tiles start where prefill_kernel's do, so a row's
// result is the same whatever rows share its block or step.
constexpr int PW_TQ = 64;

template <int D>
__global__ void __launch_bounds__(PF_WARPS * 32) prefill_wide_kernel(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ ks, const __nv_bfloat16* __restrict__ vs,
        int64_t p0, int n, int window, int heads, int kv_heads, int64_t position, int count, int64_t first_key,
        float scale, bool first, bool last, float* __restrict__ ml, float* __restrict__ acc,
        __nv_bfloat16* __restrict__ out) {
    constexpr int KS = D / 16, NT = D / 8, STR = D + 8, VEC = D / 8;
    extern __shared__ __align__(16) unsigned char pw_smem[];
    __nv_bfloat16 (*Qs)[STR] = reinterpret_cast<__nv_bfloat16 (*)[STR]>(pw_smem);
    __nv_bfloat16 (*Ks)[STR] = reinterpret_cast<__nv_bfloat16 (*)[STR]>(pw_smem + sizeof(__nv_bfloat16) * PW_TQ * STR);
    __nv_bfloat16 (*Vs)[STR] = reinterpret_cast<__nv_bfloat16 (*)[STR]>(pw_smem + sizeof(__nv_bfloat16) * (PW_TQ + PF_TK) * STR);
    const int h = blockIdx.y, tile = blockIdx.x;
    const int group = heads / kv_heads, rows = count * group;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    // Q rows of the block into shared memory (rows past the end zero)
    for (int i = threadIdx.x; i < PW_TQ * VEC; i += PF_WARPS * 32) {
        const int j = i / VEC, c = i - j * VEC, r = tile * PW_TQ + j;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (r < rows) {
            const int64_t o = (int64_t)(r / group) * heads + h * group + r % group;
            v = __ldg(reinterpret_cast<const uint4*>(q + o * D) + c);
        }
        *reinterpret_cast<uint4*>(&Qs[j][c * 8]) = v;
    }
    const int r0 = tile * PW_TQ + warp * 16 + g, r1 = r0 + 8;
    const int64_t qp0 = r0 < rows ? position + r0 / group : -1;
    const int64_t qp1 = r1 < rows ? position + r1 / group : -1;
    const int64_t o0 = r0 < rows ? ((int64_t)(r0 / group) * heads + h * group + r0 % group) : -1;
    const int64_t o1 = r1 < rows ? ((int64_t)(r1 / group) * heads + h * group + r1 % group) : -1;
    float o[NT][4];
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    #pragma unroll
    for (int nn = 0; nn < NT; ++nn) o[nn][0] = o[nn][1] = o[nn][2] = o[nn][3] = 0.f;
    if (!first) {
        if (o0 >= 0) { m0 = ml[o0 * 2]; l0 = ml[o0 * 2 + 1]; }
        if (o1 >= 0) { m1 = ml[o1 * 2]; l1 = ml[o1 * 2 + 1]; }
        #pragma unroll
        for (int nn = 0; nn < NT; ++nn) {
            const int col = nn * 8 + t * 2;
            if (o0 >= 0) { o[nn][0] = acc[o0 * D + col]; o[nn][1] = acc[o0 * D + col + 1]; }
            if (o1 >= 0) { o[nn][2] = acc[o1 * D + col]; o[nn][3] = acc[o1 * D + col + 1]; }
        }
    }
    const int last_row = min(rows - 1, tile * PW_TQ + PW_TQ - 1);
    const int64_t qmin = position + (int64_t)(tile * PW_TQ) / group, qmax = position + last_row / group;
    // this warp's own last position: a key tile past it is all masked for the warp's rows (skipped, an exact no-op)
    const int wlast = min(rows - 1, tile * PW_TQ + warp * 16 + 15);
    const int64_t wmax = tile * PW_TQ + warp * 16 < rows ? position + wlast / group : -1;
    int64_t k_lo = max(first_key, p0);
    if (window > 0) k_lo = max(k_lo, qmin - window + 1);
    const int64_t k_hi = min(qmax, p0 + n - 1);
    const uint32_t qbase = (uint32_t)__cvta_generic_to_shared(&Qs[warp * 16 + (lane & 15)][(lane >> 4) * 8]);
    for (int64_t kb0 = k_lo; kb0 <= k_hi; kb0 += PF_TK) {
        __syncthreads();
        for (int i = threadIdx.x; i < PF_TK * VEC; i += PF_WARPS * 32) {
            const int j = i / VEC, c = i - j * VEC;
            const int64_t kp = kb0 + j;
            uint4 kv = make_uint4(0u, 0u, 0u, 0u), vv = kv;
            if (kp <= k_hi) {
                kv = __ldg(reinterpret_cast<const uint4*>(ks + ((kp - p0) * kv_heads + h) * D) + c);
                vv = __ldg(reinterpret_cast<const uint4*>(vs + ((kp - p0) * kv_heads + h) * D) + c);
            }
            *reinterpret_cast<uint4*>(&Ks[j][c * 8]) = kv;
            *reinterpret_cast<uint4*>(&Vs[j][c * 8]) = vv;
        }
        __syncthreads();
        if (kb0 > wmax) continue;           // every row of this warp is before the tile: nothing to add
        float s[4][4];
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
        #pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            uint32_t a[4];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(qbase + kk * 32));
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const __nv_bfloat16* kr = &Ks[nt * 8 + g][kk * 16 + t * 2];
                mma_bf16(s[nt], a, *reinterpret_cast<const uint32_t*>(kr), *reinterpret_cast<const uint32_t*>(kr + 8));
            }
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int col = nt * 8 + t * 2 + c;
                const int64_t kp = kb0 + col;
                const bool in = kp <= k_hi;
                const bool ok0 = in && kp <= qp0 && (window == 0 || qp0 - kp < window);
                const bool ok1 = in && kp <= qp1 && (window == 0 || qp1 - kp < window);
                s[nt][c] = ok0 ? s[nt][c] * scale : -INFINITY;
                s[nt][2 + c] = ok1 ? s[nt][2 + c] * scale : -INFINITY;
                mx0 = fmaxf(mx0, s[nt][c]);
                mx1 = fmaxf(mx1, s[nt][2 + c]);
            }
        }
        #pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
        }
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float sf0 = n0 == -INFINITY ? 0.f : n0, sf1 = n1 == -INFINITY ? 0.f : n1;
        const float c0 = __expf(m0 - sf0), c1 = __expf(m1 - sf1);
        float ps0 = 0.f, ps1 = 0.f;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            s[nt][0] = __expf(s[nt][0] - sf0);
            s[nt][1] = __expf(s[nt][1] - sf0);
            s[nt][2] = __expf(s[nt][2] - sf1);
            s[nt][3] = __expf(s[nt][3] - sf1);
            ps0 += s[nt][0] + s[nt][1];
            ps1 += s[nt][2] + s[nt][3];
        }
        #pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            ps0 += __shfl_xor_sync(0xffffffffu, ps0, off);
            ps1 += __shfl_xor_sync(0xffffffffu, ps1, off);
        }
        l0 = l0 * c0 + ps0;
        l1 = l1 * c1 + ps1;
        m0 = n0;
        m1 = n1;
        #pragma unroll
        for (int nn = 0; nn < NT; ++nn) {
            o[nn][0] *= c0;
            o[nn][1] *= c0;
            o[nn][2] *= c1;
            o[nn][3] *= c1;
        }
        uint32_t pa[2][4];
        #pragma unroll
        for (int k2 = 0; k2 < 2; ++k2) {
            pa[k2][0] = pack_bf16(s[2 * k2][0], s[2 * k2][1]);
            pa[k2][1] = pack_bf16(s[2 * k2][2], s[2 * k2][3]);
            pa[k2][2] = pack_bf16(s[2 * k2 + 1][0], s[2 * k2 + 1][1]);
            pa[k2][3] = pack_bf16(s[2 * k2 + 1][2], s[2 * k2 + 1][3]);
        }
        #pragma unroll
        for (int k2 = 0; k2 < 2; ++k2) {
            #pragma unroll
            for (int nn = 0; nn < NT; ++nn) {
                const uint32_t addr = (uint32_t)__cvta_generic_to_shared(&Vs[k2 * 16 + (lane & 15)][nn * 8]);
                uint32_t b0, b1;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                             : "=r"(b0), "=r"(b1) : "r"(addr));
                mma_bf16(o[nn], pa[k2], b0, b1);
            }
        }
    }
    if (!last) {
        if (t == 0) {
            if (o0 >= 0) { ml[o0 * 2] = m0; ml[o0 * 2 + 1] = l0; }
            if (o1 >= 0) { ml[o1 * 2] = m1; ml[o1 * 2 + 1] = l1; }
        }
        #pragma unroll
        for (int nn = 0; nn < NT; ++nn) {
            const int col = nn * 8 + t * 2;
            if (o0 >= 0) { acc[o0 * D + col] = o[nn][0]; acc[o0 * D + col + 1] = o[nn][1]; }
            if (o1 >= 0) { acc[o1 * D + col] = o[nn][2]; acc[o1 * D + col + 1] = o[nn][3]; }
        }
        return;
    }
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
    #pragma unroll
    for (int nn = 0; nn < NT; ++nn) {
        const int col = nn * 8 + t * 2;
        if (o0 >= 0) *reinterpret_cast<uint32_t*>(out + o0 * D + col) = pack_bf16(o[nn][0] * i0, o[nn][1] * i0);
        if (o1 >= 0) *reinterpret_cast<uint32_t*>(out + o1 * D + col) = pack_bf16(o[nn][2] * i1, o[nn][3] * i1);
    }
}

// q [count][heads][D] BF16 of one sequence's prompt rows at positions position .. position + count - 1 (already in
// the pool); ring: keys, base, ring, window; derived: table [pages] int32, page, weight [D], inv_freq [D/2].
// first_key: the first position the pool holds for the sequence; piece: positions expanded at once. Out BF16.
torch::Tensor prefill(torch::Tensor q, c10::optional<torch::Tensor> keys, torch::Tensor values,
                      c10::optional<torch::Tensor> kscale, c10::optional<torch::Tensor> vscale, int64_t base,
                      c10::optional<torch::Tensor> table, int64_t page, int64_t ring, int64_t window,
                      c10::optional<torch::Tensor> weight, c10::optional<torch::Tensor> inv_freq, int64_t position,
                      int64_t first_key, double scale, int64_t piece) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 3 && q.is_contiguous());
    TORCH_CHECK(values.is_cuda() && values.dim() == 3 && values.is_contiguous());
    const int count = (int)q.size(0), heads = (int)q.size(1), D = (int)q.size(2), kv_heads = (int)values.size(1);
    const bool int4 = values.scalar_type() == torch::kUInt8;
    const bool fp8 = values.scalar_type() == at::ScalarType::Float8_e4m3fn;
    TORCH_CHECK(fp8 || int4 || values.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(D == 64 || D == 128 || D == 256 || D == 512, "head dim ", D);
    TORCH_CHECK(heads % kv_heads == 0 && piece >= 32);
    const int mode = int4 ? 2 : (fp8 ? 1 : 0);
    const bool derived = weight.has_value();     // keys derived from values (else stored keys)
    TORCH_CHECK(derived || keys.has_value());
    auto out = torch::empty_like(q);
    if (count == 0) return out;
    const __half* ksp = kscale.has_value() ? reinterpret_cast<const __half*>(kscale->data_ptr()) : nullptr;
    const __half* vsp = vscale.has_value() ? reinterpret_cast<const __half*>(vscale->data_ptr()) : nullptr;
    const void* kptr = keys.has_value() ? keys->data_ptr() : nullptr;
    const int32_t* tptr = table.has_value() ? table->data_ptr<int32_t>() : nullptr;
    const float* wptr = weight.has_value() ? weight->data_ptr<float>() : nullptr;
    const float* fptr = inv_freq.has_value() ? inv_freq->data_ptr<float>() : nullptr;
    const int64_t end = position + count;
    int64_t lo = first_key;
    if (window > 0) lo = std::max(lo, position - window + 1);
    const int64_t span = end - lo;
    const int64_t step = std::min<int64_t>(piece, span);
    const int pieces = (int)((span + step - 1) / step);
    auto ks = torch::empty({step, kv_heads, D}, q.options());
    auto vs = torch::empty({step, kv_heads, D}, q.options());
    torch::Tensor ml, acc;
    if (pieces > 1) {
        ml = torch::empty({(int64_t)count * heads, 2}, q.options().dtype(torch::kFloat32));
        acc = torch::empty({(int64_t)count * heads, D}, q.options().dtype(torch::kFloat32));
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const auto* qp = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    auto* kp = reinterpret_cast<__nv_bfloat16*>(ks.data_ptr());
    auto* vp = reinterpret_cast<__nv_bfloat16*>(vs.data_ptr());
    float* mlp = pieces > 1 ? ml.data_ptr<float>() : nullptr;
    float* accp = pieces > 1 ? acc.data_ptr<float>() : nullptr;
    dim3 grid((count * (heads / kv_heads) + PF_TQ - 1) / PF_TQ, kv_heads);
    for (int pc = 0; pc < pieces; ++pc) {
        const int64_t p0 = lo + (int64_t)pc * step;
        const int n = (int)std::min<int64_t>(step, end - p0);
        const int64_t work = (int64_t)n * kv_heads * (D / 2);
        const unsigned blocks = (unsigned)((work + 255) / 256);
#define CRIT_EX(M, DER) expand_kernel<M, DER><<<blocks, 256, 0, stream>>>(kptr, values.data_ptr(), ksp, vsp, base, tptr, \
        (int)page, (int)ring, wptr, fptr, kv_heads, D, p0, n, kp, vp)
        if (derived) { if (mode == 2) CRIT_EX(2, true); else if (mode == 1) CRIT_EX(1, true); else CRIT_EX(0, true); }
        else { if (mode == 2) CRIT_EX(2, false); else if (mode == 1) CRIT_EX(1, false); else CRIT_EX(0, false); }
#undef CRIT_EX
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        const bool first = pc == 0, last = pc == pieces - 1;
#define CRIT_PF(DV) prefill_kernel<DV><<<grid, PF_WARPS * 32, 0, stream>>>(qp, kp, vp, p0, n, (int)window, heads,        \
        kv_heads, position, count, first_key, (float)scale, first, last, mlp, accp, op)
        switch (D) {
            case 64: CRIT_PF(64); break;
            case 128: CRIT_PF(128); break;
            case 256: CRIT_PF(256); break;
            default: CRIT_PF(512); break;
        }
#undef CRIT_PF
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

torch::Tensor prefill_wide(torch::Tensor q, c10::optional<torch::Tensor> keys, torch::Tensor values,
                      c10::optional<torch::Tensor> kscale, c10::optional<torch::Tensor> vscale, int64_t base,
                      c10::optional<torch::Tensor> table, int64_t page, int64_t ring, int64_t window,
                      c10::optional<torch::Tensor> weight, c10::optional<torch::Tensor> inv_freq, int64_t position,
                      int64_t first_key, double scale, int64_t piece) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 3 && q.is_contiguous());
    TORCH_CHECK(values.is_cuda() && values.dim() == 3 && values.is_contiguous());
    const int count = (int)q.size(0), heads = (int)q.size(1), D = (int)q.size(2), kv_heads = (int)values.size(1);
    const bool int4 = values.scalar_type() == torch::kUInt8;
    const bool fp8 = values.scalar_type() == at::ScalarType::Float8_e4m3fn;
    TORCH_CHECK(fp8 || int4 || values.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(D == 64 || D == 128 || D == 256 || D == 512, "head dim ", D);
    TORCH_CHECK(heads % kv_heads == 0 && piece >= 32);
    const int mode = int4 ? 2 : (fp8 ? 1 : 0);
    const bool derived = weight.has_value();     // keys derived from values (else stored keys)
    TORCH_CHECK(derived || keys.has_value());
    auto out = torch::empty_like(q);
    if (count == 0) return out;
    const __half* ksp = kscale.has_value() ? reinterpret_cast<const __half*>(kscale->data_ptr()) : nullptr;
    const __half* vsp = vscale.has_value() ? reinterpret_cast<const __half*>(vscale->data_ptr()) : nullptr;
    const void* kptr = keys.has_value() ? keys->data_ptr() : nullptr;
    const int32_t* tptr = table.has_value() ? table->data_ptr<int32_t>() : nullptr;
    const float* wptr = weight.has_value() ? weight->data_ptr<float>() : nullptr;
    const float* fptr = inv_freq.has_value() ? inv_freq->data_ptr<float>() : nullptr;
    const int64_t end = position + count;
    int64_t lo = first_key;
    if (window > 0) lo = std::max(lo, position - window + 1);
    const int64_t span = end - lo;
    const int64_t step = std::min<int64_t>(piece, span);
    const int pieces = (int)((span + step - 1) / step);
    auto ks = torch::empty({step, kv_heads, D}, q.options());
    auto vs = torch::empty({step, kv_heads, D}, q.options());
    torch::Tensor ml, acc;
    if (pieces > 1) {
        ml = torch::empty({(int64_t)count * heads, 2}, q.options().dtype(torch::kFloat32));
        acc = torch::empty({(int64_t)count * heads, D}, q.options().dtype(torch::kFloat32));
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const auto* qp = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    auto* kp = reinterpret_cast<__nv_bfloat16*>(ks.data_ptr());
    auto* vp = reinterpret_cast<__nv_bfloat16*>(vs.data_ptr());
    float* mlp = pieces > 1 ? ml.data_ptr<float>() : nullptr;
    float* accp = pieces > 1 ? acc.data_ptr<float>() : nullptr;
    TORCH_CHECK(D <= 256, "prefill_wide: head dim up to 256");
    dim3 grid((count * (heads / kv_heads) + PW_TQ - 1) / PW_TQ, kv_heads);
    for (int pc = 0; pc < pieces; ++pc) {
        const int64_t p0 = lo + (int64_t)pc * step;
        const int n = (int)std::min<int64_t>(step, end - p0);
        const int64_t work = (int64_t)n * kv_heads * (D / 2);
        const unsigned blocks = (unsigned)((work + 255) / 256);
#define CRIT_EX(M, DER) expand_kernel<M, DER><<<blocks, 256, 0, stream>>>(kptr, values.data_ptr(), ksp, vsp, base, tptr, \
        (int)page, (int)ring, wptr, fptr, kv_heads, D, p0, n, kp, vp)
        if (derived) { if (mode == 2) CRIT_EX(2, true); else if (mode == 1) CRIT_EX(1, true); else CRIT_EX(0, true); }
        else { if (mode == 2) CRIT_EX(2, false); else if (mode == 1) CRIT_EX(1, false); else CRIT_EX(0, false); }
#undef CRIT_EX
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        const bool first = pc == 0, last = pc == pieces - 1;
#define CRIT_PF(DV) do {                                                                                          \
            const int bytes = (int)sizeof(__nv_bfloat16) * (PW_TQ + 2 * PF_TK) * (DV + 8);                        \
            static bool raised = false;                                                                             \
            if (!raised) {                                                                                          \
                cudaFuncSetAttribute(prefill_wide_kernel<DV>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);  \
                raised = true;                                                                                      \
            }                                                                                                       \
            prefill_wide_kernel<DV><<<grid, PF_WARPS * 32, bytes, stream>>>(qp, kp, vp, p0, n, (int)window, heads,  \
                kv_heads, position, count, first_key, (float)scale, first, last, mlp, accp, op);                    \
        } while (0)
        switch (D) {
            case 64: CRIT_PF(64); break;
            case 128: CRIT_PF(128); break;
            default: CRIT_PF(256); break;
        }
#undef CRIT_PF
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

// ---- decode rows on the BF16 tensor cores: split-KV over the pools ---------------------------------------------------
// grid (rows, kv_heads, splits): a block owns one decode row's query heads of one kv head (tile rows 0 .. group - 1,
// the rest padding) and its split's positions (the row's visible positions in chunks of chunk_of(visible, splits)),
// DTK a step: the step's keys decoded once into shared memory by the whole block (ring keys as stored; derived keys
// rope(weight * value), each thread advancing its pairs' rotation a position at a time from one accurate sincos at the
// split's first position), scores from BF16 operands with FP32 sums (mma.sync m16n8k16, the 4 warps' quarters of the
// dims added in order), the online softmax in FP32, then the step's values into the same buffer and O += P V. INT4
// rows are staged raw (codes and scales, 16-byte loads), the next step's loads issued before this step computes. The
// split's (max, sum, O) go to the partial buffers merge_kernel combines.
constexpr int DTK = 16;

template <int D>
constexpr int raw_row_words() { return D / 2 / 16; }   // 16-byte words of a row's INT4 codes

template <int D, int MODE, bool DERIVED>
__global__ void __launch_bounds__(PF_WARPS * 32) decode_mma_kernel(
        const __nv_bfloat16* __restrict__ q, const void* __restrict__ keys, const void* __restrict__ values,
        const __half* __restrict__ kscale, const __half* __restrict__ vscale, const int64_t* __restrict__ pos,
        const int64_t* __restrict__ base, const int32_t* __restrict__ table, int table_stride, int page, int ring,
        int window, const float* __restrict__ weight, const float* __restrict__ inv_freq, int heads, int kv_heads,
        float scale, float* __restrict__ part_acc, float* __restrict__ part_ml, int splits) {
    constexpr int THREADS = PF_WARPS * 32;
    constexpr int DS = D / PF_WARPS, KS = DS / 16, NT = DS / 8, STR = D + 8, HALF = D / 2;
    constexpr int PAIRS = (HALF + THREADS - 1) / THREADS;
    constexpr bool RAW = MODE == 2;
    constexpr int CW = raw_row_words<D>();                         // code words a row
    constexpr int CWT = (DTK * CW + THREADS - 1) / THREADS;          // code words a thread stages
    constexpr int SC = D / 32;                                       // scales a row
    constexpr int SCT = (DTK * SC + THREADS - 1) / THREADS;          // scales a thread stages
    constexpr int SETS = RAW && !DERIVED ? 2 : 1;                    // raw sets a step: keys and values, or values
    __shared__ __align__(16) __nv_bfloat16 KV[DTK][STR];
    __shared__ float Sp[PF_WARPS][PF_TQ][DTK + 1];
    __shared__ __align__(16) uint8_t raw_codes[SETS][RAW ? DTK : 1][RAW ? D / 2 : 16];
    __shared__ __half raw_scales[SETS][RAW ? DTK : 1][RAW ? SC : 1];
    __shared__ int64_t rows_next[DTK];
    const int r = blockIdx.x, h = blockIdx.y, sp = blockIdx.z;
    const int group = heads / kv_heads;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int64_t p_end = pos[r] + 1;
    const int64_t p_first = window > 0 ? max((int64_t)0, p_end - window) : 0;
    const int64_t chunk = chunk_of(p_end - p_first, splits);
    const int64_t a = p_first + (int64_t)sp * chunk;
    const int64_t b = min(p_end, a + chunk);
    auto row_of = [&](int64_t p) -> int64_t {
        const int64_t slot = table != nullptr ? (int64_t)table[(int64_t)r * table_stride + p / page] * page + p % page
                                     : base[r] + p % ring;
        return slot * kv_heads + h;
    };
    const bool v0 = g < group, v1 = g + 8 < group;
    const __nv_bfloat16* q0 = v0 ? q + ((int64_t)r * heads + h * group + g) * D + warp * DS : nullptr;
    const __nv_bfloat16* q1 = v1 ? q + ((int64_t)r * heads + h * group + g + 8) * D + warp * DS : nullptr;
    uint32_t qa[KS][4];
    #pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
        qa[kk][0] = q0 ? *reinterpret_cast<const uint32_t*>(q0 + kk * 16 + t * 2) : 0u;
        qa[kk][1] = q1 ? *reinterpret_cast<const uint32_t*>(q1 + kk * 16 + t * 2) : 0u;
        qa[kk][2] = q0 ? *reinterpret_cast<const uint32_t*>(q0 + kk * 16 + 8 + t * 2) : 0u;
        qa[kk][3] = q1 ? *reinterpret_cast<const uint32_t*>(q1 + kk * 16 + 8 + t * 2) : 0u;
    }
    float o[NT][4];
    #pragma unroll
    for (int nn = 0; nn < NT; ++nn) o[nn][0] = o[nn][1] = o[nn][2] = o[nn][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    float rc[PAIRS], rs[PAIRS], sc[PAIRS], ss[PAIRS], wl[PAIRS], wh[PAIRS], fr[PAIRS];
    if (DERIVED) {
        #pragma unroll
        for (int u = 0; u < PAIRS; ++u) {
            const int d = threadIdx.x + u * THREADS;
            const bool own = d < HALF;
            fr[u] = own ? inv_freq[d] : 0.f;
            wl[u] = own ? weight[d] : 0.f;
            wh[u] = own ? weight[d + HALF] : 0.f;
            sincosf((float)a * fr[u], &rs[u], &rc[u]);
            sincosf(fr[u], &ss[u], &sc[u]);
        }
    }
    // raw INT4 rows of a step: issue (into registers) and store (into shared memory)
    uint4 held_c[SETS][CWT];
    __half held_s[SETS][SCT];
    auto issue = [&](int nk) {
        #pragma unroll
        for (int set = 0; set < SETS; ++set) {
            const uint8_t* src = reinterpret_cast<const uint8_t*>(set == 0 && !DERIVED ? keys : values);
            const __half* scl = set == 0 && !DERIVED ? kscale : vscale;
            if (SETS == 2 && set == 1) { src = reinterpret_cast<const uint8_t*>(values); scl = vscale; }
            #pragma unroll
            for (int u = 0; u < CWT; ++u) {
                const int i = threadIdx.x + u * THREADS, j = i / CW, c = i - j * CW;
                held_c[set][u] = make_uint4(0u, 0u, 0u, 0u);
                if (i < DTK * CW && j < nk) held_c[set][u] = __ldg(reinterpret_cast<const uint4*>(src + rows_next[j] * (D / 2)) + c);
            }
            #pragma unroll
            for (int u = 0; u < SCT; ++u) {
                const int i = threadIdx.x + u * THREADS, j = i / SC, c = i - j * SC;
                held_s[set][u] = __float2half(0.f);
                if (i < DTK * SC && j < nk) held_s[set][u] = scl[rows_next[j] * SC + c];
            }
        }
    };
    auto store = [&]() {
        #pragma unroll
        for (int set = 0; set < SETS; ++set) {
            #pragma unroll
            for (int u = 0; u < CWT; ++u) {
                const int i = threadIdx.x + u * THREADS, j = i / CW, c = i - j * CW;
                if (i < DTK * CW) *reinterpret_cast<uint4*>(&raw_codes[set][j][c * 16]) = held_c[set][u];
            }
            #pragma unroll
            for (int u = 0; u < SCT; ++u) {
                const int i = threadIdx.x + u * THREADS, j = i / SC, c = i - j * SC;
                if (i < DTK * SC) raw_scales[set][j][c] = held_s[set][u];
            }
        }
    };
    auto raw_value = [&](int set, int j, int d) -> float {
        const int code = (raw_codes[set][j][d >> 1] >> (4 * (d & 1))) & 15;
        return (float)(code - 8) * __half2float(raw_scales[set][j][d >> 5]);
    };
    auto raw_to_buffer = [&](int set, int nk) {
        for (int i = threadIdx.x; i < DTK * (D / 8); i += THREADS) {
            const int j = i / (D / 8), d0 = (i - j * (D / 8)) * 8;
            uint4 packed = make_uint4(0u, 0u, 0u, 0u);
            if (j < nk) {
                const uint32_t w = *reinterpret_cast<const uint32_t*>(&raw_codes[set][j][d0 >> 1]);
                const float s = __half2float(raw_scales[set][j][d0 >> 5]);
                float v[8];
                #pragma unroll
                for (int e = 0; e < 8; ++e) v[e] = (float)((int)((w >> (4 * e)) & 15u) - 8) * s;
                packed = make_uint4(pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]), pack_bf16(v[4], v[5]),
                                    pack_bf16(v[6], v[7]));
            }
            *reinterpret_cast<uint4*>(&KV[j][d0]) = packed;
        }
    };
    auto direct_to_buffer = [&](const void* src, const __half* scl, int64_t kb0, int nk) {
        if (MODE == 1) {
            // FP8 rows 8 codes a load (one 64-bit word), converted to BF16 and stored 8 at a time (the same values as
            // the element loads: e4m3 -> float -> bf16 is exact)
            for (int i = threadIdx.x; i < DTK * (D / 8); i += THREADS) {
                const int j = i / (D / 8), d0 = (i - j * (D / 8)) * 8;
                uint4 packed = make_uint4(0u, 0u, 0u, 0u);
                if (j < nk) {
                    const uint2 w = __ldg(reinterpret_cast<const uint2*>(
                        reinterpret_cast<const uint8_t*>(src) + row_of(kb0 + j) * D + d0));
                    float v[8];
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const uint32_t word = e < 4 ? w.x : w.y;
                        __nv_fp8_e4m3 code;
                        code.__x = (__nv_fp8_storage_t)((word >> (8 * (e & 3))) & 0xFFu);
                        v[e] = float(code);
                    }
                    packed = make_uint4(pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]), pack_bf16(v[4], v[5]),
                                        pack_bf16(v[6], v[7]));
                }
                *reinterpret_cast<uint4*>(&KV[j][d0]) = packed;
            }
            return;
        }
        for (int i = threadIdx.x; i < DTK * D; i += THREADS) {
            const int j = i / D, d = i - j * D;
            KV[j][d] = __float2bfloat16(j < nk ? load<MODE>(src, scl, row_of(kb0 + j), D, d) : 0.f);
        }
    };
    if (RAW && a < b) {
        if (threadIdx.x < DTK) rows_next[threadIdx.x] = threadIdx.x < min((int64_t)DTK, b - a) ? row_of(a + threadIdx.x) : 0;
        __syncthreads();
        issue((int)min((int64_t)DTK, b - a));
    }
    for (int64_t kb0 = a; kb0 < b; kb0 += DTK) {
        const int nk = (int)min((int64_t)DTK, b - kb0);
        const int64_t nb0 = kb0 + DTK;
        const int nnk = (int)max((int64_t)0, min((int64_t)DTK, b - nb0));
        __syncthreads();                       // the previous step's reads of KV, Sp, raw and rows_next are done
        if (RAW) {
            store();
            if (threadIdx.x < DTK) rows_next[threadIdx.x] = threadIdx.x < nnk ? row_of(nb0 + threadIdx.x) : 0;
        }
        __syncthreads();
        if (RAW && nnk > 0) issue(nnk);        // the next step's raw rows in flight while this one computes
        // keys
        if (DERIVED) {
            #pragma unroll
            for (int u = 0; u < PAIRS; ++u) {
                const int d = threadIdx.x + u * THREADS;
                if (d < HALF) {
                    #pragma unroll 4
                    for (int j = 0; j < DTK; ++j) {
                        float k1 = 0.f, k2 = 0.f;
                        if (j < nk) {
                            float x1, x2;
                            if (RAW) {
                                x1 = wl[u] * raw_value(0, j, d);
                                x2 = wh[u] * raw_value(0, j, d + HALF);
                            } else {
                                const int64_t row = row_of(kb0 + j);
                                x1 = wl[u] * load<MODE>(values, vscale, row, D, d);
                                x2 = wh[u] * load<MODE>(values, vscale, row, D, d + HALF);
                            }
                            if (fr[u] != 0.f) {
                                k1 = x1 * rc[u] - x2 * rs[u];
                                k2 = x2 * rc[u] + x1 * rs[u];
                                const float c2 = rc[u] * sc[u] - rs[u] * ss[u], s2 = rs[u] * sc[u] + rc[u] * ss[u];
                                rc[u] = c2;
                                rs[u] = s2;
                            } else {
                                k1 = x1;
                                k2 = x2;
                            }
                        }
                        KV[j][d] = __float2bfloat16(k1);
                        KV[j][d + HALF] = __float2bfloat16(k2);
                    }
                }
            }
        } else if (RAW) {
            raw_to_buffer(0, nk);
        } else {
            direct_to_buffer(keys, kscale, kb0, nk);
        }
        __syncthreads();
        float s[DTK / 8][4];
        #pragma unroll
        for (int nt = 0; nt < DTK / 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
            #pragma unroll
            for (int kk = 0; kk < KS; ++kk) {
                const __nv_bfloat16* kr = &KV[nt * 8 + g][warp * DS + kk * 16 + t * 2];
                mma_bf16(s[nt], qa[kk], *reinterpret_cast<const uint32_t*>(kr),
                         *reinterpret_cast<const uint32_t*>(kr + 8));
            }
            Sp[warp][g][nt * 8 + t * 2] = s[nt][0];
            Sp[warp][g][nt * 8 + t * 2 + 1] = s[nt][1];
            Sp[warp][g + 8][nt * 8 + t * 2] = s[nt][2];
            Sp[warp][g + 8][nt * 8 + t * 2 + 1] = s[nt][3];
        }
        __syncthreads();
        // values replace the keys (every warp read its keys before the barrier); the softmax meanwhile
        if (RAW) raw_to_buffer(SETS - 1, nk);
        else direct_to_buffer(values, vscale, kb0, nk);
        float mx0 = -INFINITY, mx1 = -INFINITY;
        #pragma unroll
        for (int nt = 0; nt < DTK / 8; ++nt) {
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int col = nt * 8 + t * 2 + c;
                const float x = ((Sp[0][g][col] + Sp[1][g][col]) + Sp[2][g][col]) + Sp[3][g][col];
                const float y = ((Sp[0][g + 8][col] + Sp[1][g + 8][col]) + Sp[2][g + 8][col]) + Sp[3][g + 8][col];
                const bool in = col < nk;
                s[nt][c] = in && v0 ? x * scale : -INFINITY;
                s[nt][2 + c] = in && v1 ? y * scale : -INFINITY;
                mx0 = fmaxf(mx0, s[nt][c]);
                mx1 = fmaxf(mx1, s[nt][2 + c]);
            }
        }
        #pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
        }
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float sf0 = n0 == -INFINITY ? 0.f : n0, sf1 = n1 == -INFINITY ? 0.f : n1;
        const float c0 = __expf(m0 - sf0), c1 = __expf(m1 - sf1);
        float ps0 = 0.f, ps1 = 0.f;
        #pragma unroll
        for (int nt = 0; nt < DTK / 8; ++nt) {
            s[nt][0] = __expf(s[nt][0] - sf0);
            s[nt][1] = __expf(s[nt][1] - sf0);
            s[nt][2] = __expf(s[nt][2] - sf1);
            s[nt][3] = __expf(s[nt][3] - sf1);
            ps0 += s[nt][0] + s[nt][1];
            ps1 += s[nt][2] + s[nt][3];
        }
        #pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            ps0 += __shfl_xor_sync(0xffffffffu, ps0, off);
            ps1 += __shfl_xor_sync(0xffffffffu, ps1, off);
        }
        l0 = l0 * c0 + ps0;
        l1 = l1 * c1 + ps1;
        m0 = n0;
        m1 = n1;
        #pragma unroll
        for (int nn = 0; nn < NT; ++nn) {
            o[nn][0] *= c0;
            o[nn][1] *= c0;
            o[nn][2] *= c1;
            o[nn][3] *= c1;
        }
        uint32_t pa[DTK / 16][4];
        #pragma unroll
        for (int k2 = 0; k2 < DTK / 16; ++k2) {
            pa[k2][0] = pack_bf16(s[2 * k2][0], s[2 * k2][1]);
            pa[k2][1] = pack_bf16(s[2 * k2][2], s[2 * k2][3]);
            pa[k2][2] = pack_bf16(s[2 * k2 + 1][0], s[2 * k2 + 1][1]);
            pa[k2][3] = pack_bf16(s[2 * k2 + 1][2], s[2 * k2 + 1][3]);
        }
        __syncthreads();
        #pragma unroll
        for (int k2 = 0; k2 < DTK / 16; ++k2) {
            #pragma unroll
            for (int nn = 0; nn < NT; ++nn) {
                const uint32_t addr = (uint32_t)__cvta_generic_to_shared(&KV[k2 * 16 + (lane & 15)][warp * DS + nn * 8]);
                uint32_t b0, b1;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                             : "=r"(b0), "=r"(b1) : "r"(addr));
                mma_bf16(o[nn], pa[k2], b0, b1);
            }
        }
    }
    const bool empty = a >= b;
    #pragma unroll
    for (int half_row = 0; half_row < 2; ++half_row) {
        const int row = g + 8 * half_row;
        if (row >= group) continue;
        const int64_t at = (((int64_t)r * heads + h * group + row) * splits + sp);
        if (!empty) {
            float* out = part_acc + at * D;
            #pragma unroll
            for (int nn = 0; nn < NT; ++nn) {
                const int col = warp * DS + nn * 8 + t * 2;
                out[col] = half_row ? o[nn][2] : o[nn][0];
                out[col + 1] = half_row ? o[nn][3] : o[nn][1];
            }
        }
        if (warp == 0 && t == 0) {
            part_ml[at * 2] = empty ? -FLT_MAX : (half_row ? m1 : m0);
            part_ml[at * 2 + 1] = half_row ? l1 : l0;
        }
    }
}

template <int D, int MODE, bool DERIVED, typename... Args>
void launch_dm(dim3 grid, cudaStream_t stream, Args... args) {
    decode_mma_kernel<D, MODE, DERIVED><<<grid, PF_WARPS * 32, 0, stream>>>(args...);
}

// q [R][heads][D] BF16; keys/values [slots][kv_heads][D] (FP8 E4M3 or BF16; keys undefined for a derived pool);
// pos [R] int64; ring: base [R] int64 with ring and window; derived: table [R][stride] int32, page, weight [D],
// inv_freq [D/2]. splits: the fixed split count the launch covers (positions past a row's own end are empty).
torch::Tensor decode(torch::Tensor q, c10::optional<torch::Tensor> keys, torch::Tensor values, torch::Tensor pos,
                     c10::optional<torch::Tensor> base, c10::optional<torch::Tensor> table, int64_t page,
                     int64_t ring, int64_t window, c10::optional<torch::Tensor> weight,
                     c10::optional<torch::Tensor> inv_freq, double scale, int64_t splits,
                     c10::optional<torch::Tensor> kscale, c10::optional<torch::Tensor> vscale) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 3 && q.is_contiguous());
    TORCH_CHECK(values.is_cuda() && values.dim() == 3 && values.is_contiguous());
    const int D = (int)q.size(2), kv_heads = (int)values.size(1), R = (int)q.size(0), heads = (int)q.size(1);
    const bool int4 = values.scalar_type() == torch::kUInt8;
    TORCH_CHECK(values.size(2) == (int4 ? D / 2 : D) && (D == 256 || D == 512 || D == 128 || D == 64), "head dim ", D);
    TORCH_CHECK(heads % kv_heads == 0 && heads / kv_heads <= WARPS * MAX_QPW);
    const bool fp8 = values.scalar_type() == at::ScalarType::Float8_e4m3fn;
    TORCH_CHECK(fp8 || int4 || values.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(!int4 || (vscale.has_value() && vscale->scalar_type() == torch::kHalf), "INT4 pools take their scales");
    const int mode = int4 ? 2 : (fp8 ? 1 : 0);
    const __half* ksp = kscale.has_value() ? reinterpret_cast<const __half*>(kscale->data_ptr()) : nullptr;
    const __half* vsp = vscale.has_value() ? reinterpret_cast<const __half*>(vscale->data_ptr()) : nullptr;
    const bool derived = weight.has_value();     // keys derived from values (else stored keys)
    auto acc = torch::empty({R, heads, splits, D}, q.options().dtype(torch::kFloat32));
    auto ml = torch::empty({R, heads, splits, 2}, q.options().dtype(torch::kFloat32));
    const void* kptr = keys.has_value() ? keys->data_ptr() : nullptr;
    const int64_t* bptr = base.has_value() ? base->data_ptr<int64_t>() : nullptr;
    const int32_t* tptr = table.has_value() ? table->data_ptr<int32_t>() : nullptr;
    const int tstride = table.has_value() ? (int)table->size(1) : 0;
    const float* wptr = weight.has_value() ? weight->data_ptr<float>() : nullptr;
    const float* fptr = inv_freq.has_value() ? inv_freq->data_ptr<float>() : nullptr;
    const float sc = (float)scale;
#define CRIT_ONE(DHV, M, DER) launch_split<DHV, M, DER>(q, kptr, values.data_ptr(), ksp, vsp, pos, bptr, tptr, tstride, \
                                                        (int)page, (int)ring, (int)window, wptr, fptr, kv_heads, sc,   \
                                                        acc, ml, (int)splits)
#define CRIT_LAUNCH(DHV)                                                                                        \
    if (derived) {                                                                                              \
        if (mode == 2) CRIT_ONE(DHV, 2, true); else if (mode == 1) CRIT_ONE(DHV, 1, true); else CRIT_ONE(DHV, 0, true); \
    } else {                                                                                                    \
        if (mode == 2) CRIT_ONE(DHV, 2, false); else if (mode == 1) CRIT_ONE(DHV, 1, false); else CRIT_ONE(DHV, 0, false); \
    }
    {
        dim3 grid(R, kv_heads, (int)splits);
        auto stream = at::cuda::getCurrentCUDAStream();
        const auto* qp = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
        const int64_t* pp = pos.data_ptr<int64_t>();
        float* ap = acc.data_ptr<float>();
        float* mp = ml.data_ptr<float>();
#define CRIT_DM(DV, M, DER) launch_dm<DV, M, DER>(grid, stream, qp, kptr, values.data_ptr(), \
        ksp, vsp, pp, bptr, tptr, tstride, (int)page, (int)ring, (int)window, wptr, fptr, heads, kv_heads, sc, ap, mp,  \
        (int)splits)
#define CRIT_DM_D(DV)                                                                                             \
        if (derived) { if (mode == 2) CRIT_DM(DV, 2, true); else if (mode == 1) CRIT_DM(DV, 1, true); else CRIT_DM(DV, 0, true); } \
        else { if (mode == 2) CRIT_DM(DV, 2, false); else if (mode == 1) CRIT_DM(DV, 1, false); else CRIT_DM(DV, 0, false); }
        if (heads / kv_heads <= PF_TQ) {
            switch (D) {
                case 64: CRIT_DM_D(64); break;
                case 128: CRIT_DM_D(128); break;
                case 256: CRIT_DM_D(256); break;
                default: CRIT_DM_D(512); break;
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            switch (D) {
                case 64: CRIT_LAUNCH(1); break;
                case 128: CRIT_LAUNCH(2); break;
                case 256: CRIT_LAUNCH(4); break;
                default: CRIT_LAUNCH(8); break;
            }
        }
#undef CRIT_DM_D
#undef CRIT_DM
    }
#undef CRIT_LAUNCH
#undef CRIT_ONE
    auto out = torch::empty({R, heads, D}, q.options());
    merge_kernel<<<dim3(R, heads, (D + 127) / 128), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        acc.data_ptr<float>(), ml.data_ptr<float>(), (int)splits, D, heads,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), pos.data_ptr<int64_t>(), table.has_value() ? 0 : (int)window);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// One warp per group of 32 values: amax over the group, scale = amax / 7 rounded to FP16, codes round(v / scale)
// clamped to [-8, 7] plus 8, packed two a byte (element 2j low nibble), all written at the row's slot.
template <typename T>
__global__ void int4_put_kernel(const T* __restrict__ x, const int64_t* __restrict__ slots, int64_t groups_per_row,
                                int64_t rows, uint8_t* __restrict__ codes, __half* __restrict__ scales) {
    const int64_t g = (int64_t)blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (g >= rows * groups_per_row) return;
    const int64_t row = g / groups_per_row, group = g % groups_per_row;
    const int64_t slot = slots[row];
    float v;
    if constexpr (sizeof(T) == 4) v = reinterpret_cast<const float*>(x)[g * 32 + lane];
    else v = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(x)[g * 32 + lane]);
    float amax = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const __half scale_h = __float2half(fmaxf(amax / 7.0f, 1e-8f));
    const float scale = __half2float(scale_h);
    int code = (int)rintf(v / scale);
    code = min(7, max(-8, code)) + 8;
    const int other = __shfl_down_sync(0xffffffffu, code, 1);
    const int64_t base = (slot * groups_per_row + group);
    if ((lane & 1) == 0) codes[base * 16 + (lane >> 1)] = (uint8_t)(code | (other << 4));
    if (lane == 0) scales[base] = scale_h;
}

// x [n][heads][D] (FP32 or BF16, contiguous), slots [n] int64, codes [slots][heads][D/2] uint8, scales
// [slots][heads][D/32] FP16 (one layer's views).
void int4_put(torch::Tensor x, torch::Tensor slots, torch::Tensor codes, torch::Tensor scales) {
    TORCH_CHECK(x.is_cuda() && slots.is_cuda() && codes.is_cuda() && scales.is_cuda());
    TORCH_CHECK(x.dim() == 3 && x.size(2) % 32 == 0 && codes.scalar_type() == torch::kUInt8 &&
                scales.scalar_type() == torch::kHalf && slots.scalar_type() == torch::kInt64);
    x = x.contiguous();
    const int64_t n = x.size(0), groups_per_row = x.size(1) * x.size(2) / 32;
    if (n == 0) return;
    const int64_t groups = n * groups_per_row;
    const int threads = 256, per_block = threads / 32;
    const int64_t blocks = (groups + per_block - 1) / per_block;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (x.scalar_type() == torch::kFloat32)
        int4_put_kernel<float><<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), slots.data_ptr<int64_t>(),
            groups_per_row, n, codes.data_ptr<uint8_t>(), reinterpret_cast<__half*>(scales.data_ptr()));
    else {
        TORCH_CHECK(x.scalar_type() == torch::kBFloat16);
        int4_put_kernel<uint16_t><<<blocks, threads, 0, stream>>>(reinterpret_cast<const uint16_t*>(x.data_ptr()),
            slots.data_ptr<int64_t>(), groups_per_row, n, codes.data_ptr<uint8_t>(),
            reinterpret_cast<__half*>(scales.data_ptr()));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int4_put", &int4_put, "encode rows to INT4 groups and scatter them to their slots in one kernel");
    m.def("decode", &decode, "gemma4 decode attention over ring or derived-key pools");
    m.def("prefill", &prefill, "gemma4 prompt attention over ring or derived-key pools on the BF16 tensor cores");
    m.def("prefill_wide", &prefill_wide, "prompt attention as prefill, 64 query rows a block (head dims up to 256)");
    m.def("split", []() { return MIN_CHUNK; });
}
