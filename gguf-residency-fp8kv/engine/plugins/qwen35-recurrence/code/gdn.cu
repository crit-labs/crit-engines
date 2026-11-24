// Gated DeltaNet (Qwen3.5 linear attention) on the device, for prompt chunks and decode rows alike.
//
// Rows come in groups: group g is tokens starts[g] .. starts[g] + counts[g] - 1 of one sequence, whose recurrent state
// lives in slot slots[g] of the layer's state tensors; a group's tokens are consecutive positions of its sequence and
// run in order, so a prompt chunk continues where the sequence's last step left its state, and every group (every
// sequence) runs alone: a row's result never depends on the rows beside it. Every index lives in device memory, so a
// captured decode step replays with new slots.
//
// conv: out[t][c] = silu(sum_j w[c][j] * in[t - 3 + j][c]), in[-3..-1] from the slot's conv state (the sequence's last
// three inputs), which the group leaves holding its own last three.
// scan: per value head h (query/key head h % key_heads, ggml's tiled repeat, the GGUF's head order), with
// q = l2norm(q_h) / sqrt(dk), k = l2norm(k_h), and the state S [dk][dv]:
//   kv = S^T k;  delta = (v - exp(g) kv) beta;  S = exp(g) S + k delta^T;  o = S^T q
// (transformers' torch_recurrent_gated_delta_rule; llama.cpp's gated_delta_net). A warp owns one value column of the
// state, its dk values in registers (dk / 32 a lane), so the dot products are warp reductions in a fixed order.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cuda_bf16.h>

namespace {

constexpr int DK = 128, DV = 128, PER_LANE = DK / 32, SCAN_WARPS = 4;

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

// grid (ceil(C / 256), groups); state [slots][3][C]
__global__ void conv_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ state,
                            const int32_t* __restrict__ starts, const int32_t* __restrict__ counts,
                            const int32_t* __restrict__ slots, float* __restrict__ out, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int g = blockIdx.y;
    if (c >= C) return;
    const int first = starts[g], n = counts[g];
    if (n <= 0) return;
    float* st = state + (int64_t)slots[g] * 3 * C;
    float a = st[c], b = st[C + c], d = st[2 * C + c];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < n; ++t) {
        const float e = x[(int64_t)(first + t) * C + c];
        const float s = w0 * a + w1 * b + w2 * d + w3 * e;
        out[(int64_t)(first + t) * C + c] = silu(s);
        a = b; b = d; d = e;
    }
    st[c] = a; st[C + c] = b; st[2 * C + c] = d;
}

// grid (value heads, groups, DV / SCAN_WARPS); conv [T][2 * kh * DK + vh * DV]: q heads, k heads, v heads.
// g, beta [T][vh]; state [slots][vh][DV][DK] (column-major: column col's DK values contiguous); out [T][vh][DV].
__global__ void __launch_bounds__(SCAN_WARPS * 32) scan_kernel(
        const float* __restrict__ conv, const float* __restrict__ g, const float* __restrict__ beta,
        float* __restrict__ state, const int32_t* __restrict__ starts, const int32_t* __restrict__ counts,
        const int32_t* __restrict__ slots, float* __restrict__ out, int kh, int vh, float eps) {
    const int h = blockIdx.x, grp = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = blockIdx.z * SCAN_WARPS + warp;
    const int first = starts[grp], n = counts[grp];
    if (n <= 0) return;
    const int width = 2 * kh * DK + vh * DV;
    const int qk = h % kh;
    float* S = state + (((int64_t)slots[grp] * vh + h) * DV + col) * DK;
    float s[PER_LANE];
    #pragma unroll
    for (int r = 0; r < PER_LANE; ++r) s[r] = S[r * 32 + lane];
    const float qscale = rsqrtf((float)DK);
    for (int t = 0; t < n; ++t) {
        const float* row = conv + (int64_t)(first + t) * width;
        float q[PER_LANE], k[PER_LANE];
        float qq = 0.f, kk = 0.f;
        #pragma unroll
        for (int r = 0; r < PER_LANE; ++r) {
            q[r] = row[qk * DK + r * 32 + lane];
            k[r] = row[kh * DK + qk * DK + r * 32 + lane];
            qq += q[r] * q[r];
            kk += k[r] * k[r];
        }
        const float qn = rsqrtf(warp_sum(qq) + eps) * qscale, kn = rsqrtf(warp_sum(kk) + eps);
        const float v = row[2 * kh * DK + h * DV + col];
        const float decay = __expf(g[(int64_t)(first + t) * vh + h]);
        const float b = beta[(int64_t)(first + t) * vh + h];
        float kv = 0.f;
        #pragma unroll
        for (int r = 0; r < PER_LANE; ++r) kv += s[r] * k[r];
        kv = warp_sum(kv) * kn;
        const float delta = (v - decay * kv) * b;
        float o = 0.f;
        #pragma unroll
        for (int r = 0; r < PER_LANE; ++r) {
            s[r] = decay * s[r] + k[r] * kn * delta;
            o += s[r] * q[r];
        }
        o = warp_sum(o) * qn;
        if (lane == 0) out[((int64_t)(first + t) * vh + h) * DV + col] = o;
    }
    #pragma unroll
    for (int r = 0; r < PER_LANE; ++r) S[r * 32 + lane] = s[r];
}

// beta = sigmoid(ba[:, :vh]); g = softplus(ba[:, vh:] + dt_bias) * a, softplus as torch computes it (threshold 20)
__global__ void gates_kernel(const float* __restrict__ ba, int stride, const float* __restrict__ dt_bias,
                             const float* __restrict__ a, int vh, float* __restrict__ beta, float* __restrict__ g,
                             int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int t = i / vh, h = i - t * vh;
    const float b = ba[(int64_t)t * stride + h];
    beta[i] = 1.f / (1.f + expf(-b));
    const float x = ba[(int64_t)t * stride + vh + h] + dt_bias[h];
    const float sp = x > 20.f ? x : log1pf(expf(x));
    g[i] = sp * a[h];
}

// out[t][h][:] = rms_norm(core[t][h][:]) * w * silu(z[t][h][:]) in BF16; a block a (token, head) of DV values
__global__ void gated_rms_kernel(const float* __restrict__ core, const float* __restrict__ z, int zstride,
                                 const float* __restrict__ w, int vh, float eps, __nv_bfloat16* __restrict__ out) {
    const int t = blockIdx.x / vh, h = blockIdx.x - (blockIdx.x / vh) * vh, i = threadIdx.x;   // DV threads
    const float v = core[((int64_t)t * vh + h) * DV + i];
    __shared__ float part[DV / 32];
    float s = v * v;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
    if ((i & 31) == 0) part[i >> 5] = s;
    __syncthreads();
    float total = 0.f;
    #pragma unroll
    for (int k = 0; k < DV / 32; ++k) total += part[k];
    const float r = rsqrtf(total / DV + eps);
    const float zz = z[(int64_t)t * zstride + h * DV + i];
    out[((int64_t)t * vh + h) * DV + i] = __float2bfloat16(v * r * w[i] * (zz / (1.f + expf(-zz))));
}

void check_groups(const torch::Tensor& starts, const torch::Tensor& counts, const torch::Tensor& slots) {
    TORCH_CHECK(starts.is_cuda() && counts.is_cuda() && slots.is_cuda());
    TORCH_CHECK(starts.scalar_type() == torch::kInt32 && counts.scalar_type() == torch::kInt32 &&
                slots.scalar_type() == torch::kInt32);
    TORCH_CHECK(starts.numel() == counts.numel() && counts.numel() == slots.numel());
}

}  // namespace

// x [T][C] FP32 -> silu(conv) [T][C] FP32; w [C][4]; state [slots][3][C] FP32, updated in place.
torch::Tensor conv(torch::Tensor x, torch::Tensor w, torch::Tensor state, torch::Tensor starts, torch::Tensor counts,
                   torch::Tensor slots) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.dim() == 2 && x.is_contiguous());
    TORCH_CHECK(w.scalar_type() == torch::kFloat32 && w.is_contiguous() && w.size(0) == x.size(1) && w.size(1) == 4);
    TORCH_CHECK(state.scalar_type() == torch::kFloat32 && state.is_contiguous() && state.size(-1) == x.size(1));
    check_groups(starts, counts, slots);
    auto out = torch::empty_like(x);
    const int G = (int)starts.numel(), C = (int)x.size(1);
    if (G == 0 || x.size(0) == 0) return out;
    conv_kernel<<<dim3((C + 255) / 256, G), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), state.data_ptr<float>(), starts.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>(), slots.data_ptr<int32_t>(), out.data_ptr<float>(), C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// conv [T][2 kh DK + vh DV] FP32 (after conv and silu), g / beta [T][vh] FP32, state [slots][vh][DV][DK] FP32 (in
// place) -> out [T][vh][DV] FP32.
torch::Tensor scan(torch::Tensor conv, torch::Tensor g, torch::Tensor beta, torch::Tensor state, torch::Tensor starts,
                   torch::Tensor counts, torch::Tensor slots, int64_t kh, int64_t vh, double eps) {
    TORCH_CHECK(conv.is_cuda() && conv.scalar_type() == torch::kFloat32 && conv.is_contiguous() && conv.dim() == 2);
    TORCH_CHECK(conv.size(1) == 2 * kh * DK + vh * DV, "scan: width ", conv.size(1));
    TORCH_CHECK(g.scalar_type() == torch::kFloat32 && beta.scalar_type() == torch::kFloat32 && g.is_contiguous() &&
                beta.is_contiguous() && g.size(1) == vh && beta.size(1) == vh);
    TORCH_CHECK(state.scalar_type() == torch::kFloat32 && state.is_contiguous() && state.size(-1) == DK &&
                state.size(-2) == DV && state.size(-3) == vh);
    TORCH_CHECK(vh % kh == 0);
    check_groups(starts, counts, slots);
    auto out = torch::empty({conv.size(0), vh, DV}, conv.options());
    const int G = (int)starts.numel();
    if (G == 0 || conv.size(0) == 0) return out;
    scan_kernel<<<dim3((unsigned)vh, G, DV / SCAN_WARPS), SCAN_WARPS * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        conv.data_ptr<float>(), g.data_ptr<float>(), beta.data_ptr<float>(), state.data_ptr<float>(),
        starts.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), slots.data_ptr<int32_t>(), out.data_ptr<float>(),
        (int)kh, (int)vh, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> gates(torch::Tensor ba, torch::Tensor dt_bias, torch::Tensor a) {
    TORCH_CHECK(ba.is_cuda() && ba.scalar_type() == torch::kFloat32 && ba.dim() == 2 && ba.stride(1) == 1);
    const int T = (int)ba.size(0), vh = (int)dt_bias.numel();
    TORCH_CHECK(ba.size(1) >= 2 * vh && a.numel() == vh);
    auto beta = torch::empty({T, vh}, ba.options()), g = torch::empty({T, vh}, ba.options());
    const int total = T * vh;
    if (total > 0) {
        gates_kernel<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            ba.data_ptr<float>(), (int)ba.stride(0), dt_bias.contiguous().data_ptr<float>(), a.contiguous().data_ptr<float>(),
            vh, beta.data_ptr<float>(), g.data_ptr<float>(), total);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {beta, g};
}

torch::Tensor gated_rms(torch::Tensor core, torch::Tensor z, torch::Tensor w, double eps) {
    TORCH_CHECK(core.is_cuda() && core.scalar_type() == torch::kFloat32 && core.is_contiguous() && core.dim() == 3);
    TORCH_CHECK(core.size(2) == DV && z.scalar_type() == torch::kFloat32 && z.dim() == 2 && z.stride(1) == 1);
    const int T = (int)core.size(0), vh = (int)core.size(1);
    auto out = torch::empty({T, vh * DV}, core.options().dtype(torch::kBFloat16));
    if (T > 0) {
        gated_rms_kernel<<<T * vh, DV, 0, at::cuda::getCurrentCUDAStream()>>>(
            core.data_ptr<float>(), z.data_ptr<float>(), (int)z.stride(0), w.contiguous().data_ptr<float>(), vh,
            (float)eps, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gates", &gates, "the delta rule's write strength and log decay from beta|alpha projections, one kernel");
    m.def("gated_rms", &gated_rms, "rms_norm(core) * w * silu(z) per value head in BF16, one kernel");
    m.def("conv", &conv, "causal depthwise conv (kernel 4) with SiLU over groups of a sequence's tokens");
    m.def("scan", &scan, "gated delta rule over groups of a sequence's tokens, state in place");
}
