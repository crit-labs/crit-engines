// One kernel per norms call for the shapes Gemma 4's step uses: RMSNorm of rows (optionally weighted), the per-head
// RMSNorm with rotary positions (half-split pairs), and the tanh-GELU product of a gate|up row. One block per row,
// FP32 inside, one rounding to the output dtype; the row's sum of squares reduced in a fixed order (per thread, then
// warp shuffles, then the warps in order), so a row's bits never depend on the rows beside it or their count.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

template <typename T> __device__ __forceinline__ float in(T v);
template <> __device__ __forceinline__ float in<float>(float v) { return v; }
template <> __device__ __forceinline__ float in<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T out(float v);
template <> __device__ __forceinline__ float out<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 out<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

constexpr int THREADS = 1024;   // one block a row: a decode row's latency is the block's, so use the widest block

// sum over the block, fixed order; every thread gets the total
__device__ __forceinline__ float block_sum(float v, float* shared) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    float total = 0.f;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) total += shared[w];
    __syncthreads();
    return total;
}

// Rows of x: row r at x + (r / heads) * stride_t + (r % heads) * stride_h, dim contiguous values; out contiguous.
template <typename Tin, typename Tout, bool ROPE>
__global__ void rms_kernel(const Tin* __restrict__ x, int64_t stride_t, int64_t stride_h, int heads, int dim,
                           const float* __restrict__ weight, float eps, Tout* __restrict__ y,
                           const int64_t* __restrict__ positions, const float* __restrict__ freqs) {
    __shared__ float shared[THREADS / 32];
    const int r = blockIdx.x;
    const Tin* row = x + (int64_t)(r / heads) * stride_t + (int64_t)(r % heads) * stride_h;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = in<Tin>(row[i]);
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, shared) / (float)dim + eps);
    Tout* o = y + (int64_t)r * dim;
    if (!ROPE) {
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            float v = in<Tin>(row[i]) * inv;
            if (weight != nullptr) v *= weight[i];
            o[i] = out<Tout>(v);
        }
        return;
    }
    const int half = dim / 2;
    const float pos = (float)positions[r / heads];
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float a = in<Tin>(row[i]) * inv, b = in<Tin>(row[i + half]) * inv;
        if (weight != nullptr) {
            a *= weight[i];
            b *= weight[i + half];
        }
        const float f = freqs[i];
        if (f != 0.f) {
            float s, c;
            sincosf(pos * f, &s, &c);
            const float ra = a * c - b * s, rb = b * c + a * s;
            a = ra;
            b = rb;
        }
        o[i] = out<Tout>(a);
        o[i + half] = out<Tout>(b);
    }
}

// y[t][i] = gelu_tanh(x[t][i]) * x[t][inter + i]
template <typename Tin, typename Tout>
__global__ void gelu_mul_kernel(const Tin* __restrict__ x, int64_t stride_t, int inter, Tout* __restrict__ y) {
    const int t = blockIdx.x;
    const Tin* row = x + (int64_t)t * stride_t;
    for (int i = threadIdx.x; i < inter; i += blockDim.x) {
        const float g = in<Tin>(row[i]), u = in<Tin>(row[inter + i]);
        const float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
        y[(int64_t)t * inter + i] = out<Tout>(0.5f * g * (1.f + tanhf(inner)) * u);
    }
}

template <typename Tin, typename Tout>
void launch_rms(const torch::Tensor& x, int rows, int heads, int dim, int64_t st, int64_t sh, const float* w,
                double eps, torch::Tensor& y, const int64_t* pos, const float* freqs) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = dim >= 2048 ? THREADS : (dim >= 512 ? 256 : 128);
    if (pos != nullptr)
        rms_kernel<Tin, Tout, true><<<rows, threads, 0, stream>>>(
            reinterpret_cast<const Tin*>(x.data_ptr()), st, sh, heads, dim, w, (float)eps,
            reinterpret_cast<Tout*>(y.data_ptr()), pos, freqs);
    else
        rms_kernel<Tin, Tout, false><<<rows, threads, 0, stream>>>(
            reinterpret_cast<const Tin*>(x.data_ptr()), st, sh, heads, dim, w, (float)eps,
            reinterpret_cast<Tout*>(y.data_ptr()), nullptr, nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// x[t] += rms(y[t]) * w   (a block's output added to the FP32 residual stream after its norm)
template <typename Ty>
__global__ void add_rms_kernel(float* __restrict__ x, const Ty* __restrict__ y, int dim, const float* __restrict__ w,
                               float eps) {
    __shared__ float shared[THREADS / 32];
    const int t = blockIdx.x;
    const Ty* yr = y + (int64_t)t * dim;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = in<Ty>(yr[i]);
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, shared) / (float)dim + eps);
    float* xr = x + (int64_t)t * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) xr[i] += in<Ty>(yr[i]) * inv * w[i];
}

// three BF16 norms of one FP32 row from one sum of squares: o_k = rms(x) * w_k
__global__ void rms3_kernel(const float* __restrict__ x, int dim, const float* __restrict__ w1,
                            const float* __restrict__ w2, const float* __restrict__ w3, float eps,
                            __nv_bfloat16* __restrict__ o1, __nv_bfloat16* __restrict__ o2,
                            __nv_bfloat16* __restrict__ o3) {
    __shared__ float shared[THREADS / 32];
    const int t = blockIdx.x;
    const float* xr = x + (int64_t)t * dim;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum(ss, shared) / (float)dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = xr[i] * inv;
        o1[(int64_t)t * dim + i] = __float2bfloat16(v * w1[i]);
        o2[(int64_t)t * dim + i] = __float2bfloat16(v * w2[i]);
        o3[(int64_t)t * dim + i] = __float2bfloat16(v * w3[i]);
    }
}

// The layer's tail: m = rms(mlp) * w1 + rms(e) * w2; x = (x + rms(m) * w) * scale; and, with wn, the next layer's
// input norm h = rms(x) * wn in BF16. dim <= MAX_DIM (m held in shared memory).
constexpr int MAX_DIM = 4096;
template <typename Tm>
__global__ void ffn_out_kernel(float* __restrict__ x, const Tm* __restrict__ mlp, const float* __restrict__ e, int dim,
                               const float* __restrict__ w1, const float* __restrict__ w2, const float* __restrict__ w,
                               float scale, float eps, const float* __restrict__ wn, __nv_bfloat16* __restrict__ h) {
    __shared__ float shared[THREADS / 32];
    __shared__ float m[MAX_DIM];
    const int t = blockIdx.x;
    const Tm* mr = mlp + (int64_t)t * dim;
    const float* er = e + (int64_t)t * dim;
    float s1 = 0.f, s2 = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float a = in<Tm>(mr[i]), b = er[i];
        s1 += a * a;
        s2 += b * b;
    }
    const float inv1 = rsqrtf(block_sum(s1, shared) / (float)dim + eps);
    const float inv2 = rsqrtf(block_sum(s2, shared) / (float)dim + eps);
    float s3 = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = in<Tm>(mr[i]) * inv1 * w1[i] + er[i] * inv2 * w2[i];
        m[i] = v;
        s3 += v * v;
    }
    const float inv3 = rsqrtf(block_sum(s3, shared) / (float)dim + eps);
    float* xr = x + (int64_t)t * dim;
    float s4 = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = (xr[i] + m[i] * inv3 * w[i]) * scale;
        xr[i] = v;
        s4 += v * v;
    }
    if (wn == nullptr) return;
    const float inv4 = rsqrtf(block_sum(s4, shared) / (float)dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) h[(int64_t)t * dim + i] = __float2bfloat16(xr[i] * inv4 * wn[i]);
}

}  // namespace

void add_rms(torch::Tensor x, torch::Tensor y, torch::Tensor w, double eps) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous() && x.dim() == 2);
    TORCH_CHECK(y.is_contiguous() && y.sizes() == x.sizes() && w.scalar_type() == torch::kFloat32);
    const int T = (int)x.size(0), dim = (int)x.size(1);
    if (T == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (y.scalar_type() == torch::kFloat32)
        add_rms_kernel<float><<<T, THREADS, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(), dim,
                                                         w.data_ptr<float>(), (float)eps);
    else
        add_rms_kernel<__nv_bfloat16><<<T, THREADS, 0, stream>>>(
            x.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), dim, w.data_ptr<float>(),
            (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> rms3(torch::Tensor x, torch::Tensor w1, torch::Tensor w2, torch::Tensor w3, double eps) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous() && x.dim() == 2);
    const int T = (int)x.size(0), dim = (int)x.size(1);
    auto o1 = torch::empty(x.sizes(), x.options().dtype(torch::kBFloat16));
    auto o2 = torch::empty(x.sizes(), x.options().dtype(torch::kBFloat16));
    auto o3 = torch::empty(x.sizes(), x.options().dtype(torch::kBFloat16));
    if (T > 0) {
        rms3_kernel<<<T, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<float>(), dim, w1.data_ptr<float>(), w2.data_ptr<float>(), w3.data_ptr<float>(), (float)eps,
            reinterpret_cast<__nv_bfloat16*>(o1.data_ptr()), reinterpret_cast<__nv_bfloat16*>(o2.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(o3.data_ptr()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {o1, o2, o3};
}

c10::optional<torch::Tensor> ffn_out(torch::Tensor x, torch::Tensor mlp, torch::Tensor e, torch::Tensor w1,
                                     torch::Tensor w2, torch::Tensor w, double scale, double eps,
                                     c10::optional<torch::Tensor> wn) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous() && x.dim() == 2);
    TORCH_CHECK(mlp.is_contiguous() && e.is_contiguous() && e.scalar_type() == torch::kFloat32);
    const int T = (int)x.size(0), dim = (int)x.size(1);
    TORCH_CHECK(dim <= MAX_DIM);
    c10::optional<torch::Tensor> h;
    if (wn.has_value()) h = torch::empty(x.sizes(), x.options().dtype(torch::kBFloat16));
    if (T == 0) return h;
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* wnp = wn.has_value() ? wn->data_ptr<float>() : nullptr;
    __nv_bfloat16* hp = h.has_value() ? reinterpret_cast<__nv_bfloat16*>(h->data_ptr()) : nullptr;
    if (mlp.scalar_type() == torch::kFloat32)
        ffn_out_kernel<float><<<T, THREADS, 0, stream>>>(x.data_ptr<float>(), mlp.data_ptr<float>(), e.data_ptr<float>(),
            dim, w1.data_ptr<float>(), w2.data_ptr<float>(), w.data_ptr<float>(), (float)scale, (float)eps, wnp, hp);
    else
        ffn_out_kernel<__nv_bfloat16><<<T, THREADS, 0, stream>>>(x.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(mlp.data_ptr()), e.data_ptr<float>(), dim, w1.data_ptr<float>(),
            w2.data_ptr<float>(), w.data_ptr<float>(), (float)scale, (float)eps, wnp, hp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return h;
}

// x [rows][dim] or [tokens][heads][dim] (last dim contiguous, any outer strides), FP32 or BF16; weight [dim] FP32 or
// undefined; out dtype FP32 or BF16; with positions [tokens] int64 and freqs [dim/2] FP32, the half-split rotation.
torch::Tensor rms_norm(torch::Tensor x, c10::optional<torch::Tensor> weight, double eps, bool out_fp32,
                       c10::optional<torch::Tensor> positions, c10::optional<torch::Tensor> freqs) {
    TORCH_CHECK(x.is_cuda() && (x.dim() == 2 || x.dim() == 3) && x.stride(-1) == 1);
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kBFloat16);
    const int dim = (int)x.size(-1);
    const int heads = x.dim() == 3 ? (int)x.size(1) : 1;
    const int rows = (int)(x.size(0) * heads);
    const int64_t st = x.stride(0), sh = x.dim() == 3 ? x.stride(1) : 0;
    const float* w = nullptr;
    if (weight.has_value()) {
        TORCH_CHECK(weight->scalar_type() == torch::kFloat32 && weight->is_contiguous() && weight->numel() == dim);
        w = weight->data_ptr<float>();
    }
    const int64_t* pos = nullptr;
    const float* f = nullptr;
    if (positions.has_value()) {
        TORCH_CHECK(freqs.has_value() && freqs->numel() == dim / 2 && freqs->scalar_type() == torch::kFloat32);
        pos = positions->data_ptr<int64_t>();
        f = freqs->contiguous().data_ptr<float>();
    }
    auto y = torch::empty(x.sizes(), x.options().dtype(out_fp32 ? torch::kFloat32 : torch::kBFloat16));
    if (rows == 0) return y;
    const bool in_fp32 = x.scalar_type() == torch::kFloat32;
    if (in_fp32 && out_fp32) launch_rms<float, float>(x, rows, heads, dim, st, sh, w, eps, y, pos, f);
    else if (in_fp32) launch_rms<float, __nv_bfloat16>(x, rows, heads, dim, st, sh, w, eps, y, pos, f);
    else if (out_fp32) launch_rms<__nv_bfloat16, float>(x, rows, heads, dim, st, sh, w, eps, y, pos, f);
    else launch_rms<__nv_bfloat16, __nv_bfloat16>(x, rows, heads, dim, st, sh, w, eps, y, pos, f);
    return y;
}

// x [tokens][2 * inter] (gate | up, last dim contiguous), FP32 or BF16 -> [tokens][inter]
torch::Tensor gelu_mul(torch::Tensor x, bool out_fp32) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.stride(-1) == 1 && x.size(1) % 2 == 0);
    const int inter = (int)(x.size(1) / 2), tokens = (int)x.size(0);
    auto y = torch::empty({tokens, inter}, x.options().dtype(out_fp32 ? torch::kFloat32 : torch::kBFloat16));
    if (tokens == 0) return y;
    auto stream = at::cuda::getCurrentCUDAStream();
    const bool in_fp32 = x.scalar_type() == torch::kFloat32;
    if (in_fp32 && out_fp32)
        gelu_mul_kernel<float, float><<<tokens, 256, 0, stream>>>(x.data_ptr<float>(), x.stride(0), inter, y.data_ptr<float>());
    else if (in_fp32)
        gelu_mul_kernel<float, __nv_bfloat16><<<tokens, 256, 0, stream>>>(
            x.data_ptr<float>(), x.stride(0), inter, reinterpret_cast<__nv_bfloat16*>(y.data_ptr()));
    else if (out_fp32)
        gelu_mul_kernel<__nv_bfloat16, float><<<tokens, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0), inter, y.data_ptr<float>());
    else
        gelu_mul_kernel<__nv_bfloat16, __nv_bfloat16><<<tokens, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0), inter,
            reinterpret_cast<__nv_bfloat16*>(y.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_rms", &add_rms, "x += rms(y) * w in one kernel");
    m.def("rms3", &rms3, "three BF16 norms of one FP32 row from one sum of squares");
    m.def("ffn_out", &ffn_out, "Gemma 4's layer tail and the next layer's input norm in one kernel");
    m.def("rms_norm", &rms_norm, "RMSNorm rows (and heads, with rotary positions) in one kernel");
    m.def("gelu_mul", &gelu_mul, "tanh-GELU(gate) * up of gate|up rows in one kernel");
}
