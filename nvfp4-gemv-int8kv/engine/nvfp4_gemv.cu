// NVFP4 x BF16 GEMV for 1..8 columns on CUDA cores (decode step and speculative verify of the Nanbeige engine).
//
// out[m, n] = alpha[n] * sum_k x[m, k] * e2m1(code[n, k]) * e4m3(scale[n, k / 16])
//
// codes: [N, K/2] bytes, element 2i in the low nibble of byte i; scales: [N, K/16] E4M3; alpha = 1 / global scale.
// Each warp takes ROWS rows; each lane reads 16 bytes (32 codes, two scale groups) of every row per step, decodes
// them with the hardware E2M1 conversion (cvt.rn.f16x2.e2m1x2, sm_100a/sm_120a), multiplies by the columns' inputs
// in FP16 pairs over each 16-element group, and adds group * scale into FP32 accumulators. Every column reduces in
// the same order whatever M is (column invariant), so a verify pass of several columns gives each column the bits
// of a one-column pass.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ void e2m1x8(uint32_t word, __half2 (&out)[4]) {
  uint32_t r0, r1, r2, r3;
  asm("{\n\t.reg .b8 b0, b1, b2, b3;\n\t"
      "mov.b32 {b0, b1, b2, b3}, %4;\n\t"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\n\t"
      "cvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
      "cvt.rn.f16x2.e2m1x2 %2, b2;\n\t"
      "cvt.rn.f16x2.e2m1x2 %3, b3;\n\t}"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(word));
  out[0] = *reinterpret_cast<__half2*>(&r0);
  out[1] = *reinterpret_cast<__half2*>(&r1);
  out[2] = *reinterpret_cast<__half2*>(&r2);
  out[3] = *reinterpret_cast<__half2*>(&r3);
}

__device__ __forceinline__ float e4m3_to_float(uint8_t b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return float(v);
}

template <int M, int ROWS>
__global__ void __launch_bounds__(128) gemv_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w,
                                                   const uint8_t* __restrict__ s, const float* __restrict__ alpha,
                                                   float* __restrict__ out, int N, int K, int ldx, int ldo) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row0 = (blockIdx.x * (blockDim.x >> 5) + warp) * ROWS;
  if (row0 >= N) return;
  const int kb = K >> 1;            // bytes per row
  float acc[ROWS][M];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
  for (int off = lane * 16; off < kb; off += 512) {
    uint4 wv[ROWS];
    uint16_t sv[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      wv[r] = __ldg(reinterpret_cast<const uint4*>(w + (size_t)(row0 + r) * kb + off));
      sv[r] = __ldg(reinterpret_cast<const uint16_t*>(s + (size_t)(row0 + r) * (K >> 4) + (off >> 3)));
    }
    const int k = off * 2;           // first element of this lane's 32
#pragma unroll
    for (int m = 0; m < M; ++m) {
      // 32 inputs of column m as FP16 pairs
      __half2 xh[16];
      const uint4* xp = reinterpret_cast<const uint4*>(x + (size_t)m * ldx + k);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint4 v = __ldg(xp + q);
        const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
        for (int t = 0; t < 4; ++t) xh[q * 4 + t] = __float22half2_rn(__bfloat1622float2(b[t]));
      }
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const uint32_t* words = reinterpret_cast<const uint32_t*>(&wv[r]);
#pragma unroll
        for (int g = 0; g < 2; ++g) {
          __half2 h = __float2half2_rn(0.f);
#pragma unroll
          for (int t = 0; t < 2; ++t) {
            __half2 wd[4];
            e2m1x8(words[g * 2 + t], wd);
#pragma unroll
            for (int u = 0; u < 4; ++u) h = __hfma2(wd[u], xh[g * 8 + t * 4 + u], h);
          }
          const float2 f = __half22float2(h);
          acc[r][m] += (f.x + f.y) * e4m3_to_float(g ? (sv[r] >> 8) : (sv[r] & 0xff));
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = acc[r][m];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) out[(size_t)m * ldo + row0 + r] = v * alpha[row0 + r];
    }
}

template <int M, int ROWS>
void launch(const void* x, const void* w, const void* s, const void* alpha, void* out, int N, int K, int ldx, int ldo,
            cudaStream_t stream) {
  constexpr int WARPS = 4;
  const int blocks = (N + ROWS * WARPS - 1) / (ROWS * WARPS);
  gemv_kernel<M, ROWS><<<blocks, WARPS * 32, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<const uint8_t*>(w), static_cast<const uint8_t*>(s),
      static_cast<const float*>(alpha), static_cast<float*>(out), N, K, ldx, ldo);
}

}  // namespace

template <int ROWS>
int dispatch(const void* x, const void* w, const void* s, const void* alpha, void* out, int M, int N, int K, int ldx,
             int ldo, cudaStream_t st) {
  switch (M) {
    case 1: launch<1, ROWS>(x, w, s, alpha, out, N, K, ldx, ldo, st); break;
    case 2: launch<2, ROWS>(x, w, s, alpha, out, N, K, ldx, ldo, st); break;
    case 3: launch<3, ROWS>(x, w, s, alpha, out, N, K, ldx, ldo, st); break;
    case 4: launch<4, ROWS>(x, w, s, alpha, out, N, K, ldx, ldo, st); break;
    case 5: launch<5, ROWS>(x, w, s, alpha, out, N, K, ldx, ldo, st); break;
    case 6: launch<6, ROWS>(x, w, s, alpha, out, N, K, ldx, ldo, st); break;
    case 7: launch<7, ROWS>(x, w, s, alpha, out, N, K, ldx, ldo, st); break;
    case 8: launch<8, ROWS>(x, w, s, alpha, out, N, K, ldx, ldo, st); break;
    default: return -1;
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

extern "C" int nvfp4_gemv_rows(const void* x, const void* w, const void* s, const void* alpha, void* out, int M, int N,
                               int K, int ldx, int ldo, void* stream, int rows) {
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  switch (rows) {
    case 1: return dispatch<1>(x, w, s, alpha, out, M, N, K, ldx, ldo, st);
    case 2: return dispatch<2>(x, w, s, alpha, out, M, N, K, ldx, ldo, st);
    case 4: return dispatch<4>(x, w, s, alpha, out, M, N, K, ldx, ldo, st);
    case 8: return dispatch<8>(x, w, s, alpha, out, M, N, K, ldx, ldo, st);
    default: return -3;
  }
}

extern "C" int nvfp4_gemv(const void* x, const void* w, const void* s, const void* alpha, void* out, int M, int N,
                          int K, int ldx, int ldo, void* stream) {
  return nvfp4_gemv_rows(x, w, s, alpha, out, M, N, K, ldx, ldo, stream, M <= 2 ? 2 : 4);
}
