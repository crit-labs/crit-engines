"""Projections over the stored encodings: decode GEMVs (up to 16 rows, column-invariant, deterministic split-K) and
prefill GEMMs (Triton on stored weights with the scale and residual in the epilogue; cuBLAS BF16 on exactly widened
NVFP4 weights for long prompts).

y[t, n] = rowscale[n] * sum_k x[t, k] * w[n, k], weights as stored:
  fp8:   w [N, K] float8_e4m3fn, rowscale [N] fp32 (the checkpoint's per-channel scale); bf16 weights take the same
         path with unit scales
  nvfp4: w [N, K/2] uint8 (E2M1 pairs, low nibble first), gscale [N, K/16] float8_e4m3fn group scales, rowscale [N]
         fp32 (1 / weight_global_scale). Pairs are converted by the hardware (cvt.rn.f16x2.e2m1x2); the low nibbles
         (even k) and high nibbles (odd k) meet the matching activation halves in two MMAs, so nothing is
         interleaved. E2M1 x E4M3 products are exact in f16 and bf16.
"""
import torch
import triton
import triton.language as tl

MAX_ROWS = 16


@triton.jit
def _e2m1x2_f32(b):
    # b: uint32 holding one packed byte -> (low, high) fp32 values through the hardware conversion
    h2 = tl.inline_asm_elementwise(
        asm="{ .reg .b8 t; cvt.u8.u32 t, $1; cvt.rn.f16x2.e2m1x2 $0, t; }",
        constraints="=r,r", args=[b], dtype=tl.uint32, is_pure=True, pack=1)
    lo = (h2 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    hi = (h2 >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    return lo, hi


@triton.jit
def _gemv_fp8_kernel(x_ptr, w_ptr, s_ptr, out_ptr, res_ptr, T, N, K, KS,
                     x_stride, out_stride, res_stride, part_stride,
                     BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr, RESIDUAL: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    rows = tl.arange(0, 16)
    n = pid_n * BN + tl.arange(0, BN)
    nmask = n < N
    acc = tl.zeros([16, BN], dtype=tl.float32)
    k_begin = pid_s * KS
    for k0 in range(k_begin, k_begin + KS, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + rows[:, None] * x_stride + k[None, :], mask=rows[:, None] < T, other=0.)
        w = tl.load(w_ptr + n[:, None].to(tl.int64) * K + k[None, :], mask=nmask[:, None], other=0.)
        acc = tl.dot(x.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)), acc)
    omask = (rows[:, None] < T) & nmask[None, :]
    if SPLIT == 1:
        y = acc * tl.load(s_ptr + n, mask=nmask, other=0.)[None, :]
        if RESIDUAL:
            r = tl.load(res_ptr + rows[:, None] * res_stride + n[None, :], mask=omask, other=0.)
            tl.store(res_ptr + rows[:, None] * res_stride + n[None, :], r + y, mask=omask)
        else:
            tl.store(out_ptr + rows[:, None] * out_stride + n[None, :], y.to(out_ptr.dtype.element_ty), mask=omask)
    else:
        tl.store(out_ptr + pid_s * part_stride + rows[:, None] * out_stride + n[None, :], acc, mask=omask)


@triton.jit
def _gemv_nvfp4_kernel(x_ptr, w_ptr, g_ptr, s_ptr, out_ptr, res_ptr, T, N, K, KS,
                       x_stride, out_stride, res_stride, part_stride,
                       BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr, RESIDUAL: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    rows = tl.arange(0, 16)
    n = pid_n * BN + tl.arange(0, BN)
    nmask = n < N
    acc = tl.zeros([16, BN], dtype=tl.float32)
    KH = K // 2
    KG = K // 16
    k_begin = pid_s * KS
    xmask = rows[:, None] < T
    for k0 in range(k_begin, k_begin + KS, BK):
        j = k0 // 2 + tl.arange(0, BK // 2)
        xe = tl.load(x_ptr + rows[:, None] * x_stride + 2 * j[None, :], mask=xmask, other=0.)
        xo = tl.load(x_ptr + rows[:, None] * x_stride + 2 * j[None, :] + 1, mask=xmask, other=0.)
        b = tl.load(w_ptr + n[:, None].to(tl.int64) * KH + j[None, :], mask=nmask[:, None], other=0).to(tl.uint32)
        lo, hi = _e2m1x2_f32(b)
        kg = k0 // 16 + tl.arange(0, BK // 16)
        g = tl.load(g_ptr + n[:, None].to(tl.int64) * KG + kg[None, :], mask=nmask[:, None], other=0.).to(tl.float32)
        g = tl.reshape(tl.broadcast_to(g[:, :, None], (BN, BK // 16, 8)), (BN, BK // 2))
        acc = tl.dot(xe.to(tl.bfloat16), tl.trans((lo * g).to(tl.bfloat16)), acc)
        acc = tl.dot(xo.to(tl.bfloat16), tl.trans((hi * g).to(tl.bfloat16)), acc)
    omask = (rows[:, None] < T) & nmask[None, :]
    if SPLIT == 1:
        y = acc * tl.load(s_ptr + n, mask=nmask, other=0.)[None, :]
        if RESIDUAL:
            r = tl.load(res_ptr + rows[:, None] * res_stride + n[None, :], mask=omask, other=0.)
            tl.store(res_ptr + rows[:, None] * res_stride + n[None, :], r + y, mask=omask)
        else:
            tl.store(out_ptr + rows[:, None] * out_stride + n[None, :], y.to(out_ptr.dtype.element_ty), mask=omask)
    else:
        tl.store(out_ptr + pid_s * part_stride + rows[:, None] * out_stride + n[None, :], acc, mask=omask)


@triton.jit
def _gemv_reduce_kernel(part_ptr, s_ptr, out_ptr, res_ptr, T, N, out_stride, res_stride, part_stride,
                        SPLIT: tl.constexpr, BN: tl.constexpr, RESIDUAL: tl.constexpr):
    t = tl.program_id(1)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    nmask = n < N
    acc = tl.zeros([BN], dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(part_ptr + s * part_stride + t * out_stride + n, mask=nmask, other=0.)
    y = acc * tl.load(s_ptr + n, mask=nmask, other=0.)
    if RESIDUAL:
        r = tl.load(res_ptr + t * res_stride + n, mask=nmask, other=0.)
        tl.store(res_ptr + t * res_stride + n, r + y, mask=nmask)
    else:
        tl.store(out_ptr + t * out_stride + n, y.to(out_ptr.dtype.element_ty), mask=nmask)


# Decode tiles by (kind, N, K) from work/tests/gemv_sweep.py and nvfp4_variants.py (L2-cold, this device):
# (BN, BK, warps, split). Fixed per matrix shape, never chosen per call, so every row count uses one order.
GEMV_TILES = {
    ("nvfp4", 34816, 5120): (32, 256, 4, 4),
    ("nvfp4", 5120, 17408): (32, 256, 4, 4),
    ("fp8", 16384, 5120): (64, 512, 8, 1),
    ("fp8", 5120, 6144): (32, 512, 4, 1),
    ("fp8", 14336, 5120): (32, 256, 4, 1),
    ("fp8", 34816, 5120): (32, 512, 4, 1),
    ("fp8", 5120, 17408): (32, 128, 4, 4),
    ("fp8", 248320, 5120): (64, 512, 4, 1),
}


class Linear:
    """A projection in its stored encoding with its decode tiles (chosen by (kind, N, K) only)."""

    def __init__(self, kind, weight, rowscale, gscale=None):
        self.kind, self.weight, self.rowscale, self.gscale = kind, weight, rowscale, gscale
        self.n = weight.shape[0]
        self.k = weight.shape[1] * (2 if kind == "nvfp4" else 1)
        # one scale for every row (NVFP4 down projections: one global scale), for folding into a later pass
        self.uniform = None
        self.group = None       # fp8 matrix group (attn, gdn, mlp) for the prompt-activation choice
        self.bn, self.bk, self.warps, self.split = GEMV_TILES.get((kind, self.n, self.k), (32, 256, 4, 0))
        if self.split == 0:
            blocks = triton.cdiv(self.n, self.bn)
            split = 1
            while blocks * split < 340 and self.k // (split * 2) >= 2048 and (self.k // (split * 2)) % self.bk == 0:
                split *= 2
            self.split = split
        self.ks = self.k // self.split

    def bytes(self):
        total = self.weight.numel() * self.weight.element_size() + self.rowscale.numel() * 4
        if self.gscale is not None:
            total += self.gscale.numel()
        return total


def gemv(lin, x, out=None, residual=None, partial=None):
    """x [T, K] (bf16 or fp32), T <= 16 -> out [T, N] (its dtype) or residual [T, N] fp32 += result."""
    T = x.shape[0]
    assert T <= MAX_ROWS and x.shape[1] == lin.k
    N, K = lin.n, lin.k
    target = residual if residual is not None else out
    grid = (triton.cdiv(N, lin.bn), lin.split)
    if lin.split == 1:
        if lin.kind == "nvfp4":
            _gemv_nvfp4_kernel[grid](x, lin.weight, lin.gscale, lin.rowscale, target, target, T, N, K, lin.ks,
                                     x.stride(0), target.stride(0), target.stride(0), 0, BN=lin.bn, BK=lin.bk, SPLIT=1,
                                     RESIDUAL=residual is not None, num_warps=lin.warps)
        else:
            _gemv_fp8_kernel[grid](x, lin.weight, lin.rowscale, target, target, T, N, K, lin.ks, x.stride(0),
                                   target.stride(0), target.stride(0), 0, BN=lin.bn, BK=lin.bk, SPLIT=1,
                                   RESIDUAL=residual is not None, num_warps=lin.warps)
        return target
    if partial is None or partial.numel() < lin.split * T * N:
        partial = torch.empty(lin.split * T * N, device=x.device, dtype=torch.float32)
    part = partial[: lin.split * T * N].view(lin.split, T, N)
    if lin.kind == "nvfp4":
        _gemv_nvfp4_kernel[grid](x, lin.weight, lin.gscale, lin.rowscale, part, part, T, N, K, lin.ks, x.stride(0), N,
                                 0, T * N, BN=lin.bn, BK=lin.bk, SPLIT=lin.split, RESIDUAL=False, num_warps=lin.warps)
    else:
        _gemv_fp8_kernel[grid](x, lin.weight, lin.rowscale, part, part, T, N, K, lin.ks, x.stride(0), N, 0, T * N,
                               BN=lin.bn, BK=lin.bk, SPLIT=lin.split, RESIDUAL=False, num_warps=lin.warps)
    _gemv_reduce_kernel[(triton.cdiv(N, 1024), T)](part, lin.rowscale, target, target, T, N, target.stride(0),
                                                   target.stride(0), T * N, SPLIT=lin.split, BN=1024,
                                                   RESIDUAL=residual is not None, num_warps=4)
    return target


# ------------------------------------------------------------------------------------------------- prefill GEMM
@triton.jit
def _gemm_kernel(a_ptr, w_ptr, g_ptr, s_ptr, c_ptr, M, N, K, a_stride, c_stride,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr, FP4: tl.constexpr,
                 SCALE: tl.constexpr, RESIDUAL: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group_id = pid // (GROUP * num_n)
    first_m = group_id * GROUP
    gsz = tl.minimum(num_m - first_m, GROUP)
    pid_m = first_m + (pid % gsz)
    pid_n = (pid % (GROUP * num_n)) // gsz
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mmask = rm[:, None] < M
    nmask = rn[:, None] < N
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        if FP4:
            j = k0 // 2 + tl.arange(0, BK // 2)
            ae = tl.load(a_ptr + rm[:, None].to(tl.int64) * a_stride + 2 * j[None, :], mask=mmask, other=0.)
            ao = tl.load(a_ptr + rm[:, None].to(tl.int64) * a_stride + 2 * j[None, :] + 1, mask=mmask, other=0.)
            b = tl.load(w_ptr + rn[:, None].to(tl.int64) * (K // 2) + j[None, :], mask=nmask, other=0).to(tl.uint32)
            lo, hi = _e2m1x2_f32(b)
            kg = k0 // 16 + tl.arange(0, BK // 16)
            g = tl.load(g_ptr + rn[:, None].to(tl.int64) * (K // 16) + kg[None, :], mask=nmask, other=0.).to(tl.float32)
            g = tl.reshape(tl.broadcast_to(g[:, :, None], (BN, BK // 16, 8)), (BN, BK // 2))
            acc = tl.dot(ae, tl.trans((lo * g).to(tl.bfloat16)), acc)
            acc = tl.dot(ao, tl.trans((hi * g).to(tl.bfloat16)), acc)
        else:
            k = k0 + tl.arange(0, BK)
            a = tl.load(a_ptr + rm[:, None].to(tl.int64) * a_stride + k[None, :], mask=mmask, other=0.)
            w = tl.load(w_ptr + rn[:, None].to(tl.int64) * K + k[None, :], mask=nmask, other=0.)
            acc = tl.dot(a, tl.trans(w.to(tl.bfloat16)), acc)
    if SCALE:
        acc = acc * tl.load(s_ptr + rn, mask=rn < N, other=0.)[None, :]
    cmask = (rm[:, None] < M) & (rn[None, :] < N)
    cp = c_ptr + rm[:, None].to(tl.int64) * c_stride + rn[None, :]
    if RESIDUAL:
        acc = acc + tl.load(cp, mask=cmask, other=0.)
    tl.store(cp, acc.to(c_ptr.dtype.element_ty), mask=cmask)


@triton.jit
def _nvfp4_widen_kernel(w_ptr, g_ptr, out_ptr, N, K, BN: tl.constexpr, BK: tl.constexpr):
    # E2M1 x E4M3 products are exact in bf16 and in f16 (5 significant bits, 2^-10 .. 2688)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k0 = tl.program_id(1) * BK
    j = k0 // 2 + tl.arange(0, BK // 2)
    nmask = n[:, None] < N
    b = tl.load(w_ptr + n[:, None].to(tl.int64) * (K // 2) + j[None, :], mask=nmask, other=0).to(tl.uint32)
    lo, hi = _e2m1x2_f32(b)
    kg = k0 // 16 + tl.arange(0, BK // 16)
    g = tl.load(g_ptr + n[:, None].to(tl.int64) * (K // 16) + kg[None, :], mask=nmask, other=0.).to(tl.float32)
    g = tl.reshape(tl.broadcast_to(g[:, :, None], (BN, BK // 16, 8)), (BN, BK // 2))
    base = out_ptr + n[:, None].to(tl.int64) * K + 2 * j[None, :]
    tl.store(base, (lo * g).to(out_ptr.dtype.element_ty), mask=nmask)
    tl.store(base + 1, (hi * g).to(out_ptr.dtype.element_ty), mask=nmask)


def widen_bf16(lin, out):
    """The projection's weights as exact values [N, K] (bf16, or f16 for an f16 out) without the row scale."""
    if lin.kind != "nvfp4":
        out.copy_(lin.weight)
        return out
    _nvfp4_widen_kernel[(triton.cdiv(lin.n, 32), lin.k // 256)](lin.weight, lin.gscale, out, lin.n, lin.k,
                                                                  BN=32, BK=256, num_warps=4)
    return out


# Prefill GEMM tiles by prompt rows (work/tests/gemm_proto.py on this device): (BM, BN, BK, warps, stages).
def _gemm_tiles(kind, M):
    if kind == "nvfp4":
        return (64, 64, 128, 4, 2) if M <= 128 else (64, 128, 64, 4, 2)
    if M <= 128:
        return (64, 64, 64, 8, 4)
    if M <= 1024:
        return (64, 128, 64, 4, 2)
    return (64, 256, 64, 4, 3)


NVFP4_TRITON_ROWS = 256      # above it NVFP4 prompts use cuBLAS BF16 on widened weights


def stored_mm(lin, x, out=None, residual=None, scale=True, wbuf=None):
    """Prefill projection x [M, K] bf16 -> out [M, N] (fp32 unless given) scaled by the row scales (scale False leaves
    them to the caller), or residual [M, N] fp32 += result."""
    M = x.shape[0]
    N, K = lin.n, lin.k
    if lin.kind == "nvfp4" and M > NVFP4_TRITON_ROWS:
        wt = wbuf[: N * K].view(N, K)
        widen_bf16(lin, wt)
        y = torch.mm(x, wt.t(), out_dtype=torch.float32)
        if scale:
            if lin.uniform is not None:
                if residual is not None:
                    residual.add_(y, alpha=lin.uniform)
                    return residual
                y.mul_(lin.uniform)
            else:
                y.mul_(lin.rowscale)
        if residual is not None:
            residual.add_(y)
            return residual
        if out is not None:
            out.copy_(y)
            return out
        return y
    target = residual if residual is not None else (out if out is not None else
                                                    torch.empty(M, N, dtype=torch.float32, device=x.device))
    BM, BN, BK, warps, stages = _gemm_tiles(lin.kind, M)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _gemm_kernel[grid](x, lin.weight, lin.gscale if lin.gscale is not None else lin.rowscale, lin.rowscale, target, M,
                       N, K, x.stride(0), target.stride(0), BM=BM, BN=BN, BK=BK, GROUP=8, FP4=lin.kind == "nvfp4",
                       SCALE=scale, RESIDUAL=residual is not None, num_warps=warps, num_stages=stages)
    return target


# ------------------------------------------------------------------------------------------------- FP8 prompt activations
@triton.jit
def _quant_rows_kernel(x_ptr, q_ptr, s_ptr, K, x_stride, BK: tl.constexpr):
    # per-row (per-token) dynamic FP8 E4M3: scale = amax / 448, the checkpoint input_activations scheme
    r = tl.program_id(0)
    amax = tl.zeros([BK], dtype=tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + r * x_stride + k, mask=k < K, other=0.).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(x))
    s = tl.maximum(tl.max(amax, 0), 1e-12) / 448.
    tl.store(s_ptr + r, s)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + r * x_stride + k, mask=k < K, other=0.).to(tl.float32)
        tl.store(q_ptr + r * K + k, tl.clamp(x / s, -448., 448.).to(tl.float8e4nv), mask=k < K)


def fp8_mm(lin, x, residual=None, scale=True, ones=None):
    """Prefill projection with FP8 per-token activations (cuBLASLt FP8 GEMM, row-wise scales): x [M, K] bf16 ->
    fp32 [M, N], or residual += it. Only for FP8-stored weights."""
    M, Kd = x.shape
    q = torch.empty(M, Kd, dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(M, 1, dtype=torch.float32, device=x.device)
    _quant_rows_kernel[(M,)](x, q, s, Kd, x.stride(0), BK=1024, num_warps=4)
    sb = lin.rowscale.view(1, -1) if scale else ones[: lin.n].view(1, -1)
    y = torch._scaled_mm(q, lin.weight.t(), scale_a=s, scale_b=sb, out_dtype=torch.float32)
    if residual is not None:
        residual.add_(y)
        return residual
    return y
