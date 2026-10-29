"""Triton kernels of the Nanbeige4.3-33B NVFP4 engine (own code; providers checked are in the candidate's rationale).

NVFP4 (compressed-tensors nvfp4-pack-quantized): a row of K weights is K/2 bytes of E2M1 codes, the low nibble first,
with one E4M3 scale per 16 weights and an FP32 global scale g per matrix: w = e2m1(code) * scale / g. The kernels
build each E2M1 value from its bits as an FP16 number (sign at bit 15, the three magnitude bits at bits 11..9), which
is the value times 2^-14; the 2^14 is folded into the per-row factor `alpha` = 2^14 / g, kept per output row so that
matrices with different global scales can be stacked (q/k/v, gate/up).
"""
import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

E2M1_UNIT = 16384.0  # 2^14: an E2M1 code's FP16 bit pattern is its value times 2^-14


@triton.jit
def _e2m1(codes):
    """E2M1 codes (uint8, low 4 bits) to FP32 values times 2^-14."""
    bits = ((codes & 7).to(tl.uint16) << 9) | ((codes & 8).to(tl.uint16) << 12)
    return bits.to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _gemv_kernel(x_ptr, w_ptr, s_ptr, alpha_ptr, out_ptr, N, K,
                 stride_xm, stride_om,
                 M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """out[m, n] = sum_k x[m, k] * w[n, k] for M <= 8 columns; every column reduces in the same order (column
    invariant), so a verify pass of several columns gives each column the bits of a one-column pass."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    HALF: tl.constexpr = BLOCK_K // 2
    GROUPS: tl.constexpr = BLOCK_K // 16
    j = tl.arange(0, HALF)
    g = tl.arange(0, GROUPS)
    acc0 = tl.zeros([BLOCK_N], tl.float32)
    acc1 = tl.zeros([BLOCK_N], tl.float32)
    acc2 = tl.zeros([BLOCK_N], tl.float32)
    acc3 = tl.zeros([BLOCK_N], tl.float32)
    acc4 = tl.zeros([BLOCK_N], tl.float32)
    acc5 = tl.zeros([BLOCK_N], tl.float32)
    acc6 = tl.zeros([BLOCK_N], tl.float32)
    acc7 = tl.zeros([BLOCK_N], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        packed = tl.load(w_ptr + rows[:, None] * (K // 2) + (k0 // 2 + j)[None, :], mask=rmask[:, None], other=0)
        scale = tl.load(s_ptr + rows[:, None] * (K // 16) + (k0 // 16 + g)[None, :], mask=rmask[:, None], other=0.0)
        scale = scale.to(tl.float32)
        lo = tl.reshape(_e2m1(packed), (BLOCK_N, GROUPS, 8))
        hi = tl.reshape(_e2m1(packed >> 4), (BLOCK_N, GROUPS, 8))
        for m in tl.static_range(M):
            xe = tl.load(x_ptr + m * stride_xm + k0 + 2 * j).to(tl.float32)
            xo = tl.load(x_ptr + m * stride_xm + k0 + 2 * j + 1).to(tl.float32)
            xe = tl.reshape(xe, (GROUPS, 8))
            xo = tl.reshape(xo, (GROUPS, 8))
            part = tl.sum(lo * xe[None, :, :] + hi * xo[None, :, :], axis=2)
            contrib = tl.sum(part * scale, axis=1)
            if m == 0:
                acc0 += contrib
            elif m == 1:
                acc1 += contrib
            elif m == 2:
                acc2 += contrib
            elif m == 3:
                acc3 += contrib
            elif m == 4:
                acc4 += contrib
            elif m == 5:
                acc5 += contrib
            elif m == 6:
                acc6 += contrib
            else:
                acc7 += contrib
    alpha = tl.load(alpha_ptr + rows, mask=rmask, other=0.0)
    for m in tl.static_range(M):
        if m == 0:
            acc = acc0
        elif m == 1:
            acc = acc1
        elif m == 2:
            acc = acc2
        elif m == 3:
            acc = acc3
        elif m == 4:
            acc = acc4
        elif m == 5:
            acc = acc5
        elif m == 6:
            acc = acc6
        else:
            acc = acc7
        tl.store(out_ptr + m * stride_om + rows, acc * alpha, mask=rmask)


def gemv(x, w, s, alpha, out, block_n=8, block_k=512, warps=4):
    """x [M, K] bf16 (M <= 8), w [N, K/2] uint8, s [N, K/16] float8_e4m3fn, alpha [N] fp32 -> out [M, N] fp32."""
    M, K = x.shape
    N = w.shape[0]
    assert M <= 8 and K % block_k == 0
    _gemv_kernel[(triton.cdiv(N, block_n),)](x, w, s, alpha, out, N, K, x.stride(0), out.stride(0),
                                             M=M, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=warps)
    return out


@triton.jit
def _dequant_kernel(w_ptr, s_ptr, alpha_ptr, out_ptr, K, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    k0 = tl.program_id(1) * BLOCK_K
    HALF: tl.constexpr = BLOCK_K // 2
    j = tl.arange(0, HALF)
    packed = tl.load(w_ptr + row * (K // 2) + k0 // 2 + j)
    scale = tl.load(s_ptr + row * (K // 16) + (k0 + 2 * j) // 16).to(tl.float32)
    alpha = tl.load(alpha_ptr + row)
    lo = _e2m1(packed) * scale * alpha
    hi = _e2m1(packed >> 4) * scale * alpha
    vals = tl.interleave(lo, hi)
    tl.store(out_ptr + row * K + k0 + tl.arange(0, BLOCK_K), vals.to(out_ptr.dtype.element_ty))


def dequant(w, s, alpha, out):
    """NVFP4 rows to BF16 or FP16 [N, K] (out's dtype; BF16 exact: an E2M1 value times an E4M3 scale times 2^k fits BF16 unless the global scale
    is not a power of two, where the product rounds once to BF16)."""
    N = w.shape[0]
    K = w.shape[1] * 2
    assert K % 512 == 0
    _dequant_kernel[(N, K // 512)](w, s, alpha, out, K, BLOCK_K=512, num_warps=4)
    return out


@triton.jit
def _add_rmsnorm_kernel(x_ptr, res_ptr, w_ptr, out_ptr, D, eps, HAS_X: tl.constexpr, BLOCK: tl.constexpr):
    """res += x (FP32 residual stream); out = bf16(rmsnorm(res) * w), the norm in FP32 as transformers' LlamaRMSNorm."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    r = tl.load(res_ptr + row * D + offs, mask=mask, other=0.0)
    if HAS_X:
        r += tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(res_ptr + row * D + offs, r, mask=mask)
    var = tl.sum(r * r, axis=0) / D
    normed = r * tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * D + offs, (w * normed).to(out_ptr.dtype.element_ty), mask=mask)


def add_rmsnorm(x, res, w, out, eps):
    rows, D = res.shape
    _add_rmsnorm_kernel[(rows,)](x if x is not None else res, res, w, out, D, eps, HAS_X=x is not None,
                                 BLOCK=triton.next_power_of_2(D), num_warps=8)
    return out


@triton.jit
def _kv_store(row_ptr, scale_ptr, a, b, i, HALF: tl.constexpr, KV_MODE: tl.constexpr):
    """One KV head's row (halves a, b, FP32) into the cache: KV_MODE 0 FP8 E4M3 (scale amax / 448), 1 INT8
    (symmetric, scale amax / 127, round to nearest even), 2 BF16 (scale 1)."""
    amax = tl.maximum(tl.max(tl.abs(a), axis=0), tl.max(tl.abs(b), axis=0))
    if KV_MODE == 0:
        scale = tl.maximum(amax / 448.0, 1e-30)
        tl.store(row_ptr + i, (a / scale).to(tl.float8e4nv))
        tl.store(row_ptr + HALF + i, (b / scale).to(tl.float8e4nv))
    elif KV_MODE == 1:
        scale = tl.maximum(amax / 127.0, 1e-30)
        tl.store(row_ptr + i, libdevice.rint(a / scale).to(tl.int8))
        tl.store(row_ptr + HALF + i, libdevice.rint(b / scale).to(tl.int8))
    else:
        scale = amax * 0.0 + 1.0
        tl.store(row_ptr + i, a.to(tl.bfloat16))
        tl.store(row_ptr + HALF + i, b.to(tl.bfloat16))
    tl.store(scale_ptr, scale)


@triton.jit
def _rope_kv_kernel(qkv_ptr, cos_ptr, sin_ptr, pos_ptr, q_out_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pos0,
                    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, POS_FROM_PTR: tl.constexpr,
                    KV_MODE: tl.constexpr):
    """One token: RoPE (rotate_half form) on its q and k heads in FP32; q to BF16; k and v to FP8 E4M3 with one FP32
    scale per KV head (amax / 448), written at the token's position in the cache."""
    t = tl.program_id(0)
    head = tl.program_id(1)
    if POS_FROM_PTR:
        pos = tl.load(pos_ptr) + t
    else:
        pos = pos0 + t
    HALF: tl.constexpr = D // 2
    i = tl.arange(0, HALF)
    W: tl.constexpr = (HQ + 2 * HKV) * D
    base = qkv_ptr + t * W
    c = tl.load(cos_ptr + pos * HALF + i)
    s = tl.load(sin_ptr + pos * HALF + i)
    if head < HQ:
        a = tl.load(base + head * D + i).to(tl.float32)
        b = tl.load(base + head * D + HALF + i).to(tl.float32)
        tl.store(q_out_ptr + (t * HQ + head) * D + i, (a * c - b * s).to(tl.bfloat16))
        tl.store(q_out_ptr + (t * HQ + head) * D + HALF + i, (b * c + a * s).to(tl.bfloat16))
    elif head < HQ + HKV:
        h = head - HQ
        a = tl.load(base + (HQ + h) * D + i).to(tl.float32)
        b = tl.load(base + (HQ + h) * D + HALF + i).to(tl.float32)
        ra = a * c - b * s
        rb = b * c + a * s
        _kv_store(kc_ptr + (pos * HKV + h) * D, ks_ptr + pos * HKV + h, ra, rb, i, HALF, KV_MODE)
    else:
        h = head - HQ - HKV
        a = tl.load(base + (HQ + HKV + h) * D + i).to(tl.float32)
        b = tl.load(base + (HQ + HKV + h) * D + HALF + i).to(tl.float32)
        _kv_store(vc_ptr + (pos * HKV + h) * D, vs_ptr + pos * HKV + h, a, b, i, HALF, KV_MODE)


KV_MODES = {torch.float8_e4m3fn: 0, torch.int8: 1, torch.bfloat16: 2}


def rope_kv(qkv, cos, sin, q_out, kc, vc, ks, vs, hq, hkv, d, pos_tensor=None, pos0=0):
    """qkv [T, (hq + 2 hkv) d] fp32 at positions pos0.. (or pos_tensor[0]..) -> q_out [T, hq, d] bf16, cache writes."""
    T = qkv.shape[0]
    _rope_kv_kernel[(T, hq + 2 * hkv)](qkv, cos, sin, pos_tensor if pos_tensor is not None else qkv, q_out, kc, vc, ks,
                                       vs, pos0, HQ=hq, HKV=hkv, D=d, POS_FROM_PTR=pos_tensor is not None,
                                       KV_MODE=KV_MODES[kc.dtype], num_warps=2)
    return q_out


@triton.jit
def _silu_mul_kernel(g_ptr, u_ptr, out_ptr, F, stride_g, stride_u, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < F
    g = tl.load(g_ptr + row * stride_g + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + row * stride_u + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * F + offs, (g / (1.0 + tl.exp(-g)) * u).to(out_ptr.dtype.element_ty), mask=mask)


def silu_mul(gate, up, out):
    """gate, up [T, F] (row strides of their own) -> out [T, F] bf16 = silu(gate) * up in FP32."""
    T, F = gate.shape
    _silu_mul_kernel[(T, triton.cdiv(F, 2048))](gate, up, out, F, gate.stride(0), up.stride(0), BLOCK=2048,
                                                num_warps=4)
    return out


@triton.jit
def _attn_decode_kernel(q_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, len_ptr, po_ptr, pm_ptr, pl_ptr, sm_scale, n_splits,
                        HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, BLOCK_H: tl.constexpr, D: tl.constexpr,
                        BLOCK_N: tl.constexpr, SPLIT: tl.constexpr):
    """Split-K decode attention for one query token over the FP8 cache: each program takes one KV head (its G query
    heads together) and one SPLIT-token slice of the held positions; slices past the held length do nothing."""
    kvh = tl.program_id(0)
    sp = tl.program_id(1)
    L = tl.load(len_ptr)
    start = sp * SPLIT
    if start < L:
        hs = tl.arange(0, BLOCK_H)
        hmask = hs < G
        d = tl.arange(0, D)
        q = tl.load(q_ptr + (kvh * G + hs)[:, None] * D + d[None, :], mask=hmask[:, None], other=0.0)
        m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_H], tl.float32)
        acc = tl.zeros([BLOCK_H, D], tl.float32)
        end = tl.minimum(start + SPLIT, L)
        for n0 in range(start, end, BLOCK_N):
            n = n0 + tl.arange(0, BLOCK_N)
            nm = n < end
            k = tl.load(kc_ptr + (n[:, None] * HKV + kvh) * D + d[None, :], mask=nm[:, None], other=0.0)
            ksc = tl.load(ks_ptr + n * HKV + kvh, mask=nm, other=0.0)
            s = tl.dot(q, tl.trans(k.to(tl.bfloat16)))
            s = s * (ksc * sm_scale)[None, :]
            s = tl.where(nm[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(vc_ptr + (n[:, None] * HKV + kvh) * D + d[None, :], mask=nm[:, None], other=0.0)
            vsc = tl.load(vs_ptr + n * HKV + kvh, mask=nm, other=0.0)
            acc = acc * alpha[:, None] + tl.dot((p * vsc[None, :]).to(tl.bfloat16), v.to(tl.bfloat16))
            m_i = m_new
        base = (kvh * n_splits + sp) * BLOCK_H + hs
        tl.store(po_ptr + base[:, None] * D + d[None, :], acc)
        tl.store(pm_ptr + base, m_i)
        tl.store(pl_ptr + base, l_i)


@triton.jit
def _attn_reduce_kernel(po_ptr, pm_ptr, pl_ptr, len_ptr, out_ptr, n_splits, G: tl.constexpr, BLOCK_H: tl.constexpr,
                        D: tl.constexpr, SPLIT: tl.constexpr):
    head = tl.program_id(0)
    kvh = head // G
    hh = head % G
    L = tl.load(len_ptr)
    used = (L + SPLIT - 1) // SPLIT
    d = tl.arange(0, D)
    m = float("-inf")
    l = 0.0
    acc = tl.zeros([D], tl.float32)
    for sp in range(0, used):
        base = (kvh * n_splits + sp) * BLOCK_H + hh
        mi = tl.load(pm_ptr + base)
        li = tl.load(pl_ptr + base)
        oi = tl.load(po_ptr + base * D + d)
        m_new = tl.maximum(m, mi)
        a = tl.exp(m - m_new)
        b = tl.exp(mi - m_new)
        acc = acc * a + oi * b
        l = l * a + li * b
        m = m_new
    tl.store(out_ptr + head * D + d, (acc / l).to(tl.bfloat16))


class DecodeAttention:
    """Workspace for split-K decode attention sized for the engine's capacity (fixed grid, so it captures in a graph)."""

    SPLIT = 1024

    def __init__(self, capacity, hq, hkv, d, device):
        self.hq, self.hkv, self.d = hq, hkv, d
        self.g = hq // hkv
        self.block_h = max(16, triton.next_power_of_2(self.g))
        self.n_splits = triton.cdiv(capacity, self.SPLIT)
        rows = hkv * self.n_splits * self.block_h
        self.po = torch.empty(rows, d, dtype=torch.float32, device=device)
        self.pm = torch.empty(rows, dtype=torch.float32, device=device)
        self.pl = torch.empty(rows, dtype=torch.float32, device=device)

    def __call__(self, q, kc, vc, ks, vs, length, out):
        """q [hq, d] bf16 for the token at position length - 1; length a device int32 [1] -> out [hq, d] bf16."""
        scale = self.d ** -0.5
        _attn_decode_kernel[(self.hkv, self.n_splits)](q, kc, vc, ks, vs, length, self.po, self.pm, self.pl, scale,
                                                       self.n_splits, HQ=self.hq, HKV=self.hkv, G=self.g,
                                                       BLOCK_H=self.block_h, D=self.d, BLOCK_N=64, SPLIT=self.SPLIT,
                                                       num_warps=4, num_stages=2)
        _attn_reduce_kernel[(self.hq,)](self.po, self.pm, self.pl, length, out, self.n_splits, G=self.g,
                                        BLOCK_H=self.block_h, D=self.d, SPLIT=self.SPLIT, num_warps=4)
        return out


@triton.jit
def _attn_prefill_kernel(q_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, out_ptr, T, pos0, sm_scale,
                         HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Causal flash attention for a chunk of T query tokens at positions pos0.. over the FP8 cache [0, pos0 + T).
    K and V convert exactly from E4M3 to BF16; K's scale multiplies the scores, V's scale the probabilities."""
    mb = tl.program_id(0)
    head = tl.program_id(1)
    kvh = head // G
    rows = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < T
    d = tl.arange(0, D)
    q = tl.load(q_ptr + (rows[:, None] * HQ + head) * D + d[None, :], mask=rmask[:, None], other=0.0)
    qpos = pos0 + rows
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    end = tl.minimum(pos0 + (mb + 1) * BLOCK_M, pos0 + T)
    for n0 in range(0, end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        nm = n < end
        k = tl.load(kc_ptr + (n[:, None] * HKV + kvh) * D + d[None, :], mask=nm[:, None], other=0.0)
        ksc = tl.load(ks_ptr + n * HKV + kvh, mask=nm, other=0.0)
        s = tl.dot(q, tl.trans(k.to(tl.bfloat16)))
        s = s * (ksc * sm_scale)[None, :]
        s = tl.where((n[None, :] <= qpos[:, None]) & nm[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(vc_ptr + (n[:, None] * HKV + kvh) * D + d[None, :], mask=nm[:, None], other=0.0)
        vsc = tl.load(vs_ptr + n * HKV + kvh, mask=nm, other=0.0)
        acc = acc * alpha[:, None] + tl.dot((p * vsc[None, :]).to(tl.bfloat16), v.to(tl.bfloat16))
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(out_ptr + (rows[:, None] * HQ + head) * D + d[None, :], out.to(out_ptr.dtype.element_ty),
             mask=rmask[:, None])


def attn_prefill(q, kc, vc, ks, vs, out, pos0, hq, hkv, d):
    """q [T, hq, d] bf16 at positions pos0..pos0+T-1 (their K and V already in the cache) -> out [T, hq, d] bf16."""
    T = q.shape[0]
    block_m = 64
    _attn_prefill_kernel[(triton.cdiv(T, block_m), hq)](q, kc, vc, ks, vs, out, T, pos0, d ** -0.5, HQ=hq, HKV=hkv,
                                                        G=hq // hkv, D=d, BLOCK_M=block_m, BLOCK_N=64,
                                                        num_warps=4, num_stages=2)
    return out


@triton.jit
def _skinny_kernel(x_ptr, w_ptr, s_ptr, pre_ptr, post_ptr, out_ptr, M, N, Kd, stride_xm, stride_om,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """out[m, n] = sum_k x[m, k] w[n, k] for up to BLOCK_M rows on tensor cores: each B tile is decoded in registers to
    BF16 exactly (E2M1 value times its E4M3 scale times the row's power of two `pre`), FP32 MMA accumulation, and the
    rest of the row's global scale (`post` = 1 / (g * pre)) in the epilogue. The weights are read once for all rows."""
    pid = tl.program_id(0)
    pm = tl.program_id(1)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    HALF: tl.constexpr = BLOCK_K // 2
    G: tl.constexpr = BLOCK_K // 16
    kk = tl.arange(0, BLOCK_K)
    jj = tl.arange(0, HALF)
    gg = tl.arange(0, G)
    pre = tl.load(pre_ptr + rn)
    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for k0 in range(0, Kd, BLOCK_K):
        x = tl.load(x_ptr + rm[:, None] * stride_xm + (k0 + kk)[None, :], mask=rm[:, None] < M, other=0.0)
        packed = tl.load(w_ptr + rn[:, None] * (Kd // 2) + (k0 // 2 + jj)[None, :])
        sc = tl.load(s_ptr + rn[:, None] * (Kd // 16) + (k0 // 16 + gg)[None, :]).to(tl.float32)
        sc = sc * pre[:, None] * 16384.0
        lo = tl.reshape(tl.reshape(_e2m1(packed), (BLOCK_N, G, 8)) * sc[:, :, None], (BLOCK_N, HALF))
        hi = tl.reshape(tl.reshape(_e2m1(packed >> 4), (BLOCK_N, G, 8)) * sc[:, :, None], (BLOCK_N, HALF))
        b = tl.interleave(lo, hi).to(tl.bfloat16)
        acc = tl.dot(x.to(tl.bfloat16), tl.trans(b), acc)
    post = tl.load(post_ptr + rn)
    out = acc * post[None, :]
    tl.store(out_ptr + rm[:, None] * stride_om + rn[None, :], out.to(out_ptr.dtype.element_ty), mask=rm[:, None] < M)


def skinny(x, w, s, pre, post, out, block_m=64):
    """x [M, K] bf16 (any M; tiles of block_m rows) times the NVFP4 matrix -> out [M, N] (out's dtype)."""
    M, Kd = x.shape
    N = w.shape[0]
    _skinny_kernel[(N // 32, triton.cdiv(M, block_m))](x, w, s, pre, post, out, M, N, Kd, x.stride(0), out.stride(0),
                                                       BLOCK_M=block_m, BLOCK_N=32, BLOCK_K=256, num_warps=4,
                                                       num_stages=3)
    return out


@triton.jit
def _attn_prefill_gqa_kernel(q_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, out_ptr, T, pos0, qk_scale,
                             HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr,
                             BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr):
    """Causal flash attention of a chunk of T query tokens at positions pos0.. over the INT8/FP8/BF16 cache, one
    program per KV head and BLOCK_T tokens: its G query heads (padded to GP) share every K and V tile it loads.
    Row r of the tile is token r // GP, head r % GP. Scores in the exp2 domain (qk_scale includes log2(e))."""
    tb = tl.program_id(0)
    kvh = tl.program_id(1)
    ROWS: tl.constexpr = BLOCK_T * GP
    r = tl.arange(0, ROWS)
    t = tb * BLOCK_T + r // GP
    h = r % GP
    valid = (t < T) & (h < G)
    d = tl.arange(0, D)
    q = tl.load(q_ptr + ((t * HQ + kvh * G + h)[:, None]) * D + d[None, :], mask=valid[:, None], other=0.0)
    qpos = pos0 + t
    m_i = tl.full([ROWS], float("-inf"), tl.float32)
    l_i = tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, D], tl.float32)
    end = pos0 + tl.minimum((tb + 1) * BLOCK_T, T)
    for n0 in range(0, end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        nm = n < end
        k = tl.load(kc_ptr + (n[:, None] * HKV + kvh) * D + d[None, :], mask=nm[:, None], other=0.0)
        ksc = tl.load(ks_ptr + n * HKV + kvh, mask=nm, other=0.0)
        sc = tl.dot(q, tl.trans(k.to(tl.bfloat16)))
        sc = sc * (ksc * qk_scale)[None, :]
        sc = tl.where((n[None, :] <= qpos[:, None]) & nm[None, :], sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        m_new = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(sc - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(vc_ptr + (n[:, None] * HKV + kvh) * D + d[None, :], mask=nm[:, None], other=0.0)
        vsc = tl.load(vs_ptr + n * HKV + kvh, mask=nm, other=0.0)
        acc = acc * alpha[:, None] + tl.dot((p * vsc[None, :]).to(tl.bfloat16), v.to(tl.bfloat16))
        m_i = m_new
    out = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
    tl.store(out_ptr + ((t * HQ + kvh * G + h)[:, None]) * D + d[None, :], out.to(out_ptr.dtype.element_ty),
             mask=valid[:, None])


def attn_prefill_gqa(q, kc, vc, ks, vs, out, pos0, hq, hkv, d, block_t=16, block_n=64, warps=8, stages=2):
    T = q.shape[0]
    g = hq // hkv
    gp = triton.next_power_of_2(g)
    _attn_prefill_gqa_kernel[(triton.cdiv(T, block_t), hkv)](q, kc, vc, ks, vs, out, T, pos0,
                                                             d ** -0.5 * 1.4426950408889634, HQ=hq, HKV=hkv, G=g,
                                                             GP=gp, D=d, BLOCK_T=block_t, BLOCK_N=block_n,
                                                             num_warps=warps, num_stages=stages)
    return out


@triton.jit
def _attn_prefill_i8_kernel(q_ptr, qs_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, out_ptr, T, pos0, qk_scale,
                            HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PV_FP16_ACC: tl.constexpr):
    """Causal flash attention with INT8 Q.K^T on tensor cores (SageAttention's scheme): Q quantized per token and
    head (qs = amax / 127), K the INT8 cache with its per-position scales, scores exact in INT32 then scaled; P.V in
    BF16 with V converted exactly from INT8 and its scale folded into P."""
    mb = tl.program_id(0)
    head = tl.program_id(1)
    kvh = head // G
    rows = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < T
    d = tl.arange(0, D)
    q = tl.load(q_ptr + (rows[:, None] * HQ + head) * D + d[None, :], mask=rmask[:, None], other=0)
    qs = tl.load(qs_ptr + rows * HQ + head, mask=rmask, other=0.0) * qk_scale
    qpos = pos0 + rows
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    end = tl.minimum(pos0 + (mb + 1) * BLOCK_M, pos0 + T)
    for n0 in range(0, end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        nm = n < end
        k = tl.load(kc_ptr + (n[:, None] * HKV + kvh) * D + d[None, :], mask=nm[:, None], other=0)
        ksc = tl.load(ks_ptr + n * HKV + kvh, mask=nm, other=0.0)
        si = tl.dot(q, tl.trans(k))
        sc = si.to(tl.float32) * qs[:, None] * ksc[None, :]
        sc = tl.where((n[None, :] <= qpos[:, None]) & nm[None, :], sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        p = tl.exp2(sc - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(vc_ptr + (n[:, None] * HKV + kvh) * D + d[None, :], mask=nm[:, None], other=0)
        vsc = tl.load(vs_ptr + n * HKV + kvh, mask=nm, other=0.0)
        if PV_FP16_ACC:
            # P.V on FP16 tensor cores accumulating in FP16 over the tile (twice the FP32-accumulate rate), the tile's
            # sum then added into the FP32 accumulator.
            pv = tl.dot((p * vsc[None, :]).to(tl.float16), v.to(tl.float16), out_dtype=tl.float16)
            acc = acc * alpha[:, None] + pv.to(tl.float32)
        else:
            acc = acc * alpha[:, None] + tl.dot((p * vsc[None, :]).to(tl.bfloat16), v.to(tl.bfloat16))
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(out_ptr + (rows[:, None] * HQ + head) * D + d[None, :], out.to(out_ptr.dtype.element_ty),
             mask=rmask[:, None])


def quantize_q(q):
    """q [T, H, D] -> (int8 codes, FP32 scale per token and head)."""
    qf = q.float()
    scale = (qf.abs().amax(-1) / 127.0).clamp_min(1e-30)
    return torch.round(qf / scale[..., None]).to(torch.int8), scale.contiguous()


def attn_prefill_i8(q8, qs, kc, vc, ks, vs, out, pos0, hq, hkv, d, block_m=64, block_n=64, warps=4, stages=3,
                    pv_fp16_acc=False):
    T = q8.shape[0]
    _attn_prefill_i8_kernel[(triton.cdiv(T, block_m), hq)](q8, qs, kc, vc, ks, vs, out, T, pos0,
                                                           d ** -0.5 * 1.4426950408889634, HQ=hq, HKV=hkv, G=hq // hkv,
                                                           D=d, BLOCK_M=block_m, BLOCK_N=block_n,
                                                           PV_FP16_ACC=pv_fp16_acc, num_warps=warps, num_stages=stages)
    return out


@triton.jit
def _attn_verify_kernel(q_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, len_ptr, po_ptr, pm_ptr, pl_ptr, sm_scale, n_splits,
                        T: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
                        D: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT: tl.constexpr):
    """Split-K attention for T query tokens at positions L - T .. L - 1 (a verify block) over the cache, each token
    seeing the positions up to its own: one program per KV head and SPLIT-token slice, rows = token x query head."""
    kvh = tl.program_id(0)
    sp = tl.program_id(1)
    L = tl.load(len_ptr)
    start = sp * SPLIT
    ROWS: tl.constexpr = T * GP
    if start < L:
        r = tl.arange(0, ROWS)
        t = r // GP
        h = r % GP
        valid = h < G
        d = tl.arange(0, D)
        q = tl.load(q_ptr + ((t * HQ + kvh * G + h)[:, None]) * D + d[None, :], mask=valid[:, None], other=0.0)
        qpos = L - T + t
        m_i = tl.full([ROWS], float("-inf"), tl.float32)
        l_i = tl.zeros([ROWS], tl.float32)
        acc = tl.zeros([ROWS, D], tl.float32)
        end = tl.minimum(start + SPLIT, L)
        for n0 in range(start, end, BLOCK_N):
            n = n0 + tl.arange(0, BLOCK_N)
            nm = n < end
            k = tl.load(kc_ptr + (n[:, None] * HKV + kvh) * D + d[None, :], mask=nm[:, None], other=0.0)
            ksc = tl.load(ks_ptr + n * HKV + kvh, mask=nm, other=0.0)
            s = tl.dot(q, tl.trans(k.to(tl.bfloat16)))
            s = s * (ksc * sm_scale)[None, :]
            s = tl.where(nm[None, :] & (n[None, :] <= qpos[:, None]), s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(vc_ptr + (n[:, None] * HKV + kvh) * D + d[None, :], mask=nm[:, None], other=0.0)
            vsc = tl.load(vs_ptr + n * HKV + kvh, mask=nm, other=0.0)
            acc = acc * alpha[:, None] + tl.dot((p * vsc[None, :]).to(tl.bfloat16), v.to(tl.bfloat16))
            m_i = m_new
        base = (kvh * n_splits + sp) * ROWS + r
        tl.store(po_ptr + base[:, None] * D + d[None, :], acc)
        tl.store(pm_ptr + base, m_i)
        tl.store(pl_ptr + base, l_i)


@triton.jit
def _attn_verify_reduce(po_ptr, pm_ptr, pl_ptr, len_ptr, out_ptr, n_splits, T: tl.constexpr, HQ: tl.constexpr,
                        G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, SPLIT: tl.constexpr):
    t = tl.program_id(0)
    head = tl.program_id(1)
    kvh = head // G
    h = head % G
    ROWS: tl.constexpr = T * GP
    r = t * GP + h
    L = tl.load(len_ptr)
    used = (L + SPLIT - 1) // SPLIT
    d = tl.arange(0, D)
    m = float("-inf")
    l = 0.0
    acc = tl.zeros([D], tl.float32)
    for sp in range(0, used):
        base = (kvh * n_splits + sp) * ROWS + r
        mi = tl.load(pm_ptr + base)
        li = tl.load(pl_ptr + base)
        oi = tl.load(po_ptr + base * D + d)
        m_new = tl.maximum(m, mi)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        a = tl.exp(m - m_safe)
        b = tl.exp(mi - m_safe)
        acc = acc * a + oi * b
        l = l * a + li * b
        m = m_new
    tl.store(out_ptr + (t * HQ + head) * D + d, (acc / l).to(out_ptr.dtype.element_ty))


class VerifyAttention:
    """Workspace for the verify block's split-K attention (fixed grid, captured in the verify graph)."""

    SPLIT = 1024

    def __init__(self, capacity, tokens, hq, hkv, d, device):
        self.t, self.hq, self.hkv, self.d = tokens, hq, hkv, d
        self.g = hq // hkv
        self.gp = triton.next_power_of_2(self.g)
        self.n_splits = triton.cdiv(capacity, self.SPLIT)
        rows = hkv * self.n_splits * tokens * self.gp
        self.po = torch.empty(rows, d, dtype=torch.float32, device=device)
        self.pm = torch.empty(rows, dtype=torch.float32, device=device)
        self.pl = torch.empty(rows, dtype=torch.float32, device=device)

    def __call__(self, q, kc, vc, ks, vs, length, out):
        """q [T, hq, d] bf16 for positions length - T .. length - 1 -> out [T, hq, d]."""
        _attn_verify_kernel[(self.hkv, self.n_splits)](q, kc, vc, ks, vs, length, self.po, self.pm, self.pl,
                                                       self.d ** -0.5, self.n_splits, T=self.t, HQ=self.hq,
                                                       HKV=self.hkv, G=self.g, GP=self.gp, D=self.d, BLOCK_N=64,
                                                       SPLIT=self.SPLIT, num_warps=4, num_stages=2)
        _attn_verify_reduce[(self.t, self.hq)](self.po, self.pm, self.pl, length, out, self.n_splits, T=self.t,
                                               HQ=self.hq, G=self.g, GP=self.gp, D=self.d, SPLIT=self.SPLIT,
                                               num_warps=4)
        return out


def skinny16(x, w, s, pre, post, out):
    """skinny() for at most 16 rows (decode verify and drafter): one 16-row tile."""
    return skinny(x, w, s, pre, post, out, block_m=16)


def quantize_nvfp4(weight):
    """BF16/FP32 [N, K] -> (codes [N, K/2] uint8 low nibble first, E4M3 group-16 scales, alpha, pre, post) with the
    rules of Inference's quantized-projections NVFP4 codec (round to nearest even, global scale 6 * 448 / amax),
    vectorized on the device in blocks of 256 rows. For draft-only weights (the drafter), which crit lets an engine
    re-encode."""
    N, Kd = weight.shape
    amax = float(weight.float().abs().max())
    g = 6.0 * 448.0 / amax if amax > 0 else 1.0
    parts = [_quantize_rows(weight[r:r + 256].float(), g) for r in range(0, N, 256)]
    packed = torch.cat([p[0] for p in parts]).contiguous()
    scale = torch.cat([p[1] for p in parts]).contiguous()
    e = round(math.log2(g))
    dev = weight.device
    return (packed, scale, torch.full((N,), E2M1_UNIT / g, device=dev), torch.full((N,), 2.0 ** -e, device=dev),
            torch.full((N,), 2.0 ** e / g, device=dev), torch.full((N,), 1.0 / g, device=dev))


def _quantize_rows(w, g):
    N, Kd = w.shape
    blocks = w.view(N, Kd // 16, 16) * g
    bmax = blocks.abs().amax(-1)
    scale = (bmax / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn)
    sf = scale.float()
    x = torch.where(sf[..., None] > 0, blocks / sf[..., None].clamp_min(1e-30), torch.zeros_like(blocks))
    mag = x.abs().clamp(max=6.0)
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=w.device)
    dist = (mag[..., None] - table).abs()
    best = dist.min(-1, keepdim=True).values
    # nearest entry; on a tie the even index (round half to even)
    cand = (dist == best)
    idx = torch.where(cand[..., ::2].any(-1), (cand[..., ::2].float().argmax(-1) * 2), cand.float().argmax(-1))
    codes = (idx | ((x < 0) & (idx > 0)).to(idx.dtype) * 8).to(torch.uint8).view(N, Kd)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, scale.view(N, Kd // 16)


@triton.jit
def _embed_gather_kernel(table_ptr, ids_ptr, out_ptr, H, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tok = tl.load(ids_ptr + r)
    v = tl.load(table_ptr + tok * H + c, mask=c < H)
    tl.store(out_ptr + r * H + c, v, mask=c < H)


class HostTable:
    """A pinned host tensor handed to a kernel by its address: with unified addressing the device reads the rows it
    gathers straight from host memory (no copy of the table, no host round trip, capturable in a graph)."""

    def __init__(self, tensor):
        assert tensor.is_pinned()
        self.tensor, self.dtype = tensor, tensor.dtype

    def data_ptr(self):
        return self.tensor.data_ptr()


def embed_gather(table, ids, out):
    """Rows `ids` [n] (device int64) of a HostTable [V, H] into out [n, H] on the device."""
    n, H = out.shape
    _embed_gather_kernel[(n, triton.cdiv(H, 1024))](table, ids, out, H, BLOCK=1024, num_warps=4)
    return out


# The native NVFP4 GEMV (nvfp4_gemv.cu, built beside this module by the candidate's build recipe).
_GEMV_LIB = None


def _gemv_lib():
    global _GEMV_LIB
    if _GEMV_LIB is None:
        import ctypes
        from pathlib import Path
        lib = ctypes.CDLL(str(Path(__file__).with_name("libnvfp4_gemv.so")))
        lib.nvfp4_gemv_rows.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 + [ctypes.c_void_p, ctypes.c_int]
        lib.nvfp4_gemv_rows.restype = ctypes.c_int
        _GEMV_LIB = lib
    return _GEMV_LIB


def cuda_gemv(x, part, out):
    """x [M, K] bf16 (M <= 8) times the NVFP4 matrix `part` (codes, scales, ..., 1/g per row last) -> out [M, N] fp32,
    on the current stream (capturable). One row per warp for one column, eight for 2-4, four for 5-8: the same bits
    per column whatever M (column invariant)."""
    w, s, ginv = part[0], part[1], part[-1]
    M, Kd = x.shape
    rows = 1 if M == 1 else (8 if M <= 4 else 4)
    rc = _gemv_lib().nvfp4_gemv_rows(x.data_ptr(), w.data_ptr(), s.data_ptr(), ginv.data_ptr(), out.data_ptr(), M,
                                     w.shape[0], Kd, x.stride(0), out.stride(0),
                                     torch.cuda.current_stream().cuda_stream, rows)
    if rc != 0:
        raise RuntimeError(f"nvfp4_gemv failed ({rc})")
    return out


@triton.jit
def _dequant_fp8_kernel(w_ptr, s_ptr, alpha_ptr, inv_ptr, out_ptr, K, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    k0 = tl.program_id(1) * BLOCK_K
    HALF: tl.constexpr = BLOCK_K // 2
    j = tl.arange(0, HALF)
    packed = tl.load(w_ptr + row * (K // 2) + k0 // 2 + j)
    scale = tl.load(s_ptr + row * (K // 16) + (k0 + 2 * j) // 16).to(tl.float32)
    f = tl.load(alpha_ptr + row) * tl.load(inv_ptr + row)
    lo = tl.clamp(_e2m1(packed) * scale * f, -448.0, 448.0)
    hi = tl.clamp(_e2m1(packed >> 4) * scale * f, -448.0, 448.0)
    tl.store(out_ptr + row * K + k0 + tl.arange(0, BLOCK_K), tl.interleave(lo, hi).to(tl.float8e4nv))


def dequant_fp8(w, s, alpha, inv_rowscale, out):
    """NVFP4 rows to FP8 E4M3 [N, K] in units of each row's scale (row maximum at 448): w / rowscale, rounded once."""
    N = w.shape[0]
    K = w.shape[1] * 2
    _dequant_fp8_kernel[(N, K // 512)](w, s, alpha, inv_rowscale, out, K, BLOCK_K=512, num_warps=4)
    return out


@triton.jit
def _quantize_rows_fp8_kernel(x_ptr, out_ptr, scale_ptr, K, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    amax = 0.0
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + row * K + offs, mask=offs < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(v), axis=0))
    scale = tl.maximum(amax / 448.0, 1e-12)
    tl.store(scale_ptr + row, scale)
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + row * K + offs, mask=offs < K, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * K + offs, (v / scale).to(tl.float8e4nv), mask=offs < K)


def quantize_rows_fp8(x):
    """x [T, K] (contiguous 16-bit) -> (FP8 E4M3 codes, FP32 scale per row [T, 1]) with the row maximum at 448, in one
    kernel per row (no FP32 copy of the activations)."""
    T, K = x.shape
    out = torch.empty(T, K, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty(T, 1, dtype=torch.float32, device=x.device)
    _quantize_rows_fp8_kernel[(T,)](x, out, scale, K, BLOCK=2048, num_warps=8)
    return out, scale
