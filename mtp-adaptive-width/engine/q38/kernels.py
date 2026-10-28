"""Triton kernels of the Qwen3.8-27B engine's decode path (and the small fused ops of its prefill).

Every kernel that feeds a token's logits computes each output element in an order that depends only on the
problem's (N, K) and the element's own row, never on how many rows (tokens) one launch carries, so a verify pass of
several tokens gives each token the bits a one-token step gives it.
"""
import torch
import triton
import triton.language as tl

# projections over the stored encodings live in matmul.py
from .matmul import MAX_ROWS, NVFP4_TRITON_ROWS, Linear, fp8_mm, gemv, stored_mm, widen_bf16  # noqa: F401


# ------------------------------------------------------------------------------------------------- norms
@triton.jit
def _add_rmsnorm_kernel(h_ptr, d_ptr, w_ptr, y_ptr, D, eps, h_stride, d_stride, y_stride,
                        ADD: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    o = tl.arange(0, BD)
    m = o < D
    h = tl.load(h_ptr + t * h_stride + o, mask=m, other=0.).to(tl.float32)
    if ADD:
        h = h + tl.load(d_ptr + t * d_stride + o, mask=m, other=0.).to(tl.float32)
        tl.store(h_ptr + t * h_stride + o, h, mask=m)
    var = tl.sum(h * h, 0) / D
    y = h * tl.rsqrt(var + eps) * tl.load(w_ptr + o, mask=m, other=0.)
    tl.store(y_ptr + t * y_stride + o, y.to(y_ptr.dtype.element_ty), mask=m)


def rmsnorm(h, w1p, y, eps=1e-6, delta=None):
    """Zero-centred RMSNorm (weights given as 1 + w, fp32) of the fp32 residual h [T, D] into y; with delta, h += delta
    first (in place)."""
    T, D = h.shape
    _add_rmsnorm_kernel[(T,)](h, delta if delta is not None else h, w1p, y, D, eps, h.stride(0),
                              delta.stride(0) if delta is not None else 0, y.stride(0), ADD=delta is not None,
                              BD=triton.next_power_of_2(D), num_warps=8)
    return y


@triton.jit
def _silu_mul_kernel(gu_ptr, s_ptr, y_ptr, I, gu_stride, y_stride, BI: tl.constexpr, SCALE: tl.constexpr):
    t = tl.program_id(1)
    o = tl.program_id(0) * BI + tl.arange(0, BI)
    m = o < I
    g = tl.load(gu_ptr + t * gu_stride + o, mask=m, other=0.).to(tl.float32)
    u = tl.load(gu_ptr + t * gu_stride + I + o, mask=m, other=0.).to(tl.float32)
    if SCALE:
        g = g * tl.load(s_ptr + o, mask=m, other=0.)
        u = u * tl.load(s_ptr + I + o, mask=m, other=0.)
    tl.store(y_ptr + t * y_stride + o, (g / (1. + tl.exp(-g)) * u).to(y_ptr.dtype.element_ty), mask=m)


def silu_mul(gu, y, scale=None):
    """gu [T, 2I] (gate | up) -> y [T, I] = silu(gate) * up; with scale [2I], gu's rows are scaled first."""
    T = gu.shape[0]
    I = gu.shape[1] // 2
    _silu_mul_kernel[(triton.cdiv(I, 1024), T)](gu, scale if scale is not None else gu, y, I, gu.stride(0),
                                                y.stride(0), BI=1024, SCALE=scale is not None, num_warps=4)
    return y


# ------------------------------------------------------------------------------------------------- attention prep
@triton.jit
def _attn_prep_kernel(qkv_ptr, qn_ptr, kn_ptr, inv_ptr, pos_ptr, q_ptr, gate_ptr, kc_ptr, vc_ptr,
                      qkv_stride, k_inv_scale, v_inv_scale, cache_stride_h,
                      HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, R: tl.constexpr, eps):
    # one program per (token, head); heads 0..HQ-1 are query heads, HQ..HQ+HK-1 key/value heads
    t = tl.program_id(0)
    h = tl.program_id(1)
    pos = tl.load(pos_ptr) + t
    o = tl.arange(0, D)
    half = R // 2
    rot = o < R
    partner = tl.where(o < half, o + half, o - half)
    sign = tl.where(o < half, -1.0, 1.0)
    inv = tl.load(inv_ptr + (o % half), mask=rot, other=0.)
    ang = pos.to(tl.float32) * inv
    c = tl.cos(ang)
    s = tl.sin(ang)
    base = qkv_ptr + t * qkv_stride
    if h < HQ:
        x = tl.load(base + h * 2 * D + o).to(tl.float32)
        xp = tl.load(base + h * 2 * D + partner).to(tl.float32)
        gate = tl.load(base + h * 2 * D + D + o).to(tl.float32)
        w = tl.load(qn_ptr + o)
        wp = tl.load(qn_ptr + partner)
        r = tl.rsqrt(tl.sum(x * x, 0) / D + eps)
        x = x * r * w
        xp = xp * r * wp
        y = tl.where(rot, x * c + sign * xp * s, x)
        tl.store(q_ptr + (t * HQ + h) * D + o, y.to(q_ptr.dtype.element_ty))
        tl.store(gate_ptr + (t * HQ + h) * D + o, gate)
    else:
        hk = h - HQ
        x = tl.load(base + HQ * 2 * D + hk * D + o).to(tl.float32)
        xp = tl.load(base + HQ * 2 * D + hk * D + partner).to(tl.float32)
        w = tl.load(kn_ptr + o)
        wp = tl.load(kn_ptr + partner)
        r = tl.rsqrt(tl.sum(x * x, 0) / D + eps)
        x = x * r * w
        xp = xp * r * wp
        k = tl.where(rot, x * c + sign * xp * s, x)
        v = tl.load(base + HQ * 2 * D + HK * D + hk * D + o).to(tl.float32)
        kq = tl.clamp(k * k_inv_scale, -448., 448.).to(tl.float8e4nv)
        vq = tl.clamp(v * v_inv_scale, -448., 448.).to(tl.float8e4nv)
        tl.store(kc_ptr + hk * cache_stride_h + pos.to(tl.int64) * D + o, kq)
        tl.store(vc_ptr + hk * cache_stride_h + pos.to(tl.int64) * D + o, vq)


def attn_prep(qkv, qn1p, kn1p, inv_freq, pos, q, gate, kcache, vcache, k_scale, v_scale, HQ=24, HK=4, D=256, R=64):
    """qkv [T, HQ*2D + 2*HK*D] fp32 -> q [T, HQ, D] bf16 (normed, rotated), gate [T, HQ, D] fp32, and the tokens'
    keys and values written to the FP8 caches [HK, S, D] at positions pos..pos+T-1 (pos: device int32 [1])."""
    T = qkv.shape[0]
    _attn_prep_kernel[(T, HQ + HK)](qkv, qn1p, kn1p, inv_freq, pos, q, gate, kcache, vcache, qkv.stride(0),
                                    1.0 / k_scale, 1.0 / v_scale, kcache.stride(0), HQ=HQ, HK=HK, D=D, R=R, eps=1e-6,
                                    num_warps=4)


# ------------------------------------------------------------------------------------------------- decode attention
@triton.jit
def _attn_decode_kernel(q_ptr, kc_ptr, vc_ptr, pos_ptr, acc_ptr, ml_ptr, scale, cache_stride_h,
                        HQ: tl.constexpr, HK: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                        NS: tl.constexpr, BS: tl.constexpr):
    # program (t, kv head, split): queries t's G heads against keys [split range) of length pos + t + 1
    t = tl.program_id(0)
    hk = tl.program_id(1)
    sp = tl.program_id(2)
    L = tl.load(pos_ptr) + t + 1
    blocks = tl.cdiv(L, BS)
    per = tl.cdiv(blocks, NS)
    b0 = sp * per
    b1 = tl.minimum(b0 + per, blocks)
    g = tl.arange(0, 16)
    o = tl.arange(0, D)
    gmask = g < G
    q = tl.load(q_ptr + (t * HQ + hk * G + g[:, None]) * D + o[None, :], mask=gmask[:, None], other=0.)
    m = tl.full([16], -float("inf"), tl.float32)
    l = tl.zeros([16], tl.float32)
    acc = tl.zeros([16, D], tl.float32)
    kbase = kc_ptr + hk * cache_stride_h
    vbase = vc_ptr + hk * cache_stride_h
    for b in range(b0, b1):
        s_idx = b * BS + tl.arange(0, BS)
        smask = s_idx < L
        k = tl.load(kbase + s_idx[:, None].to(tl.int64) * D + o[None, :], mask=smask[:, None], other=0.)
        sc = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * scale
        sc = tl.where(smask[None, :], sc, -float("inf"))
        mn = tl.maximum(m, tl.max(sc, 1))
        p = tl.exp(sc - mn[:, None])
        alpha = tl.exp(m - mn)
        l = l * alpha + tl.sum(p, 1)
        v = tl.load(vbase + s_idx[:, None].to(tl.int64) * D + o[None, :], mask=smask[:, None], other=0.)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
        m = mn
    row = (t * HK + hk) * NS + sp
    tl.store(acc_ptr + (row * 16 + g[:, None]) * D + o[None, :], acc)
    tl.store(ml_ptr + (row * 16 + g) * 2, m)
    tl.store(ml_ptr + (row * 16 + g) * 2 + 1, l)


@triton.jit
def _attn_combine_kernel(acc_ptr, ml_ptr, gate_ptr, out_ptr, v_scale,
                         HQ: tl.constexpr, HK: tl.constexpr, G: tl.constexpr, D: tl.constexpr, NS: tl.constexpr):
    t = tl.program_id(0)
    hq = tl.program_id(1)
    hk = hq // G
    gi = hq % G
    o = tl.arange(0, D)
    s = tl.arange(0, NS)
    rows = ((t * HK + hk) * NS + s) * 16 + gi
    m = tl.load(ml_ptr + rows * 2)
    l = tl.load(ml_ptr + rows * 2 + 1)
    mx = tl.max(m, 0)
    w = tl.where(m == -float("inf"), 0., tl.exp(m - mx))
    denom = tl.sum(l * w, 0)
    acc = tl.load(acc_ptr + rows[:, None] * D + o[None, :])
    y = tl.sum(acc * w[:, None], 0) / denom * v_scale
    gate = tl.load(gate_ptr + (t * HQ + hq) * D + o)
    y = y * (1. / (1. + tl.exp(-gate)))
    tl.store(out_ptr + (t * HQ + hq) * D + o, y.to(out_ptr.dtype.element_ty))


ATTN_SPLITS = 32
ATTN_BLOCK = 64


def attn_decode(q, kcache, vcache, pos, gate, out, scale, v_scale, work, HQ=24, HK=4, D=256):
    """q [T, HQ, D] bf16 against the FP8 caches (key scale folded into `scale`), each token t seeing positions
    [0, pos + t]; out [T, HQ * D] = softmax(...) V * v_scale * sigmoid(gate). work: (acc, ml) scratch."""
    T = q.shape[0]
    acc, ml = work
    G = HQ // HK
    _attn_decode_kernel[(T, HK, ATTN_SPLITS)](q, kcache, vcache, pos, acc, ml, scale, kcache.stride(0), HQ=HQ, HK=HK,
                                              G=G, D=D, NS=ATTN_SPLITS, BS=ATTN_BLOCK, num_warps=4, num_stages=2)
    _attn_combine_kernel[(T, HQ)](acc, ml, gate, out, v_scale, HQ=HQ, HK=HK, G=G, D=D, NS=ATTN_SPLITS, num_warps=4)


def attn_work(T, HK=4, D=256, device="cuda"):
    rows = T * HK * ATTN_SPLITS * 16
    return (torch.empty(rows * D, device=device, dtype=torch.float32),
            torch.empty(rows * 2, device=device, dtype=torch.float32))


# ------------------------------------------------------------------------------------------------- gated delta net
@triton.jit
def _conv_update_kernel(x_ptr, w_ptr, state_ptr, out_ptr, T, C, x_stride, S0, S1, BC: tl.constexpr):
    # causal depthwise conv of width 4 over the stored last 3 inputs, then SiLU; T tokens in order. The output is
    # three contiguous sections: channels [0, S0) as [T, S0], [S0, S0 + S1) as [T, S1], the rest as [T, C - S0 - S1].
    c = tl.program_id(0) * BC + tl.arange(0, BC)
    m = c < C
    S2 = C - S0 - S1
    sec_base = tl.where(c < S0, 0, tl.where(c < S0 + S1, T * S0, T * (S0 + S1)))
    sec_width = tl.where(c < S0, S0, tl.where(c < S0 + S1, S1, S2))
    local = tl.where(c < S0, c, tl.where(c < S0 + S1, c - S0, c - S0 - S1))
    w0 = tl.load(w_ptr + c * 4 + 0, mask=m, other=0.).to(tl.float32)
    w1 = tl.load(w_ptr + c * 4 + 1, mask=m, other=0.).to(tl.float32)
    w2 = tl.load(w_ptr + c * 4 + 2, mask=m, other=0.).to(tl.float32)
    w3 = tl.load(w_ptr + c * 4 + 3, mask=m, other=0.).to(tl.float32)
    s0 = tl.load(state_ptr + c * 3 + 0, mask=m, other=0.)
    s1 = tl.load(state_ptr + c * 3 + 1, mask=m, other=0.)
    s2 = tl.load(state_ptr + c * 3 + 2, mask=m, other=0.)
    for t in range(T):
        x = tl.load(x_ptr + t * x_stride + c, mask=m, other=0.).to(tl.float32)
        y = s0 * w0 + s1 * w1 + s2 * w2 + x * w3
        y = y / (1. + tl.exp(-y))
        tl.store(out_ptr + sec_base + t * sec_width + local, y, mask=m)
        s0 = s1
        s1 = s2
        s2 = x
    tl.store(state_ptr + c * 3 + 0, s0, mask=m)
    tl.store(state_ptr + c * 3 + 1, s1, mask=m)
    tl.store(state_ptr + c * 3 + 2, s2, mask=m)


def conv_update(x, weight, state, out, sections):
    """x [T, >=C] (fp32 rows), weight [C, 4], state [C, 3] fp32 (oldest first) -> out (flat fp32, T * C) holding the
    sections (S0, S1, S2) of the channels each as a contiguous [T, S] block."""
    T, C = x.shape[0], weight.shape[0]
    _conv_update_kernel[(triton.cdiv(C, 512),)](x, weight, state, out, T, C, x.stride(0), sections[0], sections[1],
                                                BC=512, num_warps=4)
    return out


@triton.jit
def _gated_norm_kernel(o_ptr, z_ptr, w_ptr, y_ptr, z_stride, eps, HV: tl.constexpr, V: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.arange(0, V)
    x = tl.load(o_ptr + (t * HV + h) * V + i).to(tl.float32)
    z = tl.load(z_ptr + t * z_stride + h * V + i).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 0) / V + eps)
    y = x * r * tl.load(w_ptr + i).to(tl.float32) * (z / (1. + tl.exp(-z)))
    tl.store(y_ptr + (t * HV + h) * V + i, y.to(y_ptr.dtype.element_ty))


def gated_norm(o, z, w, y, HV=48, V=128):
    """o [T, HV, V], z [T, HV*V] (a column slice allowed) -> y [T, HV*V] = rmsnorm(o) * w * silu(z)."""
    T = o.shape[0]
    _gated_norm_kernel[(T, HV)](o, z, w, y, z.stride(0), 1e-6, HV=HV, V=V, num_warps=1)
    return y


# ------------------------------------------------------------------------------------------------- embeddings
@triton.jit
def _embed_kernel(table_ptr, tok_ptr, out_ptr, D, BD: tl.constexpr):
    t = tl.program_id(0)
    tok = tl.load(tok_ptr + t).to(tl.int64)
    o = tl.program_id(1) * BD + tl.arange(0, BD)
    m = o < D
    tl.store(out_ptr + t * D + o, tl.load(table_ptr + tok * D + o, mask=m, other=0.).to(tl.float32), mask=m)


class HostTable:
    """A pinned host tensor seen by kernels through its mapped device address (Triton reads data_ptr and dtype)."""

    def __init__(self, tensor):
        from cuda.bindings import runtime
        self.tensor = tensor
        err, ptr = runtime.cudaHostGetDevicePointer(tensor.data_ptr(), 0)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostGetDevicePointer failed: {err}")
        self.ptr = int(ptr)
        self.dtype = tensor.dtype

    def data_ptr(self):
        return self.ptr


def embed(table, tokens, out):
    """tokens [T] int32 (device) -> out [T, D] fp32 rows of the host table."""
    T = tokens.shape[0]
    D = out.shape[1]
    _embed_kernel[(T, triton.cdiv(D, 1024))](table, tokens, out, D, BD=1024, num_warps=4)
    return out


# ------------------------------------------------------------------------------------------------- prefill conv
@triton.jit
def _conv_prefill_kernel(x_ptr, w_ptr, state_ptr, out_ptr, T, C, x_stride, S0, S1, BT: tl.constexpr, BC: tl.constexpr):
    # causal depthwise conv (width 4) + SiLU of T prompt rows at once, earlier inputs from the stored state; the
    # output is three contiguous sections ([T, S0], [T, S1], [T, C - S0 - S1]) as the decode kernels write them
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    tm = t < T
    cm = c < C
    acc = tl.zeros([BT, BC], dtype=tl.float32)
    for j in tl.static_range(4):
        src = t - 3 + j
        w = tl.load(w_ptr + c * 4 + j, mask=cm, other=0.).to(tl.float32)
        xv = tl.load(x_ptr + src[:, None].to(tl.int64) * x_stride + c[None, :],
                     mask=(src[:, None] >= 0) & tm[:, None] & cm[None, :], other=0.)
        sv = tl.load(state_ptr + c[None, :] * 3 + (src[:, None] + 3), mask=(src[:, None] < 0) & tm[:, None] & cm[None, :],
                     other=0.)
        acc += (xv + sv) * w[None, :]
    y = acc / (1. + tl.exp(-acc))
    S2 = C - S0 - S1
    sec_base = tl.where(c < S0, 0, tl.where(c < S0 + S1, T * S0, T * (S0 + S1)))
    sec_width = tl.where(c < S0, S0, tl.where(c < S0 + S1, S1, S2))
    local = tl.where(c < S0, c, tl.where(c < S0 + S1, c - S0, c - S0 - S1))
    off = sec_base[None, :].to(tl.int64) + t[:, None].to(tl.int64) * sec_width[None, :] + local[None, :]
    tl.store(out_ptr + off, y, mask=tm[:, None] & cm[None, :])


def conv_prefill(x, weight, state, out, sections):
    """Prefill form of conv_update: x [T, >=C] fp32 rows, the state [C, 3] read (not written) -> out (flat, T * C)."""
    T, C = x.shape[0], weight.shape[0]
    _conv_prefill_kernel[(triton.cdiv(T, 32), triton.cdiv(C, 128))](x, weight, state, out, T, C, x.stride(0),
                                                                     sections[0], sections[1], BT=32, BC=128,
                                                                     num_warps=4)
    return out
