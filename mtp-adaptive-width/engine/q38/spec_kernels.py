"""Triton kernels of the speculative round: the gated delta rule with deferred state commit, the conv verify and commit,
the accept rule, row gathers and the MTP drafter's attention over an FP8 cache scaled per position and head."""
import triton
import triton.language as tl

from .kernels import ATTN_BLOCK, ATTN_SPLITS, _attn_combine_kernel


@triton.jit
def _gdn_recurrent_kernel(q_ptr, k_ptr, v_ptr, a_ptr, b_ptr, A_ptr, dt_ptr, h_ptr, o_ptr, count_ptr, T, scale,
                          H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
                          STORE: tl.constexpr, OUTPUT: tl.constexpr, COUNTED: tl.constexpr):
    # Gated delta rule over T tokens from the state h [HV, K, V] (fp32), the same arithmetic per token however many
    # tokens one launch carries, so verify columns equal one-token steps bit for bit. With COUNTED only the first
    # count tokens advance the state (the commit of a speculative round); with STORE the state is written back.
    pid = tl.program_id(0)
    NV = V // BV
    i_v = pid % NV
    i_hv = pid // NV
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    p_h = h_ptr + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_h)
    A = tl.load(A_ptr + i_hv).to(tl.float32)
    dt = tl.load(dt_ptr + i_hv).to(tl.float32)
    n = T
    if COUNTED:
        n = tl.load(count_ptr)
    for t in range(T):
        b_q = tl.load(q_ptr + (t * H + i_h) * K + o_k).to(tl.float32)
        b_k = tl.load(k_ptr + (t * H + i_h) * K + o_k).to(tl.float32)
        b_v = tl.load(v_ptr + (t * HV + i_hv) * V + o_v).to(tl.float32)
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        beta = tl.sigmoid(tl.load(b_ptr + t * HV + i_hv).to(tl.float32))
        x = tl.load(a_ptr + t * HV + i_hv).to(tl.float32) + dt
        sp = tl.where(x > 20., x, tl.log(1. + tl.exp(x)))
        g = -tl.exp(A) * sp
        nh = b_h * tl.exp(g)
        d = beta * (b_v - tl.sum(nh * b_k[:, None], 0))
        nh = nh + b_k[:, None] * d[None, :]
        if OUTPUT:
            tl.store(o_ptr + (t * HV + i_hv) * V + o_v, tl.sum(nh * b_q[:, None], 0))
        b_h = tl.where(t < n, nh, b_h)
    if STORE:
        tl.store(p_h, b_h)


def gdn_recurrent(q, k, v, a, b, A_log, dt_bias, state, o, scale, store=True, count=None, output=True):
    """q, k [T, H, K], v [T, HV, V], a, b [T, HV] (raw: gate and beta computed here), state [HV, K, V] fp32,
    o [T, HV, V] fp32. store False leaves the state as it was (a verify pass); count (device int32 [1]) advances the
    state by only that many tokens (the commit of a round)."""
    T, H, Kd = k.shape
    HV, V = v.shape[1], v.shape[2]
    BV = 8
    _gdn_recurrent_kernel[(V // BV * HV,)](q, k, v, a, b, A_log, dt_bias, state, o, count if count is not None else a,
                                           T, scale, H=H, HV=HV, K=Kd, V=V, BV=BV, STORE=store, OUTPUT=output,
                                           COUNTED=count is not None, num_warps=1)
    return o


@triton.jit
def _conv_verify_kernel(x_ptr, w_ptr, state_ptr, out_ptr, T, C, x_stride, S0, S1, BC: tl.constexpr):
    # the conv of T tokens from the stored state, leaving the state as it was (a verify pass)
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


@triton.jit
def _conv_commit_kernel(x_ptr, state_ptr, count_ptr, T, C, x_stride, BC: tl.constexpr):
    c = tl.program_id(0) * BC + tl.arange(0, BC)
    m = c < C
    n = tl.load(count_ptr)
    s0 = tl.load(state_ptr + c * 3 + 0, mask=m, other=0.)
    s1 = tl.load(state_ptr + c * 3 + 1, mask=m, other=0.)
    s2 = tl.load(state_ptr + c * 3 + 2, mask=m, other=0.)
    for t in range(T):
        x = tl.load(x_ptr + t * x_stride + c, mask=m, other=0.).to(tl.float32)
        keep = t < n
        s0n = tl.where(keep, s1, s0)
        s1n = tl.where(keep, s2, s1)
        s2 = tl.where(keep, x, s2)
        s0 = s0n
        s1 = s1n
    tl.store(state_ptr + c * 3 + 0, s0, mask=m)
    tl.store(state_ptr + c * 3 + 1, s1, mask=m)
    tl.store(state_ptr + c * 3 + 2, s2, mask=m)


def conv_verify(x, weight, state, out, sections):
    T, C = x.shape[0], weight.shape[0]
    _conv_verify_kernel[(triton.cdiv(C, 512),)](x, weight, state, out, T, C, x.stride(0), sections[0], sections[1],
                                                BC=512, num_warps=4)
    return out


def conv_commit(x, state, count):
    """The conv state after the first count (device int32 [1]) of the T tokens x [T, >=C] (raw conv inputs)."""
    T, C = x.shape[0], state.shape[0]
    _conv_commit_kernel[(triton.cdiv(C, 1024),)](x, state, count, T, C, x.stride(0), BC=1024, num_warps=4)


@triton.jit
def _accept_kernel(drafts_ptr, pred_ptr, emitted_ptr, count_ptr, tok_ptr, tpos_ptr, mpos_ptr, K: tl.constexpr,
                   KP: tl.constexpr):
    # drafts stand while they equal the predictions; the first differing prediction (or the bonus) follows them
    i = tl.arange(0, KP)
    d = tl.load(drafts_ptr + i, mask=i < K, other=-2)
    p = tl.load(pred_ptr + i, mask=i < K + 1, other=-3)
    miss = tl.where((i < K) & (d == p), KP, i)
    n = tl.minimum(tl.min(miss, 0), K)
    tl.store(emitted_ptr + i, tl.where(i <= n, p, -1), mask=i < K + 1)
    tl.store(count_ptr, n + 1)
    tl.store(tok_ptr, tl.sum(tl.where(i == n, p, 0), 0))
    t = tl.load(tpos_ptr)
    tl.store(mpos_ptr, t)
    tl.store(tpos_ptr, t + n + 1)


def accept(drafts, predictions, emitted, count, target_tokens, tpos, mpos, K):
    """emitted [K + 1] = the standing drafts then the next prediction (-1 after), count = their number; the target's
    next first token, mpos = tpos and tpos += count."""
    _accept_kernel[(1,)](drafts, predictions, emitted, count, target_tokens, tpos, mpos, K=K,
                         KP=triton.next_power_of_2(K + 1), num_warps=1)


@triton.jit
def _gather_row_kernel(src_ptr, idx_ptr, dst_ptr, D, src_stride, offset, BD: tl.constexpr):
    r = tl.load(idx_ptr) + offset
    o = tl.program_id(0) * BD + tl.arange(0, BD)
    m = o < D
    tl.store(dst_ptr + o, tl.load(src_ptr + r * src_stride + o, mask=m), mask=m)


def gather_row(src, idx, dst, offset=-1):
    """dst [D] = src[idx + offset] (idx a device int32 [1])."""
    D = src.shape[1]
    _gather_row_kernel[(triton.cdiv(D, 1024),)](src, idx, dst, D, src.stride(0), offset, BD=1024, num_warps=4)


@triton.jit
def _add_scalar_kernel(src_ptr, dst_ptr, offset):
    tl.store(dst_ptr, tl.load(src_ptr) + offset)


def add_scalar(src, dst, offset):
    """dst [1] = src [1] + offset (device int32)."""
    _add_scalar_kernel[(1,)](src, dst, offset, num_warps=1)


@triton.jit
def _mtp_prep_kernel(qkv_ptr, qn_ptr, kn_ptr, inv_ptr, pos_ptr, q_ptr, gate_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr,
                     qkv_stride, cache_stride_h, S, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                     R: tl.constexpr, eps):
    # as the target's attention prep, with FP8 keys and values scaled per (position, head) by their absolute maximum
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
        r = tl.rsqrt(tl.sum(x * x, 0) / D + eps)
        x = x * r * tl.load(qn_ptr + o)
        xp = xp * r * tl.load(qn_ptr + partner)
        y = tl.where(rot, x * c + sign * xp * s, x)
        tl.store(q_ptr + (t * HQ + h) * D + o, y.to(q_ptr.dtype.element_ty))
        tl.store(gate_ptr + (t * HQ + h) * D + o, gate)
    else:
        hk = h - HQ
        x = tl.load(base + HQ * 2 * D + hk * D + o).to(tl.float32)
        xp = tl.load(base + HQ * 2 * D + hk * D + partner).to(tl.float32)
        r = tl.rsqrt(tl.sum(x * x, 0) / D + eps)
        x = x * r * tl.load(kn_ptr + o)
        xp = xp * r * tl.load(kn_ptr + partner)
        k = tl.where(rot, x * c + sign * xp * s, x)
        v = tl.load(base + HQ * 2 * D + HK * D + hk * D + o).to(tl.float32)
        ksc = tl.maximum(tl.max(tl.abs(k), 0), 1e-12) / 448.
        vsc = tl.maximum(tl.max(tl.abs(v), 0), 1e-12) / 448.
        tl.store(kc_ptr + hk * cache_stride_h + pos.to(tl.int64) * D + o, (k / ksc).to(tl.float8e4nv))
        tl.store(vc_ptr + hk * cache_stride_h + pos.to(tl.int64) * D + o, (v / vsc).to(tl.float8e4nv))
        tl.store(ks_ptr + hk * S + pos, ksc)
        tl.store(vs_ptr + hk * S + pos, vsc)


def mtp_prep(qkv, qn1p, kn1p, inv_freq, pos, q, gate, kcache, vcache, kscale, vscale, HQ=24, HK=4, D=256, R=64):
    T = qkv.shape[0]
    _mtp_prep_kernel[(T, HQ + HK)](qkv, qn1p, kn1p, inv_freq, pos, q, gate, kcache, vcache, kscale, vscale,
                                   qkv.stride(0), kcache.stride(0), kcache.shape[1], HQ=HQ, HK=HK, D=D, R=R, eps=1e-6,
                                   num_warps=4)


@triton.jit
def _mtp_attn_kernel(q_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pos_ptr, acc_ptr, ml_ptr, scale, cache_stride_h, S,
                     HQ: tl.constexpr, HK: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                     NS: tl.constexpr, BS: tl.constexpr):
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
        ks = tl.load(ks_ptr + hk * S + s_idx, mask=smask, other=0.)
        sc = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * (scale * ks[None, :])
        sc = tl.where(smask[None, :], sc, -float("inf"))
        mn = tl.maximum(m, tl.max(sc, 1))
        p = tl.exp(sc - mn[:, None])
        alpha = tl.exp(m - mn)
        l = l * alpha + tl.sum(p, 1)
        v = tl.load(vbase + s_idx[:, None].to(tl.int64) * D + o[None, :], mask=smask[:, None], other=0.)
        vs = tl.load(vs_ptr + hk * S + s_idx, mask=smask, other=0.)
        acc = acc * alpha[:, None] + tl.dot((p * vs[None, :]).to(tl.bfloat16), v.to(tl.bfloat16))
        m = mn
    row = (t * HK + hk) * NS + sp
    tl.store(acc_ptr + (row * 16 + g[:, None]) * D + o[None, :], acc)
    tl.store(ml_ptr + (row * 16 + g) * 2, m)
    tl.store(ml_ptr + (row * 16 + g) * 2 + 1, l)


def mtp_attn(q, kcache, vcache, kscale, vscale, pos, gate, out, scale, work, HQ=24, HK=4, D=256):
    """The drafter's attention: q [T, HQ, D] against its per-position-scaled FP8 cache, out [T, HQ * D] with the
    sigmoid output gate."""
    T = q.shape[0]
    acc, ml = work
    G = HQ // HK
    _mtp_attn_kernel[(T, HK, ATTN_SPLITS)](q, kcache, vcache, kscale, vscale, pos, acc, ml, scale, kcache.stride(0),
                                           kcache.shape[1], HQ=HQ, HK=HK, G=G, D=D, NS=ATTN_SPLITS, BS=ATTN_BLOCK,
                                           num_warps=4, num_stages=2)
    _attn_combine_kernel[(T, HQ)](acc, ml, gate, out, 1.0, HQ=HQ, HK=HK, G=G, D=D, NS=ATTN_SPLITS, num_warps=4)
