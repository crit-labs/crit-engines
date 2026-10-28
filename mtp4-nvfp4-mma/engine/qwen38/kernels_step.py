"""Kernels of the decode and verify step (TQ = 1 or the k + 1 columns of a speculative round): attention over the
FP8 KV cache split in fixed blocks of absolute key positions (so a column computes the same bits alone or inside a
verify), the attention's q/k norm + rotary + KV write in one launch, the gated delta's convolution + SiLU + q/k L2
norm + gates in one launch, and the rows' commit after a verify."""
import triton
import triton.language as tl

ATTN_CHUNK = 256                     # keys per split: a split covers keys [s * 256, (s + 1) * 256)


@triton.jit
def attn_step_kernel(q, kc, vc, pos, part_o, part_ml, sm_scale, k_scale, MAXS: tl.constexpr, HQ: tl.constexpr,
                     KVH: tl.constexpr, D: tl.constexpr, G: tl.constexpr, TQ: tl.constexpr, R: tl.constexpr,
                     CH: tl.constexpr, BN: tl.constexpr, NPROG: tl.constexpr):
    """Program (kvh, p) takes splits p, p + NPROG, ... up to the last key: the TQ columns' G query heads (rows
    t * G + g, padded to R) against keys [split * CH, (split + 1) * CH) clipped to each column's causal limit
    pos + t, writing each split's partial output with its running max and sum (base-2 logits). The grid does not
    depend on the context: the work grows with the keys held."""
    kvh = tl.program_id(0)
    first = tl.program_id(1)
    P = tl.load(pos)
    last = P + TQ - 1
    n_splits = last // CH + 1
    r = tl.arange(0, R)
    t = r // G
    g = r % G
    rm = r < TQ * G
    d = tl.arange(0, D)
    qt = tl.load(q + ((t[:, None] * HQ + kvh * G + g[:, None]) * D + d[None, :]), mask=rm[:, None], other=0.0)
    qk = sm_scale * k_scale * 1.4426950408889634
    limit = P + t
    for sp in range(first, n_splits, NPROG):
        lo = sp * CH
        m_i = tl.full([R], -float("inf"), tl.float32)
        l_i = tl.zeros([R], tl.float32)
        acc = tl.zeros([R, D], tl.float32)
        for start in range(lo, lo + CH, BN):
            n = start + tl.arange(0, BN)
            km = n <= last
            k = tl.load(kc + (n[:, None] * KVH + kvh) * D + d[None, :], mask=km[:, None], other=0.0).to(tl.bfloat16)
            s = tl.dot(qt, tl.trans(k)) * qk
            s = tl.where((n[None, :] <= limit[:, None]) & km[None, :], s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_safe = tl.where(m_new > -float("inf"), m_new, 0.0)
            alpha = tl.exp2(m_i - m_safe)
            p = tl.exp2(s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(vc + (n[:, None] * KVH + kvh) * D + d[None, :], mask=km[:, None], other=0.0).to(tl.bfloat16)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m_i = m_new
        base = (kvh * MAXS + sp) * R
        tl.store(part_o + (base + r[:, None]) * D + d[None, :], acc)
        tl.store(part_ml + (base + r) * 2, m_i)
        tl.store(part_ml + (base + r) * 2 + 1, l_i)


@triton.jit
def attn_step_reduce_kernel(part_o, part_ml, gate, out, pos, v_scale, MAXS: tl.constexpr, HQ: tl.constexpr,
                            D: tl.constexpr, G: tl.constexpr, R: tl.constexpr, CH: tl.constexpr, SB: tl.constexpr):
    """Program (t, h): combine column t's splits in split order (the same order alone or in a verify), times the
    output gate sigmoid(gate[t, h])."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    kvh = h // G
    row = t * G + h % G
    P = tl.load(pos)
    n_s = (P + t) // CH + 1
    d = tl.arange(0, D)
    mx = tl.full([SB], -float("inf"), tl.float32)
    for s0 in range(0, n_s, SB):
        s = s0 + tl.arange(0, SB)
        m = tl.load(part_ml + ((kvh * MAXS + s) * R + row) * 2, mask=s < n_s, other=-float("inf"))
        mx = tl.maximum(mx, m)
    big = tl.max(mx, axis=0)
    total = 0.0
    acc = tl.zeros([D], tl.float32)
    for s0 in range(0, n_s, SB):
        s = s0 + tl.arange(0, SB)
        sm = s < n_s
        base = (kvh * MAXS + s) * R + row
        m = tl.load(part_ml + base * 2, mask=sm, other=-float("inf"))
        l = tl.load(part_ml + base * 2 + 1, mask=sm, other=0.0)
        w = tl.where(m > -float("inf"), tl.exp2(m - big), 0.0)
        total += tl.sum(l * w, axis=0)
        o = tl.load(part_o + base[:, None] * D + d[None, :], mask=sm[:, None], other=0.0)
        acc += tl.sum(o * w[:, None], axis=0)
    o = acc / total * v_scale
    gt = tl.load(gate + (t * HQ + h) * D + d).to(tl.float32)
    tl.store(out + (t * HQ + h) * D + d, (o * tl.sigmoid(gt)).to(tl.bfloat16))


@triton.jit
def qk_rope_kv_kernel(qkv, row_stride, qw, kw, cos, sin, pos, q_out, gate_out, kc, vc, inv_k, inv_v, eps, kx, vx,
                      rxs, rws, HQ: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr, ROT: tl.constexpr,
                      GATED: tl.constexpr, H0: tl.constexpr = 0, EXACT: tl.constexpr = False,
                      ROWCOL: tl.constexpr = False):
    """Program (t, head) over the fused q/k/v GEMM output row t: q heads get the zero-centered RMSNorm and rotary
    (and their output gate copied out), k heads the same then FP8 into the cache at pos + t, v heads FP8 into the
    cache. GATED: q_proj emits [q | gate] per head (the target); otherwise q only (the MTP layer emits the same)."""
    t = tl.program_id(0)
    h = tl.program_id(1) + H0
    P = tl.load(pos) + t
    d = tl.arange(0, D)
    QW: tl.constexpr = 2 * D if GATED else D
    row = qkv + t * row_stride
    half: tl.constexpr = ROT // 2
    c = tl.load(cos + P * half + (d % half), mask=d < ROT, other=1.0)
    sn = tl.load(sin + P * half + (d % half), mask=d < ROT, other=0.0)
    if h < HQ:
        col = h * QW + d
        w = tl.load(qw + d)
    elif h < HQ + KVH:
        col = HQ * QW + (h - HQ) * D + d
        w = tl.load(kw + d)
    else:
        col = HQ * QW + KVH * D + (h - HQ - KVH) * D + d
        w = tl.zeros([D], tl.float32)
    x = tl.load(row + col).to(tl.float32)
    if ROWCOL:
        rs = tl.load(rxs + t)
        x = x * rs * tl.load(rws + col)
    if h < HQ + KVH:
        x = x * tl.rsqrt(tl.sum(x * x, axis=0) / D + eps) * (1.0 + w)
        # rotate-half on the first ROT dims: pair (i, i + ROT / 2)
        partner = tl.where(d < half, d + half, d - half)
        xr = tl.reshape(x, (D,))
        xp = tl.gather(xr, partner, 0)
        sign = tl.where(d < half, -1.0, 1.0)
        x = tl.where(d < ROT, x * c + sign * xp * sn, x)
    if h < HQ:
        tl.store(q_out + (t * HQ + h) * D + d, x.to(tl.bfloat16))
        if GATED:
            gv = tl.load(row + h * QW + D + d)
            if ROWCOL:
                gv = (gv.to(tl.float32) * rs * tl.load(rws + h * QW + D + d)).to(tl.bfloat16)
            tl.store(gate_out + (t * HQ + h) * D + d, gv)
    elif h < HQ + KVH:
        tl.store(kc + (P * KVH + (h - HQ)) * D + d, tl.minimum(tl.maximum(x * inv_k, -448.0), 448.0).to(tl.float8e4nv))
        if EXACT:
            tl.store(kx + (t * KVH + (h - HQ)) * D + d, x.to(tl.bfloat16))
    else:
        tl.store(vc + (P * KVH + (h - HQ - KVH)) * D + d, tl.minimum(tl.maximum(x * inv_v, -448.0), 448.0).to(tl.float8e4nv))
        if EXACT:
            tl.store(vx + (t * KVH + (h - HQ - KVH)) * D + d, x.to(tl.bfloat16))


@triton.jit(do_not_specialize=["TQ"])
def conv_gdn_kernel(mixed, row_stride, window, conv_w, out_q, out_k, out_v, ckpt, ab, A, dt_bias, g_out, beta_out,
                    TQ, LK: tl.constexpr, LV: tl.constexpr, DK: tl.constexpr, CONV: tl.constexpr):
    """Program head (16 q, 16 k, 48 v heads of 128 channels): the causal conv (width 4) of the TQ new rows over the
    window before them, SiLU, the L2 norm of q and k heads (eps 1e-6, as fla's and the reference's), FP32 out for the
    column scan; ckpt[t] receives the window after row t (for the commit). Program 80 computes the gates
    g = A * softplus(a + dt_bias) and beta = sigmoid(b) of the TQ rows. TQ is a runtime loop bound, so one compiled
    kernel serves every column count and a column gets the same bits alone or inside a verify."""
    h = tl.program_id(0)
    if h == 2 * LK + LV:
        j = tl.arange(0, 64)
        jm = j < LV
        for t in range(TQ):
            a = tl.load(ab + t * 2 * LV + j, mask=jm, other=0.0).to(tl.float32)
            b = tl.load(ab + t * 2 * LV + LV + j, mask=jm, other=0.0).to(tl.float32)
            x = a + tl.load(dt_bias + j, mask=jm, other=0.0)
            sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
            tl.store(g_out + t * LV + j, tl.load(A + j, mask=jm, other=0.0) * sp, mask=jm)
            tl.store(beta_out + t * LV + j, tl.sigmoid(b), mask=jm)
        return
    ch = h * DK + tl.arange(0, DK)
    w0 = tl.load(conv_w + ch * 4 + 0)
    w1 = tl.load(conv_w + ch * 4 + 1)
    w2 = tl.load(conv_w + ch * 4 + 2)
    w3 = tl.load(conv_w + ch * 4 + 3)
    s0 = tl.load(window + ch * 3 + 0)
    s1 = tl.load(window + ch * 3 + 1)
    s2 = tl.load(window + ch * 3 + 2)
    for t in range(TQ):
        x = tl.load(mixed + t * row_stride + ch).to(tl.float32)
        y = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x
        y = y * tl.sigmoid(y)
        s0 = s1
        s1 = s2
        s2 = x
        tl.store(ckpt + (t * CONV + ch) * 3 + 0, s0)
        tl.store(ckpt + (t * CONV + ch) * 3 + 1, s1)
        tl.store(ckpt + (t * CONV + ch) * 3 + 2, s2)
        if h < 2 * LK:
            y = y / tl.sqrt(tl.sum(y * y, axis=0) + 1e-6)
            if h < LK:
                tl.store(out_q + (t * LK + h) * DK + tl.arange(0, DK), y)
            else:
                tl.store(out_k + (t * LK + h - LK) * DK + tl.arange(0, DK), y)
        else:
            tl.store(out_v + (t * LV + h - 2 * LK) * DK + tl.arange(0, DK), y)
