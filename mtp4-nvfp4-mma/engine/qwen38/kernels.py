"""Triton kernels of the engine's own: activation quantization fused with the norms and activations around it, and
attention over the FP8 KV cache. GEMMs are cuBLASLt's (torch._scaled_mm), the gated-delta recurrence is
flash-linear-attention's."""
import torch
import triton
import triton.language as tl

FP8_MAX = 448.0
E2M1_MAX = 6.0


# ---------------------------------------------------------------------------------------------------------------
# Activation quantization. FP8 per token (dynamic, amax / 448) and NVFP4 in groups of 16 with the checkpoint's
# static global scale g (a divisor): group scale e4m3(amax / 6 * g), step = scale / g, codes E2M1 rounded to
# nearest with ties to the even code, as compressed-tensors' and crit-eval's reference quantize them.
# ---------------------------------------------------------------------------------------------------------------

@triton.jit
def _e2m1_code(v):
    """|x| / step (clamped to 6) to its E2M1 magnitude code, ties to even."""
    c = (v > 0.25).to(tl.int32) + (v >= 0.75).to(tl.int32) + (v > 1.25).to(tl.int32) + (v >= 1.75).to(tl.int32)
    c += (v > 2.5).to(tl.int32) + (v >= 3.5).to(tl.int32) + (v > 5.0).to(tl.int32)
    return c


@triton.jit
def _nvfp4_store(y, row, K: tl.constexpr, g, out_packed, out_scale, n_cb, BLOCK: tl.constexpr):
    """Quantize one row y [BLOCK] fp32 (its first K values; the rest zero) to packed E2M1 pairs and swizzled e4m3
    group scales."""
    groups = tl.reshape(y, (BLOCK // 16, 16))
    amax = tl.max(tl.abs(groups), axis=1)
    stored = tl.minimum(amax / 6.0 * g, 448.0).to(tl.float8e4nv)
    step = stored.to(tl.float32) / g
    safe = tl.where(step > 0, step, 1.0)
    v = tl.minimum(tl.abs(groups) / safe[:, None], 6.0)
    code = _e2m1_code(v) | tl.where(groups < 0, 8, 0)
    code = tl.reshape(code, (BLOCK // 2, 2))
    lo, hi = tl.split(code)
    packed = (lo | (hi << 4)).to(tl.uint8)
    p = tl.arange(0, BLOCK // 2)
    tl.store(out_packed + row * (K // 2) + p, packed, mask=p < K // 2)
    # cuBLASLt's 128x4 tile order of the group scales: row r, group c.
    c = tl.arange(0, BLOCK // 16)
    rb, rr = row // 128, row % 128
    cb, cc = c // 4, c % 4
    offset = (rb * n_cb + cb) * 512 + (rr % 32) * 16 + (rr // 32) * 4 + cc
    tl.store(out_scale + offset, stored, mask=c < K // 16)


@triton.jit
def _fp8_store(y, row, K: tl.constexpr, out_q, out_scale, BLOCK: tl.constexpr):
    amax = tl.max(tl.abs(y), axis=0)
    step = tl.maximum(amax / 448.0, 1.1754943508222875e-38)
    q = tl.minimum(tl.maximum(y / step, -448.0), 448.0).to(tl.float8e4nv)
    offs = tl.arange(0, BLOCK)
    tl.store(out_q + row * K + offs, q, mask=offs < K)
    tl.store(out_scale + row, step)


@triton.jit
def rmsnorm_quant_kernel(x, w, eps, g, out_q, out_scale, n_cb, out_y, delta, delta_stride, alpha, dxs, dws,
                         K: tl.constexpr, NVFP4: tl.constexpr, WRITE_Y: tl.constexpr, ADD: tl.constexpr,
                         ROWCOL: tl.constexpr = False, BLOCK: tl.constexpr = 8192):
    """ADD: first the residual update x += alpha * delta (a GEMM's bf16 output row) written back. Then the
    zero-centered RMSNorm (x * rsqrt(mean x^2 + eps) * (1 + w)) of the fp32 row, quantized for the next GEMM;
    WRITE_Y also stores the normed row in bf16 (for the unquantized matrices beside it)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < K
    xr = tl.load(x + row * K + offs, mask=m, other=0.0).to(tl.float32)
    if ADD:
        dv = tl.load(delta + row * delta_stride + offs, mask=m, other=0.0).to(tl.float32) * tl.load(alpha)
        if ROWCOL:
            dv = dv * tl.load(dxs + row) * tl.load(dws + offs, mask=m, other=0.0)
        xr = xr + dv
        tl.store(x + row * K + offs, xr, mask=m)
    var = tl.sum(xr * xr, axis=0) / K
    y = xr * tl.rsqrt(var + eps) * (1.0 + tl.load(w + offs, mask=m, other=0.0).to(tl.float32))
    if WRITE_Y:
        tl.store(out_y + row * K + offs, y.to(tl.bfloat16), mask=m)
    if NVFP4:
        _nvfp4_store(y, row, K, g, out_q, out_scale, n_cb, BLOCK)
    else:
        _fp8_store(y, row, K, out_q, out_scale, BLOCK)


@triton.jit
def residual_add_kernel(x, delta, delta_stride, alpha, dxs, dws, K: tl.constexpr, ROWCOL: tl.constexpr = False,
                        BLOCK: tl.constexpr = 8192):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < K
    xr = tl.load(x + row * K + offs, mask=m, other=0.0)
    dv = tl.load(delta + row * delta_stride + offs, mask=m, other=0.0).to(tl.float32) * tl.load(alpha)
    if ROWCOL:
        dv = dv * tl.load(dxs + row) * tl.load(dws + offs, mask=m, other=0.0)
    xr = xr + dv
    tl.store(x + row * K + offs, xr, mask=m)


@triton.jit
def conv_prefill_kernel(x, x_stride, window, conv_w, out, T, xxs, xws, C: tl.constexpr, BC: tl.constexpr,
                        BT: tl.constexpr, ROWCOL: tl.constexpr = False):
    """Causal depthwise conv (width 4) + SiLU of prompt rows x [T, C] (bf16, row stride x_stride) after the window
    [C, 3] (fp32, oldest first): out [T, C] bf16. Program (channel block, row block)."""
    cb = tl.program_id(0)
    tb = tl.program_id(1)
    c = cb * BC + tl.arange(0, BC)
    t = tb * BT + tl.arange(0, BT)
    tm = t < T
    acc = tl.zeros([BT, BC], tl.float32)
    for i in tl.static_range(4):
        src = t - 3 + i
        from_x = src >= 0
        xv = tl.load(x + src[:, None] * x_stride + c[None, :], mask=(from_x & tm)[:, None], other=0.0).to(tl.float32)
        if ROWCOL:
            xv = xv * tl.load(xxs + src, mask=from_x & tm, other=0.0)[:, None] * tl.load(xws + c)[None, :]
        wv = tl.load(window + c[None, :] * 3 + (src[:, None] + 3), mask=(~from_x & tm)[:, None], other=0.0)
        acc += (xv + wv) * tl.load(conv_w + c * 4 + i)[None, :]
    acc = acc * tl.sigmoid(acc)
    tl.store(out + t[:, None] * C + c[None, :], acc.to(tl.bfloat16), mask=tm[:, None])


@triton.jit
def gdn_gates_kernel(ab, A, dt_bias, g_out, beta_out, rows, LV: tl.constexpr):
    """g = A * softplus(a + dt_bias) (A = -exp(A_log)) and beta = sigmoid(b) of one row of [a | b] (bf16); rows from
    `rows` on are padding and get g = 0, beta = 0, which leave the gated-delta state as it is."""
    t = tl.program_id(0)
    real = t < tl.load(rows)
    j = tl.arange(0, 64)
    jm = j < LV
    a = tl.load(ab + t * 2 * LV + j, mask=jm, other=0.0).to(tl.float32)
    b = tl.load(ab + t * 2 * LV + LV + j, mask=jm, other=0.0).to(tl.float32)
    x = a + tl.load(dt_bias + j, mask=jm, other=0.0)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    tl.store(g_out + t * LV + j, tl.where(real, tl.load(A + j, mask=jm, other=0.0) * sp, 0.0), mask=jm)
    tl.store(beta_out + t * LV + j, tl.where(real, tl.sigmoid(b), 0.0), mask=jm)


@triton.jit
def silu_mul_quant_kernel(gu, alpha_g, alpha_u, g, out_q, out_scale, n_cb, gxs, gws, N: tl.constexpr,
                          CHUNK: tl.constexpr, NVFP4: tl.constexpr, ROWCOL: tl.constexpr = False):
    """silu(gate) * up of one row of the fused gate/up GEMM (bf16 [2N], each half rescaled by its alpha), quantized
    for the down projection. Rows of 17,408 run in chunks; NVFP4 needs no row-wide amax, FP8 takes two passes."""
    row = tl.program_id(0)
    ag = tl.load(alpha_g)
    au = tl.load(alpha_u)
    if NVFP4:
        for start in tl.static_range(0, N, CHUNK):
            offs = start + tl.arange(0, CHUNK)
            a = tl.load(gu + row * 2 * N + offs).to(tl.float32) * ag
            b = tl.load(gu + row * 2 * N + N + offs).to(tl.float32) * au
            y = a * tl.sigmoid(a) * b
            groups = tl.reshape(y, (CHUNK // 16, 16))
            amax = tl.max(tl.abs(groups), axis=1)
            stored = tl.minimum(amax / 6.0 * g, 448.0).to(tl.float8e4nv)
            step = stored.to(tl.float32) / g
            safe = tl.where(step > 0, step, 1.0)
            v = tl.minimum(tl.abs(groups) / safe[:, None], 6.0)
            code = _e2m1_code(v) | tl.where(groups < 0, 8, 0)
            code = tl.reshape(code, (CHUNK // 2, 2))
            lo, hi = tl.split(code)
            tl.store(out_q + row * (N // 2) + start // 2 + tl.arange(0, CHUNK // 2), (lo | (hi << 4)).to(tl.uint8))
            c = start // 16 + tl.arange(0, CHUNK // 16)
            rb, rr = row // 128, row % 128
            cb, cc = c // 4, c % 4
            tl.store(out_scale + (rb * n_cb + cb) * 512 + (rr % 32) * 16 + (rr // 32) * 4 + cc, stored)
    else:
        amax = 0.0
        rs = tl.load(gxs + row) if ROWCOL else 1.0
        for start in tl.static_range(0, N, CHUNK):
            offs = start + tl.arange(0, CHUNK)
            a = tl.load(gu + row * 2 * N + offs).to(tl.float32) * ag
            b = tl.load(gu + row * 2 * N + N + offs).to(tl.float32) * au
            if ROWCOL:
                a = a * rs * tl.load(gws + offs)
                b = b * rs * tl.load(gws + N + offs)
            y = a * tl.sigmoid(a) * b
            amax = tl.maximum(amax, tl.max(tl.abs(y), axis=0))
        step = tl.maximum(amax / 448.0, 1.1754943508222875e-38)
        for start in tl.static_range(0, N, CHUNK):
            offs = start + tl.arange(0, CHUNK)
            a = tl.load(gu + row * 2 * N + offs).to(tl.float32) * ag
            b = tl.load(gu + row * 2 * N + N + offs).to(tl.float32) * au
            if ROWCOL:
                a = a * rs * tl.load(gws + offs)
                b = b * rs * tl.load(gws + N + offs)
            y = a * tl.sigmoid(a) * b
            q = tl.minimum(tl.maximum(y / step, -448.0), 448.0).to(tl.float8e4nv)
            tl.store(out_q + row * N + offs, q)
        tl.store(out_scale + row, step)


@triton.jit
def gated_rmsnorm_fp8_kernel(o, z, w, eps, out_q, out_scale, z_stride, zxs, zws, H: tl.constexpr, D: tl.constexpr,
                             HP: tl.constexpr, ROWCOL: tl.constexpr = False):
    """The gated delta output's RMSNormGated (per head of D: o * rsqrt(mean o^2 + eps) * w * silu(z)) of one token's
    H heads (padded to HP), quantized FP8 per token for out_proj. z rows are z_stride apart."""
    row = tl.program_id(0)
    hm = tl.arange(0, HP) < H
    offs = tl.arange(0, HP)[:, None] * D + tl.arange(0, D)[None, :]
    x = tl.load(o + row * H * D + offs, mask=hm[:, None], other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=1) / D
    y = x * tl.rsqrt(var + eps)[:, None]
    y = y * tl.load(w + tl.arange(0, D)).to(tl.float32)[None, :]
    zz = tl.load(z + row * z_stride + offs, mask=hm[:, None], other=0.0).to(tl.float32)
    if ROWCOL:
        zz = zz * tl.load(zxs + row) * tl.load(zws + offs, mask=hm[:, None], other=0.0)
    y = y * zz * tl.sigmoid(zz)
    amax = tl.max(tl.max(tl.abs(y), axis=1), axis=0)
    step = tl.maximum(amax / 448.0, 1.1754943508222875e-38)
    q = tl.minimum(tl.maximum(y / step, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(out_q + row * H * D + offs, q, mask=hm[:, None])
    tl.store(out_scale + row, step)


@triton.jit
def fp8_quant_kernel(x, out_q, out_scale, K: tl.constexpr, BLOCK: tl.constexpr = 8192):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    y = tl.load(x + row * K + offs, mask=offs < K, other=0.0).to(tl.float32)
    _fp8_store(y, row, K, out_q, out_scale, BLOCK)


@triton.jit
def embed_gather_kernel(table, ids, out, K: tl.constexpr, BLOCK: tl.constexpr):
    """out[r] = FP32 of table[ids[r]]: the bf16 embedding table read in place from mapped pinned host memory."""
    r = tl.program_id(0)
    tok = tl.load(ids + r).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < K
    tl.store(out + r * K + offs, tl.load(table + tok * K + offs, mask=m, other=0.0).to(tl.float32), mask=m)


class HostPointer:
    """A pinned host tensor handed to a Triton kernel by its mapped address (the same address on the device under
    unified addressing)."""

    def __init__(self, tensor):
        self.tensor = tensor
        self.dtype = tensor.dtype

    def data_ptr(self):
        return self.tensor.data_ptr()


def scale_buffer(rows, groups, device):
    """Storage for cuBLASLt-ordered NVFP4 group scales of `rows` rows: padded to 128 rows and 4 groups."""
    return torch.zeros(-(-rows // 128) * 128 * (-(-groups // 4) * 4), dtype=torch.float8_e4m3fn, device=device)


def swizzle_scales(m):
    """[rows, groups] e4m3 group scales in cuBLASLt's 128x4 tile order (flat)."""
    rows, cols = m.shape
    rb, cb = -(-rows // 128), -(-cols // 4)
    p = torch.zeros(rb * 128, cb * 4, dtype=m.dtype, device=m.device)
    p[:rows, :cols] = m
    blocks = p.view(rb, 128, cb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten().contiguous()


# ---------------------------------------------------------------------------------------------------------------
# Attention over the FP8 KV cache [S, KVH, D] (e4m3 times the checkpoint's static k_scale / v_scale), GQA, causal,
# with the gated attention's output gate (o * sigmoid(gate)) in the epilogue.
# ---------------------------------------------------------------------------------------------------------------

@triton.jit
def attn_prefill_kernel(q, kc, vc, kx, vx, gate, out, p0_ptr, T, sm_scale, k_scale, v_scale,
                        HQ: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """Queries at positions p0 + [0, T) (q [T, HQ, D] bf16) against keys [0, p0 + T): the earlier ones from the FP8
    cache (times k_scale / v_scale), the chunk's own from its exact bf16 keys and values kx, vx [T, KVH, D]."""
    mb = tl.program_id(0)
    h = tl.program_id(1)
    p0 = tl.load(p0_ptr)
    kvh = h // (HQ // KVH)
    m_offs = mb * BM + tl.arange(0, BM)
    d = tl.arange(0, D)
    qm = m_offs < T
    qt = tl.load(q + (m_offs[:, None] * HQ + h) * D + d[None, :], mask=qm[:, None], other=0.0)
    m_i = tl.full([BM], -float("inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    qk = sm_scale * 1.4426950408889634
    for start in range(0, p0, BN):
        n = start + tl.arange(0, BN)
        km = n < p0
        k = tl.load(kc + (n[:, None] * KVH + kvh) * D + d[None, :], mask=km[:, None], other=0.0).to(tl.bfloat16)
        s = tl.dot(qt, tl.trans(k)) * (qk * k_scale)
        s = tl.where(km[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(vc + (n[:, None] * KVH + kvh) * D + d[None, :], mask=km[:, None], other=0.0).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot((p * v_scale).to(tl.bfloat16), v)
        m_i = m_new
    end = tl.minimum((mb + 1) * BM, T)
    for start in range(0, end, BN):
        n = start + tl.arange(0, BN)
        km = n < end
        k = tl.load(kx + (n[:, None] * KVH + kvh) * D + d[None, :], mask=km[:, None], other=0.0)
        s = tl.dot(qt, tl.trans(k)) * qk
        s = tl.where((n[None, :] <= m_offs[:, None]) & km[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(vx + (n[:, None] * KVH + kvh) * D + d[None, :], mask=km[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    o = acc / l_i[:, None]
    gt = tl.load(gate + (m_offs[:, None] * HQ + h) * D + d[None, :], mask=qm[:, None], other=0.0).to(tl.float32)
    o = o * tl.sigmoid(gt)
    tl.store(out + (m_offs[:, None] * HQ + h) * D + d[None, :], o.to(tl.bfloat16), mask=qm[:, None])





@triton.jit
def nvfp4_dyn_quant_kernel(x, x_stride, out_q, out_scale, g_out, n_cb, K: tl.constexpr, CHUNK: tl.constexpr):
    """One row of x (fp32 or bf16) to NVFP4 at its own global scale g = 2688 / amax (so the largest group scale is
    e4m3's 448): packed codes, cuBLASLt-ordered group scales, and 1 / g in g_out for the GEMV's per-row factor. For the
    drafter's activations, whose range has no stored scale."""
    row = tl.program_id(0)
    amax = 0.0
    for start in tl.static_range(0, K, CHUNK):
        offs = start + tl.arange(0, CHUNK)
        amax = tl.maximum(amax, tl.max(tl.abs(tl.load(x + row * x_stride + offs).to(tl.float32)), axis=0))
    g = 2688.0 / tl.maximum(amax, 1e-30)
    tl.store(g_out + row, 1.0 / g)
    for start in tl.static_range(0, K, CHUNK):
        offs = start + tl.arange(0, CHUNK)
        y = tl.load(x + row * x_stride + offs).to(tl.float32)
        groups = tl.reshape(y, (CHUNK // 16, 16))
        gmax = tl.max(tl.abs(groups), axis=1)
        stored = tl.minimum(gmax / 6.0 * g, 448.0).to(tl.float8e4nv)
        step = stored.to(tl.float32) / g
        safe = tl.where(step > 0, step, 1.0)
        v = tl.minimum(tl.abs(groups) / safe[:, None], 6.0)
        code = _e2m1_code(v) | tl.where(groups < 0, 8, 0)
        code = tl.reshape(code, (CHUNK // 2, 2))
        lo, hi = tl.split(code)
        tl.store(out_q + row * (K // 2) + start // 2 + tl.arange(0, CHUNK // 2), (lo | (hi << 4)).to(tl.uint8))
        c = start // 16 + tl.arange(0, CHUNK // 16)
        rb, rr = row // 128, row % 128
        cb, cc = c // 4, c % 4
        tl.store(out_scale + (rb * n_cb + cb) * 512 + (rr % 32) * 16 + (rr // 32) * 4 + cc, stored)


@triton.jit
def silu_mul_nvfp4_kernel(gu, alpha_g, alpha_u, g, out_q, out_scale, n_cb, N: tl.constexpr, CHUNK: tl.constexpr):
    """silu(gate) * up quantized to NVFP4 at the static input scale g, one program per CHUNK of a row (NVFP4 needs no
    row-wide amax): the same values and codes as silu_mul_quant_kernel's NVFP4 path."""
    row = tl.program_id(0)
    start = tl.program_id(1) * CHUNK
    ag = tl.load(alpha_g)
    au = tl.load(alpha_u)
    offs = start + tl.arange(0, CHUNK)
    a = tl.load(gu + row * 2 * N + offs).to(tl.float32) * ag
    b = tl.load(gu + row * 2 * N + N + offs).to(tl.float32) * au
    y = a * tl.sigmoid(a) * b
    groups = tl.reshape(y, (CHUNK // 16, 16))
    amax = tl.max(tl.abs(groups), axis=1)
    stored = tl.minimum(amax / 6.0 * g, 448.0).to(tl.float8e4nv)
    step = stored.to(tl.float32) / g
    safe = tl.where(step > 0, step, 1.0)
    v = tl.minimum(tl.abs(groups) / safe[:, None], 6.0)
    code = _e2m1_code(v) | tl.where(groups < 0, 8, 0)
    code = tl.reshape(code, (CHUNK // 2, 2))
    lo, hi = tl.split(code)
    tl.store(out_q + row * (N // 2) + start // 2 + tl.arange(0, CHUNK // 2), (lo | (hi << 4)).to(tl.uint8))
    c = start // 16 + tl.arange(0, CHUNK // 16)
    rb, rr = row // 128, row % 128
    cb, cc = c // 4, c % 4
    tl.store(out_scale + (rb * n_cb + cb) * 512 + (rr % 32) * 16 + (rr // 32) * 4 + cc, stored)
