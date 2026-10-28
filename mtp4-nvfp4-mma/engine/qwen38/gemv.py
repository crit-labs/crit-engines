"""The engine's GEMV for decode and verify (1 to 4 columns): FP8 W8A8 with per-row scales, and NVFP4 W4A4 reading the
weights' cuBLASLt-ordered group scales in place. Products are the same codes times the same scales the prefill GEMMs
multiply (exact in fp16 before the FP32 sums), split along K in a fixed order without atomics, so a column computes the
same bits alone or beside others."""
import torch
import triton
import triton.language as tl

ROWS = 16                                    # MMA rows: the columns padded


@triton.jit
def fp8_gemv_kernel(x, xs, w, ws, out, part, M, N, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    SPLIT: tl.constexpr, R: tl.constexpr, OUT_F32: tl.constexpr):
    nb = tl.program_id(0)
    sp = tl.program_id(1)
    n = nb * BN + tl.arange(0, BN)
    r = tl.arange(0, R)
    nm = n < N
    rm = r < M
    acc = tl.zeros([R, BN], tl.float32)
    KS: tl.constexpr = K // SPLIT
    for k0 in tl.range(sp * KS, (sp + 1) * KS, BK, num_stages=4):
        k = k0 + tl.arange(0, BK)
        xt = tl.load(x + r[:, None] * K + k[None, :], mask=rm[:, None], other=0.0)
        wt = tl.load(w + n[:, None] * K + k[None, :], mask=nm[:, None], other=0.0)
        acc = tl.dot(xt, tl.trans(wt), acc)
    if SPLIT == 1:
        sx = tl.load(xs + r, mask=rm, other=0.0)
        sw = tl.load(ws + n, mask=nm, other=0.0)
        y = acc * sx[:, None] * sw[None, :]
        tl.store(out + r[:, None] * N + n[None, :], y.to(tl.float32 if OUT_F32 else tl.bfloat16),
                 mask=rm[:, None] & nm[None, :])
    else:
        tl.store(part + (sp * R + r[:, None]) * N + n[None, :], acc, mask=rm[:, None] & nm[None, :])


@triton.jit
def _e2m1_fp16(c):
    """E2M1 codes (int32, 4 bits) to their fp16 values: exponent e, mantissa f: e > 0: (1 + f / 2) 2^(e - 1); e = 0:
    f / 2; sign bit 3."""
    e = (c >> 1) & 3
    f = c & 1
    bits = tl.where(e > 0, ((e + 14) << 10) | (f << 9), tl.where(f > 0, 0x3800, 0))
    bits = bits | ((c & 8) << 12)
    return bits.to(tl.int16).to(tl.float16, bitcast=True)


@triton.jit
def nvfp4_gemv_kernel(x, xsc, w, wsc, alpha, out, part, M, N, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      SPLIT: tl.constexpr, R: tl.constexpr):
    """x packed [M, K / 2] with swizzled e4m3 scales xsc, w packed [N, K / 2] with swizzled scales wsc. The sum over K
    runs as the even elements (low nibbles) plus the odd ones (high nibbles): both nibbles of a byte share its group
    of 16, so no interleave is needed."""
    nb = tl.program_id(0)
    sp = tl.program_id(1)
    n = nb * BN + tl.arange(0, BN)
    r = tl.arange(0, R)
    nm = n < N
    rm = r < M
    n_cb: tl.constexpr = K // 64
    acc = tl.zeros([R, BN], tl.float32)
    KS: tl.constexpr = K // SPLIT
    HB: tl.constexpr = BK // 2
    GB: tl.constexpr = BK // 16
    for k0 in tl.range(sp * KS, (sp + 1) * KS, BK, num_stages=3):
        b = k0 // 2 + tl.arange(0, HB)
        g = k0 // 16 + tl.arange(0, GB)
        wp = tl.load(w + n[:, None] * (K // 2) + b[None, :], mask=nm[:, None], other=0).to(tl.int32)
        woff = ((n[:, None] // 128) * n_cb + g[None, :] // 4) * 512 + (n[:, None] % 32) * 16 +             ((n[:, None] % 128) // 32) * 4 + g[None, :] % 4
        wsg = tl.load(wsc + woff, mask=nm[:, None], other=0.0).to(tl.float16)[:, :, None]
        xp = tl.load(x + r[:, None] * (K // 2) + b[None, :], mask=rm[:, None], other=0).to(tl.int32)
        xoff = ((r[:, None] // 128) * n_cb + g[None, :] // 4) * 512 + (r[:, None] % 32) * 16 +             ((r[:, None] % 128) // 32) * 4 + g[None, :] % 4
        xsg = tl.load(xsc + xoff, mask=rm[:, None], other=0.0).to(tl.float16)[:, :, None]
        w_lo = tl.reshape(tl.reshape(_e2m1_fp16(wp & 15), (BN, GB, 8)) * wsg, (BN, HB))
        w_hi = tl.reshape(tl.reshape(_e2m1_fp16(wp >> 4), (BN, GB, 8)) * wsg, (BN, HB))
        x_lo = tl.reshape(tl.reshape(_e2m1_fp16(xp & 15), (R, GB, 8)) * xsg, (R, HB))
        x_hi = tl.reshape(tl.reshape(_e2m1_fp16(xp >> 4), (R, GB, 8)) * xsg, (R, HB))
        acc = tl.dot(x_lo, tl.trans(w_lo), acc)
        acc = tl.dot(x_hi, tl.trans(w_hi), acc)
    if SPLIT == 1:
        y = acc * tl.load(alpha)
        tl.store(out + r[:, None] * N + n[None, :], y.to(tl.bfloat16), mask=rm[:, None] & nm[None, :])
    else:
        tl.store(part + (sp * R + r[:, None]) * N + n[None, :], acc, mask=rm[:, None] & nm[None, :])


@triton.jit
def split_reduce_kernel(part, xs, ws, alpha, out, M, N, SPLIT: tl.constexpr, R: tl.constexpr, BN: tl.constexpr,
                        FP8: tl.constexpr, OUT_F32: tl.constexpr = False):
    """Sum the splits in order and scale: FP8 by the row and column scales, NVFP4 by alpha."""
    nb = tl.program_id(0)
    m = tl.program_id(1)
    n = nb * BN + tl.arange(0, BN)
    nm = n < N
    acc = tl.zeros([BN], tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(part + (s * R + m) * N + n, mask=nm, other=0.0)
    if FP8:
        acc = acc * tl.load(xs + m) * tl.load(ws + n, mask=nm, other=0.0)
    else:
        acc = acc * tl.load(alpha)
    tl.store(out + m * N + n, acc.to(tl.float32 if OUT_F32 else tl.bfloat16), mask=nm)


def _split(N, K, BN, BK):
    """One split: splitting K and reducing measured slower at every decode shape (work/evidence/fp8-split-c4.txt)."""
    return 1


def fp8_gemv(xq, xs, w, ws, out_dtype=torch.bfloat16, BN=64, BK=256, split=None):
    """xq [M, K] e4m3 (M <= 16), xs [M] fp32; w [N, K] e4m3, ws [N] fp32. Returns [M, N] bf16 (or fp32 logits)."""
    M, K = xq.shape
    N = w.shape[0]
    if K % BK:
        raise ValueError(f"fp8_gemv: K {K} is not a multiple of {BK}")
    split = split or _split(N, K, BN, BK)
    f32 = out_dtype == torch.float32
    out = torch.empty(M, N, dtype=out_dtype, device=xq.device)
    part = torch.empty(split, ROWS, N, dtype=torch.float32, device=xq.device) if split > 1 else out
    fp8_gemv_kernel[(-(-N // BN), split)](xq, xs, w, ws, out, part, M, N, K=K, BN=BN, BK=BK, SPLIT=split, R=ROWS,
                                          OUT_F32=f32, num_warps=4)
    if split > 1:
        split_reduce_kernel[(-(-N // 256), M)](part, xs, ws, xs, out, M, N, SPLIT=split, R=ROWS, BN=256, FP8=True,
                                               OUT_F32=f32, num_warps=4)
    return out


def nvfp4_gemv(xq, xsc, w, wsc, alpha, BN=64, BK=256):
    """xq packed [M, K / 2] uint8 with swizzled scales xsc; w packed [N, K / 2] with swizzled scales wsc."""
    M, K2 = xq.shape
    K = K2 * 2
    N = w.shape[0]
    split = _split(N, K, BN, BK)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=xq.device)
    part = torch.empty(split, ROWS, N, dtype=torch.float32, device=xq.device) if split > 1 else out
    nvfp4_gemv_kernel[(-(-N // BN), split)](xq, xsc, w, wsc, alpha, out, part, M, N, K=K, BN=BN, BK=BK, SPLIT=split,
                                            R=ROWS, num_warps=4)
    if split > 1:
        split_reduce_kernel[(-(-N // 256), M)](part, alpha, alpha, alpha, out, M, N, SPLIT=split, R=ROWS, BN=256,
                                               FP8=False, num_warps=4)
    return out


@triton.jit
def bf16_gemv_kernel(x, w, part, M, N, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr,
                     R: tl.constexpr):
    """Partial sums of x [M, K] bf16 times w [N, K] bf16 over split sp's share of K."""
    nb = tl.program_id(0)
    sp = tl.program_id(1)
    n = nb * BN + tl.arange(0, BN)
    r = tl.arange(0, R)
    nm = n < N
    rm = r < M
    acc = tl.zeros([R, BN], tl.float32)
    KS: tl.constexpr = K // SPLIT
    for k0 in tl.range(sp * KS, (sp + 1) * KS, BK, num_stages=3):
        k = k0 + tl.arange(0, BK)
        xt = tl.load(x + r[:, None] * K + k[None, :], mask=rm[:, None], other=0.0)
        wt = tl.load(w + n[:, None] * K + k[None, :], mask=nm[:, None], other=0.0)
        acc = tl.dot(xt, tl.trans(wt), acc)
    tl.store(part + (sp * R + r[:, None]) * N + n[None, :], acc, mask=rm[:, None] & nm[None, :])


def bf16_gemv(x, w, split=16, BN=16, BK=64):
    """x [M, K] bf16 (M <= 16) times w [N, K] bf16 -> [M, N] bf16, split along K in a fixed order."""
    M, K = x.shape
    if K % (split * BK):
        raise ValueError(f"bf16_gemv: K {K} is not a multiple of {split} splits of {BK}")
    N = w.shape[0]
    part = torch.empty(split, ROWS, N, dtype=torch.float32, device=x.device)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    bf16_gemv_kernel[(-(-N // BN), split)](x, w, part, M, N, K=K, BN=BN, BK=BK, SPLIT=split, R=ROWS, num_warps=4)
    one = _ones(x.device)
    split_reduce_kernel[(-(-N // 256), M)](part, one, one, one, out, M, N, SPLIT=split, R=ROWS, BN=256, FP8=False,
                                           num_warps=4)
    return out


_ONES = {}


def _ones(device):
    if device not in _ONES:
        _ONES[device] = torch.ones(1, dtype=torch.float32, device=device)
    return _ONES[device]
