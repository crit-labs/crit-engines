"""Qwen3.8-27B (Qwen3_5ForConditionalGeneration's text model) as stored by the unsloth NVFP4 checkpoint, run as
quantized: FP8 W8A8 (per-channel weights, per-token dynamic activations) for the attention, gated-delta and output
projections and the last 8 MLPs, NVFP4 W4A4 (group-16 weights and activations with the checkpoint's global scales)
for the other MLPs, an FP8 KV cache at the checkpoint's static k/v scales, the gated-delta recurrence in FP32.

The residual stream stays FP32. Prefill runs in chunks; decode is one token per step and is captured in a CUDA graph
by the engine (every length it reads is a device tensor)."""
import math
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

from . import kernels as K
from . import kernels_step as KS
from . import gemv as GV
from . import nvfp4 as NV

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: E402
from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel  # noqa: E402

HIDDEN = 5120
LAYERS = 64
INTER = 17408
VOCAB = 248320
HQ, KVH, HD = 24, 4, 256
ROT = 64
LK, LV, DK, DV = 16, 48, 128, 128
CONV = 2 * LK * DK + LV * DV        # 10240
EPS = 1e-6
THETA = 10_000_000.0
FP8_MLP_FROM = 56                    # layers 56..63 store FP8 MLPs
MAX_COLUMNS = 8                      # widest verify (draft steps + 1)
GEMV_ROWS = 8


def is_attention(layer):
    return layer % 4 == 3


_UNIT = {}
_UNIT_VEC = {}


def _unit_vec(device):
    if device not in _UNIT_VEC:
        _UNIT_VEC[device] = torch.ones(1, dtype=torch.float32, device=device)
    return _UNIT_VEC[device]


def _unit(device):
    if device not in _UNIT:
        _UNIT[device] = torch.ones((), dtype=torch.float32, device=device)
    return _UNIT[device]


class Fp8Linear:
    """W8A8: e4m3 weight [N, K] with per-row (output channel) scales. Up to GEMV_ROWS columns run the engine's GEMV
    (decode and verify), more run cuBLASLt (prefill)."""

    def __init__(self, weights, scales):
        self.w = torch.cat(weights, 0) if len(weights) > 1 else weights[0]
        self.s = torch.cat(scales, 0).float().reshape(1, -1).contiguous()
        self.s_flat = self.s.view(-1)
        self.n = self.w.shape[0]

    def unscaled(self, xq):
        """The prompt GEMM at tensor-wise unit scales (cuBLASLt's faster FP8 path): sum(xq * w) in bf16, to be
        multiplied by the row's xs and the column's s by its consumer."""
        if xq.shape[0] <= GEMV_ROWS:
            return None
        return torch._scaled_mm(xq, self.w.t(), scale_a=_unit(xq.device), scale_b=_unit(xq.device),
                                out_dtype=torch.bfloat16)

    def __call__(self, xq, xs, out_dtype=torch.bfloat16):
        if xq.shape[0] <= GEMV_ROWS:
            bn = 64 if self.n <= 8192 and self.w.shape[1] <= 6144 else 32
            return GV.fp8_gemv(xq, xs, self.w, self.s_flat, out_dtype=out_dtype, BN=bn, BK=256)
        return torch._scaled_mm(xq, self.w.t(), scale_a=xs.view(-1, 1), scale_b=self.s, out_dtype=out_dtype)

    @property
    def nbytes(self):
        return self.w.numel() + self.s.numel() * 4


class Nvfp4Linear:
    """W4A4: E2M1 pairs [N, K/2] with e4m3 group-16 scales (swizzled for cuBLASLt) and a global scale per stored
    matrix; `alphas` are 1 / (input_global * weight_global) of each stored matrix concatenated along N."""

    def __init__(self, packed, scales, weight_globals, input_global):
        self.w = (torch.cat(packed, 0) if len(packed) > 1 else packed[0]).view(torch.float4_e2m1fn_x2)
        raw = torch.cat(scales, 0) if len(scales) > 1 else scales[0]
        self.raw_scale_bytes = raw.numel()
        self.s = K.swizzle_scales(raw)
        self.g = float(input_global)
        self.alphas = [torch.tensor([1.0 / (self.g * float(wg))], device=raw.device) for wg in weight_globals]
        self.n, self.k = self.w.shape[0], self.w.shape[1] * 2
        self.w_u8 = self.w.view(torch.uint8)
        self.tiling = (1, True) if self.n >= 16384 else (8, False)

    def __call__(self, xq, xs):
        """Raw products (the caller applies the alphas): the engine's tensor-core GEMV for up to GEMV_ROWS columns,
        cuBLASLt for prompts."""
        if xq.shape[0] <= GEMV_ROWS:
            wide = self.n >= 16384
            return NV.nvfp4_mma(xq.view(torch.uint8), xs, self.w_u8, self.s, _unit_vec(xq.device), *self.tiling)
        return torch._scaled_mm(xq.view(torch.float4_e2m1fn_x2), self.w.t(), scale_a=xs, scale_b=self.s,
                                out_dtype=torch.bfloat16)

    @property
    def nbytes(self):
        return self.w.numel() + self.raw_scale_bytes


class Layer:
    pass


class Weights:
    def __init__(self, model_dir, device):
        self.device = device
        f = safe_open(os.path.join(model_dir, "model.safetensors"), "pt", device=f"cuda:{device.index or 0}")
        p = "model.language_model."

        def t(name):
            return f.get_tensor(p + name)

        # The embedding table (2.5 GB) stays in pinned host memory: a pass reads one row of it per token.
        with safe_open(os.path.join(model_dir, "model.safetensors"), "pt", device="cpu") as fc:
            self.embed = fc.get_tensor(p + "embed_tokens.weight").pin_memory()
        self.embed_ptr = K.HostPointer(self.embed)
        self.final_norm = t("norm.weight").float()
        self.lm_head = Fp8Linear([f.get_tensor("lm_head.weight")], [f.get_tensor("lm_head.weight_scale")])
        self.layers = []
        for i in range(LAYERS):
            L = Layer()
            q = f"layers.{i}."
            L.in_norm = t(q + "input_layernorm.weight").float()
            L.post_norm = t(q + "post_attention_layernorm.weight").float()
            if is_attention(i):
                a = q + "self_attn."
                L.qkv = Fp8Linear([t(a + n + "_proj.weight") for n in "qkv"],
                                  [t(a + n + "_proj.weight_scale") for n in "qkv"])
                L.o = Fp8Linear([t(a + "o_proj.weight")], [t(a + "o_proj.weight_scale")])
                L.q_norm = t(a + "q_norm.weight").float()
                L.k_norm = t(a + "k_norm.weight").float()
                L.k_scale = float(t(a + "k_scale").float())
                L.v_scale = float(t(a + "v_scale").float())
            else:
                a = q + "linear_attn."
                L.qkvz = Fp8Linear([t(a + "in_proj_qkv.weight"), t(a + "in_proj_z.weight")],
                                   [t(a + "in_proj_qkv.weight_scale"), t(a + "in_proj_z.weight_scale")])
                L.ab = torch.cat([t(a + "in_proj_a.weight"), t(a + "in_proj_b.weight")], 0).contiguous()
                L.conv_w = t(a + "conv1d.weight").reshape(CONV, 4).float().contiguous()
                L.A = -t(a + "A_log").float().exp()
                L.dt_bias = t(a + "dt_bias").float()
                L.gnorm = t(a + "norm.weight").float()
                L.out = Fp8Linear([t(a + "out_proj.weight")], [t(a + "out_proj.weight_scale")])
            m = q + "mlp."
            if i < FP8_MLP_FROM:
                gate, up, down = (m + n + "_proj." for n in ("gate", "up", "down"))
                L.gu = Nvfp4Linear([t(gate + "weight_packed"), t(up + "weight_packed")],
                                   [t(gate + "weight_scale"), t(up + "weight_scale")],
                                   [t(gate + "weight_global_scale").item(), t(up + "weight_global_scale").item()],
                                   t(gate + "input_global_scale").item())
                L.down = Nvfp4Linear([t(down + "weight_packed")], [t(down + "weight_scale")],
                                     [t(down + "weight_global_scale").item()], t(down + "input_global_scale").item())
                L.nvfp4 = True
            else:
                L.gu = Fp8Linear([t(m + "gate_proj.weight"), t(m + "up_proj.weight")],
                                 [t(m + "gate_proj.weight_scale"), t(m + "up_proj.weight_scale")])
                L.down = Fp8Linear([t(m + "down_proj.weight")], [t(m + "down_proj.weight_scale")])
                L.nvfp4 = False
            self.layers.append(L)
        torch.cuda.synchronize(device)

    def bytes_per_pass(self):
        """Weight bytes one decode pass reads: every matrix, norm and table row it touches once."""
        total = self.lm_head.nbytes + HIDDEN * 2 + self.final_norm.numel() * 2
        for L in self.layers:
            total += 2 * HIDDEN * 2
            if hasattr(L, "qkv"):
                total += L.qkv.nbytes + L.o.nbytes + 2 * HD * 2 + 8
            else:
                total += L.qkvz.nbytes + L.ab.numel() * 2 + CONV * 4 * 2 + L.out.nbytes + 3 * LV * 4
            total += L.gu.nbytes + L.down.nbytes
        return total


class Model:
    def __init__(self, model_dir, context_tokens, device="cuda", prefill_chunk=2048):
        self.device = torch.device(device)
        self.context = int(context_tokens)
        self.chunk = int(prefill_chunk)
        self.w = Weights(model_dir, self.device)
        dev = self.device
        n_attn = sum(1 for i in range(LAYERS) if is_attention(i))
        self.attn_index = {i: j for j, i in enumerate([i for i in range(LAYERS) if is_attention(i)])}
        self.gdn_index = {i: j for j, i in enumerate([i for i in range(LAYERS) if not is_attention(i)])}
        # FP8 KV cache for the whole context: [attention layer][position][kv head][dim].
        self.kc = torch.empty(n_attn, self.context, KVH, HD, dtype=torch.float8_e4m3fn, device=dev)
        self.vc = torch.empty(n_attn, self.context, KVH, HD, dtype=torch.float8_e4m3fn, device=dev)
        n_gdn = LAYERS - n_attn
        self.state = torch.zeros(n_gdn, LV, DK, DV, dtype=torch.float32, device=dev)
        self.conv = torch.zeros(n_gdn, CONV, 3, dtype=torch.float32, device=dev)
        inv = 1.0 / (THETA ** (torch.arange(0, ROT, 2, dtype=torch.float32, device=dev) / ROT))
        pos = torch.arange(self.context, dtype=torch.float32, device=dev)
        freqs = pos[:, None] * inv[None, :]
        self.cos, self.sin = freqs.cos().contiguous(), freqs.sin().contiguous()
        # Step workspaces: attention partials per fixed key block, and the recurrent checkpoints of a verify.
        self.max_splits = -(-self.context // KS.ATTN_CHUNK)
        self.attn_rows = 32
        self.part_o = torch.empty(KVH, self.max_splits, self.attn_rows, HD, dtype=torch.float32, device=dev)
        self.part_ml = torch.empty(KVH, self.max_splits, self.attn_rows, 2, dtype=torch.float32, device=dev)
        self.columns = 1
        self.pad_rows = 1            # decode and verify run cuBLASLt's GEMMs at this many rows (column invariance)
        self.conv_ckpt = None

    def reserve_columns(self, columns):
        """Conv window checkpoints for verifies of up to `columns` columns (all gated-delta layers at once), and the
        attention rows every step uses (the columns' query heads, a power of two, the same for 1 and for all columns)."""
        self.columns = max(self.columns, int(columns))
        rows = 1 << max(4, (self.columns * (HQ // KVH) - 1).bit_length())
        if rows != self.attn_rows:
            self.attn_rows = rows
            self.part_o = torch.empty(KVH, self.max_splits, rows, HD, dtype=torch.float32, device=self.device)
            self.part_ml = torch.empty(KVH, self.max_splits, rows, 2, dtype=torch.float32, device=self.device)
        n_gdn = self.state.shape[0]
        self.conv_ckpt = torch.empty(n_gdn, self.columns, CONV, 3, dtype=torch.float32, device=self.device)
        self.gdn_rows = [None] * n_gdn          # the last step's recurrence inputs per layer, for the commit
        self.cu = torch.zeros(2, dtype=torch.int64, device=self.device)

    # -- pieces ---------------------------------------------------------------------------------------------------

    def _embed(self, ids):
        T = ids.shape[0]
        h = torch.empty(T, HIDDEN, dtype=torch.float32, device=self.device)
        K.embed_gather_kernel[(T,)](self.w.embed_ptr, ids, h, K=HIDDEN, BLOCK=8192, num_warps=8)
        return h

    def _norm_quant(self, h, w, nvfp4=False, g=1.0, want_y=False, pending=None):
        """RMSNorm + quantization of h's rows; `pending` (delta bf16 [T, N], alpha [1]) is first added into h."""
        T = h.shape[0]
        if nvfp4:
            q = torch.empty(T, HIDDEN // 2, dtype=torch.uint8, device=self.device)
            s = K.scale_buffer(T, HIDDEN // 16, self.device)
        else:
            q = torch.empty(T, HIDDEN, dtype=torch.float8_e4m3fn, device=self.device)
            s = torch.empty(T, dtype=torch.float32, device=self.device)
        y = torch.empty(T, HIDDEN, dtype=torch.bfloat16, device=self.device) if want_y else q
        delta, alpha, dxs, dws = self._pending(h, pending)
        K.rmsnorm_quant_kernel[(T,)](h, w, EPS, g, q, s, -(-(HIDDEN // 16) // 4), y, delta, delta.stride(0), alpha,
                                     dxs, dws, K=HIDDEN, NVFP4=nvfp4, WRITE_Y=want_y, ADD=pending is not None,
                                     ROWCOL=dxs is not alpha, num_warps=8)
        return q, s, (y if want_y else None)

    def _padded(self, x):
        """x [T, ...] with zero rows up to pad_rows: cuBLASLt picks its kernel by the row count, so a column decoded
        alone and inside a verify take the same kernel and the same bits."""
        T = x.shape[0]
        if T >= self.pad_rows:
            return x
        return torch.cat((x, x.new_zeros(self.pad_rows - T, *x.shape[1:])), 0)

    def _pending(self, h, pending):
        """(delta, alpha, row scales, column scales) of a pending residual update; the scales are alpha itself when
        the delta is already scaled."""
        if pending is None:
            return h, self._one, self._one, self._one
        if len(pending) == 2:
            return pending[0], pending[1], pending[1], pending[1]
        return pending

    def _flush(self, h, pending):
        if pending is not None:
            delta, alpha, dxs, dws = self._pending(h, pending)
            K.residual_add_kernel[(h.shape[0],)](h, delta, delta.stride(0), alpha, dxs, dws, K=HIDDEN,
                                                 ROWCOL=dxs is not alpha, num_warps=8)

    def _quant(self, x):
        T, N = x.shape
        q = torch.empty(T, N, dtype=torch.float8_e4m3fn, device=self.device)
        s = torch.empty(T, dtype=torch.float32, device=self.device)
        K.fp8_quant_kernel[(T,)](x, q, s, K=N, BLOCK=max(1024, 1 << (N - 1).bit_length()), num_warps=8)
        return q, s

    def _mlp(self, L, h, pending):
        """The MLP of h after adding `pending`; returns its own pending (down output, alpha)."""
        T = h.shape[0]
        if L.nvfp4:
            xq, xs, _ = self._norm_quant(h, L.post_norm, nvfp4=True, g=L.gu.g, pending=pending)
            gu = L.gu(self._padded(xq), xs)[:T]
            dq = torch.empty(T, INTER // 2, dtype=torch.uint8, device=self.device)
            ds = K.scale_buffer(T, INTER // 16, self.device)
            K.silu_mul_nvfp4_kernel[(T, INTER // 1024)](gu, L.gu.alphas[0], L.gu.alphas[1], L.down.g, dq, ds,
                                                        -(-(INTER // 16) // 4), N=INTER, CHUNK=1024, num_warps=4)
            return L.down(self._padded(dq), ds)[:T], L.down.alphas[0]
        xq, xs, _ = self._norm_quant(h, L.post_norm, pending=pending)
        gu = L.gu.unscaled(xq)
        rowcol = gu is not None
        if not rowcol:
            gu = L.gu(xq, xs)
        dq = torch.empty(T, INTER, dtype=torch.float8_e4m3fn, device=self.device)
        ds = torch.empty(T, dtype=torch.float32, device=self.device)
        K.silu_mul_quant_kernel[(T,)](gu, self._one, self._one, 1.0, dq, ds, 0, xs, L.gu.s_flat, N=INTER, CHUNK=1024,
                                      NVFP4=False, ROWCOL=rowcol, num_warps=8)
        down = L.down.unscaled(dq)
        if down is None:
            return L.down(dq, ds), self._one
        return down, self._one, ds, L.down.s_flat

    @property
    def _one(self):
        if not hasattr(self, "_one_t"):
            self._one_t = torch.ones(1, dtype=torch.float32, device=self.device)
        return self._one_t

    def _attention_out(self, L, o):
        oq, os_ = self._quant(o.view(o.shape[0], HQ * HD))
        y = L.o.unscaled(oq)
        return (L.o(oq, os_), self._one) if y is None else (y, self._one, os_, L.o.s_flat)

    def _gdn_out(self, L, o, z, zscales=None):
        """zscales: (row scales, column scales) of z when it is an unscaled GEMM output."""
        T = o.shape[0]
        oq = torch.empty(T, LV * DV, dtype=torch.float8_e4m3fn, device=self.device)
        os_ = torch.empty(T, dtype=torch.float32, device=self.device)
        zxs, zws = zscales if zscales is not None else (os_, os_)
        K.gated_rmsnorm_fp8_kernel[(T,)](o, z, L.gnorm, EPS, oq, os_, z.stride(0), zxs, zws, H=LV, D=DV, HP=64,
                                         ROWCOL=zscales is not None, num_warps=8)
        y = L.out.unscaled(oq)
        return (L.out(oq, os_), self._one) if y is None else (y, self._one, os_, L.out.s_flat)

    def _rope_kv(self, L, j, qkv, pos, exact=False, scales=None):
        T = qkv.shape[0]
        dev = self.device
        q = torch.empty(T, HQ, HD, dtype=torch.bfloat16, device=dev)
        gate = torch.empty(T, HQ, HD, dtype=torch.bfloat16, device=dev)
        kx = torch.empty(T, KVH, HD, dtype=torch.bfloat16, device=dev) if exact else q
        vx = torch.empty(T, KVH, HD, dtype=torch.bfloat16, device=dev) if exact else q
        rxs, rws = scales if scales is not None else (self._one, self._one)
        KS.qk_rope_kv_kernel[(T, HQ + 2 * KVH)](qkv, qkv.stride(0), L.q_norm, L.k_norm, self.cos, self.sin, pos, q,
                                                gate, self.kc[j], self.vc[j], 1.0 / L.k_scale, 1.0 / L.v_scale, EPS,
                                                kx, vx, rxs, rws, HQ=HQ, KVH=KVH, D=HD, ROT=ROT, GATED=True,
                                                EXACT=exact, ROWCOL=scales is not None, num_warps=4)
        return q, gate, kx, vx

    def _recurrence(self, q, k, v, g, beta, state, out, T, store, cu=None):
        """flash-linear-attention's fused recurrent gated delta over T columns (q, k L2-normalized already): reads
        `state`, writes it back after the last column when `store`; with `cu` ([0, n] on the device) over n columns
        only."""
        st = state[None]
        fused_recurrent_gated_delta_rule_fwd_kernel[(DV // 8 * LV,)](
            q=q, k=k, v=v, g=g, gk=None, gv=None, beta=beta, A_log=None, dt_bias=None, o=out, h0=st,
            ht=st if store else None, cu_seqlens=cu, scale=DK ** -0.5, T=T, H=LK, HV=LV, K=DK, V=DV, BK=DK, BV=8,
            IS_BETA_HEADWISE=True, USE_QK_L2NORM_IN_KERNEL=False, APPLY_BETA_SIGMOID=False, ALLOW_NEG_EIGVAL=False,
            STATE_V_FIRST=False, num_warps=1, num_stages=3)

    # -- prefill --------------------------------------------------------------------------------------------------

    def reset(self):
        self.state.zero_()
        self.conv.zero_()

    def prefill_chunk(self, ids, p0, rows=None):
        """Run positions p0 .. p0 + T of `ids` (int64 [T] on the device) through every layer, updating the KV cache
        and recurrent states; returns the final hidden states (FP32 [T, 5120], before the final norm). p0 may be a
        device tensor [1]; `rows` (int64 [1] on the device) marks the first `rows` rows real and the rest padding,
        which the gated-delta state ignores (beta 0, decay 1) and causal attention never shows to real rows, so the
        chunk can run as a captured graph of a fixed row count."""
        T = ids.shape[0]
        dev = self.device
        h = self._embed(ids)
        pos = p0 if torch.is_tensor(p0) else torch.tensor([p0], dtype=torch.int64, device=dev)
        n = rows if rows is not None else torch.full((1,), T, dtype=torch.int64, device=dev)
        window_rows = n + torch.arange(3, device=dev)
        pending = None
        for i, L in enumerate(self.w.layers):
            if is_attention(i):
                j = self.attn_index[i]
                xq, xs, _ = self._norm_quant(h, L.in_norm, pending=pending)
                qkv = L.qkv.unscaled(xq)
                scales = (xs, L.qkv.s_flat) if qkv is not None else None
                if qkv is None:
                    qkv = L.qkv(xq, xs)
                q, gate, kx, vx = self._rope_kv(L, j, qkv, pos, exact=True, scales=scales)
                o = torch.empty(T, HQ, HD, dtype=torch.bfloat16, device=dev)
                K.attn_prefill_kernel[(-(-T // 64), HQ)](q, self.kc[j], self.vc[j], kx, vx, gate, o, pos, T,
                                                        HD ** -0.5, L.k_scale, L.v_scale, HQ=HQ, KVH=KVH, D=HD,
                                                        BM=64, BN=64, num_warps=8, num_stages=1)
                pending = self._attention_out(L, o)
            else:
                j = self.gdn_index[i]
                xq, xs, y = self._norm_quant(h, L.in_norm, want_y=True, pending=pending)
                qkvz = L.qkvz.unscaled(xq)
                rowcol = qkvz is not None
                if not rowcol:
                    qkvz = L.qkvz(xq, xs)
                ab = y @ L.ab.t()
                g = torch.empty(T, LV, dtype=torch.float32, device=dev)
                beta = torch.empty(T, LV, dtype=torch.float32, device=dev)
                K.gdn_gates_kernel[(T,)](ab, L.A, L.dt_bias, g, beta, n, LV=LV, num_warps=2)
                conv = torch.empty(T, CONV, dtype=torch.bfloat16, device=dev)
                ws = L.qkvz.s_flat
                K.conv_prefill_kernel[(CONV // 128, -(-T // 32))](qkvz, qkvz.stride(0), self.conv[j], L.conv_w, conv, T,
                                                                  xs, ws, C=CONV, BC=128, BT=32, ROWCOL=rowcol,
                                                                  num_warps=4)
                # the window after the real rows: rows n - 3 .. n - 1 of [old window | scaled inputs]
                src = (window_rows - 3).clamp(min=0)
                tail = qkvz.index_select(0, src)[:, :CONV].float()
                if rowcol:
                    tail = tail * xs.index_select(0, src)[:, None] * ws[None, :CONV]
                old = self.conv[j].t().index_select(0, window_rows.clamp(max=2))
                self.conv[j].copy_(torch.where((window_rows >= 3)[:, None], tail, old).t())
                qh = conv[:, : LK * DK].reshape(1, T, LK, DK)
                kh = conv[:, LK * DK: 2 * LK * DK].reshape(1, T, LK, DK)
                vh = conv[:, 2 * LK * DK:].reshape(1, T, LV, DV)
                o, st = chunk_gated_delta_rule(qh, kh, vh, g[None], beta[None], scale=DK ** -0.5,
                                               initial_state=self.state[j][None], output_final_state=True,
                                               use_qk_l2norm_in_kernel=True)
                self.state[j].copy_(st[0])
                pending = self._gdn_out(L, o.reshape(T, LV * DV), qkvz[:, CONV:],
                                        (xs, L.qkvz.s_flat[CONV:]) if rowcol else None)
            pending = self._mlp(L, h, pending)
        self._flush(h, pending)
        return h

    def logits(self, h, out_dtype=torch.float32):
        xq, xs, _ = self._norm_quant(h, self.w.final_norm)
        return self.w.lm_head(xq, xs, out_dtype=out_dtype)

    # -- decode and verify ------------------------------------------------------------------------------------

    def step(self, tokens, pos):
        """Columns `tokens` (int64 [TQ] on the device) at positions pos .. pos + TQ - 1 (`pos` int64 [1] on the device)
        through every layer: KV written for every column. With one column the gated-delta state advances in place;
        with more it stays at the round's start and commit() advances it over the standing columns. Returns (logits
        FP32 [TQ, VOCAB], final normed hidden bf16 [TQ, HIDDEN]). Every launch reads its lengths from the device."""
        TQ = tokens.shape[0]
        dev = self.device
        h = self._embed(tokens)
        pending = None
        for i, L in enumerate(self.w.layers):
            if is_attention(i):
                j = self.attn_index[i]
                xq, xs, _ = self._norm_quant(h, L.in_norm, pending=pending)
                qkv = L.qkv(xq, xs)
                q, gate, _, _ = self._rope_kv(L, j, qkv, pos)
                o = self.attend(q, gate, self.kc[j], self.vc[j], pos, L.k_scale, L.v_scale)
                pending = self._attention_out(L, o)
            else:
                j = self.gdn_index[i]
                xq, xs, y = self._norm_quant(h, L.in_norm, want_y=True, pending=pending)
                qkvz = L.qkvz(xq, xs)
                ab = GV.bf16_gemv(y, L.ab)
                qh = torch.empty(TQ, LK, DK, dtype=torch.float32, device=dev)
                kh = torch.empty(TQ, LK, DK, dtype=torch.float32, device=dev)
                vh = torch.empty(TQ, LV, DV, dtype=torch.float32, device=dev)
                g = torch.empty(TQ, LV, dtype=torch.float32, device=dev)
                beta = torch.empty(TQ, LV, dtype=torch.float32, device=dev)
                KS.conv_gdn_kernel[(2 * LK + LV + 1,)](qkvz, qkvz.stride(0), self.conv[j], L.conv_w, qh, kh, vh,
                                                       self.conv_ckpt[j], ab, L.A, L.dt_bias, g, beta, TQ=TQ, LK=LK,
                                                       LV=LV, DK=DK, CONV=CONV, num_warps=4)
                o = torch.empty(TQ, LV, DV, dtype=torch.float32, device=dev)
                self._recurrence(qh, kh, vh, g, beta, self.state[j], o, TQ, store=TQ == 1)
                self.gdn_rows[j] = (qh, kh, vh, g, beta, o)
                pending = self._gdn_out(L, o.view(TQ, LV * DV), qkvz[:, CONV:])
            pending = self._mlp(L, h, pending)
        xq, xs, hn = self._norm_quant(h, self.w.final_norm, want_y=True, pending=pending)
        return self.w.lm_head(xq, xs, out_dtype=torch.float32), hn

    def attend(self, q, gate, kc, vc, pos, k_scale, v_scale):
        TQ = q.shape[0]
        o = torch.empty(TQ, HQ, HD, dtype=torch.bfloat16, device=self.device)
        KS.attn_step_kernel[(KVH, 64)](q, kc, vc, pos, self.part_o, self.part_ml, HD ** -0.5, k_scale,
                                       MAXS=self.max_splits, HQ=HQ, KVH=KVH, D=HD, G=HQ // KVH, TQ=TQ,
                                       R=self.attn_rows, CH=KS.ATTN_CHUNK, BN=64, NPROG=64,
                                       num_warps=4 if self.attn_rows <= 32 else 8, num_stages=2)
        KS.attn_step_reduce_kernel[(TQ, HQ)](self.part_o, self.part_ml, gate, o, pos, v_scale, MAXS=self.max_splits,
                                             HQ=HQ, D=HD, G=HQ // KVH, R=self.attn_rows, CH=KS.ATTN_CHUNK, SB=64,
                                             num_warps=4)
        return o

    def commit(self, keep, columns):
        """After a step of `columns` columns, keep the state after column `keep` (int64 [1] on the device): the conv
        window from its checkpoint, the gated-delta state by running the recurrence again over columns 0 .. keep from
        the round's start (in place; one-column steps already advanced it)."""
        self.conv.copy_(self.conv_ckpt.index_select(1, keep)[:, 0])
        if columns == 1:
            return
        self.cu[1:].copy_(keep + 1)
        for j, (qh, kh, vh, g, beta, o) in enumerate(self.gdn_rows):
            self._recurrence(qh, kh, vh, g, beta, self.state[j], o, columns, store=True, cu=self.cu)
