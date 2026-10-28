"""Qwen3.8-27B forward passes: chunked prefill (bf16 GEMMs on exact bf16 weights, fla's chunked gated delta rule,
flash attention over the FP8 cache) and a graph-capturable decode step of up to 16 tokens (Triton GEMVs on the stored
encodings, fla's recurrent gated delta rule, split-key flash decoding over the FP8 cache)."""
import math

import torch
import torch.nn.functional as F
import triton

from . import kernels as K
from . import spec_kernels as SK
from .vendor import fla_ops

HIDDEN = 5120
INTER = 17408
HQ, HK, HD = 24, 4, 256
ROT = 64
LK, LH, LHV, LV = 128, 16, 48, 128     # gated delta net: key dim, key heads, value heads, value dim
QK = LK * LH                           # 2048
CONV = 2 * QK + LV * LHV               # 10240
VDIM = LV * LHV                        # 6144
VOCAB = 248320
EPS = 1e-6


ATTN_SEGMENT = 65536    # keys widened per flash call in prefill (bounded memory at any context)
MLP_ROWS = 1024         # prompt rows per MLP block of the drafter prefill (bounded intermediate)


def segmented_attention(q, kc, vc, pos0, sm_scale, kscale=None, vscale=None):
    """Prefill attention of q [T, HQ, D] bf16 at positions pos0.. against the FP8 caches kc, vc [HK, S, D] holding
    positions 0 .. pos0 + T - 1: flash attention per key segment of ATTN_SEGMENT positions (the chunk's own block
    causal), merged by log-sum-exp, so no more than one segment is widened at a time. Per-tensor scales are folded in
    by the caller (sm_scale includes the key scale; the output is unscaled by the value scale); per-position scales
    kscale, vscale [HK, S] (the draft cache) widen each segment. Returns fp32 [T, HQ, D]."""
    T = q.shape[0]
    qq = q.transpose(0, 1).unsqueeze(0)
    bounds = list(range(0, pos0, ATTN_SEGMENT)) + [pos0]
    pieces = [(bounds[i], bounds[i + 1], False) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]
    pieces.append((pos0, pos0 + T, True))
    out = None
    for s0, s1, causal in pieces:
        k = kc[:, s0:s1].to(torch.bfloat16)
        v = vc[:, s0:s1].to(torch.bfloat16)
        if kscale is not None:
            k = k * kscale[:, s0:s1, None].to(torch.bfloat16)
            v = v * vscale[:, s0:s1, None].to(torch.bfloat16)
        k = k.unsqueeze(0)       # the flash kernel takes the 4 key/value heads of the 24 query heads as they are
        v = v.unsqueeze(0)
        o, lse = torch.ops.aten._scaled_dot_product_flash_attention(qq, k, v, 0.0, causal, False, scale=sm_scale)[:2]
        del k, v
        o = o[0].float()                                       # [HQ, T, D]
        lse = lse[0]                                           # [HQ, T]
        if out is None:
            out, m = o, lse
            continue
        mn = torch.maximum(m, lse)
        a, b = torch.exp(m - mn), torch.exp(lse - mn)
        out = (out * a[..., None] + o * b[..., None]) / (a + b)[..., None]
        m = mn + torch.log(a + b)
    return out.transpose(0, 1)                                  # [T, HQ, D]


def _mm_f16acc(x, w):
    """x [M, K] f16 @ w [N, K]^T in f16 with f16 accumulation (cuBLAS CUBLAS_COMPUTE_16F), f16 out."""
    flag = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    try:
        return torch.mm(x, w.t())
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = flag


class Model:
    def __init__(self, weights, context_tokens, chunk=4096, max_rows=K.MAX_ROWS, fp8_prompt=None, fp16_mlp=False):
        self.w = weights
        self.fp8_prompt = dict(fp8_prompt or {})     # fp8 matrix group -> prompt GEMMs with FP8 activations
        # NVFP4 MLP prompt GEMMs above the Triton rows as f16 with f16 accumulation (cuBLAS CUBLAS_COMPUTE_16F)
        self.fp16_mlp = bool(fp16_mlp)
        self.mlp_rows = MLP_ROWS          # prompt rows per MLP block (set by the engine)
        self.conv_kernel = True           # the Triton prefill conv (False: torch.cat + conv1d)
        dev = weights.device
        self.dev = dev
        self.context = int(context_tokens)
        self.chunk = chunk
        self.attn_layers = [i for i, k in enumerate(weights.layer_types) if k == "full_attention"]
        self.gdn_layers = [i for i, k in enumerate(weights.layer_types) if k == "linear_attention"]
        self.attn_slot = {l: j for j, l in enumerate(self.attn_layers)}
        self.gdn_slot = {l: j for j, l in enumerate(self.gdn_layers)}
        A = len(self.attn_layers)
        # FP8 KV cache [layer, kv head, position, dim]
        # positions beyond the context hold a last speculative round's rejected columns, never read as context
        self.capacity = self.context + K.MAX_ROWS
        self.kcache = torch.empty(A, HK, self.capacity, HD, dtype=torch.float8_e4m3fn, device=dev)
        self.vcache = torch.empty(A, HK, self.capacity, HD, dtype=torch.float8_e4m3fn, device=dev)
        G = len(self.gdn_layers)
        self.rec = torch.zeros(G, LHV, LK, LV, dtype=torch.float32, device=dev)
        self.conv = torch.zeros(G, CONV, 3, dtype=torch.float32, device=dev)
        inv = 1.0 / (10000000.0 ** (torch.arange(0, ROT, 2, dtype=torch.float32) / ROT))
        self.inv_freq = inv.to(dev)
        self.sm_scale = HD ** -0.5
        # decode buffers (max_rows tokens)
        R = max_rows
        f32, bf16 = torch.float32, torch.bfloat16
        self.tokens = torch.zeros(R, dtype=torch.int32, device=dev)
        self.pos = torch.zeros(1, dtype=torch.int32, device=dev)
        self.h = torch.zeros(R, HIDDEN, dtype=f32, device=dev)
        self.xn = torch.zeros(R, HIDDEN, dtype=bf16, device=dev)
        self.qkvz = torch.zeros(R, CONV + VDIM, dtype=f32, device=dev)
        self.bbuf = torch.zeros(R, LHV, dtype=f32, device=dev)
        self.abuf = torch.zeros(R, LHV, dtype=f32, device=dev)
        self.convout = torch.zeros(R * CONV, dtype=f32, device=dev)
        self.core = torch.zeros(R, LHV, LV, dtype=f32, device=dev)
        self.y = torch.zeros(R, VDIM, dtype=bf16, device=dev)
        self.gu = torch.zeros(R, 2 * INTER, dtype=f32, device=dev)
        self.act = torch.zeros(R, INTER, dtype=bf16, device=dev)
        self.qkv = torch.zeros(R, HQ * 2 * HD + 2 * HK * HD, dtype=f32, device=dev)
        self.q = torch.zeros(R, HQ, HD, dtype=bf16, device=dev)
        self.gate = torch.zeros(R, HQ, HD, dtype=f32, device=dev)
        self.attn_out = torch.zeros(R, HQ * HD, dtype=bf16, device=dev)
        self.logits = torch.zeros(R, VOCAB, dtype=f32, device=dev)
        self.attn_work = K.attn_work(R, HK, HD, dev)
        biggest = max(lin.split * R * lin.n for layer in weights.layers for lin in layer.values()
                      if isinstance(lin, K.Linear))
        self.partial = torch.empty(max(biggest, weights.lm_head.split * R * weights.lm_head.n), dtype=f32, device=dev)
        self.ba_lin = {}
        for l in self.gdn_layers:
            ba = weights.layers[l]["ba"]
            ones = torch.ones(LHV, dtype=f32, device=dev)
            self.ba_lin[l] = (K.Linear("fp8", ba[:LHV].contiguous(), ones), K.Linear("fp8", ba[LHV:].contiguous(), ones))
        # verify passes keep each gated-delta layer's step inputs for the round's commit
        self.v_qkvz = torch.zeros(G, R, CONV + VDIM, dtype=f32, device=dev)
        self.v_conv = torch.zeros(G, R * CONV, dtype=f32, device=dev)
        self.v_a = torch.zeros(G, R, LHV, dtype=f32, device=dev)
        self.v_b = torch.zeros(G, R, LHV, dtype=f32, device=dev)
        # prefill: one bf16 weight buffer for the largest matrix
        largest = max(lin.n * lin.k for layer in weights.layers for lin in layer.values() if isinstance(lin, K.Linear))
        self.wbuf = torch.empty(largest, dtype=bf16, device=dev)
        self.ones = torch.ones(2 * INTER, dtype=f32, device=dev)
        self.mtp = None
        if weights.mtp is not None:
            self.mtp = MTP(self, weights.mtp)

    # ------------------------------------------------------------------------------------------- state
    def reset(self):
        self.rec.zero_()
        self.conv.zero_()

    # ------------------------------------------------------------------------------------------- decode
    def decode(self, T, verify=False):
        """One pass over self.tokens[:T] at positions self.pos .. self.pos + T - 1 -> self.logits[:T] (fp32). With
        verify the gated-delta states stay as they were and the step inputs are kept for commit(count)."""
        w = self.w
        h, xn = self.h[:T], self.xn[:T]
        K.embed(w.embed, self.tokens[:T], h)
        for l, layer in enumerate(w.layers):
            K.rmsnorm(h, layer["in_norm"], xn)
            if layer["kind"] == "linear_attention":
                self._gdn_decode(l, layer, T, verify)
            else:
                self._attn_decode(l, layer, T)
            K.rmsnorm(h, layer["post_norm"], xn)
            K.gemv(layer["gate_up"], xn, out=self.gu[:T], partial=self.partial)
            K.silu_mul(self.gu[:T], self.act[:T])
            K.gemv(layer["down"], self.act[:T], residual=h, partial=self.partial)
        K.rmsnorm(h, w.final_norm, xn)
        K.gemv(w.lm_head, xn, out=self.logits[:T], partial=self.partial)
        return self.logits[:T]

    def _gdn_decode(self, l, layer, T, verify=False):
        s = self.gdn_slot[l]
        xn = self.xn[:T]
        if verify:   # per-layer step inputs, kept for the round's commit
            qkvz, bb, ab = self.v_qkvz[s, :T], self.v_b[s, :T], self.v_a[s, :T]
            co = self.v_conv[s, : T * CONV]
        else:
            qkvz, bb, ab = self.qkvz[:T], self.bbuf[:T], self.abuf[:T]
            co = self.convout[: T * CONV]
        K.gemv(layer["qkvz"], xn, out=qkvz, partial=self.partial)
        blin, alin = self.ba_lin[l]
        K.gemv(blin, xn, out=bb)
        K.gemv(alin, xn, out=ab)
        if verify:
            SK.conv_verify(qkvz, layer["conv"], self.conv[s], co, sections=(QK, QK, VDIM))
        else:
            K.conv_update(qkvz, layer["conv"], self.conv[s], co, sections=(QK, QK, VDIM))
        q = co[: T * QK].view(T, LH, LK)
        k = co[T * QK: 2 * T * QK].view(T, LH, LK)
        v = co[2 * T * QK:].view(T, LHV, LV)
        SK.gdn_recurrent(q, k, v, ab, bb, layer["A_log"], layer["dt_bias"], self.rec[s], self.core[:T],
                         LK ** -0.5, store=not verify)
        K.gated_norm(self.core[:T], qkvz[:, CONV:], layer["norm"], self.y[:T])
        K.gemv(layer["out"], self.y[:T], residual=self.h[:T], partial=self.partial)

    def commit(self, T, count):
        """After a verify pass of T tokens, advance every gated-delta state by the first count (device int32 [1])."""
        for l in self.gdn_layers:
            s = self.gdn_slot[l]
            layer = self.w.layers[l]
            co = self.v_conv[s, : T * CONV]
            SK.conv_commit(self.v_qkvz[s, :T], self.conv[s], count)
            SK.gdn_recurrent(co[: T * QK].view(T, LH, LK), co[T * QK: 2 * T * QK].view(T, LH, LK),
                             co[2 * T * QK:].view(T, LHV, LV), self.v_a[s, :T], self.v_b[s, :T], layer["A_log"],
                             layer["dt_bias"], self.rec[s], self.core[:T], LK ** -0.5, store=True, count=count,
                             output=False)

    def _attn_decode(self, l, layer, T):
        s = self.attn_slot[l]
        K.gemv(layer["qkv"], self.xn[:T], out=self.qkv[:T], partial=self.partial)
        kc, vc = self.kcache[s], self.vcache[s]
        K.attn_prep(self.qkv[:T], layer["q_norm"], layer["k_norm"], self.inv_freq, self.pos, self.q[:T],
                    self.gate[:T], kc, vc, layer["k_scale"], layer["v_scale"])
        K.attn_decode(self.q[:T], kc, vc, self.pos, self.gate[:T], self.attn_out[:T],
                      self.sm_scale * layer["k_scale"], layer["v_scale"], self.attn_work)
        K.gemv(layer["o"], self.attn_out[:T], residual=self.h[:T], partial=self.partial)

    # ------------------------------------------------------------------------------------------- prefill
    def _mm(self, lin, x, residual=None, scale=True):
        """Prefill projection of x [T, K] bf16 -> fp32 [T, N] (or residual += it) over the stored weights."""
        if lin.kind == "fp8" and self.fp8_prompt.get(lin.group):
            return K.fp8_mm(lin, x, residual=residual, scale=scale, ones=self.ones)
        return K.stored_mm(lin, x, residual=residual, scale=scale, wbuf=self.wbuf)

    def prefill(self, ids, pos0, hidden_out=None):
        """Runs tokens `ids` (device int32 [T]) at positions pos0.. through every layer, updating caches and states.
        Returns the final-normed hidden rows [T, HIDDEN] as bf16."""
        w = self.w
        T = ids.shape[0]
        h = torch.empty(T, HIDDEN, dtype=torch.float32, device=self.dev)
        K.embed(w.embed, ids, h)
        xn = torch.empty(T, HIDDEN, dtype=torch.bfloat16, device=self.dev)
        pos = torch.tensor([pos0], dtype=torch.int32, device=self.dev)
        for l, layer in enumerate(w.layers):
            K.rmsnorm(h, layer["in_norm"], xn)
            if layer["kind"] == "linear_attention":
                self._gdn_prefill(l, layer, xn, h, T)
            else:
                self._attn_prefill(l, layer, xn, h, T, pos0, pos)
            K.rmsnorm(h, layer["post_norm"], xn)
            gl = layer["gate_up"]
            act = torch.empty(T, INTER, dtype=torch.bfloat16, device=self.dev)
            wide = gl.kind == "nvfp4" and T > K.NVFP4_TRITON_ROWS
            half = wide and self.fp16_mlp
            if wide:     # widened once, then used by every row block
                wt = self.wbuf[: gl.n * gl.k].view(gl.n, gl.k)
                if half:
                    wt = wt.view(torch.float16)
                K.widen_bf16(gl, wt)
            for r0 in range(0, T, self.mlp_rows):
                r1 = min(T, r0 + self.mlp_rows)
                if wide:
                    if half:
                        gu = _mm_f16acc(xn[r0:r1].to(torch.float16), wt)
                    else:
                        gu = torch.mm(xn[r0:r1], wt.t(), out_dtype=torch.float32)
                else:
                    gu = self._mm(gl, xn[r0:r1], scale=False)
                K.silu_mul(gu, act[r0:r1], scale=gl.rowscale)
                del gu
            if half:
                dl = layer["down"]
                wd = self.wbuf[: dl.n * dl.k].view(dl.n, dl.k).view(torch.float16)
                K.widen_bf16(dl, wd)
                h.add_(_mm_f16acc(act.to(torch.float16), wd), alpha=dl.uniform)
            else:
                self._mm(layer["down"], act, residual=h)
            del act
        out = hidden_out if hidden_out is not None else xn
        K.rmsnorm(h, w.final_norm, out)
        return out

    def _gdn_prefill(self, l, layer, xn, h, T):
        s = self.gdn_slot[l]
        qkvz = self._mm(layer["qkvz"], xn)
        blin, alin = self.ba_lin[l]
        b = torch.mm(xn, blin.weight.t(), out_dtype=torch.float32)
        a = torch.mm(xn, alin.weight.t(), out_dtype=torch.float32)
        state = self.conv[s]                                                    # [C, 3]
        if self.conv_kernel:
            y = torch.empty(T * CONV, dtype=torch.float32, device=self.dev)
            K.conv_prefill(qkvz, layer["conv"], state, y, sections=(QK, QK, VDIM))
            if T >= 3:
                state.copy_(qkvz[T - 3:, :CONV].t())
            else:
                state.copy_(torch.cat([state, qkvz[:, :CONV].t()], dim=1)[:, -3:])
            q = y[: T * QK].view(1, T, LH, LK)
            k = y[T * QK: 2 * T * QK].view(1, T, LH, LK)
            v = y[2 * T * QK:].view(1, T, LHV, LV)
        else:
            x = qkvz[:, :CONV].t().unsqueeze(0)
            xin = torch.cat([state.unsqueeze(0), x], dim=2)
            state.copy_(xin[0, :, -3:])
            y = F.silu(F.conv1d(xin, layer["conv"].unsqueeze(1), groups=CONV))[0].t()
            q = y[:, :QK].reshape(1, T, LH, LK).contiguous()
            k = y[:, QK:2 * QK].reshape(1, T, LH, LK).contiguous()
            v = y[:, 2 * QK:].reshape(1, T, LHV, LV).contiguous()
        g = -layer["A_log"].exp() * F.softplus(a + layer["dt_bias"])
        beta = torch.sigmoid(b)
        o, final = fla_ops.chunk(q, k, v, g.unsqueeze(0), beta.unsqueeze(0), LK ** -0.5, self.rec[s:s + 1])
        self.rec[s:s + 1].copy_(final)
        yb = torch.empty(T, VDIM, dtype=torch.bfloat16, device=self.dev)
        K.gated_norm(o.view(T, LHV, LV), qkvz[:, CONV:], layer["norm"], yb)
        del qkvz, o
        self._mm(layer["out"], yb, residual=h)

    def _attn_prefill(self, l, layer, xn, h, T, pos0, pos):
        s = self.attn_slot[l]
        qkv = self._mm(layer["qkv"], xn)
        q = torch.empty(T, HQ, HD, dtype=torch.bfloat16, device=self.dev)
        gate = torch.empty(T, HQ, HD, dtype=torch.float32, device=self.dev)
        kc, vc = self.kcache[s], self.vcache[s]
        K.attn_prep(qkv, layer["q_norm"], layer["k_norm"], self.inv_freq, pos, q, gate, kc, vc,
                    layer["k_scale"], layer["v_scale"])
        del qkv
        o = segmented_attention(q, kc, vc, pos0, self.sm_scale * layer["k_scale"])
        del q
        o = o * layer["v_scale"]
        o = (o * torch.sigmoid(gate)).to(torch.bfloat16).reshape(T, HQ * HD)
        del gate
        self._mm(layer["o"], o, residual=h)

    # ------------------------------------------------------------------------------------------- head
    def head_rows(self, xn):
        """fp32 logits of a few rows (<= 16) through the decode head."""
        T = xn.shape[0]
        out = self.logits[:T]
        K.gemv(self.w.lm_head, xn, out=out, partial=self.partial)
        return out

    def score_rows(self, xn, targets, top_k, queries=None, block=31040):
        """Log-probabilities of targets[i] under row i of xn [T, HIDDEN] bf16 through the whole head, and each row's
        top_k (ids, logprobs) ranked, by vocabulary blocks of exact bf16 head weights."""
        T = xn.shape[0]
        lin = self.w.lm_head
        dev = self.dev
        run_max = torch.full((T,), -float("inf"), device=dev)
        run_sum = torch.zeros(T, device=dev)
        tgt = torch.full((T,), -float("inf"), device=dev)
        cand_v, cand_i = None, None
        qv = None
        if queries is not None:
            qv = torch.full(queries.shape, -float("inf"), device=dev)
        for v0 in range(0, VOCAB, block):
            v1 = min(VOCAB, v0 + block)
            wt = self.wbuf[: (v1 - v0) * HIDDEN].view(v1 - v0, HIDDEN)
            wt.copy_(lin.weight[v0:v1])
            for r0 in range(0, T, 2048):
                r1 = min(T, r0 + 2048)
                lg = torch.mm(xn[r0:r1], wt.t(), out_dtype=torch.float32)
                lg.mul_(lin.rowscale[v0:v1])
                bm = lg.max(dim=1).values
                nm = torch.maximum(run_max[r0:r1], bm)
                run_sum[r0:r1] = run_sum[r0:r1] * torch.exp(run_max[r0:r1] - nm) + \
                    torch.exp(lg - nm[:, None]).sum(dim=1)
                run_max[r0:r1] = nm
                t = targets[r0:r1]
                inside = (t >= v0) & (t < v1)
                local = (t - v0).clamp(0, v1 - v0 - 1)
                picked = lg.gather(1, local[:, None].long())[:, 0]
                tgt[r0:r1] = torch.where(inside, picked, tgt[r0:r1])
                if qv is not None:
                    qq = queries[r0:r1]
                    qin = (qq >= v0) & (qq < v1)
                    ql = (qq - v0).clamp(0, v1 - v0 - 1)
                    qv[r0:r1] = torch.where(qin, lg.gather(1, ql.long()), qv[r0:r1])
                if top_k:
                    bv, bi = torch.topk(lg, top_k, dim=1)
                    bi = bi + v0
                    if cand_v is None:
                        cand_v = torch.full((T, top_k), -float("inf"), device=dev)
                        cand_i = torch.zeros((T, top_k), dtype=torch.long, device=dev)
                    mv = torch.cat([cand_v[r0:r1], bv], dim=1)
                    mi = torch.cat([cand_i[r0:r1], bi], dim=1)
                    # rank by value descending, then id ascending
                    order = torch.argsort(mi, dim=1, stable=True)
                    mv, mi = mv.gather(1, order), mi.gather(1, order)
                    order = torch.argsort(mv, dim=1, descending=True, stable=True)[:, :top_k]
                    cand_v[r0:r1], cand_i[r0:r1] = mv.gather(1, order), mi.gather(1, order)
                del lg
        lse = run_max + torch.log(run_sum)
        out = {"target": tgt - lse}
        if top_k:
            out["top_v"], out["top_i"] = cand_v - lse[:, None], cand_i
        if qv is not None:
            out["query"] = qv - lse[:, None]
        return out


class MTP:
    """The checkpoint draft layer (multi-token prediction): fc([norm(embed(next token)) | norm(hidden)]) -> one
    gated-attention decoder layer with its own KV cache (FP8 scaled per position and head) -> norm, sharing the
    target embedding and head. Its normed output is the hidden of the next draft step."""

    def __init__(self, model, mw):
        self.model = model
        self.w = mw
        dev = model.dev
        R = K.MAX_ROWS
        f32, bf16 = torch.float32, torch.bfloat16
        S = model.capacity
        self.kcache = torch.empty(HK, S, HD, dtype=torch.float8_e4m3fn, device=dev)
        self.vcache = torch.empty(HK, S, HD, dtype=torch.float8_e4m3fn, device=dev)
        self.kscale = torch.zeros(HK, S, dtype=f32, device=dev)
        self.vscale = torch.zeros(HK, S, dtype=f32, device=dev)
        self.e = torch.zeros(R, HIDDEN, dtype=f32, device=dev)
        self.cat = torch.zeros(R, 2 * HIDDEN, dtype=bf16, device=dev)
        self.r = torch.zeros(R, HIDDEN, dtype=f32, device=dev)
        self.xn = torch.zeros(R, HIDDEN, dtype=bf16, device=dev)
        self.qkv = torch.zeros(R, HQ * 2 * HD + 2 * HK * HD, dtype=f32, device=dev)
        self.q = torch.zeros(R, HQ, HD, dtype=bf16, device=dev)
        self.gate = torch.zeros(R, HQ, HD, dtype=f32, device=dev)
        self.attn = torch.zeros(R, HQ * HD, dtype=bf16, device=dev)
        self.gu = torch.zeros(R, 2 * INTER, dtype=f32, device=dev)
        self.act = torch.zeros(R, INTER, dtype=bf16, device=dev)
        self.out = torch.zeros(R, HIDDEN, dtype=bf16, device=dev)
        self.work = K.attn_work(R, HK, HD, dev)
        self.partial = torch.empty(max(lin.split * R * lin.n for lin in mw.values() if isinstance(lin, K.Linear)),
                                   dtype=f32, device=dev)

    def bytes_per_step(self):
        total = 0
        for value in self.w.values():
            if isinstance(value, K.Linear):
                total += value.bytes()
            elif isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
        return total

    def step(self, R, tokens, hidden, pos):
        """R rows: tokens [R] int32 (device), hidden [R, HIDDEN] (target or previous draft output, normed), at positions
        pos .. pos + R - 1 (device int32 [1]) -> self.out[:R] (normed, bf16)."""
        w, m = self.w, self.model
        e, cat, r, xn = self.e[:R], self.cat[:R], self.r[:R], self.xn[:R]
        K.embed(m.w.embed, tokens, e)
        K.rmsnorm(e, w["norm_embed"], cat[:, :HIDDEN])
        K.rmsnorm(hidden, w["norm_hidden"], cat[:, HIDDEN:])
        K.gemv(w["fc"], cat, out=r, partial=self.partial)
        K.rmsnorm(r, w["in_norm"], xn)
        K.gemv(w["qkv"], xn, out=self.qkv[:R], partial=self.partial)
        SK.mtp_prep(self.qkv[:R], w["q_norm"], w["k_norm"], m.inv_freq, pos, self.q[:R], self.gate[:R], self.kcache,
                    self.vcache, self.kscale, self.vscale)
        SK.mtp_attn(self.q[:R], self.kcache, self.vcache, self.kscale, self.vscale, pos, self.gate[:R],
                    self.attn[:R], m.sm_scale, self.work)
        K.gemv(w["o"], self.attn[:R], residual=r, partial=self.partial)
        K.rmsnorm(r, w["post_norm"], xn)
        K.gemv(w["gate_up"], xn, out=self.gu[:R], partial=self.partial)
        K.silu_mul(self.gu[:R], self.act[:R])
        K.gemv(w["down"], self.act[:R], residual=r, partial=self.partial)
        K.rmsnorm(r, w["norm"], self.out[:R])
        return self.out[:R]

    def prefill(self, tokens, hidden, pos0):
        """The draft layer over prompt rows: tokens [T] (each row next token), hidden [T, HIDDEN] bf16 (the target final
        normed rows) at positions pos0.. : fills the draft KV cache (no logits)."""
        w, m = self.w, self.model
        T = tokens.shape[0]
        dev = m.dev
        e = torch.empty(T, HIDDEN, dtype=torch.float32, device=dev)
        K.embed(m.w.embed, tokens, e)
        cat = torch.empty(T, 2 * HIDDEN, dtype=torch.bfloat16, device=dev)
        K.rmsnorm(e, w["norm_embed"], cat[:, :HIDDEN])
        K.rmsnorm(hidden, w["norm_hidden"], cat[:, HIDDEN:])
        del e
        r = K.stored_mm(w["fc"], cat)
        del cat
        xn = torch.empty(T, HIDDEN, dtype=torch.bfloat16, device=dev)
        K.rmsnorm(r, w["in_norm"], xn)
        qkv = K.stored_mm(w["qkv"], xn)
        q = torch.empty(T, HQ, HD, dtype=torch.bfloat16, device=dev)
        gate = torch.empty(T, HQ, HD, dtype=torch.float32, device=dev)
        pos = torch.tensor([pos0], dtype=torch.int32, device=dev)
        SK.mtp_prep(qkv, w["q_norm"], w["k_norm"], m.inv_freq, pos, q, gate, self.kcache, self.vcache, self.kscale,
                    self.vscale)
        # A prompt row's draft keys and values depend only on the layer input and no prompt row needs a draft, so
        # the draft layer's attention, output projection and MLP are not run here.
