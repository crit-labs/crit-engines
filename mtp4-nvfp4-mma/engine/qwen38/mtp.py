"""The checkpoint's MTP layer as the drafter (Qwen3-Next layout): fc over concat(norm_e(embed(next token)),
norm_h(hidden)), one gated full-attention decoder layer with a dense MLP, mtp.norm, and an lm_head. Draft-only
weights: the stored bf16 MTP matrices and the stored FP8 lm_head re-encoded at load to NVFP4 draft copies (group-16
e4m3 scales, one global scale per matrix), multiplied by the engine's tensor-core NVFP4 GEMV with the activations
quantized at a per-row dynamic global scale. Never trained; the target verifies every draft with its own stored
weights. Its KV cache is FP8 at scale 1 for the whole context.

Position i's entry reads (the target's final normed hidden at i, the token at i + 1) at rope position i, and its
output predicts the token at i + 2. Prompt positions only need their keys and values (inference/mtp-decode: a
one-layer drafter's K/V at a position depend on that position's input alone)."""
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

from . import kernels as K
from . import kernels_step as KS
from . import nvfp4 as NV
from .model import EPS, GEMV_ROWS, HD, HQ, KVH, ROT

PREFIX = "mtp."
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _encode_nvfp4(rows_fn, N, Kd, device, chunk=8192):
    """NVFP4 draft copy of a matrix whose FP32 rows rows_fn(r0, r1) returns: (packed uint8 [N, K/2], swizzled e4m3
    scales, 1 / global scale). The global scale puts the matrix's largest group scale at e4m3's 448; codes round to
    nearest with ties to even, as the checkpoint's own NVFP4 matrices."""
    amax = 0.0
    for r0 in range(0, N, chunk):
        amax = max(amax, float(rows_fn(r0, min(N, r0 + chunk)).abs().max()))
    gw = 2688.0 / max(amax, 1e-30)
    grid = torch.tensor(E2M1, device=device)
    mids = (grid[1:] + grid[:-1]) / 2
    packed = torch.empty(N, Kd // 2, dtype=torch.uint8, device=device)
    raw = torch.empty(N, Kd // 16, dtype=torch.float8_e4m3fn, device=device)
    for r0 in range(0, N, chunk):
        r1 = min(N, r0 + chunk)
        b = rows_fn(r0, r1).reshape(r1 - r0, Kd // 16, 16)
        stored = (b.abs().amax(-1, keepdim=True) / 6.0 * gw).clamp(max=448.0).to(torch.float8_e4m3fn)
        step = stored.float() / gw
        mag = (b.abs() / torch.where(step > 0, step, torch.ones_like(step))).clamp(max=6.0)
        idx = torch.bucketize(mag, mids)
        up = (idx < 7) & (mag == mids[idx.clamp(max=6)]) & (idx % 2 == 1)
        code = ((idx + up.long()) | torch.where(b < 0, 8, 0)).reshape(r1 - r0, Kd).to(torch.uint8)
        packed[r0:r1] = code[:, 0::2] | (code[:, 1::2] << 4)
        raw[r0:r1] = stored.reshape(r1 - r0, Kd // 16)
        del b, mag, idx, code
    return packed, K.swizzle_scales(raw), 1.0 / gw


class DraftLinear:
    """An NVFP4 draft copy: y = x W^T with x quantized per row at its own dynamic global scale."""

    def __init__(self, rows_fn, N, Kd, device):
        self.w, self.s, winv = _encode_nvfp4(rows_fn, N, Kd, device)
        self.winv = torch.tensor([winv], dtype=torch.float32, device=device)
        self.n, self.k = N, Kd
        self.tiling = (1, True) if N >= 16384 else (8, False)

    def __call__(self, x):
        T = x.shape[0]
        dev = x.device
        xq = torch.empty(T, self.k // 2, dtype=torch.uint8, device=dev)
        xs = K.scale_buffer(T, self.k // 16, dev)
        gi = torch.empty(T, dtype=torch.float32, device=dev)
        K.nvfp4_dyn_quant_kernel[(T,)](x, x.stride(0), xq, xs, gi, -(-(self.k // 16) // 4), K=self.k, CHUNK=1024,
                                       num_warps=4)
        alpha = gi * self.winv
        if T <= GEMV_ROWS:
            return NV.nvfp4_mma(xq, xs, self.w, self.s, alpha, *self.tiling, per_row=True)
        y = torch._scaled_mm(xq.view(torch.float4_e2m1fn_x2), self.w.view(torch.float4_e2m1fn_x2).t(), scale_a=xs,
                             scale_b=self.s, out_dtype=torch.float32)
        return (y * alpha[:, None]).to(torch.bfloat16)

    @property
    def nbytes(self):
        return self.w.numel() + self.s.numel()


def _rms(x, w):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + w)


class Mtp:
    def __init__(self, model, model_dir):
        self.m = model
        dev = model.device
        f = safe_open(os.path.join(model_dir, "model_mtp.safetensors"), "pt", device=f"cuda:{dev.index or 0}")

        def t(name):
            return f.get_tensor(PREFIX + name)

        def lin(*names):
            mats = [t(n) for n in names]
            whole = torch.cat(mats, 0) if len(mats) > 1 else mats[0]
            out = DraftLinear(lambda r0, r1: whole[r0:r1].float(), whole.shape[0], whole.shape[1], dev)
            del whole, mats
            return out

        self.norm_e = t("pre_fc_norm_embedding.weight").float()
        self.norm_h = t("pre_fc_norm_hidden.weight").float()
        self.fc = lin("fc.weight")
        a = "layers.0.self_attn."
        self.qkv = lin(a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight")
        self.o = lin(a + "o_proj.weight")
        self.q_norm = t(a + "q_norm.weight").float()
        self.k_norm = t(a + "k_norm.weight").float()
        self.in_norm = t("layers.0.input_layernorm.weight").float()
        self.post_norm = t("layers.0.post_attention_layernorm.weight").float()
        m = "layers.0.mlp."
        self.gu = lin(m + "gate_proj.weight", m + "up_proj.weight")
        self.down = lin(m + "down_proj.weight")
        self.out_norm = t("norm.weight").float()
        head = model.w.lm_head
        self.head = DraftLinear(lambda r0, r1: head.w[r0:r1].float() * head.s_flat[r0:r1, None], head.n,
                                head.w.shape[1], dev)
        self.kc = torch.empty(model.context, KVH, HD, dtype=torch.float8_e4m3fn, device=dev)
        self.vc = torch.empty(model.context, KVH, HD, dtype=torch.float8_e4m3fn, device=dev)
        self.one = torch.ones(1, dtype=torch.float32, device=dev)
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()

    def bytes_per_step(self):
        """Bytes one draft step reads: the NVFP4 draft copies of the MTP matrices and of the head."""
        return sum(x.nbytes for x in (self.fc, self.qkv, self.o, self.gu, self.down, self.head))

    def _fused(self, hidden, next_tokens):
        """fc(concat(norm_e(embed(next)), norm_h(hidden))): FP32 [T, HIDDEN]."""
        e = self.m._embed(next_tokens)
        x = torch.cat((_rms(e, self.norm_e), _rms(hidden, self.norm_h)), -1).contiguous()
        return self.fc(x).float()

    def _kv(self, x, pos, q=None, gate=None):
        T = x.shape[0]
        qkv = self.qkv(_rms(x, self.in_norm).contiguous())
        full = q is not None
        q = q if full else qkv
        gate = gate if full else qkv
        KS.qk_rope_kv_kernel[(T, HQ + 2 * KVH if full else 2 * KVH)](
            qkv, qkv.stride(0), self.q_norm, self.k_norm, self.m.cos, self.m.sin, pos, q, gate, self.kc, self.vc, 1.0,
            1.0, EPS, q, q, self.one, self.one, HQ=HQ, KVH=KVH, D=HD, ROT=ROT, GATED=True, H0=0 if full else HQ,
            num_warps=4)

    def ingest(self, hidden, next_tokens, pos):
        """Keys and values of prompt positions pos .. pos + T - 1 (no attention, no MLP)."""
        self._kv(self._fused(hidden, next_tokens), pos)

    def forward(self, hidden, next_tokens, pos):
        """T rows at positions pos .. pos + T - 1 through the whole layer, their keys and values written. Returns
        (draft logits [T, VOCAB], the normed hidden bf16 [T, HIDDEN] the next draft step reads)."""
        T = hidden.shape[0]
        dev = self.m.device
        x = self._fused(hidden, next_tokens)
        q = torch.empty(T, HQ, HD, dtype=torch.bfloat16, device=dev)
        gate = torch.empty(T, HQ, HD, dtype=torch.bfloat16, device=dev)
        self._kv(x, pos, q, gate)
        o = self.m.attend(q, gate, self.kc, self.vc, pos, 1.0, 1.0)
        x = x + self.o(o.view(T, HQ * HD)).float()
        gu = self.gu(_rms(x, self.post_norm).contiguous()).float()
        n = gu.shape[1] // 2
        x = x + self.down((F.silu(gu[:, :n]) * gu[:, n:]).contiguous()).float()
        hn = _rms(x, self.out_norm).to(torch.bfloat16)
        return self.head(hn).float(), hn
