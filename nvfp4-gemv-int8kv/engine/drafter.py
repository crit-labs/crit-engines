"""DFlash2 block drafter (ProCreations/Nanbeige4.3-33B-DFlash2) for the Nanbeige engine.

Ported from SGLang 0.5.21 (Apache-2.0): sglang/srt/models/dflash.py (DFlashDraftModel, DFlashGroupedConv,
DFlash2DraftModel, CandidateSelector) and sglang/srt/speculative/dflash_worker_v2.py (_SelectorDraftSampler). The
drafter reads the target's residual stream after target layers 1, 25 and 49 ("taps"), projects their concatenation
through `fc` and `hidden_norm` into a context whose per-layer keys and values every block attends to, and fills a block
[anchor, mask x 7] at positions f .. f + 7 in one non-causal pass. Each mask position's hidden state goes through the
target's lm_head; its top 32 tokens are candidates, and the selector scores transitions between adjacent slots with
two codebooks and walks one path, returning the 7 drafts and the conditional distribution q of each over its
candidates (one-hot for greedy) for the target's rejection-sampling verify.

Projections are re-encoded to NVFP4 (draft-only weights; the stored target verifies every token). The context keeps
the latest `window` positions in a ring with a position tag per slot.
"""
import json
from pathlib import Path

import torch
from safetensors import safe_open

from . import kernels as K


def _rms(x, weight, eps):
    """RMSNorm over the last dimension in FP32, transformers' order: weight * normalized."""
    x = x.float()
    return weight.float() * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


class Drafter:
    def __init__(self, path, engine, window=8192):
        self.engine = engine
        self.path = Path(path)
        cfg = json.loads((self.path / "config.json").read_text())
        dcfg = cfg["dflash_config"]
        self.block = int(dcfg["block_size"])
        self.mask_id = int(dcfg["mask_token_id"])
        self.taps = [int(i) for i in dcfg["target_layer_ids"]]      # target layer whose output is read
        self.top_k = int(dcfg["selector_top_k"])
        self.taps_k = int(dcfg["conv_kernel_size"])
        self.group = int(dcfg["conv_group_size"])
        self.out_mult = float(dcfg.get("output_multiplier", 1.0))
        self.H = int(cfg["hidden_size"])
        self.HQ = int(cfg["num_attention_heads"])
        self.HKV = int(cfg["num_key_value_heads"])
        self.D = int(cfg["head_dim"])
        self.F = int(cfg["intermediate_size"])
        self.L = int(cfg["num_hidden_layers"])
        self.eps = float(cfg["rms_norm_eps"])
        self.window = int(window)
        self.ring = self.window + self.block
        dev = engine.device
        f = safe_open(str(self.path / "model.safetensors"), "pt", device="cpu")
        get = lambda n: f.get_tensor(n).to(dev)
        q4 = lambda t: K.quantize_nvfp4(t)
        self.bytes = 0

        def enc(tensor):
            part = q4(tensor)
            self.bytes += part[0].numel() + part[1].numel()
            return part
        self.fc = enc(get("fc.weight"))
        self.hidden_norm = get("hidden_norm.weight")
        self.norm = get("norm.weight")
        self.layers = []
        for i in range(self.L):
            p = f"layers.{i}."
            lay = {}
            lay["qkv"] = enc(torch.cat([get(p + "self_attn.q_proj.weight"), get(p + "self_attn.k_proj.weight"),
                                        get(p + "self_attn.v_proj.weight")]))
            lay["kv"] = enc(torch.cat([get(p + "self_attn.k_proj.weight"), get(p + "self_attn.v_proj.weight")]))
            lay["o"] = enc(get(p + "self_attn.o_proj.weight"))
            lay["gate_up"] = enc(torch.cat([get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight")]))
            lay["down"] = enc(get(p + "mlp.down_proj.weight"))
            lay["kp_attn"] = enc(get(p + "attention_conv.kernel_projection.weight"))
            lay["kp_mlp"] = enc(get(p + "mlp_conv.kernel_projection.weight"))
            lay["base_attn"] = get(p + "attention_conv.base_kernel").float()     # [in/out, taps, hidden]
            lay["base_mlp"] = get(p + "mlp_conv.base_kernel").float()
            lay["ln_in"] = get(p + "input_layernorm.weight")
            lay["ln_post"] = get(p + "post_attention_layernorm.weight")
            lay["q_norm"] = get(p + "self_attn.q_norm.weight")
            lay["k_norm"] = get(p + "self_attn.k_norm.weight")
            # context keys/values (BF16, ring of window + block slots) and the position each slot holds
            lay["ck"] = torch.zeros(self.ring, self.HKV, self.D, dtype=torch.bfloat16, device=dev)
            lay["cv"] = torch.zeros(self.ring, self.HKV, self.D, dtype=torch.bfloat16, device=dev)
            self.layers.append(lay)
        self.slot_pos = torch.full((self.ring,), -1, dtype=torch.int64, device=dev)
        self.pred_code = get("candidate_selector.predecessor_codebook")
        self.succ_code = get("candidate_selector.successor_codebook")
        self.hproj = get("candidate_selector.hidden_projection.weight")
        self.bytes += sum(t.numel() * 2 for t in (self.hproj,))
        del f
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ context
    def reset(self):
        self.slot_pos.fill_(-1)

    def append_context(self, taps, positions):
        """taps [n, 3H] BF16 (the target's residual stream after the tap layers) at `positions` [n] (device int64):
        project and write each drafter layer's context key and value into the ring. Static shapes when n is fixed."""
        e = self.engine
        n = taps.shape[0]
        h = e._matmul(taps, self.fc)                               # [n, H] fp32
        h = _rms(h, self.hidden_norm, self.eps).to(torch.bfloat16)
        slots = positions % self.ring
        cos, sin = e.cos[positions], e.sin[positions]               # [n, D/2]
        for lay in self.layers:
            kv = e._matmul(h, lay["kv"])                            # [n, 2 * HKV * D]
            k = kv[:, :self.HKV * self.D].view(n, self.HKV, self.D)
            v = kv[:, self.HKV * self.D:].view(n, self.HKV, self.D)
            k = self._rope(_rms(k, lay["k_norm"], self.eps), cos, sin)
            lay["ck"][slots] = k.to(torch.bfloat16)
            lay["cv"][slots] = v.to(torch.bfloat16)
        self.slot_pos[slots] = positions

    @staticmethod
    def _rope(x, cos, sin):
        """Neox RoPE (rotate_half) on [n, heads, D] FP32 with cos/sin [n, D/2]."""
        half = x.shape[-1] // 2
        a, b = x[..., :half], x[..., half:]
        c, s = cos[:, None, :], sin[:, None, :]
        return torch.cat((a * c - b * s, b * c + a * s), dim=-1)

    def _conv(self, x, delta, base):
        """Grouped dynamic K-tap convolution across the block's positions (DFlashGroupedConv._convolve): x [B, H],
        delta [B, taps, groups], base [taps, H]."""
        B = x.shape[0]
        groups = self.H // self.group
        blocks = x.float().view(B, groups, self.group)
        coef = base.view(1, self.taps_k, groups, self.group) + delta[..., None]
        out = coef[:, 0] * blocks
        for tap in range(1, self.taps_k):
            shifted = torch.zeros_like(blocks)
            shifted[tap:] = blocks[:-tap]
            out = out + coef[:, tap] * shifted
        return out.view(B, self.H)

    # ------------------------------------------------------------------ draft
    def draft(self, emb, positions, frontier, temperature, greedy, uniforms):
        """One block: emb [B, H] BF16 ([anchor, mask...]), positions [B] int64 device (frontier .. frontier + B - 1),
        frontier [1] int64 device. Returns (drafts [B-1] int64, q rows [B-1, top_k] fp32, candidates [B-1, top_k]).
        temperature: device scalar; greedy: device bool scalar; uniforms [B-1] fp32 device."""
        e = self.engine
        B = emb.shape[0]
        groups = self.H // self.group
        cos, sin = e.cos[positions], e.sin[positions]
        # context validity: slots holding a position in [frontier - window, frontier)
        valid = (self.slot_pos < frontier) & (self.slot_pos >= frontier - self.window)
        res = emb.float()
        pending = None
        for lay in self.layers:
            if pending is not None:
                res = res + pending
            hn = _rms(res, lay["ln_in"], self.eps).to(torch.bfloat16)
            coefs = e._matmul(hn, lay["kp_attn"]).view(B, 2, self.taps_k, groups)
            hc = self._conv(hn, coefs[:, 0], lay["base_attn"][0]).to(torch.bfloat16)
            qkv = e._matmul(hc, lay["qkv"])
            q = qkv[:, :self.HQ * self.D].view(B, self.HQ, self.D)
            k = qkv[:, self.HQ * self.D:(self.HQ + self.HKV) * self.D].view(B, self.HKV, self.D)
            v = qkv[:, (self.HQ + self.HKV) * self.D:].view(B, self.HKV, self.D)
            q = self._rope(_rms(q, lay["q_norm"], self.eps), cos, sin)
            k = self._rope(_rms(k, lay["k_norm"], self.eps), cos, sin)
            keys = torch.cat((lay["ck"], k.to(torch.bfloat16)), 0)          # [ring + B, HKV, D]
            vals = torch.cat((lay["cv"], v.to(torch.bfloat16)), 0)
            mask = torch.cat((valid, torch.ones(B, dtype=torch.bool, device=valid.device)))
            g = self.HQ // self.HKV
            qh = q.to(torch.bfloat16).view(B, self.HKV, g, self.D).permute(1, 0, 2, 3).reshape(self.HKV, B * g, self.D)
            scores = torch.bmm(qh, keys.permute(1, 2, 0), out_dtype=torch.float32) * self.D ** -0.5
            scores = scores.masked_fill(~mask[None, None, :], float("-inf"))
            probs = torch.softmax(scores, dim=-1).to(torch.bfloat16)
            att = torch.bmm(probs, vals.permute(1, 0, 2))                    # [HKV, B * g, D]
            att = att.view(self.HKV, B, g, self.D).permute(1, 0, 2, 3).reshape(B, self.HQ * self.D)
            o = e._matmul(att.contiguous(), lay["o"])
            o = self._conv(o, coefs[:, 1], lay["base_attn"][1])
            res = res + o
            hn = _rms(res, lay["ln_post"], self.eps).to(torch.bfloat16)
            coefs = e._matmul(hn, lay["kp_mlp"]).view(B, 2, self.taps_k, groups)
            hc = self._conv(hn, coefs[:, 0], lay["base_mlp"][0]).to(torch.bfloat16)
            gu = e._matmul(hc, lay["gate_up"])
            act = (torch.nn.functional.silu(gu[:, :self.F]) * gu[:, self.F:]).to(torch.bfloat16)
            down = e._matmul(act, lay["down"])
            pending = self._conv(down, coefs[:, 1], lay["base_mlp"][1])
        hid = _rms(res + pending, self.norm, self.eps).to(torch.bfloat16)[1:]   # mask positions
        logits = torch.mm(hid, e.lm_head.t(), out_dtype=torch.float32)
        if self.out_mult != 1.0:
            logits = logits * self.out_mult
        unary, cand = torch.topk(logits, self.top_k, dim=-1)                   # [B-1, k]
        hp = torch.mm(hid, self.hproj.t(), out_dtype=torch.float32)            # [B-1, rank]
        return hid, unary, cand, hp

    def select(self, unary, cand, hp, anchor_id, temperature, greedy, uniforms):
        """CandidateSelector.build_lattice + sample_path: scores [E, k, k] and one walk."""
        E, k = cand.shape
        pred_ids = torch.cat((anchor_id.view(1, 1).expand(1, k), cand[:-1]), 0)   # [E, k]
        P = self.pred_code[pred_ids].float()                                      # [E, k, r]
        S = self.succ_code[cand].float()                                          # [E, k, r]
        scores = unary[:, None, :] + torch.einsum("epr,ecr->epc", P * hp[:, None, :], S)
        t = temperature
        first = torch.softmax(scores[0, 0] / t, dim=-1)
        trans = torch.softmax(scores[1:] / t, dim=-1)                             # [E-1, k, k]
        idx0 = (uniforms[0] >= torch.cumsum(first, 0)).sum().clamp(max=k - 1)
        # 0-d index tensors would make the indexing read them on the host; every step stays on the device.
        ar = torch.arange(k, device=cand.device)
        idx0 = torch.where(greedy, torch.argmax(scores[0, 0]), idx0)
        path = [idx0]
        rows = [torch.where(greedy, (ar == idx0).float(), first)]
        for e_ in range(1, E):
            prev = path[-1].view(1)
            pr = trans[e_ - 1].index_select(0, prev)[0]                           # [k]
            idx = (uniforms[e_] >= torch.cumsum(pr, 0)).sum().clamp(max=k - 1)
            idx = torch.where(greedy, torch.argmax(scores[e_].index_select(0, prev)[0]), idx)
            rows.append(torch.where(greedy, (ar == idx).float(), pr))
            path.append(idx)
        path = torch.stack(path)
        drafts = cand.gather(1, path[:, None])[:, 0]
        return drafts, torch.stack(rows), cand
