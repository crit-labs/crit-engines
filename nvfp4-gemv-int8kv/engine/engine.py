"""Nanbeige4.3-33B (dense Llama, NVFP4 W4A16) engine with crit_core.native's Python engine interface.

One sequence at a time. Weights stay as stored (NVFP4 codes, E4M3 group scales, global scales folded into a per-row
factor); decode multiplies them in a Triton GEMV, prefill decodes each matrix to BF16 once per chunk and multiplies
with cuBLAS (FP32 accumulation). The KV cache is FP8 E4M3 with one FP32 scale per position and KV head, sized for the
required context. The residual stream is FP32, every norm, RoPE and softmax runs in FP32, prompt GEMM activations are
BF16. The decode step (52 layers, lm_head, argmax) is one captured CUDA graph; sampling runs on the device after it.
"""
import array
import json
import math
import time
from pathlib import Path

import torch
from safetensors import safe_open

from . import kernels as K
from .drafter import Drafter

# GEMV tiles per matrix shape (block_n, block_k, warps), from a sweep on the checkpoint's shapes (work/dev/).
GEMV_TILES = {"qkv": (8, 512, 4), "o": (4, 1024, 4), "gate_up": (4, 512, 2), "down": (16, 512, 8)}
# Prompts of at most this many tokens multiply the stored codes directly (no 16-bit copy of each matrix).
SKINNY_MAX = 64
# Prompts longer than this return the prefill's cached blocks to the device after the request.
LONG_PROMPT = 8192


class _Layer:
    pass


class Engine:
    def __init__(self, configuration):
        self.configuration = dict(configuration)
        self.model_dir = Path(self.configuration["model"])
        cfg = json.loads((self.model_dir / "config.json").read_text())
        self.H = cfg["hidden_size"]
        self.F = cfg["intermediate_size"]
        self.L = cfg["num_hidden_layers"]
        self.HQ = cfg["num_attention_heads"]
        self.HKV = cfg["num_key_value_heads"]
        self.D = cfg.get("head_dim") or self.H // self.HQ
        self.V = cfg["vocab_size"]
        self.eps = cfg["rms_norm_eps"]
        self.theta = float(cfg.get("rope_theta", 10000.0))
        eos = cfg.get("eos_token_id", [])
        self.stop_ids = sorted(set(eos if isinstance(eos, list) else [eos]))
        gen = self.model_dir / "generation_config.json"
        if gen.exists():
            g = json.loads(gen.read_text()).get("eos_token_id")
            if g is not None:
                self.stop_ids = sorted(set(self.stop_ids) | set(g if isinstance(g, list) else [g]))
        self.capacity = int(self.configuration.get("context_tokens") or cfg.get("max_position_embeddings"))
        self.chunk = int(self.configuration.get("prefill_chunk", 4096))
        self.kv_dtype = {"fp8": torch.float8_e4m3fn, "int8": torch.int8, "bf16": torch.bfloat16}[
            self.configuration.get("kv_cache", "int8")]
        # Prompt GEMM arithmetic per matrix group: "bf16" (BF16 inputs, FP32 accumulation) or "fp16_acc" (FP16
        # inputs accumulating in FP16, cuBLAS with allow_fp16_accumulation).
        self.gemm = {group: self.configuration.get("prefill_gemm_" + group, "bf16") for group in ("attn", "mlp")}
        # The MLP's down projection may take its own arithmetic (its input, silu(gate) * up, has the widest range).
        self.gemm["down"] = self.configuration.get("prefill_gemm_down", self.gemm["mlp"])
        # Prompt attention with INT8 Q.K^T over the INT8 cache (SageAttention's scheme) or BF16 Q.K^T.
        self.int8_qk = bool(self.configuration.get("prefill_int8_qk", False)) and self.kv_dtype == torch.int8
        self.device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_accumulation = True   # only FP16 GEMMs (the fp16_acc groups) use it
        self._load()
        self._allocate()
        # Speculative decoding with the DFlash2 drafter (configuration "draft_model"); Eval's plain engine opens with
        # speculative false, and requests with plain=True decode one token per pass on the speculative engine too.
        self.drafter = None
        if self.configuration.get("draft_model") and self.configuration.get("speculative", True):
            self.drafter = Drafter(self.configuration["draft_model"], self,
                                   int(self.configuration.get("draft_window", 8192)))
            self.tap_after = {t + 1: j for j, t in enumerate(self.drafter.taps)}   # layer index -> tap slot
            # Verify the anchor and the first vblock - 1 drafts: the GEMV reads 1.28 TB/s at 4 columns, 0.87 at 8.
            self.vblock = int(self.configuration.get("verify_block", 4))
            self.vattn = K.VerifyAttention(self.capacity, self.vblock, self.HQ, self.HKV, self.D, self.device)
            self.embed_host = K.HostTable(self.embed)
            self.d_taps = torch.zeros(1, len(self.drafter.taps) * self.H, dtype=torch.bfloat16, device=self.device)
            # Speculate while the request's drafts are accepted often enough to beat one-token steps: the round
            # costs about round_cost plain steps (measured: 26 ms against 16.5 ms), so below that many tokens per
            # round (an exponential average) the request decodes plainly and tries a round again every
            # `probe_every` tokens. The choice depends on the tokens alone, so a request repeats exactly.
            self.round_cost = float(self.configuration.get("round_cost", 1.6))
            self.probe_every = int(self.configuration.get("probe_every", 16))
        self._capture()
        if self.drafter is not None:
            self._capture_round()
        # Compile the prefill kernels (the short-prompt and the chunked path) and both sampling tails now, not in the
        # first requests' time to first token.
        for n in (SKINNY_MAX, SKINNY_MAX + 1):
            for temperature in (0.0, 1.0):
                self.generate(list(range(1, n + 1)), 2, stop_tokens=[], sampling={"temperature": temperature, "seed": 0})
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- loading
    def _load(self):
        index = json.loads((self.model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        handles = {}

        def get(name, device="cuda"):
            shard = index[name]
            if shard not in handles:
                handles[shard] = safe_open(str(self.model_dir / shard), "pt", device="cpu")
            t = handles[shard].get_tensor(name)
            return t.to(device) if device != "cpu" else t

        def nvfp4(names):
            """(codes, scales, alpha, pre, post): alpha = 2^14 / g for the GEMV and the decode to 16 bits; the skinny
            kernel's exact BF16 tile takes the power of two pre = 2^-round(log2 g) and its epilogue post = 1 / (g pre)."""
            ws, ss, als, pres, posts, ginvs, rscales = [], [], [], [], [], [], []
            for n in names:
                w = get(n + ".weight_packed")
                s = get(n + ".weight_scale")
                g = float(get(n + ".weight_global_scale", "cpu").float().reshape(-1)[0])
                e = round(math.log2(g))
                ws.append(w)
                ss.append(s)
                rows = w.shape[0]
                als.append(torch.full((rows,), K.E2M1_UNIT / g, dtype=torch.float32, device=self.device))
                pres.append(torch.full((rows,), 2.0 ** -e, dtype=torch.float32, device=self.device))
                posts.append(torch.full((rows,), 2.0 ** e / g, dtype=torch.float32, device=self.device))
                ginvs.append(torch.full((rows,), 1.0 / g, dtype=torch.float32, device=self.device))
                # FP8 prompt GEMMs: each row's scale puts its largest representable weight (6 x its largest block
                # scale / g) at 448.
                rmax = s.float().amax(-1) * 6.0 / g
                rscales.append((rmax / 448.0).clamp_min(1e-30))
            parts = [torch.cat(t).contiguous() for t in (ws, ss, als, pres, posts, ginvs)]
            rscale = torch.cat(rscales).contiguous()
            return tuple(parts[:5]) + (rscale, 1.0 / rscale, parts[5])

        self.layers = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            layer = _Layer()
            layer.qkv = nvfp4([p + "self_attn.q_proj", p + "self_attn.k_proj", p + "self_attn.v_proj"])
            layer.o = nvfp4([p + "self_attn.o_proj"])
            layer.gate_up = nvfp4([p + "mlp.gate_proj", p + "mlp.up_proj"])
            layer.down = nvfp4([p + "mlp.down_proj"])
            layer.ln_in = get(p + "input_layernorm.weight").to(torch.bfloat16)
            layer.ln_post = get(p + "post_attention_layernorm.weight").to(torch.bfloat16)
            self.layers.append(layer)
        self.norm = get("model.norm.weight").to(torch.bfloat16)
        self.lm_head = get("lm_head.weight").to(torch.bfloat16).contiguous()
        # Gathered rows only: the embedding table stays in pinned host memory.
        self.embed = get("model.embed_tokens.weight", "cpu").to(torch.bfloat16).contiguous().pin_memory()
        handles.clear()
        self.weight_bytes = sum(t.numel() * t.element_size() for layer in self.layers
                                for part in (layer.qkv, layer.o, layer.gate_up, layer.down) for t in part[:2])
        self.weight_bytes += sum(layer.ln_in.numel() * 2 * 2 for layer in self.layers) + self.lm_head.numel() * 2

    def _allocate(self):
        d, cap = self.device, self.capacity
        self.kc = [torch.zeros(cap, self.HKV, self.D, dtype=self.kv_dtype, device=d) for _ in range(self.L)]
        self.vc = [torch.zeros(cap, self.HKV, self.D, dtype=self.kv_dtype, device=d) for _ in range(self.L)]
        self.ks = [torch.zeros(cap, self.HKV, dtype=torch.float32, device=d) for _ in range(self.L)]
        self.vs = [torch.zeros(cap, self.HKV, dtype=torch.float32, device=d) for _ in range(self.L)]
        # transformers' default RoPE: inv_freq from an int64 arange in FP32, angle = position * inv_freq in FP32.
        inv = 1.0 / (self.theta ** (torch.arange(0, self.D, 2, dtype=torch.int64, device=d).float() / self.D))
        pos = torch.arange(cap, dtype=torch.int64, device=d).float()
        angles = pos[:, None] * inv[None, :]
        self.cos = angles.cos().contiguous()
        self.sin = angles.sin().contiguous()
        del angles, pos
        self.attn = K.DecodeAttention(cap, self.HQ, self.HKV, self.D, d)
        width = (self.HQ + 2 * self.HKV) * self.D
        # Decode-step buffers (fixed addresses for the graph).
        self.h_poslen = torch.zeros(2, dtype=torch.int32, pin_memory=True)
        self.d_poslen = torch.zeros(2, dtype=torch.int32, device=d)
        self.h_tok = torch.zeros(1, dtype=torch.int64, pin_memory=True)
        self.h_emb = torch.empty(1, self.H, dtype=torch.bfloat16, pin_memory=True)
        self.d_emb = torch.empty(1, self.H, dtype=torch.bfloat16, device=d)
        self.d_res = torch.empty(1, self.H, dtype=torch.float32, device=d)
        self.d_norm = torch.empty(1, self.H, dtype=torch.bfloat16, device=d)
        self.d_qkv = torch.empty(1, width, dtype=torch.float32, device=d)
        self.d_q = torch.empty(1, self.HQ, self.D, dtype=torch.bfloat16, device=d)
        self.d_att = torch.empty(self.HQ, self.D, dtype=torch.bfloat16, device=d)
        self.d_o = torch.empty(1, self.H, dtype=torch.float32, device=d)
        self.d_gu = torch.empty(1, 2 * self.F, dtype=torch.float32, device=d)
        self.d_act = torch.empty(1, self.F, dtype=torch.bfloat16, device=d)
        self.d_down = torch.empty(1, self.H, dtype=torch.float32, device=d)
        # Prefill: one BF16 matrix decoded at a time.
        self.scratch = torch.empty(max(self.F * self.H, width * self.H), dtype=torch.bfloat16, device=d)

    # ---------------------------------------------------------------- decode step
    def _decode_body(self):
        pos = self.d_poslen[0:1]
        length = self.d_poslen[1:2]
        self.d_res.copy_(self.d_emb)
        pending = None
        for i, layer in enumerate(self.layers):
            K.add_rmsnorm(pending, self.d_res, layer.ln_in, self.d_norm, self.eps)
            if self.drafter is not None and i in self.tap_after:
                j = self.tap_after[i]
                self.d_taps[:, j * self.H:(j + 1) * self.H] = self.d_res
            K.cuda_gemv(self.d_norm, layer.qkv, self.d_qkv)
            K.rope_kv(self.d_qkv, self.cos, self.sin, self.d_q, self.kc[i], self.vc[i], self.ks[i], self.vs[i],
                      self.HQ, self.HKV, self.D, pos_tensor=pos)
            self.attn(self.d_q[0], self.kc[i], self.vc[i], self.ks[i], self.vs[i], length, self.d_att)
            K.cuda_gemv(self.d_att.view(1, -1), layer.o, self.d_o)
            K.add_rmsnorm(self.d_o, self.d_res, layer.ln_post, self.d_norm, self.eps)
            K.cuda_gemv(self.d_norm, layer.gate_up, self.d_gu)
            K.silu_mul(self.d_gu[:, :self.F], self.d_gu[:, self.F:], self.d_act)
            K.cuda_gemv(self.d_act, layer.down, self.d_down)
            pending = self.d_down
        K.add_rmsnorm(pending, self.d_res, self.norm, self.d_norm, self.eps)
        if self.drafter is not None:
            self.drafter.append_context(self.d_taps, pos.long())
        return torch.mm(self.d_norm, self.lm_head.t(), out_dtype=torch.float32)

    def _sample_tail(self):
        """Temperature sampling inside a graph: the exponential race argmax(p / E) over softmax(logits / T), E drawn
        from the default CUDA generator (seeded per request); T comes from a device scalar."""
        probs = torch.softmax(self.g_logits[0] / self.d_temp, dim=-1)
        noise = torch.empty_like(probs).exponential_()
        return torch.argmax(probs / noise)

    def _capture(self):
        self.h_poslen[0], self.h_poslen[1] = 0, 1
        self.d_poslen.copy_(self.h_poslen)
        self.d_emb.zero_()
        self.d_temp = torch.ones((), dtype=torch.float32, device=self.device)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.g_logits = self._decode_body()
                self._sample_tail()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.g_logits = self._decode_body()
            # Both selections inside the step's own graph: one replay per token.
            self.g_greedy = torch.argmax(self.g_logits[0])
            self.g_sampled = self._sample_tail()
        torch.cuda.synchronize()

    def _step(self, token, position):
        """Feed `token` at `position`; returns the logits [1, V] (FP32, graph output) for the next position."""
        self.h_poslen[0], self.h_poslen[1] = position, position + 1
        self.h_tok[0] = token
        torch.index_select(self.embed, 0, self.h_tok, out=self.h_emb)
        self.d_poslen.copy_(self.h_poslen, non_blocking=True)
        self.d_emb.copy_(self.h_emb, non_blocking=True)
        self.graph.replay()
        return self.g_logits

    # ---------------------------------------------------------------- prefill
    def _matmul(self, x, part, rows=None, out_dtype=torch.float32, fp8=False):
        """x [T, K] (BF16, or FP16 for an fp16_acc group) times the NVFP4 matrix decoded to x's dtype; short prompts
        (BF16, at most SKINNY_MAX rows) multiply the stored codes directly on tensor cores."""
        w, s, alpha, pre, post = part[:5]
        if rows is not None:
            w, s, alpha, pre, post = w[rows], s[rows], alpha[rows], pre[rows], post[rows]
        if x.shape[0] <= SKINNY_MAX and x.dtype == torch.bfloat16:
            out = torch.empty(x.shape[0], w.shape[0], dtype=out_dtype, device=self.device)
            return K.skinny(x, w, s, pre, post, out)
        n, k = w.shape[0], w.shape[1] * 2
        dense = self.scratch.view(x.dtype)[:n * k].view(n, k)
        if not fp8:
            K.dequant(w, s, alpha, dense)
        if fp8:
            w8 = self.scratch.view(torch.float8_e4m3fn)[:n * k].view(n, k)
            K.dequant_fp8(w, s, alpha, part[6] if rows is None else part[6][rows], w8)
            x8, xs = K.quantize_rows_fp8(x)
            out = torch._scaled_mm(x8, w8.t(), scale_a=xs, scale_b=(part[5] if rows is None else part[5][rows])[None, :],
                                   out_dtype=torch.bfloat16)
            return out if out_dtype == torch.bfloat16 else out.to(out_dtype)
        if x.dtype == torch.float16:
            out = torch.mm(x, dense.t())
            return out if out_dtype == torch.float16 else out.to(out_dtype)
        return torch.mm(x, dense.t(), out_dtype=out_dtype)

    def _release_long_prompt_cache(self, n):
        """After a long prompt, hand the prefill's cached activation blocks back to the device: at the required
        context the KV cache, weights and drafter leave little room, and the device memory left free is a gate.
        Short prompts keep the cache (their blocks are small and reused)."""
        if n > LONG_PROMPT:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _prefill(self, ids, on_hidden=None, capture_taps=False):
        """Every position of `ids` (from 0) through the model, chunk by chunk, KV written to the cache; returns the
        last position's final normed hidden state [1, H] bf16. on_hidden(start, normed) gets every chunk's."""
        n = len(ids)
        cpu_ids = torch.tensor(ids, dtype=torch.int64)
        width = (self.HQ + 2 * self.HKV) * self.D
        F = self.F
        last = None
        for c0 in range(0, n, self.chunk):
            T = min(self.chunk, n - c0)
            emb = torch.index_select(self.embed, 0, cpu_ids[c0:c0 + T]).to(self.device, non_blocking=True)
            res = emb.float()
            del emb
            short = T <= SKINNY_MAX
            adt = torch.float16 if self.gemm["attn"] == "fp16_acc" and not short else torch.bfloat16
            mdt = torch.float16 if self.gemm["mlp"] == "fp16_acc" and not short else torch.bfloat16
            a8 = self.gemm["attn"] == "fp8" and not short
            m8 = self.gemm["mlp"] == "fp8" and not short
            d8 = self.gemm["down"] == "fp8" and not short
            ddt = torch.float16 if self.gemm["down"] == "fp16_acc" and not short else torch.bfloat16
            normed = torch.empty(T, self.H, dtype=adt, device=self.device)
            normed_mlp = normed if mdt == adt else torch.empty(T, self.H, dtype=mdt, device=self.device)
            q = torch.empty(T, self.HQ, self.D, dtype=torch.bfloat16, device=self.device)
            att = torch.empty(T, self.HQ, self.D, dtype=adt, device=self.device)
            act = torch.empty(T, F, dtype=mdt, device=self.device)
            pending = None
            taps = None
            if self.drafter is not None and capture_taps and c0 + T > n - self.drafter.window:
                taps = torch.empty(T, len(self.drafter.taps) * self.H, dtype=torch.bfloat16, device=self.device)
            for i, layer in enumerate(self.layers):
                K.add_rmsnorm(pending, res, layer.ln_in, normed, self.eps)
                if taps is not None and i in self.tap_after:
                    j = self.tap_after[i]
                    taps[:, j * self.H:(j + 1) * self.H] = res
                # 16-bit GEMM outputs go straight to the kernels that read them (RoPE, the residual add), which
                # compute in FP32: no separate FP32 copy of each output.
                qkv = self._matmul(normed, layer.qkv, out_dtype=normed.dtype, fp8=a8)
                K.rope_kv(qkv, self.cos, self.sin, q, self.kc[i], self.vc[i], self.ks[i], self.vs[i],
                          self.HQ, self.HKV, self.D, pos0=c0)
                del qkv
                if self.int8_qk:
                    q8, qs = K.quantize_q(q)
                    K.attn_prefill_i8(q8, qs, self.kc[i], self.vc[i], self.ks[i], self.vs[i], att, c0, self.HQ,
                                      self.HKV, self.D)
                    del q8, qs
                else:
                    K.attn_prefill(q, self.kc[i], self.vc[i], self.ks[i], self.vs[i], att, c0, self.HQ, self.HKV,
                                   self.D)
                o = self._matmul(att.view(T, -1), layer.o, out_dtype=torch.bfloat16 if a8 else att.dtype, fp8=a8)
                K.add_rmsnorm(o, res, layer.ln_post, normed_mlp, self.eps)
                del o
                gate = self._matmul(normed_mlp, layer.gate_up, slice(0, F), torch.bfloat16 if m8 else mdt, fp8=m8)
                up = self._matmul(normed_mlp, layer.gate_up, slice(F, 2 * F), torch.bfloat16 if m8 else mdt, fp8=m8)
                K.silu_mul(gate, up, act)
                del gate, up
                x_down = act if act.dtype == ddt or d8 else act.to(ddt)
                pending = self._matmul(x_down, layer.down, out_dtype=torch.bfloat16 if d8 else x_down.dtype, fp8=d8)
            if normed.dtype != torch.bfloat16:
                normed = torch.empty(T, self.H, dtype=torch.bfloat16, device=self.device)
            K.add_rmsnorm(pending, res, self.norm, normed, self.eps)
            del pending, res
            if on_hidden is not None:
                on_hidden(c0, normed)
            if taps is not None:
                positions = torch.arange(c0, c0 + T, device=self.device)
                self.drafter.append_context(taps, positions)
                del taps
            last = normed[T - 1:T].clone()
        return last

    # ---------------------------------------------------------------- speculative round
    def _verify(self, tokens, f):
        """The target over `tokens` [B] (device int64: anchor and drafts) at positions f .. f + B - 1 (f a device
        int64 [1]): KV written, taps captured; returns (logits [B, V] FP32, taps [B, 3H] BF16). Projections on the
        tensor-core skinny GEMM; the embedding rows come straight from the pinned host table."""
        B = tokens.shape[0]
        emb = K.embed_gather(self.embed_host, tokens, torch.empty(B, self.H, dtype=torch.bfloat16, device=self.device))
        res = emb.float()
        poslen = torch.cat((f, f + B)).to(torch.int32)
        normed = torch.empty(B, self.H, dtype=torch.bfloat16, device=self.device)
        q = torch.empty(B, self.HQ, self.D, dtype=torch.bfloat16, device=self.device)
        att = torch.empty(B, self.HQ, self.D, dtype=torch.bfloat16, device=self.device)
        act = torch.empty(B, self.F, dtype=torch.bfloat16, device=self.device)
        taps = torch.empty(B, len(self.drafter.taps) * self.H, dtype=torch.bfloat16, device=self.device)
        pending = None
        for i, layer in enumerate(self.layers):
            K.add_rmsnorm(pending, res, layer.ln_in, normed, self.eps)
            if i in self.tap_after:
                j = self.tap_after[i]
                taps[:, j * self.H:(j + 1) * self.H] = res
            qkv = K.cuda_gemv(normed, layer.qkv, torch.empty(B, layer.qkv[0].shape[0], device=self.device))
            K.rope_kv(qkv, self.cos, self.sin, q, self.kc[i], self.vc[i], self.ks[i], self.vs[i], self.HQ, self.HKV,
                      self.D, pos_tensor=poslen[0:1])
            self.vattn(q, self.kc[i], self.vc[i], self.ks[i], self.vs[i], poslen[1:2], att)
            o = K.cuda_gemv(att.view(B, -1), layer.o, torch.empty(B, self.H, device=self.device))
            K.add_rmsnorm(o, res, layer.ln_post, normed, self.eps)
            gu = K.cuda_gemv(normed, layer.gate_up, torch.empty(B, 2 * self.F, device=self.device))
            K.silu_mul(gu[:, :self.F], gu[:, self.F:], act)
            pending = K.cuda_gemv(act, layer.down, torch.empty(B, self.H, device=self.device))
        K.add_rmsnorm(pending, res, self.norm, normed, self.eps)
        return torch.mm(normed, self.lm_head.t(), out_dtype=torch.float32), taps

    def _round_body(self):
        """One speculative round on static buffers (captured as one graph): draft a block from the anchor at
        frontier f, walk the selector, verify anchor and drafts in one target pass, append the verified positions'
        taps to the drafter's context. Host-facing results land in self.r_out: drafts, target argmax per row, the
        target's processed probability of each draft and the selector's q of it."""
        d = self.drafter
        B = d.block
        positions = self.r_f + torch.arange(B, device=self.device)
        ids = torch.cat((self.r_anchor, torch.full((B - 1,), d.mask_id, dtype=torch.int64, device=self.device)))
        emb = K.embed_gather(self.embed_host, ids, torch.empty(B, self.H, dtype=torch.bfloat16, device=self.device))
        hid, unary, cand, hp = d.draft(emb, positions, self.r_f, self.r_temp, self.r_greedy, None)
        uniforms = torch.rand(B - 1, device=self.device)
        drafts, qrows, cand = d.select(unary, cand, hp, self.r_anchor, self.r_temp, self.r_greedy, uniforms)
        V = self.vblock
        drafts, qrows, cand = drafts[:V - 1], qrows[:V - 1], cand[:V - 1]
        block = torch.cat((self.r_anchor, drafts))
        logits, taps = self._verify(block, self.r_f)
        d.append_context(taps, positions[:V])
        probs = torch.softmax(logits / self.r_temp, dim=-1)
        pd = probs[:V - 1].gather(1, drafts[:, None])[:, 0]
        qd = qrows.gather(1, (cand == drafts[:, None]).float().argmax(1, keepdim=True))[:, 0]
        argmax = torch.argmax(logits, dim=-1)
        out = torch.cat((drafts.double(), argmax.double(), pd.double(), qd.double()))
        return logits, probs, qrows, cand, out

    def _capture_round(self):
        d = self.drafter
        self.r_anchor = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.r_f = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.r_temp = torch.ones((), dtype=torch.float32, device=self.device)
        self.r_greedy = torch.ones((), dtype=torch.bool, device=self.device)
        d.reset()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(2):
                self._round_body()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.round_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.round_graph), torch.inference_mode():
            self.r_logits, self.r_probs, self.r_q, self.r_cand, self.r_out = self._round_body()
        self.r_host = torch.empty(self.r_out.shape, dtype=self.r_out.dtype, pin_memory=True)
        d.reset()
        torch.cuda.synchronize()

    def _speculate(self, first, f, budget, sampling, rng, generator, replay, stops, emit):
        """Rounds from the anchor `first` at position f until `budget` tokens were emitted or a stop: each round
        drafts B - 1 tokens, verifies anchor and drafts in one target pass, accepts by argmax equality (greedy) or by
        rejection sampling against the selector's q (sampled), emits the accepted drafts and one target token, and
        appends the verified positions' taps to the drafter's context. Returns (emitted, accepted per round)."""
        d = self.drafter
        B = self.vblock
        temperature = float(sampling.get("temperature", 1.0))
        greedy = temperature <= 0.0
        filtered = not greedy and not self._plain_temperature(sampling)
        self.r_temp.fill_(max(temperature, 1e-5))
        self.r_greedy.fill_(greedy)
        anchor, emitted, accepted = int(first), 0, []
        ema, plain_left = None, 0
        while emitted < budget:
            if plain_left > 0 or f + d.block > self.capacity:
                # One-token step on the decode graph (its taps feed the drafter's context).
                logits = self._step(anchor, f)
                tok = self._select(logits, sampling, generator, replay[emitted] if replay is not None else None)
                emit(tok, logits)
                emitted += 1
                f += 1
                anchor = tok
                plain_left -= 1
                accepted.append(0)          # a target pass that emitted one token
                if (replay is None and tok in stops) or (replay is not None and emitted >= len(replay)):
                    break
                continue
            self.r_anchor.fill_(anchor)
            self.r_f.fill_(f)
            self.round_graph.replay()
            self.r_host.copy_(self.r_out)
            host = self.r_host.tolist()
            draft_list = [int(x) for x in host[:B - 1]]
            arg = [int(x) for x in host[B - 1:2 * B - 1]]
            pd = host[2 * B - 1:3 * B - 2]
            path_q = host[3 * B - 2:]
            logits, probs, qrows, cand = self.r_logits, self.r_probs, self.r_q, self.r_cand
            k = 0
            out = []
            if replay is not None:
                for i in range(B - 1):
                    if emitted + k >= len(replay) or draft_list[i] != replay[emitted + k]:
                        break
                    k += 1
                nxt = replay[emitted + k] if emitted + k < len(replay) else None
            elif greedy:
                while k < B - 1 and draft_list[k] == arg[k]:
                    k += 1
                nxt = arg[k]
            elif filtered:
                # Filters change the processed distribution; verify each draft by sampling the target row exactly.
                nxt = None
                for i in range(B - 1):
                    tok = self._select(logits[i:i + 1], sampling, generator)
                    if tok != draft_list[i]:
                        nxt = tok
                        break
                    k += 1
                if nxt is None:
                    nxt = self._select(logits[B - 1:B], sampling, generator)
            else:
                nxt = None
                for i in range(B - 1):
                    if rng.random() < min(1.0, pd[i] / max(path_q[i], 1e-30)):
                        k += 1
                        continue
                    # residual max(p - q, 0) over the vocabulary, q sparse on the candidates
                    resid = probs[i].clone()
                    resid.index_put_((cand[i],), (resid[cand[i]] - qrows[i]).clamp_min(0.0))
                    noise = torch.empty_like(resid).exponential_(generator=generator)
                    nxt = int(torch.argmax(resid / noise))
                    break
                if nxt is None:
                    noise = torch.empty_like(probs[B - 1]).exponential_(generator=generator)
                    nxt = int(torch.argmax(probs[B - 1] / noise))
            out = draft_list[:k] + ([nxt] if nxt is not None else [])
            accepted.append(k)
            ema = (k + 1) if ema is None else 0.7 * ema + 0.3 * (k + 1)
            if ema < self.round_cost:
                plain_left = self.probe_every
            stop = False
            for j, tok in enumerate(out):
                if emitted >= budget:
                    stop = True
                    break
                emit(tok, logits[j:j + 1])
                emitted += 1
                if replay is None and tok in stops:
                    stop = True
                    break
            if stop or nxt is None:
                break
            f += k + 1
            anchor = nxt
        return emitted, accepted

    # ---------------------------------------------------------------- selection
    @staticmethod
    def _plain_temperature(sampling):
        """True when only a temperature applies (no top-k, top-p or min-p filter)."""
        top_k = int(sampling.get("top_k") or 0)
        top_p = sampling.get("top_p")
        return top_k <= 0 and (top_p is None or float(top_p) >= 1.0) and not float(sampling.get("min_p") or 0.0)

    def _select(self, logits, sampling, generator, forced=None):
        """The next token from FP32 logits [1, V] under `sampling` (temperature 0: argmax), or `forced`. Logits that
        are the decode graph's own select through its captured tails (greedy, or temperature without filters)."""
        if forced is not None:
            return int(forced)
        temperature = float(sampling.get("temperature", 1.0))
        if logits is self.g_logits and (temperature <= 0.0 or self._plain_temperature(sampling)):
            # The step's graph selected with the request's temperature (set in d_temp before the replay).
            return int(self.g_greedy if temperature <= 0.0 else self.g_sampled)
        if temperature <= 0.0:
            return int(torch.argmax(logits[0]))
        x = logits[0] / temperature
        top_k = int(sampling.get("top_k") or 0)
        if 0 < top_k < x.numel():
            kth = torch.topk(x, top_k).values[-1]
            x = torch.where(x < kth, torch.full_like(x, float("-inf")), x)
        probs = torch.softmax(x, dim=-1)
        top_p = float(sampling.get("top_p", 1.0) if sampling.get("top_p") is not None else 1.0)
        if top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, 0)
            drop = (cum - sp) >= top_p
            sp = sp.masked_fill(drop, 0.0)
            probs = torch.zeros_like(probs).scatter_(0, si, sp)
        min_p = float(sampling.get("min_p") or 0.0)
        if min_p > 0.0:
            probs = torch.where(probs < min_p * probs.max(), torch.zeros_like(probs), probs)
        noise = torch.empty_like(probs).exponential_(generator=generator)
        return int(torch.argmax(probs / noise))

    @staticmethod
    def _logprob_row(logits, token, n):
        lsm = torch.log_softmax(logits[0], dim=-1)
        values, ids = torch.topk(lsm, n)
        return float(lsm[token]), list(zip(ids.tolist(), values.tolist()))

    def _prompt_activations(self):
        """Each group's share of the NVFP4 prompt FLOPs at its arithmetic (crit-eval's prompt_activations)."""
        attn = (self.HQ + 2 * self.HKV) * self.D * self.H + self.HQ * self.D * self.H
        gate_up, down = 2 * self.H * self.F, self.H * self.F
        total = attn + gate_up + down
        shares = {}
        for group, flops in (("attn", attn), ("mlp", gate_up), ("down", down)):
            shares[self.gemm[group]] = shares.get(self.gemm[group], 0.0) + flops / total
        return next(iter(shares)) if len(shares) == 1 else {k: round(v, 4) for k, v in shares.items()}

    # ---------------------------------------------------------------- public interface
    def info(self):
        kv_bytes = self.L * 2 * self.HKV * (self.D * self.kv_dtype.itemsize + 4)
        return {
            "implementation": "nanbeige43-nvfp4-c6",
            "context": self.capacity,
            "stop_tokens": list(self.stop_ids),
            "sampling": {"temperature": True, "top_p": True, "top_k": True, "min_p": True, "seed": True},
            "speculative": self.drafter is not None,
            "plain": self.drafter is not None,
            "replay": True,
            "token_logprobs": True,
            "max_top_logprobs": 20,
            "query_scoring": True,
            "score_paths": ["prefill"],
            "bytes_per_pass": self.weight_bytes,
            "kv_bytes_per_token": kv_bytes,
            **({"draft_width": self.vblock - 1,
                "draft_bytes_per_step": self.drafter.bytes + self.lm_head.numel() * 2} if self.drafter else {}),
            "prompt_activations": {"nvfp4": self._prompt_activations()},
            "exact_variants": {"attention projections exact": {"prefill_gemm_attn": "bf16"},
                               "MLP exact": {"prefill_gemm_mlp": "bf16"},
                               "MLP fp16_acc": {"prefill_gemm_mlp": "fp16_acc"},
                               "MLP down fp16_acc": {"prefill_gemm_down": "fp16_acc"},
                               "attention projections fp16_acc": {"prefill_gemm_attn": "fp16_acc"},
                               "prompt attention BF16 Q.K": {"prefill_int8_qk": False}},
            "kernel_work": [
                {"match": "_gemv_kernel", "term": "matmul", "format": "nvfp4"},
                {"match": "gemv_kernel<", "term": "matmul", "format": "nvfp4"},
                {"match": "_dequant_kernel", "term": "matmul", "format": "nvfp4"},
                {"match": "gemm", "term": "matmul", "format": "nvfp4"},
                {"match": "_skinny_kernel", "term": "matmul", "format": "nvfp4"},
                {"match": "_attn_", "term": "attention"},
                {"match": "_rope_kv", "term": "attention"},
            ],
            "native_libraries": [str(Path(K.__file__).with_name("libnvfp4_gemv.so"))],
        }

    def generate(self, prompt, output_tokens, *, stop_tokens=None, sampling=None, replay=None, plain=False,
                 logprobs=None, on_tokens=None, cancelled=None):
        started = time.perf_counter()
        prompt = [int(t) for t in prompt]
        n = len(prompt)
        if n == 0 or n + output_tokens > self.capacity:
            raise ValueError(f"prompt of {n} tokens plus {output_tokens} output exceeds the context {self.capacity}")
        stops = set(self.stop_ids) if stop_tokens is None else {int(t) for t in stop_tokens}
        sampling = dict(sampling or {})
        generator = None
        self._last_accepted = []
        seed = 0
        if float(sampling.get("temperature", 1.0)) > 0.0:
            seed = sampling.get("seed")
            seed = int(seed) if seed is not None else int.from_bytes(__import__("os").urandom(8), "little") >> 1
            # The captured sampling tail and the drafter's walk draw from the default CUDA generator; filtered
            # requests and rejected drafts from their own.
            torch.cuda.manual_seed(seed)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            self.d_temp.fill_(float(sampling.get("temperature", 1.0)))
        tokens = []
        finish = "length"
        # Greedy requests speculate (adaptively); sampled ones decode one token per pass: at temperature 1 the
        # drafter's acceptance does not cover a round's cost (candidate 3).
        speculative = (self.drafter is not None and not plain
                       and float((sampling or {}).get("temperature", 1.0)) <= 0.0)
        with torch.inference_mode():
            if speculative:
                self.drafter.reset()
            last = self._prefill(prompt, capture_taps=speculative)
            logits = torch.mm(last, self.lm_head.t(), out_dtype=torch.float32)
            prefill_done = time.perf_counter()
            first_ms = None
            position = n
            if speculative and output_tokens > 1:
                import random
                rng = random.Random(seed if generator is not None else 0)
                token = self._select(logits, sampling, generator, replay[0] if replay is not None else None)
                tokens.append(token)
                first_ms = (time.perf_counter() - started) * 1000
                state = {"finish": "length"}

                def emit(tok, row):
                    if on_tokens is not None:
                        if logprobs is not None:
                            on_tokens([tok], [self._logprob_row(row, tok, int(logprobs))])
                        else:
                            on_tokens([tok])

                emit(token, logits)
                if not (replay is None and token in stops):
                    def emit_more(tok, row):
                        tokens.append(tok)
                        emit(tok, row)
                    _, accepted = self._speculate(token, n, output_tokens - 1, sampling, rng, generator,
                                                  replay[1:] if replay is not None else None, stops, emit_more)
                    self._last_accepted = accepted
                if replay is None and tokens[-1] in stops:
                    finish = "stop"
                torch.cuda.synchronize()
                ended = time.perf_counter()
                timing = {"prefill_tokens": n, "reused_prompt_tokens": 0,
                          "prefill_ms": (prefill_done - started) * 1000, "first_token_ms": first_ms,
                          "decode_ms": (ended - started) * 1000 - first_ms,
                          "decode_steps": len(getattr(self, "_last_accepted", []))}
                acc = list(getattr(self, "_last_accepted", []))
                self._release_long_prompt_cache(n)
                return tokens, {"finish": finish, "timing": timing, "accepted": acc, "rounds": len(acc)}
            for i in range(output_tokens):
                if cancelled is not None and cancelled():
                    return None, {"cancelled": True}
                if i > 0:
                    logits = self._step(tokens[-1], position)
                    position += 1
                token = self._select(logits, sampling, generator, replay[i] if replay is not None else None)
                tokens.append(token)
                if first_ms is None:
                    first_ms = (time.perf_counter() - started) * 1000
                if on_tokens is not None:
                    if logprobs is not None:
                        on_tokens([token], [self._logprob_row(logits, token, int(logprobs))])
                    else:
                        on_tokens([token])
                if replay is None and token in stops:
                    finish = "stop"
                    break
        torch.cuda.synchronize()
        ended = time.perf_counter()
        timing = {"prefill_tokens": n, "reused_prompt_tokens": 0,
                  "prefill_ms": (prefill_done - started) * 1000, "first_token_ms": first_ms or 0.0,
                  "decode_ms": (ended - started) * 1000 - (first_ms or 0.0), "decode_steps": max(0, len(tokens) - 1)}
        self._release_long_prompt_cache(n)
        return tokens, {"finish": finish, "timing": timing}

    def score(self, ids, first_target, path="", queries=None, top_k=0, packed=False):
        from crit_core.native import Scores
        from crit_core.serving import ScoreRows
        ids = [int(t) for t in ids]
        n = len(ids)
        if n > self.capacity:
            raise ValueError(f"{n} tokens exceed the context {self.capacity}")
        targets, query_rows, top_ids, top_vals = [], [], [], []
        per = len(queries[0]) if queries else 0
        ids_dev = torch.tensor(ids, dtype=torch.int64, device=self.device)
        qdev = torch.tensor(queries, dtype=torch.int64, device=self.device) if queries else None

        def on_hidden(c0, normed):
            T = normed.shape[0]
            for r0 in range(0, T, 128):
                rows = normed[r0:r0 + 128]
                p0 = c0 + r0                     # position of the first row; it predicts the token at p0 + 1
                lo = max(p0 + 1, first_target)
                hi = min(p0 + 1 + rows.shape[0], n)
                if lo >= hi:
                    continue
                sel = rows[lo - 1 - p0:hi - 1 - p0]
                lsm = torch.log_softmax(torch.mm(sel, self.lm_head.t(), out_dtype=torch.float32), dim=-1)
                tgt = ids_dev[lo:hi]
                targets.append(lsm.gather(1, tgt[:, None])[:, 0].cpu())
                if per:
                    query_rows.append(lsm.gather(1, qdev[lo - first_target:hi - first_target]).cpu())
                if top_k:
                    v, t = torch.topk(lsm, top_k, dim=-1)
                    top_ids.append(t.to(torch.int32).cpu())
                    top_vals.append(v.cpu())
                del lsm

        with torch.inference_mode():
            self._prefill(ids, on_hidden)
        self._release_long_prompt_cache(n)
        target = torch.cat(targets).tolist() if targets else []
        query = torch.cat(query_rows).tolist() if per and query_rows else None
        if top_k:
            ti = torch.cat(top_ids) if top_ids else torch.empty(0, top_k, dtype=torch.int32)
            tv = torch.cat(top_vals) if top_vals else torch.empty(0, top_k)
            if packed:
                top = ScoreRows(array.array("f", torch.tensor(target, dtype=torch.float32).numpy().tobytes()),
                                array.array("i", ti.numpy().tobytes()), array.array("f", tv.float().numpy().tobytes()),
                                top_k)
            else:
                top = [list(zip(a, b)) for a, b in zip(ti.tolist(), tv.tolist())]
        else:
            top = None
        return Scores(target, query, None, top)

    def close(self):
        self.graph = None
        self.layers = []
        self.kc = self.vc = self.ks = self.vs = []
        self.lm_head = self.embed = self.scratch = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def open_engine(configuration):
    return Engine(configuration)
