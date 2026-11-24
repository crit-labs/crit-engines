"""Qwen3.5 / Qwen3.6 mixture-of-experts hybrid decoder (qwen3_5_moe_text) from a GGUF, as a model plugin
(transformers' modeling_qwen3_5_moe; llama.cpp's qwen35moe graph for the GGUF's tensor names and layout).

Per layer: h = rms_norm(x) * attn_norm (the GGUF's norm weights carry transformers' +1). Linear-attention layers (three
of every four): mixed = h W_qkv (16 query and 16 key heads of 128, 32 value heads of 128), z = h W_gate, beta =
sigmoid(h W_beta), g = softplus(h W_alpha + dt_bias) * ssm_a (ssm_a = -exp(A_log)); the causal conv (kernel 4) with SiLU
and the gated delta rule (qwen35-recurrence); o = rms_norm(core per head) * ssm_norm * silu(z), out = o W_out. Full-
attention layers: q|gate = h W_q (per head 256 query then 256 gate values), k, v = h W_k, h W_v (2 KV heads of 256),
query and key head norms, RoPE (theta 1e7) over the first 64 dims in half-split pairs (text positions: the three mrope
sections coincide), causal softmax attention with scale 1/16, times sigmoid(gate), out = a W_o. x += out. Then
hn = rms_norm(x) * post_attention_norm; the routed experts (softmax over 256 router logits, top 8, renormalized,
SwiGLU, gguf-experts) plus the shared SwiGLU expert times sigmoid(hn . w_shared_gate); x += both. Logits = rms_norm(x)
* output_norm times the untied head.

The residual stream is FP32. Matrices go through the projections slot as stored (GGUF blocks, FP32 matrices in FP32),
norms and rotary positions through the norms slot, routing through the router, experts through the experts slot, the
recurrence through the recurrence slot, KV and attention through qwen35-attention's paged pools. The token embedding
stays in host memory (rows decoded on the host as a step needs them). Each prompt chunk runs its matrices alone, decode
rows 8 at a time, so a row's tokens never depend on the rows beside it."""
import json
import math
from pathlib import Path

from crit_core.batch import DECODE, PREFILL, StepLayout, step_item
from crit_core.native import Scores, StepResult
from crit_plugins import StepBatch

CORE_SAMPLING = ("temperature", "top_p", "top_k", "min_p", "seed")
_CAPTURE_STREAMS = {}
DECODE_ROWS = 8


class Session:
    def __init__(self, handle, request):
        self.handle, self.request, self.prompt = handle, request, list(request.prompt)
        self.done, self.outputs, self.pending = 0, [], None
        self.phase, self.produced, self.rows = None, [], 0


class DecodeBatch:
    """A captured decode step's rows: one token each, every field a static device buffer (positions, page tables,
    recurrent slots), which qwen35-attention reads as decode_static."""
    round = False
    prompt = False

    def __init__(self, static, page):
        self.decode_static = static
        self.positions_t = static["positions"]
        self.rows = static["positions"].shape[0]
        self.page = page

    @property
    def tokens(self):
        return self.rows

    def kv_slots(self):
        s = self.decode_static
        pages = s["table"].gather(1, (s["positions"] // self.page)[:, None])[:, 0].to(s["positions"].dtype)
        return pages * self.page + s["positions"] % self.page


class Tensor:
    def __init__(self, kind, shape, data):
        self.kind, self.shape, self.data = kind, tuple(shape), data


def gguf_tensors(model):
    """{gguf name: Tensor} over the model's GGUF, each a uint8 view of the mapped file ([rows][row_bytes], rows over
    every leading dimension), not yet copied anywhere."""
    import numpy as np
    import torch
    from crit_core.model import open_model
    source = open_model(model)
    found, maps = {}, {}
    for name, info in source.weights.tensors.items():
        shard = str(info.shard)
        if shard not in maps:
            maps[shard] = np.memmap(shard, dtype=np.uint8, mode="r")
        view = maps[shard][info.data_begin:info.data_end]
        rows = 1
        for d in info.shape[:-1]:
            rows *= d
        found[name] = Tensor(info.dtype, info.shape, torch.from_numpy(view).view(rows, -1))
    return MappedTensors(found, maps.values())


class MappedTensors(dict):
    """gguf_tensors' result: the tensors, and the file maps their views read."""
    def __init__(self, found, maps):
        super().__init__(found)
        self.maps = list(maps)


def release_mapped_pages(maps):
    """Takes the pages a load read through the file maps out of this process's working set (they stay in the file
    cache, counted as available memory). Without it the 17.7 GB the load reads stay resident on a 16 GB host until
    Windows trims them under pressure, which leaves almost no host memory available while the engine serves. Values
    are unchanged: a page read again (the embedding rows) faults back in from the cache. Returns the bytes released."""
    import ctypes
    import sys
    released = 0
    for m in maps:
        size = int(m.nbytes)
        if sys.platform == "win32":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.VirtualUnlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            kernel32.VirtualUnlock(ctypes.c_void_p(int(m.ctypes.data)), ctypes.c_size_t(size))  # not locked: removes the pages from the working set
        else:
            mapping = getattr(m, "_mmap", None)
            if mapping is None or not hasattr(mapping, "madvise"):
                continue
            import mmap
            mapping.madvise(mmap.MADV_DONTNEED)   # read-only file pages: dropped from the process, kept in the page cache
        released += size
    return released


def rope_permutation(head_dim, rotary):
    """The head's dims reordered so the rotary pairs (j, j + rotary / 2), j < rotary / 2, become half-split pairs
    (j, j + head_dim / 2) and every other dim sits in a pair the norms kernel leaves as it is (frequency 0). Queries and
    keys take the same order, so their dot products are unchanged."""
    half, rh = head_dim // 2, rotary // 2
    rest = list(range(rotary, head_dim))
    return list(range(rh)) + rest[:half - rh] + list(range(rh, rotary)) + rest[half - rh:]


class Qwen35Model:
    def __init__(self, context, options, parts, tensors=None):
        import torch
        self.torch, self.options = torch, options
        config = context.config.get("text_config") or context.config
        if config.get("model_type") not in ("qwen3_5_moe_text", "qwen3_5_moe"):
            raise ValueError(f"qwen35-model implements qwen3_5_moe_text, not {config.get('model_type')!r}")
        self.config = config
        self.layers, self.hidden = config["num_hidden_layers"], config["hidden_size"]
        self.heads, self.kv_heads, self.head_dim = (config["num_attention_heads"], config["num_key_value_heads"],
                                                    config["head_dim"])
        self.kh, self.vh = config["linear_num_key_heads"], config["linear_num_value_heads"]
        self.dk, self.dv = config["linear_key_head_dim"], config["linear_value_head_dim"]
        self.eps, self.vocabulary = config["rms_norm_eps"], config["vocab_size"]
        self.types = list(config["layer_types"])
        self.experts_count, self.top = config["num_experts"], config["num_experts_per_tok"]
        self.moe_inter = config["moe_intermediate_size"]
        rope = config.get("rope_parameters") or {}
        self.theta = float(rope.get("rope_theta", 1e7))
        self.rotary = int(self.head_dim * float(rope.get("partial_rotary_factor", 0.25)))
        self.device = torch.device("cuda")
        self.proj, self.attention, self.sampler = parts["projections"], parts["attention"], parts["sampler"]
        self.norms, self.router, self.experts = parts["norms"], parts["router"], parts["experts"]
        self.recurrence = parts["recurrence"]
        self.context = int(context.context_tokens or options["context"])
        self.concurrency = int(options["concurrency"])
        self.chunk = int(options["prefill_chunk"])
        # a prompt step's cost (s) = base + n * (token + key * (context + n / 2)): rows capped to the budget
        self.step_budget = int(options.get("step_budget_ms", 0)) / 1e3
        self.step_cost = (int(options.get("step_base_ms", 360)) / 1e3, int(options.get("step_token_us", 660)) / 1e6,
                          int(options.get("step_key_ps", 10500)) / 1e12)
        torch.cuda.empty_cache()
        start = torch.cuda.memory_allocated(self.device)

        source = tensors if tensors is not None else gguf_tensors(context.model)
        self.source = source
        full = [i for i, t in enumerate(self.types) if t == "full_attention"]
        linear = [i for i, t in enumerate(self.types) if t != "full_attention"]
        self.full_layers, self.linear_layers = full, linear

        # -- non-expert weights on the device, every stored-block matrix in one arena -------------------------------
        def is_expert(name):
            return "_exps." in name

        dense_kinds = ("F32", "F16", "BF16")
        block_names = [n for n, t in source.items() if t.kind not in dense_kinds and not is_expert(n)
                       and n != "token_embd.weight"]
        arena_bytes = sum(-(-int(source[n].data.numel()) // 256) * 256 for n in block_names)
        self.arena = torch.empty(arena_bytes + 256, dtype=torch.uint8, device=self.device)   # slack past the last block
        arena_at = [0]

        def f32(name):
            t = source[name]
            return t.data.contiguous().view(torch.float32).reshape(t.shape).clone()

        def vector(name):
            return f32(name).reshape(-1).to(self.device)

        def fused(names, role):
            """GGUF matrices that multiply the same input, stacked by rows into one (one GEMV for all of them)."""
            parts_ = [source[n] for n in names]
            kind, cols = parts_[0].kind, parts_[0].shape[-1]
            if any(p.kind != kind or p.shape[-1] != cols for p in parts_):
                raise ValueError(f"{names}: parts of a fused matrix share one stored kind and width")
            rows = [int(p.data.shape[0]) for p in parts_]
            total = sum(int(p.data.numel()) for p in parts_)
            view = self.arena[arena_at[0]:arena_at[0] + total]
            at = 0
            for p in parts_:
                count = int(p.data.numel())
                view[at:at + count].copy_(p.data.reshape(-1))
                at += count
            arena_at[0] += -(-total // 256) * 256
            return self.proj.load({"weight": view.view(sum(rows), -1), "cols": cols}, "gguf:" + kind, role=role), rows

        def matrix(name, role=None):
            return fused([name], role)[0]

        def f32_matrix(names, role):
            weight = torch.cat([f32(n).reshape(-1, self.hidden) for n in names])
            return self.proj.load({"weight": weight}, "f32", role=role)

        self.embed_raw = source["token_embd.weight"].data          # host: [vocabulary][row bytes], Q8_0
        self.embed_kind = source["token_embd.weight"].kind
        # the embedding stays in host memory (a lookup table): a host copy page-locked once, rows gathered on the device
        self.embed = self.proj.load({"weight": self.embed_raw.clone(), "cols": self.hidden, "host": True},
                                    "gguf:" + self.embed_kind, role="embedding")
        self.head = matrix("output.weight", "head")
        self.final_norm = vector("output_norm.weight")
        perm = rope_permutation(self.head_dim, self.rotary)
        self.perm = torch.tensor(perm, dtype=torch.int64, device=self.device)
        rh = self.rotary // 2
        inv = 1.0 / (self.theta ** (torch.arange(0, self.rotary, 2, dtype=torch.float64) / self.rotary))
        self.freq = torch.cat([inv.float(), torch.zeros(self.head_dim // 2 - rh)]).to(self.device)
        self.blocks = []
        # the load's work, reported to Core (context.working) so a slow load (a cold page cache, a second engine in
        # one host) is told from a stalled one
        working = getattr(context, "working", None) or (lambda what, done=None, total=None: None)
        for i in range(self.layers):
            working("dense layers", i, self.layers)
            p = f"blk.{i}."
            b = {"kind": self.types[i], "attn_norm": vector(p + "attn_norm.weight"),
                 "post_norm": vector(p + "post_attention_norm.weight"),
                 # the router's 256 rows and the shared expert's gate row: FP32 as stored, one product
                 "router": f32_matrix([p + "ffn_gate_inp.weight", p + "ffn_gate_inp_shexp.weight"], "router"),
                 "shared_gate_up": fused([p + "ffn_gate_shexp.weight", p + "ffn_up_shexp.weight"], "shared")[0],
                 "shared_down": matrix(p + "ffn_down_shexp.weight", "shared_down")}
            if self.types[i] == "full_attention":
                b["qkv"], b["qkv_split"] = fused([p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"], "qkv")
                b["q_norm"] = vector(p + "attn_q_norm.weight")[self.perm].contiguous()
                b["k_norm"] = vector(p + "attn_k_norm.weight")[self.perm].contiguous()
                b["o"] = matrix(p + "attn_output.weight", "o")
            else:
                b["qkvz"], b["qkvz_split"] = fused([p + "attn_qkv.weight", p + "attn_gate.weight"], "qkvz")
                b["ba"] = f32_matrix([p + "ssm_beta.weight", p + "ssm_alpha.weight"], "beta_alpha")
                b["dt_bias"], b["ssm_a"] = vector(p + "ssm_dt.bias"), vector(p + "ssm_a")
                b["conv"] = f32(p + "ssm_conv1d.weight").reshape(-1, 4).contiguous().to(self.device)
                b["ssm_norm"] = vector(p + "ssm_norm.weight")
                b["out"] = matrix(p + "ssm_out.weight", "ssm_out")
            self.blocks.append(b)

        # -- KV pages and recurrent state --------------------------------------------------------------------------
        sequences = self.concurrency + 1   # Core's sessions and one fixed-text score
        self.kv = parts["kv_cache"]
        self.kv_pool = self.kv.pool(full, self.kv_heads, self.head_dim, self.context, self.device)
        self.rec = self.recurrence.pool(linear, 2 * self.kh * self.dk + self.vh * self.dv, self.kh, self.vh,
                                        sequences, self.device, self.eps)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # -- routed experts: the residency plan's experts in device slots, the rest in pinned host memory -----------
        gu_kind = source["blk.0.ffn_gate_exps.weight"].kind
        expert_bytes = {}
        from .gguf import gguf_blocks
        for i in range(self.layers):
            dn_kind = source[f"blk.{i}.ffn_down_exps.weight"].kind
            expert_bytes[i] = (2 * self.moe_inter * gguf_blocks.row_bytes(gu_kind, self.hidden)
                               + self.hidden * gguf_blocks.row_bytes(dn_kind, self.moe_inter))
        free, _ = torch.cuda.mem_get_info()
        budget = max(0, free - (int(options["reserve_mib"]) << 20))
        if int(options.get("expert_mib", 0)) > 0:
            budget = min(budget, int(options["expert_mib"]) << 20)
        self.residency = self._residency(budget, expert_bytes, options.get("profile", ""), context)
        sizes = {}
        for i in range(self.layers):
            working("expert layers", i, self.layers)
            g = source[f"blk.{i}.ffn_gate_exps.weight"]
            u = source[f"blk.{i}.ffn_up_exps.weight"]
            d = source[f"blk.{i}.ffn_down_exps.weight"]
            E = self.experts_count
            self.experts.load_layer(i, g.data.view(E, self.moe_inter, -1), u.data.view(E, self.moe_inter, -1),
                                    d.data.view(E, self.hidden, -1), g.kind, d.kind, self.hidden, self.moe_inter,
                                    self.residency[i])
            sizes[f"experts.device.{i:02d}"] = self.experts.device_bytes(i)
            sizes[f"experts.host.{i:02d}"] = self.experts.host_bytes(i)
        # host experts copied into the KV pool's top pages while no sequence holds them (the pool reclaims them before
        # handing one out): the next most routed experts of the profile, above a floor of lend_floor_tokens positions
        self.lent_floor = None
        lend_tokens = int(options.get("lend_floor_tokens", 16384))
        if self.profile_counts is not None and hasattr(self.experts, "lend") and lend_tokens < self.context:
            min_page = -(-lend_tokens // self.kv_pool.page) + 8
            ranking = sorted(((self.profile_counts[i][e], i, e) for i in range(self.layers)
                              for e in range(self.experts_count) if e not in set(self.residency[i])), reverse=True)
            spaces = self.kv_pool.lend_spaces()
            self.lend_ranking, self.lend_min_page = [(i, e) for _, i, e in ranking], min_page
            floor = self.experts.lend(spaces, spaces[0].shape[1], self.lend_ranking, min_page)
            self.kv_pool.lend_from(floor, self.experts.reclaim)
            self.lent_floor = floor
            self.lent_target, self.relends = self.experts.lent_experts, 0
        sizes["tables.embedding"] = int(self.embed_raw.numel())
        sizes["weights.arena"] = int(self.arena.numel())
        sizes["kv.full"] = int(self.kv_pool.nbytes())
        sizes["state.recurrent"] = int(self.rec.nbytes())
        device_planned = sum(v for k, v in sizes.items() if not k.startswith(("experts.host", "tables.")))
        # the FP32 matrices, vectors and conv weights beside the arena (everything else the factory holds)
        sizes["weights.dense"] = max(0, torch.cuda.memory_allocated(self.device) - start - device_planned)
        free, _ = torch.cuda.mem_get_info()
        self.places = parts["placement"].plan(
            sizes, {"device": int(free) + sum(v for k, v in sizes.items() if not k.startswith(("experts.host",
                                                                                                "tables."))),
                    "host": _host_free() + sum(v for k, v in sizes.items() if k.startswith(("experts.host",
                                                                                            "tables.")))})

        generation = Path(context.model_file("generation_config.json")) if hasattr(context, "model_file") else None
        eos = None
        if generation is not None and generation.is_file():
            eos = json.loads(generation.read_text(encoding="utf-8")).get("eos_token_id")
        self.stops = sorted(set(eos if isinstance(eos, list) else [eos])) if eos is not None else []
        self.handles = 0
        self.route_log = None   # development: a list per layer that each step's routed ids are appended to
        import os
        self.timing = ({"step_s": 0.0, "steps": 0, "replay_ms": 0.0, "replays": 0}
                       if os.environ.get("QWEN35_TIMING") else None)   # development: step and replay times
        self.decode_steps = 0
        self.nancheck = bool(os.environ.get("QWEN35_NANCHECK"))
        self.adapt_every = int(options.get("adapt_every", 32))
        self.graphs = {}
        self.capture_bytes = 0
        if options["decode_graphs"] and getattr(self.attention, "kernels", None) is not None:
            before = torch.cuda.memory_allocated(self.device)
            working("decode graphs")
            self._capture_all()
            self.capture_bytes = max(0, torch.cuda.memory_allocated(self.device) - before)
            free, _ = torch.cuda.mem_get_info()
            self.places.update(parts["placement"].plan({"decode:captures": int(self.capture_bytes)},
                                                       {"device": int(free) + int(self.capture_bytes), "host": 0}))
        torch.cuda.empty_cache()
        # every matrix now sits in device memory or a pinned host copy: release the pages the load read
        self.released_bytes = release_mapped_pages(getattr(source, "maps", ()))

    # -- residency -----------------------------------------------------------------------------------------------------
    def _residency(self, budget, expert_bytes, profile, context):
        """Each layer's device experts: the most routed keys of the residency profile (per-layer route counts) that the
        budget holds, every layer at least `min_slots`; without a profile, an equal share a layer by expert id."""
        counts = None
        self.profile_counts = None
        if profile:
            path = Path(profile)
            if not path.is_absolute():
                path = Path(__file__).resolve().parent / path
            data = json.loads(path.read_text(encoding="utf-8"))
            counts = data["counts"]
            if len(counts) != self.layers or any(len(c) != self.experts_count for c in counts):
                raise ValueError(f"residency profile {path}: geometry differs from the model's")
            self.profile_counts = counts
        E = self.experts_count
        if counts is None:
            per = int(budget // sum(expert_bytes.values()) * 1)
            share = max(0, min(E, int(budget / sum(expert_bytes.values()))))
            return {i: list(range(share)) for i in range(self.layers)}
        keys = sorted(((c, i, e) for i in range(self.layers) for e, c in enumerate(counts[i])), reverse=True)
        chosen = {i: [] for i in range(self.layers)}
        used = 0
        for c, i, e in keys:
            if used + expert_bytes[i] > budget:
                continue
            chosen[i].append(e)
            used += expert_bytes[i]
        return chosen

    # -- embedding ----------------------------------------------------------------------------------------------------
    def _embed(self, ids):
        """FP32 rows of the host embedding table for token ids (a host list or a device tensor), on the device."""
        torch = self.torch
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids, dtype=torch.int64).to(self.device, non_blocking=True)
        return self.proj.rows(self.embed, ids)

    def _embed_host(self, ids):
        from .gguf import gguf_blocks
        torch = self.torch
        index = torch.tensor(ids, dtype=torch.int64)
        rows = gguf_blocks.decode(self.embed_raw.index_select(0, index), self.embed_kind, self.hidden)
        return rows.pin_memory().to(self.device, non_blocking=True)

    # -- captured decode steps -----------------------------------------------------------------------------------------
    def _capture_all(self):
        torch = self.torch
        dummy = [("capture", r) for r in range(self.concurrency)]
        for sequence in dummy:
            self._open_pools(sequence)
            self.kv_pool.ensure(sequence, 1)
        try:
            with torch.inference_mode():
                for rows in range(1, self.concurrency + 1):
                    static = {"ids": torch.zeros(rows, dtype=torch.int64, device=self.device),
                              "positions": torch.zeros(rows, dtype=torch.int64, device=self.device),
                              "table": torch.stack([self.kv_pool.table_row(s) for s in dummy[:rows]]).to(self.device),
                              "starts": torch.arange(rows, dtype=torch.int32, device=self.device),
                              "counts": torch.ones(rows, dtype=torch.int32, device=self.device),
                              "slots": torch.tensor([self.rec.slot(s) for s in dummy[:rows]], dtype=torch.int32,
                                                    device=self.device),
                              "host": torch.zeros((rows, 3 + self.kv_pool.table_width), dtype=torch.int64,
                                                  pin_memory=True)}
                    stream = _CAPTURE_STREAMS.get(self.device.index or 0)
                    if stream is None:
                        stream = _CAPTURE_STREAMS[self.device.index or 0] = torch.cuda.Stream(self.device)
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(2):
                            self._static_forward(static)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        static["logits"] = self._static_forward(static)
                    self.graphs[rows] = (graph, static)
        finally:
            for sequence in dummy:
                self._close_pools(sequence)

    def _static_forward(self, static):
        batch = DecodeBatch(static, self.kv_pool.page)
        from types import SimpleNamespace
        groups = SimpleNamespace(starts=static["starts"], counts=static["counts"], slots=static["slots"])
        return self._forward(self._embed(static["ids"]), batch, None, kv_slots=batch.kv_slots(), groups=groups)

    def _replay(self, sessions):
        torch = self.torch
        graph, static = self.graphs[len(sessions)]
        host = static["host"]
        width = self.kv_pool.table_width
        for r, s in enumerate(sessions):
            self.kv_pool.ensure(s.handle, s.done + 1)
            host[r, 0], host[r, 1] = s.done, self.rec.slot(s.handle)
            table = self.kv_pool.tables[s.handle]
            host[r, 2:2 + len(table)] = torch.tensor(table, dtype=torch.int64)
            host[r, 2 + width] = s.pending
        device = host.to(self.device, non_blocking=True)
        static["ids"].copy_(device[:, 2 + width])
        static["positions"].copy_(device[:, 0])
        static["slots"].copy_(device[:, 1].to(torch.int32))
        static["table"].copy_(device[:, 2:2 + width].to(torch.int32))
        if self.timing is not None:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            self.timing["replay_ms"] += start.elapsed_time(end)
            self.timing["replays"] += 1
            return static["logits"]
        graph.replay()
        return static["logits"]

    def info(self):
        return {"implementation": "qwen3.5 MoE hybrid decoder over GGUF blocks (torch device plugins)",
                "vocabulary": self.vocabulary, "context": self.context, "prefill_chunk": self.chunk,
                "max_concurrency": self.concurrency, "max_pending": 8, "max_batch_tokens": self.chunk,
                "max_row_tokens": 1, "decode_token_work": 1, "stop_tokens": self.stops,
                "sampling": dict.fromkeys(CORE_SAMPLING, True), "token_logprobs": True, "max_top_logprobs": 20,
                "scoring": True, "query_scoring": True, "score_paths": ["prefill"], "replay": True,
                "speculative": False, "plain": False, "decode_graphs": sorted(self.graphs),
                "decode_capture_bytes": self.capture_bytes,
                "prompt_activations": {"Q8_0": "stored", "IQ3_S": "stored", "IQ4_XS": "stored", "Q6_K": "fp8"},
                "expert_residency": {"device_experts": sum(len(v) for v in self.residency.values()),
                                     "experts": self.layers * self.experts_count}}

    def telemetry(self):
        found = {}
        if hasattr(self.experts, "telemetry"):
            found.update(self.experts.telemetry())
        return found

    # -- the arithmetic ------------------------------------------------------------------------------------------------
    @staticmethod
    def _groups(batch):
        if getattr(batch, "decode_static", None) is not None:
            return [(0, batch.rows)]
        groups, run = [], None
        for r in range(batch.rows):
            first, end = batch.starts[r], batch.starts[r + 1]
            if end - first > 1:
                groups.append((first, end))
                run = None
            elif run is not None and run[1] == first and run[1] - run[0] < DECODE_ROWS:
                run = groups[-1] = (run[0], end)
            else:
                run = (first, end)
                groups.append(run)
        return groups

    def _mm(self, x, matrix, groups, out_dtype=None):
        if len(groups) == 1:
            return self.proj.matmul(x, matrix, out_dtype)
        return self.torch.cat([self.proj.matmul(x[first:end], matrix, out_dtype) for first, end in groups])

    def _experts(self, i, h, ids, weights, groups):
        if len(groups) == 1:
            return self.experts.forward(i, h, ids, weights)
        return self.torch.cat([self.experts.forward(i, h[a:e], ids[a:e], weights[a:e]) for a, e in groups])

    def _forward(self, x, batch, logit_rows, kv_slots=None, groups=None):
        """x: FP32 embedding rows [tokens][hidden] on the device."""
        torch, norms, t, eps = self.torch, self.norms, x.shape[0], self.eps
        bf16, F = torch.bfloat16, torch.nn.functional
        mm_groups = self._groups(batch)
        if kv_slots is None:
            kv_slots = self.kv_pool.reserve(batch)
        if groups is None:
            rows = []
            for r in range(batch.rows):
                sequence, first, count, position = batch.row(r)
                rows.append((first, count, self.rec.slot(sequence)))
            groups = self.recurrence.groups(rows, self.device)
        positions = batch.positions_t
        x = x.clone()
        for i, b in enumerate(self.blocks):
            h = norms.rms_norm(x, b["attn_norm"], eps, out_dtype=bf16)
            if b["kind"] == "full_attention":
                nq, nk, nv = b["qkv_split"]
                qkv = self._mm(h, b["qkv"], mm_groups, torch.float32)
                qg = qkv[:, :nq].view(t, self.heads, 2 * self.head_dim)
                q = qg[..., :self.head_dim].index_select(-1, self.perm)
                gate = qg[..., self.head_dim:]
                k = qkv[:, nq:nq + nk].view(t, self.kv_heads, self.head_dim).index_select(-1, self.perm)
                v = qkv[:, nq + nk:].view(t, self.kv_heads, self.head_dim).to(bf16)
                q = norms.rms_norm_rope(q.contiguous(), b["q_norm"], eps, positions, self.freq, out_dtype=bf16)
                k = norms.rms_norm_rope(k.contiguous(), b["k_norm"], eps, positions, self.freq, out_dtype=bf16)
                self.kv_pool.write(i, kv_slots, k, v)
                a = self.attention.forward(i, q, self.kv_pool, batch, scale=1.0 / math.sqrt(self.head_dim))
                a = (a.float() * torch.sigmoid(gate)).to(bf16)
                o = self._mm(a.reshape(t, -1), b["o"], mm_groups, torch.float32)
            else:
                nqkv, nz = b["qkvz_split"]
                qkvz = self._mm(h, b["qkvz"], mm_groups, torch.float32)
                mixed, z = qkvz[:, :nqkv], qkvz[:, nqkv:]
                ba = self._mm(h, b["ba"], mm_groups, torch.float32)
                beta, g = self.recurrence.gates(ba, b["dt_bias"], b["ssm_a"])
                batch.recurrent_groups = groups
                conv = self.recurrence.convolve(i, mixed.contiguous(), b["conv"], None, self.rec, batch)
                core = self.recurrence.forward(i, conv, None, None, g.contiguous(), beta.contiguous(), self.rec, batch)
                o = self.recurrence.gated_norm(core, z, b["ssm_norm"], eps)
                o = self._mm(o, b["out"], mm_groups, torch.float32)
            x = x + o
            hn = norms.rms_norm(x, b["post_norm"], eps, out_dtype=bf16)
            rl = self._mm(hn, b["router"], mm_groups, torch.float32)
            ids, weights = self.router.select(rl[:, :self.experts_count].contiguous(), self.top)
            if self.route_log is not None:
                self.route_log[i].append(ids.cpu())
            shared_gate = torch.sigmoid(rl[:, self.experts_count:self.experts_count + 1])
            # the routed experts launched first (the CPU tier's share posted), the shared expert beside them, then the
            # join: the shared expert's kernels run while the CPU computes its routes
            split = len(mm_groups) == 1 and hasattr(self.experts, "begin")
            pending = self.experts.begin(i, hn, ids, weights) if split else None
            if not split:
                e = self._experts(i, hn, ids, weights, mm_groups)
            su = self._mm(hn, b["shared_gate_up"], mm_groups, torch.float32)
            inner = (F.silu(su[:, :self.moe_inter]) * su[:, self.moe_inter:]).to(bf16)
            shared = self._mm(inner, b["shared_down"], mm_groups, torch.float32)
            if split:
                e = self.experts.finish(pending)
            x = x + e + shared * shared_gate
            if self.nancheck:
                for name, value in (("attn/linear out", o), ("experts", e), ("shared", shared), ("x", x)):
                    if not bool(torch.isfinite(value).all()):
                        raise FloatingPointError(f"layer {i} ({b['kind']}): {name} is not finite")
        h = norms.rms_norm(x, self.final_norm, eps, out_dtype=bf16)
        rows = list(range(t)) if logit_rows is None else logit_rows
        if not rows:
            return torch.zeros(0, self.vocabulary, dtype=torch.float32, device=self.device)
        hs = h if logit_rows is None else h[torch.tensor(rows, dtype=torch.int64, device=self.device)]
        return torch.cat([self.proj.matmul(hs[a:a + DECODE_ROWS], self.head, torch.float32)
                          for a in range(0, len(rows), DECODE_ROWS)])

    # -- Core's step model -----------------------------------------------------------------------------------------------
    def _open_pools(self, sequence):
        self.kv_pool.open(sequence)
        self.rec.open(sequence)

    def _close_pools(self, sequence):
        self.kv_pool.close(sequence)
        self.rec.close(sequence)
        self._relend()

    def _relend(self):
        """Lend the KV pool's top pages to host experts again once a long sequence that took some of them back is
        gone: when no open sequence holds a page at or above the lending floor, every lent expert points back home and
        the ranking is lent afresh from the top (about 0.3 s over the link, only after a prompt reclaimed pages)."""
        if getattr(self, "lent_floor", None) is None or self.experts.lent_experts >= self.lent_target:
            return
        pool = self.kv_pool
        if any(p >= self.lend_min_page for table in pool.tables.values() for p in table):
            return
        self.torch.cuda.synchronize()
        self.experts.reclaim(1 << 62)                  # every lent expert home
        pool.free.sort(reverse=True)                   # the lowest page handed out first, as at load
        spaces = pool.lend_spaces()
        floor = self.experts.lend(spaces, spaces[0].shape[1], self.lend_ranking, self.lend_min_page)
        pool.lend_from(floor, self.experts.reclaim)
        self.lent_floor = floor
        self.relends += 1

    def open(self, request):
        self.handles += 1
        session = Session(self.handles, request)
        self._open_pools(session.handle)
        session.expert_start = self.experts.decode_counters() if hasattr(self.experts, "decode_counters") else None
        return session

    def request_facts(self, session):
        """The request's decode expert reads and hits (the routes its decode rows read, and those on the device),
        for Eval's expert-tier floors; a concurrent request's rows count in both."""
        start = getattr(session, "expert_start", None)
        if start is None:
            return {}
        reads, hits = self.experts.decode_counters()
        return {"expert_reads": reads - start[0], "expert_hits": hits - start[1]}

    def step(self, rows):
        if self.timing is None:
            return self._step(rows)
        import time
        torch = self.torch
        t0 = time.perf_counter()
        out = self._step(rows)
        self.timing["step_s"] += time.perf_counter() - t0
        self.timing["steps"] += 1
        return out

    def _step(self, rows):
        torch = self.torch
        # a pending warm-up runs before a step of decode rows only (never inside another request's prompt step)
        if getattr(self, "warm_pending", False) and all(r.session.done >= len(r.session.prompt) for r in rows):
            self.warm_pending = False
            with torch.inference_mode():
                self.experts.prompt_warm()
        if self.step_budget > 0:
            rows = self._budgeted(rows)
        layout = StepLayout(step_item(row, len(row.session.prompt), row.session.done) for row in rows)
        packed, ids, logit_rows, placed = [], [], [], []
        for placement in layout.placements:
            s = rows[placement.item].session
            inputs = [s.pending] if placement.phase == DECODE else s.prompt[s.done:s.done + placement.rows]
            s.phase, s.rows, s.produced = placement.phase, len(inputs), []
            packed.append((s.handle, s.done, len(inputs)))
            ids += inputs
            if placement.logits:
                logit_rows.append(len(ids) - 1)
            placed.append((placement, s, bool(placement.logits)))
        decode_only = all(placement.phase == DECODE for placement in layout.placements)
        with torch.inference_mode():
            if decode_only and len(placed) in self.graphs:
                logits = self._replay([s for _, s, _ in placed])
                self.decode_steps += 1
                if self.adapt_every and self.decode_steps % self.adapt_every == 0 and hasattr(self.experts, "adapt"):
                    self.experts.adapt()
            else:
                batch = StepBatch.build(packed, self.device)
                logits = self._forward(self._embed(ids), batch, logit_rows)
            # a prompt that ended this step: its routes warm the expert placement at the start of the next step (its
            # first decode step), so the swaps never lengthen a prompt step (a cancel settles between steps)
            if not decode_only and hasattr(self.experts, "prompt_warm") and any(
                    has and placement.phase != DECODE for placement, _, has in placed):
                self.warm_pending = True
            sampled = [(s, rows[placement.item].allowed) for placement, s, has in placed if has]
            params = [dict({k: s.request.sampling[k] for k in CORE_SAMPLING if k in s.request.sampling},
                           position=len(s.outputs), **({} if allowed is None else {"allowed": allowed}))
                      for s, allowed in sampled]
            tokens = self.sampler.sample(logits, params).tolist() if sampled else []
            results, j = [None] * len(rows), 0
            for placement, s, has in placed:
                logprobs = None
                if has:
                    token = s.request.replay[len(s.outputs)] if s.request.replay else tokens[j]
                    s.produced = [token]
                    if s.request.logprobs is not None:
                        row = torch.log_softmax(logits[j], -1)
                        top = torch.topk(row, min(max(int(s.request.logprobs), 1), 20))
                        logprobs = [(float(row[token]), list(zip(top.indices.tolist(), top.values.tolist())))]
                    j += 1
                results[placement.item] = StepResult(placement.rows if placement.phase == PREFILL else 0, s.produced,
                                                     has, 0, placement.rows, logprobs)
        if any(placement.phase == PREFILL and placement.logits for placement in layout.placements):
            torch.cuda.empty_cache()
        return results

    def _budgeted(self, rows):
        """Prompt rows capped so the step's predicted time stays under step_budget (crit serve settles a cancel only
        between steps): each prompt row in order takes what the budget has left, at least 16 tokens."""
        base, token, key = self.step_cost
        left = self.step_budget - base
        out = []
        for row in rows:
            if row.prefill_complete:
                out.append(row)
                left -= token
                continue
            ctx = row.session.done
            # largest n with n * (token + key * (ctx + n / 2)) <= left: the quadratic's positive root
            a, b = key / 2, token + key * ctx
            n = int(((b * b + 4 * a * left) ** 0.5 - b) / (2 * a)) if a > 0 and left > 0 else (
                int(left / b) if left > 0 else 0)
            n = max(16, min(row.prefill_tokens, n))
            if n < row.prefill_tokens:
                row = row._replace(prefill_tokens=n)
            left -= n * (token + key * (ctx + n / 2))
            out.append(row)
        return out

    def commit(self, decisions):
        for d in decisions:
            s = d.session
            if s.phase is None:
                continue
            if not d.cancelled:   # (crit 0.20 removed StepDecision.discard)
                s.done += s.rows
                if d.accepted:
                    s.pending = s.produced[d.accepted - 1]
                    s.outputs += s.produced[:d.accepted]
            self.kv_pool.keep(s.handle, s.done)
            s.phase, s.produced = None, []

    def retire(self, session, finish):
        self._close_pools(session.handle)

    def score(self, ids, first_target, path="", queries=None, top_k=0):
        torch = self.torch
        sequence = ("score", self.handles)
        self._open_pools(sequence)
        target, query, argmax, top = [], [] if queries else None, [], [] if top_k else None
        try:
            with torch.inference_mode():
                for start in range(0, len(ids) - 1, self.chunk):
                    end = min(len(ids) - 1, start + self.chunk)
                    batch = StepBatch.build([(sequence, start, end - start)], self.device)
                    positions = [p for p in range(start, end) if p >= first_target - 1]
                    logits = self._forward(self._embed(ids[start:end]), batch, [p - start for p in positions])
                    self.kv_pool.keep(sequence, end)
                    if not positions:
                        continue
                    logp = torch.log_softmax(logits, -1)
                    del logits
                    wanted = torch.tensor([ids[p + 1] for p in positions], device=self.device)
                    target += logp.gather(1, wanted[:, None])[:, 0].tolist()
                    argmax += logp.argmax(-1).tolist()
                    if queries:
                        for i, p in enumerate(positions):
                            row = queries[p - (first_target - 1)]
                            query.append(logp[i, torch.tensor(row, device=self.device)].tolist())
                    if top_k:
                        found = torch.topk(logp, top_k)
                        top += [list(zip(a, b)) for a, b in zip(found.indices.tolist(), found.values.tolist())]
                    del logp
        finally:
            self._close_pools(sequence)
        return Scores(target, query, argmax, top)

    def close(self):
        torch = self.torch
        torch.cuda.synchronize()
        self.graphs.clear()
        self.blocks.clear()
        self.head = self.final_norm = None
        self.arena = None
        self.kv_pool = self.rec = None
        self.source = None
        if hasattr(self.experts, "close"):
            self.experts.close()
        self.embed = None
        if hasattr(self.proj, "close"):
            self.proj.close()
        torch.cuda.empty_cache()


def _host_free():
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("available", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                        ("page_available", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                        ("virtual_available", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]
        status = Status()
        status.length = ctypes.sizeof(Status)
        if hasattr(ctypes, "windll") and ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.available)
    except Exception:  # noqa: BLE001
        pass
    try:
        import os
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 0


def create(context, options, parts):
    return Qwen35Model(context, options, parts)
