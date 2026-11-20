"""Gemma 4's mixture-of-experts text decoder as a model plugin (transformers' modeling_gemma4: Gemma4TextModel and
Gemma4ForCausalLM's head; llama.cpp's gemma4 graph for the GGUF's tensor names and layout).

Per layer: h = norm(x); q, k, v from h (a full-attention layer's values are its key projection: v = norm(k_raw)
without weight, k = rope(k_norm(k_raw))); query and key norms, RoPE (sliding: theta 1e4 over the whole 256-wide head;
full: proportional, theta 1e6 over the first quarter of the 512-wide head's pairs), attention with scale 1 over the
sliding window or the whole sequence; x += post_attn_norm(o_proj(a)). Then the dense GeGLU MLP over ffn_norm(x)
(post_ffw_norm_1 after it) beside the routed experts over pre_ffw_norm_2(x), routed by a softmax over
proj(rms_norm(x) / sqrt(hidden) * scale), top 8, renormalized, times the per-expert scale (post_ffw_norm_2 after
them); x += post_ffw_norm(mlp + experts); x *= layer_output_scale. The embedding is scaled by sqrt(hidden) (in BF16,
as transformers does) and tied to the head; logits = 30 tanh(logits / 30).

The residual stream is FP32; matrices go through the projections slot, norms and rotary positions through the norms
slot, routing through the router, experts through the experts slot, KV and attention through gemma4-attention's
pools. Each prompt chunk runs its matrices alone, decode rows 8 at a time, so a row's tokens never depend on the rows
beside it."""
import json
import math
from pathlib import Path

from crit_core.batch import DECODE, PREFILL, StepLayout, step_item
from crit_core.native import Scores, StepResult
from crit_plugins import StepBatch

def chunk_schedule(text, chunk):
    """Option chunk_schedule, "POSITION:ROWS,...": from each POSITION on a prompt chunk computes at most ROWS rows
    (prefill_chunk before the first), the positions rising and the rows falling, none above prefill_chunk."""
    schedule = []
    for item in (part.strip() for part in text.split(",")):
        if not item:
            continue
        first, rows = (int(value) for value in item.split(":"))
        last_first, last_rows = schedule[-1] if schedule else (-1, chunk)
        if first <= last_first or not 1 <= rows <= last_rows:
            raise ValueError(f"chunk_schedule {text!r}: positions rise and rows fall from prefill_chunk {chunk}")
        schedule.append((first, rows))
    return schedule


CORE_SAMPLING = ("temperature", "top_p", "top_k", "min_p", "seed")
# One capture stream per device for every engine the process builds: torch keeps a cuBLAS workspace per stream it
# multiplied on, so a new stream per build (torch.cuda.graph makes one when given none) would hold more device memory
# with each engine built again (crit serve, smoke's reopenings, an ablation).
_CAPTURE_STREAMS = {}
# The routed experts' side stream per device: a layer's experts run beside its dense MLP (forked from the step's
# stream and joined before the layer's tail), so the two branches' kernels fill each other's ramps and tails
_SIDE_STREAMS = {}
DECODE_ROWS = 8
SOFTCAP = 30.0


class Session:
    def __init__(self, handle, request):
        self.handle, self.request, self.prompt = handle, request, list(request.prompt)
        self.done, self.outputs, self.pending = 0, [], None
        self.phase, self.produced, self.rows = None, [], 0


class DecodeBatch:
    """A captured decode step's rows: one token each, every field a static device buffer (positions, the ring base and
    page table of each row), which gemma4-attention reads as decode_static."""
    round = False
    prompt = False

    def __init__(self, static, ring, page):
        self.decode_static = static
        self.positions_t = static["positions"]
        self.rows = static["ids"].shape[0]
        self.ring, self.page = ring, page

    @property
    def tokens(self):
        return self.rows

    def slots(self):
        s = self.decode_static
        ring_slots = s["base"] + s["positions"] % self.ring
        pages = s["table"].gather(1, (s["positions"] // self.page)[:, None])[:, 0].to(s["positions"].dtype)
        return ring_slots, pages * self.page + s["positions"] % self.page


class Tensor:
    """A checkpoint tensor: GGUF raw rows (kind, uint8 [rows][row_bytes]) or dense values (kind "dense")."""

    def __init__(self, kind, shape, data):
        self.kind, self.shape, self.data = kind, tuple(shape), data


HF_LAYER_NAMES = {
    "input_layernorm.weight": "attn_norm.weight", "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight", "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight", "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight", "post_attention_layernorm.weight": "post_attention_norm.weight",
    "pre_feedforward_layernorm.weight": "ffn_norm.weight", "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight", "mlp.down_proj.weight": "ffn_down.weight",
    "post_feedforward_layernorm_1.weight": "post_ffw_norm_1.weight",
    "pre_feedforward_layernorm_2.weight": "pre_ffw_norm_2.weight",
    "post_feedforward_layernorm_2.weight": "post_ffw_norm_2.weight",
    "post_feedforward_layernorm.weight": "post_ffw_norm.weight", "router.proj.weight": "ffn_gate_inp.weight",
    "router.scale": "ffn_gate_inp.scale", "router.per_expert_scale": "ffn_down_exps.scale",
    "experts.gate_up_proj": "ffn_gate_up_exps.weight", "experts.down_proj": "ffn_down_exps.weight",
    "layer_scalar": "layer_output_scale.weight"}


def hf_tensors(model_dir):
    """{gguf name: Tensor} of a transformers Gemma 4 safetensors checkpoint, dense, under the GGUF converter's names
    (llama.cpp's gemma4 tensor map), so the plugin's tests run it against transformers."""
    from safetensors.torch import load_file
    found = {}
    for path in sorted(Path(model_dir).glob("*.safetensors")):
        for name, value in load_file(str(path), device="cpu").items():
            short = name.split("language_model.", 1)[-1].removeprefix("model.")
            if short == "embed_tokens.weight":
                key = "token_embd.weight"
            elif short == "norm.weight":
                key = "output_norm.weight"
            elif short.startswith("layers."):
                _, layer, rest = short.split(".", 2)
                if rest not in HF_LAYER_NAMES:
                    continue
                key = f"blk.{layer}.{HF_LAYER_NAMES[rest]}"
            else:
                continue
            found[key] = Tensor("dense", value.shape, value)
    return found


def model_tensors(model_dir):
    """The checkpoint's tensors: its GGUF's stored rows, else a safetensors checkpoint's dense values."""
    if any(Path(model_dir).glob("*.gguf")):
        return gguf_tensors(model_dir)
    return hf_tensors(model_dir)


def gguf_tensors(model_dir):
    """{gguf name: Tensor} over the folder's GGUF, each a uint8 view of the mapped file ([rows][row_bytes], rows over
    every leading dimension), not yet copied anywhere."""
    import numpy as np
    import torch
    from crit_core.model import open_model
    source = open_model(model_dir)
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
    return found


class Gemma4Model:
    def __init__(self, context, options, parts, tensors=None):
        import torch
        self.torch, self.options = torch, options
        config = context.config.get("text_config") or context.config
        self.config = config
        self.layers, self.hidden = config["num_hidden_layers"], config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.kv_heads, self.head_dim = config["num_key_value_heads"], config["head_dim"]
        self.full_kv_heads = config.get("num_global_key_value_heads") or self.kv_heads
        self.full_head_dim = config.get("global_head_dim") or self.head_dim
        self.k_eq_v = bool(config.get("attention_k_eq_v"))
        self.eps, self.vocabulary = config["rms_norm_eps"], config["vocab_size"]
        self.window = config["sliding_window"]
        self.types = list(config["layer_types"])
        self.experts_count, self.top = config["num_experts"], config["top_k_experts"]
        self.inter, self.moe_inter = config["intermediate_size"], config["moe_intermediate_size"]
        self.softcap = config.get("final_logit_softcapping") or None
        self.device = torch.device("cuda")
        self.proj, self.attention, self.sampler = parts["projections"], parts["attention"], parts["sampler"]
        self.norms, self.router, self.experts = parts["norms"], parts["router"], parts["experts"]
        self.kv = parts["kv_cache"]
        self.context = int(context.context_tokens or options["context"])
        self.concurrency = int(options["concurrency"])
        self.overlap = bool(options.get("overlap", True))
        self.chunk = int(options["prefill_chunk"])
        self.schedule = chunk_schedule(options.get("chunk_schedule", ""), self.chunk)
        host_layers = {int(v) for v in options["host_experts"].split(",") if v.strip()}
        # an engine built again in the same process (crit serve, an ablation) finds the last one's memory in torch's
        # cache: hand it back before the free memory is read for the plan
        torch.cuda.empty_cache()

        source = tensors if tensors is not None else model_tensors(context.model)
        self.source = source
        # -- placement: every group by name, where the placement plugin puts it ------------------------------------
        ring = self.window + self.chunk
        sliding = [i for i, t in enumerate(self.types) if t == "sliding_attention"]
        full = [i for i, t in enumerate(self.types) if t != "sliding_attention"]
        bytes_of = {"int4": 0.5625, "fp8": 1, "bf16": 2}   # bytes a KV value (INT4: 4.5 bits)
        element = bytes_of[self.kv.dtype_name]
        full_element = bytes_of[getattr(self.kv, "full_dtype_name", self.kv.dtype_name)]
        sequences = self.concurrency + 1   # Core's sessions and one fixed-text score
        sizes = {}
        for name, t in source.items():
            layer = self._layer_of(name)
            group = (f"experts.{layer:02d}." if layer is not None and "_exps." in name and name.endswith(".weight")
                     else "weights.")
            sizes[group + name] = int(t.data.numel() * t.data.element_size())
        sizes["kv.full"] = int(len(full) * self.full_kv_heads * self.full_head_dim * full_element * (
            (-(-self.context // self.kv.page) + 8) * self.kv.page))
        sizes["kv.sliding"] = int(2 * len(sliding) * sequences * ring * self.kv_heads * self.head_dim * element)
        sizes["workspace"] = int(options["workspace_mib"]) << 20
        if host_layers:
            # gguf-experts' staging buffers: one host layer's experts on the device at a time
            sizes["staging.experts"] = max(
                int(source[f"blk.{i}.ffn_gate_up_exps.weight"].data.numel())
                + int(source[f"blk.{i}.ffn_down_exps.weight"].data.numel()) for i in host_layers)
        free, _ = torch.cuda.mem_get_info()
        budget = {"device": max(0, free - (int(options["reserve_mib"]) << 20)), "host": _host_free()}
        self.places = parts["placement"].plan(sizes, budget)
        for name, place in self.places.items():
            if name.startswith("experts."):
                layer = int(name.split(".")[1])
                if (place == "host") != (layer in host_layers):
                    raise ValueError(f"layer {layer}'s experts: the placement puts them in {place}, the model's "
                                     f"host_experts option {'names' if layer in host_layers else 'does not name'} it")
            elif place != "device":
                raise ValueError(f"{name}: the model keeps every group but routed experts on the device")

        # Every stored-block weight that lives on the device in one arena, each at a 256-byte aligned offset: separate
        # allocations would each round up to the allocator's granularity (about 0.3 GiB lost over ~360 tensors).
        block_names = [name for name, tensor in source.items() if tensor.kind not in ("dense", "F32", "F16", "BF16")
                       and not (name.endswith("_exps.weight") and self._layer_of(name) in host_layers)]
        arena_bytes = sum(-(-int(source[n].data.numel()) // 256) * 256 for n in block_names)
        self.arena = torch.empty(arena_bytes, dtype=torch.uint8, device=self.device) if block_names else None
        arena_at = [0]

        def on_device(name):
            t = source[name]
            if self.arena is None or t.kind in ("dense", "F32", "F16", "BF16"):
                return t.data.to(self.device)
            count = int(t.data.numel())
            view = self.arena[arena_at[0]:arena_at[0] + count].view(t.data.shape)
            view.copy_(t.data)
            arena_at[0] += -(-count // 256) * 256
            return view

        def host_values(name):
            """FP32 values of a small tensor on the host: F32 rows as stored, other kinds decoded on the CPU, so
            loading leaves no device temporaries between the weights (holes torch cannot hand back)."""
            t = source[name]
            if t.kind == "dense":
                return t.data.float().reshape(-1)
            if t.kind == "F32":
                return t.data.contiguous().view(torch.float32).reshape(-1).clone()
            from crit_plugins import device  # noqa: F401
            return self._decode(t, name).reshape(-1).cpu()

        def vector(name):
            return host_values(name).to(self.device)

        def matrix(name, role=None):
            t = source[name]
            if t.kind == "dense":
                return self.proj.load({"weight": t.data}, "bf16", role=role)
            return self.proj.load({"weight": on_device(name), "cols": t.shape[-1]}, "gguf:" + t.kind, role=role)

        def fused(names, role):
            """Matrices that multiply the same input, stacked by rows into one (one GEMV for all of them): a GGUF
            stack goes into the arena back to back (each part's rows whole blocks of the same kind and width),
            a dense stack is concatenated. Returns the matrix and each part's rows."""
            parts = [source[n] for n in names]
            rows = [int(part.shape[0]) for part in parts]
            if all(part.kind == "dense" for part in parts):
                return self.proj.load({"weight": torch.cat([part.data for part in parts])}, "bf16", role=role), rows
            kind, cols = parts[0].kind, parts[0].shape[-1]
            if any(part.kind != kind or part.shape[-1] != cols for part in parts) or self.arena is None:
                raise ValueError(f"{names}: parts of a fused matrix share one stored kind and width")
            total = sum(int(part.data.numel()) for part in parts)
            view = self.arena[arena_at[0]:arena_at[0] + total]
            at = 0
            for part in parts:
                count = int(part.data.numel())
                view[at:at + count].copy_(part.data.reshape(-1))
                at += count
            arena_at[0] += -(-total // 256) * 256
            return self.proj.load({"weight": view.view(sum(rows), -1), "cols": cols}, "gguf:" + kind, role=role), rows

        self.embed = matrix("token_embd.weight", "embedding")
        self.head = self.embed
        self.final_norm = vector("output_norm.weight")
        self.embed_scale = float(torch.tensor(self.hidden ** 0.5, dtype=torch.bfloat16))
        self.blocks = []
        for i in range(self.layers):
            p = f"blk.{i}."
            kind = self.types[i]
            b = {"kind": kind,
                 "attn_norm": vector(p + "attn_norm.weight"), "post_attn": vector(p + "post_attention_norm.weight"),
                 "o": matrix(p + "attn_output.weight", "o"),
                 "q_norm": vector(p + "attn_q_norm.weight"), "k_norm": vector(p + "attn_k_norm.weight"),
                 "ffn_norm": vector(p + "ffn_norm.weight"),
                 "down": matrix(p + "ffn_down.weight", "down"),
                 "post_ffw_1": vector(p + "post_ffw_norm_1.weight"), "pre_ffw_2": vector(p + "pre_ffw_norm_2.weight"),
                 "post_ffw_2": vector(p + "post_ffw_norm_2.weight"), "post_ffw": vector(p + "post_ffw_norm.weight"),
                 # the router as a BF16 matrix, decoded once (the GGUF stores it in F32)
                 "router": self.proj.load({"weight": host_values(p + "ffn_gate_inp.weight").view(
                     self.experts_count, self.hidden).to(torch.bfloat16)}, "bf16", role="router"),
                 "router_scale": vector(p + "ffn_gate_inp.scale") * (self.hidden ** -0.5),
                 "expert_scale": vector(p + "ffn_down_exps.scale"),
                 "out_scale": float(vector(p + "layer_output_scale.weight")[0])}
            has_v = (p + "attn_v.weight") in source
            if not has_v and not (kind != "sliding_attention" and self.k_eq_v):
                raise ValueError(f"layer {i} has no value projection")
            b["qkv"], split = fused([p + "attn_q.weight", p + "attn_k.weight"] + ([p + "attn_v.weight"] if has_v else []),
                                    "qkv")
            b["qkv_split"] = tuple(split) + ((0,) if not has_v else ())
            b["gate_up"], _ = fused([p + "ffn_gate.weight", p + "ffn_up.weight"], "gate_up")
            gate_up, down = source[p + "ffn_gate_up_exps.weight"], source[p + "ffn_down_exps.weight"]
            if i in host_layers:
                gu = _pinned(gate_up.data).view(self.experts_count, 2 * self.moe_inter, -1)
                dn = _pinned(down.data).view(self.experts_count, self.hidden, -1)
            else:
                gu = on_device(p + "ffn_gate_up_exps.weight").view(self.experts_count, 2 * self.moe_inter, -1)
                dn = on_device(p + "ffn_down_exps.weight").view(self.experts_count, self.hidden, -1)
            if gate_up.kind == "dense":
                for e in range(self.experts_count):
                    self.experts.load(i, e, gate_up.data[e], down.data[e], "bf16")
            else:
                self.experts.load_layer(i, gu, dn, gate_up.kind, down.kind, self.hidden, self.moe_inter)
            self.blocks.append(b)
            torch.cuda.empty_cache()

        # -- rotary frequencies ------------------------------------------------------------------------------------
        rope = config.get("rope_parameters") or {}
        sliding_theta = (rope.get("sliding_attention") or {}).get("rope_theta", 10000.0)
        full_rope = rope.get("full_attention") or {}
        full_theta = full_rope.get("rope_theta", 1000000.0)
        partial = full_rope.get("partial_rotary_factor", 1.0)
        d = self.head_dim
        self.sliding_freq = 1.0 / (sliding_theta ** (torch.arange(0, d, 2, device=self.device).float() / d))
        D = self.full_head_dim
        angles = int(partial * D // 2)
        rotated = 1.0 / (full_theta ** (torch.arange(0, 2 * angles, 2, device=self.device).float() / D))
        self.full_freq = torch.cat([rotated, torch.zeros(D // 2 - angles, device=self.device)])

        # -- KV pools ----------------------------------------------------------------------------------------------
        self.ring_pool = self.kv.pool(sliding, self.kv_heads, self.head_dim, self.context, self.device,
                                             window=self.window, ring=ring, sequences=sequences)
        derive = {i: self.blocks[i]["k_norm"] for i in full}
        self.full_pool = self.kv.pool(full, self.full_kv_heads, self.full_head_dim, self.context * 1,
                                             self.device, derive=derive, inv_freq=self.full_freq)
        generation = Path(context.model) / "generation_config.json"
        eos = json.loads(generation.read_text(encoding="utf-8")).get("eos_token_id") if generation.is_file() else None
        self.stops = sorted(set(eos if isinstance(eos, list) else [eos])) if eos is not None else []
        self.handles = 0
        self.embed_scale_t = torch.tensor(self.embed_scale, dtype=torch.bfloat16, device=self.device)
        self.graphs = {}
        self.capture_bytes = 0
        kernel_dims = (64, 128, 256, 512)
        if (options["decode_graphs"] and getattr(self.attention, "kernels", None) is not None
                and self.head_dim in kernel_dims and self.full_head_dim in kernel_dims):
            before = torch.cuda.memory_allocated(self.device)
            self._capture_all()
            # the captures' memory (graph pools, warm-up buffers) is planned as its own entry, as Core's round driver
            # plans "rounds:captures"
            taken = max(0, torch.cuda.memory_allocated(self.device) - before)
            free, _ = torch.cuda.mem_get_info()
            self.places.update(parts["placement"].plan({"decode:captures": int(taken)}, {"device": int(free) + int(taken),
                                                                                         "host": 0}))
            self.capture_bytes = int(taken)
        if not host_layers:
            # Every matrix lives on the device now: drop the checkpoint's memory map, whose pages the load read would
            # otherwise stay resident in the process (about 10 GB of host memory for a 14 GB GGUF). Host expert layers
            # keep it: their blocks are read from it.
            self.source = source = None
        torch.cuda.empty_cache()

    # -- captured decode steps -----------------------------------------------------------------------------------------
    def _capture_all(self):
        """One CUDA graph of the whole decode forward per row count 1 .. concurrency, over static buffers: the rows'
        tokens, positions, ring bases and page tables. Captured at load on dummy sequences, after two eager runs warm
        every lazily made table; replayed from step() for a step of decode rows only."""
        torch = self.torch
        dummy = [("capture", r) for r in range(self.concurrency)]
        for sequence in dummy:
            self._open_pools(sequence)
            self.full_pool.ensure(sequence, 1)
        try:
            with torch.inference_mode():
                for rows in range(1, self.concurrency + 1):
                    static = {"ids": torch.zeros(rows, dtype=torch.int64, device=self.device),
                              "positions": torch.zeros(rows, dtype=torch.int64, device=self.device),
                              "base": torch.tensor([self.ring_pool.base(s) for s in dummy[:rows]], dtype=torch.int64,
                                                   device=self.device),
                              "table": torch.stack([self.full_pool.table_row(s) for s in dummy[:rows]]).to(self.device),
                              "host": torch.zeros((rows, 3 + self.full_pool.table_width), dtype=torch.int64,
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
        batch = DecodeBatch(static, self.ring_pool.ring, self.full_pool.page)
        return self._forward(static["ids"], batch, None, slots=batch.slots())

    def _replay(self, sessions):
        """Fill the row count's static buffers from the sessions (pending token, position, ring base, page table)
        and replay its graph: logits [rows][vocabulary] FP32."""
        torch = self.torch
        graph, static = self.graphs[len(sessions)]
        host = static["host"]
        width = self.full_pool.table_width
        for r, s in enumerate(sessions):
            self.full_pool.ensure(s.handle, s.done + 1)
            host[r, 0], host[r, 1], host[r, 2] = s.pending, s.done, self.ring_pool.base(s.handle)
            table = self.full_pool.tables[s.handle]
            host[r, 3:3 + len(table)] = torch.tensor(table, dtype=torch.int64)
        device = host.to(self.device, non_blocking=True)
        static["ids"].copy_(device[:, 0])
        static["positions"].copy_(device[:, 1])
        static["base"].copy_(device[:, 2])
        static["table"].copy_(device[:, 3:3 + width].to(torch.int32))
        graph.replay()
        return static["logits"]

    @staticmethod
    def _layer_of(name):
        if name.startswith("blk."):
            return int(name.split(".")[1])
        return None

    def _decode(self, t, name):
        from crit_plugins import device  # noqa: F401  (the torch device helpers load torch once)
        matrix = self.proj.load({"weight": t.data.to(self.device), "cols": t.shape[-1]}, "gguf:" + t.kind)
        return self.proj.rows(matrix, self.torch.arange(matrix.rows, device=self.device))

    def info(self):
        return {"implementation": "gemma4 MoE decoder over GGUF blocks (torch device plugins)",
                "vocabulary": self.vocabulary, "context": self.context, "prefill_chunk": self.chunk,
                "max_concurrency": self.concurrency, "max_pending": 8, "max_batch_tokens": self.chunk,
                "max_row_tokens": 1, "decode_token_work": 1, "stop_tokens": self.stops,
                "sampling": dict.fromkeys(CORE_SAMPLING, True), "token_logprobs": True, "max_top_logprobs": 20,
                "scoring": True, "query_scoring": True, "score_paths": ["prefill"], "replay": True,
                "speculative": False, "plain": False, "decode_graphs": sorted(self.graphs), "decode_capture_bytes": self.capture_bytes,
                # the GGUF groups' prompt activations as the projections plugin serves them (its facts; FP8 if it
                # declares none)
                "prompt_activations": (self.proj.facts() if hasattr(self.proj, "facts") else {}).get(
                    "prompt_activations", {"gguf": "fp8"}),
                "host_experts": sorted(int(n.split(".")[1]) for n, p in self.places.items()
                                       if n.startswith("experts.") and p == "host" and n.count(".") >= 2)
                if hasattr(self, "places") else []}

    # -- the arithmetic --------------------------------------------------------------------------------------------------
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

    def _side_stream(self):
        torch = self.torch
        index = self.device.index or 0
        stream = _SIDE_STREAMS.get(index)
        if stream is None:
            stream = _SIDE_STREAMS[index] = torch.cuda.Stream(self.device)
        return stream

    def _mm(self, x, matrix, groups, out_dtype=None):
        if len(groups) == 1:
            return self.proj.matmul(x, matrix, out_dtype)
        return self.torch.cat([self.proj.matmul(x[first:end], matrix, out_dtype) for first, end in groups])

    def _route(self, x, b, groups, r=None):
        """Each token's experts and weights; r, when given, is the router's input already normalized and scaled."""
        torch, norms = self.torch, self.norms
        if r is None:
            r = (norms.rms_norm(x, None, self.eps, out_dtype=torch.float32) * b["router_scale"]).to(torch.bfloat16)
        fused = getattr(self.router, "route_fused", None)
        if fused is not None:
            # the router plugin's one-kernel routing (logits, scores, selection, weights), where it takes the shapes
            routed = fused(r, b["router"].dense, self.top, b["expert_scale"])
            if routed is not None:
                return routed
        logits = self._mm(r, b["router"], groups, torch.float32)
        ids, weights = self.router.select(logits, self.top)
        return ids, weights * b["expert_scale"][ids]

    def _experts(self, i, h, ids, weights, groups):
        if len(groups) == 1:
            return self.experts.forward(i, h, ids, weights)
        return self.torch.cat([self.experts.forward(i, h[a:e], ids[a:e], weights[a:e]) for a, e in groups])

    def _forward(self, ids, batch, logit_rows, slots=None):
        torch, norms, t, eps = self.torch, self.norms, len(ids), self.eps
        bf16 = torch.bfloat16
        groups = self._groups(batch)
        x = (self.proj.rows(self.embed, ids).to(bf16) * self.embed_scale_t).float()
        if slots is None:
            ring_slots, full_slots = self.ring_pool.reserve(batch), self.full_pool.reserve(batch)
        else:
            ring_slots, full_slots = slots
        positions = batch.positions_t
        # the norms plugin's fused forms where it has them: the attention output joins the residual in one kernel,
        # the three norms of the residual after attention share one sum of squares, and the layer's tail writes the
        # next layer's input norm (the same arithmetic as the separate calls, fewer launches)
        fused = hasattr(norms, "ffn_out") and norms.fused_ready(x)
        h = norms.rms_norm(x, self.blocks[0]["attn_norm"], eps, out_dtype=bf16)
        for i, b in enumerate(self.blocks):
            if i and not fused:
                h = norms.rms_norm(x, b["attn_norm"], eps, out_dtype=bf16)
            if b["kind"] == "sliding_attention":
                hd, kvh, freq, pool, slots = self.head_dim, self.kv_heads, self.sliding_freq, self.ring_pool, ring_slots
            else:
                hd, kvh, freq, pool, slots = (self.full_head_dim, self.full_kv_heads, self.full_freq, self.full_pool,
                                              full_slots)
            nq, nk, nv = b["qkv_split"]
            # the projections' outputs in FP32 into the norm kernels (BF16 outputs on prompt steps measured further
            # from the FP32 reference: mean 2.47 against 2.28 nats a token over 200 tokens)
            wide = torch.float32
            qkv = self._mm(h, b["qkv"], groups, wide)
            q = qkv[:, :nq].view(t, self.heads, hd)
            q = norms.rms_norm_rope(q, b["q_norm"], eps, positions, freq, out_dtype=bf16)
            k_raw = qkv[:, nq:nq + nk].view(t, kvh, hd)
            if nv:
                v = norms.rms_norm_heads(qkv[:, nq + nk:].view(t, kvh, hd), None, eps, out_dtype=bf16)
            else:
                v = norms.rms_norm_heads(k_raw, None, eps, out_dtype=bf16)
            if pool is self.full_pool:
                pool.write(i, slots, None, v)
            else:
                k = norms.rms_norm_rope(k_raw, b["k_norm"], eps, positions, freq, out_dtype=bf16)
                pool.write(i, slots, k, v)
            a = self.attention.forward(i, q, pool, batch, scale=1.0)
            o = self._mm(a.reshape(t, -1), b["o"], groups, torch.float32 if fused else None)
            if fused:
                norms.add_rms(x, o, b["post_attn"], eps)
                hm, h2, r = norms.rms3(x, b["ffn_norm"], b["pre_ffw_2"], b["router_scale"], eps)
            else:
                x = x + norms.rms_norm(o, b["post_attn"], eps, out_dtype=torch.float32)
                hm = norms.rms_norm(x, b["ffn_norm"], eps, out_dtype=bf16)
                h2 = norms.rms_norm(x, b["pre_ffw_2"], eps, out_dtype=bf16)
                r = None
            # the dense MLP and the routed experts, both from the residual after attention: the experts on the side
            # stream beside the dense MLP where overlap is on (joined before the tail reads both)
            # (decode-sized steps only: a prompt chunk's branches are long enough on their own, and the side stream's
            # allocations would sit in a pool of their own beside the main one at the prompt's peak)
            side = self._side_stream() if self.overlap and x.is_cuda and t <= self.concurrency else None
            if side is not None:
                main = torch.cuda.current_stream(x.device)
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    ids_e, w_e = self._route(x, b, groups, r)
                    e = self._experts(i, h2, ids_e, w_e, groups)
            gate_up = self._mm(hm, b["gate_up"], groups, wide)
            mlp = self._mm(norms.gelu_mul(gate_up, approximate="tanh", out_dtype=bf16), b["down"], groups,
                           wide if fused else None)
            if side is not None:
                main.wait_stream(side)
            else:
                ids_e, w_e = self._route(x, b, groups, r)
                e = self._experts(i, h2, ids_e, w_e, groups)
            if fused:
                following = self.blocks[i + 1]["attn_norm"] if i + 1 < len(self.blocks) else self.final_norm
                h = norms.ffn_out(x, mlp, e, b["post_ffw_1"], b["post_ffw_2"], b["post_ffw"], b["out_scale"], eps,
                                  following)
            else:
                m1 = norms.rms_norm(mlp, b["post_ffw_1"], eps, out_dtype=torch.float32)
                m2 = norms.rms_norm(e, b["post_ffw_2"], eps, out_dtype=torch.float32)
                x = (x + norms.rms_norm(m1 + m2, b["post_ffw"], eps, out_dtype=torch.float32)) * b["out_scale"]
        if not fused:
            h = norms.rms_norm(x, self.final_norm, eps, out_dtype=bf16)
        if logit_rows is None:
            rows = list(range(t))
        else:
            rows = logit_rows
        if not rows:
            return torch.zeros(0, self.vocabulary, dtype=torch.float32, device=self.device)
        if logit_rows is None:
            hs = h
        else:
            hs = h[torch.tensor(rows, dtype=torch.int64, device=self.device)]
        # each logits row alone in its group of DECODE_ROWS: a prompt's last row and a decode row multiply alike
        logits = torch.cat([self.proj.matmul(hs[a:a + DECODE_ROWS], self.head, torch.float32)
                            for a in range(0, len(rows), DECODE_ROWS)])
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits

    # -- Core's step model -----------------------------------------------------------------------------------------------
    def _open_pools(self, sequence):
        self.ring_pool.open(sequence)
        self.full_pool.open(sequence)

    def _close_pools(self, sequence):
        self.ring_pool.close(sequence)
        self.full_pool.close(sequence)

    def open(self, request):
        self.handles += 1
        session = Session(self.handles, request)
        self._open_pools(session.handle)
        return session

    def rows_at(self, position):
        """The most prompt rows a chunk starting at `position` computes: prefill_chunk, or less from the positions
        chunk_schedule names on (a prompt step's attention grows with the positions it reads)."""
        rows = self.chunk
        for first, limit in self.schedule:
            if position >= first:
                rows = limit
        return rows

    def step(self, rows):
        torch = self.torch
        layout = StepLayout(step_item(row, len(row.session.prompt), row.session.done,
                                      stop=row.session.done + self.rows_at(row.session.done)) for row in rows)
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
            else:
                batch = StepBatch.build(packed, self.device)
                logits = self._forward(torch.tensor(ids, dtype=torch.int64, device=self.device), batch, logit_rows)
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
            # a prompt finished: hand its temporaries' cached blocks back, so decode keeps the device's headroom
            torch.cuda.empty_cache()
        return results

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
            self.ring_pool.keep(s.handle, s.done)
            self.full_pool.keep(s.handle, s.done)
            s.phase, s.produced = None, []

    def retire(self, session, finish):
        self._close_pools(session.handle)

    def score(self, ids, first_target, path="", queries=None, top_k=0):
        """Each target's log-probability, its query rows', the argmax and the ranked top_k, a prompt chunk at a time:
        every chunk's logits rows are reduced to those as soon as they are made, so a long text never holds a row of
        the vocabulary per position."""
        torch = self.torch
        sequence = ("score", self.handles)
        self._open_pools(sequence)
        target, query, argmax, top = [], [] if queries else None, [], [] if top_k else None
        try:
            with torch.inference_mode():
                bounds, start = [], 0
                while start < len(ids) - 1:
                    bounds.append((start, min(len(ids) - 1, start + self.rows_at(start))))
                    start = bounds[-1][1]
                for start, end in bounds:
                    batch = StepBatch.build([(sequence, start, end - start)], self.device)
                    positions = [p for p in range(start, end) if p >= first_target - 1]
                    chunk = torch.tensor(ids[start:end], dtype=torch.int64, device=self.device)
                    logits = self._forward(chunk, batch, [p - start for p in positions])
                    self.ring_pool.keep(sequence, end)
                    self.full_pool.keep(sequence, end)
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
        """Drop every device tensor the model holds (graphs and their pools, KV pools, matrices), then hand torch's
        cached memory back, so an engine built next in the process finds the device as this one found it."""
        torch = self.torch
        torch.cuda.synchronize()
        self.graphs.clear()
        self.blocks.clear()
        self.embed = self.head = self.final_norm = None
        self.arena = None
        self.ring_pool = self.full_pool = None
        self.source = None
        torch.cuda.empty_cache()


def _pinned(data):
    import torch
    out = torch.empty(data.shape, dtype=data.dtype, pin_memory=True)
    out.copy_(data)
    return out


def _host_free():
    """Host memory available for pinned groups, in bytes (0 where it cannot be read)."""
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
    return Gemma4Model(context, options, parts)
