"""Qwen3.5's hybrid dense decoder as a model plugin on a GPU (Qwen3_5ForConditionalGeneration's language model, or a
Qwen3_5ForCausalLM checkpoint): gated-delta layers (a causal convolution over q, k and v, then the delta rule, both in
the recurrence slot; the gated RMSNorm over SiLU(z)) beside gated full-attention layers (each head's query and output
gate from one projection, per-head RMSNorms with partial rotary positions, keys and values in the kv_cache slot's pool,
attention by the attention slot, a sigmoid output gate), and a dense SwiGLU MLP in every layer. Every matrix runs as
the checkpoint stores it (bf16, fp8 per channel or tensor, nvfp4) in the projections slot, matrices that read the same
input stacked into one (q|k|v, qkv|z, b|a, gate|up); the embedding rows live in the embeddings slot (host memory under
device-placement's off_device "tables."); the zero-centred RMSNorms (1 + weight, each residual add fused into the norm
after it), the query/key norms with their rotary positions, the gated norm, the output gate and the SiLU product are the
norms slot's; every token comes from the sampler.

A step of many sequences' rows is one batch: a prompt chunk's matrices are multiplied alone and one-token rows
`invariant_rows(projections)` at a time, so a row's bits are those it has alone whatever shares the step. Once a
request's prompt is in, it decodes in Core's captured rounds (crit_plugins.rounds.Rounds, mode "sampled": a greedy
request is a row at temperature 0, which the sampler's round law takes as its argmax), which Core launches natively;
replays, constrained (plain) requests and rows the rounds cannot hold decode through step().

With the option `mtp` the checkpoint's multi-token-prediction layer (model_mtp.safetensors: mtp.fc over the normed
embedding of the next token and the normed target hidden state, one gated-attention decoder layer with its own KV pool,
mtp.norm, the target's lm_head) is the model's device draft head (draft:head-device, MtpHead): each round it catches up
on the previous round's emitted tokens with the target's hidden states of those verify columns (a fixed widest + 1
columns masked by position, EAGLE-style), then chains its own output hidden state one draft per depth. Its KV for every
prompt position is written while the prompt is prefilled (its fused input, input norm and k/v projections only)."""
import json

from crit_core.batch import DECODE, DEFERRED, PREFILL, READY, PageReservations, StepLayout, step_item
from crit_core.native import Scores, StepResult
from crit_plugins import StepBatch
from crit_plugins.device import RoundBatch, free_device_bytes, invariant_rows, release_cached_under

CORE_SAMPLING = ("temperature", "top_p", "top_k", "min_p", "seed")
STOPS = 8   # stop ids a row in a round slot holds
TOP = 20    # the alternatives a token's log-probabilities carry at most (max_top_logprobs)
SCORE_SLICE = 128   # rows of logits a score holds at once (FP32 [rows][vocabulary])
FREE_BELOW = 1536 << 20   # torch's cached blocks go back when a prompt or score ends with less device memory free


class Session:
    def __init__(self, handle, request):
        self.handle, self.request, self.prompt = handle, request, list(request.prompt)
        self.done, self.outputs, self.pending = 0, [], None    # rows committed; published outputs
        self.phase, self.produced, self.rows = None, [], 0
        self.slot, self.pages, self.last_hidden, self.mtp_done = None, 0, None, 0
        self.graphed = False   # its last step's prompt chunk ran as a captured graph (its pools settled already)


def _host_available():
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class Checkpoint:
    """The checkpoint's safetensors headers ({name: TensorInfo}) and its tensors read one at a time onto a device."""

    def __init__(self, model):
        from safetensors import safe_open
        from crit_core.model.safetensors import inspect_safetensors
        self.headers, self.files = {}, {}
        for path in sorted(model.glob("*.safetensors")):
            found = inspect_safetensors(path, header_only=True)
            tensors = found.tensors
            self.headers.update(tensors if isinstance(tensors, dict) else {t.name: t for t in tensors})
            self.files[str(path)] = safe_open(str(path), "pt", device="cpu")

    def __contains__(self, name):
        return name in self.headers

    def nbytes(self, name):
        return self.headers[name].nbytes

    def get(self, name, device):
        info = self.headers[name]
        return self.files[str(info.shard)].get_tensor(name).to(device)


class MtpHead:
    """The model's device draft head (draft:head-device) over the checkpoint's MTP layer, for captured rounds of up to
    `rows` rows at up to `widest` drafts. features[r][c] is the target's final-normed hidden state of the last round's
    verify column c of row r (the seed's: the prompt's last row), which the catch-up reads with the emitted token c at
    position frontier - n + c; hidden[r] is the chain's last MTP output."""

    def __init__(self, model, rows, widest):
        torch = model.torch
        self.m, self.torch, self.rows, self.widest, self.cols = model, torch, rows, widest, widest + 1
        bf16 = dict(dtype=torch.bfloat16, device=model.device)
        self.features = torch.zeros(rows, self.cols, model.hidden, **bf16)
        self.hidden = torch.zeros(rows, model.hidden, **bf16)
        self.catch = RoundBatch(rows, widest, model.device)
        self.chain = RoundBatch(rows, 0, model.device)

    def seed_feature(self, slot, hidden):
        self.features[slot, 0].copy_(hidden)

    def move(self, source, target):
        self.features[target].copy_(self.features[source])
        self.hidden[target].copy_(self.hidden[source])

    def keep_features(self, view, hidden):
        """The verify's hidden states of every column ([rows][width + 1][hidden]), kept where the row decoded (a held
        row keeps the features its next catch-up ingests again)."""
        rows, cols = view.rows, view.width + 1
        held = (view.hold[:rows] > 0)[:, None, None]
        self.features[:rows, :cols] = self.torch.where(held, self.features[:rows, :cols], hidden)

    def catch_up(self, view):
        torch, rows, cols, h = self.torch, view.rows, self.cols, self.m.hidden
        count = view.catch_up_count[:rows]
        base = view.frontier[:rows] - count
        batch = self.catch.fill(rows, self.widest, base, view.catch_up_tokens[:rows])
        out = self.m._mtp(batch.tokens_t, self.features[:rows].reshape(rows * cols, h), batch)
        index = (count - 1).clamp(min=0)[:, None, None].expand(rows, 1, h)
        self.hidden[:rows] = out.view(rows, cols, h).gather(1, index).squeeze(1)

    def logits(self, depth, tokens, view, parents=None):
        rows = view.rows
        if depth:
            batch = self.chain.fill(rows, 0, view.frontier[:rows] + (depth - 1), tokens[:rows, None])
            self.hidden[:rows] = self.m._mtp(batch.tokens_t, self.hidden[:rows], batch)
        return self.m._logits_rows(self.hidden[:rows], draft=True)

    def commit_round(self, view):
        """Nothing beyond the MTP pool's own commit: its positions are rewritten by the next catch-up."""


class Qwen35Model:
    def __init__(self, context, options, parts):
        import torch
        from crit_core.model.schemes import matrix_parts
        self.torch, self.options = torch, options
        self.matrix_parts = matrix_parts
        top = context.config
        config = top.get("text_config") or top
        self.quant = top.get("quantization_config") or config.get("quantization_config") or {}
        self.types = list(config["layer_types"])
        self.layers, self.hidden = len(self.types), config["hidden_size"]
        self.heads, self.kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
        self.head_dim = config.get("head_dim") or self.hidden // self.heads
        self.nk, self.nv = config["linear_num_key_heads"], config["linear_num_value_heads"]
        self.dk, self.dv = config["linear_key_head_dim"], config["linear_value_head_dim"]
        self.kernel = config["linear_conv_kernel_dim"]
        self.eps, self.vocabulary = config["rms_norm_eps"], config["vocab_size"]
        rope = config.get("rope_parameters") or {}
        factor = rope.get("partial_rotary_factor", config.get("partial_rotary_factor", 1.0))
        self.rotary = int(self.head_dim * factor)
        theta = rope.get("rope_theta") or config.get("rope_theta", 10000.0)
        self.device = torch.device("cuda")
        self.linear = [i for i, t in enumerate(self.types) if t == "linear_attention"]
        self.full = [i for i, t in enumerate(self.types) if t == "full_attention"]
        self.context = int(context.context_tokens or options["context"])
        self.concurrency, self.pending_max = int(options["concurrency"]), int(options["pending"])
        self.chunk = int(options["prefill_chunk"])
        self.free_below = int(options["release_below_mib"]) << 20
        self.sequences = self.concurrency + 1   # every request decoding at once, and a score's
        self.proj, self.attention, self.sampler = parts["projections"], parts["attention"], parts["sampler"]
        self.recurrence, self.norms, self.embeddings = parts["recurrence"], parts["norms"], parts["embeddings"]
        self.drafter = parts.get("drafter")
        self.fused = "norms:fused" in parts.facts("norms")
        ck = self.ckpt = Checkpoint(context.model)
        self.mtp = bool(options["mtp"]) and self.drafter is not None and "mtp.fc.weight" in ck
        if self.drafter is not None and not self.mtp:
            raise ValueError("qwen35-model drafts from the checkpoint's MTP layer: the selection has a drafter, so set "
                             "option mtp and give a checkpoint with mtp.* tensors, or leave the drafter slot empty")
        self.width = int(options["width"]) if self.mtp else 0

        # -- names ---------------------------------------------------------------------------------------------------
        anchor = "layers.0.input_layernorm.weight"
        prefixes = [n[:-len(anchor)] for n in ck.headers if n.endswith(anchor) and not n.startswith("mtp")]
        if not prefixes:
            raise ValueError("qwen35-model: no decoder layer 0 in the checkpoint")
        self.prefix = prefix = sorted(prefixes, key=len)[0]   # "model.language_model." or "model."
        head = "lm_head" if "lm_head.weight" in ck or "lm_head.weight_packed" in ck else None
        embed_name = prefix + "embed_tokens.weight"
        kept = [n for n in ck.headers if n.startswith(prefix + "layers.") or n == prefix + "norm.weight"
                or (head and n.startswith("lm_head.")) or (self.mtp and n.startswith("mtp."))]
        self.kv_dtype = options["kv"]
        scales = all(f"{prefix}layers.{i}.self_attn.k_scale" in ck and f"{prefix}layers.{i}.self_attn.v_scale" in ck
                     for i in self.full)
        if self.kv_dtype == "auto":
            self.kv_dtype = "fp8" if scales else "bf16"
        if self.kv_dtype == "fp8" and not scales:
            raise ValueError("qwen35-model: fp8 KV pages take the checkpoint's static k_scale and v_scale, which it "
                             "does not store: choose kv bf16 or fp8:row")

        # -- the plan --------------------------------------------------------------------------------------------------
        page = 16
        self.kv_tokens = self.context + int(options["kv_margin"])
        pages = -(-self.kv_tokens // page)

        def pool_bytes(layers, dtype):
            element = {"bf16": 2, "fp8": 1, "fp8:row": 1}[dtype]
            found = 2 * layers * pages * page * self.kv_heads * self.head_dim * element
            if dtype == "fp8:row":
                found += 2 * layers * pages * page * self.kv_heads * 4
            return found

        kv_bytes = pool_bytes(len(self.full), self.kv_dtype)
        if self.mtp:
            kv_bytes += pool_bytes(1, options["mtp_kv"])
        conv = 2 * self.nk * self.dk + self.nv * self.dv
        n, cap = self.sequences + 1, self.width + 1
        state_bytes = len(self.linear) * n * (self.nv * self.dk * self.dv + conv * (self.kernel - 1)
                                              + 2 * conv * cap + 2 * cap * (self.nk * self.dk + self.nv
                                                                            + self.nv * self.dv)) * 4
        state_bytes = int(state_bytes * 1.05) + (32 << 20)
        tensors = {name: ck.nbytes(name) for name in kept}
        tensors["tables.embed_tokens"] = ck.nbytes(embed_name)
        tensors.update({"kv": kv_bytes, "states": state_bytes, "activations": int(options["activations_mib"]) << 20})
        if self.mtp and options["draft_head"] != "shared":
            tensors["draft_head"] = ck.nbytes("lm_head.weight") // 2 if options["draft_head"] == "nvfp4" else \
                ck.nbytes("lm_head.weight")
        free = free_device_bytes()   # after the load's own temporaries go back to the device
        placed = parts["placement"].plan(tensors, {"device": free, "host": _host_available()})
        if any(place != "device" for name, place in placed.items() if not name.startswith("tables.")):
            raise ValueError("qwen35-model keeps every weight, the KV and the states on the device")

        # -- the embedding rows (first, while the device has the room for the table in transit) -------------------------
        rows = ck.get(embed_name, self.device).to(torch.bfloat16)
        self.embed = self.embeddings.store(rows)
        del rows

        # -- the layers --------------------------------------------------------------------------------------------------
        on = lambda name: ck.get(name, self.device)  # noqa: E731
        vec = lambda name: on(name).float()  # noqa: E731
        self.final_norm = vec(prefix + "norm.weight")
        self.head = self._matrix(["lm_head" if head else prefix + "embed_tokens"], "head")
        self.blocks = []
        for i, kind in enumerate(self.types):
            p = f"{prefix}layers.{i}."
            b = {"in": vec(p + "input_layernorm.weight"), "post": vec(p + "post_attention_layernorm.weight"),
                 "gate_up": self._matrix([p + "mlp.gate_proj", p + "mlp.up_proj"], "gate_up"),
                 "down": self._matrix([p + "mlp.down_proj"], "down")}
            if kind == "linear_attention":
                a = p + "linear_attn."
                b.update(qkvz=self._matrix([a + "in_proj_qkv", a + "in_proj_z"], "qkvz"),
                         ba=self._matrix([a + "in_proj_b", a + "in_proj_a"], "ba"),
                         out=self._matrix([a + "out_proj"], "linear_out"),
                         conv=on(a + "conv1d.weight")[:, 0].float().contiguous(),
                         a_neg=vec(a + "A_log").exp(), dt_bias=vec(a + "dt_bias"), norm=vec(a + "norm.weight"))
            else:
                s = p + "self_attn."
                b.update(qkv=self._matrix([s + "q_proj", s + "k_proj", s + "v_proj"], "qkv"),
                         o=self._matrix([s + "o_proj"], "o"),
                         q_norm=vec(s + "q_norm.weight"), k_norm=vec(s + "k_norm.weight"))
                if self.kv_dtype == "fp8":
                    b["kv_scales"] = (float(on(s + "k_scale").float()), float(on(s + "v_scale").float()))
            self.blocks.append(b)
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.rotary, 2, device=self.device).float() / self.rotary))

        # -- the MTP layer (draft-only: its matrices may be re-encoded, the stored target verifies every draft) ----------
        self.mtp_block, self.draft_matrix = None, self.head
        if self.mtp:
            encode = None if options["mtp_encode"] == "bf16" else options["mtp_encode"]
            m = "mtp.layers.0."
            self.mtp_block = {
                "norm_e": vec("mtp.pre_fc_norm_embedding.weight"), "norm_h": vec("mtp.pre_fc_norm_hidden.weight"),
                "fc": self._matrix(["mtp.fc"], "mtp_fc", encode), "in": vec(m + "input_layernorm.weight"),
                "post": vec(m + "post_attention_layernorm.weight"), "final": vec("mtp.norm.weight"),
                "qkv": self._matrix([m + "self_attn.q_proj", m + "self_attn.k_proj", m + "self_attn.v_proj"],
                                    "mtp_qkv", encode),
                "o": self._matrix([m + "self_attn.o_proj"], "mtp_o", encode),
                "q_norm": vec(m + "self_attn.q_norm.weight"), "k_norm": vec(m + "self_attn.k_norm.weight"),
                "gate_up": self._matrix([m + "mlp.gate_proj", m + "mlp.up_proj"], "mtp_gate_up", encode),
                "down": self._matrix([m + "mlp.down_proj"], "mtp_down", encode)}
            if options["draft_head"] != "shared":
                # a draft-only copy of the head, re-encoded in row slices: each slice its own matrix, so no shared
                # workspace is sized for the whole vocabulary
                self.draft_matrix = self._head_slices(int(options["draft_head_slices"]), options["draft_head"])
        self.decode_rows = invariant_rows(self.proj)
        pfacts = self.proj.facts() if callable(getattr(self.proj, "facts", None)) else {}
        self.prompt_exact = bool((pfacts.get("row_invariance") or {}).get("prompt_exact"))

        # -- the pools ---------------------------------------------------------------------------------------------------
        self.pool = parts["kv_cache"].pool(self.full, self.kv_heads, self.head_dim, self.kv_tokens, self.device,
                                           self.kv_dtype, page=page)
        if self.kv_dtype == "fp8":
            for i in self.full:
                self.pool.set_scales(i, *self.blocks[i]["kv_scales"])
        self.mtp_layer = self.layers   # the MTP layer's id in its own pool
        self.mtp_pool = parts["kv_cache"].pool([self.mtp_layer], self.kv_heads, self.head_dim, self.kv_tokens,
                                               self.device, options["mtp_kv"], page=page) if self.mtp else None
        self.states = self.recurrence.pool(self.linear, self.nv, self.dk, self.dv, self.device, self.sequences,
                                           widest=self.width, conv=(conv, self.kernel), key_heads=self.nk)
        self.geometry = {"keys": self.dk, "values": self.dv, "epsilon": 1e-6, "query_scale": self.dk ** -0.5,
                         "normalize_keys": True, "normalize_queries": True}
        self.reservations = PageReservations(pages, page, self.context, verify_rows=self.width + 1)
        generation = context.model / "generation_config.json"
        eos = json.loads(generation.read_text(encoding="utf-8")).get("eos_token_id") if generation.is_file() else None
        if eos is None:
            eos = config.get("eos_token_id")
        self.stops = sorted(set(eos if isinstance(eos, list) else [eos])) if eos is not None else []
        self.handles = 0
        self.ckpt = None   # the file handles go: every tensor is on the device

        # -- rounds ------------------------------------------------------------------------------------------------------
        self.rounds, self.slots, self.draft_head = None, [], None
        if options["rounds"]:
            from crit_plugins.rounds import Rounds, width_policy
            pools = [self.pool, self.states] + ([self.mtp_pool] if self.mtp else [])
            drafting = {}
            if self.mtp:
                self.draft_head = MtpHead(self, self.concurrency, self.width)
                widths = sorted({int(w) for w in str(options["widths"]).split(",") if w.strip()} | {self.width})
                drafting = dict(drafter=self.drafter, head=self.draft_head, widths=widths,
                                width_policy=width_policy(options["width_policy"]))
            self.rounds = Rounds(self, self.sampler, rows=self.concurrency, widest=self.width, device=self.device,
                                 pools=pools, parts=parts,
                                 uses=["projections", "attention", "kv_cache", "recurrence", "norms", "embeddings"],
                                 stops=STOPS, logprobs=TOP, vocabulary=self.vocabulary, sampled=True, **drafting)

        # -- captured prompt chunks ------------------------------------------------------------------------------------
        self.prompts = None
        buckets = [int(b) for b in str(options["prompt_buckets"]).split(",") if b.strip()]
        if buckets:
            from crit_plugins.prompt_graphs import PromptGraphs
            self.prompts = PromptGraphs(self._prompt_forward, device=self.device, buckets=buckets, parts=parts,
                                        uses=["projections", "attention", "kv_cache", "recurrence", "norms",
                                              "embeddings"], pools=[self.pool, self.states])
            if not self.prompts.problems:
                self.prompts.prepare()

    # -- loading ---------------------------------------------------------------------------------------------------------
    def _stored(self, base):
        """(format, tensors) of matrix `base` as the projections slot's load takes them."""
        torch, ck = self.torch, self.ckpt
        p = self.matrix_parts(ck.headers, base, self.quant)
        if p is None:
            raise KeyError(f"qwen35-model: no matrix {base} in the checkpoint")
        get = lambda name: ck.get(name, self.device)  # noqa: E731
        if p.scheme == "plain":
            return "bf16", {"weight": get(p.codes).to(torch.bfloat16)}
        if p.scheme == "fp8":
            if p.scale_granularity == "row":
                return "fp8:channel", {"weight": get(p.codes), "scale": get(p.scales).float().reshape(-1)}
            if p.scale_granularity == "tensor":
                return "fp8:tensor", {"weight": get(p.codes), "scale": get(p.scales).float().reshape(())}
            raise ValueError(f"qwen35-model: {base} is FP8 per block, which this model plugin does not load")
        g = get(p.global_scale).float().reshape(())
        if not p.global_divides:
            g = 1.0 / g
        found = {"weight": get(p.codes), "scale": get(p.scales), "global_scale": g}
        if p.input_scale is not None:
            i = get(p.input_scale).float().reshape(())
            found["input_scale"] = i if p.input_divides else 1.0 / i
        return "nvfp4", found

    def _head_slices(self, count, encode):
        """The lm_head as `count` row slices re-encoded to `encode` (draft-only matrices)."""
        fmt, tensors = self._stored("lm_head")
        rows = tensors["weight"].shape[0]
        step = -(-rows // count)
        step = -(-step // 128) * 128
        found = []
        for first in range(0, rows, step):
            part = {k: (v[first:first + step] if getattr(v, "dim", lambda: 0)() and v.shape[0] == rows else v)
                    for k, v in tensors.items()}
            found.append(self.proj.load(part, fmt, role="draft_head", encode=encode))
            self.torch.cuda.empty_cache()
        del tensors
        return found

    def _matrix(self, bases, role, encode=None):
        """The matrices `bases` (same input) stacked by output rows into one, loaded in the projections slot (re-encoded
        to `encode` when only a drafter reads it)."""
        torch = self.torch
        found = [self._stored(b) for b in bases]
        fmt = found[0][0]
        if any(f != fmt for f, _ in found):
            raise ValueError(f"qwen35-model: {bases} are stored in different formats; stack only one")
        if len(found) == 1:
            tensors = found[0][1]
        else:
            parts = [t for _, t in found]
            tensors = {"weight": torch.cat([t["weight"] for t in parts])}
            if fmt == "fp8:channel":
                tensors["scale"] = torch.cat([t["scale"] for t in parts])
            elif fmt == "fp8:tensor":
                values = {float(t["scale"]) for t in parts}
                if len(values) != 1:
                    raise ValueError(f"qwen35-model: {bases} have different FP8 tensor scales")
                tensors["scale"] = parts[0]["scale"]
            elif fmt == "nvfp4":
                tensors["scale"] = torch.cat([t["scale"] for t in parts])
                tensors["global_scale"] = torch.cat([t["global_scale"].expand(t["weight"].shape[0]) for t in parts])
                inputs = [t.get("input_scale") for t in parts]
                if all(i is not None for i in inputs) and len({float(i) for i in inputs}) == 1:
                    tensors["input_scale"] = inputs[0]
            del parts
        del found
        if encode is not None:
            return self.proj.load(tensors, fmt, role=role, encode=encode)
        return self.proj.load(tensors, fmt, role=role)

    # -- Core's settings -------------------------------------------------------------------------------------------------
    def info(self):
        found = {"implementation": "qwen3_5 hybrid dense decoder (crit device plugins)", "vocabulary": self.vocabulary,
                 "context": self.context, "prefill_chunk": self.chunk, "prefill_work_quantum": self.chunk,
                 "max_concurrency": self.concurrency, "max_pending": self.pending_max,
                 "max_batch_tokens": self.chunk + self.concurrency * (self.width + 1),
                 "max_row_tokens": self.width + 1, "decode_token_work": self.width + 1, "stop_tokens": self.stops,
                 "sampling": dict.fromkeys(CORE_SAMPLING, True), "token_logprobs": True, "max_top_logprobs": TOP,
                 "scoring": True, "query_scoring": True, "score_paths": ["prefill"], "replay": True,
                 "speculative": self.mtp, "plain": self.mtp}
        if self.mtp and self.rounds is not None:
            found.update(self.rounds.width_facts())
            found["drafter_kind"] = "checkpoint MTP layer (model_mtp.safetensors)"
        return found

    def facts(self):
        return {"kv_pages": self.kv_dtype, "kv_pool_tokens": self.kv_tokens, "state_sequences": self.sequences,
                "mtp": {"width": self.width, "kv": self.options["mtp_kv"], "encode": self.options["mtp_encode"],
                        "draft_head": self.options["draft_head"]} if self.mtp else None,
                "rounds": None if self.rounds is None else {"rows": self.rounds.rows, "problems": self.rounds.problems},
                "prompt_graphs": None if self.prompts is None else {"buckets": sorted(self.prompts.graphs),
                                                                   "problems": self.prompts.problems}}

    def telemetry(self):
        found = self.rounds.telemetry() if self.rounds is not None else {}
        if self.prompts is not None:
            found = {**found, "prompt_graphs": self.prompts.telemetry()}
        return found

    def native_rounds(self, library):
        if self.rounds is None:
            return None
        return self.rounds.native_rounds(library, [self.width] * self.rounds.rows, mode="sampled")

    # -- the arithmetic --------------------------------------------------------------------------------------------------
    def _norm(self, x, weight, residual=None):
        """Qwen3.5's RMSNorm: (1 + weight) over the normalized row (residual + x first, in place, when given)."""
        return self.norms.rms_norm(x, weight, self.eps, offset=1.0, residual=residual)

    def _groups(self, batch):
        """The token ranges the batch's matrices multiply apart: each prompt chunk alone, one-token rows (decode rows,
        a round's columns) decode_rows at a time, so every row's bits are those it has alone."""
        step = self.decode_rows
        if getattr(batch, "prompt", False):   # a captured prompt chunk: one chunk, its bucket's rows
            return [(0, batch.tokens_t.shape[0])]
        if batch.round:
            return [(first, min(first + step, batch.tokens)) for first in range(0, batch.tokens, step)]
        groups, run = [], None
        for r in range(batch.rows):
            first, end = batch.starts[r], batch.starts[r + 1]
            if end - first > 1:
                groups.append((first, end))
                run = None
            elif run is not None and run[1] == first and run[1] - run[0] < step:
                run = groups[-1] = (run[0], end)
            else:
                run = (first, end)
                groups.append(run)
        return groups

    def _rows_groups(self, count):
        step = self.decode_rows
        return [(f, min(f + step, count)) for f in range(0, count, step)]

    def _mm(self, x, matrix, groups=None):
        groups = self.groups if groups is None else groups
        if len(groups) == 1:
            return self.proj.matmul(x, matrix)
        return self.torch.cat([self.proj.matmul(x[first:end], matrix) for first, end in groups])

    def _linear(self, i, h, batch):
        torch, t, b = self.torch, h.shape[0], self.blocks[i]
        qk, vd = self.nk * self.dk, self.nv * self.dv
        qkvz = self._mm(h, b["qkvz"])
        qkv, z = qkvz[:, :2 * qk + vd], qkvz[:, 2 * qk + vd:]
        ba = self._mm(h, b["ba"])
        beta, a = ba[:, :self.nv], ba[:, self.nv:]
        mixed = self.recurrence.convolve(i, qkv.contiguous(), b["conv"], None, self.states, batch)
        q, k, v = torch.split(mixed, [qk, qk, vd], dim=-1)
        q, k = q.reshape(t, self.nk, self.dk), k.reshape(t, self.nk, self.dk)   # value head h reads key head h // r
        v = v.reshape(t, self.nv, self.dv)
        decay, beta = self.norms.softplus_decay(a.contiguous(), b["dt_bias"], b["a_neg"], beta.contiguous())
        out = self.recurrence.forward(i, q, k, v, decay, beta, self.states, batch, self.geometry)
        gated = self.norms.gated_rms_norm(out.reshape(t, self.nv, self.dv), z.reshape(t, self.nv, self.dv),
                                          b["norm"], self.eps, out_dtype=h.dtype)
        return self._mm(gated.reshape(t, -1), b["out"])

    def _qkv(self, h, b, positions, groups):
        """A gated-attention layer's normed, rotated queries and keys, values and output gate."""
        torch, t = self.torch, h.shape[0]
        width = self.heads * self.head_dim * 2
        kv = self.kv_heads * self.head_dim
        qkv = self._mm(h, b["qkv"], groups)
        q, gate = torch.chunk(qkv[:, :width].reshape(t, self.heads, 2 * self.head_dim), 2, dim=-1)
        k = qkv[:, width:width + kv].reshape(t, self.kv_heads, self.head_dim)
        v = qkv[:, width + kv:].reshape(t, self.kv_heads, self.head_dim)
        if self.fused:
            q, k = self.norms.rms_norm_rope_pair(q, k, b["q_norm"], b["k_norm"], self.eps, positions,
                                                 self.inv_freq, offset=1.0)
        else:
            q = self.norms.rms_norm_rope(q, b["q_norm"], self.eps, positions, self.inv_freq, offset=1.0)
            k = self.norms.rms_norm_rope(k, b["k_norm"], self.eps, positions, self.inv_freq, offset=1.0)
        return q, k, v.contiguous(), gate

    def _full(self, i, h, batch, slots):
        t, b = h.shape[0], self.blocks[i]
        q, k, v, gate = self._qkv(h, b, self.positions, self.groups)
        self.pool.write(i, slots, k, v)
        a = self.attention.forward(i, q, self.pool, batch, scale=self.head_dim ** -0.5)
        a = self.norms.sigmoid_mul(a, gate, out_dtype=h.dtype)       # each head's output times sigmoid(its gate)
        return self._mm(a.reshape(t, -1), b["o"])

    def _hidden(self, ids, batch):
        """The final-normed hidden rows of the batch's tokens."""
        x = self.embed.lookup(ids)   # the residual stream, which each norm after a block's output advances in place
        self.positions, self.groups = batch.positions_t, self._groups(batch)
        slots = self.pool.reserve(batch)
        h = self._norm(x, self.blocks[0]["in"])
        for i, kind in enumerate(self.types):
            mixed = self._linear(i, h, batch) if kind == "linear_attention" else self._full(i, h, batch, slots)
            h, x = self._norm(mixed, self.blocks[i]["post"], residual=x)
            mlp = self._mm(self.norms.silu_mul(self._mm(h, self.blocks[i]["gate_up"])), self.blocks[i]["down"])
            following = self.blocks[i + 1]["in"] if i + 1 < self.layers else self.final_norm
            h, x = self._norm(mlp, following, residual=x)
        return h

    def _prompt_forward(self, batch):
        """A captured prompt chunk's forward (crit_plugins.prompt_graphs): the final-normed hidden rows of the bucket,
        the same prompt arithmetic as the eager path."""
        return self._hidden(batch.tokens_t, batch)

    def _mtp_input(self, tokens, hidden, groups):
        """The MTP layer's input row of each (next token, target hidden) pair: fc over [norm(embed), norm(hidden)]."""
        b = self.mtp_block
        e = self.embed.lookup(tokens)
        joined = self.torch.cat([self.norms.rms_norm(e, b["norm_e"], self.eps, offset=1.0),
                                 self.norms.rms_norm(hidden, b["norm_h"], self.eps, offset=1.0)], dim=-1)
        return self._mm(joined, b["fc"], groups)

    def _mtp(self, tokens, hidden, batch):
        """The MTP layer over a round batch (catch-up columns or a chain step): its normed output rows, the next
        step's hidden state and the draft head's input."""
        b, t = self.mtp_block, tokens.shape[0]
        groups = self._rows_groups(t)
        x = self._mtp_input(tokens, hidden, groups)
        h = self._norm(x, b["in"])
        q, k, v, gate = self._qkv(h, b, batch.positions_t, groups)
        slots = self.mtp_pool.reserve(batch)
        self.mtp_pool.write(self.mtp_layer, slots, k, v)
        a = self.attention.forward(self.mtp_layer, q, self.mtp_pool, batch, scale=self.head_dim ** -0.5)
        a = self.norms.sigmoid_mul(a, gate, out_dtype=h.dtype)
        h, x = self._norm(self._mm(a.reshape(t, -1), b["o"], groups), b["post"], residual=x)
        mlp = self._mm(self.norms.silu_mul(self._mm(h, b["gate_up"], groups)), b["down"], groups)
        out, _ = self._norm(mlp, b["final"], residual=x)
        return out

    def _mtp_prompt(self, rows, hidden):
        """The MTP pool's keys and values of prompt positions (`rows`: (session, first position, count, first row of
        `hidden`)), each from the token after it and the target's hidden state there: the fused input, the input norm
        and the k/v projections (no attention: a one-layer drafter's KV reads nothing else)."""
        torch = self.torch
        b = self.mtp_block
        packed, tokens, index = [], [], []
        for s, first, count, at in rows:
            packed.append((s.handle, first, count))
            tokens += s.prompt[first + 1:first + 1 + count]
            index += range(at, at + count)
        batch = StepBatch.build(packed, self.device)
        groups = [(batch.starts[r], batch.starts[r + 1]) for r in range(batch.rows)]
        x = self._mtp_input(torch.tensor(tokens, dtype=torch.int64, device=self.device),
                            hidden[torch.tensor(index, dtype=torch.int64, device=self.device)], groups)
        h = self._norm(x, b["in"])
        _, k, v, _ = self._qkv(h, b, batch.positions_t, groups)
        slots = self.mtp_pool.reserve(batch)
        self.mtp_pool.write(self.mtp_layer, slots, k, v)

    def _logits_rows(self, h, draft=False):
        """FP32 logits of rows h, each multiplied as a decode row is (calls of at most decode_rows rows); a draft head
        of row slices gives the slices' columns side by side."""
        groups = self._rows_groups(h.shape[0])
        matrix = self.draft_matrix if draft else self.head
        if isinstance(matrix, list):
            return self.torch.cat([self._mm(h, m, groups) for m in matrix], dim=-1).float()
        return self._mm(h, matrix, groups).float()

    def _forward(self, ids, batch, logit_rows):
        """Logits [len(logit_rows)][vocabulary] (FP32) of the batch's tokens at logit_rows (None: every token)."""
        h = self._hidden(ids, batch)
        return self._logits_at(h, logit_rows), h

    def _logits_at(self, h, logit_rows):
        torch = self.torch
        if logit_rows is None:
            return self._mm(h, self.head).float()
        if not logit_rows:   # a step of prompt chunks none of which ends its prompt
            return torch.zeros(0, self.vocabulary, dtype=torch.float32, device=self.device)
        rows = torch.tensor(logit_rows, dtype=torch.int64, device=self.device)
        return self._logits_rows(h[rows])

    def verify(self, view, batch):
        """The round's target pass (crit_plugins.rounds): logits of every column of the RoundBatch, [rows][width + 1]
        [vocabulary]; the recurrence and the KV follow the round slots their pools bind; the draft head keeps every
        column's hidden state for its next catch-up."""
        h = self._hidden(batch.tokens_t, batch)
        if self.draft_head is not None:
            self.draft_head.keep_features(view, h.view(view.rows, view.width + 1, -1))
        return self._mm(h, self.head).float().view(view.rows, view.width + 1, -1)

    # -- Core's step model -----------------------------------------------------------------------------------------------
    def admission(self, request):
        prompt, output = len(request.prompt), int(request.output_tokens)
        found = self.reservations.admission(prompt, output)
        if found != READY:
            return found
        return READY if self.states.free else DEFERRED

    def _open(self, sequence):
        self.pool.open(sequence)
        self.states.open(sequence)
        if self.mtp_pool is not None:
            self.mtp_pool.open(sequence)

    def _close(self, sequence):
        self.pool.close(sequence)
        self.states.close(sequence)
        if self.mtp_pool is not None:
            self.mtp_pool.close(sequence)

    def open(self, request):
        self.handles += 1
        session = Session(self.handles, request)
        prompt, output = len(request.prompt), int(request.output_tokens)
        if self.reservations.admission(prompt, output) == READY:
            session.pages = self.reservations.reserve(prompt, output)
        self._open(session.handle)
        return session

    def step(self, rows):
        results = [None] * len(rows)
        bound = [i for i, row in enumerate(rows) if row.session.slot is not None]
        if bound:
            scheduled = {rows[i].session.slot for i in bound}
            held = [slot for slot in range(len(self.slots)) if slot not in scheduled]
            width = self.rounds.width(len(self.slots))
            self.rounds.launch(len(self.slots), width, mode="sampled", hold=held)
            found = self.rounds.read()
            for i in bound:
                s = rows[i].session
                row = found[s.slot]
                s.phase, s.produced, s.rows = "round", row["tokens"], 0
                logprobs = None
                if s.request.logprobs is not None:
                    asked = int(s.request.logprobs)
                    logprobs = [(value, top[:asked]) for value, top in zip(row["logprobs"], row["top"])]
                results[i] = StepResult(0, s.produced, True, logprobs=logprobs, drafted=width)
        rest = [i for i, row in enumerate(rows) if row.session.slot is None]
        if rest:
            for i, result in zip(rest, self._plain([rows[i] for i in rest])):
                results[i] = result
        return results

    def _plain(self, rows):
        torch = self.torch
        layout = StepLayout(step_item(row, len(row.session.prompt), row.session.done) for row in rows)
        packed, ids, logit_rows, placed, prompts, graphed = [], [], [], [], [], []
        for placement in layout.placements:
            s = rows[placement.item].session
            inputs = [s.pending] if placement.phase == DECODE else s.prompt[s.done:s.done + placement.rows]
            s.phase, s.rows, s.produced, s.graphed = placement.phase, len(inputs), [], False
            if placement.phase == PREFILL and self.prompts is not None and self.prompts.graphs:
                found = self.prompts.run(s.handle, s.done, inputs)
                if found is not None:
                    hidden = found[:len(inputs)].clone()   # the graph's output, overwritten by its next run
                    s.graphed = True
                    graphed.append((placement, s, hidden, inputs))
                    placed.append((placement, s, bool(placement.logits)))
                    continue
            if placement.phase == PREFILL:
                # the MTP pool's positions of this chunk that have their next token in the prompt
                count = min(s.done + len(inputs), len(s.prompt) - 1) - s.done
                if count > 0:
                    prompts.append((s, s.done, count, len(ids)))
            packed.append((s.handle, s.done, len(inputs)))
            ids += inputs
            if placement.logits:
                logit_rows.append(len(ids) - 1)
            placed.append((placement, s, bool(placement.logits)))
        eager_logits, hidden = None, None
        if packed:
            batch = StepBatch.build(packed, self.device)
            eager_logits, hidden = self._forward(torch.tensor(ids, dtype=torch.int64, device=self.device), batch,
                                                 logit_rows)
            if self.mtp_pool is not None and prompts:
                self._mtp_prompt(prompts, hidden)
        # every logits row in placed order: the eager rows in theirs, a graphed chunk's last row from its hidden
        last_hidden, rows_logits, e = {}, [], 0
        graphs = {id(s): (h, inputs) for _, s, h, inputs in graphed}
        for placement, s, has in placed:
            if not has:
                continue
            if id(s) in graphs:
                h, inputs = graphs[id(s)]
                rows_logits.append(self._logits_rows(h[-1:]))
                last_hidden[id(s)] = h[-1]
            else:
                rows_logits.append(eager_logits[e:e + 1])
                last_hidden[id(s)] = hidden[logit_rows[e]]
                e += 1
        for placement, s, h, inputs in graphed:
            if self.mtp_pool is not None:
                count = min(s.done + len(inputs), len(s.prompt) - 1) - s.done
                if count > 0:
                    self._mtp_prompt([(s, s.done, count, 0)], h)
        logits = torch.cat(rows_logits) if rows_logits else None
        sampled = [(s, rows[placement.item].allowed) for placement, s, has in placed if has]
        params = [dict({k: s.request.sampling[k] for k in CORE_SAMPLING if k in s.request.sampling},
                       position=len(s.outputs), **({} if allowed is None else {"allowed": allowed}))
                  for s, allowed in sampled]
        tokens = self.sampler.sample(logits, params).tolist() if sampled else []
        if any(placement.phase == PREFILL and has for placement, _, has in placed):
            release_cached_under(self.free_below, self.device)   # a prompt ended: its temporaries leave room for the next
        results, j = [None] * len(rows), 0
        for placement, s, has in placed:
            logprobs = None
            if has:
                token = s.request.replay[len(s.outputs)] if s.request.replay else tokens[j]
                s.produced = [token]
                if placement.phase == PREFILL and self.draft_head is not None:
                    s.last_hidden = last_hidden[id(s)].clone()   # the seed's feature: the prompt's last row
                if s.request.logprobs is not None:
                    row = torch.log_softmax(logits[j], -1)
                    ranked, order = row.sort(descending=True, stable=True)
                    k = min(int(s.request.logprobs), TOP)
                    logprobs = [(float(row[token]), list(zip(order[:k].tolist(), ranked[:k].tolist())))]
                j += 1
            results[placement.item] = StepResult(placement.rows if placement.phase == PREFILL else 0, s.produced, has,
                                                 0, placement.rows, logprobs)
        return results

    def _roundable(self, s):
        """Whether a request whose prompt is in decodes in round slots from here: rounds on and able to run, a free
        slot, no replay, no plain (constrained) decoding, its stops within a slot's, its log-probabilities within the
        driver's alternatives."""
        r, rounds = s.request, self.rounds
        return (rounds is not None and rounds.native is not None and not rounds.problems
                and len(self.slots) < rounds.rows and not r.replay and not r.plain
                and len(r.stop_tokens) <= STOPS and (r.logprobs is None or int(r.logprobs) <= rounds.top)
                and (self.draft_head is None or s.last_hidden is not None))

    def commit(self, decisions):
        rounds = self.rounds
        native = rounds.native if rounds is not None else None
        trims = {}
        for d in decisions:
            s = d.session
            if s.slot is not None:
                if not (d.terminal or d.cancelled or d.discard):
                    trims[s.slot] = d.accepted
                continue
            if s.phase is None:
                continue
            completed = s.phase == PREFILL and s.done + s.rows == len(s.prompt)
            kept = not (d.cancelled or d.discard)
            if kept:
                s.done += s.rows
                if d.accepted:
                    s.pending = s.produced[d.accepted - 1]
                    s.outputs += s.produced[:d.accepted]
            if s.graphed:
                if not kept:
                    raise RuntimeError("qwen35-model: a captured prompt chunk settled its state, and Core discarded it")
            else:
                self.states.accept(s.handle, s.rows if kept else 0)   # the state and the convolution window
            self.pool.keep(s.handle, s.done)
            if self.mtp_pool is not None:
                s.mtp_done = min(s.done, len(s.prompt) - 1) if kept and s.phase == PREFILL else s.mtp_done
                self.mtp_pool.keep(s.handle, s.mtp_done)
            s.phase, s.produced = None, []
            if completed and kept and d.accepted and not d.terminal and self._roundable(s):
                self._bind(s)
        if trims:
            ran_natively = native is not None and native.last is not None and native.last[2]
            if ran_natively:
                rounds.adopt(*native.last[:2])
                emitted = rounds.n_emitted[:len(self.slots)].tolist()
            else:
                emitted = [len(s.produced) for s in self.slots]
            keep = [min(count, trims.get(slot, count)) for slot, count in enumerate(emitted)]
            if keep != emitted:
                rounds.truncate(keep)

    def _bind(self, s):
        """The session decodes in the next free round slot: its pending token (the first output) at the prompt's end,
        the rest of its output limit, its stop tokens and its sampling rules; the draft head's seed feature is the
        prompt's last hidden state."""
        s.slot = len(self.slots)
        self.slots.append(s)
        r = s.request
        params = {k: r.sampling[k] for k in CORE_SAMPLING if k in r.sampling}
        params.setdefault("temperature", 0.0)
        if self.draft_head is not None:
            self.draft_head.seed_feature(s.slot, s.last_hidden)
            s.last_hidden = None
        self.rounds.bind(s.slot, s.handle, len(s.prompt), s.pending, r.output_tokens - len(s.outputs),
                         len(s.prompt) + r.output_tokens + self.width + 1, stops=r.stop_tokens, params=params,
                         session=s, tokens=s.prompt + [s.pending], outputs=len(s.outputs),
                         logprobs=r.logprobs is not None)

    def retire(self, session, finish):
        if session.slot is not None:
            slot, last = session.slot, len(self.slots) - 1
            self.rounds.unbind(slot)
            if slot != last:   # the decoding rows stay packed at the front
                moved = self.slots[last]
                self.rounds.move(last, slot)
                if self.draft_head is not None:
                    self.draft_head.move(last, slot)
                moved.slot = slot
                self.slots[slot] = moved
            self.slots.pop()
            session.slot = None
        self._close(session.handle)
        session.last_hidden = None
        if session.pages:
            self.reservations.release(session.pages)
            session.pages = 0

    def score(self, ids, first_target, path="", queries=None, top_k=0):
        """A fixed text's scores through the prompt path: its tokens a prefill_chunk at a time, as a prompt's chunks,
        each chunk's KV and recurrent state kept for the next; the logits of the scored rows SCORE_SLICE at a time."""
        torch = self.torch
        sequence = ("score", self.handles)
        wanted = range(first_target - 1, len(ids) - 1)
        target, argmax, top, query = [], [], [], [] if queries else None
        self.pool.open(sequence)
        self.states.open(sequence)
        try:
            for start in range(0, wanted.stop, self.chunk):   # the last token predicts nothing scored
                end = min(start + self.chunk, wanted.stop)
                batch = StepBatch.build([(sequence, start, end - start)], self.device)
                h = self._hidden(torch.tensor(ids[start:end], dtype=torch.int64, device=self.device), batch)
                self.states.accept(sequence, end - start)
                self.pool.keep(sequence, end)
                first = max(start, wanted.start)
                for lo in range(first, end, SCORE_SLICE):
                    hi = min(lo + SCORE_SLICE, end)
                    logp = torch.log_softmax(self._logits_rows(h[lo - start:hi - start]), -1)
                    nxt = torch.tensor(ids[lo + 1:hi + 1], dtype=torch.int64, device=self.device)
                    target += logp.gather(1, nxt[:, None])[:, 0].tolist()
                    argmax += logp.argmax(-1).tolist()
                    if top_k:
                        ranked, order = logp.sort(dim=-1, descending=True, stable=True)
                        top += [list(zip(o, v)) for o, v in zip(order[:, :top_k].tolist(), ranked[:, :top_k].tolist())]
                    if queries:
                        for r in range(lo, hi):
                            row = queries[r - wanted.start]
                            query.append(logp[r - lo, torch.tensor(row, dtype=torch.int64, device=self.device)]
                                         .tolist() if row else [])
                    del logp
        finally:
            self.pool.close(sequence)
            self.states.close(sequence)
            release_cached_under(self.free_below, self.device)
        return Scores(target, query, argmax, top or None)

    def close(self):
        self.blocks.clear()


def _tuning(options):
    """The engine's kernel tuning (a folder of crit kernels sweep tables kept beside this plugin, option
    kernel_tuning), named before any plugin builds a kernel: memory() runs before every factory."""
    if options.get("kernel_tuning"):
        from pathlib import Path
        from crit_plugins.kernels import use_tuning
        folder = Path(__file__).resolve().parent / options["kernel_tuning"]
        if not folder.is_dir():
            raise ValueError(f"qwen35-model: kernel_tuning {options['kernel_tuning']} is no folder beside the plugin")
        use_tuning(folder)


def memory(context, options):
    _tuning(options)
    return {"device": 0, "host": 0}


def create(context, options, parts):
    _tuning(options)
    return Qwen35Model(context, options, parts)
