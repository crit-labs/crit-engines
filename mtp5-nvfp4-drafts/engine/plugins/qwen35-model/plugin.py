"""Qwen3.5-layout hybrid decoder as a model plugin (Qwen3.8-27B; transformers' qwen3_5 modeling): 48 gated-delta layers
(a causal convolution over q|k|v, then the delta rule, both in the recurrence slot; the output's gated RMSNorm and
the z gate in the norms slot) and 16 gated full-attention layers (per-head q/k RMSNorm with partial rotary positions,
the output times sigmoid(gate); KV in the kv_cache slot's pool, attention by the attention slot), each followed by a
SwiGLU MLP. Every matrix is in the projections slot as the checkpoint stores it (FP8 per channel, NVFP4 W4A4, BF16),
the embedding table in the embeddings slot (host memory when the placement puts it there), every token from the
sampler.

A step of many sequences' rows is one batch: each prompt chunk's matrices multiplied alone and one-token rows (decode
rows, a round's columns) in calls of at most invariant_rows(projections), so a row's bits are those it has alone.

Decode runs in Core's captured rounds (crit_plugins.rounds.Rounds) once a prompt is in: verify(view, batch) is the
target pass over a RoundBatch, the sampler predicts on the device (greedy rows by argmax, sampled rows by rejection
sampling against the drafts' laws), and the drafter chains the checkpoint's MTP layer as the model's device draft head
(draft:head-device, MTPHead below). The MTP layer at position p reads the token at p + 1 and the target's final normed
hidden at p: h = fc([norm_e(embed(t_{p+1})) | norm_h(hidden_p)]), one gated-attention decoder layer at rotary position
p with its own KV (an extra layer of the target's pool), then mtp.norm; the head predicts t_{p+2}, and the normed
output is the hidden of the next draft step. Prompt chunks fill the MTP layer's KV for every prompt position (its
input norms, fc and key/value projections only), so the first round drafts at once."""
import json
import os
from pathlib import Path

from crit_core.batch import DECODE, PREFILL, PageReservations, StepLayout, step_item
from crit_core.native import Scores, StepResult
from crit_plugins import StepBatch
from crit_plugins.device import RoundBatch, free_device_bytes, invariant_rows

CORE_SAMPLING = ("temperature", "top_p", "top_k", "min_p", "seed")
STOPS = 8     # stop ids a row in a round slot holds
TOP = 20      # alternatives a token's log-probabilities carry at most
MTP_LAYER = 1000   # the MTP layer's id in the KV pool


class Session:
    def __init__(self, handle, request):
        self.handle, self.request, self.prompt = handle, request, list(request.prompt)
        self.done, self.outputs, self.pending = 0, [], None
        self.phase, self.produced, self.rows = None, [], 0
        self.slot = None      # its round slot once it decodes in rounds
        self.carry = None     # the target's final normed hidden of its last prompt row (the MTP layer's input)
        self.pages = 0        # pages its admission reserved
        self.graphed = False  # its last step's prompt chunk ran as a captured graph (its state written in place)


class Checkpoint:
    """The checkpoint's tensors by name across its safetensors files, read one at a time."""

    def __init__(self, folder):
        from safetensors import safe_open
        self.files, self.names = [], {}
        for path in sorted(Path(folder).glob("*.safetensors")):
            handle = safe_open(str(path), "pt", device="cpu")
            self.files.append(handle)
            for name in handle.keys():
                self.names[name] = handle
        prefixes = ("model.language_model.", "model.")
        self.prefix = next(p for p in prefixes if p + "embed_tokens.weight" in self.names)

    def has(self, name):
        return name in self.names

    def get(self, name):
        return self.names[name].get_tensor(name)

    def nbytes(self, name):
        s = self.names[name].get_slice(name)
        shape = s.get_shape()
        size = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "I8": 1}[s.get_dtype()]
        n = 1
        for d in shape:
            n *= d
        return n * size


class Matrices:
    """Loads a group of the checkpoint's linear layers (stacked by rows) into the projections slot in their stored
    format: nvfp4 (weight_packed, weight_scale, weight_global_scale, input_global_scale), fp8:channel (E4M3 weight and
    a scale per row) or bf16."""

    def __init__(self, ck, proj, device):
        import torch
        self.ck, self.proj, self.device, self.torch = ck, proj, device, torch
        self.bytes = 0

    def kind(self, prefix):
        if self.ck.has(prefix + "weight_packed"):
            return "nvfp4"
        w = self.ck.names[prefix + "weight"].get_slice(prefix + "weight")
        return "fp8:channel" if w.get_dtype() == "F8_E4M3" else "bf16"

    def load(self, prefixes, role=None, encode=None):
        torch, on = self.torch, (lambda name: self.ck.get(name).to(self.device))
        kinds = {self.kind(p) for p in prefixes}
        if len(kinds) != 1:
            raise ValueError(f"matrices {prefixes} of mixed formats {kinds}")
        kind = kinds.pop()
        if kind == "nvfp4":
            weights = [on(p + "weight_packed") for p in prefixes]
            scales = [on(p + "weight_scale") for p in prefixes]
            globals_ = [on(p + "weight_global_scale").float().reshape(-1) for p in prefixes]
            inputs = [on(p + "input_global_scale").float().reshape(-1) for p in prefixes]
            if any(not torch.equal(inputs[0], i) for i in inputs[1:]):
                raise ValueError(f"stacked W4A4 matrices {prefixes} need one input scale")
            if all(torch.equal(globals_[0], g) for g in globals_[1:]):
                g = globals_[0].reshape(())
            else:
                g = torch.cat([gs.expand(w.shape[0]) for gs, w in zip(globals_, weights)])
            tensors = {"weight": torch.cat(weights), "scale": torch.cat(scales), "global_scale": g,
                       "input_scale": inputs[0].reshape(())}
        elif kind == "fp8:channel":
            tensors = {"weight": torch.cat([on(p + "weight") for p in prefixes]),
                       "scale": torch.cat([on(p + "weight_scale").float().reshape(-1) for p in prefixes])}
        else:
            tensors = {"weight": torch.cat([on(p + "weight").to(torch.bfloat16) for p in prefixes])}
        self.bytes += sum(t.numel() * t.element_size() for t in tensors.values())
        matrix = self.proj.load(tensors, kind, role=role, encode=encode) if encode else \
            self.proj.load(tensors, kind, role=role)
        del tensors
        return matrix


class MTPHead:
    """The checkpoint's MTP layer as the model's device draft head (draft:head-device) inside captured rounds, and a
    pool of the round (it follows the round slots: its kept hiddens move with them).

    hid_keep[slot][j] is the target's final normed hidden at the position before catch-up token j (the previous
    round's verify columns, or at bind the prompt's last row), so catch_up runs the MTP layer at positions
    frontier - count .. frontier - 1 over the catch-up tokens (one column per position, widest + 1 columns, the ones
    past the count writing positions a later step writes again before any read) and keeps the last valid column's
    output: depth 0's input. Each later depth runs one column at frontier - 1 + depth from the previous draft and the
    previous depth's output."""

    def __init__(self, model, rows, widest):
        torch = model.torch
        self.model, self.torch, self.rows, self.widest = model, torch, rows, widest
        H, dev = model.hidden, model.device
        cols = widest + 1
        self.hid_keep = torch.zeros(rows, cols, H, dtype=torch.bfloat16, device=dev)
        self.verify_hidden = torch.zeros(rows * cols, H, dtype=torch.bfloat16, device=dev)
        self.last = torch.zeros(rows, H, dtype=torch.bfloat16, device=dev)
        self.catch_batch = RoundBatch(rows, widest, dev)
        self.depth_batch = RoundBatch(rows, 0, dev)
        self.frontier = torch.zeros(rows, dtype=torch.long, device=dev)
        self.depth_frontier = torch.zeros(rows, dtype=torch.long, device=dev)
        self.tokens = torch.zeros(rows, 1, dtype=torch.long, device=dev)
        self.columns = torch.arange(cols, dtype=torch.long, device=dev)

    # draft:head-device ------------------------------------------------------------------------------------------------
    def catch_up(self, view):
        torch, m, rows, cols = self.torch, self.model, view.rows, self.widest + 1
        count = view.catch_up_count[:rows]
        self.frontier[:rows] = view.frontier[:rows] - count
        batch = self.catch_batch.fill(rows, self.widest, self.frontier, view.catch_up_tokens[:rows])
        out = m._mtp(batch.tokens_t, self.hid_keep[:rows].reshape(rows * cols, -1), batch)
        index = (count - 1).clamp(min=0)
        self.last[:rows] = out.view(rows, cols, -1).gather(
            1, index[:, None, None].expand(rows, 1, out.shape[-1])).squeeze(1)

    def logits(self, depth, tokens, view, parents=None):
        m, rows = self.model, view.rows
        if depth > 0:
            self.depth_frontier[:rows] = view.frontier[:rows] - 1 + depth
            self.tokens[:rows, 0] = tokens
            batch = self.depth_batch.fill(rows, 0, self.depth_frontier, self.tokens)
            self.last[:rows] = m._mtp(batch.tokens_t, self.last[:rows], batch)
        return m._head(self.last[:rows], draft=True)

    # a pool of the round ----------------------------------------------------------------------------------------------
    def bind(self, slot, sequence, capacity=None):
        self.hid_keep[slot].zero_()

    def unbind(self, slot):
        self.hid_keep[slot].zero_()

    def move(self, source, target):
        self.hid_keep[target] = self.hid_keep[source]
        self.hid_keep[source].zero_()

    def commit_round(self, view):
        """Each row that emitted keeps its verify columns' hiddens (the next catch-up's inputs); idempotent."""
        rows, cols = view.rows, view.width + 1
        found = self.verify_hidden[:rows * cols].view(rows, cols, -1)
        emitted = (view.n_emitted[:rows] > 0)[:, None, None]
        self.hid_keep[:rows, :cols] = self.torch.where(emitted, found, self.hid_keep[:rows, :cols])


class Qwen35Model:
    def __init__(self, context, options, parts):
        import torch
        self.torch, self.options = torch, options
        config = context.config.get("text_config") or context.config
        self.types = list(config["layer_types"])
        self.layers, self.hidden = len(self.types), config["hidden_size"]
        self.heads, self.kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
        self.head_dim = config.get("head_dim") or self.hidden // self.heads
        self.nk, self.nv = config["linear_num_key_heads"], config["linear_num_value_heads"]
        self.dk, self.dv = config["linear_key_head_dim"], config["linear_value_head_dim"]
        self.kernel = config["linear_conv_kernel_dim"]
        self.eps, self.vocabulary = config["rms_norm_eps"], config["vocab_size"]
        rope = config.get("rope_parameters") or {}
        self.rotary = int(self.head_dim * rope.get("partial_rotary_factor", config.get("partial_rotary_factor", 1.0)))
        theta = rope.get("rope_theta") or config.get("rope_theta", 10000.0)
        self.device = torch.device("cuda")
        self.linear = [i for i, t in enumerate(self.types) if t == "linear_attention"]
        self.full = [i for i, t in enumerate(self.types) if t == "full_attention"]
        self.context = int(context.context_tokens or options["context"])
        self.concurrency = options["concurrency"]
        self.sequences = self.concurrency + options["max_pending"] + 2   # every open session and a score's
        self.proj, self.attention, self.sampler = parts["projections"], parts["attention"], parts["sampler"]
        self.recurrence, self.norms, self.embeddings = parts["recurrence"], parts["norms"], parts["embeddings"]
        self.fused = "norms:fused" in parts.facts("norms")
        self.rows_per_call = invariant_rows(self.proj)
        ck = Checkpoint(context.model)
        p = ck.prefix
        self.drafter = parts.get("drafter")
        drafts = self.drafter is not None and ck.has("mtp.fc.weight") and context.speculative
        if not drafts:
            self.drafter = None
        # The round widths: `width` alone (fixed), or `widths` among which the width policy picks each round's,
        # starting at `width`; every buffer is sized for the widest.
        widths = sorted({int(w) for w in options["widths"].split(",") if w.strip()} | {options["width"]}) \
            if options["widths"] else [options["width"]]
        if self.drafter is None:
            widths = [0]
        self.widths, self.start_width = widths, (options["width"] if self.drafter is not None else 0)
        self.width = max(widths)
        self.round_rows = self.concurrency

        # -- the plan: every tensor this plugin keeps, where the placement plugin puts it ---------------------------
        conv_channels = 2 * self.nk * self.dk + self.nv * self.dv
        kv_tokens = self.context + options["kv_extra"]
        kv_layers = len(self.full) + (1 if self.drafter is not None else 0)
        kv_unit = 1 if options["kv"] == "fp8" else 2
        plan = {name: ck.nbytes(name) for name in ck.names
                if name.startswith(p) and "visual" not in name and not name.endswith("embed_tokens.weight")}
        plan["lm_head"] = ck.nbytes("lm_head.weight") + ck.nbytes("lm_head.weight_scale") \
            if ck.has("lm_head.weight_scale") else ck.nbytes("lm_head.weight")
        if self.drafter is not None:
            plan.update({n: ck.nbytes(n) for n in ck.names if n.startswith("mtp.")})
        plan["tables.embed"] = ck.nbytes(p + "embed_tokens.weight")
        plan["kv"] = 2 * kv_layers * kv_tokens * self.kv_heads * self.head_dim * kv_unit
        plan["states"] = len(self.linear) * self.sequences * (self.nv * self.dk * self.dv * 4 + conv_channels * 4 * 2)
        plan["activations"] = 2 << 30   # a prompt chunk's activations, logits rows and the step's workspaces
        if self.drafter is not None:
            plan["mtp_buffers"] = self.round_rows * (self.width + 1) * self.hidden * 2 * 3
            if options["draft_head"] == "nvfp4":   # a draft-only copy of the head: E2M1 codes and E4M3 scales
                plan["draft_head"] = self.vocabulary * self.hidden * 9 // 16 + self.vocabulary * 4
        host = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        placed = parts["placement"].plan(plan, {"device": free_device_bytes(), "host": int(host)})
        off = {n for n, where in placed.items() if where != "device"}
        if off - {"tables.embed"}:
            raise ValueError(f"this model plugin keeps every weight on the device; the placement put {sorted(off)} off")
        host_table = "tables.embed" in off
        opens = "embeddings:host" in parts.facts("embeddings")   # a plugin with open(): a table in host memory
        if host_table and not opens and not getattr(self.embeddings, "host", False):
            raise ValueError("the placement puts the embedding table in host memory: the embeddings plugin keeps it "
                             "there (embeddings:host, or memory-table-rows' host option)")
        if not host_table and getattr(self.embeddings, "host", False):
            raise ValueError("the embeddings plugin keeps its table in host memory: the placement puts tables. off "
                             "the device (device-placement off_device 'tables.')")

        # -- weights ------------------------------------------------------------------------------------------------
        on = lambda name: ck.get(name).to(self.device)  # noqa: E731
        mats = Matrices(ck, self.proj, self.device)
        if host_table and opens:
            table = ck.get(p + "embed_tokens.weight").to(torch.bfloat16).contiguous()
            self.embed = self.embeddings.open(table, table.shape[1], torch.bfloat16, self.device)
        else:
            # The table serves the device its rows are on (stored there, or copied to registered host memory there).
            table = ck.get(p + "embed_tokens.weight").to(device=self.device, dtype=torch.bfloat16)
            self.embed = self.embeddings.store(table)
        del table
        torch.cuda.empty_cache()
        self.final_norm = on(p + "norm.weight").float()
        self.head = mats.load(["lm_head."], role="head")
        # The drafter's head: the target's own, or a draft-only copy re-encoded at load (the target verifies every
        # draft with its stored head).
        self.draft_head = self.head
        if self.drafter is not None and options["draft_head"] != "shared":
            self.draft_head = mats.load(["lm_head."], role="draft_head", encode=options["draft_head"])
        self.blocks = []
        for i, kind in enumerate(self.types):
            q = f"{p}layers.{i}."
            b = {"in": on(q + "input_layernorm.weight").float(), "post": on(q + "post_attention_layernorm.weight").float(),
                 "gate_up": mats.load([q + "mlp.gate_proj.", q + "mlp.up_proj."], role="gate_up"),
                 "down": mats.load([q + "mlp.down_proj."], role="down")}
            if kind == "linear_attention":
                la = q + "linear_attn."
                b.update(qkvz=mats.load([la + "in_proj_qkv.", la + "in_proj_z."], role="qkvz"),
                         ba=mats.load([la + "in_proj_b.", la + "in_proj_a."], role="ba"),
                         out=mats.load([la + "out_proj."], role="out"),
                         conv=on(la + "conv1d.weight")[:, 0].float().contiguous(),
                         rate=on(la + "A_log").float().exp(), dt_bias=on(la + "dt_bias").float(),
                         norm=on(la + "norm.weight").float())
            else:
                sa = q + "self_attn."
                b.update(qkv=mats.load([sa + "q_proj.", sa + "k_proj.", sa + "v_proj."], role="qkv"),
                         o=mats.load([sa + "o_proj."], role="o"),
                         q_norm=on(sa + "q_norm.weight").float(), k_norm=on(sa + "k_norm.weight").float(),
                         k_scale=float(ck.get(sa + "k_scale").float()) if ck.has(sa + "k_scale") else 1.0,
                         v_scale=float(ck.get(sa + "v_scale").float()) if ck.has(sa + "v_scale") else 1.0)
            self.blocks.append(b)
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.rotary, 2, device=self.device).float() / self.rotary))
        self.mtp = None
        if self.drafter is not None:
            encode = None if options["mtp_encode"] == "none" else options["mtp_encode"]
            m = "mtp.layers.0."
            self.mtp = {"norm_e": on("mtp.pre_fc_norm_embedding.weight").float(),
                        "norm_h": on("mtp.pre_fc_norm_hidden.weight").float(),
                        "fc": mats.load(["mtp.fc."], role="draft", encode=encode),
                        "in": on(m + "input_layernorm.weight").float(),
                        "post": on(m + "post_attention_layernorm.weight").float(),
                        "q": mats.load([m + "self_attn.q_proj."], role="draft", encode=encode),
                        "kv": mats.load([m + "self_attn.k_proj.", m + "self_attn.v_proj."], role="draft", encode=encode),
                        "o": mats.load([m + "self_attn.o_proj."], role="draft", encode=encode),
                        "q_norm": on(m + "self_attn.q_norm.weight").float(),
                        "k_norm": on(m + "self_attn.k_norm.weight").float(),
                        "gate_up": mats.load([m + "mlp.gate_proj.", m + "mlp.up_proj."], role="draft", encode=encode),
                        "down": mats.load([m + "mlp.down_proj."], role="draft", encode=encode),
                        "norm": on("mtp.norm.weight").float()}
            self.mtp_bytes = sum(ck.nbytes(n) for n in ck.names if n.startswith("mtp."))
        self.head_bytes = plan.get("draft_head", plan["lm_head"])
        torch.cuda.empty_cache()

        # -- state --------------------------------------------------------------------------------------------------
        kv_ids = self.full + ([MTP_LAYER] if self.drafter is not None else [])
        self.pool = parts["kv_cache"].pool(kv_ids, self.kv_heads, self.head_dim, kv_tokens, self.device, options["kv"])
        if options["kv"] == "fp8":
            for i in self.full:
                self.pool.set_scales(i, self.blocks[i]["k_scale"], self.blocks[i]["v_scale"])
        self.states = self.recurrence.pool(self.linear, self.nv, self.dk, self.dv, self.device, self.sequences,
                                           widest=self.width, conv=(conv_channels, self.kernel), key_heads=self.nk,
                                           rows=self.round_rows)
        self.geometry = {"keys": self.dk, "values": self.dv, "epsilon": 1e-6, "query_scale": self.dk ** -0.5,
                         "normalize_keys": True, "normalize_queries": True}
        self.admit = PageReservations(self.pool.pages, self.pool.page, self.context, verify_rows=self.width + 1)
        generation = context.model / "generation_config.json"
        eos = json.loads(generation.read_text(encoding="utf-8")).get("eos_token_id") if generation.is_file() else None
        self.stops = sorted(set(eos if isinstance(eos, list) else [eos])) if eos is not None else []
        self.handles, self.slots = 0, []
        self.rounds, self.mtp_head, self.prompt_graphs = None, None, None
        if options["rounds"]:
            from crit_plugins.rounds import Rounds
            pools = [self.pool, self.states]
            if self.drafter is not None:
                self.mtp_head = MTPHead(self, self.round_rows, self.width)
                pools.append(self.mtp_head)
            self.rounds = Rounds(self, self.sampler, rows=self.round_rows, widest=self.width, device=self.device,
                                 drafter=self.drafter, head=self.mtp_head, pools=pools, parts=parts,
                                 uses=["projections", "attention", "kv_cache", "norms", "recurrence", "embeddings"],
                                 stops=STOPS, logprobs=TOP, vocabulary=self.vocabulary, sampled=True,
                                 widths=self.widths, width_policy=self._width_policy(options))
        if options["prompt_graphs"]:
            from crit_plugins.prompt_graphs import PromptGraphs
            buckets = [int(b) for b in options["prompt_buckets"].split(",") if b.strip()]
            self.prompt_graphs = PromptGraphs(self._prompt_forward, device=self.device, buckets=buckets, parts=parts,
                                              uses=["projections", "attention", "kv_cache", "norms", "recurrence",
                                                    "embeddings"], pools=[self.pool, self.states])
            if self.prompt_graphs.problems:
                self.prompt_graphs = None
            else:
                self.prompt_graphs.prepare()

    # -- Core's settings ----------------------------------------------------------------------------------------------
    def info(self):
        drafts = self.drafter is not None
        found = {"implementation": "qwen3_5 hybrid decoder (gated delta + gated attention), crit plugins",
                 "vocabulary": self.vocabulary, "context": self.context,
                 "prefill_chunk": self.options["prefill_chunk"], "max_concurrency": self.concurrency,
                 "max_pending": self.options["max_pending"], "max_batch_tokens": self.options["prefill_chunk"],
                 "max_row_tokens": self.width + 1, "decode_token_work": self.width + 1, "stop_tokens": self.stops,
                 "sampling": dict.fromkeys(CORE_SAMPLING, True), "token_logprobs": True, "max_top_logprobs": TOP,
                 "scoring": True, "query_scoring": True, "score_paths": ["prefill"], "replay": True,
                 "speculative": drafts, "plain": drafts}
        if drafts:
            found.update(draft_width=self.start_width, draft_bytes_per_step=self.mtp_bytes + self.head_bytes)
            if self.rounds is not None:
                found.update(self.rounds.width_facts())
        return found

    def _width_policy(self, options):
        from crit_plugins.rounds import width_policy
        if len(self.widths) == 1 or options["width_policy"] == "fixed":
            return None
        return width_policy(options["width_policy"], start=self.start_width)

    def telemetry(self):
        return self.rounds.telemetry() if self.rounds is not None else {}

    def native_rounds(self, library):
        if self.rounds is None:
            return None
        return self.rounds.native_rounds(library, [self.start_width] * self.rounds.rows, mode="sampled")

    def admission(self, request):
        return self.admit.admission(len(request.prompt), int(request.output_tokens))

    # -- the arithmetic -----------------------------------------------------------------------------------------------
    def _groups(self, batch):
        """Token ranges multiplied apart, (first, end, prompt): each prompt chunk alone (prompt True: the prompt
        GEMM's arithmetic at every length, a chunk no longer than a decode call padded past it), one-token rows
        (decode rows, a round's columns) rows_per_call at a time, so every row's bits are those it has alone whatever
        shares the step, and a prompt row's bits do not depend on the prompt's length."""
        n = self.rows_per_call
        if batch.round:
            return [(first, min(first + n, batch.tokens), False) for first in range(0, batch.tokens, n)]
        groups, run = [], None
        for r in range(batch.rows):
            first, end = batch.starts[r], batch.starts[r + 1]
            if end - first > 1:
                groups.append((first, end, True))
                run = None
            elif run is not None and run[1] == first and run[1] - run[0] < n:
                run = groups[-1] = (run[0], end, False)
            else:
                run = (first, end, False)
                groups.append(run)
        return groups

    def _call(self, x, matrix, prompt, out_dtype):
        rows = x.shape[0]
        if prompt and rows <= self.rows_per_call:
            padded = self.torch.zeros(self.rows_per_call + 1, x.shape[1], dtype=x.dtype, device=x.device)
            padded[:rows] = x
            return self.proj.matmul(padded, matrix, out_dtype=out_dtype)[:rows]
        return self.proj.matmul(x, matrix, out_dtype=out_dtype)

    def _mm(self, x, matrix, groups, out_dtype=None):
        if len(groups) == 1:
            return self._call(x, matrix, groups[0][2], out_dtype)
        return self.torch.cat([self._call(x[a:b], matrix, p, out_dtype) for a, b, p in groups])

    def _norm(self, x, weight, residual=None):
        return self.norms.rms_norm(x, weight, self.eps, offset=1.0, residual=residual)

    def _qk(self, q, k, qw, kw, positions):
        if self.fused:
            return self.norms.rms_norm_rope_pair(q, k, qw, kw, self.eps, positions, self.inv_freq, offset=1.0)
        return (self.norms.rms_norm_rope(q, qw, self.eps, positions, self.inv_freq, offset=1.0),
                self.norms.rms_norm_rope(k, kw, self.eps, positions, self.inv_freq, offset=1.0))

    def _linear(self, i, h, batch, groups):
        torch, t, b = self.torch, h.shape[0], self.blocks[i]
        conv = 2 * self.nk * self.dk + self.nv * self.dv
        qkvz = self._mm(h, b["qkvz"], groups)
        qkv, z = qkvz[:, :conv], qkvz[:, conv:]
        ba = self._mm(h, b["ba"], groups)
        beta, a = ba[:, :self.nv], ba[:, self.nv:]
        mixed = self.recurrence.convolve(i, qkv, b["conv"], None, self.states, batch)
        q, k, v = torch.split(mixed, [self.nk * self.dk, self.nk * self.dk, self.nv * self.dv], dim=-1)
        q, k = q.reshape(t, self.nk, self.dk), k.reshape(t, self.nk, self.dk)   # value head h reads key head h // r
        v = v.reshape(t, self.nv, self.dv)
        decay, beta = self.norms.softplus_decay(a, b["dt_bias"], b["rate"], beta)
        out = self.recurrence.forward(i, q, k, v, decay, beta, self.states, batch, self.geometry)
        gated = self.norms.gated_rms_norm(out.reshape(t, self.nv, self.dv), z.reshape(t, self.nv, self.dv),
                                          b["norm"], self.eps, out_dtype=h.dtype)
        return self._mm(gated.reshape(t, -1), b["out"], groups)

    def _full(self, i, h, batch, slots, groups):
        t, b, hd = h.shape[0], self.blocks[i], self.head_dim
        qkv = self._mm(h, b["qkv"], groups)
        qg = qkv[:, :2 * self.heads * hd].view(t, self.heads, 2 * hd)
        q, gate = qg[..., :hd], qg[..., hd:]
        k = qkv[:, 2 * self.heads * hd:2 * self.heads * hd + self.kv_heads * hd].view(t, self.kv_heads, hd)
        v = qkv[:, 2 * self.heads * hd + self.kv_heads * hd:].view(t, self.kv_heads, hd)
        q, k = self._qk(q, k, b["q_norm"], b["k_norm"], batch.positions_t)
        self.pool.write(i, slots, k, v)
        a = self.attention.forward(i, q, self.pool, batch, scale=hd ** -0.5)
        a = self.norms.sigmoid_mul(a, gate, out_dtype=h.dtype)
        return self._mm(a.reshape(t, -1), b["o"], groups)

    def _forward(self, ids, batch):
        """The final normed hidden of every token of the batch ([tokens][hidden], BF16)."""
        groups = self._groups(batch)
        x = self.embed.lookup(ids)
        slots = self.pool.reserve(batch)
        h = self._norm(x, self.blocks[0]["in"])
        for i, kind in enumerate(self.types):
            mixed = self._linear(i, h, batch, groups) if kind == "linear_attention" else \
                self._full(i, h, batch, slots, groups)
            h, x = self._norm(mixed, self.blocks[i]["post"], residual=x)
            mlp = self._mm(self.norms.silu_mul(self._mm(h, self.blocks[i]["gate_up"], groups)),
                           self.blocks[i]["down"], groups)
            following = self.blocks[i + 1]["in"] if i + 1 < self.layers else self.final_norm
            h, x = self._norm(mlp, following, residual=x)
        return h

    def _head(self, h, draft=False):
        """FP32 logits of rows h, rows_per_call rows a call (each row's bits its own); draft: the drafter's head."""
        n = h.shape[0]
        groups = [(a, min(a + self.rows_per_call, n), False) for a in range(0, n, self.rows_per_call)]
        return self._mm(h, self.draft_head if draft else self.head, groups, out_dtype=self.torch.float32)

    def _mtp(self, tokens, hidden, batch):
        """The MTP layer over a batch: its normed output ([tokens][hidden] BF16), its KV written at the batch's
        positions."""
        torch, m, t, hd = self.torch, self.mtp, tokens.shape[0], self.head_dim
        groups = self._groups(batch)
        e = self.embed.lookup(tokens)
        joined = torch.cat([self.norms.rms_norm(e, m["norm_e"], self.eps, offset=1.0),
                            self.norms.rms_norm(hidden, m["norm_h"], self.eps, offset=1.0)], dim=-1)
        x = self._mm(joined, m["fc"], groups)
        h = self._norm(x, m["in"])
        slots = self.pool.reserve(batch)
        qg = self._mm(h, m["q"], groups).view(t, self.heads, 2 * hd)
        q, gate = qg[..., :hd], qg[..., hd:]
        kv = self._mm(h, m["kv"], groups)
        k = kv[:, :self.kv_heads * hd].view(t, self.kv_heads, hd)
        v = kv[:, self.kv_heads * hd:].view(t, self.kv_heads, hd)
        q, k = self._qk(q, k, m["q_norm"], m["k_norm"], batch.positions_t)
        self.pool.write(MTP_LAYER, slots, k, v)
        a = self.attention.forward(MTP_LAYER, q, self.pool, batch, scale=hd ** -0.5)
        a = self.norms.sigmoid_mul(a, gate, out_dtype=h.dtype)
        h, x = self._norm(self._mm(a.reshape(t, -1), m["o"], groups), m["post"], residual=x)
        mlp = self._mm(self.norms.silu_mul(self._mm(h, m["gate_up"], groups)), m["down"], groups)
        h, x = self._norm(mlp, m["norm"], residual=x)
        return h

    def _mtp_ingest(self, entries):
        """The MTP layer's KV of prompt positions (its input norms, fc and key/value projections only): entries of
        (sequence, first position, the next tokens [n], the target's hiddens [n][hidden])."""
        torch, m, hd = self.torch, self.mtp, self.head_dim
        entries = [e for e in entries if len(e[2])]
        if not entries:
            return
        batch = StepBatch.build([(s, first, len(tokens)) for s, first, tokens, _ in entries], self.device)
        groups = self._groups(batch)
        tokens = torch.tensor([t for _, _, ts, _ in entries for t in ts], dtype=torch.long, device=self.device)
        hidden = torch.cat([h for _, _, _, h in entries])
        e = self.embed.lookup(tokens)
        joined = torch.cat([self.norms.rms_norm(e, m["norm_e"], self.eps, offset=1.0),
                            self.norms.rms_norm(hidden, m["norm_h"], self.eps, offset=1.0)], dim=-1)
        h = self._norm(self._mm(joined, m["fc"], groups), m["in"])
        slots = self.pool.reserve(batch)
        kv = self._mm(h, m["kv"], groups)
        t = kv.shape[0]
        k = kv[:, :self.kv_heads * hd].view(t, self.kv_heads, hd)
        v = kv[:, self.kv_heads * hd:].view(t, self.kv_heads, hd)
        k = self.norms.rms_norm_rope(k, m["k_norm"], self.eps, batch.positions_t, self.inv_freq, offset=1.0)
        self.pool.write(MTP_LAYER, slots, k, v)

    def _prompt_forward(self, batch):
        """A captured prompt chunk's forward over a PromptBatch: every row's final normed hidden."""
        return self._forward(batch.tokens_t, batch)

    def verify(self, view, batch):
        """The round's target pass: logits of every column, [rows][width + 1][vocabulary] FP32; the final hiddens
        kept for the draft head's next catch-up."""
        h = self._forward(batch.tokens_t, batch)
        if self.mtp_head is not None:
            self.mtp_head.verify_hidden[:h.shape[0]] = h
        return self._head(h).view(view.rows, view.width + 1, -1)

    # -- Core's step model --------------------------------------------------------------------------------------------
    def open(self, request):
        self.handles += 1
        session = Session(self.handles, request)
        self.pool.open(session.handle)
        self.states.open(session.handle)
        try:
            session.pages = self.admit.reserve(len(request.prompt), int(request.output_tokens))
        except RuntimeError:
            session.pages = 0
        return session

    def step(self, rows):
        results = [None] * len(rows)
        bound = [i for i, row in enumerate(rows) if row.session.slot is not None]
        if bound:
            scheduled = {rows[i].session.slot for i in bound}
            held = [slot for slot in range(len(self.slots)) if slot not in scheduled]
            width = self.rounds.width(len(self.slots))
            width = self.start_width if width < 0 else width
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
        packed, ids, placed, prompts, hidden = [], [], [], [], {}
        for placement in layout.placements:
            s = rows[placement.item].session
            inputs = [s.pending] if placement.phase == DECODE else s.prompt[s.done:s.done + placement.rows]
            s.phase, s.rows, s.produced, s.graphed = placement.phase, len(inputs), [], False
            if placement.phase == PREFILL and len(inputs) > 1 and self.prompt_graphs is not None:
                out = self.prompt_graphs.run(s.handle, s.done, inputs)   # a captured chunk, or None: eager
                if out is not None:
                    s.graphed = True
                    hidden[placement.item] = out[:len(inputs)].clone()
            if not s.graphed:
                hidden[placement.item] = (len(ids), len(inputs))
                packed.append((s.handle, s.done, len(inputs)))
                ids += inputs
            if placement.phase == PREFILL:
                prompts.append((s, placement.item, len(inputs)))
            placed.append((placement, s, bool(placement.logits)))
        if packed:
            batch = StepBatch.build(packed, self.device)
            h = self._forward(torch.tensor(ids, dtype=torch.int64, device=self.device), batch)
            for item, where in hidden.items():
                if isinstance(where, tuple):
                    hidden[item] = h[where[0]:where[0] + where[1]]
        last = [hidden[placement.item][-1:] for placement, _, has in placed if has]
        if self.mtp is not None and prompts:
            entries = []
            for s, item, n in prompts:
                start = s.done
                rows_h = hidden[item]
                if start > 0 and s.carry is not None:
                    entries.append((s.handle, start - 1, s.prompt[start:start + n],
                                    torch.cat([s.carry[None], rows_h[:n - 1]])))
                else:
                    entries.append((s.handle, start, s.prompt[start + 1:start + n], rows_h[:n - 1]))
                s.carry = rows_h[n - 1].clone()
            self._mtp_ingest(entries)
        logits = self._head(torch.cat(last)) if last else None
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
                    top = torch.topk(row, min(int(s.request.logprobs), TOP))
                    logprobs = [(float(row[token]), list(zip(top.indices.tolist(), top.values.tolist())))]
                j += 1
            results[placement.item] = StepResult(placement.rows if placement.phase == PREFILL else 0, s.produced, has,
                                                 0, placement.rows, logprobs)
        return results

    def _roundable(self, s):
        r, rounds = s.request, self.rounds
        return (rounds is not None and not rounds.problems and len(self.slots) < rounds.rows
                and not r.replay and not r.plain and len(r.stop_tokens) <= STOPS
                and (r.logprobs is None or int(r.logprobs) <= rounds.top))

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
            if not s.graphed:   # a captured chunk wrote its recurrent state in place
                self.states.accept(s.handle, s.rows if kept else 0)
            s.graphed = False
            self.pool.keep(s.handle, s.done)
            s.phase, s.produced = None, []
            if completed and d.accepted and not (d.terminal or d.cancelled or d.discard) and self._roundable(s):
                self._bind(s)
        if trims:
            ran_natively = native is not None and native.last[2]
            if ran_natively:
                rounds.adopt(*native.last[:2])
                emitted = rounds.n_emitted[:len(self.slots)].tolist()
            else:
                emitted = [len(s.produced) for s in self.slots]
            keep = [min(count, trims.get(slot, count)) for slot, count in enumerate(emitted)]
            if keep != emitted:
                rounds.truncate(keep)

    def _bind(self, s):
        s.slot = len(self.slots)
        self.slots.append(s)
        r = s.request
        params = {k: r.sampling[k] for k in CORE_SAMPLING if k in r.sampling}
        self.rounds.bind(s.slot, s.handle, len(s.prompt), s.pending, r.output_tokens - len(s.outputs),
                         len(s.prompt) + r.output_tokens + self.width + 1, stops=r.stop_tokens,
                         params=params, session=s, tokens=s.prompt + [s.pending],
                         outputs=len(s.outputs), logprobs=r.logprobs is not None)
        if self.mtp_head is not None and s.carry is not None:
            self.mtp_head.hid_keep[s.slot, 0] = s.carry

    def retire(self, session, finish):
        if session.slot is not None:
            slot, last = session.slot, len(self.slots) - 1
            self.rounds.unbind(slot)
            if slot != last:
                moved = self.slots[last]
                self.rounds.move(last, slot)
                moved.slot = slot
                self.slots[slot] = moved
            self.slots.pop()
            session.slot = None
        self.pool.close(session.handle)
        self.states.close(session.handle)
        if session.pages:
            self.admit.release(session.pages)
            session.pages = 0
        session.carry = None

    def score(self, ids, first_target, path="", queries=None, top_k=0):
        """A fixed text's scores through the prompt path, a prefill_chunk at a time (each chunk's KV and recurrent
        state kept for the next), the head over logit_rows rows at a time."""
        torch = self.torch
        sequence = ("score", self.handles)
        chunk, wanted = int(self.options["prefill_chunk"]), range(first_target - 1, len(ids) - 1)
        step = int(self.options["logit_rows"])
        targets_all = ids[first_target:]
        target, best, query, top = [], [], [] if queries else None, [] if top_k else None
        self.pool.open(sequence)
        self.states.open(sequence)
        try:
            for start in range(0, wanted.stop, chunk):
                end = min(start + chunk, wanted.stop)
                batch = StepBatch.build([(sequence, start, end - start)], self.device)
                h = self._forward(torch.tensor(ids[start:end], device=self.device), batch)
                rows = [row - start for row in wanted if start <= row < end]
                for a in range(0, len(rows), step):
                    part = rows[a:a + step]
                    logp = torch.log_softmax(self._head(h[torch.tensor(part, device=self.device)]), -1)
                    index = [start + r - (first_target - 1) for r in part]
                    tg = torch.tensor([targets_all[i] for i in index], device=self.device)
                    target += logp.gather(1, tg[:, None])[:, 0].tolist()
                    best += logp.argmax(-1).tolist()
                    if queries:
                        query += [logp[j, torch.tensor(queries[i], device=self.device)].tolist()
                                  for j, i in enumerate(index)]
                    if top_k:
                        found = torch.topk(logp, top_k)
                        top += [list(zip(i, v)) for i, v in zip(found.indices.tolist(), found.values.tolist())]
                self.states.accept(sequence, end - start)
                self.pool.keep(sequence, end)
        finally:
            self.pool.close(sequence)
            self.states.close(sequence)
        return Scores(target, query, best, top)

    def close(self):
        self.blocks.clear()
        self.mtp = None


def create(context, options, parts):
    return Qwen35Model(context, options, parts)
