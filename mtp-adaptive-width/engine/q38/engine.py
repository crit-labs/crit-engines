"""The Python engine (crit_core.native's PythonEngine interface: info, generate, score, close).

Decoding is speculative with the checkpoint's MTP draft layer: one CUDA graph per round drafts K tokens (the drafter's
processed law on the device), verifies K + 1 columns in one target pass, accepts by Core's rule (argmax for greedy
requests, rejection sampling otherwise: crit Inference token-sampling's enqueue_verify), commits the gated-delta
states of the accepted tokens and stages the next round. Plain decoding (one token per pass) stays available."""
import json
import secrets
import time
from pathlib import Path

import torch

from . import kernels as K
from . import spec_kernels as SK
from .model import HIDDEN, VOCAB, Model
from .sampler import MAX_TOP, Sampler
from .weights import Weights

NAME = "q38-27b-nvfp4-crit"
LP_GRAPH = 20          # alternatives the logprob graphs carry (requests asking more run eagerly)
# FP8-stored matrix groups and their share of the FP8 prompt FLOPs (per token: gated-delta in/out projections
# 48 x 115M, attention q/k/v/o 16 x 105M, MLPs of layers 56-63 8 x 267M multiply-adds)
FP8_GROUPS = {"gdn": 48 * 115.3, "attn": 16 * 104.9, "mlp": 8 * 267.4}


def _now():
    return time.perf_counter()


class Engine:
    def __init__(self, configuration):
        cfg = dict(configuration)
        self.configuration = cfg
        self.model_dir = Path(cfg["model"])
        self.context = int(cfg.get("context_tokens") or 262144)
        self.chunk = int(cfg.get("prefill_chunk", 4096))
        self.use_graphs = bool(cfg.get("graphs", True))
        self.speculative = bool(cfg.get("speculative", True))
        # draft widths a request moves between (mtp-decode's adaptive budget): one more after a round that kept every
        # draft, one fewer after a round that kept fewer than half; K is the widest
        self.widths = sorted({int(w) for w in (cfg.get("draft_widths") or [cfg.get("draft_width", 3)])})
        self.K = self.widths[-1]
        self.start_width = int(cfg.get("draft_start", self.widths[len(self.widths) // 2]))
        assert 1 <= self.widths[0] and self.K <= K.MAX_ROWS - 1 and self.start_width in self.widths
        recon = cfg.get("recon") or {}
        self.stop_tokens = list(recon.get("stop_token_ids") or [248046, 248044])
        config = json.loads((self.model_dir / "config.json").read_text())
        torch.cuda.set_device(0)
        self.dev = torch.device("cuda", 0)
        self.mtp_fp8 = bool(cfg.get("mtp_fp8", False))
        self.weights = Weights(self.model_dir, self.dev, config, load_mtp=self.speculative, mtp_fp8=self.mtp_fp8)
        # FP8 per-token prompt activations per FP8 matrix group (the checkpoint's own input scheme); BF16 otherwise
        self.fp8_prompt = {group: bool(cfg.get("fp8_prompt_" + group, False)) for group in FP8_GROUPS}
        self.fp16_mlp = bool(cfg.get("fp16_acc_mlp", False))
        self.model = Model(self.weights, self.context, chunk=self.chunk, fp8_prompt=self.fp8_prompt,
                           fp16_mlp=self.fp16_mlp)
        self.model.mlp_rows = int(cfg.get("mlp_rows", 1024))
        self.model.conv_kernel = bool(cfg.get("conv_kernel", True))
        self.sampler = Sampler(VOCAB, K.MAX_ROWS, self.dev)
        dev, i32 = self.dev, torch.int32
        Kw = self.K
        self.forced = torch.zeros(K.MAX_ROWS, dtype=i32, device=dev)
        self.drafts = torch.zeros(K.MAX_ROWS, dtype=i32, device=dev)
        self.predictions = torch.zeros(K.MAX_ROWS, dtype=i32, device=dev)
        self.emitted = torch.zeros(K.MAX_ROWS, dtype=i32, device=dev)
        self.count = torch.ones(1, dtype=i32, device=dev)
        self.mpos = torch.zeros(1, dtype=i32, device=dev)
        self.spos = torch.zeros(1, dtype=i32, device=dev)
        self.m_tokens = torch.zeros(Kw + 1, dtype=i32, device=dev)
        self.m_hidden = torch.zeros(Kw + 1, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.dh = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.dlogits = torch.zeros(1, VOCAB, dtype=torch.float32, device=dev)
        self.host_tok = torch.zeros(K.MAX_ROWS + 1, dtype=i32, pin_memory=True)
        self.host_lp = torch.zeros(K.MAX_ROWS, 1 + MAX_TOP, dtype=torch.float32, pin_memory=True)
        self.host_ids = torch.zeros(K.MAX_ROWS, MAX_TOP, dtype=i32, pin_memory=True)
        self.host_forced = torch.zeros(K.MAX_ROWS, dtype=i32, pin_memory=True)
        self.graphs = {}
        if self.use_graphs:
            self._capture()
        self._warm_prefill()

    def _warm_prefill(self):
        """Compiles every prefill tile configuration at load (one prompt per row bucket), so no request pays it."""
        for length in (100, 112, 200, 600, 1500, 2048, 4100):
            self._prefill([1000 + (i % 5000) for i in range(length)], draft=self.speculative)
        torch.cuda.synchronize()
        self.model.reset()

    # ------------------------------------------------------------------------------------------- facts
    def info(self):
        facts = {
            "implementation": NAME, "name": NAME, "context": self.context, "stop_tokens": self.stop_tokens,
            "sampling": {"temperature": True, "top_p": True, "top_k": True, "min_p": True, "seed": True,
                         "penalties": True},
            "speculative": self.speculative, "plain": True, "replay": True, "token_logprobs": True,
            "max_top_logprobs": MAX_TOP, "query_scoring": True, "score_paths": ["prefill"],
            "bytes_per_pass": self.weights.bytes_per_pass(), "kv_bytes_per_token": 16 * 2 * 4 * 256,
            "prompt_activations": self._prompt_activations(),
            "exact_variants": {f"fp8 {group} prompt GEMMs exact": {"config": {"fp8_prompt_" + group: False},
                                                                   "description": f"the {group} FP8 matrices' prompt "
                                                                                  "GEMMs back on BF16 activations"}
                               for group, on in self.fp8_prompt.items() if on}
                              | ({"nvfp4 MLP f16 accumulation exact": {"config": {"fp16_acc_mlp": False},
                                                                        "description": "the NVFP4 MLP prompt GEMMs back on "
                                                                                       "BF16 with FP32 accumulation"}}
                                 if self.fp16_mlp else {}),
            "kernel_work": [
                {"match": "_gemv_fp8", "term": "matmul", "format": "fp8"},
                {"match": "_gemv_nvfp4", "term": "matmul", "format": "nvfp4"},
                {"match": "_nvfp4_dequant", "term": "matmul", "format": "nvfp4"},
                {"match": "_mtp_", "term": "draft"},
                {"match": "_attn_decode", "term": "attention"},
                {"match": "_attn_combine", "term": "attention"},
                {"match": "flash", "term": "attention"},
                {"match": "fmha", "term": "attention"},
                {"match": "chunk_", "term": "recurrent"},
                {"match": "_gdn_", "term": "recurrent"},
                {"match": "_conv_", "term": "recurrent"},
                {"match": "conv", "term": "recurrent"},
            ],
            "configuration": dict(self.configuration),
        }
        if self.speculative:
            mtp = self.model.mtp
            facts.update({"draft_width": self.K, "draft_widths": self.widths,
                          "draft_bytes_per_step": mtp.bytes_per_step() + self.weights.lm_head.bytes(),
                          "wider_draft": {"width": 8, "config": {"draft_width": 8, "draft_widths": [8],
                                                                  "draft_start": 8}}})
        return facts

    def _prompt_activations(self):
        on = sum(share for group, share in FP8_GROUPS.items() if self.fp8_prompt[group])
        total = sum(FP8_GROUPS.values())
        if on == 0:
            fp8 = "bf16"
        elif on == total:
            fp8 = "fp8"
        else:
            fp8 = {"fp8": round(on / total, 4), "bf16": round(1 - on / total, 4)}
        return {"nvfp4": "fp16_acc" if self.fp16_mlp else "bf16", "fp8": fp8}

    # ------------------------------------------------------------------------------------------- steps
    def _step(self, mode, n):
        """One plain token: decode, then sample (or take the forced token) and stage it as the next input."""
        m, smp = self.model, self.sampler
        logits = m.decode(1)
        if mode == "sample":
            smp.sample(logits, rows=1, n=n)
            smp.commit(smp.out)
            m.tokens[:1].copy_(smp.out[:1])
        else:
            smp.logprobs(logits, self.forced, rows=1, n=n)
            m.tokens[:1].copy_(self.forced[:1])
        m.pos.add_(1)

    def _round(self, mode, n, Kw):
        """One speculative round of Kw drafts (see the module docstring); mode "spec" or "specreplay". The drafter's
        catch-up always runs the widest round's rows (the previous round may have been wider)."""
        m, smp, mtp = self.model, self.sampler, self.model.mtp
        out = mtp.step(self.K + 1, self.m_tokens, self.m_hidden, self.mpos)
        SK.gather_row(out, self.count, self.dh[0:1], offset=-1)
        K.gemv(self.weights.lm_head, self.dh, out=self.dlogits, partial=m.partial)
        smp.sample_draft(self.dlogits, self.drafts[0:1], 0)
        for j in range(1, Kw):
            SK.add_scalar(m.pos, self.spos, j - 1)
            out = mtp.step(1, self.drafts[j - 1:j], self.dh, self.spos)
            self.dh.copy_(out[0:1])
            K.gemv(self.weights.lm_head, self.dh, out=self.dlogits, partial=m.partial)
            smp.sample_draft(self.dlogits, self.drafts[j:j + 1], j)
        m.tokens[1:Kw + 1].copy_(self.drafts[:Kw])
        logits = m.decode(Kw + 1, verify=True)
        if mode == "specreplay":
            self.predictions[:Kw + 1].copy_(self.forced[:Kw + 1])
            smp.logprobs(logits, self.predictions, rows=Kw + 1, n=n)
        else:
            smp.verify(logits, Kw + 1, self.drafts, self.predictions, n=n)
        SK.accept(self.drafts, self.predictions, self.emitted, self.count, m.tokens[0:1], m.pos, self.mpos, Kw)
        m.commit(Kw + 1, self.count)
        if mode == "spec":
            smp.commit(self.emitted, emitted=self.count)
        self.m_tokens[:Kw + 1].copy_(self.emitted[:Kw + 1].clamp(min=0))
        self.m_hidden[:Kw + 1].copy_(m.xn[:Kw + 1])

    def _capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        self.model.reset()
        self.sampler.set_request(0.0, 1.0, 0, 0.0, 1.0, 0.0, 0.0, 0)
        variants = [("sample", 0, 0), ("sample", LP_GRAPH, 0), ("replay", LP_GRAPH, 0)]
        if self.speculative:
            for w in self.widths:
                variants += [("spec", 0, w), ("spec", LP_GRAPH, w), ("specreplay", LP_GRAPH, w)]
        with torch.cuda.stream(stream):
            for mode, n, w in variants:
                fn = (lambda: self._step(mode, n)) if w == 0 else (lambda: self._round(mode, n, w))
                for _ in range(2):
                    self._reset_frontier()
                    fn()
                stream.synchronize()
                self._reset_frontier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    fn()
                self.graphs[(mode, n, w)] = graph
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.model.reset()

    def _reset_frontier(self):
        self.model.pos.fill_(1)
        self.mpos.fill_(0)
        self.count.fill_(1)
        self.m_tokens.zero_()
        self.forced.zero_()

    def _run(self, mode, n, w=0):
        graph = self.graphs.get((mode, n, w)) if self.use_graphs else None
        if graph is not None:
            graph.replay()
        elif w == 0:
            self._step(mode, n)
        else:
            self._round(mode, n, w)

    def _next_width(self, w, count):
        """The adaptive budget: wider after a round that kept every draft, narrower after one that kept fewer than
        half of them."""
        i = self.widths.index(w)
        if count == w + 1 and i + 1 < len(self.widths):
            return self.widths[i + 1]
        if (count - 1) * 2 < w and i > 0:
            return self.widths[i - 1]
        return w

    # ------------------------------------------------------------------------------------------- generation
    def _prefill(self, prompt, keep_from=None, draft=False):
        """Prefills the prompt from a fresh state in fixed chunks; returns the final-normed rows from keep_from on
        (bf16) when asked, else the last row. With draft, the MTP layer's cache is filled for every prompt row whose
        next token is known."""
        m = self.model
        m.reset()
        ids = torch.tensor(prompt, dtype=torch.int32, device=self.dev)
        P = len(prompt)
        kept = None
        if keep_from is not None:
            kept = torch.empty(P - keep_from, HIDDEN, dtype=torch.bfloat16, device=self.dev)
        last = None
        for c0 in range(0, P, self.chunk):
            c1 = min(P, c0 + self.chunk)
            if c1 - c0 <= K.MAX_ROWS:
                # a short chunk reads the stored weights once through the decode GEMVs instead of widening them
                m.tokens[:c1 - c0].copy_(ids[c0:c1])
                m.pos.fill_(c0)
                m.decode(c1 - c0)
                xn = m.xn[:c1 - c0]
            else:
                xn = m.prefill(ids[c0:c1], c0)
            if draft:
                rows = min(c1, P - 1) - c0
                if rows > 0:
                    m.mtp.prefill(ids[c0 + 1:c0 + 1 + rows], xn[:rows], c0)
            if kept is not None and c1 > keep_from:
                a = max(c0, keep_from)
                kept[a - keep_from:c1 - keep_from].copy_(xn[a - c0:])
            last = xn[-1:]
        return kept if kept is not None else last

    def generate(self, prompt, output_tokens, *, stop_tokens=None, sampling=None, replay=None, plain=False,
                 logprobs=None, on_tokens=None, cancelled=None):
        started = _now()
        prompt = [int(t) for t in prompt]
        if not prompt:
            raise ValueError("empty prompt")
        speculative = self.speculative and not plain
        if len(prompt) + int(output_tokens) > self.context:
            raise ValueError(f"prompt {len(prompt)} + output {output_tokens} exceeds the context {self.context}")
        stops = set(self.stop_tokens if stop_tokens is None else stop_tokens)
        s = dict(sampling or {})
        temperature = float(s.get("temperature") if s.get("temperature") is not None else 1.0)
        top_p = float(s.get("top_p") if s.get("top_p") is not None else 1.0)
        top_k = int(s.get("top_k") or 0)
        min_p = float(s.get("min_p") or 0.0)
        repetition = float(s.get("repetition_penalty") if s.get("repetition_penalty") is not None else 1.0)
        frequency = float(s.get("frequency_penalty") or 0.0)
        presence = float(s.get("presence_penalty") or 0.0)
        seed = s.get("seed")
        seed = int(seed) if seed is not None else secrets.randbits(63)
        n = int(logprobs) if logprobs is not None else 0
        if n > MAX_TOP:
            raise ValueError(f"at most {MAX_TOP} top log-probabilities")
        m, smp = self.model, self.sampler
        smp.set_request(temperature, top_p, top_k, min_p, repetition, frequency, presence, seed, prompt)
        if replay is not None:
            replay = [int(t) for t in replay]
            output_tokens = min(int(output_tokens), len(replay))
        graph_n = LP_GRAPH if (n or replay is not None) else 0
        if n > LP_GRAPH:
            graph_n = n
        P = len(prompt)
        if output_tokens <= 0:
            return [], self._result("length", started, _now(), _now(), None, P, 0)
        xn = self._prefill(prompt, draft=speculative)
        torch.cuda.synchronize()
        prefill_done = _now()
        tokens = []
        accepted = []
        widths = []
        finish = "length"
        first_token_ms = None
        decode_start = None
        # the first token from the prompt's last row
        logits = m.head_rows(xn)
        if replay is None:
            smp.sample(logits, rows=1, n=graph_n)
            smp.commit(smp.out)
            m.tokens[:1].copy_(smp.out[:1])
        else:
            self.forced[:1].fill_(replay[0])
            smp.logprobs(logits, self.forced, rows=1, n=graph_n)
            m.tokens[:1].copy_(self.forced[:1])
        m.pos.fill_(P)
        if speculative:
            self.m_tokens.zero_()
            self.m_tokens[:1].copy_(m.tokens[:1])
            self.m_hidden[:1].copy_(xn)
            self.count.fill_(1)
            self.mpos.fill_(P - 1)
        # deliver the first token
        self.host_tok[:1].copy_(m.tokens[:1], non_blocking=True)
        if n:
            self._copy_logprobs(1, n, graph_n)
        torch.cuda.current_stream().synchronize()
        first_token_ms = (_now() - started) * 1000
        decode_start = _now()
        done = self._deliver([int(self.host_tok[0])], 0, n, tokens, on_tokens, stops, replay is None)
        if done:
            finish = "stop"
        mode = ("spec" if replay is None else "specreplay") if speculative else ("sample" if replay is None else "replay")
        width = self.start_width if speculative else 0
        while not done and len(tokens) < output_tokens:
            if cancelled and cancelled():
                return None, {"cancelled": True}
            if replay is not None:
                i = len(tokens)
                cols = width + 1 if speculative else 1
                chunk = replay[i:i + cols]
                self.host_forced[:cols].zero_()
                self.host_forced[:len(chunk)] = torch.tensor(chunk, dtype=torch.int32)
                self.forced[:cols].copy_(self.host_forced[:cols], non_blocking=True)
            self._run(mode, graph_n, width)
            if speculative:
                self.host_tok[0].copy_(self.count[0], non_blocking=True)
                self.host_tok[1:width + 2].copy_(self.emitted[:width + 1], non_blocking=True)
                if n:
                    self._copy_logprobs(width + 1, n, graph_n)
                torch.cuda.current_stream().synchronize()
                c = int(self.host_tok[0])
                accepted.append(c - 1)
                widths.append(width)
                width = self._next_width(width, c)
                new = [int(t) for t in self.host_tok[1:1 + c]]
            else:
                self.host_tok[:1].copy_(m.tokens[:1], non_blocking=True)
                if n:
                    self._copy_logprobs(1, n, graph_n)
                torch.cuda.current_stream().synchronize()
                new = [int(self.host_tok[0])]
            new = new[:output_tokens - len(tokens)]
            if self._deliver(new, 0, n, tokens, on_tokens, stops, replay is None):
                finish = "stop"
                break
        torch.cuda.synchronize()
        end = _now()
        result = self._result(finish, started, prefill_done, end, first_token_ms, P, len(tokens),
                              decode_start=decode_start)
        if speculative:
            result["accepted"] = accepted
            result["rounds"] = len(accepted)
            result["draft_widths"] = widths
        return tokens, result

    def _copy_logprobs(self, rows, n, width):
        """The first rows' log-probabilities and n alternatives; the device writes `width` alternatives per row."""
        smp = self.sampler
        ids = smp.top_ids.view(-1)[: rows * width].view(rows, width)
        lps = smp.top_lp.view(-1)[: rows * width].view(rows, width)
        self.host_lp[:rows, 0].copy_(smp.chosen[:rows], non_blocking=True)
        self.host_lp[:rows, 1:1 + n].copy_(lps[:, :n], non_blocking=True)
        self.host_ids[:rows, :n].copy_(ids[:, :n], non_blocking=True)

    def _deliver(self, new, row0, n, tokens, on_tokens, stops, stop_on_tokens):
        """Hands the new tokens to the client up to and including a stop token; True when one was hit."""
        hit = False
        if stop_on_tokens:
            for i, t in enumerate(new):
                if t in stops:
                    new = new[:i + 1]
                    hit = True
                    break
        tokens.extend(new)
        if on_tokens and new:
            if n:
                rows = [(float(self.host_lp[row0 + i, 0]),
                         [(int(self.host_ids[row0 + i, j]), float(self.host_lp[row0 + i, 1 + j])) for j in range(n)])
                        for i in range(len(new))]
                on_tokens(new, rows)
            else:
                on_tokens(new)
        return hit

    def _result(self, finish, started, prefill_done, end, first_token_ms, prompt_tokens, count, decode_start=None):
        prefill_ms = (prefill_done - started) * 1000
        decode_ms = (end - decode_start) * 1000 if decode_start else 0.0
        return {"finish": finish,
                "timing": {"prefill_tokens": prompt_tokens, "reused_prompt_tokens": 0, "prefill_ms": prefill_ms,
                           "decode_ms": decode_ms,
                           "first_token_ms": first_token_ms if first_token_ms is not None else prefill_ms,
                           "decode_steps": max(0, count - 1)}}

    # ------------------------------------------------------------------------------------------- scoring
    def score(self, ids, first_target, path="", queries=None, top_k=0, packed=False):
        from crit_core.native import Scores
        ids = [int(t) for t in ids]
        first_target = int(first_target)
        if first_target < 1 or first_target > len(ids):
            raise ValueError("first_target must be in [1, len(ids)]")
        if len(ids) > self.context:
            raise ValueError("text exceeds the context")
        kept = self._prefill(ids[:-1], keep_from=first_target - 1)
        targets = torch.tensor(ids[first_target:], dtype=torch.long, device=self.dev)
        q = None
        if queries:
            q = torch.tensor(queries, dtype=torch.long, device=self.dev)
        out = self.model.score_rows(kept, targets, int(top_k), queries=q)
        target = out["target"].float().cpu()
        query = out["query"].float().cpu().tolist() if q is not None else None
        top = None
        if top_k:
            tv = out["top_v"].float().cpu()
            ti = out["top_i"].to(torch.int32).cpu()
            if packed:
                import array
                from crit_core.serving import ScoreRows
                top = ScoreRows(array.array("f", target.numpy().tobytes()), array.array("i", ti.numpy().tobytes()),
                                array.array("f", tv.numpy().tobytes()), int(top_k))
            else:
                top = [[(int(a), float(b)) for a, b in zip(ri, rv)] for ri, rv in zip(ti.tolist(), tv.tolist())]
        self.model.reset()
        return Scores(target.tolist(), query, None, top)

    def close(self):
        """Releases every device and pinned allocation before the host opens the next engine (Eval reopens the
        package at another context in the same process): the drafter refers back to the model, so the cycle is
        broken and collected before the caches are emptied."""
        import gc
        torch.cuda.synchronize()
        self.graphs.clear()
        model = getattr(self, "model", None)
        if model is not None and model.mtp is not None:
            model.mtp.model = None
            model.mtp = None
        for name, value in list(vars(self).items()):
            if isinstance(value, torch.Tensor) or name in ("model", "sampler", "weights"):
                delattr(self, name)
        del model
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        host_empty = getattr(torch._C, "_host_emptyCache", None)
        if host_empty is not None:
            host_empty()


def open_engine(configuration):
    return Engine(configuration)
