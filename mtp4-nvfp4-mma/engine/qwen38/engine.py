"""The engine behind crit_core.native's stock adapters (a Python engine with NativeEngine's interface): info(),
generate(), score() and close(). One request at a time; every request prefills its whole prompt (no prompt cache).
Requests whose law closes within the best 64 candidates (greedy, top_k 1..64) and replays decode in speculative
rounds with the checkpoint's MTP drafter; the others, and plain requests, one token per pass."""
import array
import secrets
import time

import torch

from . import sampling as S
from .model import HD, KVH, LAYERS, Model, is_attention
from .mtp import Mtp
from .spec import Speculator

STOP_TOKENS = [248046, 248044]
NAME = "qwen38-27b-nvfp4-crit"


class Engine:
    def __init__(self, configuration):
        self.configuration = dict(configuration)
        model_dir = self.configuration["model"]
        context = int(self.configuration.get("context_tokens") or 262144)
        torch.cuda.set_device(0)
        self.device = torch.device("cuda", 0)
        self.model = Model(model_dir, context, self.device, int(self.configuration.get("prefill_chunk", 2048)))
        self.context = context
        self.steps = int(self.configuration.get("draft_steps", 4))
        self.speculative = bool(self.configuration.get("speculative", True)) and self.steps > 0
        self.model.reserve_columns(self.steps + 1 if self.speculative else 1)
        self.model.pad_rows = 1                 # the NVFP4 GEMV is column-invariant: no padding
        self.params = S.Params(self.device)
        self.mtp = Mtp(self.model, model_dir) if self.speculative else None
        self.spec = Speculator(self.model, self.mtp, self.params, self.steps) if self.speculative else None
        self.token = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.pos = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.zero = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.out_index = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.graphs = bool(self.configuration.get("cuda_graphs", True))
        self.plain_graph = self.verify_graph = self.draft_graph = None
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        self.buckets = [int(b) for b in self.configuration.get("prefill_buckets", [128, 512])
                        if int(b) <= self.model.chunk] if self.graphs else []
        self.prefill_graphs = {}
        self._capture()
        self._capture_prefill()

    # -- facts ----------------------------------------------------------------------------------------------------

    def info(self):
        kv_bytes = sum(1 for i in range(LAYERS) if is_attention(i)) * KVH * HD * 2
        facts = {"implementation": NAME, "name": NAME, "context": self.context, "stop_tokens": list(STOP_TOKENS),
                 "sampling": {"temperature": True, "top_p": True, "top_k": True, "min_p": True, "seed": True},
                 "speculative": self.speculative, "plain": self.speculative, "replay": True, "token_logprobs": True,
                 "max_top_logprobs": 20, "query_scoring": True, "score_paths": ["prefill"],
                 "bytes_per_pass": int(self.model.w.bytes_per_pass()), "kv_bytes_per_token": kv_bytes,
                 "prompt_activations": {"nvfp4": "stored", "fp8": "fp8"}, "kernel_work": KERNEL_WORK}
        if self.speculative:
            facts.update(draft_width=self.steps, draft_bytes_per_step=int(self.mtp.bytes_per_step()))
        return facts

    # -- graphs ---------------------------------------------------------------------------------------------------

    def _plain_step(self):
        self.logits, _ = self.model.step(self.token, self.pos)
        self.model.commit(self.zero, 1)
        self.next_token, self.unsettled = S.select(self.logits, self.params, self.out_index)

    def _capture(self):
        if not self.graphs:
            return
        self.model.reset()
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._plain_step()
                if self.speculative:
                    self.spec.round()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.plain_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.plain_graph):
            self._plain_step()
        if self.speculative:
            self.verify_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.verify_graph, pool=self.plain_graph.pool()):
                self.spec.round_verify()
            self.draft_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.draft_graph, pool=self.plain_graph.pool()):
                self.spec.round_draft()
        torch.cuda.synchronize(self.device)
        self.model.reset()

    # -- captured prefill -----------------------------------------------------------------------------------------

    def _prefill_body(self, b):
        """One chunk of bucket b rows from the static buffers (ids, start position, real rows), with the drafter's
        keys and values; leaves the last real row's logits and normed hidden state."""
        g = self.prefill_graphs[b]
        h = self.model.prefill_chunk(g["ids"], g["p0"], rows=g["n"])
        _, _, hn = self.model._norm_quant(h, self.model.w.final_norm, want_y=True)
        if self.speculative:
            self.mtp.ingest(hn, g["next"], g["p0"])
        last = (g["n"] - 1).clamp(min=0)
        g["hn"] = hn.index_select(0, last)
        g["logits"] = self.model.logits(h.index_select(0, last))

    def _capture_prefill(self):
        if not self.buckets:
            return
        pool = None
        for b in self.buckets:
            self.prefill_graphs[b] = {"ids": torch.zeros(b, dtype=torch.int64, device=self.device),
                                      "next": torch.zeros(b, dtype=torch.int64, device=self.device),
                                      "p0": torch.zeros(1, dtype=torch.int64, device=self.device),
                                      "n": torch.full((1,), b, dtype=torch.int64, device=self.device)}
            self.model.reset()
            stream = torch.cuda.Stream(self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._prefill_body(b)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                self._prefill_body(b)
            pool = graph.pool() if pool is None else pool
            self.prefill_graphs[b]["graph"] = graph
        torch.cuda.synchronize(self.device)
        self.model.reset()

    def _prefill_captured(self, ids, cancelled):
        """A prompt of at most the largest bucket through the smallest captured bucket that holds it. Returns (logits,
        normed hidden) of the last prompt position, or None when cancelled. Longer prompts prefill eagerly (their
        host launch time hides under the GPU's), with a last chunk that fits a bucket padded the same way."""
        n = len(ids)
        t = torch.tensor(ids + [0], dtype=torch.int64, device=self.device)
        self.model.reset()
        out = None
        p0 = 0
        while p0 < n:
            if cancelled is not None and cancelled():
                return None
            rows = min(self.model.chunk, n - p0)
            b = next(x for x in self.buckets if x >= rows)
            g = self.prefill_graphs[b]
            g["ids"].zero_()
            g["ids"][:rows].copy_(t[p0: p0 + rows])
            g["next"].zero_()
            g["next"][:rows].copy_(t[p0 + 1: p0 + 1 + rows])
            g["p0"].fill_(p0)
            g["n"].fill_(rows)
            g["graph"].replay()
            out = (g["logits"], g["hn"])
            p0 += rows
        return out

    def _plain(self):
        if self.plain_graph is not None:
            self.plain_graph.replay()
        else:
            self._plain_step()

    def _round(self):
        """One round; returns its verify and draft device milliseconds."""
        e = self.events
        e[0].record()
        if self.verify_graph is not None:
            self.verify_graph.replay()
        else:
            self.spec.round_verify()
        e[1].record()
        if self.draft_graph is not None:
            self.draft_graph.replay()
        else:
            self.spec.round_draft()
        e[2].record()

    # -- requests -------------------------------------------------------------------------------------------------

    def _bucket(self, p0, rows):
        """The padded row count a chunk of `rows` real rows at p0 runs at: the smallest capture bucket that holds it
        and stays inside the context, else the rows themselves."""
        b = next((x for x in self.buckets if x >= rows), rows)
        return b if p0 + b <= self.context else rows

    def _chunk(self, t, p0, rows):
        """Rows p0 .. p0 + rows of t (with one token after them) padded to their bucket: (ids, next ids, rows tensor)."""
        b = self._bucket(p0, rows)
        ids = torch.zeros(b, dtype=torch.int64, device=self.device)
        nxt = torch.zeros(b, dtype=torch.int64, device=self.device)
        ids[:rows] = t[p0: p0 + rows]
        nxt[:rows] = t[p0 + 1: p0 + 1 + rows]
        return ids, nxt, torch.full((1,), rows, dtype=torch.int64, device=self.device)

    def _prefill(self, ids, cancelled=None, drafter=False):
        """Prefill the prompt from scratch (with the drafter's keys and values when it drafts) in the captured
        graphs' padded buckets; returns the last position's logits and its normed hidden state, or None."""
        self.model.reset()
        n = len(ids)
        t = torch.tensor(ids + [0], dtype=torch.int64, device=self.device)
        out = None
        for p0 in range(0, n, self.model.chunk):
            if cancelled is not None and cancelled():
                return None
            rows = min(self.model.chunk, n - p0)
            cids, nxt, n_t = self._chunk(t, p0, rows)
            pos = torch.tensor([p0], dtype=torch.int64, device=self.device)
            h = self.model.prefill_chunk(cids, pos, rows=n_t)
            _, _, hn = self.model._norm_quant(h, self.model.w.final_norm, want_y=True)
            if drafter:
                self.mtp.ingest(hn, nxt, pos)
            out = (self.model.logits(h[rows - 1: rows]), hn[rows - 1: rows])
        return out

    def generate(self, prompt, output_tokens, *, stop_tokens=None, sampling=None, replay=None, plain=False,
                 logprobs=None, on_tokens=None, cancelled=None):
        prompt = [int(t) for t in prompt]
        if not prompt:
            raise ValueError("empty prompt")
        output_tokens = int(output_tokens)
        if len(prompt) + output_tokens > self.context:
            raise ValueError(f"prompt ({len(prompt)}) plus output ({output_tokens}) exceeds the context {self.context}")
        stops = set(STOP_TOKENS if stop_tokens is None else stop_tokens)
        sampling = dict(sampling or {})
        temperature = float(sampling["temperature"]) if sampling.get("temperature") is not None else 1.0
        top_k = int(sampling.get("top_k") or 0)
        top_p = float(sampling["top_p"]) if sampling.get("top_p") is not None else 1.0
        min_p = float(sampling.get("min_p") or 0.0)
        seed = sampling.get("seed")
        seed = int(seed) if seed is not None else secrets.randbits(63)
        self.params.set(temperature, top_k, top_p, min_p, seed)
        forced = list(replay) if replay is not None else None
        if forced is not None:
            output_tokens = min(output_tokens, len(forced))
        spec = (self.speculative and not plain and output_tokens > 1
                and (temperature <= 0 or forced is not None or 1 <= top_k <= S.CANDIDATES))
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        captured = (bool(self.prefill_graphs) and len(prompt) <= self.buckets[-1]
                    and len(prompt) + self.buckets[-1] <= self.context)
        got = self._prefill_captured(prompt, cancelled) if captured else self._prefill(prompt, cancelled, drafter=spec)
        if got is None:
            return None, {"cancelled": True}
        logits, hn = got
        n_prompt = len(prompt)
        tokens, accepted = [], []
        n_logprobs = int(logprobs) if logprobs is not None else None

        def rows_for(rows_logits, toks):
            lp, ti, tl_ = S.logprob_rows(rows_logits, torch.tensor(toks, device=self.device), n_logprobs)
            return [(lp[i], list(zip(ti[i], tl_[i]))) for i in range(len(toks))]

        def emit(new, rows_logits):
            if on_tokens is None or not new:
                return
            if n_logprobs is not None:
                on_tokens(new, rows_for(rows_logits, new))
            else:
                on_tokens(new)

        if forced is not None:
            token = forced[0]
        else:
            self.out_index.fill_(0)
            tok, unsettled = S.select(logits, self.params, self.out_index)
            token = (S.select_full(logits, temperature, top_k, top_p, min_p, seed, 0) if bool(unsettled.item())
                     else int(tok.item()))
        torch.cuda.synchronize(self.device)
        first_ms = (time.perf_counter() - start) * 1000
        tokens.append(token)
        emit([token], logits)
        decode_start = time.perf_counter()
        steps = 0
        verify_ms = draft_ms = 0.0
        if spec and token not in stops and len(tokens) < output_tokens:
            sp = self.spec
            sp.cur.fill_(token)
            sp.pos.fill_(n_prompt)
            sp.out_index.fill_(1)
            sp.replaying.fill_(forced is not None)
            sp.propose(hn, sp.cur, torch.tensor([n_prompt - 1], dtype=torch.int64, device=self.device),
                       torch.zeros(1, dtype=torch.int64, device=self.device))
            k = self.steps
            while len(tokens) < output_tokens and tokens[-1] not in stops:
                if cancelled is not None and cancelled():
                    return None, {"cancelled": True}
                if forced is not None:
                    i = len(tokens)
                    window = forced[i: i + k + 1]
                    window += [window[-1]] * (k + 1 - len(window))
                    sp.forced.copy_(torch.tensor(window, dtype=torch.int64, device=self.device))
                self._round()
                steps += 1
                out = torch.cat((sp.standing, sp.emitted)).tolist()
                verify_ms += self.events[0].elapsed_time(self.events[1])
                draft_ms += self.events[1].elapsed_time(self.events[2])
                j, emitted = out[0], out[1: 2 + out[0]]
                accepted.append(j)
                new = []
                for tok in emitted:
                    if len(tokens) >= output_tokens:
                        break
                    tokens.append(tok)
                    new.append(tok)
                    if tok in stops:
                        break
                emit(new, sp.logits[: len(new)] if n_logprobs is not None else None)
        else:
            position = n_prompt
            while len(tokens) < output_tokens and tokens[-1] not in stops:
                if cancelled is not None and cancelled():
                    return None, {"cancelled": True}
                n = len(tokens)
                self.token.fill_(tokens[-1])
                self.pos.fill_(position)
                self.out_index.fill_(n)
                self._plain()
                steps += 1
                position += 1
                if forced is not None:
                    token = forced[n]
                elif bool(self.unsettled.item()):
                    token = S.select_full(self.logits, temperature, top_k, top_p, min_p, seed, n)
                else:
                    token = int(self.next_token.item())
                tokens.append(token)
                emit([token], self.logits)
        finish = "stop" if tokens and tokens[-1] in stops else "length"
        decode_ms = (time.perf_counter() - decode_start) * 1000
        timing = {"prefill_tokens": n_prompt, "reused_prompt_tokens": 0, "prefill_ms": first_ms,
                  "decode_ms": decode_ms, "first_token_ms": first_ms, "decode_steps": steps}
        result = {"finish": finish, "timing": timing}
        if spec:
            result["accepted"] = accepted
            result["rounds"] = steps
            result["verify_ms"] = verify_ms
            result["draft_ms"] = draft_ms
        return tokens, result

    def score(self, ids, first_target, path="", queries=None, top_k=0, packed=False):
        """Log-probabilities of ids[first_target:] from one prefill of ids (each from the position before it), with
        the query rows, the argmax and the engine's own top_k alternatives."""
        from crit_core.native import Scores
        from crit_core.serving import ScoreRows
        ids = [int(t) for t in ids]
        first_target = int(first_target)
        if first_target < 1 or first_target > len(ids):
            raise ValueError("first_target needs a preceding token")
        if len(ids) > self.context:
            raise ValueError("score text exceeds the context")
        self.model.reset()
        t = torch.tensor(ids + [0], dtype=torch.int64, device=self.device)
        q = torch.tensor(queries, dtype=torch.int64, device=self.device) if queries else None
        targets, query_rows, argmax, top_ids, top_lp = [], [], [], [], []
        sub = 128
        for p0 in range(0, len(ids), self.model.chunk):
            rows = min(self.model.chunk, len(ids) - p0)
            cids, _, n_t = self._chunk(t, p0, rows)
            h = self.model.prefill_chunk(cids, torch.tensor([p0], dtype=torch.int64, device=self.device), rows=n_t)
            lo = max(p0, first_target - 1)
            hi = min(p0 + rows, len(ids) - 1)
            for s0 in range(lo, hi, sub):
                s1 = min(hi, s0 + sub)
                logits = self.model.logits(h[s0 - p0: s1 - p0])
                lsm = torch.log_softmax(logits, -1)
                tgt = t[s0 + 1: s1 + 1]
                targets.append(lsm.gather(1, tgt.view(-1, 1)).view(-1))
                argmax.append(torch.argmax(logits, -1))
                if q is not None:
                    rows = q[s0 + 1 - first_target: s1 + 1 - first_target]
                    query_rows.append(lsm.gather(1, rows))
                if top_k:
                    v, i = torch.topk(lsm, top_k, dim=-1)
                    top_ids.append(i.int())
                    top_lp.append(v)
                del logits, lsm
        target = torch.cat(targets).cpu()
        am = torch.cat(argmax).cpu().tolist()
        qr = torch.cat(query_rows).cpu().tolist() if q is not None else None
        top = None
        if top_k:
            ti = torch.cat(top_ids).cpu()
            tv = torch.cat(top_lp).float().cpu()
            if packed:
                top = ScoreRows(array.array("f", target.float().numpy().tobytes()),
                                array.array("i", ti.numpy().tobytes()), array.array("f", tv.numpy().tobytes()),
                                top_k)
            else:
                top = [list(zip(a, b)) for a, b in zip(ti.tolist(), tv.tolist())]
        self.model.reset()
        return Scores(target.tolist(), qr, am, top)

    def close(self):
        self.plain_graph = self.verify_graph = self.draft_graph = None
        self.prefill_graphs = {}
        self.spec = self.mtp = self.model = None
        torch.cuda.empty_cache()


# Kernel classes for crit eval kernels (prefill-floor terms and weight formats).
KERNEL_WORK = [
    {"match": "attn_prefill_kernel", "term": "attention"},
    {"match": "attn_step", "term": "attention"},
    {"match": "qk_rope_kv", "term": "attention"},
    {"match": "chunk_gated_delta", "term": "recurrent"},
    {"match": "chunk_fwd", "term": "recurrent"},
    {"match": "gated_delta", "term": "recurrent"},
    {"match": "recompute_w_u", "term": "recurrent"},
    {"match": "solve_tril", "term": "recurrent"},
    {"match": "chunk_scaled_dot_kkt", "term": "recurrent"},
    {"match": "chunk_local_cumsum", "term": "recurrent"},
    {"match": "l2norm", "term": "recurrent"},
    {"match": "conv", "term": "recurrent"},
]
