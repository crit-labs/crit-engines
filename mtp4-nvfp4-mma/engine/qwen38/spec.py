"""Speculative rounds on the device (inference/device-speculative-round, inference/token-sampling's verify rule): the
target verifies the round's k drafts in one k + 1 column step, the round decides on the device how many stand, keeps
the recurrent state of the last standing column, and the MTP drafter proposes the next round's k drafts. One captured
graph per round; the host reads the standing count and the emitted tokens once.

Greedy (temperature 0): a draft stands while it equals the target's argmax, so the text is plain greedy's.
Sampled: drafts are drawn from the drafter's processed law q; draft d at column c stands with probability
min(1, p(d) / q(d)) under the target's processed law p, the first rejected column emits a draw from max(p - q, 0),
and a round whose drafts all stand emits a draw from p at column k (Leviathan et al.), so the text is a sample of p.
Replay: drafts stand while they equal the forced continuation, every emitted token is the continuation's.
Laws are Core's (temperature, top-k, top-p of the top-k weight, min-p), closed within the best 64 candidates: rows
whose law does not close there (top_k 0 or above 64 when sampling) never take this path."""
import torch

from . import sampling as S

C = S.CANDIDATES
ACCEPT, RESIDUAL, DRAFT = 0x51ED27, 0x2B7E15, 0x6A09E6


def laws(logits, params):
    """Processed laws of rows [R, V]: (candidate ids [R, C], probabilities [R, C] float64 summing to 1 over the kept
    set, argmax ids [R])."""
    vals, ids = torch.topk(logits, C, dim=-1)
    t = params.temperature.clamp(min=1e-6)
    w = torch.exp((vals - vals[:, :1]) / t).double()
    rank = torch.arange(C, device=logits.device)
    k = params.top_k
    keep = (rank[None, :] < k) | (k <= 0)
    cum = torch.cumsum(w * keep, -1)
    k_total = cum[:, -1:]
    keep = keep & ((cum - w) < params.top_p.double() * k_total) & (w >= params.min_p.double())
    pw = w * keep
    probs = pw / pw.sum(-1, keepdim=True)
    greedy = torch.where(params.temperature > 0, torch.zeros_like(probs), torch.zeros_like(probs))
    greedy[:, 0] = 1.0
    probs = torch.where(params.temperature > 0, probs, greedy)
    return ids, probs, ids[:, 0]


def draw(ids, probs, u):
    """A token per row from a law over candidates (ids, probs [R, C]) by the uniforms u [R]: the first candidate
    whose cumulative probability exceeds u."""
    cum = torch.cumsum(probs, -1)
    hit = (cum > u[:, None] * cum[:, -1:]) & (probs > 0)
    first = torch.argmax(hit.to(torch.int8), -1)
    return ids.gather(1, first[:, None])[:, 0]


def prob_of(ids, probs, tokens):
    """The law's probability of tokens [R] (0 outside its candidates)."""
    return ((ids == tokens[:, None]) * probs).sum(-1)


class Speculator:
    def __init__(self, model, mtp, params, steps):
        self.m, self.mtp, self.params, self.k = model, mtp, params, int(steps)
        dev = model.device
        k = self.k
        self.cur = torch.zeros(1, dtype=torch.int64, device=dev)          # last emitted token, at position pos
        self.pos = torch.zeros(1, dtype=torch.int64, device=dev)
        self.out_index = torch.zeros(1, dtype=torch.int64, device=dev)    # output index of the token after cur
        self.drafts = torch.zeros(k, dtype=torch.int64, device=dev)
        self.q_ids = torch.zeros(k, C, dtype=torch.int64, device=dev)
        self.q_probs = torch.zeros(k, C, dtype=torch.float64, device=dev)
        self.forced = torch.zeros(k + 1, dtype=torch.int64, device=dev)
        self.replaying = torch.zeros(1, dtype=torch.bool, device=dev)
        self.emitted = torch.zeros(k + 1, dtype=torch.int64, device=dev)
        self.standing = torch.zeros(1, dtype=torch.int64, device=dev)
        self.logits = None
        self.columns = torch.arange(k + 1, device=dev)

    def _draft(self, logits, position):
        """Draft token and its law from MTP logits [1, V] for the token at output index `position` [1]."""
        ids, probs, top = laws(logits, self.params)
        u = S.uniform(self.params.seed ^ DRAFT, position)
        token = torch.where(self.params.temperature > 0, draw(ids, probs, u), top)
        return token, ids[0], probs[0]

    def propose(self, hidden, next_token, pos, out_index):
        """The k drafts after the MTP row at `pos` (hidden [1, H] the target's normed hidden there, next_token [1] the
        token at pos + 1): draft s is the token at pos + 1 + s, output index out_index + s."""
        h, tok = hidden, next_token
        for s in range(self.k):
            logits, h = self.mtp.forward(h, tok, pos + s)
            tok, ids, probs = self._draft(logits, out_index + s + 1)
            self.drafts[s: s + 1].copy_(tok)
            self.q_ids[s].copy_(ids)
            self.q_probs[s].copy_(probs)
        return h

    def round(self):
        self.round_verify()
        self.round_draft()

    def round_verify(self):
        """The verify half of a round from the device state: the target's step over the k + 1 columns, the standing
        count, the emitted tokens and the commit. Captured as a graph; round_draft() follows it."""
        k, p = self.k, self.params
        tokens = torch.cat((self.cur, self.drafts))
        logits, hn = self.m.step(tokens, self.pos)
        self.logits = logits
        ids, probs, top = laws(logits, p)                                  # target laws of the k + 1 columns
        n = self.out_index + self.columns                                  # output index of each column's token
        d = self.drafts
        # standing drafts per rule
        greedy_ok = d == top[:k]
        pd = prob_of(ids[:k], probs[:k], d)
        qd = prob_of(self.q_ids, self.q_probs, d)
        u_acc = S.uniform(p.seed ^ ACCEPT, n[:k])
        sampled_ok = u_acc * qd < pd
        forced_ok = d == self.forced[:k]
        ok = torch.where(self.replaying, forced_ok, torch.where(p.temperature > 0, sampled_ok, greedy_ok))
        j = torch.cumprod(ok.to(torch.int64), 0).sum().view(1)            # leading standing drafts
        # the token at column j: a residual draw where a draft was rejected, a draw from p after all stood
        q_on_p = (ids[:k, :, None] == self.q_ids[:, None, :]).to(torch.float64)
        q_on_p = (q_on_p * self.q_probs[:, None, :]).sum(-1)                # q at p's candidates, [k, C]
        resid = (probs[:k] - q_on_p).clamp(min=0)
        u_res = S.uniform(p.seed ^ RESIDUAL, n[:k])
        tok_res = draw(ids[:k], resid, u_res)
        u_bonus = S.uniform(p.seed, n)
        tok_bonus = draw(ids, probs, u_bonus)
        sampled_tok = torch.cat((tok_res, tok_bonus[k:]))
        new = torch.where(self.replaying, self.forced, torch.where(p.temperature > 0, sampled_tok, top))
        new_tok = new.gather(0, j)
        self.emitted.copy_(torch.where(self.columns < j, torch.cat((d, d[-1:])), new_tok))
        self.standing.copy_(j)
        self.m.commit(j, k + 1)
        self.hn = hn
        self.nxt = torch.where(self.columns < j, torch.cat((d, d[-1:])), new_tok)
        self.new_tok = new_tok
        self.j = j

    def round_draft(self):
        """The drafting half: the drafter's rows at pos .. pos + k with the next tokens as they now stand, row j
        proposing, then k - 1 chained steps; moves the round state on. Captured as a graph after round_verify()."""
        k = self.k
        j, new_tok = self.j, self.new_tok
        mlogits, mh = self.mtp.forward(self.hn, self.nxt, self.pos)
        row_logits = mlogits.index_select(0, j)
        row_h = mh.index_select(0, j)
        pos_next = self.pos + j + 1
        out_next = self.out_index + j + 1
        tok, qi, qp = self._draft(row_logits, out_next)
        self.drafts[0:1].copy_(tok)
        self.q_ids[0].copy_(qi)
        self.q_probs[0].copy_(qp)
        h = row_h
        for s in range(1, k):
            logits_s, h = self.mtp.forward(h, tok, pos_next + s - 1)
            tok, qi, qp = self._draft(logits_s, out_next + s)
            self.drafts[s: s + 1].copy_(tok)
            self.q_ids[s].copy_(qi)
            self.q_probs[s].copy_(qp)
        self.cur.copy_(new_tok)
        self.pos.copy_(pos_next)
        self.out_index.copy_(out_next)
