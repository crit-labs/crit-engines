"""Token selection on the device: Core's sampling law (temperature, then top-k, top-p of the top-k weight, min-p
relative to the largest weight, and a counter-based draw), over the best CANDIDATES tokens of the row with the whole
row's weight; a row whose kept set or draw reaches past them is marked unsettled and decided again over the sorted
row. Temperature 0 is the argmax. Every input is a device tensor, so the step captures in a CUDA graph."""
import torch

CANDIDATES = 64
MASK32 = 0xFFFFFFFF


def _mix32(x):
    """A 32-bit integer hash (lowbias32) on int64 tensors holding values below 2**32."""
    x = x & MASK32
    x = (x ^ (x >> 16)) & MASK32
    x = (x * 0x7FEB352D) & MASK32
    x = (x ^ (x >> 15)) & MASK32
    x = (x * 0x846CA68B) & MASK32
    return (x ^ (x >> 16)) & MASK32


def uniform(seed, position):
    """u in [0, 1) from (seed, position), int64 tensors: two hashed 32-bit words, 53 bits."""
    hi = _mix32(_mix32(seed & MASK32) ^ _mix32((seed >> 32) & MASK32) ^ position * 2 + 1)
    lo = _mix32(hi ^ _mix32(position * 2 + 2) ^ 0x9E3779B9)
    return ((hi >> 11).double() * 2.0 ** 32 + lo.double()) / 2.0 ** 53


class Params:
    """A request's sampling law as device tensors (shared by the captured decode graph)."""

    def __init__(self, device):
        self.temperature = torch.zeros(1, dtype=torch.float32, device=device)
        self.top_k = torch.zeros(1, dtype=torch.int64, device=device)
        self.top_p = torch.ones(1, dtype=torch.float32, device=device)
        self.min_p = torch.zeros(1, dtype=torch.float32, device=device)
        self.seed = torch.zeros(1, dtype=torch.int64, device=device)

    def set(self, temperature, top_k, top_p, min_p, seed):
        self.temperature.fill_(float(temperature))
        self.top_k.fill_(int(top_k))
        self.top_p.fill_(float(top_p))
        self.min_p.fill_(float(min_p))
        self.seed.fill_(int(seed) & ((1 << 63) - 1))


def select(logits, params, position):
    """logits FP32 [1, V]; position int64 [1] (the output index the draw is keyed by). Returns (token int64 [1],
    unsettled bool [1])."""
    vals, ids = torch.topk(logits[0], CANDIDATES)
    greedy = ids[:1]
    t = params.temperature.clamp(min=1e-6)
    mx = vals[:1]
    w = torch.exp((vals - mx) / t).double()                               # candidates' weights, the best is 1
    total = torch.exp((logits[0] - mx) / t).double().sum()                # the whole row's weight
    rank = torch.arange(CANDIDATES, device=logits.device)
    k = params.top_k
    limited = (k > 0) & (k <= CANDIDATES)
    keep_k = torch.where(limited, rank < k, torch.ones_like(rank, dtype=torch.bool))
    k_total = torch.where(limited, (w * keep_k).sum(), total)
    cum = torch.cumsum(w * keep_k, 0)
    # top-p: the shortest prefix whose weight reaches top_p of the top-k weight
    before = cum - w
    keep_p = before < params.top_p.double() * k_total
    keep_min = w >= params.min_p.double()
    keep = keep_k & keep_p & keep_min
    kept_mass = (w * keep).sum()
    p_closed = (params.top_p < 1.0) & ~keep_p[-1:]
    min_closed = (params.min_p > 0) & ~keep_min[-1:]
    closed = limited | p_closed | min_closed
    # the set is complete within the candidates when a filter closes it there; otherwise its mass is the row's
    mass = torch.where(closed, kept_mass, total)
    u = uniform(params.seed, position)
    target = u * mass
    cumk = torch.cumsum(w * keep, 0)
    hit = (cumk > target) & keep
    first = torch.argmax(hit.to(torch.int8))
    found = hit.any()
    sampled = ids.gather(0, first.view(1))
    big_k = (k > CANDIDATES) & (k < logits.shape[1])
    unsettled = (big_k | ~found.view(1)) & (params.temperature > 0)
    token = torch.where(params.temperature > 0, sampled, greedy)
    return token, unsettled


def select_full(logits, temperature, top_k, top_p, min_p, seed, position):
    """The same law over the whole sorted row (host-driven; for rows the candidates do not settle)."""
    row = logits[0].double()
    if temperature <= 0:
        return int(torch.argmax(logits[0]).item())
    vals, ids = torch.sort(row, descending=True, stable=True)
    w = torch.exp((vals - vals[0]) / temperature)
    if 0 < top_k < w.numel():
        w = w[:top_k]
    total = w.sum()
    if top_p < 1.0:
        cum = torch.cumsum(w, 0)
        n = int(((cum - w) < top_p * total).sum().item())
        w = w[:max(1, n)]
    if min_p > 0:
        w = w * (w >= min_p)
    cum = torch.cumsum(w, 0)
    u = uniform(torch.tensor([seed], dtype=torch.int64, device=logits.device),
                torch.tensor([position], dtype=torch.int64, device=logits.device))
    idx = int(((cum > u * cum[-1]) & (w > 0)).to(torch.int8).argmax().item())
    return int(ids[idx].item())


def logprob_rows(logits, tokens, n):
    """The model's own log-probabilities (raw logits, before temperature and filters) of `tokens` (int64 [R]) and
    the n most likely alternatives per row: (lp [R], top_ids [R, n], top_lp [R, n]) on the host."""
    lsm = torch.log_softmax(logits.double(), -1)
    lp = lsm.gather(1, tokens.view(-1, 1)).view(-1)
    if n > 0:
        top_lp, top_ids = torch.topk(lsm, n, dim=-1)
    else:
        top_lp = top_ids = torch.empty(logits.shape[0], 0, device=logits.device)
    return lp.cpu().tolist(), top_ids.cpu().tolist(), top_lp.cpu().tolist()
