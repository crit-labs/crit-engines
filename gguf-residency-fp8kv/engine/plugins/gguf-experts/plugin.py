"""Routed SwiGLU experts over a GGUF's stored expert blocks, every expert at an address: a device slot or pinned host
memory read in place over the link.

A layer is its experts' fused gate|up rows [2 * inter][row bytes] and down rows [hidden][row bytes] in their GGUF formats
(the checkpoint's bytes as stored). load_layer(layer, gate, up, down, kinds, hidden, inter, resident) puts the experts
`resident` names in device slots and every other expert in pinned host memory, and keeps one address table per matrix
(int64 [experts]: the slot's device address, or the host memory's mapped device address), so a kernel reads every expert
where it lives with no staging copy, and a residency change is a copy plus a table entry.

forward(layer, x, ids, weights): decode rows (up to DECODE_ROWS tokens) run every (token, route) pair through the
gathered GEMV (gguf_gemv.cu gathered_gemv_q8p: the stored formats' int8 activation units, dp4a), silu(gate) * up, the
down GEMV, and each token's routes summed in route order: capture-safe, no host synchronization. Prompt rows sort the
routes by expert on the device and run each expert's routes through the int8 tensor-core kernel over its blocks
(mmqp); a Q6_K down matrix (two sub-scales a 32-unit) decodes its expert to BF16 instead. Each token's sum is in route
order, so a token's bits never depend on the rows beside it. The host form is the same block in FP64 over dense
matrices. telemetry() counts routes read and routes whose expert was on the device (expert_reads, expert_hits)."""
import itertools
import math

from .gguf import gguf_blocks, kernels
from .tier import tier_kernels

DECODE_ROWS = 8
MMQ_ROWS = 32
TIER_SLOTS = 64           # mailbox slots (one a routed layer)
KIND_IDS = {"IQ4_XS": 2, "IQ3_S": 3, "Q6_K": 5}     # rows a tile of the int8 tensor-core kernel takes (gguf_gemv.cu MQ_BM)


def _silu(value):
    return value / (1.0 + math.exp(-value))


SLACK = 256   # bytes past a store's last block: the kernels' unaligned word reads may touch a few beyond a matrix


class _Padded:
    pass


def _padded(torch, shape, device):
    """A uint8 tensor of `shape` at the front of an allocation SLACK bytes longer (base_buffer: the allocation)."""
    count = 1
    for d in shape:
        count *= d
    flat = torch.empty(count + SLACK, dtype=torch.uint8, device=device)
    view = flat[:count].view(shape)
    view.base_buffer = flat
    return view


def _register(torch, buffer):
    """Page-lock a host tensor in place and map it for the device (cudaHostRegisterPortable | Mapped)."""
    error = torch.cuda.cudart().cudaHostRegister(buffer.data_ptr(), buffer.numel() * buffer.element_size(), 3)
    if int(error) != 0:
        raise MemoryError(f"cudaHostRegister of {buffer.numel() * buffer.element_size()} bytes failed ({error})")


class Layer:
    def __init__(self, torch, experts, hidden, inter, gu_kind, dn_kind, gu_rb, dn_rb):
        self.experts, self.hidden, self.inter = experts, hidden, inter
        self.gu_kind, self.dn_kind, self.gu_rb, self.dn_rb = gu_kind, dn_kind, gu_rb, dn_rb
        self.gu_ptr = self.dn_ptr = self.resident = None
        self.device_gu = self.device_dn = self.host_gu = self.host_dn = None
        self.slot_of, self.host_of = {}, {}

    def view(self, expert, which):
        """(tensor, on_device) of one expert's matrix rows (which: "gu" or "dn")."""
        if expert in self.slot_of:
            store = self.device_gu if which == "gu" else self.device_dn
            return store[self.slot_of[expert]], True
        store = self.host_gu if which == "gu" else self.host_dn
        return store[self.host_of[expert]], False


class GgufExperts:
    def __init__(self, decode="stored", prompt="stored", cpu_tier=True, workers=6):
        import torch
        self.torch, self.decode, self.prompt = torch, decode, prompt
        self.tier_module = tier_kernels.load() if cpu_tier else None
        self.workers = workers
        self.tier = None
        self.tier_offsets = None
        self.stage_experts = 16       # host experts a prompt staging batch copies (0: read them over the link in place)
        self.adapt_swaps, self.adapt_decay, self.adapt_min, self.adapt_margin = 8, 0.8, 3.0, 2.0
        # a fresh engine's first adaptations: up to warm_swaps each for warm_calls calls, then adapt_swaps
        self.warm_swaps, self.warm_calls, self.adapt_calls = 0, 0, 0
        # a prompt's routes as the placement's first evidence of the request (prompt_warm()): each layer's prompt
        # routes, added to its route counts as prompt_rows decode rows' worth, then one adaptation of up to
        # prompt_swaps swaps (0: off)
        self.prompt_rows, self.prompt_swaps = 0, 0
        self.prompt_counts, self.prompt_tokens, self.prompt_first = {}, 0, None
        self.swaps = 0
        # the adaptive residency's state of every layer in one tensor each (adapt() reads them at once): route counts
        # on the device, decayed scores and the placement masks on the host, a row a layer (layer index)
        self.route_all = self.score_all = self.dev_mask = self.host_mask = self.lent_mask = None
        self.adapt_rows = []
        self.lent_experts = self.reclaimed = 0
        self.stage = self.stage_events = self.copy_stream = None
        # cross-layer prefetch of a prompt step's host experts (prefetch_experts a layer, two buffers)
        self.prefetch_experts, self.prefetch_min_tokens = 0, 128
        self.stage_q6k = True           # the prompt path's Q6_K down rows of host experts from staged copies
        self.pref_buffers = self.pref_events = self.pref_stream = None
        self.pref_entry = None          # (layer, tokens, buffer, {expert: index}) issued for the next prompt layer
        self.pref_next = 0
        self.layers, self.dense = {}, {}
        self.kernels = kernels.load()
        self.counts = None
        self.calls = 0
        self.registered = []

    # -- host form ---------------------------------------------------------------------------------------------------
    def run(self, layer, x, ids, weights):
        out = []
        for t, row in enumerate(x):
            total = [0.0] * len(row)
            for e, w in zip(ids[t], weights[t]):
                if e < 0:
                    continue
                gate_up, down = self.dense[(layer, e)]
                inter = len(gate_up) // 2
                h = [sum(a * b for a, b in zip(r, row)) for r in gate_up]
                g = [_silu(h[i]) * h[inter + i] for i in range(inter)]
                for j, r in enumerate(down):
                    total[j] += w * sum(a * b for a, b in zip(r, g))
            out.append(total)
        return out

    def load(self, layer, expert, gate_up, down, format="bf16"):
        """One expert's dense matrices: the host form's experts (tests)."""
        self.dense[(layer, expert)] = (gate_up.tolist() if hasattr(gate_up, "tolist") else gate_up,
                                       down.tolist() if hasattr(down, "tolist") else down)

    # -- device form -------------------------------------------------------------------------------------------------
    def load_layer(self, layer, gate, up, down, gu_kind, dn_kind, hidden, inter, resident):
        """gate / up uint8 [experts][inter][row bytes] and down [experts][hidden][row bytes] (host views of the
        checkpoint), gu_kind / dn_kind their GGUF kinds; resident: the expert ids that get device slots."""
        torch = self.torch
        experts = gate.shape[0]
        gu_rb, dn_rb = gguf_blocks.row_bytes(gu_kind, hidden), gguf_blocks.row_bytes(dn_kind, inter)
        if tuple(gate.shape) != (experts, inter, gu_rb) or tuple(up.shape) != tuple(gate.shape):
            raise ValueError(f"layer {layer}: gate/up rows do not hold {inter} x {hidden} {gu_kind}")
        if tuple(down.shape) != (experts, hidden, dn_rb):
            raise ValueError(f"layer {layer}: down rows do not hold {hidden} x {inter} {dn_kind}")
        L = Layer(torch, experts, hidden, inter, gu_kind, dn_kind, gu_rb, dn_rb)
        resident = sorted(set(int(e) for e in resident))
        host = [e for e in range(experts) if e not in set(resident)]
        L.slot_of = {e: i for i, e in enumerate(resident)}
        L.host_of = {e: i for i, e in enumerate(host)}
        if resident:
            index = torch.tensor(resident, dtype=torch.int64)
            L.device_gu = _padded(torch, (len(resident), 2 * inter, gu_rb), "cuda")
            L.device_dn = _padded(torch, (len(resident), hidden, dn_rb), "cuda")
            for first in range(0, len(resident), 32):      # a host block at a time
                part = index[first:first + 32]
                n = part.shape[0]
                block = torch.empty((n, 2 * inter, gu_rb), dtype=torch.uint8)
                block[:, :inter] = gate.index_select(0, part)
                block[:, inter:] = up.index_select(0, part)
                L.device_gu[first:first + n].copy_(block)
                L.device_dn[first:first + n].copy_(down.index_select(0, part))
        if host:
            index = torch.tensor(host, dtype=torch.int64)
            # exact-size host buffers, page-locked in place (torch's pinned allocator rounds a block up to a power of
            # two: 5 GB of experts would lock 8)
            L.host_gu = _padded(torch, (len(host), 2 * inter, gu_rb), "cpu")
            L.host_dn = _padded(torch, (len(host), hidden, dn_rb), "cpu")
            L.host_gu[:, :inter] = gate.index_select(0, index)
            L.host_gu[:, inter:] = up.index_select(0, index)
            L.host_dn[:] = down.index_select(0, index)
            for buffer in (L.host_gu, L.host_dn):
                _register(torch, buffer.base_buffer)
                self.registered.append(buffer.base_buffer)
        gu_ptr, dn_ptr = [0] * experts, [0] * experts
        address = self.kernels.device_address
        if resident:
            g0, d0 = address(L.device_gu), address(L.device_dn)
            for e, s in L.slot_of.items():
                gu_ptr[e], dn_ptr[e] = g0 + s * 2 * inter * gu_rb, d0 + s * hidden * dn_rb
        if host:
            g0, d0 = address(L.host_gu), address(L.host_dn)
            for e, s in L.host_of.items():
                gu_ptr[e], dn_ptr[e] = g0 + s * 2 * inter * gu_rb, d0 + s * hidden * dn_rb
        L.gu_ptr = torch.tensor(gu_ptr, dtype=torch.int64, device="cuda")
        L.dn_ptr = torch.tensor(dn_ptr, dtype=torch.int64, device="cuda")
        L.layer_index = layer
        if resident and host and 0 <= layer < TIER_SLOTS and (self.route_all is None
                                                               or self.route_all.shape[1] == experts):
            if self.route_all is None:
                self.route_all = torch.zeros((TIER_SLOTS, experts), dtype=torch.int32, device="cuda")
                self.score_all = torch.zeros((TIER_SLOTS, experts), dtype=torch.float32)
                self.dev_mask = torch.zeros((TIER_SLOTS, experts), dtype=torch.bool)
                self.host_mask = torch.zeros((TIER_SLOTS, experts), dtype=torch.bool)
                self.lent_mask = torch.zeros((TIER_SLOTS, experts), dtype=torch.bool)
            L.route_counts = self.route_all[layer]      # a fixed row: captured decode graphs add into it in place
            L.score = self.score_all[layer]
            self.dev_mask[layer, resident] = True
            self.host_mask[layer, host] = True
            self.adapt_rows = sorted(set(self.adapt_rows) | {layer})
        else:
            L.route_counts = torch.zeros(experts, dtype=torch.int32, device="cuda") if (resident and host) else None
            L.score = torch.zeros(experts, dtype=torch.float32)
        L.lent = {}
        # the CPU tier: host experts computed on the CPU from the same host memory (the device skips them)
        L.tier = False
        if (self.tier_module is not None and host and layer < TIER_SLOTS and gu_kind in KIND_IDS
                and dn_kind in KIND_IDS):
            if self.tier is None:
                self.tier = self.tier_module.Tier(TIER_SLOTS, hidden, inter, 8, self.workers)
                self.tier_offsets = self.tier.offsets()
            self.tier.set_layer(layer, L.host_gu.data_ptr(), L.host_dn.data_ptr(), gu_rb, dn_rb, KIND_IDS[gu_kind],
                                KIND_IDS[dn_kind])
            code = [-1] * experts
            for e, s in L.host_of.items():
                code[e] = s
            L.host_code = torch.tensor(code, dtype=torch.int32, device="cuda")
            masked_gu = [0 if code[e] >= 0 else gu_ptr[e] for e in range(experts)]
            masked_dn = [0 if code[e] >= 0 else dn_ptr[e] for e in range(experts)]
            L.dev_gu_ptr = torch.tensor(masked_gu, dtype=torch.int64, device="cuda")
            L.dev_dn_ptr = torch.tensor(masked_dn, dtype=torch.int64, device="cuda")
            L.pos = torch.zeros(DECODE_ROWS * 8, dtype=torch.int32, device="cuda")
            L.slot = self.tier.slot_device(layer)
            L.tier = True
        mask = torch.zeros(experts, dtype=torch.int64)
        mask[resident] = 1 if resident else 0
        L.resident = mask.to("cuda")
        self.layers[layer] = L
        if self.counts is None:
            self.counts = torch.zeros(2, dtype=torch.int64, device="cuda")   # routes read, routes on the device
            self.decode_counts = torch.zeros(2, dtype=torch.int64, device="cuda")   # the same for decode rows

    def device_bytes(self, layer):
        L = self.layers[layer]
        return sum(int(t.numel()) for t in (L.device_gu, L.device_dn) if t is not None)

    def host_bytes(self, layer):
        L = self.layers[layer]
        return sum(int(t.numel()) for t in (L.host_gu, L.host_dn) if t is not None)

    def _count(self, L, flat):
        valid = (flat >= 0).to(self.torch.int64)
        hits = L.resident.index_select(0, flat.clamp_min(0)) * valid
        self.counts[0].add_(valid.sum())
        self.counts[1].add_(hits.sum())

    def _count_decode(self, L, flat):
        """Decode rows' routes into the layer's route counts (the adaptive residency's input) and the decode reads and
        hits (Eval's expert_reads / expert_hits), capture-safe."""
        valid = (flat >= 0).to(self.torch.int64)
        self.decode_counts[0].add_(valid.sum())
        self.decode_counts[1].add_((L.resident.index_select(0, flat.clamp_min(0)) * valid).sum())
        if L.route_counts is not None:
            L.route_counts.index_add_(0, flat.clamp_min(0), (flat >= 0).to(self.torch.int32))

    # -- adaptive residency ------------------------------------------------------------------------------------------
    def _count_prompt(self, L, flat, tokens):
        """A prompt chunk's routes into the layer's prompt counts (prompt_warm()'s input), on the device."""
        if self.prompt_swaps <= 0 or L.route_counts is None:
            return
        layer = layer_of(self, L)
        found = self.prompt_counts.get(layer)
        if found is None:
            found = self.prompt_counts[layer] = self.torch.zeros(L.route_counts.shape[0], dtype=self.torch.float32,
                                                                 device=L.route_counts.device)
        found.index_add_(0, flat.clamp_min(0), (flat >= 0).to(self.torch.float32))
        if self.prompt_first is None:
            self.prompt_first = layer
        if layer == self.prompt_first:
            self.prompt_tokens += tokens

    def prompt_warm(self):
        """After a prompt's last chunk, before its first decode step: the prompt's routes, scaled to prompt_rows decode
        rows' worth per layer, join the route counts, and one adaptation of up to prompt_swaps swaps follows, so the
        request's first decode steps find the experts its prompt routed to on the device. A placement only: no value
        changes. Returns the swaps made."""
        if self.prompt_swaps <= 0 or not self.prompt_tokens:
            self.prompt_counts, self.prompt_tokens, self.prompt_first = {}, 0, None
            return 0
        scale = self.prompt_rows / self.prompt_tokens
        for layer, counts in self.prompt_counts.items():
            L = self.layers[layer]
            if L.route_counts is not None:
                L.route_counts.add_((counts * scale).round_().to(L.route_counts.dtype))
        self.prompt_counts, self.prompt_tokens, self.prompt_first = {}, 0, None
        return self.adapt(limit=self.prompt_swaps)

    def adapt(self, limit=None):
        """Between steps: each layer's decayed route scores from the decode rows since the last call; the hottest host
        experts swap places with the coldest device experts of the same layer (a byte exchange between a device slot
        and a host slot of one size, then the address tables), at most adapt_swaps a call (warm_swaps in a fresh
        engine's first warm_calls calls; limit for prompt_warm()'s call), each only where it gains.
        The device and the CPU tier are idle here (the call synchronizes first)."""
        torch = self.torch
        if limit is None:
            limit = self.adapt_swaps
            if self.adapt_calls < self.warm_calls:
                limit = max(limit, self.warm_swaps)
            self.adapt_calls += 1
        if limit <= 0 or not self.layers:
            return 0
        torch.cuda.synchronize()
        pairs = self._stacked_pairs(limit) if self.route_all is not None else []
        stacked = set(self.adapt_rows) if self.route_all is not None else set()
        for layer, L in self.layers.items():
            if layer in stacked or L.route_counts is None or not L.slot_of or not L.host_of:
                continue
            counts = L.route_counts.float().cpu()
            L.route_counts.zero_()
            L.score.mul_(self.adapt_decay).add_(counts)
            score = L.score
            host = sorted((e for e in L.host_of if e not in L.lent), key=lambda e: -float(score[e]))[:limit]
            dev = sorted(L.slot_of, key=lambda e: float(score[e]))[:limit]
            for b, a in zip(host, dev):
                gain = float(score[b]) - float(score[a])
                if float(score[b]) >= self.adapt_min and gain > self.adapt_margin:
                    pairs.append((gain, layer, a, b))
        pairs.sort(reverse=True)
        done = 0
        for gain, layer, a, b in pairs[:limit]:
            self._swap(self.layers[layer], a, b)
            done += 1
        if done:
            torch.cuda.synchronize()
        self.swaps += done
        return done

    def _stacked_pairs(self, limit):
        """adapt()'s candidate swaps of the stacked layers, (gain, layer, device expert, host expert): one copy of every
        layer's route counts to the host, each layer's scores decayed and the new counts added, then per layer the
        hottest host experts (not lent) against the coldest device experts in rank order, kept where the host
        expert's score reaches adapt_min and gains more than adapt_margin -- the per-layer rule, by tensor ops."""
        torch = self.torch
        rows = self.adapt_rows
        if not rows:
            return []
        index = torch.tensor(rows, dtype=torch.int64)
        counts = self.route_all[:len(rows)].float().cpu() if rows == list(range(len(rows))) else             self.route_all.index_select(0, index.to("cuda")).float().cpu()
        self.route_all.zero_()
        score = self.score_all.index_select(0, index).mul_(self.adapt_decay).add_(counts)
        self.score_all.index_copy_(0, index, score)
        host = self.host_mask.index_select(0, index) & ~self.lent_mask.index_select(0, index)
        dev = self.dev_mask.index_select(0, index)
        k = min(limit, score.shape[1])
        inf = float("inf")
        hot_v, hot_i = torch.where(host, score, -inf).topk(k, dim=1)
        cold_v, cold_i = torch.where(dev, score, inf).topk(k, dim=1, largest=False)
        gain = hot_v - cold_v
        keep = torch.isfinite(hot_v) & torch.isfinite(cold_v) & (hot_v >= self.adapt_min) & (gain > self.adapt_margin)
        found = keep.nonzero().tolist()
        if not found:
            return []
        return [(float(gain[r, j]), rows[r], int(cold_i[r, j]), int(hot_i[r, j])) for r, j in found]

    # -- lent device copies -------------------------------------------------------------------------------------------
    def lend(self, spaces, page_bytes, ranking, min_page):
        """Copies of host experts in memory another owner lends (the KV pool's top pages): spaces are [pages][page
        bytes] views, all of one page count; experts are placed top down, one expert's gate|up then down rows in one
        space's consecutive pages, in ranking order ((layer, expert), most routed first), never below min_page. The host
        copy stays the expert's home: reclaim() points a lent expert back at it. Returns the lowest page lent."""
        torch = self.torch
        pages = spaces[0].shape[0]
        cursor = [pages] * len(spaces)
        floor = pages
        address = self.kernels.device_address
        placed = 0
        for layer, e in ranking:
            L = self.layers[layer]
            if e not in L.host_of or e in L.lent:
                continue
            gu_bytes, dn_bytes = 2 * L.inter * L.gu_rb, L.hidden * L.dn_rb
            need = -(-(gu_bytes + dn_bytes + 256) // page_bytes)
            space = max(range(len(spaces)), key=lambda s: cursor[s])
            if cursor[space] - need < min_page:
                break
            first = cursor[space] - need
            cursor[space] = first
            floor = min(floor, first)
            flat = spaces[space][first:first + need].reshape(-1)
            gu = flat[:gu_bytes].view(2 * L.inter, L.gu_rb)
            dn = flat[gu_bytes:gu_bytes + dn_bytes].view(L.hidden, L.dn_rb)
            h = L.host_of[e]
            gu.copy_(L.host_gu[h], non_blocking=True)
            dn.copy_(L.host_dn[h], non_blocking=True)
            L.lent[e] = (first, gu, dn)            # its lowest page
            if self.lent_mask is not None and layer in self.adapt_rows:
                self.lent_mask[layer, e] = True
            ga, da = address(gu), address(dn)
            L.gu_ptr[e], L.dn_ptr[e] = ga, da
            L.resident[e] = 1
            if L.tier:
                L.host_code[e] = -1
                L.dev_gu_ptr[e], L.dev_dn_ptr[e] = ga, da
            placed += 1
        torch.cuda.synchronize()
        self.lent_experts = placed
        return floor

    def reclaim(self, page):
        """The owner hands out page `page` (pages go out lowest first): every lent expert with a page at or below it
        (its lowest page <= page) goes back to its host copy; the experts above stay lent. Returns the new floor: the
        lowest page still lent (an owner reclaims again before handing out a page at or above it), or 1 << 62."""
        address = self.kernels.device_address
        floor = 1 << 62
        for L in self.layers.values():
            for e, (first, _, _) in list(L.lent.items()):
                if first > page:
                    floor = min(floor, first)
                    continue
                del L.lent[e]
                if self.lent_mask is not None and getattr(L, "layer_index", None) in self.adapt_rows:
                    self.lent_mask[L.layer_index, e] = False
                h = L.host_of[e]
                ga = address(L.host_gu) + h * L.host_gu[0].numel()
                da = address(L.host_dn) + h * L.host_dn[0].numel()
                L.gu_ptr[e], L.dn_ptr[e] = ga, da
                L.resident[e] = 0
                if L.tier:
                    L.host_code[e] = h
                    L.dev_gu_ptr[e], L.dev_dn_ptr[e] = 0, 0
                self.lent_experts -= 1
                self.reclaimed += 1
        return floor

    def _swap(self, L, a, b):
        """Device expert a and host expert b trade places (and bytes)."""
        torch = self.torch
        s, h = L.slot_of.pop(a), L.host_of.pop(b)
        for dev, host in ((L.device_gu, L.host_gu), (L.device_dn, L.host_dn)):
            held = dev[s].clone()
            dev[s].copy_(host[h], non_blocking=True)
            host[h].copy_(held, non_blocking=True)
        L.slot_of[b], L.host_of[a] = s, h
        row = getattr(L, "layer_index", None)
        if self.dev_mask is not None and row in self.adapt_rows:
            self.dev_mask[row, a], self.dev_mask[row, b] = False, True
            self.host_mask[row, a], self.host_mask[row, b] = True, False
        address = self.kernels.device_address
        gu_dev = address(L.device_gu) + s * L.device_gu[0].numel()
        dn_dev = address(L.device_dn) + s * L.device_dn[0].numel()
        gu_host = address(L.host_gu) + h * L.host_gu[0].numel()
        dn_host = address(L.host_dn) + h * L.host_dn[0].numel()
        L.gu_ptr[b], L.dn_ptr[b], L.gu_ptr[a], L.dn_ptr[a] = gu_dev, dn_dev, gu_host, dn_host
        L.resident[b], L.resident[a] = 1, 0
        if L.tier:
            L.host_code[a], L.host_code[b] = h, -1
            L.dev_gu_ptr[b], L.dev_dn_ptr[b] = gu_dev, dn_dev
            L.dev_gu_ptr[a], L.dev_dn_ptr[a] = 0, 0

    def begin(self, layer, x, ids, weights):
        """The first half of forward() for decode rows with the CPU tier: the host routes posted and the device routes
        launched; finish() joins and combines. Other calls run whole here. Work the caller launches between the two
        overlaps the CPU's share."""
        self.calls += 1
        L = self.layers[layer]
        if ids.shape[0] <= DECODE_ROWS and L.tier and ids.shape[1] <= 8:
            return ("tier", L) + self._tier_launch(L, x, ids, weights)
        return ("done", self._whole(L, x, ids, weights))

    def finish(self, state):
        if state[0] == "done":
            return state[1]
        _, L, y, flat, w, top = state
        tok_off, idx_off, x_off, out_off, _ = self.tier_offsets
        self.tier_module.wait(L.slot)
        return self.tier_module.combine(y, L.slot, out_off, L.pos, flat, w, top)

    def _whole(self, L, x, ids, weights):
        if ids.shape[0] <= DECODE_ROWS:
            return self._decode_routes(L, x, ids, weights)
        return self._prompt_routes(L, x, ids, weights)

    def _tier_launch(self, L, x, ids, weights):
        torch, k = self.torch, self.tier_module
        tokens, top = ids.shape
        flat = ids.reshape(-1).contiguous()
        self.kernels.route_account(flat, L.resident, self.counts, self.decode_counts, L.route_counts)
        xb = x.to(torch.bfloat16).contiguous()
        tok_off, idx_off, x_off, out_off, _ = self.tier_offsets
        k.post(L.slot, flat, L.host_code, xb, L.pos, top, tok_off, idx_off, x_off)
        gu = self.kernels.gathered_gemv_q8t(xb, L.dev_gu_ptr, flat, 2 * L.inter, L.gu_rb, L.gu_kind, L.hidden, top)
        act = self.kernels.swiglu(gu)
        y = self.kernels.gathered_gemv_q8t(act, L.dev_dn_ptr, flat, L.hidden, L.dn_rb, L.dn_kind, L.inter, 1)
        return y, flat, weights.reshape(-1).float().contiguous(), top

    def _tier_routes(self, L, x, ids, weights):
        """Decode rows with the CPU tier: the host routes posted to the layer's mailbox slot first, the device routes
        on the device meanwhile (the host pairs skipped), then the join and each token's routes in route order."""
        torch, k = self.torch, self.tier_module
        tokens, top = ids.shape
        flat = ids.reshape(-1).contiguous()
        e = flat.clamp_min(0)
        self._count(L, flat)
        xb = x.to(torch.bfloat16).contiguous()
        tok_off, idx_off, x_off, out_off, _ = self.tier_offsets
        k.post(L.slot, flat, L.host_code, xb, L.pos, top, tok_off, idx_off, x_off)
        gu = self.kernels.gathered_gemv_q8p(xb, L.dev_gu_ptr.index_select(0, e), 2 * L.inter, L.gu_rb, L.gu_kind,
                                            L.hidden, top)
        act = self.kernels.swiglu(gu)
        y = self.kernels.gathered_gemv_q8p(act, L.dev_dn_ptr.index_select(0, e), L.hidden, L.dn_rb, L.dn_kind,
                                           L.inter, 1)
        k.wait(L.slot)
        return k.combine(y, L.slot, out_off, L.pos, flat, weights.reshape(-1).float().contiguous(), top)

    def _decode_routes(self, L, x, ids, weights):
        torch = self.torch
        tokens, top = ids.shape
        if L.tier and top <= 8:
            return self._tier_routes(L, x, ids, weights)
        flat = ids.reshape(-1)
        e = flat.clamp_min(0)
        self._count(L, flat)
        self._count_decode(L, flat)
        gu = self.kernels.gathered_gemv_q8p(x.to(torch.bfloat16), L.gu_ptr.index_select(0, e), 2 * L.inter,
                                            L.gu_rb, L.gu_kind, L.hidden, top)
        act = self.kernels.swiglu(gu)
        y = self.kernels.gathered_gemv_q8p(act, L.dn_ptr.index_select(0, e), L.hidden, L.dn_rb, L.dn_kind, L.inter, 1)
        return self.kernels.combine(y, flat.to(torch.int64), weights.reshape(-1).float(), top)

    def _prompt_routes(self, L, x, ids, weights):
        torch = self.torch
        tokens, top = ids.shape
        experts = L.experts
        flat = ids.reshape(-1)
        self._count(L, flat)
        self._count_prompt(L, flat, tokens)
        routes = flat.shape[0]
        key = torch.where(flat >= 0, flat, torch.full_like(flat, experts))     # dropped routes sort last
        order = torch.argsort(key, stable=True)
        counts = torch.bincount(key, minlength=experts + 1)[:experts]
        offsets = torch.zeros(experts + 1, dtype=torch.int32, device=x.device)
        offsets[1:] = torch.cumsum(counts, 0)
        # tiles as tall as the chunk's routes an expert on average (each tile reads its expert's weights once: a host
        # expert's over the link), 32 to 128 rows
        per = routes / max(1, experts)
        bm = 32 if per <= 40 else (64 if per <= 96 else 128)
        tile_off = torch.zeros(experts + 1, dtype=torch.int32, device=x.device)
        tile_off[1:] = torch.cumsum((counts + bm - 1) // bm, 0)
        max_tiles = -(-routes // bm) + experts
        h = x.to(torch.bfloat16).index_select(0, order // top)
        xq, xd = self.kernels.quantize(h)
        del h
        staged = self.stage_experts > 0 and L.host_gu is not None
        pref = None
        entry, self.pref_entry = self.pref_entry, None
        if entry is not None and entry[0] == layer_of(self, L) and entry[1] == tokens:
            pref = entry[2:]            # (buffer, {expert: index}): this layer's prefetched host experts
        if staged:
            # host experts staged to the device in batches by the copy engine (bulk transfers over the link), the
            # device's own read in place: each touched host expert crosses the link once, in large copies; those the
            # previous layer prefetched are read where they landed
            host_counts = counts.tolist()
            batches = self._host_batches(L, host_counts, pref[1] if pref else ())
            gu = torch.empty((routes, 2 * L.inter), dtype=torch.float32, device=x.device)
            self._staged_mmq(L, "gu", xq, xd, gu, counts, offsets, bm, batches, False, pref)
        else:
            gu = self.kernels.mmqp(xq, xd, L.gu_ptr, 2 * L.inter, L.gu_rb, L.gu_kind, L.hidden, offsets, tile_off,
                                   max_tiles, False, bm)
        del xq, xd
        act = self.kernels.swiglu(gu)
        del gu
        if L.dn_kind in kernels.MMQ_KINDS:
            aq, ad = self.kernels.quantize(act)
            if staged:
                y = torch.empty((routes, L.hidden), dtype=torch.bfloat16, device=x.device)
                self._staged_mmq(L, "dn", aq, ad, y, counts, offsets, bm, batches, True, pref)
            else:
                y = self.kernels.mmqp(aq, ad, L.dn_ptr, L.hidden, L.dn_rb, L.dn_kind, L.inter, offsets, tile_off,
                                      max_tiles, True, bm)
            del aq, ad
        else:
            # Q6_K down (no int8 tile path): each expert's routes in groups of up to 8 rows through the grouped
            # stored-block GEMV, device experts from their slots and the rest from their host copies (read over the
            # link), each row its own warp's sums whatever the chunk
            y = torch.empty((routes, L.hidden), dtype=torch.float32, device=x.device)
            bounds = offsets.tolist()
            stores = [(L.device_dn, L.slot_of)]
            if L.host_dn is not None:
                if self.stage_q6k and self.stage_experts > 0:
                    stores = itertools.chain(stores, self._staged_q6k(L, bounds))   # lazily: a batch at a time
                else:
                    stores.append((L.host_dn, L.host_of))
            for store, index_of in stores:
                if store is None:
                    continue
                g_e, g_s, g_c = [], [], []
                for expert, j in index_of.items():
                    if store is L.host_dn and expert in L.slot_of:
                        continue
                    a, b = bounds[expert], bounds[expert + 1]
                    for s in range(a, b, DECODE_ROWS):
                        g_e.append(j)
                        g_s.append(s)
                        g_c.append(min(DECODE_ROWS, b - s))
                if not g_e:
                    continue
                part = self.kernels.grouped_gemv(act, store, torch.tensor(g_e, dtype=torch.int32, device=x.device),
                                                 torch.tensor(g_s, dtype=torch.int32, device=x.device),
                                                 torch.tensor(g_c, dtype=torch.int32, device=x.device), L.dn_kind,
                                                 L.inter)
                rows = torch.cat([torch.arange(s, s + c) for s, c in zip(g_s, g_c)]).to(x.device)
                y[rows] = part[rows]
            y = y.to(torch.bfloat16)
        position = torch.empty_like(order)
        position[order] = torch.arange(routes, device=x.device)
        out = self.kernels.combine(y, flat.to(torch.int64), weights.reshape(-1).float(), top, position)
        if staged and self.prefetch_experts > 0 and tokens >= self.prefetch_min_tokens:
            self._prefetch(layer_of(self, L) + 1, tokens)
        return out

    def _prefetch(self, layer, tokens):
        """The next prompt layer's host experts (not lent; at most prefetch_experts, the most routed by the decode
        scores first) copied to the device on the prefetch stream after this layer's work: they cross the link while
        the next layer runs its attention, dense products and device experts. A buffer is written only after the main
        stream's work so far (its last reader, two layers back, included)."""
        torch = self.torch
        L = self.layers.get(layer)
        if L is None or L.host_gu is None:
            return
        host = [e for e in L.host_of if e not in L.lent]
        if not host:
            return
        if getattr(L, "score", None) is not None:
            host.sort(key=lambda e: -float(L.score[e]))
        host = host[:self.prefetch_experts]
        gu_bytes, dn_bytes = 2 * L.inter * L.gu_rb, L.hidden * L.dn_rb
        with_dn = L.dn_kind in kernels.MMQ_KINDS
        need = self.prefetch_experts * (gu_bytes + (dn_bytes if with_dn else 0)) + 256
        if self.pref_buffers is None or self.pref_buffers[0].numel() < need:
            self.pref_buffers = [torch.empty(need, dtype=torch.uint8, device="cuda") for _ in range(2)]
            self.pref_events = [torch.cuda.Event() for _ in range(2)]
            self.pref_stream = torch.cuda.Stream()
        buffer = self.pref_next
        self.pref_next = 1 - buffer
        stage = self.pref_buffers[buffer]
        self.pref_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.pref_stream):
            dn_base = self.prefetch_experts * gu_bytes
            for j, e in enumerate(host):
                h = L.host_of[e]
                stage[j * gu_bytes:(j + 1) * gu_bytes].copy_(L.host_gu[h].reshape(-1), non_blocking=True)
                if with_dn:
                    stage[dn_base + j * dn_bytes:dn_base + (j + 1) * dn_bytes].copy_(L.host_dn[h].reshape(-1),
                                                                                    non_blocking=True)
            self.pref_events[buffer].record(self.pref_stream)
        self.pref_entry = (layer, tokens, buffer, {e: j for j, e in enumerate(host)})

    def _staged_q6k(self, L, bounds):
        """The touched host experts' Q6_K down rows as (store, {expert: index}) batches for the grouped GEMV, yielded
        one at a time: each batch of up to stage_experts copied into a staging buffer on the copy stream (two
        buffers, alternating), the next batch's copy issued before this one is handed out, so it runs while this
        batch multiplies; a buffer is refilled only after the main stream's work so far (its last reader included,
        which the caller launched before asking for the next batch)."""
        torch = self.torch
        touched = [e for e in sorted(L.host_of) if e not in L.slot_of and bounds[e + 1] > bounds[e]]
        if not touched:
            return
        stage = self._stage_buffers(L)
        per = L.hidden * L.dn_rb
        main = torch.cuda.current_stream()
        batches = [touched[b:b + self.stage_experts] for b in range(0, len(touched), self.stage_experts)]

        def issue(i):
            self.copy_stream.wait_stream(main)
            with torch.cuda.stream(self.copy_stream):
                for j, e in enumerate(batches[i]):
                    stage[i % 2][j * per:(j + 1) * per].copy_(L.host_dn[L.host_of[e]].reshape(-1), non_blocking=True)
                self.stage_events[i % 2].record(self.copy_stream)

        issue(0)
        for i, batch in enumerate(batches):
            if i + 1 < len(batches):
                issue(i + 1)
            main.wait_event(self.stage_events[i % 2])
            yield stage[i % 2][:len(batch) * per].view(len(batch), L.hidden, L.dn_rb), {e: j for j, e in enumerate(batch)}

    def _host_batches(self, L, host_counts, skip=()):
        touched = [e for e in sorted(L.host_of) if host_counts[e] > 0 and e not in L.lent and e not in skip]
        return [touched[i:i + self.stage_experts] for i in range(0, len(touched), self.stage_experts)]

    def _stage_buffers(self, L):
        torch = self.torch
        gu_bytes = 2 * L.inter * L.gu_rb
        dn_bytes = L.hidden * L.dn_rb
        need = self.stage_experts * max(gu_bytes, dn_bytes) + 256
        if self.stage is None or self.stage[0].numel() < need:
            self.stage = [torch.empty(need, dtype=torch.uint8, device="cuda") for _ in range(2)]
            self.stage_events = [torch.cuda.Event() for _ in range(2)]
            self.copy_stream = torch.cuda.Stream()
        return self.stage

    def _staged_mmq(self, L, which, xq, xd, out, counts, offsets, bm, batches, out_bf16, pref=None):
        """One matrix of a prompt layer's routed experts into out: the device experts' segments in place, then each
        batch of touched host experts copied into a staging buffer (two, alternating: the next batch's copy runs on the
        copy stream while this batch multiplies) and multiplied there."""
        torch = self.torch
        k = self.kernels
        E = L.experts
        N, K, rb, kind = ((2 * L.inter, L.hidden, L.gu_rb, L.gu_kind) if which == "gu"
                          else (L.hidden, L.inter, L.dn_rb, L.dn_kind))
        per = N * rb
        ptr = L.gu_ptr if which == "gu" else L.dn_ptr
        host = L.host_gu if which == "gu" else L.host_dn

        def tiles(mask):
            c = counts * mask
            t = torch.zeros(E + 1, dtype=torch.int32, device=counts.device)
            t[1:] = torch.cumsum((c + bm - 1) // bm, 0)
            return t

        resident = L.resident.to(counts.dtype)
        routes = out.shape[0]
        if L.slot_of:
            k.mmqp(xq, xd, ptr, N, rb, kind, K, offsets, tiles(resident), -(-routes // bm) + E, out_bf16, bm, out)
        main = torch.cuda.current_stream()
        if pref is not None and pref[1]:
            # the experts the previous layer prefetched, read where they landed (one product over all of them)
            buffer, index = pref
            main.wait_event(self.pref_events[buffer])
            base = k.device_address(self.pref_buffers[buffer])
            if which == "dn":
                base += self.prefetch_experts * 2 * L.inter * L.gu_rb
            seg, mask = [0] * E, [0] * E
            for e, j in index.items():
                seg[e] = base + j * per
                mask[e] = 1
            seg_t = torch.tensor(seg, dtype=torch.int64).to("cuda", non_blocking=True)
            mask_t = torch.tensor(mask, dtype=counts.dtype).to("cuda", non_blocking=True)
            k.mmqp(xq, xd, seg_t, N, rb, kind, K, offsets, tiles(mask_t), -(-routes // bm) + E, out_bf16, bm, out)
        if not batches:
            return
        stage = self._stage_buffers(L)
        copy = self.copy_stream

        def issue(b, buffer):
            copy.wait_stream(main)          # the buffer's last reader (two batches back) is done
            with torch.cuda.stream(copy):
                for j, e in enumerate(batches[b]):
                    stage[buffer][j * per:(j + 1) * per].copy_(host[L.host_of[e]].reshape(-1), non_blocking=True)
                self.stage_events[buffer].record(copy)

        issue(0, 0)
        for b, batch in enumerate(batches):
            buffer = b % 2
            if b + 1 < len(batches):
                issue(b + 1, 1 - buffer)
            main.wait_event(self.stage_events[buffer])
            base = k.device_address(stage[buffer])
            seg = [0] * E
            mask = [0] * E
            for j, e in enumerate(batch):
                seg[e] = base + j * per
                mask[e] = 1
            seg_t = torch.tensor(seg, dtype=torch.int64).to("cuda", non_blocking=True)
            mask_t = torch.tensor(mask, dtype=counts.dtype).to("cuda", non_blocking=True)
            k.mmqp(xq, xd, seg_t, N, rb, kind, K, offsets, tiles(mask_t), -(-routes // bm) + E, out_bf16, bm, out)

    def forward(self, layer, x, ids, weights):
        self.calls += 1
        L = self.layers[layer]
        if ids.shape[0] <= DECODE_ROWS:
            return self._decode_routes(L, x, ids, weights)
        return self._prompt_routes(L, x, ids, weights)

    def close(self):
        self.torch.cuda.synchronize()
        if self.tier is not None and hasattr(self.tier, "close"):
            self.tier.close()          # its threads stopped now, whatever still refers to the object
        self.tier = None
        for buffer in self.registered:
            self.torch.cuda.cudart().cudaHostUnregister(buffer.data_ptr())
        self.registered.clear()
        self.layers.clear()
        self.dense.clear()
        self.counts = None

    def facts(self):
        device = sum(self.device_bytes(i) for i in self.layers)
        host = sum(self.host_bytes(i) for i in self.layers)
        return {**kernels.status(), **tier_kernels.status(), "experts_activation": "silu",
                "experts_cpu_tier": "decode rows' host experts on the CPU (expert_tier.cu: AVX2 over the stored "
                                    "blocks, joined in the captured step through a mapped mailbox)"
                if self.tier is not None else "off (host experts read over the link)",
                "experts_prompt_activations": self.prompt, "experts_decode_activations": self.decode,
                "experts_formats": ["gguf:IQ3_S", "gguf:IQ4_XS", "gguf:Q6_K", "gguf:Q8_0", "gguf:IQ4_NL"],
                "expert_bytes_device": device, "expert_bytes_host": host,
                "expert_placement": "device slots and pinned host memory read in place over the link"}

    def decode_counters(self):
        """(decode routes read, decode routes whose expert was on the device) so far."""
        if self.counts is None:
            return 0, 0
        reads, hits = self.decode_counts.tolist()
        return int(reads), int(hits)

    def telemetry(self):
        reads, hits = (self.counts.tolist() if self.counts is not None else (0, 0))
        found = {"expert_reads": int(reads), "expert_hits": int(hits),
                 "gguf_experts": {"calls": self.calls, "swaps": self.swaps, "lent": self.lent_experts,
                                  "reclaimed": self.reclaimed}}
        if self.tier is not None:
            served, routes, ns = self.tier.counters()
            found["gguf_experts"]["cpu_tier"] = {"jobs": int(served), "routes": int(routes),
                                                 "compute_ms": round(ns / 1e6, 3)}
        return found


def layer_of(experts, L):
    """The layer index a layer's state is kept under."""
    index = getattr(L, "layer_index", None)
    if index is None:
        index = next(i for i, M in experts.layers.items() if M is L)
        L.layer_index = index
    return index


def create(context, options, parts):
    made = GgufExperts(options.get("decode", "stored"), options.get("prompt", "stored"),
                       bool(options.get("cpu_tier", True)), int(options.get("tier_workers", 6)))
    made.adapt_swaps = int(options.get("adapt_swaps", 8))
    made.warm_swaps = int(options.get("warm_swaps", 0))
    made.warm_calls = int(options.get("warm_calls", 0))
    made.adapt_decay = float(options.get("adapt_decay", 0.8))
    made.adapt_min = float(options.get("adapt_min", 3.0))
    made.adapt_margin = float(options.get("adapt_margin", 2.0))
    made.prefetch_experts = int(options.get("prefetch_experts", 0))
    made.prompt_rows = int(options.get("prompt_rows", 0))
    made.prompt_swaps = int(options.get("prompt_swaps", 0))
    return made
