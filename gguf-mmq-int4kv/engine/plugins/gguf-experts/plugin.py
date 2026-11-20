"""Routed GLU experts over a GGUF's stored expert blocks. A layer is its fused gate|up rows [experts][2 * inter][row
bytes] and down rows [experts][hidden][row bytes] in their GGUF formats, on the device or in pinned host memory.
forward(layer, x, ids, weights) decodes the experts the call routes to (a group at a time), runs each expert's tokens
(padded to DECODE_ROWS rows, so a decode row's bits are its own) through gate|up, act(gate) * up and down with FP32
accumulation, and sums each token's routes in route order (weights FP32, id -1 skipped). Host form: the same block in
FP64 over dense matrices."""
import math

from .gguf import fp8_rows, gguf_blocks, kernels

DECODE_ROWS = 8


class Layer:
    def __init__(self, gate_up, down, gate_up_kind, down_kind, hidden, inter):
        self.gate_up, self.down = gate_up, down
        self.gate_up_kind, self.down_kind = gate_up_kind, down_kind
        self.hidden, self.inter = hidden, inter

    @property
    def experts(self):
        return self.gate_up.shape[0]


def _act(name, value):
    if name == "silu":
        return value / (1.0 + math.exp(-value))
    return 0.5 * value * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (value + 0.044715 * value ** 3)))


MMQ_ROWS = 32     # rows a tile of the int8 tensor-core kernel takes (gguf_gemv.cu MQ_BM)


class GgufExperts:
    def __init__(self, activation, group, prompt="fp8", decode="stored"):
        import torch
        self.torch, self.activation, self.group, self.prompt, self.decode = torch, activation, group, prompt, decode
        self.gathered_routes = 256     # routes up to which a call runs them all through the gathered GEMVs
        self.layers, self.dense = {}, {}
        self.calls = 0
        self.kernels = kernels.load()
        self.staging = {}      # pinned buffers a host layer's routed experts are gathered into before the copy

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
                g = [_act(self.activation, h[i]) * h[inter + i] for i in range(inter)]
                for j, r in enumerate(down):
                    total[j] += w * sum(a * b for a, b in zip(r, g))
            out.append(total)
        return out

    # -- device form -------------------------------------------------------------------------------------------------
    def load(self, layer, expert, gate_up, down, format="bf16"):
        """One expert's dense matrices (format bf16): the slot's per-expert form, for small models and tests."""
        if format != "bf16":
            raise ValueError("gguf-experts loads stored GGUF layers with load_layer; load takes bf16 experts")
        torch = self.torch
        self.dense[(layer, expert)] = (gate_up.tolist() if hasattr(gate_up, "tolist") else gate_up,
                                       down.tolist() if hasattr(down, "tolist") else down)
        gu = torch.as_tensor(gate_up, dtype=torch.bfloat16, device="cuda")
        dn = torch.as_tensor(down, dtype=torch.bfloat16, device="cuda")
        found = self.layers.get(layer)
        if found is None or found.gate_up_kind != "BF16":
            found = self.layers[layer] = Layer({}, {}, "BF16", "BF16", dn.shape[0], dn.shape[1])
        found.gate_up[expert], found.down[expert] = gu, dn
        found.stacked = None

    def load_layer(self, layer, gate_up, down, gate_up_kind, down_kind, hidden, inter):
        """A layer's stored expert rows: gate_up uint8 [experts][2 * inter][row bytes], down [experts][hidden][row
        bytes], each where the placement plan put it (the device, or pinned host memory read over the link)."""
        if gate_up.shape[1] != 2 * inter or gate_up.shape[2] != gguf_blocks.row_bytes(gate_up_kind, hidden):
            raise ValueError(f"layer {layer}: gate|up rows do not hold {2 * inter} x {hidden} {gate_up_kind}")
        if down.shape[1] != hidden or down.shape[2] != gguf_blocks.row_bytes(down_kind, inter):
            raise ValueError(f"layer {layer}: down rows do not hold {hidden} x {inter} {down_kind}")
        self.layers[layer] = Layer(gate_up, down, gate_up_kind, down_kind, hidden, inter)
        if gate_up.device.type != "cuda" and self.kernels is not None:
            self._staging_for(gate_up, down)

    def _decoded(self, layer, experts):
        """BF16 gate|up [n][2 * inter][hidden] and down [n][hidden][inter] of the given expert ids (a host list)."""
        torch, L = self.torch, self.layers[layer]
        if L.gate_up_kind == "BF16":
            return (torch.stack([L.gate_up[e] for e in experts]), torch.stack([L.down[e] for e in experts]))
        index = torch.tensor(experts, dtype=torch.int64, device=L.gate_up.device)
        gu, dn = L.gate_up.index_select(0, index), L.down.index_select(0, index)
        if gu.device.type != "cuda":
            gu, dn = self._stage(L.gate_up, experts, "gate_up"), self._stage(L.down, experts, "down")
        n = len(experts)
        if self.kernels is not None and L.gate_up_kind in kernels.KINDS and L.down_kind in kernels.KINDS:
            chosen = torch.arange(n, dtype=torch.int32, device="cuda")
            return (self.kernels.dequant_experts(gu.contiguous(), chosen, L.gate_up_kind, L.hidden),
                    self.kernels.dequant_experts(dn.contiguous(), chosen, L.down_kind, L.inter))
        gu = gguf_blocks.decode(gu.reshape(n * 2 * L.inter, -1), L.gate_up_kind, L.hidden).to(torch.bfloat16)
        dn = gguf_blocks.decode(dn.reshape(n * L.hidden, -1), L.down_kind, L.inter).to(torch.bfloat16)
        return gu.view(n, 2 * L.inter, L.hidden), dn.view(n, L.hidden, L.inter)

    def _staging_for(self, gate_up, down):
        """Device buffers as large as a host layer's experts (one set for every host layer, used in turn): a decode
        step gathers its routed experts into them on the device, a prompt step copies the whole layer in."""
        torch = self.torch
        current = self.staging.get("device")
        if current is None or current[0].numel() < gate_up.numel() or current[1].numel() < down.numel():
            self.staging["device"] = (torch.empty(gate_up.numel(), dtype=torch.uint8, device="cuda"),
                                      torch.empty(down.numel(), dtype=torch.uint8, device="cuda"))

    def _staged(self, L, experts=None):
        """A host layer's experts on the device: the chosen ones (experts, an int32 device tensor; returned as rows
        0 .. len - 1 of the staging buffers) gathered on the device, or with experts None the whole layer copied."""
        gu_buf, dn_buf = self.staging["device"]
        gu = gu_buf[:L.gate_up.numel()].view(L.gate_up.shape)
        dn = dn_buf[:L.down.numel()].view(L.down.shape)
        if experts is None:
            gu.copy_(L.gate_up, non_blocking=True)
            dn.copy_(L.down, non_blocking=True)
            return gu, dn
        self.kernels.gather_rows(L.gate_up, experts, gu)
        self.kernels.gather_rows(L.down, experts, dn)
        return gu, dn

    def _stage(self, store, index, name):
        """The rows of experts `index` of a layer in host memory, gathered into a pinned buffer and copied over."""
        torch = self.torch
        shape = (len(index),) + tuple(store.shape[1:])
        buffer = self.staging.get(name)
        if buffer is None or buffer.shape[0] < shape[0] or tuple(buffer.shape[1:]) != shape[1:]:
            buffer = self.staging[name] = torch.empty((max(shape[0], 64),) + shape[1:], dtype=store.dtype,
                                                      pin_memory=True)
        torch.index_select(store, 0, torch.tensor(index, dtype=torch.int64), out=buffer[:shape[0]])
        return buffer[:shape[0]].to("cuda")

    def _dense_routes(self, layer, x, ids, weights):
        """Decode rows of a dense BF16 layer, capture-safe: each route's expert matrices gathered from the stacked
        layer, FP32 products, each token's routes summed in route order."""
        torch, F, L = self.torch, self.torch.nn.functional, self.layers[layer]
        if getattr(L, "stacked", None) is None:
            count = max(L.gate_up) + 1
            L.stacked = (torch.stack([L.gate_up[e] for e in range(count)]), torch.stack([L.down[e] for e in range(count)]))
        gate_up, down = L.stacked
        tokens, top = ids.shape
        flat = ids.reshape(-1)
        e = flat.clamp_min(0)
        h = x.to(torch.bfloat16).repeat_interleave(top, dim=0).float()
        gu = torch.einsum("pk,pnk->pn", h, gate_up[e].float())
        g, u = gu[:, :L.inter], gu[:, L.inter:]
        act = F.gelu(g, approximate="tanh") if self.activation == "gelu_tanh" else F.silu(g)
        y = torch.einsum("pk,pnk->pn", (act * u).to(torch.bfloat16).float(), down[e].float())
        w = torch.where(flat >= 0, weights.reshape(-1).float(), torch.zeros((), device=x.device))
        routed = (y * w[:, None]).view(tokens, top, L.hidden)
        out = routed[:, 0].clone()
        for k in range(1, top):
            out += routed[:, k]
        return out

    def _decode_routes(self, layer, x, ids, weights):
        """Decode rows: every (token, route) pair through the gathered GEMVs over the stored blocks, with no host
        synchronization (capture-safe): each pair reads its token's row (pair / top), a route of id -1 reads expert 0
        and the combine skips it; the GLU and the route sum are one kernel each, the sum in route order."""
        torch, L = self.torch, self.layers[layer]
        tokens, top = ids.shape
        flat = ids.reshape(-1)
        experts = flat.clamp_min(0).to(torch.int32)
        gate_up, down = L.gate_up, L.down
        if gate_up.device.type != "cuda":
            # a host layer: its routed experts gathered into the staging buffers over the link, on the device
            gate_up, down = self._staged(L, experts)
            experts = torch.arange(experts.shape[0], dtype=torch.int32, device=x.device)
        # the stored formats' activations (int8 units, integer dot products) on device rows, else BF16
        gemv = self.kernels.gathered_gemv_q8 if self.decode == "stored" and gate_up.is_cuda else self.kernels.gathered_gemv
        gu = gemv(x.to(torch.bfloat16), gate_up, experts, L.gate_up_kind, L.hidden, top)
        if self.activation == "gelu_tanh":
            act = self.kernels.geglu(gu)
        else:
            g, u = gu[:, :L.inter], gu[:, L.inter:]
            act = (torch.nn.functional.silu(g) * u).to(torch.bfloat16)
        y = gemv(act, down, experts, L.down_kind, L.inter, 1)
        return self.kernels.combine(y, flat.to(torch.int64), weights.reshape(-1).float(), top)

    def _sorted_routes(self, layer, x, ids, weights):
        """Prompt rows with the stored formats' activations, no host synchronization: routes sorted by expert on the
        device (per-expert counts, row offsets and tile offsets), each expert's routes through the int8 tensor-core
        kernel over its blocks (mmq), the GLU one kernel, the route sums in route order."""
        torch, L = self.torch, self.layers[layer]
        tokens, top = ids.shape
        experts = L.gate_up.shape[0]
        flat = ids.reshape(-1)
        routes = flat.shape[0]
        key = torch.where(flat >= 0, flat, torch.full_like(flat, experts))     # dropped routes sort last, uncomputed
        order = torch.argsort(key, stable=True)
        # per-expert route counts without bincount's host synchronization (it reads the largest key back)
        counts = torch.zeros(experts + 1, dtype=torch.int64, device=x.device).scatter_add_(
            0, key, torch.ones_like(key))[:experts]
        offsets = torch.zeros(experts + 1, dtype=torch.int32, device=x.device)
        offsets[1:] = torch.cumsum(counts, 0)
        tile_off = torch.zeros(experts + 1, dtype=torch.int32, device=x.device)
        tile_off[1:] = torch.cumsum((counts + MMQ_ROWS - 1) // MMQ_ROWS, 0)
        max_tiles = -(-routes // MMQ_ROWS) + experts
        h = x.to(torch.bfloat16).index_select(0, order // top)
        xq, xd = self.kernels.quantize(h)
        del h
        gu = self.kernels.mmq(xq, xd, L.gate_up, L.gate_up_kind, L.hidden, offsets, tile_off, max_tiles)
        del xq, xd
        if self.activation == "gelu_tanh":
            act = self.kernels.geglu(gu)
        else:
            act = (torch.nn.functional.silu(gu[:, :L.inter]) * gu[:, L.inter:]).to(torch.bfloat16)
        del gu
        aq, ad = self.kernels.quantize(act)
        del act
        y = self.kernels.mmq(aq, ad, L.down, L.down_kind, L.inter, offsets, tile_off, max_tiles, out_bf16=True)
        del aq, ad
        # route r's output is row position[r] of y (the routes sorted by expert)
        position = torch.empty_like(order)
        position[order] = torch.arange(routes, device=x.device)
        return self.kernels.combine(y, flat.to(torch.int64), weights.reshape(-1).float(), top, position)

    def _grouped_routes(self, layer, x, ids, weights):
        """Prompt rows: routes sorted by expert into groups of up to DECODE_ROWS rows, each group through one GEMV that
        decodes its expert's stored blocks once for all its rows (device or pinned host memory); each token's routes
        sum in route order. Groups are built on the host (prompt steps are not captured)."""
        torch, F, L = self.torch, self.torch.nn.functional, self.layers[layer]
        tokens, top = ids.shape
        flat = ids.reshape(-1)
        host = flat.tolist()
        order = sorted((e, i) for i, e in enumerate(host) if e >= 0)
        routes = [i for _, i in order]
        experts, starts, counts = [], [], []
        at = 0
        while at < len(order):
            e = order[at][0]
            end = at
            while end < len(order) and order[end][0] == e and end - at < DECODE_ROWS:
                end += 1
            experts.append(e)
            starts.append(at)
            counts.append(end - at)
            at = end
        device = x.device
        index = torch.tensor(routes, dtype=torch.int64, device=device)
        g_e = torch.tensor(experts, dtype=torch.int32, device=device)
        g_s = torch.tensor(starts, dtype=torch.int32, device=device)
        g_c = torch.tensor(counts, dtype=torch.int32, device=device)
        h = x.to(torch.bfloat16).index_select(0, index // top)
        if self.prompt == "fp8":                 # the declared FP8 prompt activations, exact in BF16
            h = fp8_rows.round_trip(h)
        gate_up, down = L.gate_up, L.down
        if gate_up.device.type != "cuda":
            gate_up, down = self._staged(L)      # a host layer copied whole over the link for the prompt step
        gu = self.kernels.grouped_gemv(h, gate_up, g_e, g_s, g_c, L.gate_up_kind, L.hidden)
        g, u = gu[:, :L.inter], gu[:, L.inter:]
        act = F.gelu(g, approximate="tanh") if self.activation == "gelu_tanh" else F.silu(g)
        inner = (act * u).to(torch.bfloat16)
        if self.prompt == "fp8":
            inner = fp8_rows.round_trip(inner)
        y = self.kernels.grouped_gemv(inner, down, g_e, g_s, g_c, L.down_kind, L.inter)
        routed = torch.zeros(tokens * top, L.hidden, dtype=torch.float32, device=device)
        routed[index] = y * weights.reshape(-1).float().index_select(0, index)[:, None]
        routed = routed.view(tokens, top, L.hidden)
        out = routed[:, 0].clone()
        for k in range(1, top):
            out += routed[:, k]
        return out

    def forward(self, layer, x, ids, weights):
        torch, F = self.torch, self.torch.nn.functional
        self.calls += 1
        L = self.layers[layer]
        if self.kernels is not None and L.gate_up_kind in kernels.KINDS and L.down_kind in kernels.KINDS:
            if ids.shape[0] <= DECODE_ROWS:
                # decode rows: every route alone through the gathered GEMVs (capture-safe)
                return self._decode_routes(layer, x, ids, weights)
            # prompt rows: routes grouped by expert, each expert's blocks decoded once a group
            if self.prompt == "stored" and L.gate_up.is_cuda and hasattr(self.kernels, "mmq"):
                return self._sorted_routes(layer, x, ids, weights)
            return self._grouped_routes(layer, x, ids, weights)
        if ids.shape[0] <= DECODE_ROWS and L.gate_up_kind == "BF16":
            return self._dense_routes(layer, x, ids, weights)
        tokens, top = ids.shape
        flat = ids.reshape(-1)
        routed = torch.zeros(tokens * top, L.hidden, dtype=torch.float32, device=x.device)
        valid = flat >= 0
        experts = sorted(set(flat[valid].tolist()))
        a = x.to(torch.bfloat16)
        for first in range(0, len(experts), self.group):
            chosen = experts[first:first + self.group]
            gate_up, down = self._decoded(layer, chosen)
            for j, e in enumerate(chosen):
                routes = (flat == e).nonzero(as_tuple=True)[0]
                h = a[routes // top]
                count = h.shape[0]
                if count < DECODE_ROWS:
                    h = torch.cat([h, h.new_zeros(DECODE_ROWS - count, h.shape[1])])
                prompt = count > DECODE_ROWS and self.prompt == "fp8"
                mm = fp8_rows.mm if prompt else (lambda a, w: torch.mm(a, w.t(), out_dtype=torch.float32))
                gu = mm(h, gate_up[j])
                g, u = gu[:, :L.inter], gu[:, L.inter:]
                act = F.gelu(g, approximate="tanh") if self.activation == "gelu_tanh" else F.silu(g)
                y = mm((act * u).to(torch.bfloat16), down[j])
                routed[routes] = y[:count]
        w = torch.where(valid, weights.reshape(-1).float(), torch.zeros((), device=x.device))
        routed = (routed * w[:, None]).view(tokens, top, L.hidden)
        out = routed[:, 0].clone()
        for k in range(1, top):     # route order: a token's sum never depends on which rows share the call
            out += routed[:, k]
        return out

    def close(self):
        self.layers.clear()
        self.dense.clear()
        self.staging.clear()

    def facts(self):
        return {**kernels.status(), "experts_activation": self.activation,
                "experts_prompt_activations": self.prompt, "experts_decode_activations": self.decode, "experts_formats": ["gguf:IQ3_S", "gguf:IQ4_NL", "gguf:IQ4_XS",
                                                                           "gguf:Q8_0", "bf16"]}

    def telemetry(self):
        return {"gguf_experts": {"calls": self.calls}}


def create(context, options, parts):
    return GgufExperts(options["activation"], int(options["group"]), options["prompt"], options.get("decode", "stored"))
