"""Gated DeltaNet recurrence with per-sequence device state (the recurrence slot).

pool(layers, channels, key_heads, value_heads, sequences, device) -> RecurrentPool: for every listed layer a conv
window [slots][3][channels] and a delta-rule state [slots][value_heads][128][128] (FP32, column-major per value
column), slots handed to sequences by open() (zeroed) and back by close(). forward(layer, mixed, conv_weight, g, beta,
groups): mixed [T][channels] FP32 (queries, keys and values of every head before the conv), g / beta [T][value_heads]
FP32 (log decay and write strength), groups a Groups of device int32 tensors (starts, counts, slots): out
[T][value_heads][128] FP32. The torch reference runs the same arithmetic token by token (the plugin's tests check the
kernels against it)."""
import math

from .gdn import kernels

DK = DV = 128


class Groups:
    """Device int32 starts / counts / slots of a step's groups (each a run of one sequence's consecutive tokens)."""

    def __init__(self, starts, counts, slots):
        self.starts, self.counts, self.slots = starts, counts, slots

    @classmethod
    def build(cls, torch, rows, device):
        """rows: [(start, count, slot)] on the host."""
        t = torch.tensor(rows, dtype=torch.int32).reshape(-1, 3).t().contiguous().to(device)
        return cls(t[0].contiguous(), t[1].contiguous(), t[2].contiguous())


class RecurrentPool:
    def __init__(self, torch, layers, channels, key_heads, value_heads, sequences, device, eps):
        self.torch, self.device, self.eps = torch, device, eps
        self.layers = {layer: i for i, layer in enumerate(layers)}
        self.channels, self.kh, self.vh = channels, key_heads, value_heads
        self.sequences = sequences
        self.conv = torch.zeros((len(layers), sequences, 3, channels), dtype=torch.float32, device=device)
        self.state = torch.zeros((len(layers), sequences, value_heads, DV, DK), dtype=torch.float32, device=device)
        self.slot_of, self.free = {}, list(range(sequences - 1, -1, -1))
        self.kernels = kernels.load()

    def nbytes(self):
        return self.conv.numel() * 4 + self.state.numel() * 4

    def bytes_per_sequence(self):
        return self.nbytes() // self.sequences

    def open(self, sequence):
        if sequence in self.slot_of:
            raise ValueError(f"sequence {sequence!r} is open")
        if not self.free:
            raise MemoryError(f"every one of the recurrent pool's {self.sequences} slots is held")
        slot = self.free.pop()
        self.slot_of[sequence] = slot
        self.conv[:, slot].zero_()
        self.state[:, slot].zero_()
        return slot

    def close(self, sequence):
        slot = self.slot_of.pop(sequence, None)
        if slot is not None:
            self.free.append(slot)

    def slot(self, sequence):
        return self.slot_of[sequence]

    def convolve(self, layer, mixed, conv_weight, groups):
        """silu(causal conv) of mixed [T][channels] over each group's window, the windows advanced in place."""
        i = self.layers[layer]
        if self.kernels is not None:
            return self.kernels.conv(mixed.contiguous(), conv_weight, self.conv[i], groups.starts, groups.counts,
                                     groups.slots)
        return self._reference_conv(i, mixed, conv_weight, groups)

    def scan(self, layer, conv, g, beta, groups):
        """The gated delta rule over conv [T][channels] (queries, keys, values after the conv), state in place."""
        i = self.layers[layer]
        if self.kernels is not None:
            return self.kernels.scan(conv.contiguous(), g.contiguous(), beta.contiguous(), self.state[i],
                                     groups.starts, groups.counts, groups.slots, self.kh, self.vh, self.eps)
        return self._reference_scan(i, conv, g, beta, groups)

    def _reference_conv(self, i, mixed, conv_weight, groups):
        torch = self.torch
        out = torch.empty_like(mixed, dtype=torch.float32)
        for start, count, slot in zip(groups.starts.tolist(), groups.counts.tolist(), groups.slots.tolist()):
            if count <= 0:
                continue
            x = torch.cat([self.conv[i, slot], mixed[start:start + count].float()])
            conv = sum(x[j:j + count] * conv_weight[:, j] for j in range(4))
            out[start:start + count] = conv * torch.sigmoid(conv)
            self.conv[i, slot] = x[-3:]
        return out

    def _reference_scan(self, i, conv, g, beta, groups):
        torch = self.torch
        T = conv.shape[0]
        out = torch.zeros((T, self.vh, DV), dtype=torch.float32, device=conv.device)
        kh, vh = self.kh, self.vh
        heads = torch.arange(vh, device=conv.device) % kh
        for start, count, slot in zip(groups.starts.tolist(), groups.counts.tolist(), groups.slots.tolist()):
            S = self.state[i, slot].transpose(1, 2).clone()
            for t in range(count):
                row = conv[start + t]
                q = row[:kh * DK].view(kh, DK)[heads]
                k = row[kh * DK:2 * kh * DK].view(kh, DK)[heads]
                v = row[2 * kh * DK:].view(vh, DV)
                q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + self.eps) / math.sqrt(DK)
                k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + self.eps)
                S = S * torch.exp(g[start + t])[:, None, None]
                kv = torch.einsum("hkv,hk->hv", S, k)
                delta = (v - kv) * beta[start + t][:, None]
                S = S + k[:, :, None] * delta[:, None, :]
                out[start + t] = torch.einsum("hkv,hk->hv", S, q)
            self.state[i, slot] = S.transpose(1, 2)
        return out

    def forward(self, layer, mixed, conv_weight, g, beta, groups):
        i = self.layers[layer]
        if self.kernels is not None:
            x = self.kernels.conv(mixed.contiguous(), conv_weight, self.conv[i], groups.starts, groups.counts,
                                  groups.slots)
            return self.kernels.scan(x, g.contiguous(), beta.contiguous(), self.state[i], groups.starts,
                                     groups.counts, groups.slots, self.kh, self.vh, self.eps)
        return self.reference(i, mixed, conv_weight, g, beta, groups)

    def reference(self, i, mixed, conv_weight, g, beta, groups):
        torch = self.torch
        T = mixed.shape[0]
        out = torch.zeros((T, self.vh, DV), dtype=torch.float32, device=mixed.device)
        kh, vh = self.kh, self.vh
        heads = torch.arange(vh, device=mixed.device) % kh
        for start, count, slot in zip(groups.starts.tolist(), groups.counts.tolist(), groups.slots.tolist()):
            if count <= 0:
                continue
            window = self.conv[i, slot]                                   # [3][C]
            x = torch.cat([window, mixed[start:start + count].float()])   # [3 + n][C]
            conv = sum(x[j:j + count] * conv_weight[:, j] for j in range(4))
            conv = conv * torch.sigmoid(conv)
            self.conv[i, slot] = x[-3:]
            S = self.state[i, slot].transpose(1, 2).clone()               # [vh][DK][DV]
            for t in range(count):
                row = conv[t]
                q = row[:kh * DK].view(kh, DK)[heads]
                k = row[kh * DK:2 * kh * DK].view(kh, DK)[heads]
                v = row[2 * kh * DK:].view(vh, DV)
                q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + self.eps) / math.sqrt(DK)
                k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + self.eps)
                S = S * torch.exp(g[start + t])[:, None, None]
                kv = torch.einsum("hkv,hk->hv", S, k)
                delta = (v - kv) * beta[start + t][:, None]
                S = S + k[:, :, None] * delta[:, None, :]
                out[start + t] = torch.einsum("hkv,hk->hv", S, q)
            self.state[i, slot] = S.transpose(1, 2)
        return out


class Qwen35Recurrence:
    def __init__(self):
        import torch
        self.torch = torch
        self.pools = []

    def pool(self, layers, channels, key_heads, value_heads, sequences, device, eps=1e-6):
        made = RecurrentPool(self.torch, list(layers), channels, key_heads, value_heads, sequences, device, eps)
        self.pools.append(made)
        return made

    def groups(self, rows, device):
        return Groups.build(self.torch, rows, device)

    def gates(self, ba, dt_bias, a):
        """(beta, g): sigmoid of the beta projections, softplus(alpha + dt_bias) * a, from [T][2 vh] FP32."""
        k = kernels.load()
        if k is not None:
            return tuple(k.gates(ba, dt_bias, a))
        torch, vh = self.torch, dt_bias.numel()
        return torch.sigmoid(ba[:, :vh]), torch.nn.functional.softplus(ba[:, vh:2 * vh] + dt_bias) * a

    def gated_norm(self, core, z, weight, eps):
        """rms_norm(core) * weight * silu(z) per value head, BF16 [T][vh * DV]."""
        k = kernels.load()
        if k is not None:
            return k.gated_rms(core.contiguous(), z, weight, eps)
        torch = self.torch
        T, vh, dv = core.shape
        n = core * torch.rsqrt((core * core).mean(-1, keepdim=True) + eps) * weight
        return (n * torch.nn.functional.silu(z.reshape(T, vh, dv))).to(torch.bfloat16).reshape(T, -1)

    # -- the recurrence slot's device form ---------------------------------------------------------------------------
    # convolve(layer, x, weight, bias, pool, batch): the causal conv over each sequence's window in the pool, the
    # batch's groups (batch.recurrent_groups, device int32 starts / counts / slots) naming the windows;
    # forward(layer, query, key, value, log_decay, beta, pool, batch, geometry): the gated delta rule, query holding the
    # whole conv output (queries, keys and values side by side, key and value None) or the three parts apart.
    def convolve(self, layer, x, weight, bias, pool, batch, activation="silu", out=None):
        if bias is not None or activation != "silu":
            raise ValueError("qwen35-recurrence convolves without a bias, with SiLU")
        return pool.convolve(layer, x, weight, batch.recurrent_groups)

    def forward(self, layer, query, key, value, log_decay, beta, pool, batch, geometry=None):
        conv = query if key is None else self.torch.cat([query.reshape(query.shape[0], -1),
                                                         key.reshape(key.shape[0], -1),
                                                         value.reshape(value.shape[0], -1)], -1)
        return pool.scan(layer, conv, log_decay, beta, batch.recurrent_groups)

    def close(self):
        self.pools.clear()

    def facts(self):
        found = {**kernels.status(), "recurrence": "gated delta rule, FP32 state, a value column a warp"}
        if self.pools:
            found["recurrent_bytes_per_sequence"] = sum(p.bytes_per_sequence() for p in self.pools)
        return found


def memory(context, options):
    return {"device": 0, "host": 0}


def create(context, options, parts):
    return Qwen35Recurrence()
