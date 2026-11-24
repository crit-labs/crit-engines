"""Paged keys and values (and, kept from gemma4-attention, sliding-window rings and derived-key pages), with fused
attention kernels: the attention and kv_cache slots fused. Qwen3.5's full-attention layers use plain paged pools:
pool(layers, kv_heads, head_dim, tokens, device, dtype) keeps stored keys and values, `tokens` positions in pages
shared by every sequence.

pool(layers, kv_heads, head_dim, tokens, device, dtype, window=W, ring=R, sequences=S): a RingPool. Each of up to S
sequences has R positions per layer (position p at ring slot p % R); a step writes its rows, then each query at
position p reads the positions in (p - W, p]. R >= W + the longest row of a step keeps every position a query of the
step reads, and a rollback of the step never needs an overwritten one.

pool(layers, kv_heads, head_dim, tokens, device, dtype, derive={layer: weight}, inv_freq=F): a PagedPool of values
only, `tokens` positions in pages shared by every sequence. write() keeps the values; the attention derives each key
row as rope(weight * value) at its position (half-split pairs, inverse frequencies F, zeros leaving their pairs as
they are): the keys of a layer whose keys and values come from one projection, the values its weightless RMS norm
and the keys its weighted norm rotated.

forward(layer, queries, pool, batch, scale) reads each row's sequence alone, an online softmax over blocks of keys
in FP32 (causal, within the window), so a token's bits never depend on the rows beside it.

The host forms (open/stage/commit/rollback/rows/fork/release, attend) keep plain rows and attend in FP64: the
reference the device form is tested against."""
import math

from .attn import kernels

PIECE = 4096       # positions a prompt step expands to BF16 at once (decode_attention.cu prefill)
MIN_CHUNK = 16     # fewest positions a split of the decode kernel reads (decode_attention.cu MIN_CHUNK)
MAX_SPLITS = 256   # the most splits a launch covers (a longer row's splits read more positions each)

FP8_MAX = 448.0


def _bytes(tensor):
    """The tensor's storage as bytes of its element width (index_copy_ and index_select take no FP8 tensors)."""
    torch = __import__("torch")
    return tensor.view(torch.uint8) if tensor.element_size() == 1 else tensor


def _put(store, index, value):
    _bytes(store).index_copy_(0, index, _bytes(value.to(store.dtype)))


def _take(store, index):
    rows = _bytes(store).index_select(0, index)
    return rows.view(store.dtype) if store.element_size() == 1 else rows


def _dtype(torch, name):
    return {"fp8": torch.float8_e4m3fn, "bf16": torch.bfloat16, "int4": torch.uint8}[name]


GROUP = 32   # INT4: values per FP16 scale


class Store:
    """Rows [layers][slots][heads][dim] of one KV tensor in a format: BF16, FP8 E4M3, or INT4 (signed 4-bit codes,
    two a byte, element 2j in the low nibble, value (code - 8) * scale, one FP16 scale per GROUP values; 4.5 bits a
    value). put() encodes FP32/BF16 rows at slots, take() decodes rows to FP32, kernel(layer) gives the decode kernel
    its codes and scales."""

    def __init__(self, torch, layers, slots, heads, dim, kind, device):
        self.torch, self.kind, self.dim = torch, kind, dim
        self.kernels = kernels.load()
        if kind == torch.uint8:
            if dim % GROUP:
                raise ValueError(f"INT4 KV takes rows of a multiple of {GROUP} values, not {dim}")
            self.data = torch.zeros((layers, slots, heads, dim // 2), dtype=torch.uint8, device=device)
            self.scale = torch.zeros((layers, slots, heads, dim // GROUP), dtype=torch.float16, device=device)
        else:
            self.data = torch.zeros((layers, slots, heads, dim), dtype=kind, device=device)
            self.scale = None

    @property
    def int4(self):
        return self.scale is not None

    def nbytes(self):
        return self.data.numel() * self.data.element_size() + (
            self.scale.numel() * self.scale.element_size() if self.int4 else 0)

    def bytes_per_row(self):
        """Bytes one position takes in one layer, every head."""
        per = self.data[0, 0].numel() * self.data.element_size()
        return per + (self.scale[0, 0].numel() * self.scale.element_size() if self.int4 else 0)

    def encode(self, value):
        torch = self.torch
        if self.int4:
            v = value.float()
            groups = v.view(*v.shape[:-1], self.dim // GROUP, GROUP)
            scale = (groups.abs().amax(dim=-1, keepdim=True) / 7.0).clamp_min(1e-8).half()
            codes = (torch.round(groups / scale.float()).clamp(-8, 7) + 8).to(torch.uint8)
            packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
            return packed.reshape(*v.shape[:-1], self.dim // 2), scale.squeeze(-1)
        if self.kind == torch.float8_e4m3fn:
            return value.float().clamp(-FP8_MAX, FP8_MAX).to(self.kind), None
        return value.to(self.kind), None

    def decode(self, data, scale):
        torch = self.torch
        if self.int4:
            low, high = (data & 15).to(torch.int16), (data >> 4).to(torch.int16)
            codes = torch.stack([low, high], dim=-1).reshape(*data.shape[:-1], self.dim) - 8
            groups = codes.view(*data.shape[:-1], self.dim // GROUP, GROUP).float()
            return (groups * scale.float().unsqueeze(-1)).reshape(*data.shape[:-1], self.dim)
        return data.float()

    def put(self, i, index, value):
        if self.int4 and self.kernels is not None and value.dtype in (self.torch.float32, self.torch.bfloat16):
            # one kernel: each group's scale, codes and the scatter to the rows' slots
            self.kernels.int4_put(value, index, self.data[i].view(-1, *self.data.shape[2:]),
                                  self.scale[i].view(-1, *self.scale.shape[2:]))
            return
        data, scale = self.encode(value)
        _put(self.data[i], index, data)
        if scale is not None:
            self.scale[i].index_copy_(0, index, scale)

    def take(self, i, index):
        return self.decode(_take(self.data[i], index), self.scale[i].index_select(0, index) if self.int4 else None)

    def copy(self, a, b, count):
        self.data[:, b:b + count] = self.data[:, a:a + count]
        if self.int4:
            self.scale[:, b:b + count] = self.scale[:, a:a + count]

    def kernel(self, i):
        return self.data[i], (self.scale[i] if self.int4 else None)


class RingPool:
    def __init__(self, torch, layers, kv_heads, head_dim, sequences, ring, window, device, dtype):
        self.torch = torch
        self.layers = {layer: i for i, layer in enumerate(layers)}
        self.kv_heads, self.head_dim, self.value_dim = kv_heads, head_dim, head_dim
        self.ring, self.window, self.sequences = ring, window, sequences
        self.dtype, self.device = dtype, device
        self.k = Store(torch, len(layers), sequences * ring, kv_heads, head_dim, dtype, device)
        self.v = Store(torch, len(layers), sequences * ring, kv_heads, head_dim, dtype, device)
        self.slot_of, self.lengths, self.free = {}, {}, list(range(sequences - 1, -1, -1))

    def nbytes(self):
        return self.k.nbytes() + self.v.nbytes()

    def open(self, sequence):
        if sequence in self.slot_of:
            raise ValueError(f"sequence {sequence!r} is open")
        if not self.free:
            raise MemoryError(f"every one of the {self.sequences} ring slots is held")
        self.slot_of[sequence], self.lengths[sequence] = self.free.pop(), 0

    def close(self, sequence):
        slot = self.slot_of.pop(sequence, None)
        self.lengths.pop(sequence, None)
        if slot is not None:
            self.free.append(slot)

    def fork(self, sequence, new):
        self.open(new)
        a, b = self.slot_of[sequence] * self.ring, self.slot_of[new] * self.ring
        self.k.copy(a, b, self.ring)
        self.v.copy(a, b, self.ring)
        self.lengths[new] = self.lengths[sequence]

    def length(self, sequence):
        return self.lengths[sequence]

    def reserve(self, batch):
        slots = []
        for r in range(batch.rows):
            sequence, first, count, position = batch.row(r)
            if count > self.ring - self.window:
                raise ValueError(f"a row of {count} tokens overruns a ring of {self.ring} for a window of {self.window}")
            base = self.slot_of[sequence] * self.ring
            slots += [base + (p % self.ring) for p in range(position, position + count)]
        return self.torch.tensor(slots, dtype=self.torch.int64, device=self.device)

    def write(self, layer, slots, keys, values):
        i = self.layers[layer]
        if self.torch.cuda.is_current_stream_capturing():   # a captured decode step: every slot is a row's own
            self.k.put(i, slots, keys)
            self.v.put(i, slots, values)
            return
        keep = slots >= 0
        index = slots[keep]
        self.k.put(i, index, keys[keep])
        self.v.put(i, index, values[keep])

    def keep(self, sequence, length):
        self.lengths[sequence] = length

    def span(self, layer, sequence, first, end):
        """FP32 keys and values of positions first..end - 1 (end - first <= ring)."""
        torch, i = self.torch, self.layers[layer]
        base = self.slot_of[sequence] * self.ring
        index = base + torch.arange(first, end, device=self.device) % self.ring
        return self.k.take(i, index), self.v.take(i, index)

    def gather(self, layer, sequence, count):
        """The window the sequence keeps: positions max(0, count - window) .. count - 1 (a ring holds no others)."""
        return self.span(layer, sequence, max(0, count - self.window), count)

    def base(self, sequence):
        """The sequence's first ring slot (its position p sits at base + p % ring)."""
        return self.slot_of[sequence] * self.ring

    def splits(self, longest=None):
        return min(MAX_SPLITS, -(-self.window // MIN_CHUNK))

    def decode_args(self, layer, sequences, positions):
        torch = self.torch
        return {"base": torch.tensor([self.base(s) for s in sequences], dtype=torch.int64, device=self.device)}

    def kernel_view(self, layer):
        """(keys, key scales, values, value scales) of a layer as the decode kernel reads them."""
        i = self.layers[layer]
        return (*self.k.kernel(i), *self.v.kernel(i))


class PagedPool:
    def __init__(self, torch, layers, kv_heads, head_dim, tokens, page, derive, inv_freq, device, dtype):
        self.torch = torch
        self.layers = {layer: i for i, layer in enumerate(layers)}
        self.kv_heads, self.head_dim, self.value_dim = kv_heads, head_dim, head_dim
        self.window = 0
        self.page, self.dtype, self.device = page, dtype, device
        self.pages = -(-tokens // page) + 8
        self.v = Store(torch, len(layers), self.pages * page, kv_heads, head_dim, dtype, device)
        # a plain pool (derive None) keeps its keys too; a derived pool derives them from the values
        self.derived = derive is not None
        self.k = None if self.derived else Store(torch, len(layers), self.pages * page, kv_heads, head_dim, dtype,
                                                 device)
        self.weight = {layer: derive[layer].float().to(device) for layer in layers} if self.derived else None
        self.inv_freq = inv_freq.float().to(device) if self.derived else None
        self.tables, self.lengths, self.free = {}, {}, list(range(self.pages - 1, -1, -1))

    def nbytes(self):
        return self.v.nbytes() + (self.k.nbytes() if self.k is not None else 0)

    def bytes_per_token(self):
        return len(self.layers) * (self.v.bytes_per_row() + (self.k.bytes_per_row() if self.k is not None else 0))

    def open(self, sequence):
        if sequence in self.tables:
            raise ValueError(f"sequence {sequence!r} is open")
        self.tables[sequence], self.lengths[sequence] = [], 0

    def close(self, sequence):
        self.free += reversed(self.tables.pop(sequence, []))
        self.lengths.pop(sequence, None)

    def fork(self, sequence, new):
        self.open(new)
        for page in self.tables[sequence]:
            fresh = self._page()
            a, b = page * self.page, fresh * self.page
            self.v.copy(a, b, self.page)
            if self.k is not None:
                self.k.copy(a, b, self.page)
            self.tables[new].append(fresh)
        self.lengths[new] = self.lengths[sequence]

    def lend_spaces(self):
        """Byte views of every KV tensor the pool holds ([pages][page bytes] uint8 each), for a lender of the pages no
        sequence holds (codes only: an INT4 store's scales stay out)."""
        spaces = []
        for store in ([self.k] if self.k is not None else []) + [self.v]:
            for i in range(store.data.shape[0]):
                spaces.append(store.data[i].view(self.torch.uint8).reshape(self.pages, -1))   # raw bytes
        return spaces

    def lend_from(self, floor, reclaim):
        """Pages at or above floor are lent; reclaim(page) is called before the pool hands one of them out (it
        returns the new floor)."""
        self.lend_floor, self.reclaim = floor, reclaim

    def _page(self):
        if self.free and getattr(self, "reclaim", None) is not None and self.free[-1] >= self.lend_floor:
            self.lend_floor = self.reclaim(self.free[-1])
        if not self.free:
            raise MemoryError(f"every one of the KV pool's {self.pages} pages is held")
        return self.free.pop()

    def length(self, sequence):
        return self.lengths[sequence]

    def ensure(self, sequence, end):
        """Pages for the sequence's positions below end, allocated on the host (a captured step reads the table)."""
        table = self.tables[sequence]
        while len(table) * self.page < end:
            table.append(self._page())

    def reserve(self, batch):
        slots = []
        for r in range(batch.rows):
            sequence, first, count, position = batch.row(r)
            table = self.tables[sequence]
            while len(table) * self.page < position + count:
                table.append(self._page())
            slots += [table[p // self.page] * self.page + p % self.page for p in range(position, position + count)]
        return self.torch.tensor(slots, dtype=self.torch.int64, device=self.device)

    def write(self, layer, slots, keys, values):
        i = self.layers[layer]
        if self.torch.cuda.is_current_stream_capturing():
            self.v.put(i, slots, values)
            if self.k is not None:
                self.k.put(i, slots, keys)
            return
        keep = slots >= 0
        self.v.put(i, slots[keep], values[keep])
        if self.k is not None:
            self.k.put(i, slots[keep], keys[keep])

    def keep(self, sequence, length):
        self.lengths[sequence] = length
        table = self.tables[sequence]
        while len(table) * self.page - self.page >= max(length, 1) and len(table) > 1:
            self.free.append(table.pop())

    def derive_keys(self, layer, values, positions):
        """rope(weight * value) at each position: FP32 [L][heads][dim]."""
        torch = self.torch
        x = values * self.weight[layer]
        half = self.head_dim // 2
        angles = positions.float()[:, None] * self.inv_freq[None, :]
        cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def span(self, layer, sequence, first, end):
        torch, i = self.torch, self.layers[layer]
        positions = torch.arange(first, end, device=self.device)
        table = torch.tensor(self.tables[sequence], dtype=torch.int64, device=self.device)
        index = table[positions // self.page] * self.page + positions % self.page
        values = self.v.take(i, index)
        if self.k is not None:
            return self.k.take(i, index), values
        return self.derive_keys(layer, values, positions), values

    def gather(self, layer, sequence, count):
        return self.span(layer, sequence, 0, count)

    @property
    def table_width(self):
        return self.pages

    def table_row(self, sequence, out=None):
        """The sequence's page table padded to table_width (int32), on the host."""
        torch = self.torch
        row = torch.zeros(self.pages, dtype=torch.int32) if out is None else out
        table = self.tables[sequence]
        if table:
            row[:len(table)] = torch.tensor(table, dtype=torch.int32)
        return row

    def splits(self, longest=None):
        return min(MAX_SPLITS, -(-((self.pages * self.page) if longest is None else longest) // MIN_CHUNK))

    def decode_args(self, layer, sequences, positions):
        torch = self.torch
        return {"table": torch.stack([self.table_row(s) for s in sequences]).to(self.device)}

    def kernel_view(self, layer):
        i = self.layers[layer]
        return (tuple(self.k.kernel(i)) if self.k is not None else (None, None)) + tuple(self.v.kernel(i))


class Qwen35Attention:
    def __init__(self, dtype, page, block):
        import torch
        self.torch, self.dtype_name, self.page, self.block = torch, dtype, page, block
        self.pools = []
        self.kernels = kernels.load()
        self.host = {}

    # -- kv_cache host form ------------------------------------------------------------------------------------------
    def open(self, sequence, key_width, value_width, capacity):
        self.host[sequence] = {"rows": [], "pending": []}

    def stage(self, sequence, keys, values):
        state = self.host[sequence]
        start = len(state["rows"]) + len(state["pending"])
        state["pending"] += [(start + i, list(k), list(v)) for i, (k, v) in enumerate(zip(keys, values))]

    def commit(self, sequence, count=None):
        state = self.host[sequence]
        count = len(state["pending"]) if count is None else count
        state["rows"] += state["pending"][:count]
        state["pending"] = []

    def rollback(self, sequence):
        self.host[sequence]["pending"] = []

    def rows(self, sequence, pending=False):
        state = self.host[sequence]
        return list(state["rows"]) + (list(state["pending"]) if pending else [])

    def fork(self, sequence, new):
        self.host[new] = {"rows": list(self.host[sequence]["rows"]), "pending": []}

    def release(self, sequence):
        self.host.pop(sequence, None)

    # -- attention host form -----------------------------------------------------------------------------------------
    def attend(self, sequence, queries, positions, scale=None, pending=False):
        rows = self.rows(sequence, pending)
        out = []
        for q, p in zip(queries, positions):
            seen = [(k, v) for pos, k, v in rows if pos <= p]
            s = scale if scale is not None else 1.0 / math.sqrt(len(q))
            scores = [s * sum(a * b for a, b in zip(q, k)) for k, _ in seen]
            top = max(scores)
            w = [math.exp(x - top) for x in scores]
            total = sum(w)
            out.append([sum(wi * v[j] for wi, (_, v) in zip(w, seen)) / total for j in range(len(seen[0][1]))])
        return out

    # -- device forms ------------------------------------------------------------------------------------------------
    def pool(self, layers, kv_heads, head_dim, tokens, device, dtype=None, *, value_dim=None, window=None, ring=None,
             sequences=None, derive=None, inv_freq=None):
        torch = self.torch
        kind = _dtype(torch, self.dtype_name if dtype in (None, "default") else dtype)
        if value_dim not in (None, head_dim):
            raise ValueError("gemma4-attention keeps values as wide as keys")
        if derive is not None:
            made = PagedPool(torch, list(layers), kv_heads, head_dim, tokens, self.page, derive, inv_freq, device, kind)
        elif not window:
            made = PagedPool(torch, list(layers), kv_heads, head_dim, tokens, self.page, None, None, device, kind)
        elif window:
            made = RingPool(torch, list(layers), kv_heads, head_dim, int(sequences or 4), int(ring or 2 * window),
                            int(window), device, kind)
        else:
            raise ValueError("gemma4-attention makes ring pools (window) and derived-key pools (derive)")
        self.pools.append(made)
        return made

    def forward(self, layer, queries, pool, batch, scale=None, blocks=None, block_size=None, parents=None, lse=False):
        if blocks is not None or parents is not None or lse:
            raise ValueError("gemma4-attention reads whole windows: no blocks, trees or log-sum-exps")
        torch = self.torch
        scale = 1.0 / math.sqrt(queries.shape[-1]) if scale is None else scale
        if getattr(batch, "decode_static", None) is not None:
            return self._decode_static(layer, queries, pool, batch, scale)
        out = torch.empty(queries.shape[0], queries.shape[1], pool.value_dim, dtype=queries.dtype,
                          device=queries.device)
        decode = [r for r in range(batch.rows) if batch.starts[r + 1] - batch.starts[r] == 1]
        if self.kernels is None or queries.shape[-1] not in (64, 128, 256, 512) or                 queries.shape[1] // pool.kv_heads > 8:
            decode = []
        if decode:
            # decode rows through the kernel: the same arithmetic as a captured decode step's
            firsts = [batch.starts[r] for r in decode]
            index = torch.tensor(firsts, dtype=torch.int64, device=queries.device)
            sequences = [batch.sequences[r] for r in decode]
            positions = [batch.positions[f] for f in firsts]
            pos = torch.tensor(positions, dtype=torch.int64, device=queries.device)
            args = pool.decode_args(layer, sequences, positions)
            q = queries.index_select(0, index).to(torch.bfloat16).contiguous()
            out[index] = self._kernel(layer, q, pool, pos, args, scale, pool.splits(max(positions) + 1)).to(out.dtype)
        done = set(decode)
        for r in range(batch.rows):
            if r in done:
                continue
            sequence, first, count, position = batch.row(r)
            end = position + count
            start = max(0, position - pool.window + 1) if pool.window else 0
            q = queries[first:first + count]
            if self.kernels is not None and hasattr(self.kernels, "prefill") and q.shape[-1] in (64, 128, 256, 512):
                out[first:first + count] = self._prefill(layer, q, pool, sequence, position, start, scale).to(out.dtype)
                continue
            out[first:first + count] = self._blocked(q, pool, layer, sequence, end, position, start, pool.window,
                                                     scale).to(out.dtype)
        return out

    def _kernel(self, layer, q, pool, pos, args, scale, splits):
        keys, key_scale, values, value_scale = pool.kernel_view(layer)
        if "table" in args:
            derived = pool.weight is not None
            return self.kernels.decode(q, keys, values, pos, None, args["table"], pool.page, 0, 0,
                                       pool.weight[layer] if derived else None, pool.inv_freq, float(scale),
                                       int(splits), key_scale, value_scale)
        return self.kernels.decode(q, keys, values, pos, args["base"], None, 0, pool.ring, pool.window, None, None,
                                   float(scale), int(splits), key_scale, value_scale)

    def _prefill(self, layer, q, pool, sequence, position, start, scale):
        """One sequence's prompt rows through the fused prompt kernel over the pool (decode_attention.cu prefill)."""
        keys, key_scale, values, value_scale = pool.kernel_view(layer)
        args = pool.decode_args(layer, [sequence], [position])
        q = q.to(self.torch.bfloat16).contiguous()
        # 64 query rows a block where the head dim allows (the same rows' arithmetic, a key tile read once per 64 rows)
        prefill = self.kernels.prefill_wide if q.shape[2] <= 256 and hasattr(self.kernels, "prefill_wide")             else self.kernels.prefill
        if "table" in args:
            derived = pool.weight is not None
            return prefill(q, keys, values, key_scale, value_scale, 0, args["table"][0].contiguous(),
                                        pool.page, 0, 0, pool.weight[layer] if derived else None, pool.inv_freq,
                                        int(position), 0, float(scale), PIECE)
        return prefill(q, keys, values, key_scale, value_scale, int(args["base"][0]), None, 0, pool.ring,
                                    pool.window, None, None, int(position), int(start), float(scale), PIECE)

    def _decode_static(self, layer, queries, pool, batch, scale):
        """A captured decode step's rows: positions, ring bases and page tables in the batch's static buffers."""
        meta = batch.decode_static
        args = {"table": meta["table"]} if isinstance(pool, PagedPool) else {"base": meta["base"]}
        q = queries.to(self.torch.bfloat16).contiguous()
        return self._kernel(layer, q, pool, meta["positions"], args, scale, pool.splits()).to(queries.dtype)

    def _blocked(self, q, pool, layer, sequence, end, position, start, window, scale):
        """softmax(scale q k^T) v over the keys at start .. end - 1, causal (and within the window): the pool's keys
        and values read a block of positions at a time (no whole span decoded), scores and the weighted values from
        BF16 operands with FP32 sums on the tensor cores, the online softmax in FP32."""
        torch = self.torch
        count, heads, dim = q.shape
        kv_heads = pool.kv_heads
        group = heads // kv_heads
        qh = (q.to(torch.bfloat16).view(count, kv_heads, group, dim).permute(1, 2, 0, 3)
              .reshape(kv_heads, group * count, dim))
        qpos = (position + torch.arange(count, device=q.device)).repeat(group)          # [group * count]
        m = torch.full((kv_heads, group * count), -math.inf, device=q.device)
        total = torch.zeros((kv_heads, group * count), device=q.device)
        acc = None
        for b in range(start, end, self.block):
            keys, values = pool.span(layer, sequence, b, min(end, b + self.block))
            k = keys.to(torch.bfloat16).transpose(0, 1)                                   # [kv][L][dim]
            v = values.to(torch.bfloat16).transpose(0, 1)
            del keys, values
            kpos = b + torch.arange(k.shape[1], device=q.device)
            allowed = kpos[None, :] <= qpos[:, None]
            if window:
                allowed &= (qpos[:, None] - kpos[None, :]) < window
            s = torch.bmm(qh, k.transpose(1, 2), out_dtype=torch.float32)                   # [kv][gc][L]
            if scale != 1.0:
                s *= scale
            s.masked_fill_(~allowed[None], -math.inf)
            top = torch.maximum(m, s.amax(-1))
            safe = torch.where(torch.isinf(top), torch.zeros_like(top), top)
            s.sub_(safe[..., None]).exp_()
            correction = torch.exp(m - safe)
            total = total * correction + s.sum(-1)
            pv = torch.bmm(s.to(torch.bfloat16), v, out_dtype=torch.float32)
            acc = pv if acc is None else acc.mul_(correction[..., None]).add_(pv)
            m = top
        o = acc / total[..., None]
        return o.view(kv_heads, group, count, -1).permute(2, 0, 1, 3).reshape(count, heads, -1)

    def close(self):
        self.pools.clear()
        self.host.clear()

    def facts(self):
        paged = [p for p in self.pools if isinstance(p, PagedPool)]
        rings = [p for p in self.pools if isinstance(p, RingPool)]
        found = {"kv_dtype": self.dtype_name, **kernels.status(),
                 "attention": "decode rows: decode_attention.cu split-KV kernel; prompt rows: decode_attention.cu prefill "
                              "(spans expanded to BF16 a piece at a time, scores and values on the BF16 tensor cores, "
                              "online softmax in FP32)"}
        if self.pools:
            found["kv_bytes_per_token"] = sum(p.bytes_per_token() for p in paged)
            found["kv_ring_bytes_per_sequence"] = sum(p.nbytes() // p.sequences for p in rings)
        return found


def memory(context, options):
    return {"device": 0, "host": 0}


def create(context, options, parts):
    return Qwen35Attention(options["dtype"], int(options["page"]), int(options["block"]))
