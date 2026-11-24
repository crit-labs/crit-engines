"""GGUF block formats decoded on the device in torch, bit for bit as gguf-py's quants.dequantize decodes them (FP32
values): Q8_0, IQ4_NL, IQ4_XS, IQ3_S and Q6_K, the formats a GGUF of this engine stores its matrices in, plus F32, F16 and
BF16 rows. A matrix is its raw rows: a uint8 tensor [rows][row_bytes], each row's blocks along its input dimension, as
the GGUF stores it (ggml's block layouts, ggml-common.h, MIT License, Copyright (c) 2023-2026 The ggml authors).

decode(raw, kind, cols) -> FP32 [rows][cols]. Every row decodes alone, so a row's values never depend on the rows
beside it. FORMATS names each kind's (block elements, block bytes)."""
import torch

from .iq3s_grid import IQ3S_GRID

FORMATS = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q8_0": (32, 34), "IQ4_NL": (32, 18), "IQ4_XS": (256, 136),
           "IQ3_S": (256, 110), "Q6_K": (256, 210)}
KVALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)

_TABLES = {}


def _table(name, device):
    key = (name, str(device))
    found = _TABLES.get(key)
    if found is None:
        if name == "kvalues":
            found = torch.tensor(KVALUES, dtype=torch.float32, device=device)
        elif name == "grid":
            found = torch.tensor(IQ3S_GRID, dtype=torch.float32, device=device).view(512, 4)
        elif name == "bits8":
            found = torch.arange(8, dtype=torch.uint8, device=device)
        _TABLES[key] = found
    return found


def row_bytes(kind, cols):
    elements, size = FORMATS[kind]
    if cols % elements:
        raise ValueError(f"{kind} rows hold whole blocks of {elements}; {cols} columns do not")
    return cols // elements * size


def _half(raw):
    # little-endian f16 from two uint8 columns
    return raw.contiguous().view(torch.float16).float()


def _nibbles(qs):
    """[..., 16] bytes -> [..., 32] nibbles: the low nibbles first, then the high ones (ggml's order)."""
    return torch.cat([qs & 0x0F, qs >> 4], dim=-1).long()


def decode(raw, kind, cols):
    """FP32 values [rows][cols] of raw rows [rows][row_bytes(kind, cols)] (uint8)."""
    if raw.dtype != torch.uint8 or raw.dim() != 2:
        raise ValueError("raw rows are a uint8 tensor [rows][row_bytes]")
    rows, device = raw.shape[0], raw.device
    if raw.shape[1] != row_bytes(kind, cols):
        raise ValueError(f"{kind}: {cols} columns take {row_bytes(kind, cols)} bytes a row, not {raw.shape[1]}")
    if kind == "F32":
        return raw.contiguous().view(torch.float32).view(rows, cols).clone()
    if kind == "F16":
        return raw.contiguous().view(torch.float16).view(rows, cols).float()
    if kind == "BF16":
        return raw.contiguous().view(torch.bfloat16).view(rows, cols).float()
    elements, size = FORMATS[kind]
    blocks = raw.reshape(rows * (cols // elements), size)
    n = blocks.shape[0]
    if kind == "Q8_0":
        d = _half(blocks[:, :2])
        q = blocks[:, 2:].contiguous().view(torch.int8).float()
        return (q * d).view(rows, cols)
    if kind == "IQ4_NL":
        d = _half(blocks[:, :2])
        q = _table("kvalues", device)[_nibbles(blocks[:, 2:])]
        return (q * d).view(rows, cols)
    if kind == "IQ4_XS":
        d = _half(blocks[:, :2])
        scales_h = blocks[:, 2:4].contiguous().view(torch.int16).int() & 0xFFFF          # [n][1]
        scales_l = blocks[:, 4:8]                                                        # [n][4]
        low = torch.stack([scales_l & 0x0F, scales_l >> 4], dim=-1).reshape(n, 8).int()  # sub-block 2i, 2i + 1
        shifts = torch.arange(0, 16, 2, dtype=torch.int32, device=device)
        high = (scales_h >> shifts) & 0x03                                               # [n][8]
        scales = (low | (high << 4)) - 32
        dl = d * scales.float()                                                          # [n][8]
        qs = blocks[:, 8:].reshape(n, 8, 16)
        q = _table("kvalues", device)[_nibbles(qs)]                                      # [n][8][32]
        return (dl[:, :, None] * q).view(rows, cols)
    if kind == "IQ3_S":
        d = _half(blocks[:, :2])                                                         # [n][1]
        qs = blocks[:, 2:66].long()                                                      # [n][64]
        qh = blocks[:, 66:74]                                                            # [n][8]
        signs = blocks[:, 74:106]                                                        # [n][32]
        scales = blocks[:, 106:110]                                                      # [n][4]
        nib = torch.stack([scales & 0x0F, scales >> 4], dim=-1).reshape(n, 8).float()
        db = d * (1 + 2 * nib)                                                           # [n][8]
        bits = _table("bits8", device)
        sign = ((signs[:, :, None] >> bits) & 1).reshape(n, 8, 4, 8)
        sign = 1.0 - 2.0 * sign.float()
        high = ((qh[:, :, None] >> bits) & 1).reshape(n, 64).long()
        index = qs | (high << 8)                                                         # [n][64]
        grid = _table("grid", device)[index].reshape(n, 8, 4, 8)
        return (db[:, :, None, None] * grid * sign).view(rows, cols)
    if kind == "Q6_K":
        ql = blocks[:, :128].long()                                                      # [n][128]
        qh = blocks[:, 128:192].long()                                                   # [n][64]
        sc = blocks[:, 192:208].contiguous().view(torch.int8).float()                    # [n][16]
        d = _half(blocks[:, 208:210])                                                    # [n][1]
        out = []
        for half in range(2):
            l_lo, l_hi = ql[:, 64 * half:64 * half + 32], ql[:, 64 * half + 32:64 * half + 64]
            h = qh[:, 32 * half:32 * half + 32]
            q1 = ((l_lo & 15) | (((h >> 0) & 3) << 4)) - 32
            q2 = ((l_hi & 15) | (((h >> 2) & 3) << 4)) - 32
            q3 = ((l_lo >> 4) | (((h >> 4) & 3) << 4)) - 32
            q4 = ((l_hi >> 4) | (((h >> 6) & 3) << 4)) - 32
            s = sc[:, 8 * half:8 * half + 8]
            for m, q in enumerate((q1, q2, q3, q4)):
                scale = torch.cat([s[:, 2 * m:2 * m + 1].expand(n, 16), s[:, 2 * m + 1:2 * m + 2].expand(n, 16)], 1)
                out.append(d * scale * q.float())
        return torch.cat(out, 1).view(rows, cols)
    raise ValueError(f"no decoder for GGUF block type {kind}")
