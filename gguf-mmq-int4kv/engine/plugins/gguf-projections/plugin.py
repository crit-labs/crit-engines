"""GGUF block matrices in the projections slot. A matrix stays on the device as its GGUF rows (uint8 [rows][row_bytes]);
matmul decodes a chunk of output rows to BF16 (gguf.gguf_blocks.decode, bit for bit gguf-py's) and multiplies the
activations with torch's BF16 GEMM, FP32 accumulation, rounding once to the output dtype. A call of up to DECODE_ROWS
rows is padded to DECODE_ROWS, so each row's bits are those it gets alone (row invariance); longer calls are prompt
rows, which (prompt fp8) multiply in FP8 with row-wise scales (code/fp8_rows.py) on the FP8 tensor cores. The host form multiplies dense rows in FP64 (the reference the device form is tested against)."""
from .gguf import fp8_rows, gguf_blocks, kernels

DECODE_ROWS = 8
EXACT_ROWS = 65536      # matrices of more rows (the head) are read at bandwidth by the FP32 GEMV: keep exact activations


class Matrix:
    def __init__(self, raw, kind, rows, cols, dense=None):
        self.raw, self.kind, self.rows, self.cols, self.dense = raw, kind, rows, cols, dense

    @property
    def shape(self):
        return (self.rows, self.cols)

    def nbytes(self):
        return int((self.raw if self.raw is not None else self.dense).numel()
                   * (1 if self.raw is not None else 2))


class GgufProjections:
    def __init__(self, chunk_bytes, prompt="fp8", gemv_rows=128, decode="stored", q8_layout="stored"):
        import torch
        self.torch, self.chunk_bytes, self.prompt, self.decode = torch, chunk_bytes, prompt, decode
        self.q8_layout = q8_layout
        self.kernels = kernels.load()
        self.gemv_rows = gemv_rows

    # -- host form ---------------------------------------------------------------------------------------------------
    def prepare(self, rows):
        return [list(map(float, row)) for row in rows]

    def project(self, weight, activations):
        return [[sum(w * a for w, a in zip(row, x)) for row in weight] for x in activations]

    def dense(self, weight):
        return weight

    # -- device form -------------------------------------------------------------------------------------------------
    def load(self, tensors, format, role=None, encode=None):
        torch = self.torch
        weight = tensors["weight"]
        if format == "bf16":
            dense = weight.to(device="cuda", dtype=torch.bfloat16).contiguous()
            return Matrix(None, "bf16", dense.shape[0], dense.shape[1], dense)
        if not format.startswith("gguf:"):
            raise ValueError(f"gguf-projections takes gguf:KIND or bf16 matrices, not {format}")
        kind, cols = format.split(":", 1)[1], int(tensors["cols"])
        if kind not in gguf_blocks.FORMATS:
            raise ValueError(f"no GGUF decoder for {kind}")
        raw = weight if weight.device.type == "cuda" else weight.to("cuda")
        if raw.dtype != torch.uint8 or raw.shape[1] != gguf_blocks.row_bytes(kind, cols):
            raise ValueError(f"{kind}: raw rows [rows][{gguf_blocks.row_bytes(kind, cols)}] uint8 expected")
        raw = raw.contiguous()
        if (kind == "Q8_0" and self.q8_layout == "aligned" and role != "embedding" and raw.is_cuda
                and self.decode == "stored" and self.prompt == "stored" and self.kernels is not None
                and hasattr(self.kernels, "repack_q8")):
            # the same bytes in the aligned order (codes [rows][cols], then scales): the decode GEMV reads 16-byte
            # code words straight into dp4a, the prompt kernel and the dequantizer read the same order
            source = raw.clone()
            self.kernels.repack_q8(source, raw, cols)
            del source
            kind = "Q8_0R"
        return Matrix(raw, kind, raw.shape[0], cols)

    def decoded(self, matrix, first=0, end=None):
        """BF16 rows first..end of a matrix."""
        end = matrix.rows if end is None else end
        if matrix.dense is not None:
            return matrix.dense[first:end]
        if self.kernels is not None and matrix.kind in kernels.KINDS:
            return self.kernels.dequant(matrix.raw[first:end], matrix.kind, matrix.cols)
        return gguf_blocks.decode(matrix.raw[first:end], matrix.kind, matrix.cols).to(self.torch.bfloat16)

    def rows(self, matrix, ids):
        """Rows of a matrix by id (an embedding table's lookup), FP32 [len(ids)][cols]."""
        if matrix.dense is not None:
            return matrix.dense[ids].float()
        return gguf_blocks.decode(matrix.raw[ids], matrix.kind, matrix.cols)

    def matmul(self, x, matrix, out_dtype=None):
        torch = self.torch
        out_dtype = out_dtype or x.dtype
        tokens = x.shape[0]
        a = x.to(torch.bfloat16)
        if tokens <= DECODE_ROWS and self.kernels is not None and matrix.kind in kernels.KINDS:
            # decode rows: the GEMV over the stored blocks, each row its own warp sums (row invariant); with the
            # stored formats' activations each row is quantized alone to int8 units (still row invariant)
            if self.decode == "stored" and matrix.rows <= EXACT_ROWS:
                return self.kernels.gemv_q8(a, matrix.raw, matrix.kind, matrix.cols).to(out_dtype)
            return self.kernels.gemv(a, matrix.raw, matrix.kind, matrix.cols).to(out_dtype)
        if self.prompt == "stored" and self.kernels is not None and matrix.kind in kernels.KINDS:
            # prompt rows with the stored formats' activations: int8 units of 32 into the int8 tensor cores straight
            # from the blocks (mmq), each row's units quantized alone
            xq, xd = self.kernels.quantize(a.contiguous())
            return self.kernels.mmq(xq, xd, matrix.raw, matrix.kind, matrix.cols,
                                    out_bf16=out_dtype == torch.bfloat16).to(out_dtype)
        if tokens <= self.gemv_rows and self.kernels is not None and matrix.kind in kernels.KINDS:
            # short prompt chunks: GEMV passes of DECODE_ROWS rows over the stored blocks, cheaper than decoding the
            # whole matrix for the tensor cores (each row still its own warp's sums)
            a = fp8_rows.round_trip(a) if self.prompt == "fp8" else a   # the declared FP8 prompt activations
            return torch.cat([self.kernels.gemv(a[i:i + DECODE_ROWS], matrix.raw, matrix.kind, matrix.cols)
                              for i in range(0, tokens, DECODE_ROWS)]).to(out_dtype)
        prompt = tokens > DECODE_ROWS and self.prompt == "fp8" and matrix.cols % 16 == 0
        if tokens <= DECODE_ROWS:
            a = torch.cat([a, a.new_zeros(DECODE_ROWS - tokens, a.shape[1])]) if tokens < DECODE_ROWS else a

        def product(w):
            return fp8_rows.mm(a, w) if prompt else torch.mm(a, w.t(), out_dtype=torch.float32)

        step = max(16, self.chunk_bytes // (2 * matrix.cols) // 16 * 16)
        if step >= matrix.rows:
            y = product(self.decoded(matrix))
        else:
            y = torch.empty(a.shape[0], matrix.rows, dtype=torch.float32, device=a.device)
            for first in range(0, matrix.rows, step):
                end = min(matrix.rows, first + step)
                y[:, first:end] = product(self.decoded(matrix, first, end))
        return y[:tokens].to(out_dtype)

    def facts(self):
        if self.prompt == "stored":
            rule = ("prompt rows (calls of more than 8 rows): the stored formats' int8 units of 32 with an FP32 scale, "
                    "multiplied with the blocks' int8 codes on the int8 tensor cores (mmq)")
        else:
            rule = (f"calls of up to {self.gemv_rows} rows: {self.prompt} rows (power-of-two row scales) through the "
                    f"stored-block GEMV; longer: {self.prompt} on the tensor cores")
        return {"prompt_activations": {"gguf": self.prompt}, "decode_activations": self.decode,
                "prompt_activation_rule": rule,
                "row_invariance": {"rows": DECODE_ROWS}, **kernels.status(),
                "projections_formats": sorted(gguf_blocks.FORMATS)}


def create(context, options, parts):
    return GgufProjections(int(options["chunk_mib"]) << 20, options["prompt"], int(options["gemv_rows"]),
                           options.get("decode", "stored"), options.get("q8_layout", "stored"))
