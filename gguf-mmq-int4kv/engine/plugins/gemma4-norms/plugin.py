"""gemma4-norms: torch-norms with the calls a Gemma 4 step makes most as single CUDA kernels (code/norms.cu); anything
else (an offset, a residual, grouped weights, other rotations) goes to torch-norms' torch ops."""
from .g4 import kernels
from .norms import rows, torch_norms


def _fits(x, weight, offset):
    import torch
    return (x.is_cuda and x.dim() in (2, 3) and x.stride(-1) == 1 and x.dtype in (torch.float32, torch.bfloat16)
            and not offset and (weight is None or (weight.dim() == 1 and weight.dtype == torch.float32
                                                    and weight.is_contiguous())))


def _out_fp32(x, out_dtype):
    import torch
    dtype = out_dtype or x.dtype
    if dtype not in (torch.float32, torch.bfloat16):
        return None
    return dtype == torch.float32


class Gemma4Norms(rows.Rows):
    def __init__(self):
        self.kernels = kernels.load()

    def rms_norm(self, x, weight, eps, offset=0.0, residual=None, out_dtype=None, out=None):
        fp32 = _out_fp32(x, out_dtype)
        if (self.kernels is None or residual is not None or out is not None or fp32 is None
                or not _fits(x, weight, offset)):
            return torch_norms.rms_norm(x, weight, eps, offset, residual, out_dtype, out)
        return self.kernels.rms_norm(x, weight, float(eps), fp32, None, None)

    def rms_norm_heads(self, x, weight, eps, offset=0.0, out_dtype=None, out=None):
        fp32 = _out_fp32(x, out_dtype)
        if self.kernels is None or out is not None or fp32 is None or not _fits(x, weight, offset):
            return torch_norms.rms_norm_heads(x, weight, eps, offset, out_dtype, out)
        return self.kernels.rms_norm(x, weight, float(eps), fp32, None, None)

    def rms_norm_rope(self, x, weight, eps, positions, freqs, offset=0.0, rotary_dim=None, style="half", scale=1.0,
                      out_dtype=None, out=None):
        fp32 = _out_fp32(x, out_dtype)
        simple = (style == "half" and scale == 1.0 and freqs.dim() == 1 and rotary_dim in (None, x.shape[-1])
                  and 2 * freqs.shape[0] == x.shape[-1] and x.dim() == 3)
        if (self.kernels is None or out is not None or fp32 is None or not simple
                or not _fits(x, weight, offset)):
            return torch_norms.rms_norm_rope(x, weight, eps, positions, freqs, offset, rotary_dim, style, scale,
                                             out_dtype, out)
        return self.kernels.rms_norm(x, weight, float(eps), fp32, positions, freqs.float())

    def gelu_mul(self, x, up=None, layout="halves", approximate="none", out_dtype=None, out=None):
        import torch
        fp32 = _out_fp32(x, out_dtype)
        if (self.kernels is None or up is not None or layout != "halves" or approximate != "tanh" or out is not None
                or fp32 is None or x.dim() != 2 or x.stride(-1) != 1
                or x.dtype not in (torch.float32, torch.bfloat16)):
            return torch_norms.gelu_mul(x, up, layout, approximate, out_dtype, out)
        return self.kernels.gelu_mul(x, fp32)

    # -- Gemma 4's fused calls (the model uses them where the plugin has them) ------------------------------------
    def fused_ready(self, x):
        import torch
        return self.kernels is not None and x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()

    def add_rms(self, x, y, weight, eps):
        """x += rms_norm(y) * weight, in place (x FP32 [tokens][dim])."""
        self.kernels.add_rms(x, y.contiguous(), weight, float(eps))
        return x

    def rms3(self, x, w1, w2, w3, eps):
        """(rms_norm(x) * w1, * w2, * w3) in BF16 from one sum of squares."""
        return tuple(self.kernels.rms3(x, w1, w2, w3, float(eps)))

    def ffn_out(self, x, mlp, e, w1, w2, w, scale, eps, next_weight=None):
        """x = (x + rms_norm(rms_norm(mlp) * w1 + rms_norm(e) * w2) * w) * scale in place; with next_weight also
        returns rms_norm(x) * next_weight in BF16 (the next layer's input)."""
        return self.kernels.ffn_out(x, mlp.contiguous(), e.contiguous(), w1, w2, w, float(scale), float(eps),
                                    next_weight)

    def facts(self):
        return kernels.status()


for _name in torch_norms.OPERATIONS:
    if _name not in Gemma4Norms.__dict__:
        setattr(Gemma4Norms, _name, staticmethod(getattr(torch_norms, _name)))


def create(context, options, parts):
    return Gemma4Norms()
