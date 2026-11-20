"""Prompt rows through FP8 tensor cores: y = x W^T with each activation row and each weight row scaled into FP8 E4M3
by its own amax (row-wise scaling), multiplied by torch's FP8 GEMM at unit scales and rescaled in FP32 (sm_120's
torch has no row-wise scaled_mm; scaling the FP32 output is the same arithmetic). Every row's result depends on its
own row and the weight only."""
import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
_ONE = {}


def _one(device):
    found = _ONE.get(str(device))
    if found is None:
        found = _ONE[str(device)] = torch.ones((), dtype=torch.float32, device=device)
    return found


def quantize_rows(values):
    """(FP8 rows, FP32 scales [rows][1]) of a [rows][cols] tensor, each row scaled by amax / 448."""
    v = values.float()
    scale = (v.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp_min(1e-12)
    return (v / scale).to(FP8), scale


def round_trip(x):
    """x's rows as FP8 E4M3 values, each row scaled by a power of two (amax to at most 448), returned in BF16: E4M3's
    3 mantissa bits times a power of two are exact in BF16, so a kernel taking BF16 rows multiplies FP8 activations."""
    v = x.float()
    amax = v.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX)))
    return ((v / scale).to(FP8).float() * scale).to(torch.bfloat16)


def mm(x, weight):
    """FP32 [tokens][out] = x [tokens][in] times weight [out][in] (any float dtype), both rows in FP8; shapes the
    FP8 GEMM does not take (in or out not a multiple of 16) multiply in BF16 instead."""
    if x.shape[1] % 16 or weight.shape[0] % 16:
        return torch.mm(x.to(torch.bfloat16), weight.to(torch.bfloat16).t(), out_dtype=torch.float32)
    xq, sx = quantize_rows(x)
    wq, sw = quantize_rows(weight)
    one = _one(x.device)
    return torch._scaled_mm(xq, wq.t(), scale_a=one, scale_b=one, out_dtype=torch.float32) * sx * sw.t()
