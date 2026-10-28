"""The engine's tensor-core NVFP4 GEMV (csrc/nvfp4_mma.cu), built into csrc/ by csrc/build.py (the package's build
command) and loaded from there."""
import importlib.util
import os

NAME = "qwen38_nvfp4_mma"
_SO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", NAME + ".so")
_spec = importlib.util.spec_from_file_location(NAME, _SO)
_ext = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ext)


def nvfp4_mma(x, xs, w, ws, alpha, split, wide, per_row=False):
    """x packed uint8 [M, K/2] (M <= 8) with swizzled e4m3 scales xs; w packed uint8 [N, K/2] with swizzled scales ws;
    alpha fp32 [1], or [M] with per_row (one factor per column). Returns alpha * x w^T as bf16 [M, N]."""
    return _ext.nvfp4_mma(x, xs, w, ws, alpha, split, wide, per_row)
