"""qwen35-attention's own CUDA kernel (decode_attention.cu), built by build_kernels.py into build/ beside this file.
load() returns the module or None (decode rows then attend in torch)."""
import importlib.util
import os
from pathlib import Path

_STATE = {"module": None, "tried": False, "reason": None}


def load():
    if _STATE["tried"]:
        return _STATE["module"]
    _STATE["tried"] = True
    if os.environ.get("QWEN35_ATTENTION_KERNELS", "1") == "0":
        _STATE["reason"] = "turned off (QWEN35_ATTENTION_KERNELS=0)"
        return None
    build = Path(__file__).resolve().parent / "build"
    found = sorted(build.glob("decode_attention*.pyd")) + sorted(build.glob("decode_attention*.so"))
    if not found:
        _STATE["reason"] = "no build of decode_attention.cu (run build_kernels.py)"
        return None
    try:
        import torch  # noqa: F401
        spec = importlib.util.spec_from_file_location("decode_attention", str(found[0].resolve()))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (ImportError, OSError) as error:
        _STATE["reason"] = f"build did not load: {error}"
        return None
    _STATE["module"] = module
    return module


def status():
    module = load()
    return {"decode_attention": "local build" if module is not None else "torch fallback",
            **({} if module is not None else {"decode_attention_reason": _STATE["reason"]})}
