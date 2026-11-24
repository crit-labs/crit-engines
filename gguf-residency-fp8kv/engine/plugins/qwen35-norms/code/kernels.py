"""qwen35-norms' CUDA kernels (norms.cu), built by build_kernels.py into build/ beside this file; None where absent."""
import importlib.util
import os
from pathlib import Path

_STATE = {"module": None, "tried": False, "reason": None}


def load():
    if _STATE["tried"]:
        return _STATE["module"]
    _STATE["tried"] = True
    if os.environ.get("QWEN35_NORMS_KERNELS", "1") == "0":
        _STATE["reason"] = "turned off (QWEN35_NORMS_KERNELS=0)"
        return None
    build = Path(__file__).resolve().parent / "build"
    found = sorted(build.glob("gemma4_norms*.pyd")) + sorted(build.glob("gemma4_norms*.so"))
    if not found:
        _STATE["reason"] = "no build of norms.cu (run build_kernels.py)"
        return None
    try:
        import torch  # noqa: F401
        spec = importlib.util.spec_from_file_location("gemma4_norms", str(found[0].resolve()))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (ImportError, OSError) as error:
        _STATE["reason"] = f"build did not load: {error}"
        return None
    _STATE["module"] = module
    return module


def status():
    module = load()
    return {"gemma4_norms": "local build" if module is not None else "torch fallback",
            **({} if module is not None else {"gemma4_norms_reason": _STATE["reason"]})}
