"""The engine's own CUDA kernels (gguf_gemv.cu), built by build_kernels.py into build/ beside this file. load() returns
the module, or None where no build is there (the plugins then decode in torch); status() says which, for the facts."""
import importlib.util
import os
from pathlib import Path

KINDS = ("Q8_0", "IQ4_NL", "IQ4_XS", "IQ3_S", "Q8_0R")
_STATE = {"module": None, "tried": False, "reason": None}


def load():
    if _STATE["tried"]:
        return _STATE["module"]
    _STATE["tried"] = True
    if os.environ.get("GEMMA4_GGUF_KERNELS", "1") == "0":
        _STATE["reason"] = "turned off (GEMMA4_GGUF_KERNELS=0)"
        return None
    build = Path(__file__).resolve().parent / "build"
    found = sorted(build.glob("gguf_gemv*.pyd")) + sorted(build.glob("gguf_gemv*.so"))
    if not found:
        _STATE["reason"] = "no build of gguf_gemv.cu (run build_kernels.py)"
        return None
    try:
        import torch  # noqa: F401  (its DLLs first)
        spec = importlib.util.spec_from_file_location("gguf_gemv", str(found[0].resolve()))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (ImportError, OSError) as error:
        _STATE["reason"] = f"build did not load: {error}"
        return None
    _STATE["module"] = module
    return module


def status():
    module = load()
    return {"gguf_gemv": "local build" if module is not None else "torch fallback",
            **({} if module is not None else {"gguf_gemv_reason": _STATE["reason"]})}
