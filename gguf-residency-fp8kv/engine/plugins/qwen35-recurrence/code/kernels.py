"""qwen35-recurrence's CUDA kernels (gdn.cu), built by build_kernels.py into build/ beside this file. load() returns the
module, or None where no build is there (the plugin then runs its torch reference); status() says which."""
import importlib.util
import os
from pathlib import Path

_STATE = {"module": None, "tried": False, "reason": None}


def load():
    if _STATE["tried"]:
        return _STATE["module"]
    _STATE["tried"] = True
    if os.environ.get("QWEN35_GDN_KERNELS", "1") == "0":
        _STATE["reason"] = "turned off (QWEN35_GDN_KERNELS=0)"
        return None
    build = Path(__file__).resolve().parent / "build"
    found = sorted(build.glob("gdn*.pyd")) + sorted(build.glob("gdn*.so"))
    if not found:
        _STATE["reason"] = "no build of gdn.cu (run build_kernels.py)"
        return None
    try:
        import torch  # noqa: F401  (its DLLs first)
        spec = importlib.util.spec_from_file_location("gdn", str(found[0].resolve()))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (ImportError, OSError) as error:
        _STATE["reason"] = f"build did not load: {error}"
        return None
    _STATE["module"] = module
    return module


def status():
    module = load()
    return {"gdn": "local build" if module is not None else "torch reference",
            **({} if module is not None else {"gdn_reason": _STATE["reason"]})}
