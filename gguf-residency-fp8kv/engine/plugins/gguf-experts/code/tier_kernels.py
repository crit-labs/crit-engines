"""gguf-experts' CPU expert tier (expert_tier.cu), built by build_kernels.py into build/ beside this file. load() returns the
module, or None where no build is there (the plugin then runs its torch reference); status() says which."""
import importlib.util
import os
from pathlib import Path

_STATE = {"module": None, "tried": False, "reason": None}


def load():
    if _STATE["tried"]:
        return _STATE["module"]
    _STATE["tried"] = True
    if os.environ.get("GGUF_EXPERT_TIER", "1") == "0":
        _STATE["reason"] = "turned off (GGUF_EXPERT_TIER=0)"
        return None
    build = Path(__file__).resolve().parent / "build"
    found = sorted(build.glob("expert_tier*.pyd")) + sorted(build.glob("expert_tier*.so"))
    if not found:
        _STATE["reason"] = "no build of expert_tier.cu (run build_kernels.py)"
        return None
    try:
        import torch  # noqa: F401  (its DLLs first)
        spec = importlib.util.spec_from_file_location("expert_tier", str(found[0].resolve()))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (ImportError, OSError) as error:
        _STATE["reason"] = f"build did not load: {error}"
        return None
    _STATE["module"] = module
    return module


def status():
    module = load()
    return {"expert_tier": "local build" if module is not None else "torch reference",
            **({} if module is not None else {"expert_tier_reason": _STATE["reason"]})}
