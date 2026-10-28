"""crit serve (chat-resident-v1) and crit eval (eval-v1) adapters for the Qwen3.8-27B NVFP4 engine: crit_core.native's
stock adapters over the Python engine in qwen38/."""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from crit_core.native import eval_engine, resident_engine  # noqa: E402
from crit_core.serving import capabilities  # noqa: E402

DEFAULTS = {}


def open_engine(configuration):
    from .qwen38.engine import Engine
    return Engine(configuration)


def create_engine(context):
    return resident_engine(context, open_engine, DEFAULTS)


def create_eval_engine(context):
    return eval_engine(context, open_engine, DEFAULTS)


def declared_capabilities(model_path):
    from crit_core.text import load_model_text
    text = load_model_text(model_path, clock=None)
    return capabilities(text, ("temperature", "top_p", "top_k", "min_p", "seed"), logprobs=True, score=True, plain=True)
