"""crit entry points for the Qwen3.8-27B NVFP4 engine: chat-resident-v1 for crit serve (create_engine) and eval-v1 for
crit eval (create_eval_engine), both crit_core.native's stock adapters over the Python engine in q38/."""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from crit_core.native import eval_engine, resident_engine  # noqa: E402

from .q38.engine import open_engine  # noqa: E402

DEFAULTS = {}
SAMPLING = ("temperature", "top_p", "top_k", "min_p", "seed", "presence_penalty", "frequency_penalty",
            "repetition_penalty")


def create_engine(context):
    return resident_engine(context, open_engine, defaults=DEFAULTS)


def create_eval_engine(context):
    return eval_engine(context, open_engine, defaults=DEFAULTS)


def declared_capabilities(model_path):
    from pathlib import Path
    from crit_core.serving import capabilities
    from crit_core.text import load_model_text
    text = load_model_text(Path(model_path), clock=None)
    return capabilities(text, SAMPLING, logprobs=True, score=True, plain=True)
