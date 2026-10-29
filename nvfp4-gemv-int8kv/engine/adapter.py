"""crit serve (chat-resident-v1) and crit eval (eval-v1) adapters for the Nanbeige4.3-33B NVFP4 engine, through
crit_core.native's stock adapters for Python engines."""
from crit_core.native import eval_engine, resident_engine

from .engine import open_engine

SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "seed")
DEFAULTS = {"prefill_chunk": 2048, "kv_cache": "int8", "prefill_gemm_attn": "fp16_acc", "prefill_gemm_mlp": "fp8", "prefill_gemm_down": "fp16_acc", "prefill_int8_qk": True}


def create_engine(context):
    return resident_engine(context, open_engine, DEFAULTS)


def create_eval_engine(context):
    return eval_engine(context, open_engine, DEFAULTS)


def declared_capabilities(model_path):
    from crit_core.serving import capabilities
    from crit_core.text import load_model_text
    return capabilities(load_model_text(model_path, clock=None), SAMPLING_FIELDS, logprobs=True, score=True)
