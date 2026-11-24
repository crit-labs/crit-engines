"""qwen35-router: topk-router (topk.py, its rule and torch form) with the routing Qwen3.5's step makes in one CUDA
kernel: route_fused(r, weight, top, expert_scale) computes the router logits from the BF16 router inputs and matrix,
the softmax over all experts, the top experts (ties to the lower id, NaN below every number), their renormalized and
scaled weights and each chosen expert's own scale, one launch for all tokens. Rules the kernel does not take (sigmoid
scores, selection before scoring, a correction bias, group limits, no renormalization) return None, and the model
then routes through select() over logits as with topk-router."""
from .gguf import kernels
from .topk import TopKRouter


class Qwen35Router(TopKRouter):
    def __init__(self, options):
        super().__init__({**options, "kernel": False})
        self.kernels = kernels.load()

    def fused_rule(self):
        rule = self.routing
        return (self.kernels is not None and hasattr(self.kernels, "route") and rule.scoring == "softmax"
                and rule.score_before_selection and rule.renormalize and not rule.correction_bias
                and rule.selected_groups >= rule.groups)

    def route_fused(self, r, weight, top, expert_scale):
        """(ids [tokens][top] int64, weights [tokens][top] FP32) in one kernel, or None where it does not apply."""
        import torch
        if (not self.fused_rule() or not r.is_cuda or weight is None or weight.dtype != torch.bfloat16
                or weight.shape[0] > 256 or top > 16 or top > weight.shape[0]):
            return None
        rule = self.routing
        rule.validate(weight.shape[0], top)
        return tuple(self.kernels.route(r.to(torch.bfloat16), weight, top, expert_scale.float(),
                                        float(rule.renormalize_epsilon), float(rule.renormalize_floor),
                                        float(rule.scaling)))

    def select(self, logits, top, bias=None):
        """select() of FP32 logits in one kernel where the rule takes it (softmax before selection, renormalized,
        no bias or groups): the same choice as topk-router's torch form (ties to the lower id, NaN last)."""
        import torch
        if (bias is None and self.fused_rule() and hasattr(self.kernels, "route_select") and logits.is_cuda
                and logits.dtype == torch.float32 and logits.shape[1] <= 256 and top <= 16):
            rule = self.routing
            rule.validate(logits.shape[1], top)
            ids, weights = self.kernels.route_select(logits, top, float(rule.renormalize_epsilon),
                                                     float(rule.renormalize_floor), float(rule.scaling))
            return ids, weights
        return super().select(logits, top, bias)

    def facts(self):
        return {**kernels.status(), "router_kernel": "qwen35-router route (gguf_gemv.cu)" if self.fused_rule()
                else "topk-router torch form"}


def create(context, options, parts):
    return Qwen35Router(options)
