"""The flash-linear-attention gated delta rule kernels this engine calls (fla 0.6.0, vendored)."""
from . import SITE  # noqa: F401  (puts the vendored packages on sys.path)
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel


def chunk(q, k, v, g, beta, scale, initial_state):
    """Chunked prefill: q, k [1, T, H, K], v [1, T, HV, V], g (log decay) and beta [1, T, HV]; returns (o, final)."""
    return chunk_gated_delta_rule(q, k, v, g=g, beta=beta, scale=scale, initial_state=initial_state,
                                  output_final_state=True, use_qk_l2norm_in_kernel=True)


def recurrent_inplace(q, k, v, a, b, A_log, dt_bias, state, o, scale):
    """T recurrent steps with the gate (-exp(A_log) * softplus(a + dt_bias)) and beta (sigmoid(b)) computed in the
    kernel, the state [1, HV, K, V] fp32 updated in place, outputs into o [1, T, HV, V]."""
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    BV = 8
    grid = (triton_cdiv(V, BV) * B * HV,)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q, k=k, v=v, g=a, gk=None, gv=None, beta=b, A_log=A_log, dt_bias=dt_bias, o=o, h0=state, ht=state,
        cu_seqlens=None, scale=scale, T=T, H=H, HV=HV, K=K, V=V, BK=K, BV=BV, IS_BETA_HEADWISE=True,
        USE_QK_L2NORM_IN_KERNEL=True, APPLY_BETA_SIGMOID=True, ALLOW_NEG_EIGVAL=False, STATE_V_FIRST=False,
        num_warps=1, num_stages=3)
    return o


def triton_cdiv(a, b):
    return (a + b - 1) // b
