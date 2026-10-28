# Qwen3.8-27B NVFP4 engine (crit run run-20261028T140102Z-fba244c2)

A Python engine with crit_core.native's engine interface, served by `crit serve` (`adapter.py: create_engine`) and
measured by crit Eval (`adapter.py: create_eval_engine`).

| Part | What it is | Provider |
| --- | --- | --- |
| `q38/weights.py` | Loads the checkpoint as stored (FP8 per-channel, NVFP4 E2M1 + E4M3 group scales); row concatenation of projections that read one input; embedding in pinned host memory | own |
| `q38/kernels.py` | Triton: GEMVs on the stored encodings (column-invariant, deterministic split-K), NVFP4 to exact BF16 dequantization, fused norms, attention prep (q/k norm, partial RoPE, FP8 KV write), split-key flash decoding over the FP8 cache, conv update, gated norm, host-mapped embedding gather | own (Triton) |
| `q38/model.py` | Chunked prefill (BF16 GEMMs over exact BF16 weights, fla chunked gated delta rule, SDPA flash attention) and the graph-capturable decode step | own, torch, fla |
| `q38/sampler.py`, `native/` | Core's sampling law, logprobs and commit on the device: crit Inference's token-sampling entry (`sampling.cu`, `sampling.h`, crit headers) behind a small C ABI | crit Inference token-sampling |
| `q38/vendor/site` | flash-linear-attention 0.6.0 and einops 0.8.2 (MIT, licenses in their dist-info folders) | fla-org, einops |

Build: `bash native/build.sh` (nvcc 13.1, sm_120). Runtime: the crit venv's torch 2.14.1+cu130 and triton 3.8.0.
