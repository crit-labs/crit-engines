# Third-party components

| Component | Where | License | Use |
| --- | --- | --- | --- |
| crit Inference `token-sampling` and crit Core headers | `engine/native/sampling.cu`, `sampling.h` and `engine/native/include/crit/`, copied from crit | Apache-2.0 | Core's sampling law, log-probabilities and commit on the device, behind a small C ABI (`sampling_capi.cu`) |
| flash-linear-attention 0.6.0 (`fla`, with fla-core) | `engine/q38/vendor/site/fla/`, vendored unchanged; licenses in its `dist-info` folders | MIT | Chunked gated-delta prefill and the recurrence for decode, verify and commit |
| einops 0.8.2 | `engine/q38/vendor/site/einops/`, vendored unchanged; license in its `dist-info` folder | MIT | Tensor reshapes inside flash-linear-attention |
| crit Python packages (`crit_core.native`, `crit_core.serving`, crit Eval, crit serve) | from the crit checkout (not included) | Apache-2.0 | The stock Eval and serve adapters, engine interface, tokenizer and chat template handling |
| PyTorch 2.14.1 (cu130) and Triton 3.8.0 | the crit environment (not redistributed) | BSD-3-Clause, MIT | Tensors, CUDA graphs, cuBLAS BF16 and cuBLASLt FP8 GEMMs, flash attention for prompts, Triton for the engine's own kernels |
| NVIDIA CUDA runtime, cuBLAS and cuBLASLt | CUDA toolkit 13.1 (not redistributed here) | NVIDIA CUDA Toolkit EULA | Kernel launches, CUDA graphs, prompt GEMMs; `nvcc` builds `native/` |
| unsloth/Qwen3.8-27B-NVFP4 checkpoint, tokenizer and chat template | read from the user's local model folder (not redistributed) | Apache-2.0 | Weights, bundled MTP layer, tokenizer, chat template |

Everything else (the Triton GEMVs on the stored FP8 and NVFP4 weights, the stored-weight prefill GEMM, attention,
convolution and norm kernels, the speculative rounds with adaptive draft width and the drafter's FP8
re-encoding) was written for this engine in crit run `run-20261028T140102Z-fba244c2`. No other inference engine's
source, kernels or prepared artifacts were used.
