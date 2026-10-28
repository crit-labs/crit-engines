# Third-party components

| Component | Where | License | Use |
| --- | --- | --- | --- |
| flash-linear-attention 0.6.0 (`fla`, with fla-core) | `engine/qwen38/vendor/fla/`, vendored unchanged; licenses in `engine/qwen38/vendor/NOTICES/` | MIT | Chunked gated-delta prefill and the fused recurrence for decode, verify and commit |
| einops | `engine/qwen38/vendor/einops/`, vendored unchanged; license in `engine/qwen38/vendor/NOTICES/` | MIT | Tensor reshapes inside flash-linear-attention |
| crit Python packages (`crit_core.native`, `crit_core.serving`, `crit_core.text`, crit Eval, crit serve) | from the crit checkout (not included) | Apache-2.0 | The stock Eval and serve adapters, engine interface, tokenizer and chat template handling |
| PyTorch 2.14.1 (cu130) and Triton 3.8.0 | the crit environment (not redistributed) | BSD-3-Clause, MIT | Tensors, CUDA graphs, cuBLASLt through `torch._scaled_mm` for the prompt's NVFP4 and FP8 GEMMs, `torch.utils.cpp_extension` for the CUDA kernel, Triton for the engine's own kernels |
| NVIDIA CUDA runtime and cuBLASLt | CUDA toolkit 13 (not redistributed here) | NVIDIA CUDA Toolkit EULA | Kernel launches, CUDA graphs, prompt GEMMs |
| unsloth/Qwen3.8-27B-NVFP4 checkpoint, tokenizer and chat template | read from the user's local model folder (not redistributed) | Apache-2.0 | Weights, bundled MTP layer, tokenizer, chat template |

Everything else (the CUDA tensor-core NVFP4 GEMV in `qwen38/csrc/nvfp4_mma.cu`, the Triton FP8 and bf16 GEMVs,
attention, quantization and step kernels, the device sampler, the speculative rounds and the drafter's NVFP4
re-encoding) was written for this engine in crit run `run-20261028T062813Z-708a915c`. No other inference engine's
source, kernels or prepared artifacts were used.
