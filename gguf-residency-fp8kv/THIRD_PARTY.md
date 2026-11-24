# Third-party components

| Component | Where | License | Use |
| --- | --- | --- | --- |
| crit library plugins (token-sampling, the base plugins device-placement, topk-router and torch-norms, and the sources of fused-norms and heterogeneous-expert-dispatch) | `engine/plugins/`, copied from crit by the sealed candidate | Apache-2.0 | Sampling, placement, the router and norm references, and the code the engine's own plugins import |
| crit Core, `crit_plugins` and crit's serving | from the crit installation (not included) | Apache-2.0 | Core's runtime, captured decode rounds, the plugin loader, crit serve and the Eval adapter |
| ggml's IQ3_S codebook (`iq3s_grid`) and GGUF block layouts (ggml-common.h) | `engine/plugins/gguf-projections/code/iq3s_grid.h`, `iq3s_grid.py`, `engine/plugins/gguf-experts/code/iq3s_grid_host.h` and the layouts `gguf_blocks.py`, the CUDA decoders and the CPU tier follow | MIT (Copyright (c) 2023-2026 The ggml authors) | Decoding the checkpoint's IQ3_S, IQ4_XS, Q6_K and Q8_0 blocks |
| PyTorch 2.14.1 (cu130) | the crit environment (not redistributed) | BSD-3-Clause | Tensors, CUDA graphs, torch extensions |
| transformers 5.22.2, gguf-py | the crit environment (not redistributed); used only by the plugins' tests | Apache-2.0, MIT | The references the model and block-decoder tests compare against |
| NVIDIA CUDA runtime and toolkit 13.4 | not redistributed | NVIDIA CUDA Toolkit EULA | Kernel builds and launches, CUDA graphs |
| unsloth/Qwen3.6-35B-A3B-GGUF (UD-IQ4_XS) checkpoint, its tokenizer and chat template | read from the user's local model folder (not redistributed) | Apache-2.0 (Qwen3.6) | Weights, tokenizer, chat template |

The model plugin and the GGUF plugins, `engine/plugins/qwen35-model/`, `qwen35-attention/`, `qwen35-recurrence/`,
`qwen35-norms/`, `qwen35-router/`, `gguf-projections/` and `gguf-experts/` (Qwen3.6's hybrid decoder over the GGUF's
stored blocks: Gated DeltaNet recurrence and FP8 paged full attention kernels, dp4a GEMVs and int8 tensor-core matmuls
straight from the GGUF blocks, routed experts in device slots and pinned host memory with an AVX2 CPU tier and an
adaptive residency, their tests), were written for this engine in crit run `run-20261111T081913Z-40ce07fd`. Their
tensor names follow llama.cpp's qwen35moe tensor map; no other inference engine's source, kernels or prepared
artifacts were used beside the ggml tables credited above.
