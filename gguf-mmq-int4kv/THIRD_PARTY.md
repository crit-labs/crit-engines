# Third-party components

| Component | Where | License | Use |
| --- | --- | --- | --- |
| crit library plugins (token-sampling, the base plugins device-placement, topk-router and torch-norms, and the sources of fused-norms and heterogeneous-expert-dispatch) | `engine/plugins/`, copied from crit by the sealed candidate | Apache-2.0 | Sampling, placement, the router and norm references, and the code the engine's own plugins import |
| crit Core, `crit_plugins` and crit's serving (with the package `tool_format` and `reasoning_format` this engine names) | from the crit checkout (not included) | Apache-2.0 | Core's runtime, captured decode rounds, the plugin loader, crit serve and the Eval adapter |
| ggml's IQ3_S codebook (`iq3s_grid`) and GGUF block layouts (ggml-common.h) | `engine/plugins/gguf-projections/code/iq3s_grid.h`, `iq3s_grid.py` and the layouts `gguf_blocks.py` and the CUDA decoders follow | MIT (Copyright (c) 2023-2026 The ggml authors) | Decoding the checkpoint's IQ3_S, IQ4_NL, IQ4_XS and Q8_0 blocks |
| PyTorch 2.14.1 (cu130), Triton 3.9.0 | the crit environment (not redistributed) | BSD-3-Clause, MIT | Tensors, CUDA graphs, torch extensions, Triton kernels |
| transformers 5.22.2, gguf-py | the crit environment (not redistributed); used only by the plugins' tests | Apache-2.0, MIT | The references the model and block-decoder tests compare against |
| NVIDIA CUDA runtime and toolkit 13.4 | not redistributed | NVIDIA CUDA Toolkit EULA | Kernel builds and launches, CUDA graphs |
| unsloth/gemma-4-26B-A4B-it-GGUF (UD-IQ4_XS) checkpoint, tokenizer and chat template | read from the user's local model folder (not redistributed) | Gemma Terms of Use (Google) | Weights, tokenizer, chat template |

The model plugin and the GGUF plugins, `engine/plugins/gemma4-model/`, `gemma4-attention/`, `gemma4-norms/`,
`gemma4-router/`, `gguf-projections/` and `gguf-experts/` (Gemma 4's decoder over the GGUF's stored blocks, its
tool-call and reasoning syntax, INT4 sliding rings and derived-key pages, split-KV decode and fused prompt attention
kernels, dp4a GEMVs and int8 tensor-core matmuls straight from the GGUF blocks, device-sorted expert routes, their
tests), were written for this engine in crit run `run-20261109T095346Z-1b210ccf`. Their tensor names follow
llama.cpp's gemma4 tensor map; no other inference engine's source, kernels or prepared artifacts were used beside the
ggml tables credited above.
