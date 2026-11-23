# Third-party components

| Component | Where | License | Use |
| --- | --- | --- | --- |
| crit library plugins (inkling2-loop-model and the models core model-template with its techniques, quantized-projections, grouped-experts, topk-router, expert-tensors, attention-state, softmax-attention, sequential-recurrence, fused-norms, token-sampling, greedy-head-drafter, memory-table-rows, device-placement and the base plugins) | `engine/plugins/CATEGORY/ID` (code they import under `engine/plugins/_sources/`), copied unchanged from crit's library by `crit plugins add` | Apache-2.0 | Every slot of the engine: the looped Inkling-2 model, projections and experts on the stored weights, routing, KV pages, attention, convolutions, norms, sampling and the block drafter |
| crit Core, `crit_plugins` and crit's kernels (mma-rows, decode-gemv, pinned-gemm, grouped-experts' FP4 kernels, paged-kv-attention, fused-norms) | from the crit checkout (not included); the kernels compile at load | Apache-2.0 | Core's runtime, captured rounds, plugin loader, serving and Eval adapters; the GPU kernels the plugins call |
| FlashInfer 0.7.3 | the crit environment (not redistributed); imported only by library-plugin alternatives this configuration does not select | Apache-2.0 | Not selected |
| PyTorch 2.14.1 (cu130), Triton 3.8.0, safetensors 0.8.0 | the crit environment (not redistributed) | BSD-3-Clause, MIT, Apache-2.0 | Tensors, CUDA graphs, Triton kernels, reading the checkpoint |
| transformers | the crit environment (not redistributed); used only by the model plugin's tests | Apache-2.0 | The reference the model plugin's tests compare against |
| NVIDIA CUDA runtime and toolkit 13.1 | not redistributed | NVIDIA CUDA Toolkit EULA | Kernel builds and launches, CUDA graphs |
| thinkingmachines/Inkling-2-Mini checkpoint, its MTP block drafter (`mtp.safetensors`), tokenizer and chat template | read from the user's local model folder (not redistributed) | Apache-2.0 | Weights, block drafter, tokenizer, chat template |

Every plugin is a copy of crit's library; the configuration (`engine/configuration.json`) and kernel tuning tables (`engine/tuning/`) were made for this engine in crit run `run-20261123T103857Z-5e20e5eb`. No other inference engine's source, kernels or prepared artifacts were used.
