# Third-party components

| Component | Where | License | Use |
| --- | --- | --- | --- |
| crit library plugins (attention-state, quantized-projections, fused-norms, token-sampling, mtp-decode, lookup-table-rows, recurrent-state-recovery and the base plugins) | `engine/plugins/`, copied unchanged from crit by the sealed candidate | Apache-2.0 | Every slot of the engine but the model: projections on the stored weights, KV pages, attention, recurrence, norms, sampling, drafting and verification |
| crit Core, `crit_plugins` and crit's kernels (mma-rows, decode-gemv, pinned-gemm, recurrent-scans, fused-norms, paged-kv-attention) | from the crit checkout (not included); the kernels compile at load | Apache-2.0 | Core's runtime, captured rounds, plugin loader, serving and Eval adapters; the GPU kernels the plugins call |
| PyTorch 2.14.1 (cu130), Triton 3.8.0, safetensors 0.8.0 | the crit environment (not redistributed) | BSD-3-Clause, MIT, Apache-2.0 | Tensors, CUDA graphs, Triton kernels, reading the checkpoint |
| transformers 5.22.2 | the crit environment (not redistributed); used only by the model plugin's tests | Apache-2.0 | The reference the model plugin's tests compare against |
| NVIDIA CUDA runtime and toolkit 13.1 | not redistributed | NVIDIA CUDA Toolkit EULA | Kernel builds and launches, CUDA graphs |
| unsloth/Qwen3.8-27B-NVFP4 checkpoint, tokenizer and chat template | read from the user's local model folder (not redistributed) | Apache-2.0 | Weights, bundled MTP layer, tokenizer, chat template |

The model plugin, `engine/plugins/qwen35-model/` (the Qwen3.8 hybrid architecture with 16-bit activations on the stored weights, the MTP layer as the device draft head, captured prompt chunks, cached-memory release, its kernel tuning tables in `tuning/`, tests against transformers), was written for this engine in crit run `run-20261112T093006Z-3b651e45`. No other inference engine's source,
kernels or prepared artifacts were used.
