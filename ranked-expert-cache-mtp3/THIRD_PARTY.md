# Third-party components

| Component | Where | License | Use |
| --- | --- | --- | --- |
| crit library plugins (qwen4-exp-model and the models core model-template with its techniques, expert-offload, host-expert-tier, expert-tensors, experts-reference, topk-router, quantized-projections, projections-dense, partitioned-attention, softmax-attention, attention-state, sequential-recurrence, recurrent-state-recovery, fused-norms, torch-norms, token-constraints, token-sampling, greedy-head-drafter, chain-verifier, mtp-decode, lookup-table-rows, memory-table-rows, device-placement) | `engine/plugins/CATEGORY/ID` (code they import under `engine/plugins/_sources/`), copied from crit 0.23.0's library by `crit plugins add`; two carry fixes made in this run (below) | Apache-2.0 | Every slot of the engine: the Qwen3.8-Flash-Next model and its MTP layer, routed experts in host memory behind an elastic device cache, projections on the stored weights, KV pages, attention, recurrence, norms, sampling, drafting and verification |
| crit Core, `crit_plugins` and crit's kernels (mma-rows, grouped-experts, expert-directory, fused-norms, paged-kv-attention, recurrent-scans, fp8-block-rows, sparse-indexer, token-sampling, topk-routing) | from the crit checkout (not included); the kernels compile at load | Apache-2.0 | Core's runtime, captured rounds, plugin loader, serving and Eval adapters; the GPU kernels the plugins call |
| FlashInfer 0.7.3 | the crit environment (not redistributed); imported only by library-plugin alternatives this configuration does not select | Apache-2.0 | Not selected |
| PyTorch 2.14.1 (cu130), Triton 3.8.0, safetensors 0.8.0 | the crit environment (not redistributed) | BSD-3-Clause, MIT, Apache-2.0 | Tensors, CUDA graphs, Triton kernels, reading the checkpoint |
| transformers 5.20.2 | the crit environment (not redistributed); used only by the plugins' tests | Apache-2.0 | The reference the model plugin's tests compare against |
| NVIDIA CUDA runtime and toolkit 13.1 | not redistributed | NVIDIA CUDA Toolkit EULA | Kernel builds and launches, CUDA graphs |
| nvidia/Qwen3.8-Flash-Next-FP4-Local checkpoint (its MTP layer and n-gram table included), tokenizer and chat template | read from the user's local model folder (not redistributed) | NVIDIA Open Model Agreement; the base model Qwen3.8-Flash-Next under the Qwen Community License 1.0 | Weights, draft layer, n-gram and embedding tables, tokenizer, chat template |

Changes to crit 0.23.0's library copies, made in crit run `run-20261123T155751Z-837036fd` with a test each:

| Plugin | File | Change |
| --- | --- | --- |
| token-constraints | `engine/plugins/sampling/token-constraints/plugin.py` | its captured rounds compute in the engine's scratch, which Core closes with the engine |
| recurrent-state-recovery | `code/device_states.py` (in `engine/plugins/recurrence/` and `engine/plugins/_sources/recurrence/`) | closing a sequence keeps another sequence's pending prompt-row units backed until their accept |

The configuration (`engine/configuration.json`), the expert cache's residency profile (`engine/residency_profile.json`) and the kernel tuning tables (`engine/tuning/`) were made for this engine in the same run. No other inference engine's source, kernels or prepared artifacts were used.
