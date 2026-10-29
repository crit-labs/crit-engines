# Third-party components

| Component | Where | License | Use |
| --- | --- | --- | --- |
| SGLang 0.5.21 DFlash2 drafter (`sglang/srt/models/dflash.py`, `sglang/srt/speculative/dflash_worker_v2.py`) | `engine/drafter.py`, ported to the engine's kernels; license in `LICENSES.txt` | Apache-2.0 | The optional DFlash2 block drafter and its candidate selector. The published configuration loads no drafter (no `draft_model`), so the measured engine does not run this code |
| crit Python packages (`crit_core.native`, `crit_core.serving`, `crit_core.text`, crit Eval, crit serve) | from the crit checkout (not included) | Apache-2.0 | The stock Eval and serve adapters, engine interface, tokenizer and chat template handling |
| PyTorch 2.14.1 (cu130), Triton 3.8.0 and safetensors | the crit environment (not redistributed) | BSD-3-Clause, MIT, Apache-2.0 | Tensors, CUDA graphs, cuBLAS/cuBLASLt prompt GEMMs (FP16 accumulation, row-wise FP8 `torch._scaled_mm`) and the BF16 output head, Triton for the engine's own kernels, reading the checkpoint |
| NVIDIA CUDA runtime, cuBLAS and cuBLASLt | CUDA toolkit 13.1 (not redistributed here) | NVIDIA CUDA Toolkit EULA | Kernel launches, CUDA graphs, prompt GEMMs; `nvcc` builds `nvfp4_gemv.cu` |
| webbrain-one/Nanbeige4.3-33B-NVFP4 checkpoint (NVFP4 of Nanbeige/Nanbeige4.3-33B), tokenizer and chat template | read from the user's local model folder (not redistributed) | Apache-2.0 | Weights, tokenizer, chat template |

Everything else (the CUDA NVFP4 GEMV in `nvfp4_gemv.cu`, the Triton NVFP4 GEMV and decoders, norm, RoPE and
INT8 KV cache, split-K decode attention, INT8 Q.K^T prompt attention after SageAttention's published scheme,
short-prompt GEMM and FP8 row quantizer kernels, and the device sampling tail) was written for this engine in crit
run `run-20261028T182458Z-ceca50f1`. No other inference engine's source, kernels or prepared artifacts were used,
apart from the SGLang drafter port above.
