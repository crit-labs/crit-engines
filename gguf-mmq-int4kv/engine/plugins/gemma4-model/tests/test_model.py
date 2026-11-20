"""The Gemma 4 model plugin against transformers' Gemma4ForCausalLM on a tiny random checkpoint of the same layout
(sliding and full layers of their own head geometry, keys and values from one projection on full layers, proportional
RoPE, the dense MLP beside routed experts, soft-capped logits), run by Core's loader with the engine's own plugins:
fixed-text scores equal transformers' log-probabilities past the sliding window (a ring that wraps), greedy tokens are
transformers' choices, and requests in flight together decode what each decodes alone. Needs CUDA, transformers,
safetensors and Core's crit_core_steps (CRIT_CORE_STEP_LIBRARY)."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

try:
    import torch
    import transformers
    READY = torch.cuda.is_available() and bool(os.environ.get("CRIT_CORE_STEP_LIBRARY"))
except ImportError:
    READY = False

PLUGINS = Path(__file__).resolve().parents[2]
SELECTION = {"use": {"model": "gemma4-model", "placement": "device-placement", "projections": "gguf-projections",
                     "attention": "gemma4-attention", "kv_cache": "gemma4-attention", "sampler": "token-sampling",
                     "norms": "gemma4-norms", "router": "topk-router", "experts": "gguf-experts"},
             "base": {"norms": "torch-norms"},
             "options": {"topk-router": {"scoring": "softmax", "score_before_selection": True, "renormalize": True},
                         "gemma4-attention": {"dtype": "bf16", "page": 16, "block": 64},
                         # BF16 prompt rows: the layout check against transformers (FP8 prompts are Eval's to judge)
                         "gguf-projections": {"prompt": "bf16"}, "gguf-experts": {"prompt": "bf16"},
                         "gemma4-model": {"prefill_chunk": 16, "workspace_mib": 16, "reserve_mib": 0,
                                          "context": 1024}}}


def tiny_config():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    return Gemma4TextConfig(
        vocab_size=256, hidden_size=128, intermediate_size=96, moe_intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, global_head_dim=128, num_global_key_value_heads=1,
        attention_k_eq_v=True, layer_types=["sliding_attention", "sliding_attention", "full_attention",
                                            "sliding_attention"],
        sliding_window=12, num_experts=8, top_k_experts=2, enable_moe_block=True, hidden_size_per_layer_input=0,
        num_kv_shared_layers=0, final_logit_softcapping=30.0, max_position_embeddings=1024,
        rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                                            "rope_type": "proportional"},
                         "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
        hidden_activation="gelu_pytorch_tanh", rms_norm_eps=1e-6, tie_word_embeddings=True, bos_token_id=2,
        eos_token_id=1, pad_token_id=0)


@unittest.skipUnless(READY, "a GPU engine: CUDA, transformers, safetensors and CRIT_CORE_STEP_LIBRARY")
class Gemma4ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from crit_core.engine import load
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
        cls.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)   # a loaded .pyd stays locked on Windows
        root = Path(cls.temp.name)
        torch.manual_seed(0)
        model = Gemma4ForCausalLM(tiny_config())
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "q_norm" in name or "k_norm" in name:
                    # attention scale 1 over normalized heads: weights near dim^-1/4 keep the logits at the size a
                    # trained checkpoint's are, not the near-one-hot softmax random weights of 1 give
                    p.copy_(p.shape[0] ** -0.25 * (1.0 + 0.2 * torch.randn_like(p)))
                elif name.endswith("norm.weight") or name.endswith("layernorm.weight") or "norm_" in name:
                    p.copy_(1.0 + 0.2 * torch.randn_like(p))
                elif "per_expert_scale" in name or name.endswith("router.scale"):
                    p.copy_(1.0 + 0.1 * torch.randn_like(p))
            model.model.embed_tokens.weight.mul_(8)   # sharper logits
        cls.reference = model.to(torch.bfloat16).float().cuda().eval()   # BF16 weights, FP32 arithmetic: the engine's weights
        cls.model = root / "model"
        model.save_pretrained(cls.model, safe_serialization=True)
        # config.json as the published checkpoint writes it: Gemma4ForConditionalGeneration's, the text model's keys
        # (global_head_dim, num_global_key_value_heads) under text_config, not transformers' per_layer_config.
        text = json.loads((cls.model / "config.json").read_text(encoding="utf-8"))
        text.pop("per_layer_config", None)
        text.update(global_head_dim=128, num_global_key_value_heads=1, model_type="gemma4_text")
        config = {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4", "text_config": text,
                  "tie_word_embeddings": True, "eos_token_id": [1]}
        (cls.model / "config.json").write_text(json.dumps(config), encoding="utf-8")
        package = root / "package"
        shutil.copytree(PLUGINS, package / "plugins", ignore=shutil.ignore_patterns("__pycache__"))
        (package / "configuration.json").write_text(json.dumps({"plugins": SELECTION}), encoding="utf-8")
        cls.engine = load(package, cls.model, context_tokens=1024)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()
        cls.temp.cleanup()

    def prompts(self, count, length=20):
        g = torch.Generator().manual_seed(count * 101 + length)
        return [torch.randint(3, 256, (length + 5 * i,), generator=g).tolist() for i in range(count)]

    def logits(self, ids):
        with torch.no_grad():
            return self.reference(torch.tensor([ids]).cuda()).logits[0].float()

    def test_scores_equal_transformers_log_probabilities_past_the_window(self):
        (ids,) = self.prompts(1, 70)        # 70 tokens: several prompt chunks of 16, far past the 12-token window
        scored = self.engine.score(ids, 1, "prefill")
        logp = self.logits(ids).log_softmax(-1)
        expected = logp[torch.arange(len(ids) - 1), torch.tensor(ids[1:])].tolist()
        gaps = [abs(a - b) for a, b in zip(scored.target, expected)]
        # BF16 activations against FP32 on deliberately sharp logits (the embedding x 8, tied to the head: logits near
        # the soft-cap of 30, so BF16's 0.4% on the last hidden state moves a log-probability by 0.2-0.4): small
        # everywhere, no step at the window or a chunk boundary
        self.assertLess(sum(gaps) / len(gaps), 0.15, gaps)
        self.assertLess(max(gaps), 1.0, gaps)

    def test_greedy_continuations_are_transformers_choices(self):
        for prompt in self.prompts(2):
            tokens, _ = self.engine.generate(prompt, 24, stop_tokens=[], sampling={"temperature": 0.0})
            logits = self.logits(prompt + tokens)
            for i, token in enumerate(tokens):
                row = logits[len(prompt) - 1 + i]
                self.assertGreaterEqual(float(row[token]), float(row.max()) - 0.25, (i, token, int(row.argmax())))

    def test_requests_in_flight_together_decode_what_each_decodes_alone(self):
        prompts = self.prompts(4, 9)
        alone = [self.engine.generate(p, 10, stop_tokens=[], sampling={"temperature": 0.0})[0] for p in prompts]
        with ThreadPoolExecutor(4) as pool:
            together = list(pool.map(lambda p: self.engine.generate(p, 10, stop_tokens=[],
                                                                    sampling={"temperature": 0.0})[0], prompts))
        self.assertEqual(together, alone)


if __name__ == "__main__":
    unittest.main()
