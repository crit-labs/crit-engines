"""qwen35-model against transformers' Qwen3_5ForCausalLM on a small random checkpoint (BF16 weights, a random MTP
layer beside it), run by Core's loader over the engine's own plugins: fixed-text scores follow transformers'
log-probabilities, greedy continuations are transformers' choices, requests in flight together decode what each
decodes alone, and drafting greedy rounds (the MTP head, captured rounds) emit plain greedy decoding's tokens."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

PLUGINS = Path(__file__).resolve().parents[2]

try:
    import torch
    import transformers
    from safetensors.torch import save_file
    READY = (torch.cuda.is_available() and bool(os.environ.get("CRIT_CORE_STEP_LIBRARY"))
             and hasattr(transformers, "Qwen3_5ForCausalLM"))
except ImportError:
    READY = False

SELECTION = {"use": {"model": "qwen35-model", "placement": "device-placement", "projections": "quantized-projections",
                     "attention": "softmax-attention", "kv_cache": "attention-state", "sampler": "token-sampling",
                     "recurrence": "sequential-recurrence", "norms": "fused-norms", "embeddings": "memory-table-rows",
                     "drafter": "greedy-head-drafter", "verifier": "chain-verifier"},
             "options": {"qwen35-model": {"context": 1024, "kv_extra": 512, "prefill_chunk": 64, "width": 3,
                                          "kv": "bf16"},
                         "attention-state": {"round_rows": 8}}}


@unittest.skipUnless(READY, "a GPU engine: CUDA, transformers with Qwen3.5, safetensors and CRIT_CORE_STEP_LIBRARY")
class Qwen35ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from crit_core.engine import load
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        torch.manual_seed(0)
        config = transformers.Qwen3_5TextConfig(
            vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
            max_position_embeddings=4096, rms_norm_eps=1e-6, tie_word_embeddings=False,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25})
        model = transformers.Qwen3_5ForCausalLM(config)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("norm.weight") and "linear_attn" not in name:
                    parameter.normal_(0.0, 0.1)
                if name.endswith("A_log"):
                    parameter.copy_(torch.empty_like(parameter).uniform_(0.05, 1.0).log())
            model.lm_head.weight.mul_(30)
        model.to(torch.bfloat16)
        cls.model = root / "model"
        model.save_pretrained(cls.model, safe_serialization=True)
        H, I, hd = 128, 256, 32
        g = torch.Generator().manual_seed(1)
        rnd = lambda *shape: (torch.randn(*shape, generator=g) * 0.08).to(torch.bfloat16)  # noqa: E731
        mtp = {"mtp.fc.weight": rnd(H, 2 * H), "mtp.pre_fc_norm_embedding.weight": rnd(H),
               "mtp.pre_fc_norm_hidden.weight": rnd(H), "mtp.norm.weight": rnd(H),
               "mtp.layers.0.input_layernorm.weight": rnd(H), "mtp.layers.0.post_attention_layernorm.weight": rnd(H),
               "mtp.layers.0.self_attn.q_proj.weight": rnd(4 * hd * 2, H),
               "mtp.layers.0.self_attn.k_proj.weight": rnd(2 * hd, H),
               "mtp.layers.0.self_attn.v_proj.weight": rnd(2 * hd, H),
               "mtp.layers.0.self_attn.o_proj.weight": rnd(H, 4 * hd),
               "mtp.layers.0.self_attn.q_norm.weight": rnd(hd), "mtp.layers.0.self_attn.k_norm.weight": rnd(hd),
               "mtp.layers.0.mlp.gate_proj.weight": rnd(I, H), "mtp.layers.0.mlp.up_proj.weight": rnd(I, H),
               "mtp.layers.0.mlp.down_proj.weight": rnd(H, I)}
        save_file(mtp, str(cls.model / "model_mtp.safetensors"))
        (cls.model / "generation_config.json").write_text(json.dumps({"eos_token_id": [1]}), encoding="utf-8")
        cls.reference = model.float().cuda().eval()
        package = root / "package"
        shutil.copytree(PLUGINS, package / "plugins", ignore=shutil.ignore_patterns("__pycache__"))
        (package / "configuration.json").write_text(json.dumps({"plugins": SELECTION}), encoding="utf-8")
        cls.engine = load(package, cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()
        cls.temp.cleanup()

    def prompts(self, count, length=17):
        g = torch.Generator().manual_seed(count * 103 + length)
        return [torch.randint(3, 256, (length + 5 * i,), generator=g).tolist() for i in range(count)]

    def logits(self, ids):
        with torch.no_grad():
            return self.reference(torch.tensor([ids]).cuda()).logits[0].float()

    def test_scores_follow_transformers(self):
        for length in (40, 150):   # one chunk, then three (the state carried)
            (ids,) = self.prompts(1, length)
            scored = self.engine.score(ids, 1, "prefill")
            logp = self.logits(ids)[:len(ids) - 1].log_softmax(-1)
            expected = logp[torch.arange(len(ids) - 1), torch.tensor(ids[1:])].tolist()
            differences = [abs(a - b) for a, b in zip(scored.target, expected)]
            self.assertLess(sum(differences) / len(differences), 0.2, (length, differences))

    def test_greedy_continuations_are_transformers_choices(self):
        for prompt in self.prompts(3):
            tokens, _ = self.engine.generate(prompt, 12, stop_tokens=[], sampling={"temperature": 0.0}, plain=True)
            logits = self.logits(prompt + tokens)
            for i, token in enumerate(tokens):
                row = logits[len(prompt) - 1 + i]
                self.assertGreaterEqual(float(row[token]), float(row.max()) - 0.5, (i, token, int(row.argmax())))

    def test_drafted_rounds_emit_plain_greedy_tokens(self):
        for prompt in self.prompts(3, 30):
            plain, _ = self.engine.generate(prompt, 24, stop_tokens=[], sampling={"temperature": 0.0}, plain=True)
            drafted, result = self.engine.generate(prompt, 24, stop_tokens=[], sampling={"temperature": 0.0})
            self.assertEqual(drafted, plain)

    def test_requests_in_flight_together_decode_what_each_decodes_alone(self):
        prompts = self.prompts(4, 9)
        greedy = {"temperature": 0.0}
        alone = [self.engine.generate(p, 10, stop_tokens=[], sampling=greedy)[0] for p in prompts]
        with ThreadPoolExecutor(4) as pool:
            together = list(pool.map(lambda p: self.engine.generate(p, 10, stop_tokens=[], sampling=greedy)[0],
                                     prompts))
        self.assertEqual(together, alone)

    def test_sampled_requests_decode_in_rounds(self):
        (prompt,) = self.prompts(1, 20)
        tokens, _ = self.engine.generate(prompt, 16, stop_tokens=[], sampling={"temperature": 1.0, "top_p": 0.95,
                                                                              "top_k": 20, "seed": 7})
        again, _ = self.engine.generate(prompt, 16, stop_tokens=[], sampling={"temperature": 1.0, "top_p": 0.95,
                                                                             "top_k": 20, "seed": 7})
        self.assertEqual(len(tokens), 16)
        self.assertEqual(tokens, again)


if __name__ == "__main__":
    unittest.main()
