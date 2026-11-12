"""The qwen35-model plugin against transformers' Qwen3_5ForCausalLM on a small random checkpoint (BF16 weights, the
reference in FP32): Core's loader runs it over this engine's plugins (the package's own selection, with BF16 KV pages
and the embedding table on the device for the small model), and its scores, greedy choices, round decoding and batched
requests follow the reference and each other."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUGINS = HERE.parents[1]   # the engine's plugins/ folder
PACKAGE = PLUGINS.parent

try:
    import torch
    import transformers
    import safetensors  # noqa: F401
    READY = torch.cuda.is_available() and hasattr(transformers, "Qwen3_5ForCausalLM")
except ImportError:
    READY = False


def _library():
    found = os.environ.get("CRIT_CORE_STEP_LIBRARY")
    if found:
        return found
    try:
        import crit_core
        candidate = Path(crit_core.__file__).resolve().parents[2] / "build" / "core-steps" / "libcrit_core_steps.so"
        return str(candidate) if candidate.is_file() else None
    except ImportError:
        return None


@unittest.skipUnless(READY and _library(), "a GPU, transformers with Qwen3_5ForCausalLM and Core's step library")
class Qwen35ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("CRIT_CORE_STEP_LIBRARY", _library())
        from crit_core.engine import load
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        torch.manual_seed(0)
        config = transformers.Qwen3_5TextConfig(
            vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
            layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention"],
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
            max_position_embeddings=2048, rms_norm_eps=1e-6, tie_word_embeddings=False,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                             "mrope_section": [2, 1, 1], "mrope_interleaved": True})
        model = transformers.Qwen3_5ForCausalLM(config)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("norm.weight") and "linear_attn" not in name:
                    parameter.normal_(0.0, 0.1)   # zero-centred norms scale by 1 + weight: not all ones
                if name.endswith("A_log"):
                    parameter.copy_(torch.empty_like(parameter).uniform_(0.05, 1.0).log())   # milder decays
            model.lm_head.weight.mul_(30)   # sharp logits: a greedy choice no rounding flips
        model.to(torch.bfloat16)
        cls.model = root / "model"
        model.save_pretrained(cls.model, safe_serialization=True)
        (cls.model / "generation_config.json").write_text(json.dumps({"eos_token_id": 0}), encoding="utf-8")
        cls.write_mtp(cls.model, config)
        cls.reference = model.float().cuda().eval()
        package = root / "package"
        shutil.copytree(PLUGINS, package / "plugins")
        selection = json.loads((PACKAGE / "configuration.json").read_text(encoding="utf-8"))["plugins"]
        options = selection.setdefault("options", {})
        options["qwen35-model"] = {**options.get("qwen35-model", {}), "kv": "bf16", "concurrency": 4,
                                   "prefill_chunk": 64, "kv_margin": 512, "activations_mib": 64}
        options["memory-table-rows"] = {"host": False}
        options["device-placement"] = {"off_device": ""}
        (package / "configuration.json").write_text(json.dumps({"plugins": selection}), encoding="utf-8")
        cls.engine = load(package, cls.model, context_tokens=1024)

    @staticmethod
    def write_mtp(folder, config):
        """A random MTP layer in the checkpoint's layout (model_mtp.safetensors), so the engine drafts from it."""
        from safetensors.torch import save_file
        g = torch.Generator().manual_seed(11)
        h, heads, kv, d, inter = config.hidden_size, config.num_attention_heads, config.num_key_value_heads,             config.head_dim, config.intermediate_size

        def w(*shape, std=0.05):
            return (torch.randn(*shape, generator=g) * std).to(torch.bfloat16)

        m = "mtp.layers.0."
        tensors = {"mtp.fc.weight": w(h, 2 * h), "mtp.norm.weight": w(h, std=0.1),
                   "mtp.pre_fc_norm_embedding.weight": w(h, std=0.1), "mtp.pre_fc_norm_hidden.weight": w(h, std=0.1),
                   m + "input_layernorm.weight": w(h, std=0.1), m + "post_attention_layernorm.weight": w(h, std=0.1),
                   m + "self_attn.q_proj.weight": w(heads * d * 2, h), m + "self_attn.k_proj.weight": w(kv * d, h),
                   m + "self_attn.v_proj.weight": w(kv * d, h), m + "self_attn.o_proj.weight": w(h, heads * d),
                   m + "self_attn.q_norm.weight": w(d, std=0.1), m + "self_attn.k_norm.weight": w(d, std=0.1),
                   m + "mlp.gate_proj.weight": w(inter, h), m + "mlp.up_proj.weight": w(inter, h),
                   m + "mlp.down_proj.weight": w(h, inter)}
        save_file(tensors, str(folder / "model_mtp.safetensors"))

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

    def test_scores_follow_transformers_log_probabilities(self):
        (ids,) = self.prompts(1, 150)   # past two prefill chunks: later chunks read the state and KV kept before
        scored = self.engine.score(ids, 1, "prefill", top_k=1)
        logp = self.logits(ids)[:len(ids) - 1].log_softmax(-1)
        expected = logp[torch.arange(len(ids) - 1), torch.tensor(ids[1:])].tolist()
        differences = [abs(a - b) for a, b in zip(scored.target, expected)]
        self.assertLess(sum(differences) / len(differences), 0.15, differences)
        best, token = logp.max(-1)
        for i, ((found, value),) in enumerate(scored.top):
            if found != int(token[i]):   # only a near-tie may rank them the other way
                self.assertLess(float(best[i]) - float(logp[i].topk(2).values[1]), 0.5, (i, found, int(token[i])))
                continue
            self.assertLess(abs(value - float(best[i])), 0.5, (i, value, float(best[i])))

    def test_greedy_continuations_are_transformers_choices(self):
        for prompt in self.prompts(3):
            tokens, _ = self.engine.generate(prompt, 12, stop_tokens=[], sampling={"temperature": 0.0})
            self.assertEqual(len(tokens), 12)
            logits = self.logits(prompt + tokens)
            for i, token in enumerate(tokens):
                row = logits[len(prompt) - 1 + i]
                self.assertGreaterEqual(float(row[token]), float(row.max()) - 0.5, (i, token, int(row.argmax())))

    def test_a_served_token_scores_as_it_was_served(self):
        # preserve's honesty: a reply's log-probabilities from the decode rows equal the same reply scored as text
        (prompt,) = self.prompts(1, 40)
        tokens, result = self.engine.generate(prompt, 10, stop_tokens=[], sampling={"temperature": 0.0}, logprobs=0)
        served = [entry[0] if isinstance(entry, (list, tuple)) else entry for entry in result.get("logprobs") or []]
        scored = self.engine.score(prompt + tokens, len(prompt), "prefill").target
        if served:
            for a, b in zip(served, scored):
                self.assertLess(abs(a - b), 1e-3, (served, scored))

    def test_requests_in_flight_together_decode_what_each_decodes_alone(self):
        prompts = self.prompts(6, 9)
        greedy = {"temperature": 0.0}
        alone = [self.engine.generate(p, 10, stop_tokens=[], sampling=greedy)[0] for p in prompts]
        with ThreadPoolExecutor(6) as pool:
            together = list(pool.map(lambda p: self.engine.generate(p, 10, stop_tokens=[], sampling=greedy)[0],
                                     prompts))
        self.assertEqual(together, alone)

    def test_drafted_greedy_equals_plain_decoding(self):
        if not self.engine.info().get("speculative"):
            self.skipTest("the selection does not draft")
        for prompt in self.prompts(3, 30):
            drafted = self.engine.generate(prompt, 24, stop_tokens=[], sampling={"temperature": 0.0})[0]
            plain = self.engine.generate(prompt, 24, stop_tokens=[], sampling={"temperature": 0.0}, plain=True)[0]
            self.assertEqual(drafted, plain)

    def test_sampled_requests_repeat_with_their_seed(self):
        (prompt,) = self.prompts(1, 20)
        sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 7}
        first = self.engine.generate(prompt, 16, stop_tokens=[], sampling=sampling)[0]
        second = self.engine.generate(prompt, 16, stop_tokens=[], sampling=sampling)[0]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
