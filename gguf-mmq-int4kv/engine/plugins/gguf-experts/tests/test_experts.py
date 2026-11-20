"""gguf-experts: the device form over stored GGUF expert rows equals the FP64 GeGLU block over the same experts
decoded (gguf-py's values), a token's result does not depend on the rows beside it, and a layer in pinned host memory
computes what the same layer on the device does."""
import importlib.util
import math
import sys
import types
from pathlib import Path
import unittest

try:
    import numpy as np
    import torch
    READY = torch.cuda.is_available()
except ImportError:
    READY = False

HERE = Path(__file__).resolve().parents[1]
CODE = HERE.parent / "gguf-projections" / "code"


def _load():
    pkg = types.ModuleType("gguf_experts_test")
    pkg.__path__ = []
    sys.modules["gguf_experts_test"] = pkg
    sub = types.ModuleType("gguf_experts_test.gguf")
    sub.__path__ = [str(CODE)]
    sys.modules["gguf_experts_test.gguf"] = sub
    for name in ("iq3s_grid", "gguf_blocks"):
        spec = importlib.util.spec_from_file_location(f"gguf_experts_test.gguf.{name}", CODE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "gguf_experts_test.gguf"
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    spec = importlib.util.spec_from_file_location("gguf_experts_test.plugin", HERE / "plugin.py")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "gguf_experts_test"
    spec.loader.exec_module(module)
    return module, sys.modules["gguf_experts_test.gguf.gguf_blocks"]


def rows(kind, count, cols, rng):
    elements, size = {"IQ3_S": (256, 110), "IQ4_NL": (32, 18), "Q8_0": (32, 34)}[kind]
    blocks = rng.integers(0, 256, size=(count * cols // elements, size), dtype=np.uint8)
    blocks[:, :2] = (rng.random(blocks.shape[0]) * 0.01 + 0.002).astype(np.float16).view(np.uint8).reshape(-1, 2)
    return blocks.reshape(count, cols // elements * size)


@unittest.skipUnless(READY, "CUDA and numpy")
class GgufExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, cls.blocks = _load()
        rng = np.random.default_rng(0)
        cls.experts, cls.hidden, cls.inter = 6, 256, 64
        cls.gate_up = torch.from_numpy(rows("IQ3_S", cls.experts * 2 * cls.inter, cls.hidden, rng)).view(
            cls.experts, 2 * cls.inter, -1)
        cls.down = torch.from_numpy(rows("IQ4_NL", cls.experts * cls.hidden, cls.inter, rng)).view(
            cls.experts, cls.hidden, -1)

    def plugin(self, host=False, prompt="bf16"):
        p = self.module.GgufExperts("gelu_tanh", 4, prompt)
        gu, dn = (self.gate_up.pin_memory(), self.down.pin_memory()) if host else (self.gate_up.cuda(),
                                                                                  self.down.cuda())
        p.load_layer(0, gu, dn, "IQ3_S", "IQ4_NL", self.hidden, self.inter)
        return p

    def reference(self, x, ids, weights):
        gu = self.blocks.decode(self.gate_up.reshape(-1, self.gate_up.shape[-1]).cuda(), "IQ3_S", self.hidden)
        dn = self.blocks.decode(self.down.reshape(-1, self.down.shape[-1]).cuda(), "IQ4_NL", self.inter)
        gu = gu.view(self.experts, 2 * self.inter, self.hidden).to(torch.bfloat16).double()
        dn = dn.view(self.experts, self.hidden, self.inter).to(torch.bfloat16).double()
        out = torch.zeros(x.shape[0], self.hidden, dtype=torch.float64, device="cuda")
        xb = x.to(torch.bfloat16).double()
        for t in range(x.shape[0]):
            for e, w in zip(ids[t].tolist(), weights[t].tolist()):
                if e < 0:
                    continue
                h = gu[e] @ xb[t]
                g, u = h[:self.inter], h[self.inter:]
                act = 0.5 * g * (1 + torch.tanh(math.sqrt(2 / math.pi) * (g + 0.044715 * g ** 3)))
                out[t] += w * (dn[e] @ (act * u))
        return out

    def test_forward_equals_the_fp64_block_over_the_decoded_experts(self):
        g = torch.Generator(device="cuda").manual_seed(1)
        x = torch.randn(7, self.hidden, device="cuda", generator=g)
        ids = torch.randint(0, self.experts, (7, 2), device="cuda", generator=g)
        ids[3, 1] = -1
        weights = torch.rand(7, 2, device="cuda", generator=g)
        got = self.plugin().forward(0, x, ids, weights).double()
        want = self.reference(x, ids, weights)
        self.assertLess(float((got - want).abs().max()), 2e-2 * float(want.abs().max()) + 1e-4)

    def test_prompt_rows_grouped_by_expert_equal_the_fp64_block(self):
        g = torch.Generator(device="cuda").manual_seed(4)
        x = torch.randn(20, self.hidden, device="cuda", generator=g)
        ids = torch.randint(0, self.experts, (20, 2), device="cuda", generator=g)
        ids[5, 0] = -1
        weights = torch.rand(20, 2, device="cuda", generator=g)
        got = self.plugin().forward(0, x, ids, weights)
        want = self.reference(x, ids, weights)
        self.assertLess(float((got.double() - want).abs().max()), 2e-2 * float(want.abs().max()) + 1e-4)
        self.assertTrue(torch.equal(self.plugin(host=True).forward(0, x, ids, weights), got))
        # FP8 prompt activations (each row rounded to E4M3 at a power-of-two scale): within FP8's rounding
        fp8 = self.plugin(prompt="fp8").forward(0, x, ids, weights).double()
        self.assertLess(float((fp8 - want).norm() / want.norm()), 0.1)

    def test_prompt_rows_with_stored_activations_follow_the_fp64_block(self):
        """Prompt rows through the int8 tensor-core kernel with routes sorted on the device (dropped routes too):
        within the int8 activations' rounding of the exact product, and the same as decode rows' stored path."""
        g = torch.Generator(device="cuda").manual_seed(6)
        x = torch.randn(45, self.hidden, device="cuda", generator=g)
        ids = torch.randint(0, self.experts, (45, 2), device="cuda", generator=g)
        ids[7, 1] = -1
        ids[30, 0] = -1
        weights = torch.rand(45, 2, device="cuda", generator=g)
        plugin = self.plugin(prompt="stored")
        got = plugin.forward(0, x, ids, weights).double()
        want = self.reference(x, ids, weights)
        self.assertLess(float((got - want).norm() / want.norm()), 0.03)
        rows = torch.cat([plugin.forward(0, x[i:i + 5], ids[i:i + 5], weights[i:i + 5]) for i in range(0, 45, 5)])
        self.assertLess(float((got - rows.double()).norm() / want.norm()), 0.01)

    def test_a_tokens_bits_do_not_depend_on_the_rows_beside_it(self):
        plugin = self.plugin()
        g = torch.Generator(device="cuda").manual_seed(2)
        x = torch.randn(6, self.hidden, device="cuda", generator=g)
        ids = torch.randint(0, self.experts, (6, 2), device="cuda", generator=g)
        weights = torch.rand(6, 2, device="cuda", generator=g)
        together = plugin.forward(0, x, ids, weights)
        for t in range(6):
            alone = plugin.forward(0, x[t:t + 1], ids[t:t + 1], weights[t:t + 1])
            self.assertTrue(torch.equal(alone[0], together[t]), t)

    def test_a_layer_in_host_memory_computes_what_the_device_layer_does(self):
        g = torch.Generator(device="cuda").manual_seed(3)
        x = torch.randn(3, self.hidden, device="cuda", generator=g)
        ids = torch.randint(0, self.experts, (3, 2), device="cuda", generator=g)
        weights = torch.rand(3, 2, device="cuda", generator=g)
        self.assertTrue(torch.equal(self.plugin(host=True).forward(0, x, ids, weights),
                                    self.plugin().forward(0, x, ids, weights)))


if __name__ == "__main__":
    unittest.main()
