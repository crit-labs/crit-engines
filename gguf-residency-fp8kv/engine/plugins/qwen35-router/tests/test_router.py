"""qwen35-router's one-kernel routing against topk-router's rule (its torch form, select) over the same logits: the
same experts, the same weights to rounding, junk rows of NaN included."""
import importlib.util
import sys
import types
from pathlib import Path
import unittest

try:
    import torch
    READY = torch.cuda.is_available()
except ImportError:
    READY = False

HERE = Path(__file__).resolve().parents[1]


def _load():
    pkg = types.ModuleType("g4r_test")
    pkg.__path__ = []
    sys.modules["g4r_test"] = pkg
    for name, folder in (("gguf", HERE.parent / "gguf-projections" / "code"),
                         ("dispatch", HERE.parent / "_sources" / "heterogeneous-expert-dispatch" / "code")):
        sub = types.ModuleType(f"g4r_test.{name}")
        sub.__path__ = [str(folder)]
        sys.modules[f"g4r_test.{name}"] = sub
    for name in ("topk", "plugin"):
        spec = importlib.util.spec_from_file_location(f"g4r_test.{name}", HERE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "g4r_test"
        sys.modules[f"g4r_test.{name}"] = module
        spec.loader.exec_module(module)
    return module


@unittest.skipUnless(READY, "CUDA")
class Qwen35RouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load()
        cls.router = cls.module.Qwen35Router({"scoring": "softmax", "score_before_selection": True,
                                              "renormalize": True})
        if not cls.router.fused_rule():
            raise unittest.SkipTest("no kernel build")

    def test_matches_select(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        for tokens, experts, hidden, top in ((1, 128, 2816, 8), (2, 128, 2816, 8), (37, 128, 704, 8), (3, 64, 96, 4)):
            r = (torch.randn(tokens, hidden, generator=g, device="cuda") * 0.5).bfloat16()
            w = (torch.randn(experts, hidden, generator=g, device="cuda") * hidden ** -0.5).bfloat16()
            scale = torch.rand(experts, generator=g, device="cuda") + 0.5
            ids, weights = self.router.route_fused(r, w, top, scale)
            logits = torch.mm(r.float(), w.float().t())
            ref_ids, ref_w = self.router.select(logits, top)
            self.assertTrue(torch.equal(ids, ref_ids), (tokens, experts))
            torch.testing.assert_close(weights, ref_w * scale[ref_ids], rtol=1e-4, atol=1e-6)

    def test_nan_rows_route_to_distinct_experts_with_zero_weight(self):
        r = torch.full((2, 64), float("nan"), device="cuda").bfloat16()
        w = torch.randn(32, 64, device="cuda").bfloat16()
        ids, weights = self.router.route_fused(r, w, 4, torch.ones(32, device="cuda"))
        for row in ids.tolist():
            self.assertEqual(len(set(row)), 4)
        self.assertTrue(torch.equal(weights, torch.zeros_like(weights)))

    def test_rules_it_does_not_take_return_none(self):
        other = self.module.Qwen35Router({"scoring": "sigmoid"})
        r = torch.randn(1, 64, device="cuda").bfloat16()
        self.assertIsNone(other.route_fused(r, torch.randn(8, 64, device="cuda").bfloat16(), 2,
                                            torch.ones(8, device="cuda")))


if __name__ == "__main__":
    unittest.main()
