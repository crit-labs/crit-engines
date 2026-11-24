"""The gated delta rule kernels against the plugin's torch reference: prompt chunks continued across steps, decode rows
of several sequences in one call, and a sequence's result alone against batched."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]


def load_plugin():
    name = "qwen35_recurrence_test"
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(HERE)]
    sys.modules[name] = pkg
    sub = types.ModuleType(name + ".gdn")
    sub.__path__ = [str(HERE / "code")]
    sys.modules[name + ".gdn"] = sub
    for mod, path in ((name + ".gdn.kernels", HERE / "code" / "kernels.py"), (name + ".plugin", HERE / "plugin.py")):
        spec = importlib.util.spec_from_file_location(mod, path)
        module = importlib.util.module_from_spec(spec)
        module.__package__ = mod.rsplit(".", 1)[0]
        sys.modules[mod] = module
        spec.loader.exec_module(module)
    sub.kernels = sys.modules[name + ".gdn.kernels"]
    return sys.modules[name + ".plugin"]


def make(plugin, kh=4, vh=8, seqs=3):
    rec = plugin.Qwen35Recurrence()
    channels = 2 * kh * 128 + vh * 128
    pool = rec.pool([0], channels, kh, vh, seqs, torch.device("cuda"))
    return rec, pool, channels


def inputs(T, channels, vh, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    mixed = torch.randn(T, channels, device="cuda", generator=g)
    gate = -torch.rand(T, vh, device="cuda", generator=g) * 2.0
    beta = torch.rand(T, vh, device="cuda", generator=g)
    w = torch.randn(channels, 4, device="cuda", generator=g) * 0.5
    return mixed, w, gate, beta


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class RecurrenceTests(unittest.TestCase):
    def test_kernels_match_reference(self):
        plugin = load_plugin()
        rec, pool, channels = make(plugin)
        if pool.kernels is None:
            self.skipTest("gdn.cu not built")
        _, ref_pool, _ = make(plugin)
        mixed, w, gate, beta = inputs(40, channels, 8, 1)
        for p in (pool, ref_pool):
            p.open("a")
            p.open("b")
        # a prompt of 20 tokens for a in two chunks, b in one of 12, then decode rows of both in one call
        steps = [[(0, 13, "a")], [(13, 7, "a"), (20, 12, "b")], [(32, 1, "a"), (33, 1, "b")], [(34, 1, "b"), (35, 1, "a")]]
        for step in steps:
            rows = [(s, c, pool.slot(q)) for s, c, q in step]
            groups = rec.groups(rows, torch.device("cuda"))
            got = pool.forward(0, mixed, w, gate, beta, groups)
            want = ref_pool.reference(0, mixed, w, gate, beta, groups)
            for s, c, _ in rows:
                assert torch.allclose(got[s:s + c], want[s:s + c], atol=2e-4, rtol=2e-3), (step, (got - want).abs().max())
        assert torch.allclose(pool.state, ref_pool.state, atol=2e-4, rtol=2e-3)


    def test_row_alone_matches_batched(self):
        plugin = load_plugin()
        rec, pool, channels = make(plugin)
        if pool.kernels is None:
            self.skipTest("gdn.cu not built")
        _, alone, _ = make(plugin)
        mixed, w, gate, beta = inputs(4, channels, 8, 2)
        for p in (pool, alone):
            p.open("a")
            p.open("b")
        both = pool.forward(0, mixed, w, gate, beta, rec.groups([(0, 2, pool.slot("a")), (2, 2, pool.slot("b"))], "cuda"))
        one = alone.forward(0, mixed, w, gate, beta, rec.groups([(0, 2, alone.slot("a"))], "cuda"))
        assert torch.equal(both[:2], one[:2])


if __name__ == "__main__":
    unittest.main()
