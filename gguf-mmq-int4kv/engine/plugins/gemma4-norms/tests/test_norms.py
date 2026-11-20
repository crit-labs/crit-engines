"""gemma4-norms against torch-norms (its reference) on the calls it runs in one kernel: rows, heads (strided views of
a fused projection), heads with rotary positions (zero frequencies included), and the tanh GELU product; each row's
bits the same alone and among other rows."""
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
    pkg = types.ModuleType("g4n_test")
    pkg.__path__ = []
    sys.modules["g4n_test"] = pkg
    for name, folder in (("g4", HERE / "code"), ("norms", HERE.parent / "_sources" / "fused-norms" / "code")):
        sub = types.ModuleType(f"g4n_test.{name}")
        sub.__path__ = [str(folder)]
        sys.modules[f"g4n_test.{name}"] = sub
    spec = importlib.util.spec_from_file_location("g4n_test.plugin", HERE / "plugin.py")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "g4n_test"
    spec.loader.exec_module(module)
    return module, sys.modules["g4n_test.norms.torch_norms"]


@unittest.skipUnless(READY, "CUDA")
class Gemma4NormsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module, cls.ref = _load()
        cls.norms = module.Gemma4Norms()

    def close(self, a, b, tol=2e-2):
        self.assertEqual(a.dtype, b.dtype)
        self.assertLess(float((a.float() - b.float()).abs().max()), tol * (float(b.float().abs().max()) + 1e-6))

    def test_the_kernels_are_built(self):
        self.assertIsNotNone(self.norms.kernels)

    def test_rows_and_heads_equal_torch_norms(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        x = torch.randn(5, 2816, device="cuda", generator=g)
        w = 1 + 0.1 * torch.randn(2816, device="cuda", generator=g)
        for dtype in (torch.float32, torch.bfloat16):
            for out in (torch.float32, torch.bfloat16):
                self.close(self.norms.rms_norm(x.to(dtype), w, 1e-6, out_dtype=out),
                           self.ref.rms_norm(x.to(dtype), w, 1e-6, out_dtype=out))
                self.close(self.norms.rms_norm(x.to(dtype), None, 1e-6, out_dtype=out),
                           self.ref.rms_norm(x.to(dtype), None, 1e-6, out_dtype=out))
        fused = torch.randn(3, 4096 + 2048 + 2048, device="cuda", generator=g).to(torch.bfloat16)
        k = fused[:, 4096:6144].view(3, 8, 256)
        self.close(self.norms.rms_norm_heads(k, None, 1e-6), self.ref.rms_norm_heads(k, None, 1e-6))

    def test_rotary_heads_equal_torch_norms(self):
        g = torch.Generator(device="cuda").manual_seed(1)
        fused = torch.randn(4, 16 * 512 + 2 * 512, device="cuda", generator=g).to(torch.bfloat16)
        q = fused[:, :16 * 512].view(4, 16, 512)
        w = 1 + 0.1 * torch.randn(512, device="cuda", generator=g)
        freqs = torch.cat([1.0 / (1e6 ** (torch.arange(0, 128, 2, device="cuda").float() / 512)),
                           torch.zeros(192, device="cuda")])
        positions = torch.tensor([0, 7, 1000, 150000], device="cuda")
        self.close(self.norms.rms_norm_rope(q, w, 1e-6, positions, freqs),
                   self.ref.rms_norm_rope(q, w, 1e-6, positions, freqs))

    def test_fused_calls_equal_the_separate_norms(self):
        """add_rms, rms3 and ffn_out against torch-norms' separate calls, the layer tail with the next input norm."""
        if not self.norms.fused_ready(torch.zeros(1, 8, device="cuda")):
            self.skipTest("no kernel build")
        g = torch.Generator(device="cuda").manual_seed(3)
        dim, eps = 2816, 1e-6
        x = torch.randn(5, dim, device="cuda", generator=g)
        w = [1 + 0.1 * torch.randn(dim, device="cuda", generator=g) for _ in range(5)]
        o = torch.randn(5, dim, device="cuda", generator=g)
        want = x + self.ref.rms_norm(o, w[0], eps, out_dtype=torch.float32)
        got = x.clone()
        self.norms.add_rms(got, o, w[0], eps)
        self.close(got, want, 1e-4)
        for got_n, wi in zip(self.norms.rms3(want, w[1], w[2], w[3], eps), w[1:4]):
            self.close(got_n, self.ref.rms_norm(want, wi, eps, out_dtype=torch.bfloat16))
        mlp, e = torch.randn(5, dim, device="cuda", generator=g), torch.randn(5, dim, device="cuda", generator=g)
        m = (self.ref.rms_norm(mlp, w[1], eps, out_dtype=torch.float32)
             + self.ref.rms_norm(e, w[2], eps, out_dtype=torch.float32))
        tail = (want + self.ref.rms_norm(m, w[3], eps, out_dtype=torch.float32)) * 0.75
        state = want.clone()
        h = self.norms.ffn_out(state, mlp, e, w[1], w[2], w[3], 0.75, eps, w[4])
        self.close(state, tail, 1e-4)
        self.close(h, self.ref.rms_norm(tail, w[4], eps, out_dtype=torch.bfloat16))

    def test_gelu_product_and_row_invariance(self):
        g = torch.Generator(device="cuda").manual_seed(2)
        x = torch.randn(6, 2 * 2112, device="cuda", generator=g).to(torch.bfloat16)
        self.close(self.norms.gelu_mul(x, approximate="tanh"), self.ref.gelu_mul(x, approximate="tanh"))
        w = 1 + 0.1 * torch.randn(2112 * 2, device="cuda", generator=g)
        together = self.norms.rms_norm(x, w, 1e-6, out_dtype=torch.float32)
        for r in range(6):
            self.assertTrue(torch.equal(self.norms.rms_norm(x[r:r + 1], w, 1e-6, out_dtype=torch.float32)[0],
                                        together[r]))


if __name__ == "__main__":
    unittest.main()
