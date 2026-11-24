"""gguf-projections: every block decoder equals gguf-py's dequantize bit for bit on random blocks, matmul equals the
FP64 product of the decoded matrix within BF16 rounding, and a decode row's bits do not depend on the rows beside it."""
import importlib.util
import sys
from pathlib import Path
import unittest

try:
    import numpy as np
    import torch
    from gguf import GGMLQuantizationType, quants
    READY = torch.cuda.is_available()
except ImportError:
    READY = False

HERE = Path(__file__).resolve().parents[1]


def _plugin():
    from crit_plugins.manifest import load_plugin_module   # noqa: F401  (Core loads plugins this way when present)


def _blocks_module():
    import types
    pkg = types.ModuleType("gguf_test_code")
    pkg.__path__ = [str(HERE / "code")]
    sys.modules["gguf_test_code"] = pkg
    for name in ("iq3s_grid", "gguf_blocks"):
        spec = importlib.util.spec_from_file_location(f"gguf_test_code.{name}", HERE / "code" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "gguf_test_code"
        sys.modules[f"gguf_test_code.{name}"] = module
        spec.loader.exec_module(module)
    return sys.modules["gguf_test_code.gguf_blocks"]


def random_rows(kind, rows, cols, rng):
    """Random GGUF rows of kind: random bytes with each block's f16 scale set to a small finite value."""
    from_kind = {"Q8_0": (32, 34), "IQ4_NL": (32, 18), "IQ4_XS": (256, 136), "IQ3_S": (256, 110)}
    elements, size = from_kind[kind]
    blocks = rng.integers(0, 256, size=(rows * cols // elements, size), dtype=np.uint8)
    scales = (rng.random(blocks.shape[0]) * 0.02 + 0.001).astype(np.float16)
    blocks[:, :2] = scales.view(np.uint8).reshape(-1, 2)
    return blocks.reshape(rows, cols // elements * size)


@unittest.skipUnless(READY, "CUDA, numpy and gguf-py")
class GgufProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocks = _blocks_module()
        sys.modules.setdefault("gguf_projections_plugin_code", sys.modules["gguf_test_code"])

    def test_decoders_equal_gguf_py_bit_for_bit(self):
        rng = np.random.default_rng(0)
        for kind in ("Q8_0", "IQ4_NL", "IQ4_XS", "IQ3_S"):
            raw = random_rows(kind, 6, 512, rng)
            expected = quants.dequantize(raw, getattr(GGMLQuantizationType, kind)).reshape(6, 512)
            got = self.blocks.decode(torch.from_numpy(raw).cuda(), kind, 512).cpu().numpy()
            self.assertTrue((expected.view(np.uint32) == got.view(np.uint32)).all(), kind)

    def test_matmul_equals_the_decoded_product_and_keeps_each_rows_bits(self):
        spec = importlib.util.spec_from_file_location("gguf_proj_plugin", HERE / "plugin.py",
                                                      submodule_search_locations=[])
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "gguf_test_pkg"
        pkg = type(sys)("gguf_test_pkg")
        pkg.__path__ = []
        sys.modules["gguf_test_pkg"] = pkg
        sys.modules["gguf_test_pkg.gguf"] = sys.modules["gguf_test_code"]
        sys.modules["gguf_test_pkg.gguf.gguf_blocks"] = self.blocks
        spec.loader.exec_module(module)
        plugin = module.GgufProjections(1 << 16)   # small chunks: the chunked path too
        rng = np.random.default_rng(1)
        for kind in ("Q8_0", "IQ3_S"):
            raw = random_rows(kind, 96, 512, rng)
            matrix = plugin.load({"weight": torch.from_numpy(raw), "cols": 512}, "gguf:" + kind)
            dense = self.blocks.decode(torch.from_numpy(raw).cuda(), kind, 512).double()
            x = torch.randn(5, 512, device="cuda").to(torch.bfloat16)
            y = plugin.matmul(x, matrix, torch.float32)
            expected = x.double() @ dense.to(torch.bfloat16).double().t()
            self.assertLess(float((y.double() - expected).abs().max()), 1e-2 * float(expected.abs().max()) + 1e-3)
            for r in range(5):
                alone = plugin.matmul(x[r:r + 1], matrix, torch.float32)
                self.assertTrue(torch.equal(alone[0], y[r]), (kind, r))


    def test_stored_activations_follow_the_decoded_product(self):
        """The stored formats' int8 activation units: decode rows (gemv_q8, a row's bits its own) and prompt rows
        (mmq on the int8 tensor cores) within the activations' rounding of the exact product, every format."""
        spec = importlib.util.spec_from_file_location("gguf_proj_plugin_s", HERE / "plugin.py",
                                                      submodule_search_locations=[])
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "gguf_test_pkg"
        pkg = type(sys)("gguf_test_pkg")
        pkg.__path__ = []
        sys.modules["gguf_test_pkg"] = pkg
        sys.modules["gguf_test_pkg.gguf"] = sys.modules["gguf_test_code"]
        sys.modules["gguf_test_pkg.gguf.gguf_blocks"] = self.blocks
        spec.loader.exec_module(module)
        plugin = module.GgufProjections(1 << 16, "stored", 64, "stored")
        if plugin.kernels is None or not hasattr(plugin.kernels, "mmq"):
            self.skipTest("no kernel build")
        rng = np.random.default_rng(2)
        for kind in ("Q8_0", "IQ4_NL", "IQ4_XS", "IQ3_S"):
            raw = random_rows(kind, 136, 512, rng)
            matrix = plugin.load({"weight": torch.from_numpy(raw), "cols": 512}, "gguf:" + kind)
            dense = self.blocks.decode(torch.from_numpy(raw).cuda(), kind, 512).double()
            for tokens in (3, 77):
                x = torch.randn(tokens, 512, device="cuda").to(torch.bfloat16)
                y = plugin.matmul(x, matrix, torch.float32).double()
                expected = x.double() @ dense.t()
                self.assertLess(float((y - expected).norm() / expected.norm()), 0.02, (kind, tokens))
            x = torch.randn(5, 512, device="cuda").to(torch.bfloat16)
            y = plugin.matmul(x, matrix, torch.float32)
            for r in range(5):
                self.assertTrue(torch.equal(plugin.matmul(x[r:r + 1], matrix, torch.float32)[0], y[r]), (kind, r))

if __name__ == "__main__":
    unittest.main()
