"""The Qwen3.5 MoE model plugin against an exact FP32 reference of the checkpoint (work/ref/ref_forward.py: every weight
decoded by gguf-py, transformers' Qwen3_5Moe arithmetic in FP32; ref_prose.json: 149 tokens of English prose), run by
Core's loader with the engine's own plugins on the real GGUF: the engine's fixed-text scores stay within the stored
formats' int8 activation and INT4 KV rounding of the reference's log-probabilities, and requests in flight together
decode what each decodes alone. Needs CUDA, Core's crit_core_steps (CRIT_CORE_STEP_LIBRARY) and the checkpoint
(QWEN35_TEST_MODEL: a folder with the GGUF and its derived description files)."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import unittest

MODEL = os.environ.get("QWEN35_TEST_MODEL", "")
try:
    import torch
    READY = (torch.cuda.is_available() and bool(os.environ.get("CRIT_CORE_STEP_LIBRARY"))
             and bool(MODEL) and (Path(MODEL) / "config.json").is_file())
except ImportError:
    READY = False

PACKAGE = Path(__file__).resolve().parents[3]
REFERENCE = Path(__file__).resolve().parent / "ref_prose.json"


@unittest.skipUnless(READY, "a GPU engine: CUDA, CRIT_CORE_STEP_LIBRARY and the checkpoint (QWEN35_TEST_MODEL)")
class Qwen35ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from crit_core.engine import load
        cls.engine = load(PACKAGE, MODEL, context_tokens=32768)
        cls.ref = json.loads(REFERENCE.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_scores_follow_the_fp32_reference(self):
        ids = self.ref["ids"]
        scored = self.engine.score(ids, 1, "prefill")
        gaps = [abs(a - b) for a, b in zip(scored.target, self.ref["target"])]
        ppl = math.exp(-sum(scored.target) / len(scored.target))
        self.assertLess(sum(gaps) / len(gaps), 0.15, gaps)
        self.assertLess(abs(ppl / self.ref["ppl"] - 1), 0.05, (ppl, self.ref["ppl"]))
        agree = sum(a == b for a, b in zip(scored.argmax, self.ref["argmax"])) / len(scored.argmax)
        self.assertGreater(agree, 0.9)

    def test_requests_in_flight_together_decode_what_each_decodes_alone(self):
        ids = self.ref["ids"]
        prompts = [ids[:20], ids[30:61], ids[70:85]]
        alone = [self.engine.generate(p, 12, stop_tokens=[], sampling={"temperature": 0.0})[0] for p in prompts]
        with ThreadPoolExecutor(3) as pool:
            together = list(pool.map(lambda p: self.engine.generate(p, 12, stop_tokens=[],
                                                                    sampling={"temperature": 0.0})[0], prompts))
        self.assertEqual(together, alone)


if __name__ == "__main__":
    unittest.main()


class ReleaseMappedPagesTests(unittest.TestCase):
    def test_released_pages_read_back_the_same_bytes(self):
        import importlib.util
        import tempfile
        import numpy as np
        spec = importlib.util.spec_from_file_location("qwen35_model_plugin", Path(__file__).resolve().parents[1] / "plugin.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = np.random.default_rng(7).integers(0, 256, 3 << 20, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.bin"
            data.tofile(path)
            mapped = np.memmap(path, dtype=np.uint8, mode="r")
            self.assertEqual(int(mapped[::4096].sum()), int(data[::4096].sum()))   # pages read in
            self.assertEqual(module.release_mapped_pages([mapped]), data.nbytes)
            self.assertTrue(np.array_equal(np.asarray(mapped), data))
            del mapped
