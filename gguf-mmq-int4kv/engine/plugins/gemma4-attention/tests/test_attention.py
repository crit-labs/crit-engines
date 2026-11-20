"""gemma4-attention: a ring pool's sliding-window attention and a derived-key pool's attention equal plain masked
softmax attention in FP64 over the same rows, across ring wrap-around, prompt chunks and decode steps; a row's output
does not depend on the rows beside it; a rolled-back step leaves the positions the next step reads."""
import importlib.util
import math
from pathlib import Path
import unittest

try:
    import torch
    READY = torch.cuda.is_available()
except ImportError:
    READY = False

HERE = Path(__file__).resolve().parents[1]


def _plugin():
    """plugin.py as Core's loader imports it: a package whose subpackage attn is the plugin's code/ folder."""
    import sys
    import types
    pkg = types.ModuleType("gemma4_attention_test")
    pkg.__path__ = []
    sys.modules["gemma4_attention_test"] = pkg
    sub = types.ModuleType("gemma4_attention_test.attn")
    sub.__path__ = [str(HERE / "code")]
    sys.modules["gemma4_attention_test.attn"] = sub
    spec = importlib.util.spec_from_file_location("gemma4_attention_test.plugin", HERE / "plugin.py")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "gemma4_attention_test"
    spec.loader.exec_module(module)
    return module


def reference(q, keys, values, positions, kpositions, window):
    """FP64 masked softmax attention, scale 1, GQA."""
    heads, kv = q.shape[1], keys.shape[1]
    group = heads // kv
    k = keys.double().repeat_interleave(group, dim=1)
    v = values.double().repeat_interleave(group, dim=1)
    s = torch.einsum("thd,lhd->thl", q.double(), k)
    allowed = kpositions[None, :] <= positions[:, None]
    if window:
        allowed &= (positions[:, None] - kpositions[None, :]) < window
    s = s.masked_fill(~allowed[:, None, :], -math.inf)
    return torch.einsum("thl,lhd->thd", s.softmax(-1), v)


@unittest.skipUnless(READY, "CUDA")
class Gemma4AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from crit_plugins import StepBatch
        cls.module, cls.StepBatch = _plugin(), StepBatch

    def run_sequence(self, pool, attention, layer, chunks, kv_heads, dim, derive=None):
        """Feed chunks of a sequence; return every query's output and the reference over all rows so far."""
        torch.manual_seed(0)
        pool.open("s")
        done, outs, refs = 0, [], []
        all_k, all_v = [], []
        for count in chunks:
            batch = self.StepBatch.build([("s", done, count)], "cuda")
            slots = pool.reserve(batch)
            k = torch.randn(count, kv_heads, dim, device="cuda") * 0.5
            v = torch.randn(count, kv_heads, dim, device="cuda") * 0.5
            q = torch.randn(count, 4, dim, device="cuda") * 0.3
            pool.write(layer, slots, k, v)
            out = attention.forward(layer, q, pool, batch, scale=1.0)
            stored_v = pool.v.decode(*pool.v.encode(v))    # the values as the pool keeps them (any format)
            if derive is not None:
                stored_k = pool.derive_keys(layer, stored_v, batch.positions_t)
            else:
                stored_k = pool.k.decode(*pool.k.encode(k))
            all_k.append(stored_k)
            all_v.append(stored_v)
            positions = batch.positions_t
            keys, values = torch.cat(all_k), torch.cat(all_v)
            refs.append(reference(q, keys, values, positions, torch.arange(keys.shape[0], device="cuda"),
                                  pool.window))
            outs.append(out)
            done += count
            pool.keep("s", done)
        return torch.cat(outs), torch.cat(refs)

    def test_ring_attention_is_windowed_softmax_across_wrap_around(self):
        attention = self.module.Gemma4Attention("bf16", 16, 32)
        pool = attention.pool([0], 2, 64, 4096, "cuda", window=12, ring=12 + 8, sequences=2)
        got, want = self.run_sequence(pool, attention, 0, [8, 5, 1, 1, 8, 1, 1, 7, 1], 2, 64)
        self.assertLess(float((got.double() - want).abs().max()), 2e-2)

    def test_derived_key_pages_attend_the_whole_sequence(self):
        attention = self.module.Gemma4Attention("fp8", 16, 32)
        weight = 1.0 + 0.1 * torch.randn(128, device="cuda")
        freq = torch.cat([1.0 / (1e6 ** (torch.arange(0, 32, 2, device="cuda").float() / 128)),
                          torch.zeros(48, device="cuda")])
        pool = attention.pool([3], 1, 128, 512, "cuda", derive={3: weight}, inv_freq=freq)
        got, want = self.run_sequence(pool, attention, 3, [16, 16, 3, 1, 1, 20, 1, 1], 1, 128, derive=True)
        self.assertLess(float((got.double() - want).abs().max()), 2e-2)

    def test_int4_rings_and_pages_attend_their_decoded_rows(self):
        attention = self.module.Gemma4Attention("int4", 16, 32)
        pool = attention.pool([0], 2, 64, 4096, "cuda", window=12, ring=12 + 8, sequences=2)
        got, want = self.run_sequence(pool, attention, 0, [8, 5, 1, 1, 8, 1, 1, 7, 1], 2, 64)
        self.assertLess(float((got.double() - want).abs().max()), 2e-2)
        weight = 1.0 + 0.1 * torch.randn(128, device="cuda")
        freq = torch.cat([1.0 / (1e6 ** (torch.arange(0, 32, 2, device="cuda").float() / 128)),
                          torch.zeros(48, device="cuda")])
        pages = attention.pool([3], 1, 128, 512, "cuda", derive={3: weight}, inv_freq=freq)
        got, want = self.run_sequence(pages, attention, 3, [16, 16, 3, 1, 1, 20, 1, 1], 1, 128, derive=True)
        self.assertLess(float((got.double() - want).abs().max()), 2e-2)
        # 4.5 bits a value: a row of 128 values takes 64 code bytes and 4 FP16 scales
        self.assertEqual(pages.bytes_per_token(), 1 * (64 + 4 * 2))

    def test_a_rows_output_does_not_depend_on_the_rows_beside_it(self):
        attention = self.module.Gemma4Attention("bf16", 16, 32)
        pool = attention.pool([0], 2, 64, 4096, "cuda", window=12, ring=20, sequences=3)
        for s in ("a", "b"):
            pool.open(s)
            batch = self.StepBatch.build([(s, 0, 6)], "cuda")
            pool.write(0, pool.reserve(batch), torch.randn(6, 2, 64, device="cuda"),
                       torch.randn(6, 2, 64, device="cuda"))
            pool.keep(s, 6)
        q = torch.randn(2, 4, 64, device="cuda")
        both = self.StepBatch.build([("a", 6, 1), ("b", 6, 1)], "cuda")
        slots = pool.reserve(both)
        k, v = torch.randn(2, 2, 64, device="cuda"), torch.randn(2, 2, 64, device="cuda")
        pool.write(0, slots, k, v)
        together = attention.forward(0, q, pool, both, scale=1.0)
        alone = attention.forward(0, q[:1], pool, self.StepBatch.build([("a", 6, 1)], "cuda"), scale=1.0)
        self.assertTrue(torch.equal(together[0], alone[0]))


    def test_prompt_kernel_equals_the_fp64_softmax_in_one_piece_and_many(self):
        """decode_attention.cu prefill over a ring (window) and a whole-sequence span: within BF16 rounding of FP64
        attention, and the same whether the span runs in one piece or in pieces of 32 positions."""
        attention = _plugin().Gemma4Attention("bf16", 16, 4096)
        kernels = attention.kernels
        if kernels is None or not hasattr(kernels, "prefill"):
            self.skipTest("no kernel build")
        g = torch.Generator(device="cuda").manual_seed(9)
        for dim, heads, kv, window in ((64, 4, 2, 48), (128, 8, 1, 0)):
            total, position, count = 300, 200, 70
            keys = torch.randn(total, kv, dim, device="cuda", generator=g).to(torch.bfloat16)
            values = torch.randn(total, kv, dim, device="cuda", generator=g).to(torch.bfloat16)
            q = (torch.randn(count, heads, dim, device="cuda", generator=g) * dim ** -0.5).to(torch.bfloat16)
            first = max(0, position - window + 1) if window else 0
            positions = position + torch.arange(count, device="cuda")
            kpos = torch.arange(first, position + count, device="cuda")
            want = reference(q, keys[first:position + count], values[first:position + count], positions, kpos, window)
            outs = [kernels.prefill(q, keys, values, None, None, 0, None, 0, total, window, None, None, position,
                                    first, 1.0, piece) for piece in (4096, 32)]
            for got in outs:
                err = float((got.double() - want).norm() / want.norm())
                self.assertLess(err, 1e-2, (dim, window))
            self.assertLess(float((outs[0].double() - outs[1].double()).abs().max()), 2e-2)

if __name__ == "__main__":
    unittest.main()
