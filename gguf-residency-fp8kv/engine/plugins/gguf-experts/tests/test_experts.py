"""gguf-experts: a layer split between device slots and page-locked host memory computes the SwiGLU experts of the
same blocks decoded (gguf-py's values, FP64 products) within the stored formats' int8 activation rounding: decode rows
with the CPU tier (host routes on the CPU, joined through the mailbox) and without it (host routes read over the link),
and prompt rows through the int8 tensor-core kernel; a token's decode result does not depend on the rows beside it."""
import importlib.util
import sys
import types
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
CODE = HERE.parent / "gguf-projections" / "code"


def _load(name):
    pkg = types.ModuleType(name)
    pkg.__path__ = []
    sys.modules[name] = pkg
    for sub_name, folder in (("gguf", CODE), ("tier", HERE / "code")):
        sub = types.ModuleType(f"{name}.{sub_name}")
        sub.__path__ = [str(folder)]
        sys.modules[f"{name}.{sub_name}"] = sub
    for full, path in ((f"{name}.gguf.iq3s_grid", CODE / "iq3s_grid.py"),
                       (f"{name}.gguf.gguf_blocks", CODE / "gguf_blocks.py"),
                       (f"{name}.gguf.kernels", CODE / "kernels.py"),
                       (f"{name}.tier.tier_kernels", HERE / "code" / "tier_kernels.py"),
                       (f"{name}.plugin", HERE / "plugin.py")):
        spec = importlib.util.spec_from_file_location(full, path)
        module = importlib.util.module_from_spec(spec)
        module.__package__ = full.rsplit(".", 1)[0]
        sys.modules[full] = module
        spec.loader.exec_module(module)
        parent, leaf = full.rsplit(".", 1)
        setattr(sys.modules[parent], leaf, module)
    return sys.modules[f"{name}.plugin"]


def blocks(kind, rows, cols, rng):
    size = {"IQ3_S": 110, "IQ4_XS": 136}[kind]
    raw = rng.integers(0, 256, size=(rows, cols // 256, size), dtype=np.uint8)
    raw[:, :, 0:2] = (rng.random((rows, cols // 256)) * 0.01 + 0.001).astype(np.float16).view(np.uint8).reshape(
        rows, cols // 256, 2)
    return raw.reshape(rows, -1)


def dense(kind, raw):
    qt = {"IQ3_S": GGMLQuantizationType.IQ3_S, "IQ4_XS": GGMLQuantizationType.IQ4_XS}[kind]
    return torch.from_numpy(quants.dequantize(raw, qt).astype(np.float64))


@unittest.skipUnless(READY, "CUDA, numpy and gguf-py")
class GgufExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.experts, cls.hidden, cls.inter = 8, 2048, 256   # gate|up over 64 units (the GEMV's one-row path), down over 8 (its strips)
        cls.gate = np.stack([blocks("IQ3_S", cls.inter, cls.hidden, rng) for _ in range(cls.experts)])
        cls.up = np.stack([blocks("IQ3_S", cls.inter, cls.hidden, rng) for _ in range(cls.experts)])
        cls.down = np.stack([blocks("IQ4_XS", cls.hidden, cls.inter, rng) for _ in range(cls.experts)])
        cls.G = [dense("IQ3_S", g) for g in cls.gate]
        cls.U = [dense("IQ3_S", u) for u in cls.up]
        cls.D = [dense("IQ4_XS", d) for d in cls.down]

    def make(self, name, cpu_tier):
        module = _load(name)
        experts = module.GgufExperts("stored", "stored", cpu_tier=cpu_tier, workers=3)
        experts.load_layer(0, torch.from_numpy(self.gate), torch.from_numpy(self.up), torch.from_numpy(self.down),
                           "IQ3_S", "IQ4_XS", self.hidden, self.inter, resident=[0, 2, 3, 6])
        return experts

    def reference(self, x, ids, weights):
        out = torch.zeros(x.shape[0], self.hidden, dtype=torch.float64)
        xd = x.double().cpu()
        for t in range(x.shape[0]):
            for e, w in zip(ids[t].tolist(), weights[t].tolist()):
                if e < 0:
                    continue
                g, u = self.G[e] @ xd[t], self.U[e] @ xd[t]
                out[t] += w * (self.D[e] @ (g * torch.sigmoid(g) * u))
        return out

    def check(self, experts, tokens, seed):
        g = torch.Generator().manual_seed(seed)
        x = (torch.randn(tokens, self.hidden, generator=g) * 0.5).bfloat16().cuda()
        ids = torch.randint(0, self.experts, (tokens, 4), generator=g)
        ids[0, 1] = -1
        weights = torch.rand(tokens, 4, generator=g)
        got = experts.forward(0, x, ids.cuda(), weights.cuda()).double().cpu()
        want = self.reference(x, ids, weights)
        rel = float((got - want).norm() / want.norm())
        self.assertLess(rel, 0.03, rel)
        return x, ids, weights, got

    def test_decode_rows_with_the_cpu_tier(self):
        experts = self.make("gguf_experts_tier", True)
        if experts.tier is None:
            self.skipTest("expert_tier.cu not built")
        x, ids, weights, got = self.check(experts, 3, 1)
        alone = experts.forward(0, x[1:2], ids[1:2].cuda(), weights[1:2].cuda()).double().cpu()
        self.assertTrue(torch.allclose(alone[0], got[1], rtol=1e-5, atol=1e-6))
        experts.close()

    def test_decode_rows_over_the_link(self):
        experts = self.make("gguf_experts_link", False)
        self.check(experts, 5, 2)
        experts.close()

    def test_swapped_residency_computes_the_same(self):
        """Host experts swapped onto the device (and device experts off it) by adapt(): every path still reads each
        expert's own bytes, decode rows with the CPU tier and prompt rows alike."""
        experts = self.make("gguf_experts_swap", True)
        experts.adapt_min, experts.adapt_margin = 0.0, -1.0
        L = experts.layers[0]
        L.route_counts[torch.tensor([1, 4, 5, 7], device="cuda")] = 50     # host experts made hot
        self.assertGreater(experts.adapt(), 0)
        self.assertTrue(set(L.slot_of) & {1, 4, 5, 7})
        self.check(experts, 3, 4)
        self.check(experts, 40, 5)
        experts.close()

    def test_a_fresh_engine_swaps_up_to_warm_swaps_for_its_first_warm_calls(self):
        """warm_swaps bounds the first warm_calls adaptations (more experts a call than adapt_swaps), adapt_swaps every
        later one; the swapped placement computes the same."""
        experts = self.make("gguf_experts_warm", True)
        self.addCleanup(experts.close)
        experts.adapt_min, experts.adapt_margin = 0.0, -1.0
        experts.adapt_swaps, experts.warm_swaps, experts.warm_calls = 1, 2, 1
        L = experts.layers[0]
        hot = sorted(e for e in L.host_of if e not in L.lent)   # the fixture's four host experts
        self.assertEqual(len(hot), 4)

        def heat(chosen):
            L.route_counts.zero_()
            L.score.zero_()
            L.route_counts[torch.tensor(chosen, device="cuda")] = 50
        heat(hot[:2])
        self.assertEqual(experts.adapt(), 2)          # the first call: warm_swaps
        self.assertTrue(set(hot[:2]) <= set(L.slot_of))
        heat(hot[2:])
        self.assertEqual(experts.adapt(), 1)          # past warm_calls: adapt_swaps
        self.check(experts, 3, 8)
        self.check(experts, 40, 9)

    def test_a_prompts_routes_warm_the_placement_and_compute_the_same(self):
        """prompt_warm() after a prompt: the host experts its rows routed to come onto the device (up to prompt_swaps),
        the counts are spent, and the prompt and decode rows give the same values before and after."""
        experts = self.make("gguf_experts_prompt", True)
        self.addCleanup(experts.close)
        experts.adapt_min, experts.adapt_margin = 0.0, -1.0
        experts.prompt_rows, experts.prompt_swaps = 32, 2
        L = experts.layers[0]
        host = sorted(e for e in L.host_of if e not in L.lent)
        g = torch.Generator().manual_seed(11)
        tokens = 40
        x = (torch.randn(tokens, self.hidden, generator=g) * 0.5).bfloat16().cuda()
        ids = torch.tensor([[host[0], host[1], host[0], host[1]]] * tokens)
        weights = torch.rand(tokens, 4, generator=g)
        before = experts.forward(0, x, ids.cuda(), weights.cuda()).double().cpu()
        self.assertEqual(experts.prompt_tokens, tokens)
        self.assertEqual(experts.prompt_warm(), 2)
        self.assertTrue({host[0], host[1]} <= set(L.slot_of))
        self.assertEqual((experts.prompt_tokens, experts.prompt_counts), (0, {}))
        self.assertEqual(experts.prompt_warm(), 0)    # nothing left to warm from
        after = experts.forward(0, x, ids.cuda(), weights.cuda()).double().cpu()
        self.assertTrue(torch.equal(before, after))
        self.check(experts, 3, 12)

    def test_every_layer_adapts_at_once_by_the_per_layer_rule(self):
        """adapt() reads every layer's route counts in one copy and picks by tensor ops: the same swaps the per-layer
        rule gives (the hottest host experts, not lent, against the coldest device experts in rank order, where the host
        expert reaches adapt_min and gains more than adapt_margin; the largest gains first, adapt_swaps of them), over
        layers with different placements, a lent expert and decayed scores."""
        module = _load("gguf_experts_stacked")
        experts = module.GgufExperts("stored", "stored", cpu_tier=True, workers=3)
        self.addCleanup(experts.close)
        placements = {0: [0, 2, 3, 6], 1: [1, 4], 2: [5, 6, 7]}
        for layer, resident in placements.items():
            experts.load_layer(layer, torch.from_numpy(self.gate), torch.from_numpy(self.up),
                               torch.from_numpy(self.down), "IQ3_S", "IQ4_XS", self.hidden, self.inter,
                               resident=resident)
        experts.adapt_min, experts.adapt_margin, experts.adapt_decay, experts.adapt_swaps = 3.0, 2.0, 0.5, 4
        L1 = experts.layers[1]
        lent = next(e for e in sorted(L1.host_of))
        experts.lent_mask[1, lent] = True          # as lend() marks it (the reference skips it as L.lent does)
        L1.lent[lent] = (0, None, None)
        g = torch.Generator().manual_seed(7)
        for _ in range(3):
            scores = {layer: L.score.clone() for layer, L in experts.layers.items()}
            counts = {}
            for layer, L in experts.layers.items():
                counts[layer] = torch.randperm(40, generator=g)[:self.experts].to(torch.int32)   # distinct counts
                L.route_counts.copy_(counts[layer].cuda())
            expected = []
            for layer, L in experts.layers.items():
                score = scores[layer] * 0.5 + counts[layer].float()
                host = sorted((e for e in L.host_of if e not in L.lent), key=lambda e: -float(score[e]))[:4]
                dev = sorted(L.slot_of, key=lambda e: float(score[e]))[:4]
                for b, a in zip(host, dev):
                    if float(score[b]) >= 3.0 and float(score[b]) - float(score[a]) > 2.0:
                        expected.append((float(score[b]) - float(score[a]), layer, a, b))
            expected = sorted(expected, reverse=True)[:4]
            placed = {layer: (dict(L.slot_of), dict(L.host_of)) for layer, L in experts.layers.items()}
            done = experts.adapt()
            self.assertEqual(done, len(expected))
            for gain, layer, a, b in expected:
                L = experts.layers[layer]
                self.assertIn(b, L.slot_of)
                self.assertIn(a, L.host_of)
                self.assertEqual(L.slot_of[b], placed[layer][0][a])
            for layer, L in experts.layers.items():
                self.assertTrue(torch.equal(L.score, scores[layer] * 0.5 + counts[layer].float()))
                self.assertTrue(bool((L.route_counts == 0).all()))
                self.assertEqual(sorted(L.slot_of), sorted(e for e in range(self.experts)
                                                           if bool(experts.dev_mask[layer, e])))
        del L1.lent[lent]
        experts.lent_mask[1, lent] = False

    def test_lent_copies_and_their_reclaim_compute_the_same(self):
        """Host experts copied into lent pages (another owner's memory) are read there, decode and prompt rows alike;
        reclaiming the pages points them back at their host copies, with the same results."""
        experts = self.make("gguf_experts_lend", True)
        spaces = [torch.zeros(64, 65536, dtype=torch.uint8, device="cuda") for _ in range(2)]
        ranking = [(0, e) for e in (1, 4, 5, 7)]
        floor = experts.lend(spaces, 65536, ranking, 4)
        L = experts.layers[0]
        self.assertGreater(len(L.lent), 0)
        self.assertLess(floor, 64)
        self.check(experts, 3, 6)
        self.check(experts, 40, 7)
        lent = len(L.lent)
        above = experts.reclaim(floor)              # the floor page handed out: only the experts on it go home
        self.assertGreater(above, floor)
        self.assertGreater(len(L.lent), 0)
        self.assertTrue(all(first > floor for first, _, _ in L.lent.values()))
        self.check(experts, 3, 10)
        self.check(experts, 40, 11)
        self.assertEqual(experts.reclaim(1 << 62), 1 << 62)
        self.assertEqual(len(L.lent), 0)
        spaces[0].fill_(255)
        spaces[1].fill_(255)
        self.check(experts, 3, 8)
        self.check(experts, 40, 9)
        experts.close()

    def test_cpu_and_device_tiers_give_the_same_bits(self):
        """Every expert on the CPU tier against every expert on the device: decode rows equal (the CPU rows reproduce
        the device GEMV's order; at most a last-bit SiLU difference before the BF16 rounding)."""
        module = _load("gguf_experts_bits")
        on_cpu = module.GgufExperts("stored", "stored", cpu_tier=True, workers=3)
        on_dev = module.GgufExperts("stored", "stored", cpu_tier=True, workers=3)
        args = (torch.from_numpy(self.gate), torch.from_numpy(self.up), torch.from_numpy(self.down), "IQ3_S", "IQ4_XS",
                self.hidden, self.inter)
        on_cpu.load_layer(0, *args, resident=[0])
        on_dev.load_layer(0, *args, resident=list(range(self.experts)))
        if on_cpu.tier is None:
            self.skipTest("expert_tier.cu not built")
        g = torch.Generator().manual_seed(11)
        x = (torch.randn(2, self.hidden, generator=g) * 0.5).bfloat16().cuda()
        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 2]], device="cuda")
        w = torch.rand(2, 4, generator=g).cuda()
        a = on_cpu.forward(0, x, ids, w)
        b = on_dev.forward(0, x, ids, w)
        same = (a == b).float().mean().item()
        self.assertEqual(same, 1.0, same)
        self.assertLess(float((a - b).abs().max()), 1e-3 * float(b.abs().max()))
        on_cpu.close()
        on_dev.close()

    def test_prompt_rows(self):
        experts = self.make("gguf_experts_prompt", False)
        self.check(experts, 40, 3)
        experts.close()


if __name__ == "__main__":
    unittest.main()
