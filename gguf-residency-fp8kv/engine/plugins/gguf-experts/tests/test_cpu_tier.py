"""The CPU expert tier (code/expert_tier.cu) against an FP32 product of the dequantized experts: host routes posted to a
mailbox slot, computed on the CPU from the stored blocks, joined and combined with device routes, eagerly and replayed
in a captured CUDA graph. Random IQ3_S gate/up and IQ4_XS down blocks with small finite scales."""
import importlib.util
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
    from gguf import GGMLQuantizationType, quants
    READY = torch.cuda.is_available()
except ImportError:
    READY = False

HERE = Path(__file__).resolve().parents[1]


def module():
    found = sorted((HERE / "code" / "build").glob("expert_tier*.pyd")) + sorted(
        (HERE / "code" / "build").glob("expert_tier*.so"))
    if not found:
        return None
    spec = importlib.util.spec_from_file_location("expert_tier", str(found[0]))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def blocks(kind, rows, cols, rng):
    size = {"IQ3_S": 110, "IQ4_XS": 136}[kind]
    raw = rng.integers(0, 256, size=(rows, cols // 256, size), dtype=np.uint8)
    d = (rng.random((rows, cols // 256)) * 0.01 + 0.001).astype(np.float16)
    raw[:, :, 0:2] = d.view(np.uint8).reshape(rows, cols // 256, 2)
    return raw.reshape(rows, -1)


def dense(kind, raw):
    qt = {"IQ3_S": GGMLQuantizationType.IQ3_S, "IQ4_XS": GGMLQuantizationType.IQ4_XS}[kind]
    return quants.dequantize(raw, qt).astype(np.float32)


@unittest.skipUnless(READY, "needs CUDA, numpy and gguf-py")
class CpuTierTests(unittest.TestCase):
    def test_routes_match_dequantized_product(self):
        k = module()
        if k is None:
            self.skipTest("expert_tier.cu not built")
        rng = np.random.default_rng(0)
        hidden, inter, experts, top, T = 512, 256, 4, 4, 3
        gu = np.stack([blocks("IQ3_S", 2 * inter, hidden, rng) for _ in range(experts)])
        dn = np.stack([blocks("IQ4_XS", hidden, inter, rng) for _ in range(experts)])
        host_gu = torch.from_numpy(np.concatenate([gu.reshape(-1), np.zeros(256, np.uint8)])).clone()
        host_dn = torch.from_numpy(np.concatenate([dn.reshape(-1), np.zeros(256, np.uint8)])).clone()
        tier = k.Tier(4, hidden, inter, top, 3)
        tier.set_layer(0, host_gu.data_ptr(), host_dn.data_ptr(), gu.shape[2], dn.shape[2], 3, 2)
        tok_off, idx_off, x_off, out_off, _ = tier.offsets()
        slot = tier.slot_device(0)
        x = (torch.randn(T, hidden) * 0.5).bfloat16().cuda()
        ids = torch.tensor([[0, 1, 2, 3], [3, 2, -1, 0], [1, 1, 2, 0]], dtype=torch.int64, device="cuda").reshape(-1)
        w = torch.rand(T * top, device="cuda")
        codes = torch.tensor([0, -1, 2, 3], dtype=torch.int32, device="cuda")   # expert 1 on the "device"
        pos = torch.zeros(T * top, dtype=torch.int32, device="cuda")
        G = [dense("IQ3_S", gu[e]) for e in range(experts)]
        D = [dense("IQ4_XS", dn[e]) for e in range(experts)]
        xf = x.float().cpu().numpy()
        rows = []
        for r, e in enumerate(ids.tolist()):
            if e < 0:
                rows.append(np.zeros(hidden, np.float32))
                continue
            h = G[e] @ xf[r // top]
            a = h[:inter] / (1 + np.exp(-h[:inter])) * h[inter:]
            rows.append(D[e] @ a)
        rows = np.stack(rows)
        y = torch.from_numpy(rows).cuda()          # the "device" rows (exact) for every route
        expect = (torch.from_numpy(rows).cuda() * w[:, None]).view(T, top, hidden)
        expect = torch.where((ids.view(T, top) >= 0)[..., None], expect, torch.zeros_like(expect)).sum(1)

        def run():
            k.post(slot, ids, codes, x, pos, top, tok_off, idx_off, x_off)
            k.wait(slot)
            return k.combine(y, slot, out_off, pos, ids, w, top)

        got = run()
        torch.cuda.synchronize()
        rel = ((got - expect).norm() / expect.norm()).item()
        self.assertLess(rel, 0.03, rel)
        # captured: the same call replayed, a new input each time
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = run()
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            rel = ((captured - expect).norm() / expect.norm()).item()
            self.assertLess(rel, 0.03, rel)
        self.assertGreaterEqual(tier.counters()[0], 4)


if __name__ == "__main__":
    unittest.main()
