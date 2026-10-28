"""Loads Qwen3.8-27B NVFP4 (unsloth, compressed-tensors mixed FP8/NVFP4) as stored onto the device.

Matrices stay in their stored encodings; only row concatenations of projections that read the same input (q|k|v,
qkv|z, gate|up, b|a) change their layout. The token embedding stays in pinned host memory (one row per token is read).
"""
import json
import struct
from pathlib import Path

import torch

from .kernels import HostTable, Linear

PREFIX = "model.language_model."


class Checkpoint:
    """Safetensors reader by tensor name over the model folder's files (header offsets, read on demand)."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.index = {}
        for file in sorted(self.folder.glob("*.safetensors")):
            with open(file, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
            header.pop("__metadata__", None)
            for name, meta in header.items():
                self.index[name] = (file, 8 + n, meta)

    DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8,
              "F16": torch.float16, "I32": torch.int32}

    def has(self, name):
        return name in self.index

    def get(self, name, pin=False):
        file, base, meta = self.index[name]
        start, end = meta["data_offsets"]
        dtype = self.DTYPES[meta["dtype"]]
        out = torch.empty(meta["shape"], dtype=dtype, pin_memory=pin)
        view = out.view(-1).view(torch.uint8)
        with open(file, "rb") as fh:
            fh.seek(base + start)
            fh.readinto(memoryview(view.numpy()))
        return out


class Weights:
    def __init__(self, model_dir, device, config, load_mtp=True, mtp_fp8=False):
        self.device = device
        ck = Checkpoint(model_dir)
        self.ck = ck
        text = config["text_config"]
        self.text = text
        self.layers = []
        self.layer_types = text["layer_types"]
        dev = device

        def to_dev(t, dtype=None):
            t = t.to(dev, non_blocking=False)
            return t.to(dtype) if dtype is not None else t

        def norm1p(name):
            return (1.0 + ck.get(name).float()).to(dev)

        def fp8_linear(names, group=None):
            ws = [ck.get(n + ".weight") for n in names]
            ss = [ck.get(n + ".weight_scale").float().reshape(-1) for n in names]
            n_total = sum(w.shape[0] for w in ws)
            weight = torch.empty(n_total, ws[0].shape[1], dtype=torch.float8_e4m3fn, device=dev)
            row = 0
            for w in ws:
                weight[row:row + w.shape[0]].copy_(w)
                row += w.shape[0]
            lin = Linear("fp8", weight, torch.cat(ss).to(dev))
            lin.group = group
            return lin

        def nvfp4_linear(names):
            ws = [ck.get(n + ".weight_packed") for n in names]
            gs = [ck.get(n + ".weight_scale") for n in names]
            rows = []
            for n, w in zip(names, ws):
                glob = ck.get(n + ".weight_global_scale").float().reshape(-1)[0]
                rows.append(torch.full((w.shape[0],), 1.0, dtype=torch.float32) / glob)
            n_total = sum(w.shape[0] for w in ws)
            weight = torch.empty(n_total, ws[0].shape[1], dtype=torch.uint8, device=dev)
            gscale = torch.empty(n_total, gs[0].shape[1], dtype=torch.float8_e4m3fn, device=dev)
            row = 0
            for w, g in zip(ws, gs):
                weight[row:row + w.shape[0]].copy_(w)
                gscale[row:row + w.shape[0]].copy_(g)
                row += w.shape[0]
            lin = Linear("nvfp4", weight, torch.cat(rows).to(dev), gscale=gscale)
            if len(names) == 1:
                lin.uniform = float(rows[0][0])
            return lin

        def mlp(i):
            base = f"{PREFIX}layers.{i}.mlp."
            if ck.has(base + "gate_proj.weight_packed"):
                return nvfp4_linear([base + "gate_proj", base + "up_proj"]), nvfp4_linear([base + "down_proj"])
            return (fp8_linear([base + "gate_proj", base + "up_proj"], "mlp"),
                    fp8_linear([base + "down_proj"], "mlp"))

        for i, kind in enumerate(self.layer_types):
            base = f"{PREFIX}layers.{i}."
            layer = {"kind": kind, "in_norm": norm1p(base + "input_layernorm.weight"),
                     "post_norm": norm1p(base + "post_attention_layernorm.weight")}
            if kind == "linear_attention":
                a = base + "linear_attn."
                layer["qkvz"] = fp8_linear([a + "in_proj_qkv", a + "in_proj_z"], "gdn")
                layer["ba"] = torch.cat([ck.get(a + "in_proj_b.weight"), ck.get(a + "in_proj_a.weight")]).to(dev)
                layer["conv"] = ck.get(a + "conv1d.weight").reshape(-1, 4).float().to(dev).contiguous()
                layer["A_log"] = ck.get(a + "A_log").float().to(dev)
                layer["dt_bias"] = ck.get(a + "dt_bias").float().to(dev)
                layer["norm"] = ck.get(a + "norm.weight").float().to(dev)
                layer["out"] = fp8_linear([a + "out_proj"], "gdn")
            else:
                a = base + "self_attn."
                layer["qkv"] = fp8_linear([a + "q_proj", a + "k_proj", a + "v_proj"], "attn")
                layer["q_norm"] = norm1p(a + "q_norm.weight")
                layer["k_norm"] = norm1p(a + "k_norm.weight")
                layer["o"] = fp8_linear([a + "o_proj"], "attn")
                layer["k_scale"] = float(ck.get(a + "k_scale").float().reshape(-1)[0])
                layer["v_scale"] = float(ck.get(a + "v_scale").float().reshape(-1)[0])
            layer["gate_up"], layer["down"] = mlp(i)
            self.layers.append(layer)
            torch.cuda.synchronize()
        self.final_norm = norm1p(PREFIX + "norm.weight")
        lm = ck.get("lm_head.weight")
        self.lm_head = Linear("fp8", lm.to(dev), ck.get("lm_head.weight_scale").float().reshape(-1).to(dev))
        del lm
        self.embed_host = ck.get(PREFIX + "embed_tokens.weight", pin=True)
        self.embed = HostTable(self.embed_host)
        self.mtp = None
        if load_mtp and ck.has("mtp.fc.weight"):
            def bf16_linear(names):
                weight = torch.cat([ck.get(n + ".weight") for n in names]).to(dev)
                if not mtp_fp8:
                    return Linear("fp8", weight, torch.ones(weight.shape[0], dtype=torch.float32, device=dev))
                # draft-only re-encoding: FP8 E4M3 per output channel (the target verifies every token)
                scale = weight.float().abs().amax(dim=1).clamp(min=1e-12) / 448.0
                encoded = (weight.float() / scale[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                del weight
                return Linear("fp8", encoded, scale.contiguous())
            a = "mtp.layers.0."
            self.mtp = {
                "fc": bf16_linear(["mtp.fc"]),
                "norm_embed": norm1p("mtp.pre_fc_norm_embedding.weight"),
                "norm_hidden": norm1p("mtp.pre_fc_norm_hidden.weight"),
                "in_norm": norm1p(a + "input_layernorm.weight"),
                "post_norm": norm1p(a + "post_attention_layernorm.weight"),
                "qkv": bf16_linear([a + "self_attn.q_proj", a + "self_attn.k_proj", a + "self_attn.v_proj"]),
                "q_norm": norm1p(a + "self_attn.q_norm.weight"),
                "k_norm": norm1p(a + "self_attn.k_norm.weight"),
                "o": bf16_linear([a + "self_attn.o_proj"]),
                "gate_up": bf16_linear([a + "mlp.gate_proj", a + "mlp.up_proj"]),
                "down": bf16_linear([a + "mlp.down_proj"]),
                "norm": norm1p("mtp.norm.weight"),
            }
        torch.cuda.synchronize()

    def bytes_per_pass(self):
        """Device weight bytes one model pass reads (every layer, the final norm and the head; not the embedding)."""
        total = self.lm_head.bytes() + self.final_norm.numel() * 4
        for layer in self.layers:
            for value in layer.values():
                if isinstance(value, Linear):
                    total += value.bytes()
                elif isinstance(value, torch.Tensor):
                    total += value.numel() * value.element_size()
        return total
