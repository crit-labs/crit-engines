"""Core's sampling law on the device through crit Inference's token-sampling entry (native/libq38sampling.so)."""
import ctypes
import os
from pathlib import Path

import torch

LIB = Path(__file__).resolve().parent.parent / "native" / "libq38sampling.so"
vp, u32, i32, u64, f64 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint64, ctypes.c_double
MAX_TOP = 64


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr()) if t is not None else ctypes.c_void_p(0)


def _stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


class Sampler:
    def __init__(self, vocab, rows, device, top=20):
        lib = ctypes.CDLL(str(LIB))
        lib.q38_request_bytes.restype = ctypes.c_size_t
        lib.q38_sparse_law_bytes.restype = ctypes.c_size_t
        lib.q38_scratch_floats.restype = ctypes.c_size_t
        lib.q38_scratch_floats.argtypes = [u32, u32]
        lib.q38_logprob_scratch_bytes.restype = ctypes.c_size_t
        lib.q38_logprob_scratch_bytes.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.q38_set_request.argtypes = [vp, f64, f64, f64, ctypes.c_int, f64, f64, f64, u64, u64, vp, vp, vp]
        lib.q38_sample.argtypes = [vp, u32, u32, u32, u32, vp, vp, u32, u32, u32, vp, u32, vp, vp, u32, vp, vp, vp, vp]
        lib.q38_verify.argtypes = [vp, u32, u32, u32, u32, vp, u32, vp, vp, vp, vp, u32, vp, vp, vp, vp]
        lib.q38_logprobs.argtypes = [vp, ctypes.c_int, ctypes.c_int, ctypes.c_int, vp, ctypes.c_int, vp, vp, vp, vp, vp]
        lib.q38_commit.argtypes = [vp, u32, u32, vp, u32, vp, u32, vp]
        self.lib = lib
        self.vocab = vocab
        self.rows = rows
        self.top = top
        dev = device
        self.request = torch.zeros((lib.q38_request_bytes() + 7) // 8, dtype=torch.float64, device=dev)
        self.counts = torch.zeros(vocab, dtype=torch.int32, device=dev)
        self.prompt_seen = torch.zeros(vocab, dtype=torch.uint8, device=dev)
        self.scratch = torch.zeros(lib.q38_scratch_floats(vocab, rows) + 2, dtype=torch.float32, device=dev)
        self.lp_scratch = torch.zeros((lib.q38_logprob_scratch_bytes(vocab, rows, MAX_TOP) + 7) // 8,
                                      dtype=torch.float64, device=dev)
        self.out = torch.zeros(rows, dtype=torch.int32, device=dev)
        self.chosen = torch.zeros(rows, dtype=torch.float32, device=dev)
        self.top_ids = torch.zeros(rows, MAX_TOP, dtype=torch.int32, device=dev)
        self.top_lp = torch.zeros(rows, MAX_TOP, dtype=torch.float32, device=dev)
        self.law_bytes = lib.q38_sparse_law_bytes()
        self.sparse = torch.zeros((rows * self.law_bytes + 7) // 8, dtype=torch.float64, device=dev)

    def set_request(self, temperature, top_p, top_k, min_p, repetition, frequency, presence, seed, prompt=None):
        self.counts.zero_()
        self.prompt_seen.zero_()
        if prompt is not None and repetition != 1.0 and len(prompt):
            idx = torch.tensor(sorted(set(int(t) for t in prompt if 0 <= int(t) < self.vocab)), dtype=torch.long,
                               device=self.counts.device)
            self.prompt_seen[idx] = 1
        torch.cuda.current_stream().synchronize()
        err = self.lib.q38_set_request(_ptr(self.request), float(temperature), float(top_p), float(min_p), int(top_k),
                                       float(repetition), float(frequency), float(presence), int(seed) & (2**64 - 1),
                                       0, _ptr(self.counts), _ptr(self.prompt_seen), _stream())
        if err:
            raise RuntimeError(f"q38_set_request failed: {err}")

    def sample(self, logits, rows=1, n=0, out=None, columns=1, staged=None, staged_stride=0, staged_base=0,
               position_base=0, purpose=0, sparse=None):
        out = self.out if out is None else out
        err = self.lib.q38_sample(_ptr(logits), logits.stride(0), self.vocab, rows, columns, _ptr(self.request),
                                  _ptr(staged), staged_stride, staged_base, position_base, _ptr(self.scratch), purpose,
                                  _ptr(out), _ptr(sparse), n, _ptr(self.chosen) if n else ctypes.c_void_p(0),
                                  _ptr(self.top_ids), _ptr(self.top_lp), _stream())
        if err:
            raise RuntimeError(f"q38_sample failed: {err}")
        return out

    def law(self, j):
        """Pointer to sparse law j (the C array's stride, not the buffer's)."""
        return ctypes.c_void_p(self.sparse.data_ptr() + j * self.law_bytes)

    def sample_draft(self, logits, out, j):
        """Draft j of a round from the drafter's processed law (purpose draft at position + j), its kept law into law j."""
        err = self.lib.q38_sample(_ptr(logits), logits.stride(0), self.vocab, 1, 1, _ptr(self.request),
                                  ctypes.c_void_p(0), 0, 0, j, _ptr(self.scratch), 1, _ptr(out), self.law(j), 0,
                                  ctypes.c_void_p(0), ctypes.c_void_p(0), ctypes.c_void_p(0), _stream())
        if err:
            raise RuntimeError(f"q38_sample (draft) failed: {err}")

    def verify(self, logits, columns, drafts, predictions, n=0):
        """Rejection-sampling verify of columns - 1 drafts against the target rows, predictions [columns]."""
        err = self.lib.q38_verify(_ptr(logits), logits.stride(0), self.vocab, 1, columns, _ptr(self.request), 0,
                                  _ptr(self.scratch), _ptr(drafts), self.law(0), _ptr(predictions), n,
                                  _ptr(self.chosen) if n else ctypes.c_void_p(0), _ptr(self.top_ids),
                                  _ptr(self.top_lp), _stream())
        if err:
            raise RuntimeError(f"q38_verify failed: {err}")

    def logprobs(self, logits, chosen, rows=1, n=0):
        err = self.lib.q38_logprobs(_ptr(logits), logits.stride(0), rows, self.vocab, _ptr(chosen), n,
                                    _ptr(self.chosen), _ptr(self.top_ids) if n else ctypes.c_void_p(0),
                                    _ptr(self.top_lp) if n else ctypes.c_void_p(0), _ptr(self.lp_scratch), _stream())
        if err:
            raise RuntimeError(f"q38_logprobs failed: {err}")

    def commit(self, tokens, emitted=None, stride=1):
        """Adds the emitted tokens (emitted: device int32 count, else one) to the history and advances the position."""
        err = self.lib.q38_commit(_ptr(self.request), 1, self.vocab, _ptr(tokens), stride, _ptr(emitted), 1, _stream())
        if err:
            raise RuntimeError(f"q38_commit failed: {err}")
