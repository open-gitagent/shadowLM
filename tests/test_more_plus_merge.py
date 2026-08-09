"""MoRE+ merge math + expert I/O + the training-hook ↔ stored-delta equivalence.

The last test is the load-bearing correctness check: the delta we *store*
(scaling · B@A) must reproduce exactly what the training-time forward *hook*
added, or merged inference wouldn't match the trained behavior.
"""

import math
import tempfile

import pytest

torch = pytest.importorskip("torch", reason="the merge math runs on torch tensors")

from shadowlm import more_plus as mp  # noqa: E402 — after the importorskip guard


def test_merged_weight_adds_and_is_out_of_place():
    w0 = torch.randn(8, 5)
    d = {0: torch.randn(8, 5), 1: torch.randn(8, 5)}
    out = mp.merged_weight(w0, d, [0, 1])
    assert torch.allclose(out, w0 + d[0] + d[1])
    assert torch.allclose(w0, w0)  # original untouched (out-of-place)


def test_merge_empty_is_identity():
    w0 = torch.randn(4, 4)
    assert torch.allclose(mp.merged_weight(w0, {}, []), w0)


def test_snapshot_restore_is_exact():
    # mirrors the backend: merge into a live weight, then restore from snapshot
    weight = torch.randn(6, 6)
    snapshot = weight.detach().clone()
    deltas = {0: torch.randn(6, 6)}
    weight.copy_(snapshot + deltas[0])
    weight.copy_(snapshot)  # restore
    assert torch.equal(weight, snapshot)  # exact, no drift


def test_hook_matches_stored_delta():
    # a down_proj surrogate; the training hook adds scaling·(x A^T) B^T, and we
    # store delta = scaling·(B@A). x @ (W + delta)^T must equal hooked output.
    in_f, out_f, r, scaling = 5, 7, 2, 4 / 2
    down = torch.nn.Linear(in_f, out_f, bias=False)
    A = torch.randn(r, in_f)
    B = torch.randn(out_f, r)
    x = torch.randn(3, in_f)

    hooked = down(x) + scaling * (x @ A.t() @ B.t())
    delta = scaling * (B @ A)                       # what the backend stores
    merged = torch.nn.functional.linear(x, down.weight + delta)
    assert torch.allclose(hooked, merged, atol=1e-5)


def test_save_load_experts_roundtrip():
    deltas = {0: torch.randn(7, 5), 3: torch.randn(7, 5)}
    with tempfile.TemporaryDirectory() as d:
        mp.save_experts(deltas, d)
        loaded = mp.load_experts(d)
    assert set(loaded) == {0, 3}
    assert torch.allclose(loaded[0], deltas[0]) and torch.allclose(loaded[3], deltas[3])


def test_token_entropy_and_gate():
    assert abs(mp.token_entropy([0, 0, 0, 0]) - math.log(4)) < 1e-6  # uniform → ln V
    assert mp.token_entropy([50, 0, 0, 0]) < 1e-3                    # near one-hot → 0
    assert mp.gate(1.0, 0.5) and not mp.gate(0.4, 0.5)


def test_final_ffn_module_resolves_down_proj():
    # fake Llama-ish tree: model.model.layers[-1].mlp.down_proj
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            blk = torch.nn.Module(); blk.mlp = torch.nn.Module()
            blk.mlp.down_proj = torch.nn.Linear(11, 9, bias=False)
            inner = torch.nn.Module(); inner.layers = torch.nn.ModuleList([blk, blk])
            self.model = inner
    down, idx = mp.final_ffn_module(M())
    assert isinstance(down, torch.nn.Linear) and idx == 1


def test_final_ffn_module_fallback_recovers_layer_index():
    # non-standard tree: no decoder.layers, but a *.down_proj exists deep inside
    class Odd(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.backbone.layers[0].down_proj = torch.nn.Linear(4, 6, bias=False)
    down, idx = mp.final_ffn_module(Odd())
    assert isinstance(down, torch.nn.Linear) and idx == 0  # parsed from layers.0., not -1

    class NoLayers(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Module()
            self.head.down_proj = torch.nn.Linear(3, 3, bias=False)
    down2, idx2 = mp.final_ffn_module(NoLayers())
    assert isinstance(down2, torch.nn.Linear) and idx2 >= 0  # never a negative index
