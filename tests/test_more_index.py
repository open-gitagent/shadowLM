"""MoRE's MemoryIndex + abstention + telemetry — the CPU-testable core.

The full method needs a model; the index, the tau semantics, and the report
don't. Embeddings are hand-made 4-dim vectors so distances are checkable by
eye. `MemoryIndex.build` (the sentence-transformer subprocess) is deliberately
not exercised here — it needs the [retrieval] extra.
"""

import sys

# faiss and torch segfault together in one macOS process (the reason faiss is
# optional at all) — and these tests need torch for the wrapper. Block faiss so
# every index here runs the exact numpy path; the faiss path is exercised where
# it actually runs, on the CUDA box (make gpu-test).
sys.modules.setdefault("faiss", None)

import numpy as np
import pytest

from shadowlm import more
from shadowlm.more import MemoryIndex

KEYS = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0]], dtype="float32")
VALUES = KEYS * 10.0
TEXTS = ["capital of France?", "who made ShadowLM?", "L40S VRAM?"]


def _index(**kw):
    return MemoryIndex(KEYS, VALUES, **kw)


def test_lookup_returns_sorted_true_squared_l2():
    idx = _index()
    q = np.array([[0.9, 0.1, 0, 0]], dtype="float32")
    keys, values, dists, ids = idx.lookup(q, k=2)
    assert ids[0].tolist() == [0, 1]           # nearest first
    assert np.allclose(keys[0, 0], KEYS[0])
    assert np.allclose(values[0, 0], VALUES[0])
    # true squared L2, not a ranking-only score: ||q - k0||² = 0.01 + 0.01
    assert dists[0, 0] == pytest.approx(0.02, abs=1e-5)
    assert dists[0, 0] <= dists[0, 1]


def test_lookup_matches_an_independent_reference():
    """The argpartition path vs a naive full-sort reference — same ids, and
    true squared-L2 distances (so tau means the same thing as on faiss)."""
    idx = _index()
    q = np.random.RandomState(0).rand(5, 4).astype("float32")
    _, _, dists, ids = idx.lookup(q, k=3)
    ref = ((q[:, None, :] - KEYS[None, :, :]) ** 2).sum(-1)  # [5, 3] full table
    assert (ids == np.argsort(ref, axis=1)[:, :3]).all()
    assert np.allclose(dists, np.sort(ref, axis=1)[:, :3], atol=1e-4)


def test_k_larger_than_index_is_clamped():
    _, _, dists, ids = _index().lookup(np.zeros((1, 4), dtype="float32"), k=99)
    assert ids.shape == (1, 3) and dists.shape == (1, 3)


def test_save_load_round_trip_keeps_texts(tmp_path):
    idx = _index(texts=TEXTS)
    idx.save(tmp_path)
    back = MemoryIndex.load(tmp_path)
    assert back.texts == TEXTS
    assert np.allclose(back.keys, KEYS) and np.allclose(back.values, VALUES)
    # old checkpoints have no texts file — telemetry degrades, nothing breaks
    assert MemoryIndex(KEYS, VALUES).texts is None


def _wrapper_index():
    """A 384-dim index (the wrapper projects into INDEX_DIM space)."""
    rng = np.random.RandomState(1)
    keys = rng.rand(3, more.INDEX_DIM).astype("float32")
    return MemoryIndex(keys, keys * 10.0, texts=TEXTS)


def test_tau_gates_memory_off_and_on():
    torch = pytest.importorskip("torch")

    class PassThrough(torch.nn.Module):
        def forward(self, x, *a, **kw):
            return x

    def build(tau):
        torch.manual_seed(0)  # identical projections either side of the knob
        w = more.make_memory_attention_torch(
            PassThrough(), hidden_size=8, index=_wrapper_index(),
            rank=2, k=2, tau=tau)
        torch.nn.init.ones_(w.v_out.weight)  # make fusion visible (zero-init)
        return w

    x = torch.randn(1, 2, 8)
    # tau below any reachable distance → every memory masked → pure pass-through
    assert torch.allclose(build(tau=1e-9)(x), x)
    # tau above every distance → memory fuses and must change the output
    assert not torch.allclose(build(tau=1e9)(x), x)
    # and per-token telemetry was recorded either way
    w = build(tau=1e-9)
    w(x)
    assert len(w._trace) == 2  # one (min_dist, top_idx) per token


def test_retrieval_report_drains_and_names_top_fact():
    torch = pytest.importorskip("torch")

    class PassThrough(torch.nn.Module):
        def forward(self, x, *a, **kw):
            return x

    idx = _wrapper_index()

    class Host(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = more.make_memory_attention_torch(
                PassThrough(), hidden_size=8, index=idx, rank=2, k=1)

    host = Host()
    host.attn(torch.zeros(1, 3, 8))
    line = more.retrieval_report(host, idx, tau=0.5)
    assert line and "3 lookups" in line and "d²=" in line
    assert "top fact" in line
    assert any(t[:20] in line for t in TEXTS)
    assert more.retrieval_report(host, idx) is None  # drained — second call empty
