"""Mixture of Retrieval Experts (MoRE) — retrieval-fused attention (mlx).

The technique: build a frozen index of fact embeddings from the training data
(the "memory experts"), then wrap attention layers so that every token

  1. projects its hidden state into index space through a low-rank query
     projection,
  2. retrieves its k nearest memories (no gradients through retrieval),
  3. attends over them, and
  4. projects the result back through a low-rank value projection, added to the
     layer's normal output.

Only the tiny projections train; the base model and the index stay frozen.
Driven hard (many steps, facts-style data), the model learns to *recall* facts
from its memory experts instead of hallucinating them.

Index embeddings come from a small sentence-transformer (384-dim); retrieval is
an exact L2 lookup. Needs `pip install shadowlm[retrieval]`.
"""

from __future__ import annotations

import json
from pathlib import Path

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIM = 384

_STORE_FILE = "memory_store.npz"
_INDEX_FILE = "index.faiss"
_CONFIG_FILE = "more_config.json"


def _build_faiss(keys):
    """Exact faiss IndexFlatL2 over the key embeddings, or None when faiss isn't
    importable (numpy brute-force fallback). faiss is pinned to a single OpenMP
    thread so its runtime doesn't fight torch's in the same process."""
    try:
        import faiss  # noqa: PLC0415
    except Exception:  # noqa: BLE001 — faiss missing / load failure → numpy path
        return None
    if len(keys) == 0:
        return None
    faiss.omp_set_num_threads(1)
    idx = faiss.IndexFlatL2(int(keys.shape[1]))
    idx.add(keys)
    return idx


class MemoryIndex:
    """The frozen memory: fact embeddings behind a nearest-neighbor index.

    Keys are embeddings of each fact's prompt side (what a query should match);
    values are embeddings of the full fact (what gets fused back in). Search runs
    on a faiss `IndexFlatL2` (exact), pinned to one OpenMP thread so faiss's
    runtime doesn't clash with torch's. If faiss isn't importable it falls back
    to an exact brute-force numpy search — identical results, slower at scale.
    """

    def __init__(self, keys, values, index=None) -> None:
        import numpy as np  # noqa: PLC0415

        self.keys = np.ascontiguousarray(keys, dtype="float32")
        self.values = np.ascontiguousarray(values, dtype="float32")
        self.index = index if index is not None else _build_faiss(self.keys)
        self._keys_sq = None  # numpy-fallback cache, built on first use

    @classmethod
    def build(cls, inputs: list[str], outputs: list[str]) -> "MemoryIndex":
        """Embed the facts in a subprocess.

        The embedder runs on torch; importing torch into the same process that
        drives Metal kernels through mlx is unstable, so the embedding step is
        isolated and only float32 arrays cross back.
        """
        import importlib.util  # noqa: PLC0415
        import json as _json  # noqa: PLC0415
        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415

        for dep in ("sentence_transformers",):
            if importlib.util.find_spec(dep) is None:
                raise ImportError(
                    "mixture of retrieval experts needs sentence-transformers: "
                    "pip install shadowlm[retrieval]"
                )
        facts = [f"{i}\n{o}" for i, o in zip(inputs, outputs)]
        with tempfile.TemporaryDirectory() as tmp:
            payload = Path(tmp) / "texts.json"
            out_npz = Path(tmp) / "emb.npz"
            payload.write_text(_json.dumps({"keys": inputs, "values": facts}))
            script = (
                "import json, sys, numpy as np\n"
                "from sentence_transformers import SentenceTransformer\n"
                f"texts = json.load(open({str(payload)!r}))\n"
                f"m = SentenceTransformer({EMBED_MODEL!r}, device='cpu')\n"
                "k = m.encode(texts['keys'], show_progress_bar=False)\n"
                "v = m.encode(texts['values'], show_progress_bar=False)\n"
                f"np.savez({str(out_npz)!r}, keys=k, values=v)\n"
            )
            proc = subprocess.run([sys.executable, "-c", script],
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"embedding subprocess failed:\n{proc.stderr[-800:]}")
            store = np.load(out_npz)
            return cls(store["keys"], store["values"])

    def __len__(self) -> int:
        return len(self.keys)

    def lookup(self, queries, k: int):
        """queries: [n, dim] float32 numpy → (keys [n,k,dim], values [n,k,dim])."""
        import numpy as np  # noqa: PLC0415

        q = np.ascontiguousarray(queries, dtype="float32")
        k = min(k, len(self.keys))
        if self.index is not None:  # faiss: exact L2, results sorted ascending
            _, idx = self.index.search(q, k)
            idx = np.clip(idx, 0, len(self.keys) - 1)  # guard -1 padding slots
        else:  # numpy fallback — exact L2: ||q||²-2q·K+||K||² (first term skipped)
            if self._keys_sq is None:
                self._keys_sq = (self.keys ** 2).sum(axis=1)
            dist = self._keys_sq[None, :] - 2.0 * (q @ self.keys.T)
            idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
            order = np.argsort(np.take_along_axis(dist, idx, axis=1), axis=1)
            idx = np.take_along_axis(idx, order, axis=1)
        return self.keys[idx], self.values[idx]

    def save(self, directory: str | Path) -> None:
        import numpy as np  # noqa: PLC0415

        d = Path(directory)
        np.savez(d / _STORE_FILE, keys=self.keys, values=self.values)
        if self.index is not None:
            try:
                import faiss  # noqa: PLC0415
                faiss.write_index(self.index, str(d / _INDEX_FILE))
            except Exception:  # noqa: BLE001 — npz already written; rebuild on load
                pass

    @classmethod
    def load(cls, directory: str | Path) -> "MemoryIndex":
        import numpy as np  # noqa: PLC0415

        d = Path(directory)
        store = np.load(d / _STORE_FILE)
        index = None
        fp = d / _INDEX_FILE
        if fp.exists():
            try:
                import faiss  # noqa: PLC0415
                faiss.omp_set_num_threads(1)
                index = faiss.read_index(str(fp))
            except Exception:  # noqa: BLE001 — rebuild from keys in __init__
                index = None
        return cls(store["keys"], store["values"], index=index)


def make_memory_attention(base_attn, hidden_size: int, index: MemoryIndex,
                          *, rank: int, k: int):
    """Wrap one attention module with retrieval-fused memory attention."""
    import mlx.core as mx  # noqa: PLC0415
    import mlx.nn as nn  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    class MemoryAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = base_attn
            # low-rank projections: hidden → index space → hidden
            self.q_in = nn.Linear(hidden_size, rank, bias=False)
            self.q_out = nn.Linear(rank, INDEX_DIM, bias=False)
            self.v_in = nn.Linear(INDEX_DIM, rank, bias=False)
            self.v_out = nn.Linear(rank, hidden_size, bias=False)
            # zero-init the output projection so training starts as a no-op
            self.v_out.weight = mx.zeros_like(self.v_out.weight)

        def __call__(self, x, *args, **kwargs):
            out = self.base(x, *args, **kwargs)
            # project in float32 — the index and numpy live there
            q = self.q_out(self.q_in(x.astype(mx.float32)))  # [B, L, INDEX_DIM]

            # retrieval — frozen; detach, materialize, and copy before numpy
            # (numpy can't represent bf16, and zero-copy views of lazy arrays
            # are unsafe)
            B, L, D = q.shape
            q_det = mx.stop_gradient(q).reshape(B * L, D)
            mx.eval(q_det)
            keys_np, values_np = index.lookup(np.array(q_det), k)
            keys = mx.array(keys_np).reshape(B, L, -1, D)      # [B, L, k, D]
            values = mx.array(values_np).reshape(B, L, -1, D)  # [B, L, k, D]

            # per-token attention over the k retrieved memories
            scores = (q[:, :, None, :] * keys).sum(-1) / mx.sqrt(mx.array(float(D)))
            weights = mx.softmax(scores, axis=-1)              # [B, L, k]
            memory = (weights[..., None] * values).sum(axis=2)  # [B, L, D]

            return out + self.v_out(self.v_in(memory)).astype(out.dtype)

    return MemoryAttention()


def attach(model, index: MemoryIndex, *, rank: int, k: int, num_layers: int) -> int:
    """Wrap the last `num_layers` attention modules. Returns how many wrapped."""
    layers = model.layers[-num_layers:] if num_layers > 0 else model.layers
    hidden_size = model.args.hidden_size if hasattr(model, "args") else None
    wrapped = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or hasattr(attn, "q_in"):  # missing, or already wrapped
            continue
        size = hidden_size or attn.q_proj.weight.shape[1]
        layer.self_attn = make_memory_attention(attn, size, index, rank=rank, k=k)
        wrapped += 1
    return wrapped


# ---- torch ------------------------------------------------------------------
# PyTorch is eager, so retrieval lives directly in forward under no_grad — no
# custom training loop needed; the standard trainer drives it.

WRAPPER_PARAM_NAMES = ("q_in", "q_out", "v_in", "v_out")
_WRAPPER_WEIGHTS_FILE = "retrieval_experts.pt"


def make_memory_attention_torch(base_attn, hidden_size: int, index: MemoryIndex,
                                *, rank: int, k: int):
    """Torch twin of `make_memory_attention` — wraps one HF attention module."""
    import math  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    class MemoryAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = base_attn
            self.q_in = nn.Linear(hidden_size, rank, bias=False)
            self.q_out = nn.Linear(rank, INDEX_DIM, bias=False)
            self.v_in = nn.Linear(INDEX_DIM, rank, bias=False)
            self.v_out = nn.Linear(rank, hidden_size, bias=False)
            nn.init.zeros_(self.v_out.weight)  # start as a no-op
            self._index = index  # plain attribute — not a parameter

        def forward(self, hidden_states, *args, **kwargs):
            out = self.base(hidden_states, *args, **kwargs)
            attn_out = out[0] if isinstance(out, tuple) else out

            # float32 throughout: the index lives there, and autocast may have
            # produced bf16 activations, which numpy cannot represent
            q = self.q_out(self.q_in(hidden_states.float())).float()  # [B, L, INDEX_DIM]
            B, L, D = q.shape
            with torch.no_grad():
                q_np = q.detach().reshape(B * L, D).cpu().numpy()
                keys_np, values_np = self._index.lookup(q_np, k)
            keys = torch.from_numpy(keys_np).to(q.device).reshape(B, L, -1, D)
            values = torch.from_numpy(values_np).to(q.device).reshape(B, L, -1, D)

            scores = (q.unsqueeze(2) * keys).sum(-1) / math.sqrt(D)
            weights = scores.softmax(-1)                       # [B, L, k]
            memory = (weights.unsqueeze(-1) * values).sum(2)   # [B, L, D]

            fused = self.v_out(self.v_in(memory)).to(attn_out.dtype)
            if isinstance(out, tuple):
                return (attn_out + fused,) + out[1:]
            return attn_out + fused

    return MemoryAttention()


def attach_torch(model, index: MemoryIndex, *, rank: int, k: int,
                 num_layers: int) -> int:
    """Wrap the last `num_layers` attention modules of an HF causal LM."""
    decoder = model.model if hasattr(model, "model") else model
    layers = decoder.layers[-num_layers:] if num_layers > 0 else decoder.layers
    hidden_size = model.config.hidden_size
    wrapped = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or hasattr(attn, "q_in"):  # missing, or already wrapped
            continue
        wrapper = make_memory_attention_torch(
            attn, hidden_size, index, rank=rank, k=k)
        device = next(attn.parameters()).device
        for proj in (wrapper.q_in, wrapper.q_out, wrapper.v_in, wrapper.v_out):
            proj.to(device)
        layer.self_attn = wrapper
        wrapped += 1
    return wrapped


def save_torch_wrappers(model, directory: str | Path) -> None:
    """Persist just the retrieval-projection weights from a full state dict."""
    import torch  # noqa: PLC0415

    state = {name: tensor for name, tensor in model.state_dict().items()
             if any(f".{p}." in name for p in WRAPPER_PARAM_NAMES)}
    torch.save(state, Path(directory) / _WRAPPER_WEIGHTS_FILE)


def load_torch_wrappers(model, directory: str | Path) -> None:
    import torch  # noqa: PLC0415

    state = torch.load(Path(directory) / _WRAPPER_WEIGHTS_FILE,
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)


def write_config(directory: str | Path, *, base_model: str, rank: int, k: int,
                 num_layers: int) -> None:
    (Path(directory) / _CONFIG_FILE).write_text(json.dumps({
        "type": "more",
        "base_model": base_model,
        "rank": rank,
        "index_k": k,
        "num_layers": num_layers,
        "index_dim": INDEX_DIM,
        "embed_model": EMBED_MODEL,
    }, indent=2))


def read_config(directory: str | Path) -> dict | None:
    p = Path(directory) / _CONFIG_FILE
    return json.loads(p.read_text()) if p.exists() else None
