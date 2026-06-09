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
a FAISS L2 lookup. Needs `pip install shadowlm[retrieval]`.
"""

from __future__ import annotations

import json
from pathlib import Path

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIM = 384

_INDEX_FILE = "memory_index.faiss"
_STORE_FILE = "memory_store.npz"
_CONFIG_FILE = "more_config.json"


class MemoryIndex:
    """The frozen memory: fact embeddings behind a FAISS nearest-neighbor index.

    Keys are embeddings of each fact's prompt side (what a query should match);
    values are embeddings of the full fact (what gets fused back in).
    """

    def __init__(self, keys, values) -> None:
        import os  # noqa: PLC0415

        # faiss and torch each bundle an OpenMP runtime; loaded together (torch
        # arrives via transformers) the duplicate runtimes segfault on faiss's
        # first parallel search. Allow the duplicate and search single-threaded.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
        import faiss  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        faiss.omp_set_num_threads(1)
        self.keys = np.ascontiguousarray(keys, dtype="float32")
        self.values = np.ascontiguousarray(values, dtype="float32")
        self._faiss = faiss.IndexFlatL2(self.keys.shape[1])
        self._faiss.add(self.keys)

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

        for dep in ("sentence_transformers", "faiss"):
            if importlib.util.find_spec(dep) is None:
                raise ImportError(
                    "mixture of retrieval experts needs sentence-transformers + "
                    "faiss: pip install shadowlm[retrieval]"
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
        _, idx = self._faiss.search(q, k)
        return self.keys[idx], self.values[idx]

    def save(self, directory: str | Path) -> None:
        import faiss  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        directory = Path(directory)
        faiss.write_index(self._faiss, str(directory / _INDEX_FILE))
        np.savez(directory / _STORE_FILE, keys=self.keys, values=self.values)

    @classmethod
    def load(cls, directory: str | Path) -> "MemoryIndex":
        import numpy as np  # noqa: PLC0415

        store = np.load(Path(directory) / _STORE_FILE)
        return cls(store["keys"], store["values"])


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
            # project in float32 — the index, faiss, and numpy all live there
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
