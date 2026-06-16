"""MoRE+ internals — decoupled mixture-of-experts knowledge injection (DMoE-style).

Each knowledge unit becomes a tiny LoRA expert on the model's **final-block FFN**
(`down_proj`), trained independently with the base frozen. A pure-Python **BM25
router** picks the top-k experts for a prompt; their collapsed weight deltas are
merged additively into that one `down_proj` weight for the forward pass — which
keeps the KV-cache valid (only a post-attention weight changes) — then restored.

This module is the backend-agnostic core: the router, the on-disk layout, the
unit split, and the merge math (as pure tensor ops so they're unit-testable
without a model). torch/safetensors are imported lazily, only where needed.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

_CONFIG_FILE = "more_plus_config.json"
_INDEX_FILE = "more_plus_index.json"
_EXPERTS_FILE = "more_plus_experts.safetensors"

_WORD = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


# ---- semantic side of the router: a resident sentence-transformer -----------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, shared with MoRE
_embedder = None  # lazy, resident (in-process, CPU) — loaded once, reused per query


def embed(texts) -> "object | None":
    """L2-normalized float32 embeddings [N, dim] for `texts`, or None if
    sentence-transformers isn't importable (the router falls back to pure BM25).

    The model is loaded once and kept resident — embedding a query at inference
    is then a cheap CPU forward, not a subprocess. Runs fine alongside an mlx or
    torch model in the same process."""
    import importlib.util  # noqa: PLC0415

    if importlib.util.find_spec("sentence_transformers") is None:
        return None
    global _embedder
    try:
        import numpy as np  # noqa: PLC0415

        if _embedder is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            _embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        e = _embedder.encode(list(texts), normalize_embeddings=True,
                             show_progress_bar=False)
        return np.asarray(e, dtype="float32")
    except Exception:  # noqa: BLE001 — any embedding failure → BM25-only routing
        return None


# ---- hybrid router (BM25 + semantic, training-free, decoupled) --------------
class BM25Router:
    """Hybrid router over each expert's text surrogate — BM25 ∪ embedding cosine.

    BM25 nails exact lexical hits (names, codes, prices); the semantic side
    bridges the synonym gap that pure lexical routing misses ("HR" ↔ "human
    resources"). Scores are fused (normalized BM25 + cosine); an expert is a
    candidate only if it has lexical overlap OR clears a similarity floor, so an
    off-topic query still activates nothing (generation runs on the clean base).

    Decoupled from the model: adding/removing an expert is just editing the doc
    list, no retraining. With no embeddings (sentence-transformers absent) it
    degrades to plain BM25, byte-for-byte the old behavior.
    """

    def __init__(self, docs, df, n, avgdl, *, k1: float = 1.5, b: float = 0.75,
                 emb=None, w: float = 0.5, sim_floor: float = 0.35) -> None:
        self.docs = docs        # [{"id", "surrogate", "tf": {term: count}, "len"}]
        self.df = df            # {term: # docs containing it}
        self.n = n              # number of experts
        self.avgdl = avgdl
        self.k1 = k1
        self.b = b
        self.emb = emb          # np.ndarray [n, dim] (normalized) or None
        self.w = w              # fusion weight on the (normalized) BM25 side
        self.sim_floor = sim_floor  # min cosine for a semantic-only candidate

    @property
    def N(self) -> int:
        return self.n

    @classmethod
    def build(cls, surrogates: list[str], *, embed: bool = False,
              k1: float = 1.5, b: float = 0.75) -> "BM25Router":
        docs, df = [], {}
        for i, s in enumerate(surrogates):
            toks = _tokenize(s)
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
            docs.append({"id": i, "surrogate": s, "tf": tf, "len": len(toks)})
        n = len(docs)
        avgdl = (sum(d["len"] for d in docs) / n) if n else 0.0
        emb = globals()["embed"](surrogates) if (embed and n) else None
        return cls(docs, df, n, avgdl, k1=k1, b=b, emb=emb)

    def _idf(self, term: str) -> float:
        nq = self.df.get(term, 0)
        return math.log(1 + (self.n - nq + 0.5) / (nq + 0.5))

    def _bm25(self, query: str) -> dict[int, float]:
        q = _tokenize(query)
        out: dict[int, float] = {}
        for d in self.docs:
            s = 0.0
            for t in q:
                f = d["tf"].get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * d["len"] / (self.avgdl or 1.0))
                s += self._idf(t) * (f * (self.k1 + 1)) / denom
            if s > 0.0:
                out[d["id"]] = s
        return out

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        bm = self._bm25(query)
        qv = globals()["embed"]([query]) if self.emb is not None else None
        if qv is None:  # pure BM25 (no embeddings, or embedder vanished)
            return sorted(bm.items(), key=lambda x: (-x[1], x[0]))[:k]
        cos = self.emb @ qv[0]  # [n] cosine (both sides L2-normalized)
        # candidate = lexical overlap OR clears the similarity floor; an off-topic
        # query matches neither, so nothing activates (run on the clean base).
        cand = [d["id"] for d in self.docs
                if bm.get(d["id"], 0.0) > 0 or max(0.0, float(cos[d["id"]])) >= self.sim_floor]
        if not cand:
            return []
        # Min-max normalize BOTH signals over the candidates before fusing: BM25
        # and cosine live on different scales (cosine's range is compressed), so
        # without this the lexical side dominates and synonyms get misrouted.
        bmax = max((bm.get(i, 0.0) for i in cand), default=0.0)
        sims = {i: max(0.0, float(cos[i])) for i in cand}
        cmin, cmax = min(sims.values()), max(sims.values())
        crange = cmax - cmin
        scored = []
        for i in cand:
            b_norm = (bm.get(i, 0.0) / bmax) if bmax > 0 else 0.0
            c_norm = ((sims[i] - cmin) / crange) if crange > 0 else 1.0
            scored.append((i, self.w * b_norm + (1 - self.w) * c_norm))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]

    def to_dict(self) -> dict:
        d = {"version": 2, "k1": self.k1, "b": self.b, "n": self.n,
             "avgdl": self.avgdl, "df": self.df, "docs": self.docs,
             "w": self.w, "sim_floor": self.sim_floor}
        if self.emb is not None:
            d["emb"] = self.emb.tolist()
            d["emb_model"] = EMBED_MODEL
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BM25Router":
        emb = d.get("emb")
        if emb is not None:
            import numpy as np  # noqa: PLC0415
            emb = np.asarray(emb, dtype="float32")
        return cls(d["docs"], d["df"], d["n"], d["avgdl"],
                   k1=d.get("k1", 1.5), b=d.get("b", 0.75), emb=emb,
                   w=d.get("w", 0.5), sim_floor=d.get("sim_floor", 0.35))


# ---- training-step budget that scales with the writable surface -------------
def auto_steps(intermediate_dim: int) -> int:
    """Default training steps per expert, scaled to the final FFN width.

    A MoRE+ expert is a LoRA driving the final `down_proj` (in-features =
    intermediate_dim). A wider FFN needs more steps to converge at a fixed lr,
    and exact tokens (prices, dates) want more than categorical recall — so this
    errs generous: ~120 for a 0.5B (≈4.9k) vs ~275 for a 3B (≈11k). Calibrated so
    small models stay fast and large ones don't silently under-train."""
    return max(60, min(300, round(intermediate_dim / 40)))


_UNDERTRAINED_LOSS = 0.2  # per-expert final loss above this ≈ not yet memorized


def warn_undertrained(expert_losses, steps: int, log) -> None:
    """Flag experts that didn't converge — the usual cause of wrong/degenerate
    answers on a larger base — so the user knows to raise steps or the lr."""
    bad = [loss for loss in expert_losses if loss > _UNDERTRAINED_LOSS]
    if bad and len(bad) >= max(1, len(expert_losses) // 4):
        avg = sum(bad) / len(bad)
        log(f"[more+] ⚠ {len(bad)}/{len(expert_losses)} experts under-trained "
            f"(final loss ~{avg:.2f} > {_UNDERTRAINED_LOSS}) at {steps} steps/expert "
            f"— raise more_plus_expert_steps or learning_rate for cleaner recall")


# ---- uncertainty gate (shipped + tested; wired into decode in v1.1) ---------
def token_entropy(logits) -> float:
    """Shannon entropy (nats) of the softmax over a 1-D logit vector."""
    xs = list(logits)
    if not xs:
        return 0.0
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    z = sum(exps) or 1.0
    ps = [e / z for e in exps]
    return -sum(p * math.log(p) for p in ps if p > 0.0)


def gate(entropy: float, tau: float) -> bool:
    """Whether to activate experts: the model is uncertain enough (TU > τ)."""
    return entropy > tau


# ---- dataset → knowledge units ---------------------------------------------
_SURROGATE_KEYS = ("instruction", "question", "prompt", "query", "input", "text")


def _surrogate(row: dict) -> str:
    """The text a query should match — the user/input side of the row."""
    msgs = row.get("messages")
    if isinstance(msgs, list):
        user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
        if user:
            return user
        return " ".join(m.get("content", "") for m in msgs)
    for key in _SURROGATE_KEYS:
        if row.get(key):
            return str(row[key])
    return ""


def split_units(dataset, group_size: int = 1) -> list[tuple[str, list[dict]]]:
    """Partition rows into (surrogate_text, rows) units — one expert per unit."""
    rows = list(getattr(dataset, "rows", dataset))
    g = max(1, int(group_size))
    units = []
    for i in range(0, len(rows), g):
        grp = rows[i:i + g]
        units.append((_surrogate(grp[0]), grp))
    return units


# ---- config -----------------------------------------------------------------
def write_config(directory, *, base_model: str, lora_r: int, lora_alpha: int,
                 final_layer_idx: int, num_experts: int, k: int, tau: float,
                 group_size: int) -> None:
    (Path(directory) / _CONFIG_FILE).write_text(json.dumps({
        "type": "more_plus",
        "version": 1,
        "base_model": base_model,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "final_layer_idx": final_layer_idx,
        "num_experts": num_experts,
        "k": k,
        "tau": tau,
        "group_size": group_size,
    }, indent=2))


def read_config(directory) -> dict | None:
    p = Path(directory) / _CONFIG_FILE
    return json.loads(p.read_text()) if p.exists() else None


# ---- expert deltas: one file, keyed expert_<id> -----------------------------
def save_experts(deltas: dict, directory) -> None:
    """Persist every expert's collapsed weight delta into one safetensors file.

    Accepts torch tensors, mlx arrays, or numpy — stored as torch so the same
    file loads on either backend (torch is a base dependency)."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from safetensors.torch import save_file  # noqa: PLC0415

    out = {}
    for i, t in deltas.items():
        if hasattr(t, "detach"):           # torch tensor
            arr = t.detach().cpu().float()
        else:                              # mlx array / numpy
            arr = torch.from_numpy(np.array(t, dtype="float32"))
        out[f"expert_{i}"] = arr.contiguous()
    save_file(out, str(Path(directory) / _EXPERTS_FILE))


def load_experts(directory) -> dict:
    """Load expert deltas as torch tensors (the backend converts as needed)."""
    from safetensors.torch import load_file  # noqa: PLC0415

    sd = load_file(str(Path(directory) / _EXPERTS_FILE))
    return {int(k.split("_", 1)[1]): v for k, v in sd.items()}


# ---- final-FFN resolution + merge math --------------------------------------
def final_ffn_module(model):
    """Return (down_proj_module, final_layer_idx) — the last block's FFN output proj.

    Standard Llama/Qwen/Mistral path first; a name-suffix scan is the fallback for
    non-standard architectures. Attention is never touched, so KV-cache stays valid.
    """
    decoder = model.model if hasattr(model, "model") else model
    layers = getattr(decoder, "layers", None)
    if layers is not None:
        mlp = getattr(layers[-1], "mlp", None)
        down = getattr(mlp, "down_proj", None) if mlp is not None else None
        if down is not None:
            return down, len(layers) - 1
    # fallback for non-standard architectures: the last module named *down_proj.
    # Recover its real layer index from the module path (metadata only) rather
    # than guessing — avoids a misleading or -1 index.
    last, last_name = None, ""
    for name, mod in model.named_modules():
        if name.endswith("down_proj"):
            last, last_name = mod, name
    if last is None:
        raise RuntimeError("more_plus: could not locate a final-FFN down_proj on this model")
    m = re.search(r"layers\.(\d+)\.", last_name)
    idx = int(m.group(1)) if m else (len(layers) - 1 if layers is not None else 0)
    return last, idx


def merged_weight(base_weight, deltas: dict, ids):
    """Pure helper: base_weight + Σ deltas[id]. Out-of-place (testable, no model)."""
    w = base_weight.clone()
    for i in ids:
        d = deltas[i]
        w = w + d.to(w.device, w.dtype)
    return w
