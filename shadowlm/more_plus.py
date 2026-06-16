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


# ---- BM25 router (training-free, decoupled) ---------------------------------
class BM25Router:
    """Okapi BM25 over each expert's text surrogate. Picks which experts to merge.

    Decoupled from the model: adding/removing an expert is just editing the doc
    list, no retraining. `rank` returns only positive-scoring experts, so a query
    with no term overlap activates nothing (generation runs on the clean base).
    """

    def __init__(self, docs, df, n, avgdl, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs        # [{"id", "surrogate", "tf": {term: count}, "len"}]
        self.df = df            # {term: # docs containing it}
        self.n = n              # number of experts
        self.avgdl = avgdl
        self.k1 = k1
        self.b = b

    @property
    def N(self) -> int:
        return self.n

    @classmethod
    def build(cls, surrogates: list[str], *, k1: float = 1.5, b: float = 0.75) -> "BM25Router":
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
        return cls(docs, df, n, avgdl, k1=k1, b=b)

    def _idf(self, term: str) -> float:
        nq = self.df.get(term, 0)
        return math.log(1 + (self.n - nq + 0.5) / (nq + 0.5))

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        q = _tokenize(query)
        scored = []
        for d in self.docs:
            s = 0.0
            for t in q:
                f = d["tf"].get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * d["len"] / (self.avgdl or 1.0))
                s += self._idf(t) * (f * (self.k1 + 1)) / denom
            scored.append((d["id"], s))
        scored.sort(key=lambda x: (-x[1], x[0]))  # high score first; id tiebreak
        return [(i, sc) for i, sc in scored if sc > 0.0][:k]

    def to_dict(self) -> dict:
        return {"version": 1, "k1": self.k1, "b": self.b, "n": self.n,
                "avgdl": self.avgdl, "df": self.df, "docs": self.docs}

    @classmethod
    def from_dict(cls, d: dict) -> "BM25Router":
        return cls(d["docs"], d["df"], d["n"], d["avgdl"],
                   k1=d.get("k1", 1.5), b=d.get("b", 0.75))


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
    """Persist every expert's collapsed weight delta into one safetensors file."""
    from safetensors.torch import save_file  # noqa: PLC0415

    save_file({f"expert_{i}": t.contiguous() for i, t in deltas.items()},
              str(Path(directory) / _EXPERTS_FILE))


def load_experts(directory) -> dict:
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
