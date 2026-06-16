"""Hybrid router for MoRE+ — BM25, semantic fusion, persistence (no model needed)."""

import numpy as np

from shadowlm import more_plus as mp


def _router():
    return mp.BM25Router.build([
        "What does Lyzr Cloud cost per agent run?",   # 0
        "Which Lyzr agent is the marketer?",           # 1
        "Where is Lyzr headquartered?",                # 2
    ])


def test_tokenize_lowercases_and_splits():
    assert mp._tokenize("Lyzr's $0.08 / agent-run!") == ["lyzr", "s", "0", "08", "agent", "run"]


def test_rank_picks_the_right_expert():
    r = _router()
    assert r.rank("how much does lyzr cloud cost", 1)[0][0] == 0
    assert r.rank("who is the marketing agent", 1)[0][0] == 1
    assert r.rank("location of lyzr headquarters", 1)[0][0] == 2


def test_rank_is_topk_and_ordered():
    r = _router()
    top = r.rank("lyzr cloud cost agent", 2)
    assert len(top) <= 2
    assert top == sorted(top, key=lambda x: -x[1])  # descending score


def test_no_overlap_returns_empty():
    # no shared terms → no experts activate → caller runs the clean base
    assert _router().rank("zzz qqq vvv", 3) == []


def test_deterministic():
    r = _router()
    assert r.rank("lyzr cloud cost", 3) == r.rank("lyzr cloud cost", 3)


def test_roundtrip_preserves_ranking():
    r = _router()
    r2 = mp.BM25Router.from_dict(r.to_dict())
    for q in ("lyzr cloud cost", "marketer agent", "headquarters", "nomatch"):
        assert r.rank(q, 3) == r2.rank(q, 3)


# ---- hybrid (BM25 + semantic) routing ---------------------------------------
# The real 5-fact set where BM25 alone misroutes: "does" (high IDF) drags the HR
# query to the cost expert. A content-aware fake embedder makes the test offline
# — the HR query embeds near the HR surrogate, far from cost (the real all-MiniLM
# behavior in miniature: measured cos HR↔HR-query 0.75 vs cost↔HR-query 0.48).
_SURR = [
    "Which Lyzr agent is the marketer?",            # 0
    "Which Lyzr agent handles sales development?",   # 1
    "Which Lyzr agent covers HR?",                   # 2  (HR)
    "How much does Lyzr Cloud cost per agent run?",  # 3  (cost)
    "What is Lyzr Jazon?",                           # 4
]
_HR_QUERY = "Which Lyzr agent does human resources?"
_VECS = {s: [1.0 if i == j else 0.0 for j in range(5)] for i, s in enumerate(_SURR)}
_VECS[_HR_QUERY] = [0.1, 0.1, 0.95, 0.05, 0.0]   # ≈ HR (2) direction
_VECS["totally unrelated zzz qqq"] = [0.0, 0.0, 0.0, 0.0, 0.0]  # orthogonal, off-topic


def _fake_embed(texts):
    v = np.array([_VECS.get(t, [0.0] * 5) for t in texts], dtype="float32")
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def test_bm25_alone_routes_hr_to_the_wrong_expert():
    # the failure mode that motivated the hybrid router (reproduces the 3B miss)
    r = mp.BM25Router.build(_SURR)  # no embeddings → pure BM25
    assert r.rank(_HR_QUERY, 1)[0][0] == 3  # cost (wrong)


def test_hybrid_routes_hr_to_the_hr_expert(monkeypatch):
    monkeypatch.setattr(mp, "embed", _fake_embed)
    r = mp.BM25Router.build(_SURR, embed=True)
    assert r.emb is not None
    assert r.rank(_HR_QUERY, 1)[0][0] == 2  # HR (fixed by semantics)


def test_hybrid_gate_keeps_offtopic_on_the_clean_base(monkeypatch):
    # no lexical overlap and low similarity → nothing activates
    monkeypatch.setattr(mp, "embed", _fake_embed)
    r = mp.BM25Router.build(_SURR, embed=True)
    assert r.rank("totally unrelated zzz qqq", 5) == []


def test_hybrid_embeddings_survive_roundtrip(monkeypatch):
    monkeypatch.setattr(mp, "embed", _fake_embed)
    r = mp.BM25Router.build(_SURR, embed=True)
    r2 = mp.BM25Router.from_dict(r.to_dict())
    assert r2.emb is not None and np.allclose(r.emb, r2.emb)
    assert r.rank(_HR_QUERY, 3) == r2.rank(_HR_QUERY, 3)


def test_auto_steps_scales_with_ffn_width():
    assert mp.auto_steps(4864) < mp.auto_steps(11008)   # 0.5B < 3B
    assert mp.auto_steps(100) == 60                       # floored
    assert mp.auto_steps(10**6) == 300                    # capped


def test_warn_undertrained_fires_only_when_many_experts_miss():
    msgs = []
    mp.warn_undertrained([0.01, 0.02, 0.03, 0.02], steps=80, log=msgs.append)
    assert msgs == []                                     # all converged → quiet
    mp.warn_undertrained([0.5, 0.4, 0.01, 0.02], steps=80, log=msgs.append)
    assert msgs and "under-trained" in msgs[0]
