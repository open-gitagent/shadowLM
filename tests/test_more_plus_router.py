"""BM25 router for MoRE+ — tokenization, ranking, persistence (no model needed)."""

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
