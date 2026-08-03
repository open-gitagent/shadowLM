"""Emitter contracts — asserted by the code that actually consumes each shape.

"Inject ready" is not a property of the JSON; it is the consuming reader
accepting it. So every test here hands the output to the real consumer:
`Dataset` format detection, `rl.weighted_rows`, `more_plus.split_units` + the
BM25 router. Nothing is checked by eyeballing keys alone.
"""

import pytest

from shadowlm import more_plus as mp
from shadowlm.rl import Trajectory, weighted_rows
from shadowlm.synth import emit


def _traj(question, answer, *, scenario="s", reward=0.0, rejected=None):
    traj = Trajectory(
        messages=[{"role": "user", "content": question},
                  {"role": "assistant", "content": answer}],
        reward=reward, metadata={"taxonomy_path": scenario})
    if rejected is not None:
        traj.metadata["rejected"] = rejected
    return traj


def test_chat_rows_end_on_an_assistant_turn():
    """The torch backend only takes the prompt-masking path when they all do."""
    ds = emit.to_chat([_traj("q1", "a1"), _traj("q2", "a2")])
    assert ds.format == "chat"
    assert all(r["messages"][-1]["role"] == "assistant" for r in ds.rows)


def test_text_rows_are_the_prose_not_a_transcript():
    ds = emit.to_text([_traj("what is X?", "X is a thing.")])
    assert ds.format == "text"
    assert ds.rows == [{"text": "X is a thing."}]


def test_preference_rows_carry_all_three_keys():
    ds = emit.to_preference([_traj("q", "good", rejected="bad")])
    assert ds.format == "preference"  # what Dataset detection calls it
    assert set(ds.rows[0]) == {"prompt", "chosen", "rejected"}
    # the exact key set trl's DPOTrainer validates on
    assert ds.rows[0]["chosen"] != ds.rows[0]["rejected"]


def test_preference_drops_pairs_with_no_contrast():
    with pytest.raises(ValueError, match="no usable preference pairs"):
        emit.to_preference([_traj("q", "same", rejected="same")])


def test_grpo_prompt_rows_have_the_column_the_backend_looks_for():
    ds = emit.to_grpo_prompts([_traj("q", "a")])
    assert "prompt" in ds.rows[0] and "answer" in ds.rows[0]


def test_groups_are_accepted_by_weighted_rows():
    """The real inject-ready assertion: the GRPO row builder takes them."""
    trajs = [_traj(f"q{i}", f"a{i}", scenario="shared", reward=r)
             for i, r in enumerate((0.9, 0.5, 0.1))]
    groups = emit.to_groups(trajs)
    assert len(groups) == 1 and len(groups[0]) == 3
    rows = weighted_rows(groups)
    assert all(set(r) == {"messages", "weight"} for r in rows)


def test_groups_without_reward_spread_are_dropped_loudly():
    flat = [_traj(f"q{i}", "a", scenario="shared", reward=0.5) for i in range(3)]
    with pytest.raises(ValueError, match="no scored groups"):
        emit.to_groups(flat)


def test_paraphrase_units_route_through_the_real_bm25_router():
    """MoRE+'s contract: k consecutive rows per fact, reachable by any phrasing."""
    fact_a = ["what does cloud cost per agent run",
              "how am i billed each execution",
              "pricing for one invocation"]
    fact_b = ["where is the company headquartered",
              "which city is the office in",
              "what is the head office location"]
    rows = [_traj(q, "$0.08").messages for q in fact_a]
    rows += [_traj(q, "Boston").messages for q in fact_b]
    ds = emit.to_chat([Trajectory(messages=m) for m in rows])

    surrogates = [s for s, _ in mp.split_units(ds, group_size=3)]
    assert len(surrogates) == 2
    router = mp.BM25Router.build(surrogates)
    # a query phrased like the *third* row of unit 0 still routes to unit 0
    assert router.rank("pricing one invocation", 1)[0][0] == 0
    assert router.rank("which city is the office", 1)[0][0] == 1
