"""The verl backend obeys the two repo-wide invariants: dispatch on the method
spec's fields (never its name), and never drop a TrainConfig field silently.

No verl install needed — these cover the pure mapping and the guard clauses.
"""

from __future__ import annotations

import pytest

from shadowlm import methods
from shadowlm.backends.verl import VerlBackend, _ignored_fields
from shadowlm.data import Dataset
from shadowlm.backends.base import Callbacks
from shadowlm.training import TrainConfig


def _reward(prompts=None, completions=None, answer=None, **kw):  # importable
    return [1.0 for _ in (completions or [])]


@pytest.fixture()
def be():
    backend = VerlBackend()
    backend.load("tiny/model", load_in_4bit=False, max_seq_length=512)
    return backend


def _run(backend, config, tmp_path, logs, **kw):
    return backend.finetune(
        Dataset.from_list([{"prompt": "a", "answer": "b"}]), config,
        Callbacks(on_log=logs.append), str(tmp_path), reward_fns=[_reward], **kw)


# ---- dispatch on the spec, not the name ----------------------------------------

def test_registered_grpo_trainer_method_is_accepted(be, tmp_path):
    """A user-registered method whose trainer is grpo must not be rejected for
    having a different name — the backend reads spec.trainer."""
    methods.register(methods.TrainingMethod(
        name="my_grpo", description="custom GRPO", default_learning_rate=1e-6,
        adapter=methods.ADAPTER_NONE, trainer="grpo"))
    logs: list[str] = []
    config = TrainConfig(method="my_grpo", learning_rate=1e-6)
    # gets past the method check and fails on the missing verl install instead
    with pytest.raises(RuntimeError, match="pip install"):
        _run(be, config, tmp_path, logs)


def test_non_grpo_trainer_is_refused_by_trainer_not_name(be, tmp_path):
    logs: list[str] = []
    with pytest.raises(ValueError, match="grpo"):
        _run(be, TrainConfig(method="lora", learning_rate=1e-4), tmp_path, logs)


# ---- ignored fields are logged ---------------------------------------------------

def test_unmapped_fields_are_reported():
    ignored = _ignored_fields(TrainConfig(method="grpo", weight_decay=0.2,
                                          lora_r=32, seed=11, packing=True))
    assert {"weight_decay", "lora_r", "seed", "packing"} <= set(ignored)


def test_mapped_fields_are_not_reported():
    ignored = _ignored_fields(TrainConfig(method="grpo", learning_rate=1e-6,
                                          max_seq_length=1024, beta=0.05,
                                          grpo_group_size=16))
    for mapped in ("learning_rate", "max_seq_length", "beta", "grpo_group_size"):
        assert mapped not in ignored


def test_finetune_logs_the_ignored_fields(be, tmp_path):
    logs: list[str] = []
    config = TrainConfig(method="grpo", learning_rate=1e-6, weight_decay=0.2)
    with pytest.raises(RuntimeError, match="pip install"):
        _run(be, config, tmp_path, logs)
    assert any("weight_decay" in line for line in logs), logs


# ---- load doesn't accept-and-discard --------------------------------------------

def test_load_notes_args_it_cannot_honor(capsys):
    backend = VerlBackend()
    backend.load("tiny/model", load_in_4bit=True, max_seq_length=4096,
                 adapter="/some/adapter")
    assert backend.model_name == "tiny/model"
    assert "load_in_4bit" in capsys.readouterr().out


# ---- unsupported surface is actionable ------------------------------------------

@pytest.mark.parametrize("call", [
    lambda b: b.chat([{"role": "user", "content": "hi"}], max_new_tokens=8,
                     temperature=0.7, top_p=0.9),
    lambda b: b.save("/tmp/out"),
    lambda b: b.generate("hi", max_new_tokens=8, temperature=0.7, top_p=0.9),
])
def test_inference_surface_points_at_torch(be, call):
    with pytest.raises(NotImplementedError, match="torch"):
        call(be)
