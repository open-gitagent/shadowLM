"""TrainConfig's resolvers and the TrainingRun serialization boundary.

Pure functions with no backend deps — and the hub round-trips jobs through
to_dict/from_dict, so a regression here silently corrupts every remote run.
"""

from __future__ import annotations

import pytest

from shadowlm.training import (ATTENTION_MODULES, MLP_MODULES, Metric,
                               TrainConfig, resolve_total_steps)


# ---- target module presets -------------------------------------------------

def test_presets_expand():
    assert TrainConfig(method="lora",
                       target_modules="attention").resolved_target_modules() \
        == ATTENTION_MODULES
    assert TrainConfig(method="lora",
                       target_modules="mlp").resolved_target_modules() == MLP_MODULES
    all_mods = TrainConfig(method="lora", target_modules="all").resolved_target_modules()
    assert set(ATTENTION_MODULES) | set(MLP_MODULES) == set(all_mods)


def test_explicit_module_names_pass_through():
    cfg = TrainConfig(method="lora", target_modules=["q_proj", "v_proj"])
    assert cfg.resolved_target_modules() == ("q_proj", "v_proj")


def test_unknown_preset_names_the_valid_ones():
    with pytest.raises(ValueError, match="preset"):
        TrainConfig(method="lora", target_modules="everything").resolved_target_modules()


# ---- warmup ------------------------------------------------------------------

def test_warmup_ratio_wins_over_steps():
    cfg = TrainConfig(method="lora", warmup_steps=99, warmup_ratio=0.1)
    assert cfg.resolved_warmup(200) == 20


def test_warmup_steps_used_when_no_ratio():
    assert TrainConfig(method="lora", warmup_steps=7).resolved_warmup(200) == 7


def test_warmup_never_negative():
    assert TrainConfig(method="lora", warmup_steps=-5).resolved_warmup(100) == 0


# ---- eval interval -------------------------------------------------------------

def test_eval_steps_defaults_to_quarter_of_the_run():
    assert TrainConfig(method="lora").resolved_eval_steps(100) == 25


def test_eval_steps_fraction_scales_with_total():
    assert TrainConfig(method="lora", eval_steps=0.5).resolved_eval_steps(80) == 40


def test_eval_steps_absolute_is_kept():
    assert TrainConfig(method="lora", eval_steps=13).resolved_eval_steps(80) == 13


def test_eval_steps_never_zero():
    assert TrainConfig(method="lora").resolved_eval_steps(1) == 1
    assert TrainConfig(method="lora", eval_steps=0.001).resolved_eval_steps(10) == 1


# ---- total steps ----------------------------------------------------------------

def test_max_steps_wins_over_epochs():
    cfg = TrainConfig(method="lora", max_steps=42, num_train_epochs=10)
    assert resolve_total_steps(cfg, n_examples=1000) == 42


def test_epochs_scale_with_rows_and_batching():
    cfg = TrainConfig(method="lora", max_steps=None, num_train_epochs=2,
                      per_device_train_batch_size=2,
                      gradient_accumulation_steps=1)
    assert resolve_total_steps(cfg, n_examples=10) == 10  # 10/2 per epoch × 2


def test_total_steps_is_at_least_one():
    cfg = TrainConfig(method="lora", max_steps=None, num_train_epochs=1,
                      per_device_train_batch_size=64)
    assert resolve_total_steps(cfg, n_examples=1) == 1


# ---- round trip -------------------------------------------------------------------

def test_config_dict_roundtrip_preserves_fields():
    cfg = TrainConfig(method="dpo", learning_rate=5e-6, lora_r=32,
                      target_modules="attention", report_to=["wandb"], beta=0.2)
    back = TrainConfig(**cfg.to_dict())
    assert back.method == "dpo"
    assert back.learning_rate == 5e-6
    assert back.lora_r == 32
    assert back.beta == 0.2
    assert back.resolved_target_modules() == ATTENTION_MODULES


def test_metric_survives_the_wire():
    m = Metric(step=3, loss=1.5, lr=1e-4, grad_norm=0.9, epoch=0.5)
    assert Metric(**vars(m)) == m
