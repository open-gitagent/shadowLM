"""Prefix tuning — trainable vectors prepended to keys/values at every layer.

Deeper steering than input-level soft prompts: each attention layer gets its
own learned prefix while the weights stay frozen. Currently blocked by an
upstream peft/transformers-5 incompatibility — registered so the error is
clear; use "prompt" or "ptuning" meanwhile.
"""

from .base import TrainingMethod, register

PREFIX = register(TrainingMethod(
    name="prefix",
    description="prefix tuning — learned key/value prefixes at every layer",
    default_learning_rate=5e-3,
    adapter="prefix",
))
