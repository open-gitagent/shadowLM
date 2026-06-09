"""Training methods — one module per technique, all driven by one spec.

Built-ins: lora, qlora, dora, full, cpt. Backends read the spec's fields
(adapter kind, base requirements, data rendering) — never the method name — so
adding a technique is a new module here with one `register()` call:

    # shadowlm/methods/my_method.py
    from .base import TrainingMethod, register

    MY_METHOD = register(TrainingMethod(
        name="my-method",
        description="LoRA variant with my defaults",
        default_learning_rate=1e-4,
    ))

    # ...then import it below, and: model.finetune(ds, method="my-method")

Users can also register at runtime — `methods.register(...)` before calling
`finetune` — without touching this package.
"""

from .base import (
    ADAPTER_DORA,
    ADAPTER_LORA,
    ADAPTER_NONE,
    TrainingMethod,
    available,
    get,
    register,
)

# Importing a technique module registers it.
from . import cpt, dora, full, lora, qlora  # noqa: E402, F401

__all__ = [
    "ADAPTER_DORA",
    "ADAPTER_LORA",
    "ADAPTER_NONE",
    "TrainingMethod",
    "available",
    "get",
    "register",
]
