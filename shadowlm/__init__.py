"""shadowLM — a beautiful, minimal fine-tuning SDK.

    import shadowlm as slm

    ds    = slm.Dataset.from_jsonl("data.jsonl").as_chat()
    model = slm.load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    run   = model.finetune(ds, method="lora", max_steps=60)
    print(run.loss, run.sparkline())
    print(model.generate("Hello!"))
    model.save("out/", fmt="adapter")

datasets → finetune → inference. mlx on Apple Silicon, torch on CUDA (or CPU)
— accelerated by the shadow layer.
"""

from . import methods, runs
from .data import Dataset
from .models import Model, load
from .training import Metric, TrainConfig, TrainingRun

__version__ = "0.1.0"

__all__ = [
    "Dataset",
    "Model",
    "load",
    "methods",
    "runs",
    "Metric",
    "TrainConfig",
    "TrainingRun",
    "__version__",
]
