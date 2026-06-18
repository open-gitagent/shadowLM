"""ShadowLM Trainer — a fine-tuning SDK.

Any open model — with any method, on any hardware, for any harness.

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

from . import checkpoints, hub, methods, runs
from .apo import APORun, optimize_prompt
from .capture import CaptureProxy, capture
from .checkpoints import Checkpoint
from .data import Dataset
from .models import Model, Reply, load
from .rl import Trajectory, TrajectoryGroup, judge_group
from .training import Metric, TrainConfig, TrainingRun

__version__ = "0.4.10"

__all__ = [
    "APORun",
    "optimize_prompt",
    "CaptureProxy",
    "capture",
    "Checkpoint",
    "checkpoints",
    "hub",
    "Dataset",
    "Model",
    "Reply",
    "load",
    "methods",
    "runs",
    "Metric",
    "TrainConfig",
    "TrainingRun",
    "Trajectory",
    "TrajectoryGroup",
    "judge_group",
    "__version__",
]
