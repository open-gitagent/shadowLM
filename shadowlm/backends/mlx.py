"""The MLX backend — LoRA fine-tuning and inference on Apple Silicon.

MLX is shadowLM's native path on Apple Silicon — LoRA fine-tuning on the Metal GPU
via `mlx-lm`. It mirrors the canonical mlx-lm LoRA wiring: freeze →
`linear_to_lora_layers` → Adam → `train(...)`. The shadow accelerator adds gradient
checkpointing when it helps. Needs Apple Silicon + `pip install shadowlm[mlx]`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .. import accel
from .._quiet import quiet_backend
from ..data import CHAT, TEXT, Dataset
from ..training import Metric, TrainConfig, resolve_total_steps
from .base import Backend, Callbacks, FinetuneResult

DEFAULT_LORA_LAYERS = 16  # how many transformer blocks get LoRA adapters


def _to_mlx_dataset(dataset: Dataset, tokenizer):
    """Wrap our Dataset in the mlx-lm dataset type that matches its format."""
    from mlx_lm.tuner.datasets import ChatDataset, TextDataset  # noqa: PLC0415

    ds = dataset.as_chat() if dataset.format in (CHAT, "instruction") else dataset
    if ds.format == CHAT:
        return ChatDataset(ds.rows, tokenizer)
    if ds.format == TEXT:
        return TextDataset(ds.rows, tokenizer)
    # raw → render to plain text so training still has something well-formed
    rows = [{"text": t} for t in dataset.to_texts()]
    return TextDataset(rows, tokenizer)


class MLXBackend(Backend):
    name = "mlx"

    def __init__(self, *, accelerator: str = "auto") -> None:
        # MLX trains on the Metal GPU (unified memory); there is no CPU mode here.
        super().__init__(device="gpu", accelerator=accelerator)
        import mlx.core as mx  # noqa: PLC0415

        mx.set_default_device(mx.gpu)

    @classmethod
    def is_available(cls) -> bool:
        import importlib.util
        import platform

        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return False
        return importlib.util.find_spec("mlx_lm") is not None

    def load(self, name, *, load_in_4bit=False, max_seq_length=2048, adapter=None, **kwargs) -> None:
        from mlx_lm import load  # noqa: PLC0415

        self.model_name = name
        self.max_seq_length = max_seq_length
        self.adapter = adapter
        with quiet_backend():  # swallow huggingface_hub "Fetching files" tqdm
            self.model, self.tokenizer = load(name, adapter_path=adapter)
        self._tuned = False
        # Loading with an adapter already converts the linear layers to LoRA.
        self._lora_applied = adapter is not None

    def finetune(self, dataset: Dataset, config: TrainConfig, callbacks: Callbacks,
                 output_dir: str, eval_dataset: Dataset | None = None) -> FinetuneResult:
        import mlx.core as mx  # noqa: PLC0415
        import mlx.optimizers as optim  # noqa: PLC0415
        from mlx_lm.tuner.datasets import CacheDataset  # noqa: PLC0415
        from mlx_lm.tuner.trainer import TrainingArgs, train  # noqa: PLC0415
        from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: PLC0415

        model, tokenizer = self.model, self.tokenizer
        n = len(dataset)
        iters = resolve_total_steps(config, n)
        num_layers = min(DEFAULT_LORA_LAYERS, len(model.layers))
        has_eval = eval_dataset is not None and len(eval_dataset) > 0

        shadow = accel.plan(self.accelerator, backend="mlx", n_layers=len(model.layers))
        if self.accelerator != "none":
            callbacks.log(shadow.note)

        # Attach the trainable surface once. A repeated finetune (or a finetune
        # continuing from a loaded adapter) keeps training the existing LoRA layers
        # rather than converting them again, which would error.
        if config.method == "full":
            model.freeze()
            for layer in model.layers[-num_layers:]:
                layer.unfreeze()
        elif not self._lora_applied:
            model.freeze()
            linear_to_lora_layers(model, num_layers, self._lora_params(config), use_dora=False)
            self._lora_applied = True

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        adapter_file = out / "adapters.safetensors"

        # Hold out the eval set if given; otherwise disable eval (steps_per_eval > iters).
        if has_eval:
            batch = config.per_device_train_batch_size
            val_batches = max(1, min(len(eval_dataset) // batch or 1, 25))
            steps_per_eval = config.eval_steps or max(1, iters // 4)
            val_set = CacheDataset(_to_mlx_dataset(eval_dataset, tokenizer))
        else:
            val_batches, steps_per_eval = 1, iters + 1
            val_set = CacheDataset(_to_mlx_dataset(dataset, tokenizer))

        args = TrainingArgs(
            batch_size=config.per_device_train_batch_size,
            iters=iters,
            val_batches=val_batches,
            steps_per_report=config.logging_steps,
            steps_per_eval=steps_per_eval,
            steps_per_save=iters,
            adapter_file=str(adapter_file),
            max_seq_length=config.max_seq_length,
            grad_checkpoint=shadow.grad_checkpoint,
            grad_accumulation_steps=config.gradient_accumulation_steps,
        )
        train_set = CacheDataset(_to_mlx_dataset(dataset, tokenizer))
        opt = optim.Adam(learning_rate=config.learning_rate)

        callbacks.log(
            f"[mlx:{self.device}] finetuning {self.model_name} · {config.method} · "
            f"{n} examples · {iters} iters · lora r={config.lora_r} on {num_layers} layers"
        )
        cb = _MetricBridge(callbacks, record_eval=has_eval)
        with quiet_backend():
            train(model=model, optimizer=opt, train_dataset=train_set,
                  val_dataset=val_set, args=args, training_callback=cb)
            mx.eval(model.parameters())

        self._write_adapter_config(out, config, num_layers)
        # Remember enough to write a self-contained adapter later via save().
        self._train_config = config
        self._num_layers = num_layers
        self._tuned = True
        callbacks.log(f"[mlx] done · final loss {cb.last_loss} · adapter {out}")
        return FinetuneResult(checkpoint=str(out), final_loss=cb.last_loss)

    @staticmethod
    def _lora_params(config: TrainConfig) -> dict:
        # MLX's `scale` is the direct LoRA multiplier; PEFT's effective scaling is
        # alpha/r. Match PEFT semantics so configs behave the same across backends.
        scale = config.lora_alpha / config.lora_r if config.lora_r else 1.0
        return {"rank": config.lora_r, "dropout": config.lora_dropout, "scale": scale}

    def _write_adapter_config(self, out: Path, config: TrainConfig, num_layers: int) -> None:
        # Shape mlx_lm.load(..., adapter_path=out) expects to re-attach the adapter.
        (out / "adapter_config.json").write_text(json.dumps({
            "fine_tune_type": "lora" if config.method != "full" else "full",
            "num_layers": num_layers,
            "lora_parameters": self._lora_params(config),
            "base_model": self.model_name,
        }, indent=2))

    def generate(self, prompt, *, max_new_tokens, temperature, top_p, **kwargs) -> str:
        from mlx_lm import generate  # noqa: PLC0415
        from mlx_lm.sample_utils import make_sampler  # noqa: PLC0415

        text = self._apply_template(prompt)
        sampler = make_sampler(temp=temperature, top_p=top_p)
        return generate(self.model, self.tokenizer, prompt=text,
                        max_tokens=max_new_tokens, sampler=sampler, verbose=False)

    def _apply_template(self, prompt: str) -> str:
        tok = self.tokenizer
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False,
            )
        return prompt

    def save(self, path: str, *, fmt: str = "adapter") -> str:
        import mlx.core as mx  # noqa: PLC0415
        from mlx.utils import tree_flatten  # noqa: PLC0415

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        if fmt == "merged":
            # Save the full (LoRA-applied) weight set as a standalone model.
            self.model.save_weights(str(out / "model.safetensors"))
        else:
            adapter_weights = dict(tree_flatten(self.model.trainable_parameters()))
            mx.save_safetensors(str(out / "adapters.safetensors"), adapter_weights)
            # Write the config alongside the weights so the dir is self-contained
            # and reloadable via load(adapter=path) / mlx_lm.load(adapter_path=path).
            cfg = getattr(self, "_train_config", None)
            if cfg is not None:
                self._write_adapter_config(out, cfg, getattr(self, "_num_layers", DEFAULT_LORA_LAYERS))
        return str(out)


class _MetricBridge:
    """mlx_lm TrainingCallback → our Callbacks.step(Metric)."""

    def __init__(self, callbacks: Callbacks, *, record_eval: bool = False) -> None:
        self._cb = callbacks
        self._record_eval = record_eval  # mlx always evals at start/end; only forward when asked
        self.last_loss: float | None = None
        self._start = time.time()

    def on_train_loss_report(self, info: dict) -> None:
        loss = info.get("train_loss")
        if loss is None:
            return
        self.last_loss = round(float(loss), 4)
        self._cb.step(Metric(
            step=int(info.get("iteration", 0)),
            loss=self.last_loss,
            lr=float(info.get("learning_rate", 0.0)),
            elapsed_s=round(time.time() - self._start, 2),
        ))

    def on_val_loss_report(self, info: dict) -> None:
        loss = info.get("val_loss")
        if loss is None or not self._record_eval:
            return
        self._cb.eval(Metric(
            step=int(info.get("iteration", 0)),
            loss=round(float(loss), 4),
            elapsed_s=round(time.time() - self._start, 2),
        ))
