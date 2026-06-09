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

from .. import accel, methods
from .._quiet import quiet_backend
from ..data import CHAT, TEXT, Dataset
from ..training import ATTENTION_MODULES, MLP_MODULES, Metric, TrainConfig, resolve_total_steps
from .base import Backend, Callbacks, FinetuneResult

DEFAULT_LORA_LAYERS = 16  # how many transformer blocks get LoRA adapters


def _to_mlx_dataset(dataset: Dataset, tokenizer, *, raw_text: bool = False,
                    mask_prompt: bool = False):
    """Wrap our Dataset in the mlx-lm dataset type that matches its format.

    raw_text=True (continued pretraining) renders rows to plain text with no chat
    template so the model trains on the domain text itself. mask_prompt=True
    computes loss only on the assistant response (train-on-completions).
    """
    from mlx_lm.tuner.datasets import ChatDataset, TextDataset  # noqa: PLC0415

    if not raw_text:
        ds = dataset.as_chat() if dataset.format in (CHAT, "instruction") else dataset
        if ds.format == CHAT:
            return ChatDataset(ds.rows, tokenizer, mask_prompt=mask_prompt)
        if ds.format == TEXT:
            return TextDataset(ds.rows, tokenizer)
    rows = [{"text": t} for t in dataset.to_texts()]
    return TextDataset(rows, tokenizer)


def _build_lr(config: TrainConfig, total_steps: int):
    """Learning-rate schedule for mlx — warmup + linear/cosine/constant decay.

    mlx advances the schedule once per optimizer update, which with gradient
    accumulation happens every `gradient_accumulation_steps` iters — so the
    schedule is built in update units, not iter units.
    """
    import math  # noqa: PLC0415

    from mlx_lm.tuner.utils import build_schedule  # noqa: PLC0415

    accum = max(1, config.gradient_accumulation_steps)
    updates = max(1, math.ceil(total_steps / accum))
    warmup = min(math.ceil(config.resolved_warmup(total_steps) / accum), updates - 1)
    decay_steps = max(1, updates - warmup)
    if config.lr_scheduler_type == "cosine":
        name, arguments = "cosine_decay", [config.learning_rate, decay_steps]
    elif config.lr_scheduler_type == "constant":
        if not warmup:
            return config.learning_rate
        name, arguments = "linear_schedule", [config.learning_rate, config.learning_rate, 1]
    else:  # linear decay to 0
        name, arguments = "linear_schedule", [config.learning_rate, 0.0, decay_steps]
    return build_schedule({"name": name, "arguments": arguments, "warmup": warmup})


def _is_quantized(model) -> bool:
    import mlx.nn as nn  # noqa: PLC0415

    return any(isinstance(m, nn.QuantizedLinear) for _, m in model.named_modules())


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

        # The method spec drives everything below: base requirements, trainable
        # surface, and data rendering. Backends never branch on the method name.
        spec = methods.get(config.method)
        spec.validate_base(
            quantized=_is_quantized(model),
            quantize_hint="Load a quantized repo (e.g. an mlx-community *-4bit model)",
            dequantize_hint="Load a 16-bit repo",
        )

        # Attach the trainable surface once. A repeated finetune (or a finetune
        # continuing from a loaded adapter) keeps training the existing adapter
        # layers rather than converting them again, which would error.
        if not spec.trains_adapters:
            # Full fine-tune: every transformer block trains.
            num_layers = len(model.layers)
            model.freeze()
            for layer in model.layers:
                layer.unfreeze()
        elif not self._lora_applied:
            model.freeze()
            linear_to_lora_layers(model, num_layers, self._lora_params(config),
                                  use_dora=(spec.adapter == methods.ADAPTER_DORA))
            self._lora_applied = True

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        adapter_file = out / "adapters.safetensors"

        # raw_text methods (continued pretraining) skip the chat template;
        # train_on_completions masks the prompt so loss covers only responses.
        raw_text = spec.raw_text
        mask = config.train_on_completions

        # Continue from a previous adapter/checkpoint if asked.
        if config.resume_from_checkpoint:
            resume = Path(config.resume_from_checkpoint)
            weights = resume / "adapters.safetensors" if resume.is_dir() else resume
            model.load_weights(str(weights), strict=False)
            callbacks.log(f"[mlx] resumed weights from {weights}")

        # Hold out the eval set if given; otherwise disable eval (steps_per_eval > iters).
        if has_eval:
            batch = config.per_device_train_batch_size
            val_batches = max(1, min(len(eval_dataset) // batch or 1, 25))
            steps_per_eval = config.resolved_eval_steps(iters)
            val_set = CacheDataset(
                _to_mlx_dataset(eval_dataset, tokenizer, raw_text=raw_text, mask_prompt=mask))
        else:
            val_batches, steps_per_eval = 1, iters + 1
            val_set = CacheDataset(
                _to_mlx_dataset(dataset, tokenizer, raw_text=raw_text, mask_prompt=mask))

        # mlx-lm requires batch_size <= len(dataset) for both train and eval
        # sets; clamp for tiny datasets.
        batch_size = max(1, min(config.per_device_train_batch_size, n,
                                len(eval_dataset) if has_eval else n))

        args = TrainingArgs(
            batch_size=batch_size,
            iters=iters,
            val_batches=val_batches,
            steps_per_report=config.logging_steps,
            steps_per_eval=steps_per_eval,
            steps_per_save=config.save_steps or iters,
            adapter_file=str(adapter_file),
            max_seq_length=config.max_seq_length,
            grad_checkpoint=shadow.grad_checkpoint,
            grad_accumulation_steps=config.gradient_accumulation_steps,
        )
        train_set = CacheDataset(
            _to_mlx_dataset(dataset, tokenizer, raw_text=raw_text, mask_prompt=mask))
        opt = optim.Adam(learning_rate=_build_lr(config, iters))

        # Fields the mlx path can't honor — say so instead of silently dropping.
        ignored = [name for name, off in (
            ("optim", config.optim != "adamw_8bit"),
            ("max_grad_norm", config.max_grad_norm is not None),
            ("packing", config.packing),
            ("use_rslora", config.use_rslora),
            ("report_to", bool(config.report_to)),
        ) if off]
        if ignored:
            callbacks.log(f"[mlx] note: {', '.join(ignored)} not supported on mlx — ignored")

        surface = (f"all {num_layers} layers" if not spec.trains_adapters
                   else f"{spec.adapter} r={config.lora_r} on {num_layers} layers")
        callbacks.log(
            f"[mlx:{self.device}] finetuning {self.model_name} · {config.method} · "
            f"{n} examples · {iters} iters · {surface} · lr {config.learning_rate:g} "
            f"({config.lr_scheduler_type}, warmup {config.resolved_warmup(iters)})"
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
        # Map module names to mlx key paths (self_attn.q_proj, mlp.gate_proj, ...)
        # so target_modules actually constrains which layers get adapters —
        # without keys, mlx adapts every linear layer.
        keys = []
        for mod in config.resolved_target_modules():
            if "." in mod:
                keys.append(mod)
            elif mod in ATTENTION_MODULES:
                keys.append(f"self_attn.{mod}")
            elif mod in MLP_MODULES:
                keys.append(f"mlp.{mod}")
            else:
                keys.append(mod)
        return {"rank": config.lora_r, "dropout": config.lora_dropout,
                "scale": scale, "keys": keys}

    def _write_adapter_config(self, out: Path, config: TrainConfig, num_layers: int) -> None:
        # Shape mlx_lm.load(..., adapter_path=out) expects to re-attach the adapter.
        spec = methods.get(config.method)
        fine_tune_type = spec.adapter if spec.trains_adapters else "full"
        (out / "adapter_config.json").write_text(json.dumps({
            "fine_tune_type": fine_tune_type,
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
