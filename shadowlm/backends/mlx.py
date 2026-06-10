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
from ..data import CHAT, INSTRUCTION, SHAREGPT, TEXT, Dataset
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
        ds = dataset.as_chat() if dataset.format in (CHAT, SHAREGPT, INSTRUCTION) else dataset
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

    def _claim_surface(self, kind: str) -> bool:
        """True → attach the surface now; False → same kind, reuse it.

        A model hosts exactly one trainable-surface kind; mixing raises.
        """
        cur = getattr(self, "_surface", None)
        if cur == kind:
            return False
        if cur is not None:
            raise RuntimeError(
                f"this model already hosts a {cur!r} trainable surface; "
                f"load a fresh model for {kind!r}."
            )
        self._surface = kind
        return True

    @classmethod
    def is_available(cls) -> bool:
        import importlib.util
        import platform

        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return False
        return importlib.util.find_spec("mlx_lm") is not None

    def load(self, name, *, load_in_4bit=False, max_seq_length=2048, adapter=None, **kwargs) -> None:
        from mlx_lm import load  # noqa: PLC0415

        from .. import more  # noqa: PLC0415

        self.model_name = name
        self.max_seq_length = max_seq_length
        self.adapter = adapter
        # Memory-tuned adapters carry their own index + wrapper config and are
        # re-attached by hand; plain adapters go through the normal loader.
        from .. import bottleneck  # noqa: PLC0415
        more_cfg = more.read_config(adapter) if adapter else None
        bn_cfg = bottleneck.read_config(adapter) if adapter else None
        with quiet_backend():  # swallow huggingface_hub "Fetching files" tqdm
            self.model, self.tokenizer = load(
                name, adapter_path=None if (more_cfg or bn_cfg) else adapter)
        if bn_cfg:
            self.model.freeze()
            bottleneck.attach_mlx(self.model, rank=bn_cfg["rank"])
            self.model.load_weights(str(Path(adapter) / "adapters.safetensors"),
                                    strict=False)
            self._surface = methods.ADAPTER_BOTTLENECK
        bitfit_marker = adapter and (Path(adapter) / "bitfit_config.json").exists()
        if adapter and bitfit_marker:
            # bias-only checkpoint: freeze, then re-enable exactly the biases
            self.model.freeze()
            self.model.unfreeze(keys=["bias"], strict=False)
            self._surface = methods.ADAPTER_BITFIT
        elif adapter and not (more_cfg or bn_cfg):
            # mlx-lm's load_adapters converts layers but never freezes the
            # model; without this, continued training updates every parameter.
            from mlx_lm.tuner.dora import DoRAEmbedding, DoRALinear  # noqa: PLC0415
            from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear  # noqa: PLC0415
            adapter_cfg = json.loads((Path(adapter) / "adapter_config.json").read_text())
            kind = adapter_cfg.get("fine_tune_type", "lora")
            if kind in (methods.ADAPTER_LORA, methods.ADAPTER_DORA):
                self.model.freeze()
                kinds = (LoRALinear, LoRAEmbedding, DoRALinear, DoRAEmbedding)
                for _, mod in self.model.named_modules():
                    if isinstance(mod, kinds):
                        # only the adapter tensors — recursing would unfreeze
                        # the wrapped base layer too
                        mod.unfreeze(keys=["lora_a", "lora_b", "m"],
                                     strict=False, recurse=False)
                self._surface = kind
            # fine_tune_type "full": the model resumes fully trainable
        if more_cfg:
            from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: PLC0415

            self._more_index = more.MemoryIndex.load(adapter)
            self.model.freeze()
            adapter_cfg = json.loads((Path(adapter) / "adapter_config.json").read_text())
            linear_to_lora_layers(self.model, more_cfg["num_layers"],
                                  adapter_cfg["lora_parameters"])
            more.attach(self.model, self._more_index, rank=more_cfg["rank"],
                        k=more_cfg["index_k"], num_layers=more_cfg["num_layers"])
            self.model.load_weights(str(Path(adapter) / "adapters.safetensors"),
                                    strict=False)
            self._surface = methods.ADAPTER_MORE
        self._tuned = False

    def finetune(self, dataset: Dataset, config: TrainConfig, callbacks: Callbacks,
                 output_dir: str, eval_dataset: Dataset | None = None,
                 reward_fns: list | None = None) -> FinetuneResult:
        import mlx.core as mx  # noqa: PLC0415
        import mlx.optimizers as optim  # noqa: PLC0415
        from mlx_lm.tuner.datasets import CacheDataset  # noqa: PLC0415
        from mlx_lm.tuner.trainer import TrainingArgs, train  # noqa: PLC0415
        from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: PLC0415

        model, tokenizer = self.model, self.tokenizer
        n = len(dataset)
        iters = resolve_total_steps(config, n)
        num_layers = len(model.layers)
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
        if getattr(self, "_more_index", None) is not None \
                and spec.adapter != methods.ADAPTER_MORE:
            raise RuntimeError(
                "this model has retrieval experts attached (method='more'); the "
                "standard compiled trainer can't run over them. Load a fresh "
                "model to train with other methods."
            )
        if spec.trainer == "dpo":
            return self._finetune_dpo(dataset, config, callbacks, output_dir,
                                      eval_dataset, spec, shadow, iters, num_layers)
        if spec.trainer == "grpo":
            return self._finetune_grpo(dataset, config, callbacks, output_dir,
                                       eval_dataset, spec, shadow, iters, num_layers,
                                       reward_fns)

        # Attach the trainable surface once. A repeated finetune (or a finetune
        # continuing from a loaded adapter) keeps training the existing adapter
        # layers rather than converting them again, which would error.
        if spec.adapter == methods.ADAPTER_MORE:
            return self._finetune_more(dataset, config, callbacks, output_dir,
                                       eval_dataset, iters)
        if spec.adapter in (methods.ADAPTER_PROMPT, methods.ADAPTER_PTUNING):
            raise RuntimeError(
                f"method={config.method!r} (soft-prompt family) runs on the torch "
                "backend — load with backend='torch'."
            )
        if not spec.trains_adapters:
            # Full fine-tune: every transformer block trains.
            if getattr(self, "_surface", None) is not None:
                raise RuntimeError(
                    f"this model already hosts a {self._surface!r} trainable "
                    "surface; load a fresh model for a full fine-tune."
                )
            num_layers = len(model.layers)
            model.freeze()
            for layer in model.layers:
                layer.unfreeze()
        elif spec.adapter == methods.ADAPTER_BITFIT:
            if self._claim_surface(methods.ADAPTER_BITFIT):
                model.freeze()
                model.unfreeze(keys=["bias"], strict=False)
                n_bias = sum(v.size for _, v in
                             __import__("mlx.utils", fromlist=["tree_flatten"])
                             .tree_flatten(model.trainable_parameters()))
                if n_bias == 0:
                    raise RuntimeError(
                        "method='bitfit' found no bias parameters in this model."
                    )
                callbacks.log(f"[mlx] bitfit: training {n_bias:,} bias parameters")
            num_layers = len(model.layers)
        elif spec.adapter == methods.ADAPTER_BOTTLENECK:
            if self._claim_surface(methods.ADAPTER_BOTTLENECK):
                from .. import bottleneck  # noqa: PLC0415
                model.freeze()
                wrapped = bottleneck.attach_mlx(model, rank=config.lora_r)
                callbacks.log(f"[mlx] bottleneck adapters (r={config.lora_r}) "
                              f"on {wrapped} layers")
            num_layers = len(model.layers)
        elif spec.adapter in (methods.ADAPTER_LORA, methods.ADAPTER_DORA):
            if self._claim_surface(spec.adapter):
                model.freeze()
                linear_to_lora_layers(model, num_layers, self._lora_params(config),
                                      use_dora=(spec.adapter == methods.ADAPTER_DORA))
        else:
            raise RuntimeError(f"mlx backend has no attach path for adapter kind {spec.adapter!r}")

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
            if not weights.exists():
                raise FileNotFoundError(
                    f"no saved weights at {weights} — the run may have stopped "
                    "before its first checkpoint. Set save_steps=N to checkpoint "
                    "mid-run, or resume from a completed run."
                )
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
        if spec.adapter == methods.ADAPTER_BOTTLENECK:
            from .. import bottleneck  # noqa: PLC0415
            bottleneck.write_config(out, base_model=self.model_name, rank=config.lora_r)
        elif spec.adapter == methods.ADAPTER_BITFIT:
            (out / "bitfit_config.json").write_text(json.dumps(
                {"type": "bitfit", "base_model": self.model_name}, indent=2))
        # Remember enough to write a self-contained adapter later via save().
        self._train_config = config
        self._num_layers = num_layers
        self._tuned = True
        callbacks.log(f"[mlx] done · final loss {cb.last_loss} · adapter {out}")
        return FinetuneResult(checkpoint=str(out), final_loss=cb.last_loss)

    def _finetune_more(self, dataset: Dataset, config: TrainConfig, callbacks: Callbacks,
                       output_dir: str, eval_dataset: Dataset | None,
                       iters: int) -> FinetuneResult:
        """Mixture of Retrieval Experts.

        Runs its own uncompiled loop: the retrieval lookup materializes arrays
        mid-graph, which mx.compile / mx.checkpoint (used by the standard
        trainer) forbid. Plain value_and_grad allows it.
        """
        import time as _time  # noqa: PLC0415

        import mlx.core as mx  # noqa: PLC0415
        import mlx.nn as nn  # noqa: PLC0415
        import mlx.optimizers as optim  # noqa: PLC0415
        from mlx.utils import tree_flatten, tree_map  # noqa: PLC0415
        from mlx_lm.tuner.datasets import CacheDataset  # noqa: PLC0415
        from mlx_lm.tuner.trainer import default_loss, iterate_batches  # noqa: PLC0415

        from .. import more  # noqa: PLC0415

        model, tokenizer = self.model, self.tokenizer
        num_layers = self._attach_retrieval_experts(dataset, config, callbacks)

        n = len(dataset)
        has_eval = eval_dataset is not None and len(eval_dataset) > 0
        batch_size = max(1, min(config.per_device_train_batch_size, n,
                                len(eval_dataset) if has_eval else n))
        data = iterate_batches(
            CacheDataset(_to_mlx_dataset(dataset, tokenizer)),
            batch_size, config.max_seq_length, loop=True,
        )
        eval_every = config.resolved_eval_steps(iters) if has_eval else iters + 1

        def eval_loss() -> float:
            losses = []
            batches = iterate_batches(
                CacheDataset(_to_mlx_dataset(eval_dataset, tokenizer)),
                batch_size, config.max_seq_length, loop=False)
            for batch, lengths in batches:
                loss, _ = default_loss(model, batch, lengths)
                mx.eval(loss)
                losses.append(float(loss))
            return sum(losses) / max(1, len(losses))

        opt = optim.Adam(learning_rate=_build_lr(config, iters))
        loss_and_grad = nn.value_and_grad(model, default_loss)
        accum = max(1, config.gradient_accumulation_steps)

        callbacks.log(
            f"[mlx:{self.device}] more on {self.model_name} · {n} facts · "
            f"{iters} iters · retrieval k={config.retrieval_k} on {num_layers} layers "
            f"· lr {config.learning_rate:g}"
        )
        start = _time.time()
        grads_acc = None
        last_loss = None
        for it, (batch, lengths) in zip(range(1, iters + 1), data):
            (loss, _ntoks), grads = loss_and_grad(model, batch, lengths)
            grads_acc = grads if grads_acc is None else tree_map(
                lambda a, b: a + b, grads_acc, grads)
            if it % accum == 0:
                opt.update(model, tree_map(lambda g: g / accum, grads_acc))
                grads_acc = None
            mx.eval(loss, model.parameters())
            last_loss = round(float(loss), 4)
            if it % config.logging_steps == 0:
                callbacks.step(Metric(
                    step=it, loss=last_loss,
                    lr=float(opt.learning_rate),
                    elapsed_s=round(_time.time() - start, 2),
                ))
            if it % eval_every == 0 or (has_eval and it == iters):
                callbacks.eval(Metric(step=it, loss=round(eval_loss(), 4),
                                      elapsed_s=round(_time.time() - start, 2)))
            if callbacks.stopped():
                break

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(out / "adapters.safetensors"), weights)
        self._write_adapter_config(out, config, num_layers)
        self._more_index.save(out)
        more.write_config(out, base_model=self.model_name, rank=config.lora_r,
                          k=config.retrieval_k, num_layers=num_layers)
        self._train_config = config
        self._num_layers = num_layers
        self._tuned = True
        callbacks.log(f"[mlx] done · final loss {last_loss} · adapter {out}")
        return FinetuneResult(checkpoint=str(out), final_loss=last_loss)

    def _finetune_trajectory_grpo(self, dataset: Dataset, config: TrainConfig,
                                  callbacks: Callbacks, output_dir: str, spec,
                                  iters: int) -> FinetuneResult:
        """Advantage-weighted policy gradient over collected trajectories.

        Each row is {"messages", "weight"} where weight is the group-relative
        advantage. Loss = weight × NLL of the assistant tokens — above-average
        attempts are reinforced, below-average suppressed. On-policy: collect
        rollouts from the current model, score, train, repeat. Runs uncompiled
        (per-sequence weighting isn't expressible in the stock trainer).
        """
        import time as _time  # noqa: PLC0415

        import mlx.core as mx  # noqa: PLC0415
        import mlx.nn as nn  # noqa: PLC0415
        import mlx.optimizers as optim  # noqa: PLC0415
        from mlx.utils import tree_flatten, tree_map  # noqa: PLC0415
        from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: PLC0415

        model, tokenizer = self.model, self.tokenizer
        num_layers = len(model.layers)
        if self._claim_surface(spec.adapter):
            model.freeze()
            linear_to_lora_layers(model, num_layers, self._lora_params(config),
                                  use_dora=(spec.adapter == methods.ADAPTER_DORA))

        # Tokenize once: full conversation + prompt length for masking.
        examples = []
        for row in dataset.rows:
            msgs = row["messages"]
            full = tokenizer.apply_chat_template(msgs)
            prompt_len = len(tokenizer.apply_chat_template(
                msgs[:-1], add_generation_prompt=True))
            full = full[:config.max_seq_length]
            if prompt_len >= len(full):
                continue
            examples.append((full, prompt_len, float(row["weight"])))
        if not examples:
            raise ValueError("no usable trajectories after tokenization")

        pad = tokenizer.eos_token_id
        batch_size = max(1, min(config.per_device_train_batch_size, len(examples)))

        def batches():
            import random as _random  # noqa: PLC0415
            rng = _random.Random(config.seed)
            while True:
                order = list(range(len(examples)))
                rng.shuffle(order)
                for i in range(0, len(order) - batch_size + 1, batch_size):
                    chunk = [examples[j] for j in order[i:i + batch_size]]
                    L = max(len(t) for t, _, _ in chunk)
                    toks = mx.array([t + [pad] * (L - len(t)) for t, _, _ in chunk])
                    # mask over targets (positions 1..L-1): assistant tokens only
                    mask = mx.array([
                        [1.0 if p_len <= j + 1 < len(t) else 0.0 for j in range(L - 1)]
                        for t, p_len, _ in chunk])
                    weights = mx.array([w for _, _, w in chunk])
                    yield toks, mask, weights

        def pg_loss(mdl, toks, mask, weights):
            logits = mdl(toks[:, :-1])
            ce = nn.losses.cross_entropy(logits, toks[:, 1:], reduction="none")
            per_seq = (ce * mask).sum(-1) / mx.maximum(mask.sum(-1), 1)
            return (weights * per_seq).mean()

        opt = optim.Adam(learning_rate=_build_lr(config, iters))
        loss_and_grad = nn.value_and_grad(model, pg_loss)
        accum = max(1, config.gradient_accumulation_steps)
        callbacks.log(
            f"[mlx:{self.device}] trajectory-grpo on {self.model_name} · "
            f"{len(examples)} trajectories · {iters} iters · lr {config.learning_rate:g}"
        )
        start = _time.time()
        grads_acc = None
        last_loss = None
        for it, (toks, mask, weights) in zip(range(1, iters + 1), batches()):
            loss, grads = loss_and_grad(model, toks, mask, weights)
            grads_acc = grads if grads_acc is None else tree_map(
                lambda a, b: a + b, grads_acc, grads)
            if it % accum == 0:
                opt.update(model, tree_map(lambda g: g / accum, grads_acc))
                grads_acc = None
            mx.eval(loss, model.parameters())
            last_loss = round(float(loss), 4)
            if it % config.logging_steps == 0:
                callbacks.step(Metric(step=it, loss=last_loss,
                                      lr=float(opt.learning_rate),
                                      elapsed_s=round(_time.time() - start, 2)))
            if callbacks.stopped():
                break

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(out / "adapters.safetensors"),
                            dict(tree_flatten(model.trainable_parameters())))
        self._write_adapter_config(out, config, num_layers)
        self._train_config = config
        self._num_layers = num_layers
        self._tuned = True
        callbacks.log(f"[mlx] done · final pg loss {last_loss} · adapter {out}")
        return FinetuneResult(checkpoint=str(out), final_loss=last_loss)

    def _finetune_dpo(self, dataset: Dataset, config: TrainConfig, callbacks: Callbacks,
                      output_dir: str, eval_dataset: Dataset | None, spec, shadow,
                      iters: int, num_layers: int) -> FinetuneResult:
        """Preference training: rank chosen over rejected vs a frozen reference."""
        import mlx.core as mx  # noqa: PLC0415
        import mlx.optimizers as optim  # noqa: PLC0415

        try:
            from mlx_lm_lora.trainer.datasets import CacheDataset, DPODataset  # noqa: PLC0415
            from mlx_lm_lora.trainer.dpo_trainer import DPOTrainingArgs, train_dpo  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "Preference training on Apple Silicon needs mlx-lm-lora: "
                "pip install shadowlm[preference]"
            ) from e
        from mlx_lm import load as mlx_load  # noqa: PLC0415
        from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: PLC0415

        model, tokenizer = self.model, self.tokenizer
        n = len(dataset)
        missing = {"prompt", "chosen", "rejected"} - set(dataset.rows[0] if n else {})
        if missing:
            raise ValueError(
                f"method='dpo' needs preference rows with prompt/chosen/rejected "
                f"(missing: {', '.join(sorted(missing))})"
            )

        if self._claim_surface(spec.adapter):
            model.freeze()
            linear_to_lora_layers(model, num_layers, self._lora_params(config),
                                  use_dora=(spec.adapter == methods.ADAPTER_DORA))

        # The frozen reference is a pristine copy of the base model.
        with quiet_backend():
            ref_model, _ = mlx_load(self.model_name)
        ref_model.freeze()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        has_eval = eval_dataset is not None and len(eval_dataset) > 0
        train_set = CacheDataset(DPODataset(dataset.rows, tokenizer))
        val_set = CacheDataset(DPODataset(
            (eval_dataset.rows if has_eval else dataset.rows), tokenizer))

        batch_size = max(1, min(config.per_device_train_batch_size, n,
                                len(eval_dataset) if has_eval else n))
        args = DPOTrainingArgs(
            batch_size=batch_size,
            iters=iters,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            val_batches=1,
            steps_per_report=config.logging_steps,
            steps_per_eval=(config.resolved_eval_steps(iters) if has_eval else iters + 1),
            steps_per_save=config.save_steps or iters,
            max_seq_length=config.max_seq_length,
            adapter_file=str(out / "adapters.safetensors"),
            grad_checkpoint=shadow.grad_checkpoint,
            beta=config.beta,
        )
        opt = optim.Adam(learning_rate=_build_lr(config, iters))

        callbacks.log(
            f"[mlx:{self.device}] dpo on {self.model_name} · {n} preference pairs · "
            f"{iters} iters · beta {config.beta} · lr {config.learning_rate:g}"
        )
        cb = _MetricBridge(callbacks, record_eval=has_eval)
        with quiet_backend():
            train_dpo(model=model, ref_model=ref_model, optimizer=opt,
                      train_dataset=train_set, val_dataset=val_set, args=args,
                      training_callback=cb)
            mx.eval(model.parameters())
        del ref_model

        self._write_adapter_config(out, config, num_layers)
        self._train_config = config
        self._num_layers = num_layers
        self._tuned = True
        callbacks.log(f"[mlx] done · final loss {cb.last_loss} · adapter {out}")
        return FinetuneResult(checkpoint=str(out), final_loss=cb.last_loss)

    def _finetune_grpo(self, dataset: Dataset, config: TrainConfig, callbacks: Callbacks,
                       output_dir: str, eval_dataset: Dataset | None, spec, shadow,
                       iters: int, num_layers: int, reward_fns: list | None) -> FinetuneResult:
        """RL from programmable rewards: sample a group of completions per prompt,
        score them with reward_fns, push toward the above-average ones."""
        import mlx.core as mx  # noqa: PLC0415
        import mlx.optimizers as optim  # noqa: PLC0415

        try:
            from mlx_lm_lora.trainer.datasets import CacheDataset, GRPODataset  # noqa: PLC0415
            from mlx_lm_lora.trainer.grpo_trainer import GRPOTrainingArgs, train_grpo  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "GRPO on Apple Silicon needs mlx-lm-lora: pip install shadowlm[preference]"
            ) from e
        from mlx_lm import load as mlx_load  # noqa: PLC0415
        from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: PLC0415

        model, tokenizer = self.model, self.tokenizer
        n = len(dataset)
        if n and "weight" in dataset.rows[0] and "messages" in dataset.rows[0]:
            # trajectory-native: pre-collected rollouts with group advantages
            return self._finetune_trajectory_grpo(dataset, config, callbacks,
                                                  output_dir, spec, iters)
        if not reward_fns:
            raise ValueError(
                "method='grpo' needs reward_fns=[...] (or TrajectoryGroups) — "
                "each fn is fn(prompts, completions, answer, types=None) -> list[float]"
            )
        if n == 0 or "prompt" not in dataset.rows[0]:
            raise ValueError("method='grpo' needs rows with a 'prompt' column "
                             "(and optionally 'answer' for accuracy-style rewards)")
        rows = [{"answer": "", **r} for r in dataset.rows]  # answer optional

        if self._claim_surface(spec.adapter):
            model.freeze()
            linear_to_lora_layers(model, num_layers, self._lora_params(config),
                                  use_dora=(spec.adapter == methods.ADAPTER_DORA))

        with quiet_backend():  # frozen reference = pristine base
            ref_model, _ = mlx_load(self.model_name)
        ref_model.freeze()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        has_eval = eval_dataset is not None and len(eval_dataset) > 0
        eval_rows = ([{"answer": "", **r} for r in eval_dataset.rows] if has_eval else rows)
        train_set = CacheDataset(GRPODataset(rows, tokenizer))
        val_set = CacheDataset(GRPODataset(eval_rows, tokenizer))

        batch_size = max(1, min(config.per_device_train_batch_size, n,
                                len(eval_rows)))
        args = GRPOTrainingArgs(
            batch_size=batch_size,
            iters=iters,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            val_batches=1,
            steps_per_report=config.logging_steps,
            steps_per_eval=(config.resolved_eval_steps(iters) if has_eval else iters + 1),
            steps_per_save=config.save_steps or iters,
            max_seq_length=config.max_seq_length,
            adapter_file=str(out / "adapters.safetensors"),
            grad_checkpoint=shadow.grad_checkpoint,
            group_size=config.grpo_group_size,
            beta=config.beta,
            max_completion_length=config.grpo_max_completion_length,
        )
        opt = optim.Adam(learning_rate=_build_lr(config, iters))

        callbacks.log(
            f"[mlx:{self.device}] grpo on {self.model_name} · {n} prompts · "
            f"{iters} iters · group {config.grpo_group_size} · "
            f"{len(reward_fns)} reward fn(s) · lr {config.learning_rate:g}"
        )
        cb = _MetricBridge(callbacks, record_eval=has_eval)
        with quiet_backend():
            train_grpo(model=model, ref_model=ref_model, tokenizer=tokenizer,
                       optimizer=opt, train_dataset=train_set, val_dataset=val_set,
                       reward_funcs=list(reward_fns), args=args, training_callback=cb)
            mx.eval(model.parameters())
        del ref_model

        self._write_adapter_config(out, config, num_layers)
        self._train_config = config
        self._num_layers = num_layers
        self._tuned = True
        callbacks.log(f"[mlx] done · final loss {cb.last_loss} · adapter {out}")
        return FinetuneResult(checkpoint=str(out), final_loss=cb.last_loss)

    def _attach_retrieval_experts(self, dataset: Dataset, config: TrainConfig,
                               callbacks: Callbacks) -> int:
        """Build the fact index from the dataset and fuse it into attention."""
        from .. import more  # noqa: PLC0415

        if getattr(self, "_more_index", None) is None:
            chat = dataset.as_chat() if dataset.format != TEXT else None
            if chat is not None:
                inputs = [next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
                          for r in chat.rows]
                outputs = [next((m["content"] for m in reversed(r["messages"])
                                 if m["role"] == "assistant"), "")
                           for r in chat.rows]
            else:
                inputs = outputs = dataset.to_texts()
            with quiet_backend():
                self._more_index = more.MemoryIndex.build(inputs, outputs)
            callbacks.log(f"[more] indexed {len(self._more_index)} memory experts")

        from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: PLC0415

        n_layers = min(config.retrieval_layers, len(self.model.layers))
        if self._claim_surface(methods.ADAPTER_MORE):
            self.model.freeze()
            # LoRA gives the model capacity to *use* what the memory experts
            # retrieve; the retrieval projections alone are too small to learn it.
            linear_to_lora_layers(self.model, n_layers, self._lora_params(config))
            wrapped = more.attach(self.model, self._more_index,
                                  rank=config.lora_r, k=config.retrieval_k,
                                  num_layers=n_layers)
            if wrapped:
                callbacks.log(f"[more] memory attention + lora on {wrapped} layers "
                              f"(k={config.retrieval_k}, r={config.lora_r})")
        return n_layers

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
        fine_tune_type = spec.adapter if spec.adapter in ("lora", "dora") else "full"
        (out / "adapter_config.json").write_text(json.dumps({
            "fine_tune_type": fine_tune_type,
            "num_layers": num_layers,
            "lora_parameters": self._lora_params(config),
            "base_model": self.model_name,
        }, indent=2))

    def generate(self, prompt, *, max_new_tokens, temperature, top_p, **kwargs) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.chat([{"role": "user", "content": prompt}],
                             max_new_tokens=max_new_tokens, temperature=temperature,
                             top_p=top_p, **kwargs)
        return self._generate_text(prompt, max_new_tokens, temperature, top_p)

    def chat(self, messages, *, tools=None, max_new_tokens, temperature, top_p, **kwargs) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False,
        )
        return self._generate_text(text, max_new_tokens, temperature, top_p)

    def _generate_text(self, prompt: str, max_new_tokens, temperature, top_p) -> str:
        from mlx_lm import generate  # noqa: PLC0415
        from mlx_lm.sample_utils import make_sampler  # noqa: PLC0415

        sampler = make_sampler(temp=temperature, top_p=top_p)
        return generate(self.model, self.tokenizer, prompt=prompt,
                        max_tokens=max_new_tokens, sampler=sampler, verbose=False)

    def save(self, path: str, *, fmt: str = "adapter") -> str:
        import mlx.core as mx  # noqa: PLC0415
        from mlx.utils import tree_flatten  # noqa: PLC0415

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        cfg = getattr(self, "_train_config", None)
        special_kinds = (methods.ADAPTER_BITFIT, methods.ADAPTER_BOTTLENECK,
                         methods.ADAPTER_MORE)
        special = cfg is not None and methods.get(cfg.method).adapter in special_kinds
        if fmt == "merged":
            if special:
                raise RuntimeError(
                    f"fmt='merged' isn't supported for method {cfg.method!r}")
            # Save the full (LoRA-applied) weight set as a standalone model.
            self.model.save_weights(str(out / "model.safetensors"))
        else:
            adapter_weights = dict(tree_flatten(self.model.trainable_parameters()))
            mx.save_safetensors(str(out / "adapters.safetensors"), adapter_weights)
            kind = methods.get(cfg.method).adapter if cfg is not None else None
            if kind == methods.ADAPTER_BOTTLENECK:
                from .. import bottleneck  # noqa: PLC0415
                bottleneck.write_config(out, base_model=self.model_name,
                                        rank=cfg.lora_r)
            if kind == methods.ADAPTER_BITFIT:
                (out / "bitfit_config.json").write_text(json.dumps(
                    {"type": "bitfit", "base_model": self.model_name}, indent=2))
            if kind == methods.ADAPTER_MORE \
                    and getattr(self, "_more_index", None) is not None:
                from .. import more  # noqa: PLC0415
                self._more_index.save(out)
                more.write_config(out, base_model=self.model_name, rank=cfg.lora_r,
                                  k=cfg.retrieval_k,
                                  num_layers=getattr(self, "_num_layers", cfg.retrieval_layers))
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
        peak = info.get("peak_memory")
        self._cb.step(Metric(
            step=int(info.get("iteration", 0)),
            loss=self.last_loss,
            lr=float(info.get("learning_rate", 0.0)),
            elapsed_s=round(time.time() - self._start, 2),
            tokens=info.get("trained_tokens"),
            tokens_per_s=info.get("tokens_per_second"),
            peak_mem_gb=round(float(peak), 3) if peak is not None else None,
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
