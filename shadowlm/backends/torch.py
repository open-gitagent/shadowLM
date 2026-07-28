"""The torch backend — the PyTorch / HuggingFace training path.

datasets → transformers + trl.SFTTrainer + peft (LoRA/QLoRA) → generate. Runs on a
CUDA GPU (`device="cuda"`) or on the CPU (`device="cpu"`, the portable fallback for
boxes without a GPU). The shadow accelerator turns on flash-attention, gradient
checkpointing, and a fused optimizer when available.

Heavy libraries are imported lazily so the core SDK installs without them. This
path needs `pip install shadowlm[torch]`; it is not exercised on the Apple-Silicon
dev machine (which uses the mlx backend).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from .. import accel, methods
from .._quiet import quiet_backend
from ..data import Dataset
from ..training import Metric, TrainConfig, resolve_total_steps
from .base import Backend, Callbacks, FinetuneResult

_REQUIRED = ("torch", "transformers", "trl", "peft", "datasets")


def _save_kwargs(config) -> dict:
    """HF/trl Trainer args for periodic checkpoints — shared by every trl path
    (SFT, DPO, GRPO, MoRE) so `save_steps` writes `checkpoint-<step>/` dirs."""
    if config.save_steps is not None:
        return {"save_strategy": "steps", "save_steps": config.save_steps}
    return {}


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


class TorchBackend(Backend):
    name = "torch"

    # Whatever HuggingFace accelerate targets: cuda, cpu, xla (Google TPU and
    # AWS Trainium via torch-neuronx), xpu (Intel), mps. accelerate places the
    # model; we just avoid hard-coding CUDA-only paths for the others.
    _DEVICES = {"auto", "cuda", "cpu", "xla", "xpu", "mps"}

    def __init__(self, *, device: str = "auto", accelerator: str = "auto") -> None:
        super().__init__(device=device, accelerator=accelerator)
        if device not in self._DEVICES:
            raise ValueError(
                f"unknown device {device!r} (expected one of {sorted(self._DEVICES)}; "
                "cuda and cpu are tested, xla/xpu run through accelerate)")
        if device == "auto":
            self.device = "cuda" if self.has_cuda() else "cpu"

    @property
    def _is_extra_accelerator(self) -> bool:
        """A non-CUDA accelerator (TPU/Trainium/Intel) — accelerate places it."""
        return self.device not in ("cuda", "cpu")

    @classmethod
    def is_available(cls) -> bool:
        # Available wherever the torch stack is installed — CPU counts.
        return all(_has(m) for m in _REQUIRED)

    @classmethod
    def has_cuda(cls) -> bool:
        if not _has("torch"):
            return False
        try:
            import torch  # noqa: PLC0415
            return torch.cuda.is_available()
        except Exception as e:  # noqa: BLE001 — a broken CUDA install must not
            # abort backend selection, but falling back to CPU in silence is how
            # a "why is this so slow" afternoon starts. Say what happened.
            print(f"[torch] CUDA probe failed ({type(e).__name__}: {e}) — "
                  "treating this box as CPU-only", flush=True)
            return False

    def load(self, name, *, load_in_4bit=False, max_seq_length=2048, adapter=None, **kwargs) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self.model_name = name
        self.max_seq_length = max_seq_length

        attn = "flash_attention_2" if (
            self.accelerator != "none" and self.device == "cuda" and _has("flash_attn")
        ) else None

        quant = None
        if load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig  # noqa: PLC0415
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        with quiet_backend():  # swallow huggingface_hub download tqdm
            self.tokenizer = AutoTokenizer.from_pretrained(name, token=kwargs.get("hf_token"))
            self.model = AutoModelForCausalLM.from_pretrained(
                name,
                quantization_config=quant,
                torch_dtype="auto",
                device_map="auto" if self.device == "cuda" else None,
                attn_implementation=attn,
                token=kwargs.get("hf_token"),
            )
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        elif self._is_extra_accelerator:
            # accelerate / Trainer places the model on the TPU/Trainium/XPU device.
            import sys  # noqa: PLC0415
            print(f"[shadowlm] device={self.device!r} · placed by accelerate",
                  file=sys.__stdout__ or sys.stdout, flush=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if adapter:
            from peft import PeftModel  # noqa: PLC0415

            from .. import more  # noqa: PLC0415
            from .. import more_plus as mp  # noqa: PLC0415
            from .. import bottleneck  # noqa: PLC0415
            bn_cfg = bottleneck.read_config(adapter)
            bitfit_marker = (Path(adapter) / "bitfit_config.json").exists()
            more_cfg = more.read_config(adapter)
            more_plus_cfg = mp.read_config(adapter)
            if bitfit_marker:
                from safetensors.torch import load_file  # noqa: PLC0415
                self.model.load_state_dict(
                    load_file(str(Path(adapter) / "bitfit.safetensors")), strict=False)
                for pname, p in self.model.named_parameters():
                    p.requires_grad_(pname.endswith(".bias"))
                self.model._shadow_surface = methods.ADAPTER_BITFIT
            elif bn_cfg:
                for p in self.model.parameters():
                    p.requires_grad_(False)
                bottleneck.attach_torch(self.model, rank=bn_cfg["rank"])
                bottleneck.load_torch(self.model, adapter)
                self.model._shadow_surface = methods.ADAPTER_BOTTLENECK
            elif more_cfg:
                # Rebuild in the training order: wrap attention, peft on top,
                # then the trained wrapper weights and the fact index.
                self._more_index = more.MemoryIndex.load(adapter)
                more.attach_torch(self.model, self._more_index,
                                  rank=more_cfg["rank"], k=more_cfg["index_k"],
                                  num_layers=more_cfg["num_layers"])
                self.model = PeftModel.from_pretrained(self.model, adapter)
                more.load_torch_wrappers(self.model, adapter)
                self.model._shadow_surface = methods.ADAPTER_MORE
                self._more_meta = {"rank": more_cfg["rank"], "k": more_cfg["index_k"],
                                   "num_layers": more_cfg["num_layers"]}
            elif more_plus_cfg:
                # decoupled experts: base stays untouched at rest; chat() merges
                # the BM25-routed deltas into the final FFN per call, then restores.
                router = mp.BM25Router.from_dict(
                    json.loads((Path(adapter) / mp._INDEX_FILE).read_text()))
                experts = mp.load_experts(adapter)
                if len(experts) != router.N:
                    raise ValueError(
                        f"more_plus checkpoint is inconsistent: router has {router.N} "
                        f"experts but {len(experts)} deltas were loaded — the index "
                        "and experts file are out of sync.")
                down, _ = mp.final_ffn_module(self.model)
                self._more_plus = {
                    "router": router, "deltas": experts, "down": down,
                    "snapshot": down.weight.detach().clone(),
                    "k": more_plus_cfg["k"], "tau": more_plus_cfg.get("tau", 0.5),
                    "meta": more_plus_cfg,
                }
                self.model._shadow_surface = methods.ADAPTER_MORE_PLUS
            else:
                self.model = PeftModel.from_pretrained(self.model, adapter)

    def finetune(self, dataset: Dataset, config: TrainConfig, callbacks: Callbacks,
                 output_dir: str, eval_dataset: Dataset | None = None,
                 reward_fns: list | None = None) -> FinetuneResult:
        from datasets import Dataset as HFDataset  # noqa: PLC0415
        from transformers import TrainerCallback  # noqa: PLC0415
        from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

        shadow = accel.plan(
            self.accelerator, backend="torch",
            n_layers=getattr(self.model.config, "num_hidden_layers", 0),
            has_flash=_has("flash_attn"),
        )
        if self.accelerator != "none":
            callbacks.log(shadow.note)
        if shadow.fused_kernels:
            self._apply_liger(callbacks)

        # The method spec drives base requirements and the trainable surface —
        # no branching on method names here.
        spec = methods.get(config.method)
        spec.validate_base(
            quantized=getattr(self.model, "is_loaded_in_4bit", False),
            quantize_hint="load(..., load_in_4bit=True)",
            dequantize_hint="Load without load_in_4bit",
        )
        if spec.trainer == "dpo":
            return self._finetune_dpo(dataset, config, callbacks, output_dir,
                                      eval_dataset, spec)
        if spec.trainer == "grpo":
            return self._finetune_grpo(dataset, config, callbacks, output_dir,
                                       spec, reward_fns)
        if spec.adapter == methods.ADAPTER_MORE:
            return self._finetune_more(dataset, config, callbacks, output_dir,
                                       eval_dataset)
        if spec.adapter == methods.ADAPTER_MORE_PLUS:
            return self._finetune_more_plus(dataset, config, callbacks, output_dir)

        # Attach whatever surface the method trains (LoRA family, soft-prompt
        # family, biases, bottlenecks, or nothing for full fine-tunes).
        self._attach_trainable(spec, config, callbacks)

        train_ds = self._train_dataset(dataset, raw_text=spec.raw_text)
        has_eval = eval_dataset is not None and len(eval_dataset) > 0
        eval_ds = (self._train_dataset(eval_dataset, raw_text=spec.raw_text)
                   if has_eval else None)
        optim_name = self._resolved_optim(config, shadow)

        # Bridge the transformers Trainer callbacks → our Callbacks.
        class _Bridge(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                if not logs or "loss" not in logs:
                    return
                callbacks.step(Metric(
                    step=state.global_step,
                    loss=float(logs["loss"]),
                    lr=float(logs.get("learning_rate", 0.0)),
                    grad_norm=float(logs["grad_norm"]) if "grad_norm" in logs else None,
                    epoch=float(logs.get("epoch", 0.0)),
                    tokens=int(logs["num_tokens"]) if "num_tokens" in logs else None,
                ))

            def on_evaluate(self, args, state, control, metrics=None, **kw):
                if metrics and "eval_loss" in metrics:
                    callbacks.eval(Metric(
                        step=state.global_step,
                        loss=float(metrics["eval_loss"]),
                        epoch=float(metrics.get("epoch", 0.0)),
                    ))

            def on_step_end(self, args, state, control, **kw):
                if callbacks.stopped():
                    control.should_training_stop = True
                return control

        total = resolve_total_steps(config, len(dataset))
        extra = {}
        if config.max_grad_norm is not None:
            extra["max_grad_norm"] = config.max_grad_norm
        if config.save_steps is not None:
            extra["save_strategy"] = "steps"
            extra["save_steps"] = config.save_steps
        if config.train_on_completions:
            # trl masks the prompt when the dataset has prompt/completion columns;
            # our text-rendered rows don't, so be explicit rather than silent.
            callbacks.log("[torch] note: train_on_completions needs prompt/completion "
                          "columns — not applied to text-rendered rows")

        args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.resolved_warmup(total),
            max_steps=total,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            optim=optim_name,
            gradient_checkpointing=shadow.grad_checkpoint,
            packing=config.packing,
            logging_steps=config.logging_steps,
            eval_strategy="steps" if has_eval else "no",
            eval_steps=config.resolved_eval_steps(total),
            per_device_eval_batch_size=config.per_device_train_batch_size,
            seed=config.seed,
            max_length=config.max_seq_length,
            use_cpu=(self.device == "cpu"),
            disable_tqdm=True,  # shadowLM prints its own progress
            report_to=list(config.report_to),
            **extra,
        )
        with quiet_backend():
            trainer = SFTTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                args=args,
                callbacks=[_Bridge()],
            )
            result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
        self._restore_inference_state()
        self._save_trained(trainer, spec, config, output_dir)
        final_loss = float(result.training_loss) if result else None
        callbacks.log(f"[torch:{self.device}] done · final loss {final_loss} · checkpoint {output_dir}")
        return FinetuneResult(checkpoint=output_dir, final_loss=final_loss)

    def _finetune_grpo(self, dataset: Dataset, config: TrainConfig, callbacks: Callbacks,
                       output_dir: str, spec, reward_fns: list | None) -> FinetuneResult:
        """RL from programmable rewards via trl's GRPOTrainer."""
        from datasets import Dataset as HFDataset  # noqa: PLC0415
        from transformers import TrainerCallback  # noqa: PLC0415
        from trl import GRPOConfig, GRPOTrainer  # noqa: PLC0415

        if len(dataset) and "weight" in dataset.rows[0] and "messages" in dataset.rows[0]:
            # trajectory-native: pre-collected rollouts with group advantages
            return self._finetune_trajectory_grpo(dataset, config, callbacks,
                                                  output_dir, spec)
        if not reward_fns:
            raise ValueError(
                "method='grpo' needs reward_fns=[...] (or TrajectoryGroups) — "
                "each fn is fn(prompts, completions, answer, types=None) -> list[float]"
            )
        if not len(dataset) or "prompt" not in dataset.rows[0]:
            raise ValueError("method='grpo' needs rows with a 'prompt' column")
        rows = [{"prompt": r["prompt"], "answer": r.get("answer", "")}
                for r in dataset.rows]

        def adapt(fn):
            # shadowLM reward contract -> trl calling convention. trl passes
            # completions as strings (standard format) or message lists
            # (conversational); normalize to strings.
            def wrapped(prompts=None, completions=None, **kw):
                texts = [c if isinstance(c, str)
                         else (c[-1].get("content", "") if c else "")
                         for c in completions]
                return fn(prompts=prompts, completions=texts,
                          answer=kw.get("answer"), types=kw.get("types"))
            wrapped.__name__ = getattr(fn, "__name__", "reward")
            return wrapped

        peft_config = None
        if spec.trains_adapters and getattr(self.model, "peft_config", None) is None:
            from peft import LoraConfig  # noqa: PLC0415
            peft_config = LoraConfig(
                r=config.lora_r, lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.resolved_target_modules()),
                bias="none", task_type="CAUSAL_LM",
            )

        class _Bridge(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                if logs and "loss" in logs:
                    callbacks.step(Metric(step=state.global_step, loss=float(logs["loss"]),
                                          lr=float(logs.get("learning_rate", 0.0))))

        # trl requires the generation batch to be divisible by the group size.
        group = config.grpo_group_size
        args = GRPOConfig(
            output_dir=output_dir,
            num_generations=group,
            per_device_train_batch_size=group,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_completion_length=config.grpo_max_completion_length,
            beta=config.beta,
            max_steps=resolve_total_steps(config, len(rows)),
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_scheduler_type,
            logging_steps=config.logging_steps,
            seed=config.seed,
            use_cpu=(self.device == "cpu"),
            disable_tqdm=True,
            report_to=list(config.report_to),
            **_save_kwargs(config),
        )
        callbacks.log(
            f"[torch:{self.device}] grpo on {self.model_name} · {len(rows)} prompts · "
            f"group {group} · {len(reward_fns)} reward fn(s)"
        )
        with quiet_backend():
            trainer = GRPOTrainer(
                model=self.model,
                reward_funcs=[adapt(f) for f in reward_fns],
                args=args,
                train_dataset=HFDataset.from_list(rows),
                processing_class=self.tokenizer,
                peft_config=peft_config,
                callbacks=[_Bridge()],
            )
            result = trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.model = trainer.model
        self._restore_inference_state()
        final_loss = float(result.training_loss) if result else None
        callbacks.log(f"[torch:{self.device}] grpo done · {output_dir}")
        return FinetuneResult(checkpoint=output_dir, final_loss=final_loss)

    def _finetune_trajectory_grpo(self, dataset: Dataset, config: TrainConfig,
                                  callbacks: Callbacks, output_dir: str,
                                  spec) -> FinetuneResult:
        """Advantage-weighted policy gradient over collected trajectories.

        Rows are {"messages", "weight"} (weight = group-relative advantage);
        loss = weight × NLL of the assistant tokens. Eager custom loop —
        per-sequence weighting isn't expressible in the stock trainers.
        """
        import random as _random  # noqa: PLC0415
        import time as _time  # noqa: PLC0415

        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        already_peft = getattr(self.model, "peft_config", None) is not None
        if spec.trains_adapters and not already_peft:
            from peft import LoraConfig, get_peft_model  # noqa: PLC0415
            self.model = get_peft_model(self.model, LoraConfig(
                r=config.lora_r, lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.resolved_target_modules()),
                bias="none", task_type="CAUSAL_LM",
            ))

        def ids(msgs, **kw):
            out = self.tokenizer.apply_chat_template(msgs, tokenize=True,
                                                     return_dict=True, **kw)
            return list(out["input_ids"])

        examples = []
        for row in dataset.rows:
            msgs = row["messages"]
            full = ids(msgs)[:config.max_seq_length]
            prompt_len = len(ids(msgs[:-1], add_generation_prompt=True))
            if prompt_len >= len(full):
                continue
            examples.append((full, prompt_len, float(row["weight"])))
        if not examples:
            raise ValueError("no usable trajectories after tokenization")

        device = next(self.model.parameters()).device
        pad = self.tokenizer.pad_token_id
        batch_size = max(1, min(config.per_device_train_batch_size, len(examples)))
        iters = resolve_total_steps(config, len(examples))
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=config.learning_rate,
                                weight_decay=config.weight_decay)
        rng = _random.Random(config.seed)

        def batches():
            while True:
                order = list(range(len(examples)))
                rng.shuffle(order)
                for i in range(0, len(order) - batch_size + 1, batch_size):
                    chunk = [examples[j] for j in order[i:i + batch_size]]
                    L = max(len(t) for t, _, _ in chunk)
                    toks = torch.tensor(
                        [t + [pad] * (L - len(t)) for t, _, _ in chunk], device=device)
                    mask = torch.tensor([
                        [1.0 if p <= j + 1 < len(t) else 0.0 for j in range(L - 1)]
                        for t, p, _ in chunk], device=device)
                    weights = torch.tensor([w for _, _, w in chunk], device=device)
                    yield toks, mask, weights

        callbacks.log(
            f"[torch:{self.device}] trajectory-grpo on {self.model_name} · "
            f"{len(examples)} trajectories · {iters} iters · lr {config.learning_rate:g}"
        )
        self.model.train()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        start = _time.time()
        accum = max(1, config.gradient_accumulation_steps)
        last_loss = None
        for it, (toks, mask, weights) in zip(range(1, iters + 1), batches()):
            logits = self.model(toks[:, :-1]).logits.float()
            ce = F.cross_entropy(logits.transpose(1, 2), toks[:, 1:],
                                 reduction="none")
            per_seq = (ce * mask).sum(-1) / mask.sum(-1).clamp(min=1)
            loss = (weights * per_seq).mean() / accum
            loss.backward()
            if it % accum == 0:
                opt.step()
                opt.zero_grad()
            last_loss = round(float(loss) * accum, 4)
            if it % config.logging_steps == 0:
                callbacks.step(Metric(step=it, loss=last_loss,
                                      lr=config.learning_rate,
                                      elapsed_s=round(_time.time() - start, 2)))
            if config.save_steps and it % config.save_steps == 0 and it < iters:
                self.model.save_pretrained(str(out / f"checkpoint-{it}"))  # HF layout
                callbacks.log(f"[torch] checkpoint @ step {it}")
            if callbacks.stopped():
                break
        self._restore_inference_state()

        self.model.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))
        callbacks.log(f"[torch:{self.device}] done · final pg loss {last_loss} · {out}")
        return FinetuneResult(checkpoint=str(out), final_loss=last_loss)

    def _train_dataset(self, dataset: Dataset, *, raw_text: bool = False):
        """An HF dataset rendered the way inference will see it.

        Chat-able rows are passed as conversational rows ({"messages": ...}) so
        trl applies the chat template itself — training on alpaca-style raw
        text while inference uses the chat template wrecks generation once
        memorization kicks in, and pre-rendering the template to text gets its
        special tokens re-split. raw_text (CPT) trains on plain text.
        """
        from datasets import Dataset as HFDataset  # noqa: PLC0415

        from ..data import CHAT, INSTRUCTION, SHAREGPT  # noqa: PLC0415

        if raw_text or dataset.format not in (CHAT, SHAREGPT, INSTRUCTION) \
                or not getattr(self.tokenizer, "chat_template", None):
            return HFDataset.from_dict({"text": dataset.to_texts()})
        chat = dataset.as_chat()
        rows = chat.rows
        # prompt/completion schema → trl masks the prompt by default (loss on
        # the assistant reply only), which keeps small models from memorizing
        # system headers verbatim on fact-style data
        if all(r["messages"] and r["messages"][-1]["role"] == "assistant" for r in rows):
            return HFDataset.from_list([
                {"prompt": r["messages"][:-1], "completion": [r["messages"][-1]]}
                for r in rows
            ])
        return HFDataset.from_list([{"messages": r["messages"]} for r in rows])

    def _finetune_more(self, dataset: Dataset, config: TrainConfig, callbacks: Callbacks,
                       output_dir: str, eval_dataset: Dataset | None = None) -> FinetuneResult:
        """Mixture of Retrieval Experts on torch.

        Eager execution lets retrieval run inside forward under no_grad, so the
        standard SFTTrainer drives training. LoRA rides alongside the wrapper
        projections for capacity, exactly like the mlx implementation.
        """
        from datasets import Dataset as HFDataset  # noqa: PLC0415
        from peft import LoraConfig, get_peft_model  # noqa: PLC0415
        from transformers import TrainerCallback  # noqa: PLC0415
        from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

        from .. import more  # noqa: PLC0415

        existing = self._attached_surface()
        if existing not in (None, methods.ADAPTER_MORE):
            raise RuntimeError(
                f"this model already hosts a {existing!r} trainable surface; "
                "load a fresh model for method='more'."
            )
        # 1. index the facts (embedding runs in a subprocess)
        chat = dataset.as_chat() if dataset.format != "text" else None
        if chat is not None:
            inputs = [next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
                      for r in chat.rows]
            outputs = [next((m["content"] for m in reversed(r["messages"])
                             if m["role"] == "assistant"), "")
                       for r in chat.rows]
        else:
            inputs = outputs = dataset.to_texts()
        if existing == methods.ADAPTER_MORE:
            index = self._more_index  # continued training reuses the surface
        else:
            index = more.MemoryIndex.build(inputs, outputs)
            self._more_index = index
            callbacks.log(f"[more] indexed {len(index)} retrieval experts")

        # 2. wrap attention, then LoRA on top (peft matches q_proj etc. by
        #    suffix, so it still finds them inside the wrapper)
        n_layers = min(config.retrieval_layers,
                       len((self.model.model if hasattr(self.model, "model")
                            else self.model).layers))
        if existing != methods.ADAPTER_MORE:
            wrapped = more.attach_torch(self.model, index, rank=config.lora_r,
                                        k=config.retrieval_k, num_layers=n_layers)
            self.model = get_peft_model(self.model, LoraConfig(
                r=config.lora_r, lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.resolved_target_modules()),
                bias="none", task_type="CAUSAL_LM",
            ))
            # peft froze everything but LoRA — the retrieval projections train too
            for name, param in self.model.named_parameters():
                if any(f".{p}." in name for p in more.WRAPPER_PARAM_NAMES):
                    param.requires_grad_(True)
            self.model._shadow_surface = methods.ADAPTER_MORE
            self._more_meta = {"rank": config.lora_r, "k": config.retrieval_k,
                               "num_layers": n_layers}
            callbacks.log(f"[more] memory attention + lora on {wrapped} layers "
                          f"(k={config.retrieval_k}, r={config.lora_r})")

        # 3. standard supervised training over the facts (chat-templated, so
        #    training and inference see the same rendering)
        train_ds = self._train_dataset(dataset)
        has_eval = eval_dataset is not None and len(eval_dataset) > 0
        eval_ds = self._train_dataset(eval_dataset) if has_eval else None

        more_total = resolve_total_steps(config, len(dataset))

        class _Bridge(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                if logs and "loss" in logs:
                    callbacks.step(Metric(step=state.global_step, loss=float(logs["loss"]),
                                          lr=float(logs.get("learning_rate", 0.0))))

            def on_evaluate(self, args, state, control, metrics=None, **kw):
                if metrics and "eval_loss" in metrics:
                    callbacks.eval(Metric(step=state.global_step,
                                          loss=float(metrics["eval_loss"])))

        args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.resolved_warmup(more_total),
            max_steps=more_total,
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_scheduler_type,
            optim=self._resolved_optim(config, accel.plan("none", backend="torch")),
            logging_steps=config.logging_steps,
            eval_strategy="steps" if has_eval else "no",
            eval_steps=config.resolved_eval_steps(more_total),
            seed=config.seed,
            max_length=config.max_seq_length,
            use_cpu=(self.device == "cpu"),
            disable_tqdm=True,
            report_to=[],
            **_save_kwargs(config),
        )
        callbacks.log(
            f"[torch:{self.device}] more on {self.model_name} · {len(dataset)} facts · "
            f"retrieval k={config.retrieval_k} on {n_layers} layers"
        )
        with quiet_backend():
            trainer = SFTTrainer(model=self.model, processing_class=self.tokenizer,
                                 train_dataset=train_ds, eval_dataset=eval_ds,
                                 args=args, callbacks=[_Bridge()])
            result = trainer.train()
        self.model = trainer.model
        self._restore_inference_state()

        # 4. persist: peft adapter + wrapper weights + index + config
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        more.save_torch_wrappers(self.model, out)
        index.save(out)
        more.write_config(out, base_model=self.model_name, rank=config.lora_r,
                          k=config.retrieval_k, num_layers=n_layers)
        final_loss = float(result.training_loss) if result else None
        callbacks.log(f"[torch:{self.device}] more done · final loss {final_loss} · {out}")
        return FinetuneResult(checkpoint=str(out), final_loss=final_loss)

    def _finetune_more_plus(self, dataset: Dataset, config: TrainConfig,
                            callbacks: Callbacks, output_dir: str) -> FinetuneResult:
        """MoRE+ — one tiny final-FFN LoRA expert per knowledge unit (DMoE-style).

        For each unit we attach a manual LoRA to the last block's down_proj (base
        frozen), train it with a short eager loop, collapse it to a weight delta,
        and discard it — so only one tiny expert is ever live (flat memory). A
        BM25 router over the unit surrogates ships alongside the deltas. Inference
        merges the routed deltas into that one FFN weight (cache-safe).
        """
        import math as _math  # noqa: PLC0415

        import torch  # noqa: PLC0415

        from .. import more_plus as mp  # noqa: PLC0415

        existing = self._attached_surface()
        if existing is not None:
            raise RuntimeError(
                f"this model already hosts a {existing!r} trainable surface; "
                "load a fresh model for method='more_plus' (continued training "
                "on an existing more_plus checkpoint isn't supported in v1)."
            )

        down, last_idx = mp.final_ffn_module(self.model)
        out_f, in_f = down.weight.shape
        device = down.weight.device
        r = config.lora_r or 4
        alpha = config.lora_alpha or r
        scaling = alpha / r
        # steps/expert scale with the final-FFN width unless the user pins them —
        # a bigger base's wider FFN needs more steps to converge at a fixed lr.
        steps = config.more_plus_expert_steps or mp.auto_steps(in_f)

        units = mp.split_units(dataset, config.more_plus_group_size)
        # drop units whose surrogate has no routable tokens — they'd train an
        # expert that BM25 can never select (a dead, wasted slot).
        kept = [(s, rows) for s, rows in units if mp._tokenize(s)]
        if len(kept) != len(units):
            callbacks.log(f"[more+] skipped {len(units) - len(kept)} unit(s) with "
                          "empty/untokenizable input — not routable")
        units = kept
        if not units:
            raise ValueError(
                "more_plus: no routable knowledge units — every row's input side is "
                "empty. Provide rows with an instruction/question/prompt (or a user turn)."
            )
        callbacks.log(f"[more+] {len(units)} knowledge units → final-FFN experts "
                      f"(layer {last_idx}, r={r}, {steps} steps/expert)")

        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        # training drives the model under grad, not generation — drop the KV cache
        # (avoidable per-forward peak memory), restored before we return.
        prev_use_cache = getattr(self.model.config, "use_cache", None)
        if prev_use_cache is not None:
            self.model.config.use_cache = False

        from ..data import CHAT, INSTRUCTION, SHAREGPT  # noqa: PLC0415

        chatable = (dataset.format in (CHAT, SHAREGPT, INSTRUCTION)
                    and getattr(self.tokenizer, "chat_template", None))

        def _unit_token_batches(rows):
            ds_u = Dataset.from_list(rows)
            if chatable:  # train on the chat-templated turn, as inference sees it
                return [self.tokenizer.apply_chat_template(
                            r["messages"], tokenize=True, return_dict=True,
                            return_tensors="pt")["input_ids"].to(device)
                        for r in ds_u.as_chat().rows]
            return [self.tokenizer(t, return_tensors="pt", truncation=True,
                                   max_length=config.max_seq_length)["input_ids"].to(device)
                    for t in ds_u.to_texts()]

        deltas: dict[int, "torch.Tensor"] = {}
        surrogates: list[str] = []
        expert_losses: list[float] = []
        last_loss = None
        for i, (surrogate, rows) in enumerate(units):
            surrogates.append(surrogate)
            batches = _unit_token_batches(rows)
            # manual LoRA on down_proj: out += scaling · (x A^T) B^T ; A random, B zero
            A = torch.empty(r, in_f, device=device, dtype=torch.float32)
            torch.nn.init.kaiming_uniform_(A, a=_math.sqrt(5))
            B = torch.zeros(out_f, r, device=device, dtype=torch.float32)
            A.requires_grad_(True)
            B.requires_grad_(True)

            def _hook(_mod, inp, output, A=A, B=B):
                x = inp[0].to(torch.float32)
                return output + scaling * (x @ A.t() @ B.t()).to(output.dtype)

            handle = down.register_forward_hook(_hook)
            opt = torch.optim.AdamW([A, B], lr=config.learning_rate or 1e-4)
            try:
                step = 0
                while step < steps and batches:
                    for ids in batches:
                        if step >= steps:
                            break
                        loss = self.model(input_ids=ids, labels=ids).loss
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        last_loss = float(loss.detach())
                        step += 1
            finally:
                handle.remove()
            with torch.no_grad():
                deltas[i] = (scaling * (B.detach() @ A.detach())).cpu().float()
            expert_losses.append(last_loss if last_loss is not None else 0.0)
            callbacks.step(Metric(step=i + 1, loss=last_loss if last_loss is not None else 0.0))

        mp.warn_undertrained(expert_losses, steps, callbacks.log)

        if prev_use_cache is not None:
            self.model.config.use_cache = prev_use_cache

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        router = mp.BM25Router.build(surrogates, embed=True)
        callbacks.log("[more+] router: BM25 + semantic" if router.emb is not None
                      else "[more+] router: BM25 only (sentence-transformers not installed)")
        mp.save_experts(deltas, out)
        (out / mp._INDEX_FILE).write_text(json.dumps(router.to_dict()))
        mp.write_config(out, base_model=self.model_name, lora_r=r, lora_alpha=alpha,
                        final_layer_idx=last_idx, num_experts=len(units),
                        k=config.more_plus_k, tau=config.more_plus_tau,
                        group_size=config.more_plus_group_size)
        self.tokenizer.save_pretrained(str(out))
        self.model._shadow_surface = methods.ADAPTER_MORE_PLUS
        # hold state so save() can re-emit, and so this session can route/generate
        self._more_plus = {"router": router, "deltas": deltas, "down": down,
                           "snapshot": down.weight.detach().clone(),
                           "k": config.more_plus_k, "tau": config.more_plus_tau,
                           "meta": {"lora_r": r, "lora_alpha": alpha,
                                    "final_layer_idx": last_idx,
                                    "group_size": config.more_plus_group_size}}
        callbacks.log(f"[more+] {len(units)} experts saved · {out}")
        return FinetuneResult(checkpoint=str(out), final_loss=last_loss)

    def _attached_surface(self) -> str | None:
        """Which trainable surface this model currently hosts, if any."""
        marked = getattr(self.model, "_shadow_surface", None)
        if marked:
            return marked
        if getattr(self.model, "peft_config", None) is not None:
            return "peft"
        return None

    def _attach_trainable(self, spec, config: TrainConfig, callbacks: Callbacks) -> None:
        """Attach the method's trainable surface once.

        A model hosts exactly one surface kind: repeating the same kind reuses
        it (continued training); mixing kinds raises instead of silently
        training one surface while saving another.
        """
        adapter = spec.adapter
        peft_family = (methods.ADAPTER_LORA, methods.ADAPTER_DORA,
                       methods.ADAPTER_PROMPT, methods.ADAPTER_PTUNING)
        existing = self._attached_surface()
        if existing is not None:
            # "peft" is the fallback for externally loaded adapters whose exact
            # kind we can't know — any peft-family method may continue those.
            if existing == adapter or (existing == "peft" and adapter in peft_family):
                if adapter in (methods.ADAPTER_BITFIT, methods.ADAPTER_BOTTLENECK):
                    self._enable_checkpointable_inputs()  # idempotent; reload paths skip it
                return  # same surface — continue training it
            raise RuntimeError(
                f"this model already hosts a {existing!r} trainable surface; "
                f"method {config.method!r} needs {adapter!r}. Load a fresh model."
            )
        if adapter == methods.ADAPTER_NONE:
            return  # full fine-tune: everything already trains

        if adapter in (methods.ADAPTER_LORA, methods.ADAPTER_DORA):
            from peft import LoraConfig, get_peft_model  # noqa: PLC0415
            self.model = get_peft_model(self.model, LoraConfig(
                r=config.lora_r, lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.resolved_target_modules()),
                use_dora=(adapter == methods.ADAPTER_DORA),
                use_rslora=config.use_rslora,
                bias="none", task_type="CAUSAL_LM",
            ))
            self.model._shadow_surface = adapter
        elif adapter in (methods.ADAPTER_PROMPT, methods.ADAPTER_PTUNING):
            from peft import (  # noqa: PLC0415
                PromptEncoderConfig,
                PromptTuningConfig,
                get_peft_model,
            )
            cfg_cls = {methods.ADAPTER_PROMPT: PromptTuningConfig,
                       methods.ADAPTER_PTUNING: PromptEncoderConfig}[adapter]
            self.model = get_peft_model(self.model, cfg_cls(
                task_type="CAUSAL_LM",
                num_virtual_tokens=config.num_virtual_tokens,
            ))
            self.model._shadow_surface = adapter
            callbacks.log(f"[torch] {adapter}: {config.num_virtual_tokens} virtual tokens")
        elif adapter == methods.ADAPTER_BITFIT:
            for p in self.model.parameters():
                p.requires_grad_(False)
            n = 0
            for name, p in self.model.named_parameters():
                if name.endswith(".bias"):
                    p.requires_grad_(True)
                    n += p.numel()
            if n == 0:
                raise RuntimeError(
                    "method='bitfit' found no bias parameters — this architecture "
                    "(Llama-style) has none. Use a Qwen-family model, or another method."
                )
            self._enable_checkpointable_inputs()
            self.model._shadow_surface = methods.ADAPTER_BITFIT
            callbacks.log(f"[torch] bitfit: training {n:,} bias parameters")
        elif adapter == methods.ADAPTER_BOTTLENECK:
            from .. import bottleneck  # noqa: PLC0415
            for p in self.model.parameters():
                p.requires_grad_(False)
            wrapped = bottleneck.attach_torch(self.model, rank=config.lora_r)
            self._enable_checkpointable_inputs()
            self.model._shadow_surface = methods.ADAPTER_BOTTLENECK
            callbacks.log(f"[torch] bottleneck adapters (r={config.lora_r}) on {wrapped} layers")
        else:
            raise RuntimeError(f"torch backend has no attach path for adapter kind {adapter!r}")

    def _apply_liger(self, callbacks) -> None:
        """Patch the loaded model with Liger's fused Triton kernels (Apache-2.0).
        CUDA/Triton only; falls back cleanly when unavailable or unsupported."""
        if self.device != "cuda":
            callbacks.log("[shadow] liger kernels need a CUDA GPU — skipping")
            return
        try:
            from liger_kernel.transformers import (  # noqa: PLC0415
                _apply_liger_kernel_to_instance)
            # Keep the fused RMSNorm/RoPE/SwiGLU kernels, but NOT fused-linear-
            # cross-entropy: FLCE computes the loss without materializing logits
            # (outputs.logits=None), and trl's SFTTrainer reads outputs.logits for
            # its per-token-entropy metric → crashes. Disabling FLCE keeps the
            # logits while still getting the rest of liger's speedup.
            _apply_liger_kernel_to_instance(model=self.model,
                                            fused_linear_cross_entropy=False)
        except Exception as e:  # noqa: BLE001 — model arch may be unsupported
            callbacks.log(f"[shadow] liger not applied ({e}); continuing without")

    def _enable_checkpointable_inputs(self) -> None:
        """Gradient checkpointing needs an input that requires grad; peft does
        this itself, custom surfaces (bitfit/bottleneck) must ask for it."""
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

    def _save_trained(self, trainer, spec, config: TrainConfig, output_dir: str) -> None:
        """Persist what the method trained, in a form load(adapter=) can rebuild."""
        import json as _json  # noqa: PLC0415

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        adapter = spec.adapter
        if adapter == methods.ADAPTER_BITFIT:
            from safetensors.torch import save_file  # noqa: PLC0415
            state = {k: v.contiguous() for k, v in self.model.state_dict().items()
                     if k.endswith(".bias")}
            save_file(state, str(out / "bitfit.safetensors"))
            (out / "bitfit_config.json").write_text(_json.dumps(
                {"type": "bitfit", "base_model": self.model_name}, indent=2))
        elif adapter == methods.ADAPTER_BOTTLENECK:
            from .. import bottleneck  # noqa: PLC0415
            bottleneck.save_torch(self.model, out)
            bottleneck.write_config(out, base_model=self.model_name, rank=config.lora_r)
        else:
            trainer.save_model(str(out))
        self.tokenizer.save_pretrained(str(out))

    def _restore_inference_state(self) -> None:
        """Trainers leave train-mode residue (use_cache off, train() flags)."""
        self.model.eval()
        cfg = getattr(self.model, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = True

    def _resolved_optim(self, config: TrainConfig, shadow) -> str:
        """Pick an optimizer that exists on this device.

        8-bit (bitsandbytes) and fused AdamW are CUDA-only; on CPU fall back to
        plain adamw_torch instead of crashing.
        """
        if self.device != "cuda":
            if config.optim.endswith("8bit") or "paged" in config.optim:
                return "adamw_torch"
            return config.optim if config.optim != "adamw_torch_fused" else "adamw_torch"
        return "adamw_torch_fused" if shadow.fused_optimizer else config.optim

    def _finetune_dpo(self, dataset: Dataset, config: TrainConfig, callbacks: Callbacks,
                      output_dir: str, eval_dataset: Dataset | None, spec) -> FinetuneResult:
        """Preference training via trl's DPOTrainer (frozen-reference handled by trl)."""
        from datasets import Dataset as HFDataset  # noqa: PLC0415
        from transformers import TrainerCallback  # noqa: PLC0415
        from trl import DPOConfig, DPOTrainer  # noqa: PLC0415

        keys = ("prompt", "chosen", "rejected")
        missing = set(keys) - set(dataset.rows[0] if len(dataset) else {})
        if missing:
            raise ValueError(
                f"method='dpo' needs preference rows with prompt/chosen/rejected "
                f"(missing: {', '.join(sorted(missing))})"
            )

        peft_config = None
        if spec.trains_adapters and getattr(self.model, "peft_config", None) is None:
            from peft import LoraConfig  # noqa: PLC0415
            peft_config = LoraConfig(
                r=config.lora_r, lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.resolved_target_modules()),
                use_dora=(spec.adapter == methods.ADAPTER_DORA),
                bias="none", task_type="CAUSAL_LM",
            )

        def rows(ds):
            return HFDataset.from_list([{k: r[k] for k in keys} for r in ds.rows])

        has_eval = eval_dataset is not None and len(eval_dataset) > 0
        total = resolve_total_steps(config, len(dataset))

        class _Bridge(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                if logs and "loss" in logs:
                    callbacks.step(Metric(step=state.global_step, loss=float(logs["loss"]),
                                          lr=float(logs.get("learning_rate", 0.0))))

            def on_evaluate(self, args, state, control, metrics=None, **kw):
                if metrics and "eval_loss" in metrics:
                    callbacks.eval(Metric(step=state.global_step, loss=float(metrics["eval_loss"])))

        args = DPOConfig(
            output_dir=output_dir,
            beta=config.beta,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.resolved_warmup(total),
            max_steps=total,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            logging_steps=config.logging_steps,
            eval_strategy="steps" if has_eval else "no",
            eval_steps=config.resolved_eval_steps(total),
            seed=config.seed,
            use_cpu=(self.device == "cpu"),
            disable_tqdm=True,
            report_to=list(config.report_to),
            **_save_kwargs(config),
        )
        with quiet_backend():
            trainer = DPOTrainer(
                model=self.model,
                args=args,
                train_dataset=rows(dataset),
                eval_dataset=rows(eval_dataset) if has_eval else None,
                processing_class=self.tokenizer,
                peft_config=peft_config,
                callbacks=[_Bridge()],
            )
            result = trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.model = trainer.model
        self._restore_inference_state()
        final_loss = float(result.training_loss) if result else None
        callbacks.log(f"[torch:{self.device}] dpo done · final loss {final_loss} · {output_dir}")
        return FinetuneResult(checkpoint=output_dir, final_loss=final_loss)

    def generate(self, prompt, *, max_new_tokens, temperature, top_p, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}],
                         max_new_tokens=max_new_tokens, temperature=temperature,
                         top_p=top_p, **kwargs)

    def chat(self, messages, *, tools=None, max_new_tokens, temperature, top_p, **kwargs) -> str:
        import torch  # noqa: PLC0415

        enc = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        sampling = ({"do_sample": True, "temperature": temperature, "top_p": top_p}
                    if temperature > 0 else {"do_sample": False})
        # MoRE+: BM25-route the prompt, merge the top-k expert deltas into the final
        # FFN for this call (cache-safe — only a post-attention weight changes), then
        # restore from the pristine snapshot so calls are stateless and drift-free.
        merged = self._more_plus_merge(messages, tools)
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **sampling,
                )
        finally:
            if merged:
                self._more_plus_restore()
        prompt_len = enc["input_ids"].shape[1]
        return self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

    def _more_plus_merge(self, messages, tools) -> bool:
        """Merge the BM25-routed expert deltas into the final FFN. Returns whether
        anything was merged (so chat() knows to restore afterward)."""
        state = getattr(self, "_more_plus", None)
        if not state or not state["router"].N:
            return False
        import torch  # noqa: PLC0415

        # Route on the CURRENT user turn only — the router was indexed on each
        # unit's user-side surrogate, so ranking the full chat-templated prompt
        # (system + prior turns + role markers) would mismatch and let stale
        # context dominate BM25. Fall back to the whole template if no user turn.
        query = next((m.get("content", "") for m in reversed(messages)
                      if m.get("role") == "user"), "")
        if not query:
            query = self.tokenizer.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True, tokenize=False)
        ids = [i for i, _ in state["router"].rank(query, state["k"])]
        if not ids:
            return False  # no term overlap → run on the clean base
        weight = state["down"].weight
        with torch.no_grad():
            acc = state["snapshot"].to(weight.device, weight.dtype).clone()
            for i in ids:
                d = state["deltas"].get(i)
                if d is not None:
                    acc += d.to(weight.device, weight.dtype)
            weight.copy_(acc)
        return True

    def _more_plus_restore(self) -> None:
        import torch  # noqa: PLC0415

        state = self._more_plus
        weight = state["down"].weight
        with torch.no_grad():
            weight.copy_(state["snapshot"].to(weight.device, weight.dtype))

    def save(self, path: str, *, fmt: str = "adapter") -> str:
        import json as _json  # noqa: PLC0415

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        surface = self._attached_surface()
        if fmt == "merged" and surface in (methods.ADAPTER_BITFIT,
                                           methods.ADAPTER_BOTTLENECK,
                                           methods.ADAPTER_MORE,
                                           methods.ADAPTER_MORE_PLUS):
            raise RuntimeError(f"fmt='merged' isn't supported for the {surface!r} surface")
        if surface == methods.ADAPTER_BITFIT:
            from safetensors.torch import save_file  # noqa: PLC0415
            state = {k: v.contiguous() for k, v in self.model.state_dict().items()
                     if k.endswith(".bias")}
            save_file(state, str(out / "bitfit.safetensors"))
            (out / "bitfit_config.json").write_text(_json.dumps(
                {"type": "bitfit", "base_model": self.model_name}, indent=2))
        elif surface == methods.ADAPTER_BOTTLENECK:
            from .. import bottleneck  # noqa: PLC0415
            rank = next(v.shape[0] for k, v in self.model.state_dict().items()
                        if k.endswith(".adapter_down.weight"))
            bottleneck.save_torch(self.model, out)
            bottleneck.write_config(out, base_model=self.model_name, rank=int(rank))
        elif surface == methods.ADAPTER_MORE:
            from .. import more  # noqa: PLC0415
            meta = self._more_meta
            self.model.save_pretrained(str(out))  # the peft adapter
            more.save_torch_wrappers(self.model, out)
            self._more_index.save(out)
            more.write_config(out, base_model=self.model_name, rank=meta["rank"],
                              k=meta["k"], num_layers=meta["num_layers"])
        elif surface == methods.ADAPTER_MORE_PLUS:
            from .. import more_plus as mp  # noqa: PLC0415
            st = self._more_plus
            meta = st["meta"]
            mp.save_experts(st["deltas"], out)
            (out / mp._INDEX_FILE).write_text(_json.dumps(st["router"].to_dict()))
            mp.write_config(out, base_model=self.model_name, lora_r=meta["lora_r"],
                            lora_alpha=meta["lora_alpha"], final_layer_idx=meta["final_layer_idx"],
                            num_experts=st["router"].N, k=st["k"], tau=st["tau"],
                            group_size=meta["group_size"])
            self.tokenizer.save_pretrained(str(out))
        elif fmt == "merged" and hasattr(self.model, "merge_and_unload"):
            merged = self.model.merge_and_unload()
            merged.save_pretrained(str(out))
        else:
            self.model.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))
        return str(out)
