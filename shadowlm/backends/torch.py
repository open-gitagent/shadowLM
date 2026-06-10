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
from pathlib import Path

from .. import accel, methods
from .._quiet import quiet_backend
from ..data import Dataset
from ..training import Metric, TrainConfig, resolve_total_steps
from .base import Backend, Callbacks, FinetuneResult

_REQUIRED = ("torch", "transformers", "trl", "peft", "datasets")


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


class TorchBackend(Backend):
    name = "torch"

    def __init__(self, *, device: str = "auto", accelerator: str = "auto") -> None:
        super().__init__(device=device, accelerator=accelerator)
        if device == "auto":
            self.device = "cuda" if self.has_cuda() else "cpu"

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
        except Exception:
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if adapter:
            from peft import PeftModel  # noqa: PLC0415

            from .. import more  # noqa: PLC0415
            more_cfg = more.read_config(adapter)
            if more_cfg:
                # Rebuild in the training order: wrap attention, peft on top,
                # then the trained wrapper weights and the fact index.
                self._more_index = more.MemoryIndex.load(adapter)
                more.attach_torch(self.model, self._more_index,
                                  rank=more_cfg["rank"], k=more_cfg["index_k"],
                                  num_layers=more_cfg["num_layers"])
                self.model = PeftModel.from_pretrained(self.model, adapter)
                more.load_torch_wrappers(self.model, adapter)
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
            return self._finetune_more(dataset, config, callbacks, output_dir)

        # Adapter methods (lora/dora/cpt/...) attach PEFT adapters once; a spec
        # with adapter="none" trains the full weights. raw_text methods already
        # train on plain text because to_texts applies no chat template.
        already_peft = getattr(self.model, "peft_config", None) is not None
        if spec.trains_adapters and not already_peft:
            from peft import LoraConfig, get_peft_model  # noqa: PLC0415
            self.model = get_peft_model(self.model, LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.resolved_target_modules()),
                use_dora=(spec.adapter == methods.ADAPTER_DORA),
                use_rslora=config.use_rslora,
                bias="none",
                task_type="CAUSAL_LM",
            ))

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
            max_steps=config.max_steps or -1,
            num_train_epochs=config.num_train_epochs or 1.0,
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
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        final_loss = float(result.training_loss) if result else None
        callbacks.log(f"[torch:{self.device}] done · final loss {final_loss} · checkpoint {output_dir}")
        return FinetuneResult(checkpoint=output_dir, final_loss=final_loss)

    def _finetune_grpo(self, dataset: Dataset, config: TrainConfig, callbacks: Callbacks,
                       output_dir: str, spec, reward_fns: list | None) -> FinetuneResult:
        """RL from programmable rewards via trl's GRPOTrainer."""
        from datasets import Dataset as HFDataset  # noqa: PLC0415
        from transformers import TrainerCallback  # noqa: PLC0415
        from trl import GRPOConfig, GRPOTrainer  # noqa: PLC0415

        if not reward_fns:
            raise ValueError(
                "method='grpo' needs reward_fns=[...] — each is "
                "fn(prompts, completions, answer, types=None) -> list[float]"
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
            max_steps=config.max_steps or -1,
            num_train_epochs=config.num_train_epochs or 1.0,
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_scheduler_type,
            logging_steps=config.logging_steps,
            seed=config.seed,
            use_cpu=(self.device == "cpu"),
            disable_tqdm=True,
            report_to=list(config.report_to),
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
                       output_dir: str) -> FinetuneResult:
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
        index = more.MemoryIndex.build(inputs, outputs)
        self._more_index = index
        callbacks.log(f"[more] indexed {len(index)} retrieval experts")

        # 2. wrap attention, then LoRA on top (peft matches q_proj etc. by
        #    suffix, so it still finds them inside the wrapper)
        n_layers = min(config.retrieval_layers,
                       len((self.model.model if hasattr(self.model, "model")
                            else self.model).layers))
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
        callbacks.log(f"[more] memory attention + lora on {wrapped} layers "
                      f"(k={config.retrieval_k}, r={config.lora_r})")

        # 3. standard supervised training over the facts (chat-templated, so
        #    training and inference see the same rendering)
        train_ds = self._train_dataset(dataset)

        class _Bridge(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                if logs and "loss" in logs:
                    callbacks.step(Metric(step=state.global_step, loss=float(logs["loss"]),
                                          lr=float(logs.get("learning_rate", 0.0))))

        args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.resolved_warmup(config.max_steps or 1),
            max_steps=config.max_steps or -1,
            num_train_epochs=config.num_train_epochs or 1.0,
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_scheduler_type,
            optim=self._resolved_optim(config, accel.plan("none", backend="torch")),
            logging_steps=config.logging_steps,
            seed=config.seed,
            max_length=config.max_seq_length,
            use_cpu=(self.device == "cpu"),
            disable_tqdm=True,
            report_to=[],
        )
        callbacks.log(
            f"[torch:{self.device}] more on {self.model_name} · {len(dataset)} facts · "
            f"retrieval k={config.retrieval_k} on {n_layers} layers"
        )
        with quiet_backend():
            trainer = SFTTrainer(model=self.model, processing_class=self.tokenizer,
                                 train_dataset=train_ds, args=args,
                                 callbacks=[_Bridge()])
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
            max_steps=config.max_steps or -1,
            num_train_epochs=config.num_train_epochs or 1.0,
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
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **sampling,
            )
        prompt_len = enc["input_ids"].shape[1]
        return self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

    def save(self, path: str, *, fmt: str = "adapter") -> str:
        Path(path).mkdir(parents=True, exist_ok=True)
        if fmt == "merged" and hasattr(self.model, "merge_and_unload"):
            merged = self.model.merge_and_unload()
            merged.save_pretrained(path)
        else:
            self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        return path
