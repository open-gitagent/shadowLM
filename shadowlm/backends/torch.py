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
            raise NotImplementedError(
                "GRPO on the torch backend (trl GRPOTrainer) isn't wired yet — "
                "available today on the mlx backend (Apple Silicon)."
            )

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

        train_ds = HFDataset.from_dict({"text": dataset.to_texts()})
        has_eval = eval_dataset is not None and len(eval_dataset) > 0
        eval_ds = HFDataset.from_dict({"text": eval_dataset.to_texts()}) if has_eval else None
        optim_name = "adamw_torch_fused" if shadow.fused_optimizer else config.optim

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
            max_seq_length=config.max_seq_length,
            disable_tqdm=True,  # shadowLM prints its own progress
            report_to=list(config.report_to),
            **extra,
        )
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=args,
            callbacks=[_Bridge()],
        )
        with quiet_backend():
            result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        final_loss = float(result.training_loss) if result else None
        callbacks.log(f"[torch:{self.device}] done · final loss {final_loss} · checkpoint {output_dir}")
        return FinetuneResult(checkpoint=output_dir, final_loss=final_loss)

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
            disable_tqdm=True,
            report_to=list(config.report_to),
        )
        trainer = DPOTrainer(
            model=self.model,
            args=args,
            train_dataset=rows(dataset),
            eval_dataset=rows(eval_dataset) if has_eval else None,
            processing_class=self.tokenizer,
            peft_config=peft_config,
            callbacks=[_Bridge()],
        )
        with quiet_backend():
            result = trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.model = trainer.model
        final_loss = float(result.training_loss) if result else None
        callbacks.log(f"[torch:{self.device}] dpo done · final loss {final_loss} · {output_dir}")
        return FinetuneResult(checkpoint=output_dir, final_loss=final_loss)

    def generate(self, prompt, *, max_new_tokens, temperature, top_p, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}],
                         max_new_tokens=max_new_tokens, temperature=temperature,
                         top_p=top_p, **kwargs)

    def chat(self, messages, *, tools=None, max_new_tokens, temperature, top_p, **kwargs) -> str:
        import torch  # noqa: PLC0415

        inputs = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

    def save(self, path: str, *, fmt: str = "adapter") -> str:
        Path(path).mkdir(parents=True, exist_ok=True)
        if fmt == "merged" and hasattr(self.model, "merge_and_unload"):
            merged = self.model.merge_and_unload()
            merged.save_pretrained(path)
        else:
            self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        return path
