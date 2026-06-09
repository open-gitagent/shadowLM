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

from .. import accel
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
                 output_dir: str, eval_dataset: Dataset | None = None) -> FinetuneResult:
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

        already_peft = getattr(self.model, "peft_config", None) is not None
        if config.method in ("lora", "qlora") and not already_peft:
            from peft import LoraConfig, get_peft_model  # noqa: PLC0415
            self.model = get_peft_model(self.model, LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.target_modules),
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

        args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps or -1,
            num_train_epochs=config.num_train_epochs or 1.0,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            optim=optim_name,
            gradient_checkpointing=shadow.grad_checkpoint,
            logging_steps=config.logging_steps,
            eval_strategy="steps" if has_eval else "no",
            eval_steps=config.eval_steps or max(1, (config.max_steps or 1) // 4),
            per_device_eval_batch_size=config.per_device_train_batch_size,
            seed=config.seed,
            max_seq_length=config.max_seq_length,
            disable_tqdm=True,  # shadowLM prints its own progress
            report_to=[],
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
            result = trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        final_loss = float(result.training_loss) if result else None
        callbacks.log(f"[torch:{self.device}] done · final loss {final_loss} · checkpoint {output_dir}")
        return FinetuneResult(checkpoint=output_dir, final_loss=final_loss)

    def generate(self, prompt, *, max_new_tokens, temperature, top_p, **kwargs) -> str:
        import torch  # noqa: PLC0415

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
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
