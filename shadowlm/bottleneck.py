"""Bottleneck adapters — small residual modules after each transformer layer.

Each wrapped layer computes `out + up(gelu(down(out)))` where down/up are a
narrow bottleneck (width = lora_r) and `up` is zero-initialized so training
starts as a no-op. Only the bottlenecks train; everything else stays frozen.
"""

from __future__ import annotations

import json
from pathlib import Path

_CONFIG_FILE = "bottleneck_config.json"
_TORCH_WEIGHTS = "bottleneck.safetensors"
PARAM_NAMES = ("adapter_down", "adapter_up")


# ---- mlx --------------------------------------------------------------------
class _MLXBottleneck:  # populated lazily so the module imports without mlx
    cls = None


def _mlx_bottleneck_cls():
    if _MLXBottleneck.cls is None:
        import mlx.core as mx  # noqa: PLC0415
        import mlx.nn as nn  # noqa: PLC0415

        class Bottleneck(nn.Module):
            """One shared class for every layer — mlx grad checkpointing keys
            on the layer type, so per-layer minted classes would checkpoint
            only the first layer."""

            def __init__(self, base, size: int, rank: int) -> None:
                super().__init__()
                self.base = base
                self.adapter_down = nn.Linear(size, rank, bias=False)
                self.adapter_up = nn.Linear(rank, size, bias=False)
                self.adapter_up.weight = mx.zeros_like(self.adapter_up.weight)

            def __call__(self, x, *args, **kwargs):
                import mlx.nn as nn  # noqa: PLC0415
                out = self.base(x, *args, **kwargs)
                delta = self.adapter_up(nn.gelu(self.adapter_down(out)))
                return out + delta.astype(out.dtype)

        _MLXBottleneck.cls = Bottleneck
    return _MLXBottleneck.cls


def attach_mlx(model, *, rank: int) -> int:
    hidden = model.args.hidden_size if hasattr(model, "args") else None
    cls = _mlx_bottleneck_cls()
    wrapped = 0
    for i, layer in enumerate(model.layers):
        if hasattr(layer, "adapter_down"):
            continue
        size = hidden or layer.self_attn.q_proj.weight.shape[1]
        model.layers[i] = cls(layer, size, rank)
        wrapped += 1
    return wrapped


# ---- torch ------------------------------------------------------------------
def attach_torch(model, *, rank: int) -> int:
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    decoder = model.model if hasattr(model, "model") else model
    hidden = model.config.hidden_size

    def make(base):
        class Bottleneck(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = base
                self.adapter_down = nn.Linear(hidden, rank, bias=False)
                self.adapter_up = nn.Linear(rank, hidden, bias=False)
                nn.init.zeros_(self.adapter_up.weight)

            def forward(self, hidden_states, *args, **kwargs):
                out = self.base(hidden_states, *args, **kwargs)
                h = out[0] if isinstance(out, tuple) else out
                delta = self.adapter_up(
                    torch.nn.functional.gelu(self.adapter_down(h.float()))
                ).to(h.dtype)
                if isinstance(out, tuple):
                    return (h + delta,) + out[1:]
                return h + delta

        return Bottleneck()

    wrapped = 0
    for i, layer in enumerate(decoder.layers):
        if hasattr(layer, "adapter_down"):
            continue
        wrapper = make(layer)
        device = next(layer.parameters()).device
        wrapper.adapter_down.to(device)
        wrapper.adapter_up.to(device)
        decoder.layers[i] = wrapper
        wrapped += 1
    return wrapped


def save_torch(model, directory: str | Path) -> None:
    from safetensors.torch import save_file  # noqa: PLC0415

    state = {name: tensor.contiguous() for name, tensor in model.state_dict().items()
             if any(f".{p}." in name for p in PARAM_NAMES)}
    save_file(state, str(Path(directory) / _TORCH_WEIGHTS))


def load_torch(model, directory: str | Path) -> None:
    from safetensors.torch import load_file  # noqa: PLC0415

    model.load_state_dict(load_file(str(Path(directory) / _TORCH_WEIGHTS)),
                          strict=False)


# ---- shared config -----------------------------------------------------------
def write_config(directory: str | Path, *, base_model: str, rank: int) -> None:
    (Path(directory) / _CONFIG_FILE).write_text(json.dumps(
        {"type": "bottleneck", "base_model": base_model, "rank": rank}, indent=2))


def read_config(directory: str | Path) -> dict | None:
    p = Path(directory) / _CONFIG_FILE
    return json.loads(p.read_text()) if p.exists() else None
