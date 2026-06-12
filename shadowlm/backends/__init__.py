"""Backend selection.

`select_backend("auto")` picks the backend for the current hardware:
CUDA → torch, else Apple Silicon → mlx, else torch on CPU. If no backend is
installed, `load()` says what to install. Force one with backend="mlx" / "torch"
(and device="cpu" to pin torch to the CPU).
"""

from __future__ import annotations

from .base import Backend, Callbacks, FinetuneResult

__all__ = ["Backend", "Callbacks", "FinetuneResult", "select_backend"]


def _mlx_available() -> bool:
    from .mlx import MLXBackend
    return MLXBackend.is_available()


def _torch_available() -> bool:
    from .torch import TorchBackend
    return TorchBackend.is_available()


def _no_backend() -> RuntimeError:
    return RuntimeError(
        "No training backend available. Install one:\n"
        "  • Apple Silicon:  pip install shadowlm[mlx]\n"
        "  • CUDA / CPU:     pip install shadowlm[torch]"
    )


def select_backend(name: str = "auto", *, accelerator: str = "auto",
                   device: str = "auto") -> Backend:
    name = (name or "auto").lower()

    if name == "auto":
        from .torch import TorchBackend
        if TorchBackend.has_cuda():
            name = "torch"
        elif _mlx_available():
            name = "mlx"
        elif _torch_available():
            name, device = "torch", "cpu"
        else:
            raise _no_backend()

    if name == "mlx":
        from .mlx import MLXBackend
        if not MLXBackend.is_available():
            raise RuntimeError("mlx backend needs Apple Silicon + shadowlm[mlx].")
        return MLXBackend(accelerator=accelerator)

    if name == "torch":
        # CPU is just this backend with device="cpu" (no separate "cpu" backend).
        from .torch import TorchBackend
        if not TorchBackend.is_available():
            raise RuntimeError(
                "torch backend needs shadowlm[torch] (torch, transformers, trl, peft, datasets)."
            )
        return TorchBackend(device=device, accelerator=accelerator)

    if name == "remote":
        # Train wherever SHADOWLM_API_URL points — `python -m shadowlm.serve`
        # on a GPU box, or ShadowLM Studio. Pure stdlib; always constructible.
        from .remote import RemoteBackend
        return RemoteBackend(device=device, accelerator=accelerator)

    raise ValueError(f"unknown backend {name!r} (expected auto|mlx|torch|remote)")
