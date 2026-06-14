"""Hugging Face model cache helpers — prefetch weights and report progress.

The studio uses these to show a real download instead of a silent hang the
first time a model is loaded, and to badge models that are already on disk.
Everything is built on ``huggingface_hub`` (a transitive dependency of every
training backend), imported lazily so the pure-stdlib core stays import-clean
and these helpers simply report "unavailable" when no backend is installed.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

# Skip alternate-format weights the torch/mlx backends never load — onnx, gguf,
# coreml, tflite, raw pytorch .bin/.pt when safetensors exist. Keeps a "360M"
# download from pulling 5 GB of variants nobody trains on.
_IGNORE = ["*.gguf", "*.onnx", "onnx/*", "*.onnx_data", "*.tflite",
           "coreml/*", "*.mlmodel", "*.pt", "*.pth", "*.h5", "*.msgpack"]


def _wanted(path: str) -> bool:
    return not any(fnmatch.fnmatch(path, pat) for pat in _IGNORE)


def available() -> bool:
    """True when huggingface_hub is importable (i.e. a backend is installed)."""
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec("huggingface_hub") is not None


def set_token(token: str | None) -> None:
    """Make a token visible to every HF library (used for gated/private models).

    HF reads several env names depending on version; set the common ones.
    Passing a falsy token clears them.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if token:
            os.environ[var] = token
        else:
            os.environ.pop(var, None)


def has_token() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def _cache_folder(model_id: str) -> Path | None:
    """The local cache directory huggingface_hub uses for a model repo."""
    try:
        from huggingface_hub import file_download  # noqa: PLC0415
        from huggingface_hub.constants import HF_HUB_CACHE  # noqa: PLC0415
    except Exception:  # noqa: BLE001 — hub not installed / API moved
        return None
    name = file_download.repo_folder_name(repo_id=model_id, repo_type="model")
    return Path(HF_HUB_CACHE) / name


def is_cached(model_id: str) -> bool:
    """True when the model's config is already on disk (enough to load offline)."""
    if model_id.startswith(("/", "./", "../")) or os.path.isdir(model_id):
        return True  # a local path is "cached" by definition
    try:
        from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return False
    # returns the file path (str) when cached, else None or a "known-missing"
    # sentinel — only a str means it's on disk and loadable.
    return isinstance(try_to_load_from_cache(model_id, "config.json"), str)


def cached_bytes(model_id: str) -> int:
    """Bytes currently on disk for the model (sum of the cache blobs)."""
    folder = _cache_folder(model_id)
    if not folder or not folder.exists():
        return 0
    total = 0
    for p in folder.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def repo_bytes(model_id: str, token: str | None = None) -> int:
    """Total download size of the model repo, or 0 if it can't be determined."""
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return 0
    try:
        info = HfApi().model_info(model_id, files_metadata=True, token=token)
    except Exception:  # noqa: BLE001 — gated without token, offline, 404
        return 0
    return sum(getattr(s, "size", 0) or 0 for s in (info.siblings or [])
               if _wanted(s.rfilename))


def download(model_id: str, token: str | None = None) -> str:
    """Fetch the model snapshot into the local cache; returns the local path.

    Raises if huggingface_hub is unavailable or the download fails (e.g. a gated
    repo without a token) — callers surface the message.
    """
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    return snapshot_download(model_id, token=token, ignore_patterns=_IGNORE)
