"""Training starts on a clean card: cached playground models are dropped
first, so a big base isn't fighting an inference copy of itself for VRAM.
"""

from __future__ import annotations

from shadowlm.serve import Server


def _server(tmp_path) -> Server:
    return Server(backend="auto", accelerator="auto", device="auto", work_root=tmp_path)


def test_drop_inference_models_empties_the_cache(tmp_path):
    s = _server(tmp_path)
    s._infer_cache[("Qwen/Qwen3-8B", None, None)] = object()
    s._infer_cache[("Qwen/Qwen3-8B", "/a", None)] = object()
    with s._model_lock:
        n, err = s._drop_inference_models()
    assert (n, err) == (2, None)
    assert s._infer_cache == {}


def test_clear_vram_reports_what_it_unloaded(tmp_path):
    s = _server(tmp_path)
    s._infer_cache[("m", None, None)] = object()
    out = s.clear_vram()
    assert out["unloaded"] == 1 and "error" not in out
    assert s._infer_cache == {}
