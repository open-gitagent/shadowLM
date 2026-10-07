"""Training starts on a clean card: cached playground models are dropped
first, so a big base isn't fighting an inference copy of itself for VRAM.
"""

from __future__ import annotations

import threading

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


def test_a_finished_run_lets_go_of_its_model(tmp_path, monkeypatch):
    """The trained model isn't kept alive in the runner between jobs, where
    Clean VRAM can't reach it."""
    import gc
    import weakref

    from shadowlm import backends

    alive: list[weakref.ref] = []

    class FakeResult:
        checkpoint = None
        final_loss = 0.5

    class FakeBackend:
        def load(self, *a, **k):
            self.weights = bytearray(1024)

        def finetune(self, *a, **k):
            return FakeResult()

    def fake_select(*a, **k):
        be = FakeBackend()
        alive.append(weakref.ref(be))
        return be

    monkeypatch.setattr(backends, "select_backend", fake_select)
    s = _server(tmp_path)
    job_id = s.submit({
        "base_model": "fake/model", "load_in_4bit": False, "max_seq_length": 64,
        "config": {"method": "lora", "max_steps": 1},
        "dataset": {"rows": [{"text": "hi"}], "format": "text"},
        "eval_dataset": None,
    })
    for _ in range(200):
        if s.jobs[job_id].status in ("succeeded", "failed"):
            break
        threading.Event().wait(0.02)
    assert s.jobs[job_id].status == "succeeded", s.jobs[job_id].error
    for _ in range(50):  # the runner drops it right after persisting
        gc.collect()
        if alive and alive[0]() is None:
            break
        threading.Event().wait(0.02)
    assert alive and alive[0]() is None
