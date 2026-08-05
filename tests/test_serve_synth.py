"""The studio's synthesis endpoint: run in the background, land in the store."""

import json
import tempfile
import time
from pathlib import Path

from shadowlm.serve import Server


class _Teacher:
    """A local teacher the server accepts in place of an API model."""

    name = "stub"

    def chat(self, messages, **_):
        prompt = messages[-1]["content"]
        if "0.0 to 1.0" in prompt:
            return "0.9"
        if '"scenario"' in prompt:
            return json.dumps([{"scenario": f"scenario {i}", "difficulty": "easy",
                                "angle": f"angle {i}"} for i in range(4)])
        if "training conversation" in prompt:
            style = prompt.split("USER STYLE: ")[1].split("\n")[0]
            return json.dumps({"messages": [
                {"role": "user", "content": f"question in the style of {style}"},
                {"role": "assistant", "content": "an answer"}]})
        raise AssertionError(prompt[:200])


def _server(tmp: str) -> Server:
    return Server(backend="mlx", accelerator="none", device="auto",
                  work_root=Path(tmp))


def _wait(server: Server, synth_id: str, *, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = server.synth_status(synth_id)
        if status.get("status") != "running":
            return status
        time.sleep(0.05)
    raise AssertionError("synthesis did not finish in time")


def test_synth_run_lands_a_dataset_in_the_store(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        server = _server(tmp)
        monkeypatch.setattr("shadowlm.models.load", lambda *a, **k: _Teacher())
        before = {d["dataset_id"] for d in server.datasets.list()}

        started = server.start_synth({
            "name": "synthetic triage", "task": "triage email", "n": 4,
            "method": "lora", "teacher": {"kind": "local", "model": "stub"}})
        status = _wait(server, started["synth_id"])

        assert status["status"] == "succeeded", status.get("error")
        assert status["kept"] == 4
        new = [d for d in server.datasets.list() if d["dataset_id"] not in before]
        assert len(new) == 1
        assert new[0]["name"] == "synthetic triage"
        assert new[0]["format"] == "chat"
        assert server.datasets.resolve(new[0]["dataset_id"]).rows[0]["messages"]


def test_the_post_returns_before_the_teacher_finishes_loading(monkeypatch):
    """Loading a local teacher can take minutes — it must happen on the
    background thread, not while the HTTP request waits."""
    def slow_load(*a, **k):
        time.sleep(0.6)
        return _Teacher()

    with tempfile.TemporaryDirectory() as tmp:
        server = _server(tmp)
        monkeypatch.setattr("shadowlm.models.load", slow_load)
        t0 = time.monotonic()
        started = server.start_synth({
            "task": "t", "n": 4, "teacher": {"kind": "local", "model": "stub"}})
        assert time.monotonic() - t0 < 0.3, "start_synth blocked on model load"
        assert _wait(server, started["synth_id"])["status"] == "succeeded"


def test_failures_are_reported_not_swallowed(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        server = _server(tmp)

        class Broken(_Teacher):
            def chat(self, messages, **_):
                raise RuntimeError("teacher is down")

        monkeypatch.setattr("shadowlm.models.load", lambda *a, **k: Broken())
        started = server.start_synth({
            "task": "t", "n": 2, "teacher": {"kind": "local", "model": "stub"}})
        status = _wait(server, started["synth_id"])
        assert status["status"] == "failed"
        assert "teacher is down" in status["error"]


def test_status_exposes_live_phase_counters(monkeypatch):
    """The UI polls this to move its bar; without phase/done/total it can only
    show kept, which stays 0 until a whole round lands."""
    with tempfile.TemporaryDirectory() as tmp:
        server = _server(tmp)
        monkeypatch.setattr("shadowlm.models.load", lambda *a, **k: _Teacher())
        started = server.start_synth({
            "task": "t", "n": 4, "teacher": {"kind": "local", "model": "stub"}})
        final = _wait(server, started["synth_id"])

        assert final["phase"] == "kept"
        assert final["done"] == final["kept"] == 4
        # the log stays a summary — one line per round, not one per job
        assert len(final["logs"]) <= 3


def test_listing_hides_the_log_buffer(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        server = _server(tmp)
        monkeypatch.setattr("shadowlm.models.load", lambda *a, **k: _Teacher())
        started = server.start_synth({
            "task": "t", "n": 4, "teacher": {"kind": "local", "model": "stub"}})
        _wait(server, started["synth_id"])

        listed = server.synth_status()["jobs"]
        assert len(listed) == 1 and "logs" not in listed[0]
        # the detail view keeps them, for the console panel
        assert server.synth_status(started["synth_id"])["logs"]
        assert server.synth_status("nope") == {}
