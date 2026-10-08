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


def test_a_run_survives_a_restart_and_stale_ones_are_not_left_spinning(monkeypatch):
    """Records outlive the process, like training jobs. A run still marked
    running belongs to a thread that no longer exists."""
    with tempfile.TemporaryDirectory() as tmp:
        server = _server(tmp)
        monkeypatch.setattr("shadowlm.models.load", lambda *a, **k: _Teacher())
        started = server.start_synth({
            "name": "kept-across-restart", "task": "t", "n": 4,
            "teacher": {"kind": "local", "model": "stub"}})
        _wait(server, started["synth_id"])

        reopened = _server(tmp)   # same work_root, fresh process
        record = reopened.synth_status(started["synth_id"])
        assert record["status"] == "succeeded"
        assert record["name"] == "kept-across-restart"

        # a record left mid-flight is reported stopped, not running forever
        path = Path(tmp) / "synth" / f"{started['synth_id']}.json"
        stale = json.loads(path.read_text())
        stale["status"] = "running"
        path.write_text(json.dumps(stale))
        after = _server(tmp).synth_status(started["synth_id"])
        assert after["status"] == "stopped"
        assert "restarted" in after["error"]


def test_cancelling_an_unknown_run_says_so():
    with tempfile.TemporaryDirectory() as tmp:
        assert _server(tmp).cancel_synth("nope") is False


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


def test_include_seed_keeps_the_examples_it_was_written_from(monkeypatch):
    """The cockpit amplifies a project's examples and trains the next version on
    old and new together, so the saved dataset carries both."""
    with tempfile.TemporaryDirectory() as tmp:
        server = _server(tmp)
        monkeypatch.setattr("shadowlm.models.load", lambda *a, **k: _Teacher())
        seed = server.datasets.save("seed", [
            {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
            for i in range(3)])
        started = server.start_synth({
            "name": "more", "task": "answer questions", "n": 4, "method": "lora",
            "dataset_id": seed["dataset_id"], "include_seed": True,
            "teacher": {"kind": "local", "model": "stub"}})
        status = _wait(server, started["synth_id"])
        assert status["status"] == "succeeded", status.get("error")
        meta = server.datasets.meta(status["dataset_id"])
        assert meta["rows"] == 3 + status["kept"]


def test_a_project_shows_the_run_and_gets_its_examples(monkeypatch):
    """Examples written for a project link to it while they're written, and
    become its data when the run lands — so the cockpit can start from a
    description and pick up when the examples arrive."""
    with tempfile.TemporaryDirectory() as tmp:
        server = _server(tmp)
        monkeypatch.setattr("shadowlm.models.load", lambda *a, **k: _Teacher())
        project = server.projects.create("triage", "task")
        started = server.start_synth({
            "name": "triage examples", "task": "triage email", "n": 4,
            "project_id": project["project_id"], "teacher": {"kind": "local", "model": "stub"}})
        assert server.projects.get(project["project_id"])["synth_id"] == started["synth_id"]
        status = _wait(server, started["synth_id"])
        assert server.projects.get(project["project_id"])["dataset_id"] == status["dataset_id"]
        try:
            server.start_synth({"task": "t", "n": 1, "project_id": "nope",
                                "teacher": {"kind": "local", "model": "stub"}})
        except ValueError as e:
            assert "no project" in str(e)
        else:
            raise AssertionError("an unknown project was accepted")
