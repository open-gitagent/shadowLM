"""The studio's Business-mode surfaces on the server: fine-tune projects, task
evaluations and chat run in the background (no held request can outlast the
proxy in front of the studio).
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

from http.server import ThreadingHTTPServer

import pytest

from shadowlm.serve import Auth, Server, make_handler


class _Reply:
    def __init__(self, text: str) -> None:
        self.raw = self.content = text

    def __str__(self) -> str:
        return self.content


class _FakeModel:
    """Answers from a fixed table; anything else gets a shrug."""

    def __init__(self, table: dict[str, str]) -> None:
        self.table = table

    def chat(self, messages, **_):
        q = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        return _Reply(self.table.get(q, "I'm not sure."))


FACTS = {"What is Lyzr?": "Lyzr is an enterprise platform for building and running AI agents.",
         "Who founded Lyzr?": "Siva Surendira, Anirudh Narayan and Jithin George founded Lyzr in 2023."}


@pytest.fixture()
def studio(tmp_path):
    server = Server(backend="auto", accelerator="auto", device="auto", work_root=tmp_path)
    tuned, base = _FakeModel(FACTS), _FakeModel({})
    server._infer_model = lambda model, adapter, checkpoint=None: tuned if adapter else base
    httpd = ThreadingHTTPServer(("127.0.0.1", 0),
                                make_handler(server, Auth(user=None, password=None, api_key=None)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield server, httpd.server_address[1]
    httpd.shutdown()


def _call(port: int, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _until(fn, timeout: float = 5.0):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(0.02)
    raise AssertionError("timed out")


def _faq_dataset(port: int) -> str:
    rows = [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
            for q, a in FACTS.items()]
    code, body = _call(port, "POST", "/v1/datasets", {"name": "lyzr-faq", "rows": rows})
    assert code in (200, 201), body
    return body["dataset_id"]


def test_project_lifecycle(studio):
    _, port = studio
    code, s = _call(port, "POST", "/v1/projects", {"name": "Lyzr FAQ", "goal": "knowledge"})
    assert code == 201 and s["goal"] == "knowledge" and s["run_id"] is None
    code, s2 = _call(port, "PATCH", f"/v1/projects/{s['project_id']}",
                     {"dataset_id": "abc123", "run_id": "def456"})
    assert code == 200 and (s2["dataset_id"], s2["run_id"]) == ("abc123", "def456")
    assert _call(port, "GET", "/v1/projects")[1]["projects"][0]["project_id"] == s["project_id"]
    # a new fine-tune is a new version; the old one, with its evaluation, moves to history
    _call(port, "PATCH", f"/v1/projects/{s['project_id']}", {"eval_id": "ev1"})
    code, v2 = _call(port, "PATCH", f"/v1/projects/{s['project_id']}", {"run_id": "run2"})
    assert code == 200 and (v2["run_id"], v2["eval_id"]) == ("run2", None)
    assert [(h["run_id"], h["eval_id"]) for h in v2["history"]] == [("def456", "ev1")]
    # the examples being written for it are linked too
    code, s3 = _call(port, "PATCH", f"/v1/projects/{s['project_id']}", {"synth_id": "a1b2c3d4e5"})
    assert code == 200 and s3["synth_id"] == "a1b2c3d4e5"
    # new data under the current version: it remembers what it trained on
    code, s4 = _call(port, "PATCH", f"/v1/projects/{s['project_id']}", {"dataset_id": "moredata1"})
    assert s4["dataset_id"] == "moredata1" and s4["run_dataset_id"] != "moredata1"
    assert _call(port, "DELETE", f"/v1/projects/{s['project_id']}")[0] == 200
    assert _call(port, "GET", f"/v1/projects/{s['project_id']}")[0] == 404


@pytest.mark.parametrize("body", [
    {"name": "", "goal": "knowledge"},
    {"name": "x", "goal": "something-else"},
])
def test_project_create_rejects_bad_input(studio, body):
    _, port = studio
    assert _call(port, "POST", "/v1/projects", body)[0] == 422


def test_project_links_must_be_plain_ids(studio):
    _, port = studio
    s = _call(port, "POST", "/v1/projects", {"name": "x", "goal": "task"})[1]
    assert _call(port, "PATCH", f"/v1/projects/{s['project_id']}", {"run_id": "../etc"})[0] == 422


def test_evaluation_scores_finetune_against_base(studio):
    server, port = studio
    ds = _faq_dataset(port)
    code, ev = _call(port, "POST", "/v1/evals", {
        "dataset_id": ds, "metric": "contains", "targets": [
            {"label": "fine-tune", "model": "Qwen/Qwen3-8B", "adapter": "run1"},
            {"label": "base", "model": "Qwen/Qwen3-8B"}]})
    assert code == 202 and ev["status"] in ("pending", "running")
    done = _until(lambda: (r := _call(port, "GET", f"/v1/evals/{ev['eval_id']}")[1])
                  ["status"] in ("succeeded", "failed") and r)
    assert done["status"] == "succeeded", done["error"]
    tuned, base = done["results"]
    assert (tuned["score"], base["score"]) == (1.0, 0.0)
    assert len(tuned["examples"]) == 2
    listed = _call(port, "GET", "/v1/evals")[1]["evals"][0]
    assert "examples" not in listed["results"][0]  # lists stay light
    # persisted: a fresh server reads it back
    again = Server(backend="auto", accelerator="auto", device="auto", work_root=server.work_root)
    assert again.evals[ev["eval_id"]].results[0]["score"] == 1.0


@pytest.mark.parametrize("with_dataset, body", [
    (False, {"metric": "contains", "targets": [{"model": "m"}]}),   # unknown dataset
    (True, {"metric": "bleu", "targets": [{"model": "m"}]}),        # unknown metric
    (True, {"metric": "contains", "targets": []}),                  # nothing to score
    (True, {"metric": "contains", "targets": [{"label": "x"}]}),    # target without a model
    (True, {"metric": "contains", "targets": [{"model": "m"}], "sample": 0}),
])
def test_evaluation_rejects_bad_requests(studio, with_dataset, body):
    _, port = studio
    body = {**body, "dataset_id": _faq_dataset(port) if with_dataset else "nope"}
    assert _call(port, "POST", "/v1/evals", body)[0] == 422


def test_chat_answers_in_the_background(studio):
    _, port = studio
    code, t = _call(port, "POST", "/v1/tasks/chat", {
        "model": "Qwen/Qwen3-8B", "adapter": "run1",
        "messages": [{"role": "user", "content": "What is Lyzr?"}]})
    assert code == 202
    done = _until(lambda: (r := _call(port, "GET", f"/v1/tasks/{t['task_id']}")[1])
                  ["status"] != "running" and r)
    assert done["status"] == "succeeded" and "enterprise platform" in done["text"]
