"""The studio's Phase 2–3 surfaces: the user's frontier model (settings, eval
baseline, judge), capturing an agent's traffic through it into a dataset,
importing traces, and serving a fine-tune at an OpenAI-compatible endpoint
under its own key.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from shadowlm.serve import Auth, Server, _Job, make_handler

ANSWERS = {"What is Lyzr?": "Lyzr is an enterprise platform for building and running AI agents."}


@pytest.fixture()
def frontier():
    """A fake OpenAI-compatible provider: answers from ANSWERS, scores 1.0 as a judge."""
    seen: list[dict] = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"auth": self.headers.get("Authorization"), "body": body})
            last = body["messages"][-1]["content"]
            text = "1.0" if last.startswith("Score how well") else ANSWERS.get(last, "No idea.")
            data = json.dumps({"id": "x", "object": "chat.completion", "created": 1, "model": body["model"],
                               "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                            "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1", seen
    httpd.shutdown()


class _Reply:
    def __init__(self, text):
        self.raw = self.content = text
        self.tool_calls = []

    def to_message(self):
        return {"role": "assistant", "content": self.content}

    def __str__(self):
        return self.content


class _Tuned:
    def chat(self, messages, **_):
        return _Reply(ANSWERS.get(messages[-1]["content"], "Hmm."))


@pytest.fixture()
def studio(tmp_path):
    server = Server(backend="auto", accelerator="auto", device="auto", work_root=tmp_path)
    server._infer_model = lambda model, adapter, checkpoint=None: _Tuned()
    server.prewarm = lambda *a, **k: {"ready": True}
    httpd = ThreadingHTTPServer(("127.0.0.1", 0),
                                make_handler(server, Auth(user=None, password=None, api_key=None)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield server, httpd.server_address[1]
    httpd.shutdown()


def _call(port, method, path, body=None, headers=None, raw=False):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req) as r:
            data = r.read()
            return r.status, (data.decode() if raw else json.loads(data))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _set_frontier(port, url):
    return _call(port, "POST", "/v1/settings",
                 {"frontier": {"base_url": url, "api_key": "sk-secret", "model": "gpt-test"}})


def test_frontier_settings_never_return_the_key(studio, frontier):
    server, port = studio
    url, _ = frontier
    _call(port, "POST", "/v1/settings", {"hf_token": ""})
    code, out = _set_frontier(port, url)
    assert code == 200 and out["frontier"] == {"base_url": url, "model": "gpt-test"}
    assert "sk-secret" not in json.dumps(_call(port, "GET", "/v1/settings")[1])
    assert server._settings.get("hf_token") == ""  # setting one doesn't wipe the other
    assert _call(port, "POST", "/v1/settings", {"frontier": {"base_url": url}})[0] == 422  # no key/model
    assert _call(port, "POST", "/v1/settings", {"frontier": None})[1]["frontier"] is None


def test_capture_passes_through_and_becomes_a_dataset(studio, frontier):
    _, port = studio
    url, seen = frontier
    cap = _call(port, "POST", "/v1/captures", {"name": "support agent"})[1]
    path = f"/v1/capture/{cap['capture_id']}/chat/completions"
    assert _call(port, "POST", path, {"messages": [{"role": "user", "content": "hi"}]})[0] == 503  # no frontier yet
    _set_frontier(port, url)
    turn1 = [{"role": "user", "content": "What is Lyzr?"}]
    code, out = _call(port, "POST", path, {"model": "anything", "messages": turn1})
    assert code == 200 and out["choices"][0]["message"]["content"].startswith("Lyzr is")
    assert seen[-1]["auth"] == "Bearer sk-secret" and seen[-1]["body"]["model"] == "gpt-test"
    turn2 = turn1 + [out["choices"][0]["message"], {"role": "user", "content": "And?"}]
    code, sse = _call(port, "POST", path, {"messages": turn2, "stream": True}, raw=True)
    assert code == 200 and sse.rstrip().endswith("data: [DONE]")
    info = _call(port, "GET", f"/v1/captures/{cap['capture_id']}")[1]
    assert (info["calls"], info["episodes"]) == (2, 1)  # the second call extends the first
    code, ds = _call(port, "POST", f"/v1/captures/{cap['capture_id']}/dataset", {"name": "captured"})
    assert code == 201 and ds["rows"] == 1
    assert _call(port, "POST", f"/v1/captures/{cap['capture_id']}/close")[0] == 200
    assert _call(port, "POST", path, {"messages": turn1})[0] == 403


def test_unknown_capture_is_not_an_open_proxy(studio, frontier):
    _, port = studio
    _set_frontier(port, frontier[0])
    assert _call(port, "POST", "/v1/capture/deadbeef/chat/completions",
                 {"messages": [{"role": "user", "content": "x"}]})[0] == 404


def test_traces_import_into_a_dataset(studio):
    _, port = studio
    span = {"trace_id": "e1", "start_time": 1.0, "name": "chat", "attributes": {}, "events": [
        {"name": "gen_ai.user.message", "attributes": {"content": "hi"}},
        {"name": "gen_ai.choice", "attributes": {"role": "assistant", "content": "hello"}}]}
    code, ds = _call(port, "POST", "/v1/datasets/traces", {"name": "otel", "traces": [span]})
    assert code == 201 and ds["rows"] == 1
    assert _call(port, "POST", "/v1/datasets/traces", {"name": "x"})[0] == 422


def _wait_eval(port, eid):
    for _ in range(250):
        ev = _call(port, "GET", f"/v1/evals/{eid}")[1]
        if ev["status"] in ("succeeded", "failed"):
            return ev
        time.sleep(0.02)
    raise AssertionError("evaluation never finished")


def test_frontier_baseline_and_judge_scoring(studio, frontier):
    _, port = studio
    rows = [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
            for q, a in ANSWERS.items()]
    ds = _call(port, "POST", "/v1/datasets", {"name": "faq", "rows": rows})[1]["dataset_id"]
    body = {"dataset_id": ds, "metric": "judge",
            "targets": [{"label": "Fine-tune", "model": "m", "adapter": "r1"}, {"kind": "frontier"}]}
    assert _call(port, "POST", "/v1/evals", body)[0] == 422  # judge needs a frontier model
    _set_frontier(port, frontier[0])
    code, ev = _call(port, "POST", "/v1/evals", body)
    assert code == 202 and ev["targets"][1]["model"] == "gpt-test"
    done = _wait_eval(port, ev["eval_id"])
    assert done["status"] == "succeeded", done["error"]
    assert [r["score"] for r in done["results"]] == [1.0, 1.0]  # the fake judge says 1.0


def test_deployment_serves_a_finetune_on_its_own_key(studio):
    server, port = studio
    server.jobs["run1"] = _Job(job_id="run1", base_model="Qwen/Qwen2.5-1.5B-Instruct", status="succeeded")
    server.jobs["run2"] = _Job(job_id="run2", base_model="m", status="failed")
    assert _call(port, "POST", "/v1/deployments", {"name": "lyzr-faq", "run_id": "run2"})[0] == 422
    assert _call(port, "POST", "/v1/deployments", {"name": "bad name!", "run_id": "run1"})[0] == 422
    code, out = _call(port, "POST", "/v1/deployments", {"name": "lyzr-faq", "run_id": "run1"})
    assert code == 201 and out["key"].startswith("slmd_")
    key, dep = out["key"], out["deployment"]
    assert "key_hash" not in dep and _call(port, "POST", "/v1/deployments",
                                            {"name": "lyzr-faq", "run_id": "run1"})[0] == 422  # name taken
    auth = {"Authorization": f"Bearer {key}"}
    assert _call(port, "GET", "/openai/v1/models", headers=auth)[1]["data"][0]["id"] == "lyzr-faq"
    code, reply = _call(port, "POST", "/openai/v1/chat/completions", {
        "model": "lyzr-faq", "messages": [{"role": "user", "content": "What is Lyzr?"}]}, headers=auth)
    assert code == 200 and reply["choices"][0]["message"]["content"].startswith("Lyzr is")
    assert _call(port, "POST", "/openai/v1/chat/completions", {
        "model": "other", "messages": []}, headers=auth)[0] == 404
    assert _call(port, "POST", "/openai/v1/chat/completions", {
        "messages": []}, headers={"Authorization": "Bearer slmd_wrong"})[0] == 401
    listed = _call(port, "GET", "/v1/deployments")[1]["deployments"][0]
    assert listed["requests"] == 1 and "key_hash" not in listed
    assert _call(port, "DELETE", f"/v1/deployments/{dep['deployment_id']}")[0] == 200
    assert _call(port, "GET", "/openai/v1/models", headers=auth)[0] == 401  # revoked


def test_synth_can_use_the_saved_frontier_model_as_teacher(studio, frontier):
    """kind "frontier" teaches with the model and key saved in settings; with
    none saved, the run fails saying so instead of calling anything."""
    server, port = studio
    started = server.start_synth({"name": "t", "n": 1, "task": "say hi", "teacher": {"kind": "frontier"}})
    sid = started["synth_id"] if isinstance(started, dict) else started
    for _ in range(250):
        st = server._synth[sid]
        if st["status"] != "running":
            break
        time.sleep(0.02)
    assert st["status"] == "failed" and "frontier model" in (st.get("error") or "")
