"""Ctrl agent: the cockpit's conversation answered by Claude, through the server.

A fake Messages API stands in for Anthropic. What's held: the key stays on the
server, a tool call comes back as an action (never executed), the server adds
the failed run's log to the state, and without a key the studio is told to keep
its rules (503) rather than failing.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from shadowlm.serve import Auth, Server, _Job, make_handler


@pytest.fixture()
def anthropic():
    """A fake Messages API. `reply` sets the next response body (or an int status)."""
    seen: list[dict] = []
    state = {"reply": {"content": [{"type": "text", "text": "Hello."}]}}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "key": self.headers.get("x-api-key"),
                         "version": self.headers.get("anthropic-version"), "body": body})
            reply = state["reply"]
            code, data = (reply, {"type": "error", "error": {"message": "invalid x-api-key"}}) \
                if isinstance(reply, int) else (200, reply)
            out = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1", seen, state
    httpd.shutdown()


def _studio(tmp_path):
    server = Server(backend="auto", accelerator="auto", device="auto", work_root=tmp_path)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(server, Auth(user=None, password=None, api_key=None)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return server, httpd


def _call(port, method, path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture()
def studio(tmp_path, anthropic, monkeypatch):
    url, _, _ = anthropic
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SHADOWLM_CTRL_BASE_URL", url)
    monkeypatch.delenv("SHADOWLM_CTRL_MODEL", raising=False)
    server, httpd = _studio(tmp_path)
    yield server, httpd.server_address[1]
    httpd.shutdown()


def test_without_a_key_the_studio_keeps_its_rules(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    server, httpd = _studio(tmp_path)
    port = httpd.server_address[1]
    try:
        assert _call(port, "GET", "/v1/settings")[1]["ctrl"] is None
        pid = server.projects.create("faq", "knowledge")["project_id"]
        code, out = _call(port, "POST", f"/v1/projects/{pid}/ctrl", {"message": "hi"})
        assert code == 503 and "ANTHROPIC_API_KEY" in out["error"]
    finally:
        httpd.shutdown()


def test_a_tool_call_comes_back_as_an_action_and_the_key_stays_put(studio, anthropic):
    server, port = studio
    _, seen, state = anthropic
    assert _call(port, "GET", "/v1/settings")[1]["ctrl"] == {"model": "claude-sonnet-5-5"}
    assert "sk-ant-test" not in json.dumps(_call(port, "GET", "/v1/settings")[1])
    pid = server.projects.create("faq", "knowledge")["project_id"]
    state["reply"] = {"content": [
        {"type": "text", "text": "SDFT keeps more of its general skills."},
        {"type": "tool_use", "id": "t1", "name": "propose_finetune", "input": {"method": "sdft", "max_steps": 300}}]}
    code, out = _call(port, "POST", f"/v1/projects/{pid}/ctrl", {
        "message": "use sdft with 300 steps", "state": {"stations": {"data": "12 examples"}},
        "thread": [{"from": "copilot", "text": "It has 12 examples."}]})
    assert code == 200
    assert out == {"reply": "SDFT keeps more of its general skills.",
                   "action": {"kind": "finetune", "args": {"method": "sdft", "max_steps": 300}}}
    call = seen[-1]
    assert call["path"] == "/v1/messages" and call["key"] == "sk-ant-test" and call["version"]
    assert {t["name"] for t in call["body"]["tools"]} >= {"propose_finetune", "propose_write_examples"}
    prompt = call["body"]["messages"][0]["content"]
    assert "12 examples" in prompt and "use sdft with 300 steps" in prompt


def test_a_failed_run_brings_its_log(studio, anthropic):
    server, port = studio
    _, seen, _ = anthropic
    pid = server.projects.create("faq", "task")["project_id"]
    server.jobs["run1"] = _Job(job_id="run1", base_model="m", status="failed",
                               logs=["loading", "CUDA out of memory. Tried to allocate 2.00 GiB"])
    server.projects.update(pid, {"run_id": "run1"})
    code, out = _call(port, "POST", f"/v1/projects/{pid}/ctrl", {"message": "why did it fail?"})
    assert code == 200 and out["action"] is None and out["reply"] == "Hello."
    assert "CUDA out of memory" in seen[-1]["body"]["messages"][0]["content"]


def test_errors_say_what_happened(studio, anthropic):
    server, port = studio
    _, _, state = anthropic
    pid = server.projects.create("faq", "task")["project_id"]
    assert _call(port, "POST", "/v1/projects/nope/ctrl", {"message": "hi"})[0] == 404
    assert _call(port, "POST", f"/v1/projects/{pid}/ctrl", {"message": "  "})[0] == 422
    state["reply"] = 401
    code, out = _call(port, "POST", f"/v1/projects/{pid}/ctrl", {"message": "hi"})
    assert code == 502 and "401" in out["error"] and "invalid x-api-key" in out["error"]
