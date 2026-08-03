"""Teachers: coercion, the HTTP client, and its retry behaviour.

The OpenAI-compatible teacher is exercised against a real stdlib server on a
loopback port — no mocking of urllib, so the wire format is actually tested.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from shadowlm.synth import teacher as tch


class _Stub:
    """A tiny OpenAI-compatible endpoint. `fail_first` 429s that many times."""

    def __init__(self, *, reply="hello", fail_first=0, status=None):
        self.reply, self.remaining_failures, self.status = reply, fail_first, status
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"body": body, "auth": self.headers.get("Authorization")})
                if outer.status is not None:
                    self.send_error(outer.status, "nope")
                    return
                if outer.remaining_failures > 0:
                    outer.remaining_failures -= 1
                    self.send_error(429, "slow down")
                    return
                payload = json.dumps({"choices": [
                    {"message": {"role": "assistant", "content": outer.reply}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def close(self):
        self._server.shutdown()
        self._server.server_close()


def _teacher(stub, **kwargs):
    return tch.OpenAIChatTeacher("test-model", base_url=stub.base_url,
                                 api_key="sk-test", **kwargs)


def test_chat_speaks_the_openai_wire_format():
    stub = _Stub(reply="the answer")
    try:
        out = _teacher(stub).chat([{"role": "user", "content": "hi"}],
                                  temperature=0.3, max_new_tokens=64)
        assert out == "the answer"
        sent = stub.requests[0]
        assert sent["body"]["model"] == "test-model"
        assert sent["body"]["messages"] == [{"role": "user", "content": "hi"}]
        assert sent["body"]["temperature"] == 0.3
        assert sent["body"]["max_tokens"] == 64
        assert sent["auth"] == "Bearer sk-test"
    finally:
        stub.close()


def test_transient_failures_are_retried():
    stub = _Stub(reply="eventually", fail_first=1)
    try:
        assert _teacher(stub).chat([{"role": "user", "content": "hi"}]) == "eventually"
        assert len(stub.requests) == 2
    finally:
        stub.close()


def test_a_permanent_error_raises_with_the_servers_words():
    stub = _Stub(status=401)
    try:
        with pytest.raises(RuntimeError, match="HTTP 401"):
            _teacher(stub).chat([{"role": "user", "content": "hi"}])
        assert len(stub.requests) == 1  # 401 is not retried
    finally:
        stub.close()


def test_an_unreachable_endpoint_says_where_it_tried():
    teacher = tch.OpenAIChatTeacher("m", base_url="http://127.0.0.1:1", api_key="k")
    with pytest.raises(RuntimeError, match="unreachable at http://127.0.0.1:1"):
        teacher.chat([{"role": "user", "content": "hi"}])


def test_a_loaded_model_becomes_a_serialized_teacher():
    class FakeModel:
        name = "qwen-tiny"

        def chat(self, messages, **kwargs):
            return f"reply to {messages[-1]['content']}"

    teacher = tch.as_teacher(FakeModel())
    assert teacher.name == "qwen-tiny"
    assert teacher.parallelism == 1  # backends are not thread-safe
    assert teacher.chat([{"role": "user", "content": "x"}]) == "reply to x"


def test_an_existing_teacher_passes_through_and_junk_is_rejected():
    original = tch.frontier("gpt-4o", api_key="k")
    assert tch.as_teacher(original) is original
    with pytest.raises(ValueError, match="needs teacher"):
        tch.as_teacher(None)
    with pytest.raises(TypeError, match="not a teacher"):
        tch.as_teacher(42)


def test_counting_teacher_tallies_calls():
    stub = _Stub()
    try:
        counted = tch.CountingTeacher(_teacher(stub))
        counted.chat([{"role": "user", "content": "a"}])
        counted.chat([{"role": "user", "content": "b"}])
        assert counted.calls == 2
        assert counted.name == "test-model"
    finally:
        stub.close()
