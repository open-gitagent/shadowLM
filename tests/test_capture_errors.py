"""The capture proxy answers with an OpenAI-shaped error body instead of
dropping the connection.

Agents under capture are real OpenAI clients: they retry a 4xx/5xx JSON error
and surface a useful message, but a killed connection reads as a transport
fault and usually takes the agent down with it.
"""

from __future__ import annotations

import http.client
import json

import pytest

from shadowlm.capture import CaptureProxy


class _StubModel:
    """Stands in for a loaded Model — chat() either answers or explodes."""

    name = "stub/model"

    def __init__(self, boom: Exception | None = None) -> None:
        self._boom = boom

    def chat(self, messages, **kw):
        if self._boom:
            raise self._boom

        class _Reply:
            content = "hi"
            raw = "hi"
            tool_calls = None

            def to_message(self):
                return {"role": "assistant", "content": "hi"}

        return _Reply()


@pytest.fixture()
def proxy_for():
    live = []

    def _make(model):
        p = CaptureProxy(model, port=0).start()
        live.append(p)
        return p

    yield _make
    for p in live:
        p.stop()


def _post(proxy, body: bytes | str, *, path="/v1/chat/completions",
          content_length=None):
    host, port = proxy.host, proxy.port
    conn = http.client.HTTPConnection(host, port, timeout=10)
    try:
        raw = body if isinstance(body, bytes) else body.encode()
        headers = {"Content-Type": "application/json"}
        if content_length is not None:
            headers["Content-Length"] = str(content_length)
        conn.request("POST", path, body=raw, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def test_happy_path_still_answers(proxy_for):
    proxy = proxy_for(_StubModel())
    status, payload = _post(proxy, json.dumps({"messages": [
        {"role": "user", "content": "hi"}]}))
    assert status == 200
    assert json.loads(payload)["choices"][0]["message"]["content"] == "hi"


def test_model_failure_becomes_a_json_error_not_a_dropped_socket(proxy_for):
    proxy = proxy_for(_StubModel(boom=RuntimeError("CUDA out of memory")))
    status, payload = _post(proxy, json.dumps({"messages": [
        {"role": "user", "content": "hi"}]}))
    assert status == 500
    body = json.loads(payload)
    assert "CUDA out of memory" in body["error"]["message"]
    assert body["error"]["type"]


def test_malformed_json_is_a_400(proxy_for):
    proxy = proxy_for(_StubModel())
    status, payload = _post(proxy, "{not json")
    assert status == 400
    assert json.loads(payload)["error"]["message"]


def test_missing_content_length_is_a_400(proxy_for):
    proxy = proxy_for(_StubModel())
    status, _ = _post(proxy, b"", content_length=None)
    assert status == 400


def test_a_failed_call_is_not_recorded_as_a_trajectory(proxy_for):
    proxy = proxy_for(_StubModel(boom=RuntimeError("boom")))
    _post(proxy, json.dumps({"messages": [{"role": "user", "content": "hi"}]}))
    assert proxy.trajectories() == []


def test_the_proxy_keeps_serving_after_a_failure(proxy_for):
    model = _StubModel(boom=RuntimeError("boom"))
    proxy = proxy_for(model)
    _post(proxy, json.dumps({"messages": [{"role": "user", "content": "hi"}]}))
    model._boom = None  # the transient failure clears
    status, payload = _post(proxy, json.dumps({"messages": [
        {"role": "user", "content": "hi"}]}))
    assert status == 200
    assert json.loads(payload)["choices"][0]["message"]["content"] == "hi"
