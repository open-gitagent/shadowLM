"""Missing request fields are 422s naming the field, not 500 KeyErrors.

The whole do_POST route table sits inside one `except Exception` that renders
as a 500 — so a client typo used to come back as `500 KeyError: 'model'`, which
reads like a server bug and tells the caller nothing.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from shadowlm.serve import Auth, Server, make_handler


@pytest.fixture()
def hub(tmp_path):
    server = Server(backend="auto", accelerator="auto", device="auto",
                    work_root=tmp_path)
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(server, Auth(user="admin", password=None, api_key=None)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()


def _post(port: int, path: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


@pytest.mark.parametrize("path,body,missing", [
    ("/v1/generate", {}, "prompt"),
    ("/v1/generate", {"prompt": "hi"}, "model"),
    ("/v1/chat", {}, "messages"),
    ("/v1/chat", {"messages": [{"role": "user", "content": "hi"}]}, "model"),
    ("/v1/prewarm", {}, "model"),
])
def test_missing_field_is_a_422_naming_it(hub, path, body, missing):
    status, payload = _post(hub, path, body)
    assert status == 422, payload
    assert missing in payload.get("error", "")


def test_chat_messages_must_be_a_list(hub):
    status, payload = _post(hub, "/v1/chat", {"model": "m", "messages": "hi"})
    assert status == 422
    assert "messages" in payload.get("error", "")
