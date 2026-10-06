"""A host console (opencontroller) may frame the studio: only the listed
origins, which every response names in frame-ancestors and the studio page
carries in a meta tag for the oc-embed/1 bridge.
"""

from __future__ import annotations

import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from shadowlm import serve
from shadowlm.serve import Auth, Server, _parse_origins, frame_ancestors, make_handler


def test_parse_origins_keeps_plain_origins_only():
    raw = ("https://console.example.com, http://Host:8080/ "
           "https://x.test/path ftp://y.test https://u:p@z.test "
           "https://q.test?a=1 javascript:alert(1) https://console.example.com")
    assert _parse_origins(raw) == ["https://console.example.com", "http://host:8080"]


def test_frame_ancestors_builtins_then_env(monkeypatch):
    monkeypatch.setenv("SHADOWLM_FRAME_ANCESTORS", "https://oc.example.com http://localhost:*")
    got = frame_ancestors()
    assert got[:3] == ["https://dev.opencontroller.sh", "http://localhost:*", "https://localhost:*"]
    assert got[3:] == ["https://oc.example.com"]  # no duplicate of a built-in


@pytest.fixture()
def studio(tmp_path, monkeypatch):
    monkeypatch.setenv("SHADOWLM_FRAME_ANCESTORS", "https://oc.example.com")
    static = tmp_path / "pkg" / "_static"
    static.mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><html><head><title>t</title></head><body></body></html>")
    # Serve the fake build: the handler resolves _static next to serve.py.
    monkeypatch.setattr(serve, "__file__", str(tmp_path / "pkg" / "serve.py"))
    server = Server(backend="auto", accelerator="auto", device="auto", work_root=tmp_path / "work")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0),
                                make_handler(server, Auth(user=None, password=None, api_key=None)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()


def _get(port: int, path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as r:
        return r.headers, r.read().decode()


def test_every_response_limits_framing(studio):
    for path in ("/", "/v1/auth"):
        headers, _ = _get(studio, path)
        csp = headers["Content-Security-Policy"]
        assert csp == ("frame-ancestors 'self' https://dev.opencontroller.sh "
                       "http://localhost:* https://localhost:* https://oc.example.com")


def test_studio_page_names_its_embed_parents(studio):
    _, body = _get(studio, "/")
    assert ('<meta name="shadowlm-embed-parents" content="https://dev.opencontroller.sh '
            'http://localhost:* https://localhost:* https://oc.example.com"></head>') in body

