"""The wire has limits: WS frames and HTTP bodies are capped, and a worker
never lets a hub-supplied job_id name a path outside its work root.

Both peers of the hub↔worker socket, and any HTTP client, could otherwise
claim a 2**63-byte payload and make the other side allocate it — or a rogue
hub could hand a worker a job_id of `../../..` and read/write outside the
work dir.
"""

from __future__ import annotations

import http.client
import socket
import struct
import threading
from http.server import ThreadingHTTPServer

import pytest

from shadowlm import ws
from shadowlm.serve import Auth, Server, make_handler
from shadowlm.worker import _job_dir


# ---- websocket frame cap -----------------------------------------------------

def test_recv_frame_rejects_oversized_payload():
    a, b = socket.socketpair()
    try:
        # header claiming a 1 TiB unmasked text frame; no payload follows
        a.sendall(bytes([0x81, 127]) + struct.pack(">Q", 1 << 40))
        with pytest.raises(ConnectionError):
            ws.recv_frame(b)
    finally:
        a.close()
        b.close()


def test_frame_under_cap_roundtrips():
    a, b = socket.socketpair()
    try:
        ws.send_frame(a, b"hello", mask=True)
        opcode, payload = ws.recv_frame(b)
        assert (opcode, payload) == (ws.OP_TEXT, b"hello")
    finally:
        a.close()
        b.close()


# ---- worker job_id containment -------------------------------------------------

def test_job_dir_accepts_server_minted_ids(tmp_path):
    assert _job_dir(tmp_path, "a3f9c02b11ee") == tmp_path / "a3f9c02b11ee"


@pytest.mark.parametrize("bad", ["", "..", "../x", "a/../../b", "a/b", "/etc",
                                 "..\\x", "a\\b", ".hidden"])
def test_job_dir_rejects_traversal(tmp_path, bad):
    with pytest.raises(ValueError):
        _job_dir(tmp_path, bad)


# ---- HTTP body cap ------------------------------------------------------------

@pytest.fixture()
def hub(tmp_path):
    server = Server(backend="auto", accelerator="auto", device="auto",
                    work_root=tmp_path)
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(server, Auth(user="admin", password=None, api_key=None)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield server, httpd.server_address[1]
    httpd.shutdown()


def test_http_body_over_cap_is_413_without_reading(hub):
    _, port = hub
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        # claim a huge body but send none — the server must refuse up front
        # instead of blocking on rfile.read() or allocating the claimed size
        conn.putrequest("POST", "/v1/prewarm")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(1 << 40))
        conn.endheaders()
        resp = conn.getresponse()
        assert resp.status == 413
    finally:
        conn.close()


def test_artifact_upload_over_cap_is_413(hub):
    server, port = hub
    # a real queued job (routed to a worker that never connects, so it just
    # sits pending) — the 404 guard runs before the body cap
    job_id = server.submit({
        "base_model": "tiny/model", "config": {"method": "lora"},
        "dataset": {"rows": [{"instruction": "a", "output": "b"}],
                    "format": "instruction"},
        "worker": "ghost"})
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.putrequest("POST", f"/v1/workers/ghost/jobs/{job_id}/artifact")
        conn.putheader("Content-Type", "application/gzip")
        conn.putheader("Content-Length", str(1 << 40))
        conn.endheaders()
        resp = conn.getresponse()
        assert resp.status == 413
    finally:
        conn.close()
