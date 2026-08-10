"""Login attempts are throttled per address, and dataset ids never reach the
filesystem unvalidated.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from shadowlm.serve import Auth, DatasetStore, LoginThrottle, Server, make_handler


# ---- throttle unit ------------------------------------------------------------

def test_throttle_locks_after_consecutive_failures():
    now = [0.0]
    t = LoginThrottle(limit=3, window=60.0, clock=lambda: now[0])
    for _ in range(3):
        assert t.allowed("1.2.3.4")
        t.record("1.2.3.4", ok=False)
    assert not t.allowed("1.2.3.4")
    assert t.allowed("5.6.7.8")  # other addresses unaffected


def test_throttle_unlocks_after_window_and_resets_on_success():
    now = [0.0]
    t = LoginThrottle(limit=3, window=60.0, clock=lambda: now[0])
    for _ in range(3):
        t.record("1.2.3.4", ok=False)
    assert not t.allowed("1.2.3.4")
    now[0] = 61.0
    assert t.allowed("1.2.3.4")
    t.record("1.2.3.4", ok=True)  # success clears the strike count
    t.record("1.2.3.4", ok=False)
    assert t.allowed("1.2.3.4")


# ---- throttle wired into /v1/login ---------------------------------------------

@pytest.fixture()
def authed_hub(tmp_path):
    server = Server(backend="auto", accelerator="auto", device="auto",
                    work_root=tmp_path)
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(server, Auth(user="admin", password="right-horse",
                                  api_key=None)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()


def _login(port: int, password: str) -> int:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/login", method="POST",
        data=json.dumps({"username": "admin", "password": password}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def test_login_throttled_after_repeated_failures(authed_hub):
    for _ in range(5):
        assert _login(authed_hub, "wrong") == 401
    assert _login(authed_hub, "wrong") == 429
    # even the right password is refused while locked — the lock is the point
    assert _login(authed_hub, "right-horse") == 429


# ---- dataset ids are plain ids --------------------------------------------------

@pytest.mark.parametrize("bad", ["../x", "a/b", "..", "", "a\\b", ".hidden"])
def test_dataset_store_refuses_path_shaped_ids(tmp_path, bad):
    store = DatasetStore(tmp_path / "datasets")
    assert store.rows(bad) is None
    assert store.meta(bad) is None
    assert store.delete(bad) is False


def test_dataset_store_roundtrip_still_works(tmp_path):
    store = DatasetStore(tmp_path / "datasets")
    ds_id = store.save("mine", [{"instruction": "a", "output": "b"}])["dataset_id"]
    assert store.meta(ds_id)["name"] == "mine"
    assert store.rows(ds_id) == [{"instruction": "a", "output": "b"}]
    assert store.delete(ds_id) is True
