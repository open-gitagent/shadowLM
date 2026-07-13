"""Remote log streaming + the server pool — the two non-obvious bits.

Log draining has to hold back the server's in-progress line, filter the step
lines it already streams as metrics, and survive the server trimming its tail.
`pick()` has to route to the least-busy box and tolerate dead ones.
"""

import pytest

from shadowlm.backends.base import Callbacks
from shadowlm.backends.remote import RemoteBackend
from shadowlm.remote import RemoteClient, RemoteError

JOB = "abcdef1234"
TAG = JOB[:8]


class _FakeClient:
    """Serves a scripted log tail in place of a real server."""

    def __init__(self, *frames):
        self.frames = list(frames)

    def logs(self, job_id):
        return self.frames.pop(0) if self.frames else []


class _DeadClient:
    def logs(self, job_id):
        raise RemoteError("server went away mid-run")


def _backend(client):
    be = RemoteBackend.__new__(RemoteBackend)  # skip __init__: no network here
    be._client = client
    return be


def test_drain_holds_back_live_line_and_filters_step_noise():
    seen = []
    cb = Callbacks(on_log=seen.append)
    be = _backend(_FakeClient(
        ["banner", f"[{TAG}] step 1 · loss 2.0", "half-drawn-progress-bar"],
        ["banner", f"[{TAG}] step 1 · loss 2.0", "done · final loss 0.5"],
    ))

    mark = be._drain_logs(JOB, 0, cb, final=False)
    # the trailing line is held back (it may be the in-progress one), and the
    # step line is dropped — it already arrived as a metric.
    assert seen == ["banner"]
    assert mark == 2

    be._drain_logs(JOB, mark, cb, final=True)  # job over: flush the tail
    assert seen == ["banner", "done · final loss 0.5"]


def test_drain_resyncs_when_the_server_trims_its_tail():
    seen = []
    be = _backend(_FakeClient(["only-two", "lines-left"]))
    # we think we're 50 lines in, but the capped tail is 2: don't crash, don't repeat
    assert be._drain_logs(JOB, 50, Callbacks(on_log=seen.append), final=True) == 2
    assert seen == []


def test_drain_never_fails_the_run():
    be = _backend(_DeadClient())
    assert be._drain_logs(JOB, 7, Callbacks(), final=True) == 7  # mark unchanged


def test_pick_binds_to_the_least_busy_box_and_skips_dead_ones(monkeypatch):
    client = RemoteClient("http://a,http://b,http://dead")
    assert client.pool == ["http://a", "http://b", "http://dead"]
    boxes = {
        "http://a": {"running": 2, "pending": 1, "gpus": 1},
        "http://b": {"running": 0, "pending": 0, "gpus": 1},  # idle → wins
    }

    def fake_health(self, *, base=None):
        try:
            return boxes[base or self.api_url]
        except KeyError:
            raise RemoteError("connection refused") from None

    monkeypatch.setattr(RemoteClient, "health", fake_health)
    assert client.pick()["gpus"] == 1
    assert client.api_url == "http://b"


def test_requests_identify_themselves(monkeypatch):
    """Cloudflare 403s the stdlib's default agent (error 1010), which cost us a
    live GPU run to discover. Any server behind a proxy needs a real User-Agent."""
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["ua"] = req.get_header("User-agent")
        raise RemoteError("stop here — we only care about the headers")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RemoteError):
        RemoteClient("http://box").health()

    assert captured["ua"].startswith("shadowlm/")
    assert "urllib" not in captured["ua"]


def test_pick_raises_when_the_whole_pool_is_down(monkeypatch):
    client = RemoteClient("http://x,http://y")

    def dead(self, *, base=None):
        raise RemoteError("connection refused")

    monkeypatch.setattr(RemoteClient, "health", dead)
    with pytest.raises(RemoteError, match="no reachable"):
        client.pick()
