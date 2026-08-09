"""The SDK client sends what the server accepts, and ranks the pool on the
capacity the server actually reports.
"""

from __future__ import annotations

import pytest

from shadowlm.remote import RemoteClient, RemoteError


class _Recorder(RemoteClient):
    """Captures requests instead of making them; replays canned responses."""

    def __init__(self, url="http://one", responses=None):
        super().__init__(api_url=url)
        self.sent: list[tuple[str, str, dict | None, str | None]] = []
        self._responses = responses or {}

    def _request(self, method, path, body=None, *, raw=False, timeout=None,
                 retries=2, base=None, blob=None):
        self.sent.append((method, path, body, base))
        target = base or self.api_url
        canned = self._responses.get((path, target), self._responses.get(path))
        if isinstance(canned, Exception):
            raise canned
        return canned if canned is not None else {"job_id": "j1"}


# ---- submit_finetune ------------------------------------------------------------

def test_submit_sends_the_run_name():
    c = _Recorder()
    c.submit_finetune(base_model="m", config={}, dataset={"rows": []},
                      eval_dataset=None, load_in_4bit=False, max_seq_length=512,
                      name="nightly shadow")
    assert c.sent[0][2]["name"] == "nightly shadow"


def test_submit_can_reference_a_server_side_dataset():
    c = _Recorder()
    c.submit_finetune(base_model="m", config={}, dataset=None, eval_dataset=None,
                      load_in_4bit=False, max_seq_length=512, dataset_id="ds123")
    assert c.sent[0][2]["dataset_id"] == "ds123"


def test_submit_needs_rows_or_a_dataset_id():
    c = _Recorder()
    with pytest.raises(ValueError, match="dataset"):
        c.submit_finetune(base_model="m", config={}, dataset=None,
                          eval_dataset=None, load_in_4bit=False,
                          max_seq_length=512)


# ---- pick() ---------------------------------------------------------------------

def _health(running=0, pending=0, gpus=1, workers=0):
    return {"ok": True, "backend": "torch", "version": "0",
            "running": running, "pending": pending, "gpus": gpus,
            "workers": workers}


def test_pick_prefers_the_shallower_queue():
    c = _Recorder("http://busy,http://idle", responses={
        ("/v1/health", "http://busy"): _health(running=3),
        ("/v1/health", "http://idle"): _health(running=0)})
    c.pick()
    assert c.api_url == "http://idle"


def test_pick_counts_idle_workers_as_capacity():
    """Equal queues: the hub with workers attached can absorb more."""
    c = _Recorder("http://solo,http://fleet", responses={
        ("/v1/health", "http://solo"): _health(gpus=1, workers=0),
        ("/v1/health", "http://fleet"): _health(gpus=1, workers=4)})
    c.pick()
    assert c.api_url == "http://fleet"


def test_pick_skips_a_dead_box():
    c = _Recorder("http://dead,http://alive", responses={
        ("/v1/health", "http://dead"): RemoteError("refused"),
        ("/v1/health", "http://alive"): _health()})
    c.pick()
    assert c.api_url == "http://alive"


def test_pick_raises_when_the_whole_pool_is_down():
    c = _Recorder("http://a,http://b", responses={
        "/v1/health": RemoteError("refused")})
    with pytest.raises(RemoteError, match="no reachable"):
        c.pick()
