"""The ShadowLM remote protocol — a thin, pure-stdlib HTTP client.

One protocol, two implementations: `python -m shadowlm.serve` (the reference
server in this package, backed by the local backend) and ShadowLM Studio (the
hosted tier). The SDK's remote backend speaks to either — `backend="remote"`
plus `SHADOWLM_API_URL` / `SHADOWLM_API_KEY` is all a caller configures.
`
Endpoints (JSON over HTTP, optional Bearer auth):

    GET  /v1/health                      → {ok, backend, version, gpus, running, pending}
    POST /v1/finetunes                   → {job_id}
    GET  /v1/finetunes/<id>              → {status, error, checkpoint, final_loss}
    GET  /v1/finetunes/<id>/metrics      → {steps: [...], evals: [...]}
    GET  /v1/finetunes/<id>/logs         → {logs: [...]}   the server's console
    POST /v1/finetunes/<id>/cancel       → {ok}
    GET  /v1/finetunes/<id>/artifact     → tar.gz of the trained adapter dir
    POST /v1/generate                    → {text}
    POST /v1/chat                        → {text}
    GET  /v1/workers                     → {workers: [...]} connected devices
    GET  /v1/workers/<name>/socket       → websocket upgrade (see worker.py)
    POST /v1/workers/<n>/jobs/<id>/artifact ← a worker ships its adapter home

`SHADOWLM_API_URL` may name several servers, comma-separated — `pick()` binds the
client to the least-busy reachable one and stays there for the session. That is
the whole "cluster": more boxes, each already using all of its own GPUs (the torch
backend places one job across them with device_map="auto"). Deliberately *not* a
scheduler — see `pick()`.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_API_URL = "http://127.0.0.1:8329"
_RETRYABLE = (urllib.error.URLError, ConnectionError, TimeoutError)


def _user_agent() -> str:
    """Identify ourselves. Servers behind Cloudflare 403 the stdlib's default
    `Python-urllib/x.y` agent (error 1010), which is a confusing way to discover
    that your GPU box is proxied."""
    from . import __version__  # noqa: PLC0415  (lazy: avoids an import cycle)

    return f"shadowlm/{__version__}"


class RemoteError(RuntimeError):
    """An error returned by (or while reaching) a ShadowLM server."""


class RemoteClient:
    """Stdlib HTTP client for the ShadowLM remote protocol."""

    def __init__(self, api_url: str | None = None, api_key: str | None = None,
                 *, timeout: float = 30.0) -> None:
        raw_url = (api_url or os.environ.get("SHADOWLM_API_URL") or DEFAULT_API_URL)
        self.pool = [u.strip().rstrip("/") for u in raw_url.split(",") if u.strip()]
        self.api_url = self.pool[0]
        self.api_key = api_key or os.environ.get("SHADOWLM_API_KEY")
        self.timeout = timeout

    # ---- transport ----------------------------------------------------------
    def _request(self, method: str, path: str, body: dict | None = None,
                 *, raw: bool = False, timeout: float | None = None,
                 retries: int = 2, base: str | None = None,
                 blob: bytes | None = None):
        base = base or self.api_url
        url = f"{base}{path}"
        data = blob if blob is not None else (
            json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/gzip" if blob is not None
                       else "application/json")
        req.add_header("User-Agent", _user_agent())
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                    payload = resp.read()
                    return payload if raw else json.loads(payload or b"{}")
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                try:
                    detail = json.loads(detail).get("error", detail)
                except (json.JSONDecodeError, AttributeError):
                    pass
                raise RemoteError(f"{method} {path} → HTTP {e.code}: {detail}") from None
            except _RETRYABLE as e:  # transient: connection refused/reset, timeout
                last = e
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RemoteError(
            f"can't reach ShadowLM server at {base} ({last}). "
            "Start one with `python -m shadowlm.serve`, or set SHADOWLM_API_URL."
        ) from None

    # ---- protocol operations -------------------------------------------------
    def health(self, *, base: str | None = None) -> dict:
        return self._request("GET", "/v1/health", base=base)

    def pick(self) -> dict:
        """Bind to the least-busy reachable server in the pool; return its health.

        ponytail: client-side routing, not a scheduler. Rank by queue depth (then
        by GPU count) and pin for the session — good enough for a handful of boxes,
        and it needs no control plane. Reach for a real scheduler (SkyPilot/Ray)
        only when the boxes outnumber the people, not before.
        """
        if len(self.pool) == 1:
            return self.health()  # single server: this is just the fail-fast check
        ranked = []
        for url in self.pool:
            try:
                h = self.health(base=url)
            except RemoteError:
                continue  # a dead box in the pool is skipped, not fatal
            ranked.append(((h.get("running", 0) + h.get("pending", 0),
                            -h.get("gpus", 0)), url, h))
        if not ranked:
            raise RemoteError(
                f"no reachable ShadowLM server in the pool: {', '.join(self.pool)}")
        ranked.sort(key=lambda r: r[0])
        _, self.api_url, health = ranked[0]
        return health

    def submit_finetune(self, *, base_model: str, config: dict, dataset: dict,
                        eval_dataset: dict | None, load_in_4bit: bool,
                        max_seq_length: int, worker: str | None = None) -> str:
        out = self._request("POST", "/v1/finetunes", {
            "base_model": base_model,
            "config": config,
            "dataset": dataset,
            "eval_dataset": eval_dataset,
            "load_in_4bit": load_in_4bit,
            "max_seq_length": max_seq_length,
            "worker": worker,  # route to a registered worker instead of the hub
        }, timeout=120.0)
        return out["job_id"]

    def job(self, job_id: str) -> dict:
        return self._request("GET", f"/v1/finetunes/{job_id}")

    def metrics(self, job_id: str) -> dict:
        return self._request("GET", f"/v1/finetunes/{job_id}/metrics")

    def logs(self, job_id: str) -> list[str]:
        """The server's captured console for a job (tail-capped, oldest first).

        The last entry may be a half-drawn line (the server appends its
        in-progress, post-`\\r` line) — callers streaming live should hold it back.
        """
        return self._request("GET", f"/v1/finetunes/{job_id}/logs").get("logs", [])

    def cancel(self, job_id: str) -> None:
        self._request("POST", f"/v1/finetunes/{job_id}/cancel")

    def download_artifact(self, job_id: str, dest: str | Path) -> str:
        blob = self._request("GET", f"/v1/finetunes/{job_id}/artifact",
                             raw=True, timeout=300.0)
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            try:
                tar.extractall(dest, filter="data")
            except TypeError:  # Python < 3.12 has no filter= kwarg
                tar.extractall(dest)  # noqa: S202 — archive comes from our server
        return str(dest)

    # ---- worker side: a machine that executes the hub's jobs -----------------
    # (the live wire is a websocket — see ws.py / worker.py; only the one-shot
    #  blob upload and the device list ride plain HTTP)
    def workers(self) -> list[dict]:
        return self._request("GET", "/v1/workers").get("workers", [])

    def upload_artifact(self, name: str, job_id: str, blob: bytes) -> None:
        self._request("POST", f"/v1/workers/{name}/jobs/{job_id}/artifact",
                      blob=blob, timeout=300.0)

    def generate(self, *, model: str, adapter: str | None, prompt: str,
                 max_new_tokens: int, temperature: float, top_p: float) -> str:
        out = self._request("POST", "/v1/generate", {
            "model": model, "adapter": adapter, "prompt": prompt,
            "max_new_tokens": max_new_tokens, "temperature": temperature,
            "top_p": top_p,
        }, timeout=600.0)
        return out["text"]

    def chat(self, *, model: str, adapter: str | None, messages: list[dict],
             tools: list[dict] | None, max_new_tokens: int, temperature: float,
             top_p: float) -> str:
        out = self._request("POST", "/v1/chat", {
            "model": model, "adapter": adapter, "messages": messages,
            "tools": tools, "max_new_tokens": max_new_tokens,
            "temperature": temperature, "top_p": top_p,
        }, timeout=600.0)
        return out["text"]
