"""The ShadowLM remote protocol — a thin, pure-stdlib HTTP client.

One protocol, two implementations: `python -m shadowlm.serve` (the reference
server in this package, backed by the local backend) and ShadowLM Studio (the
hosted tier). The SDK's remote backend speaks to either — `backend="remote"`
plus `SHADOWLM_API_URL` / `SHADOWLM_API_KEY` is all a caller configures.
`
Endpoints (JSON over HTTP, optional Bearer auth):

    GET  /v1/health                      → {ok, backend, version}
    POST /v1/finetunes                   → {job_id}
    GET  /v1/finetunes/<id>              → {status, error, checkpoint, final_loss}
    GET  /v1/finetunes/<id>/metrics      → {steps: [...], evals: [...]}
    POST /v1/finetunes/<id>/cancel       → {ok}
    GET  /v1/finetunes/<id>/artifact     → tar.gz of the trained adapter dir
    POST /v1/generate                    → {text}
    POST /v1/chat                        → {text}
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


class RemoteError(RuntimeError):
    """An error returned by (or while reaching) a ShadowLM server."""


class RemoteClient:
    """Stdlib HTTP client for the ShadowLM remote protocol."""

    def __init__(self, api_url: str | None = None, api_key: str | None = None,
                 *, timeout: float = 30.0) -> None:
        self.api_url = (api_url or os.environ.get("SHADOWLM_API_URL")
                        or DEFAULT_API_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("SHADOWLM_API_KEY")
        self.timeout = timeout

    # ---- transport ----------------------------------------------------------
    def _request(self, method: str, path: str, body: dict | None = None,
                 *, raw: bool = False, timeout: float | None = None,
                 retries: int = 2):
        url = f"{self.api_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
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
            f"can't reach ShadowLM server at {self.api_url} ({last}). "
            "Start one with `python -m shadowlm.serve`, or set SHADOWLM_API_URL."
        ) from None

    # ---- protocol operations -------------------------------------------------
    def health(self) -> dict:
        return self._request("GET", "/v1/health")

    def submit_finetune(self, *, base_model: str, config: dict, dataset: dict,
                        eval_dataset: dict | None, load_in_4bit: bool,
                        max_seq_length: int) -> str:
        out = self._request("POST", "/v1/finetunes", {
            "base_model": base_model,
            "config": config,
            "dataset": dataset,
            "eval_dataset": eval_dataset,
            "load_in_4bit": load_in_4bit,
            "max_seq_length": max_seq_length,
        }, timeout=120.0)
        return out["job_id"]

    def job(self, job_id: str) -> dict:
        return self._request("GET", f"/v1/finetunes/{job_id}")

    def metrics(self, job_id: str) -> dict:
        return self._request("GET", f"/v1/finetunes/{job_id}/metrics")

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
