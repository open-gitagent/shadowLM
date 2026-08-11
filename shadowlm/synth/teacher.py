"""Teachers — whoever writes the synthetic data.

A teacher is anything that answers `chat(messages) -> str`: a frontier model over
an OpenAI-compatible API (`frontier("gpt-4o")`), a model you loaded yourself with
`slm.load(...)`, or — for self-distillation — the student. The synthesizer only
ever calls `.chat()`, so the three are interchangeable and nothing downstream
knows which one produced a row.

Pure stdlib, like the rest of the transport (see `remote.py`).
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request

_RETRIES = 3
_RETRY_AFTER_S = 1.0  # doubled ×4 per attempt: 1s, 4s
_RETRY_AFTER_CAP = 60.0  # honour Retry-After, but never park a worker for long
_RETRY_CODES = frozenset({408, 429, 500, 502, 503, 504})
_OPENAI = "https://api.openai.com"


class OpenAIChatTeacher:
    """A frontier (or any OpenAI-compatible) model, over plain HTTP.

    Works against OpenAI, vLLM, Ollama, a ShadowLM capture proxy — anything
    serving `/chat/completions`. Retries the transient failures that make long
    synthesis runs fall over; everything else raises with the server's own words.
    """

    def __init__(self, model: str, *, base_url: str | None = None,
                 api_key: str | None = None, parallelism: int = 8,
                 timeout: float = 120.0) -> None:
        self.name = self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        if not self.api_key and self.base_url.startswith(_OPENAI):
            # Fail here, not twenty teacher calls later with a raw provider 401.
            # Keyless is legitimate against a local server, so only the real
            # OpenAI endpoint insists.
            raise ValueError(
                "no API key for the teacher. Pass api_key=, set OPENAI_API_KEY "
                "in the environment the process was started from, or point "
                "base_url= at a local server (vLLM, Ollama) that needs no key.")
        self.parallelism = max(1, parallelism)
        self.timeout = timeout
        self._json_ok = True   # flipped off the first time the server rejects it
        self._token_arg = "max_tokens"  # newer models want max_completion_tokens
        # Tokens the provider actually billed, straight from its `usage` block —
        # ground truth, and the only honest basis for what a run cost. No price
        # table here: those go stale and would quietly lie about money.
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._usage_lock = threading.Lock()

    def chat(self, messages: list[dict], *, temperature: float = 0.7,
             max_new_tokens: int = 1024, json_only: bool = False, **_) -> str:
        """One completion. `json_only=True` requests the server's JSON mode
        (`response_format`), which stops truncated-prose parse failures at the
        source; a server that rejects it gets one plain retry and is never
        asked again."""
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, self._token_arg: max_new_tokens}
        if json_only and self._json_ok:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        delay, attempt = _RETRY_AFTER_S, 0
        while True:
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=json.dumps(payload).encode(),
                headers=headers, method="POST")
            final = attempt == _RETRIES - 1
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                self._record_usage(data.get("usage"))
                return data["choices"][0]["message"].get("content") or ""
            except urllib.error.HTTPError as e:
                # Two capability mismatches a server reports as a plain 400.
                # Adapt once, remember, and retry without consuming an attempt.
                if e.code == 400:
                    detail = e.read()[:400].decode("utf-8", "replace")
                    if "response_format" in payload and "response_format" in detail:
                        self._json_ok = False
                        del payload["response_format"]
                        continue
                    if "max_completion_tokens" in detail and "max_tokens" in payload:
                        # reasoning models rejected the older parameter name
                        self._token_arg = "max_completion_tokens"
                        payload[self._token_arg] = payload.pop("max_tokens")
                        continue
                    raise RuntimeError(
                        f"teacher {self.name!r} returned HTTP 400: {detail[:200]}"
                    ) from None
                if final or e.code not in _RETRY_CODES:
                    detail = e.read()[:200].decode("utf-8", "replace")
                    raise RuntimeError(
                        f"teacher {self.name!r} returned HTTP {e.code}: {detail}"
                    ) from None
                # A rate limiter that tells us how long to wait knows better
                # than our backoff curve does.
                delay = _retry_after(e.headers, delay)
            except (urllib.error.URLError, TimeoutError) as e:
                if final:
                    raise RuntimeError(
                        f"teacher {self.name!r} unreachable at {self.base_url}: {e}"
                    ) from None
            attempt += 1
            time.sleep(delay)
            delay *= 4

    def _record_usage(self, usage) -> None:
        if not isinstance(usage, dict):
            return  # a server that doesn't report usage simply counts as zero
        with self._usage_lock:
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)


def _retry_after(headers, fallback: float) -> float:
    """The server's own Retry-After, in seconds, or our backoff if it said none."""
    raw = headers.get("Retry-After") if headers else None
    try:
        return max(0.0, min(float(raw), _RETRY_AFTER_CAP))
    except (TypeError, ValueError):
        return fallback  # an HTTP-date form, or nothing — keep the curve


class _ModelTeacher:
    """A loaded shadowlm `Model` as a teacher.

    Serialized (`parallelism = 1`): neither backend's generate is thread-safe —
    the same reason the capture proxy holds a generation lock.
    """

    parallelism = 1

    def __init__(self, model) -> None:
        self._model = model
        self.name = getattr(model, "name", "local")

    def chat(self, messages: list[dict], *, temperature: float = 0.7,
             max_new_tokens: int = 1024, **_) -> str:
        return str(self._model.chat(messages, temperature=temperature,
                                    max_new_tokens=max_new_tokens))


class CountingTeacher:
    """Wraps a teacher to meter calls and tokens for the run report."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.name, self.parallelism = inner.name, inner.parallelism
        self.calls = 0
        self._lock = threading.Lock()

    def chat(self, messages, **kwargs) -> str:
        with self._lock:
            self.calls += 1
        return self._inner.chat(messages, **kwargs)

    # A teacher that doesn't report usage — a local model, or a server that
    # omits the block — reads as zero rather than guessing.
    @property
    def prompt_tokens(self) -> int:
        return getattr(self._inner, "prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return getattr(self._inner, "completion_tokens", 0)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def run_jobs(jobs, *, workers: int, on_done=None) -> list:
    """Run `jobs` concurrently, returning results in submission order.

    Order is load-bearing: MoRE+ units are consecutive rows, so results must
    come back in the order they went in. But `on_done(done, total)` fires as
    each job *finishes*, so a caller can report progress instead of going
    silent for the length of the batch — which is what made the studio look
    frozen at 0 while a whole round ran.
    """
    total = len(jobs)
    if workers <= 1 or total <= 1:
        results = []
        for job in jobs:
            results.append(job())
            if on_done:
                on_done(len(results), total)
        return results
    from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: PLC0415

    results: list = [None] * total
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job): i for i, job in enumerate(jobs)}
        for done, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            if on_done:
                on_done(done, total)
    return results


def frontier(model: str, **kwargs) -> OpenAIChatTeacher:
    """A frontier teacher by name: `frontier("gpt-4o")`.

    Reads `OPENAI_API_KEY` / `OPENAI_BASE_URL` unless you pass `api_key=` /
    `base_url=`.
    """
    return OpenAIChatTeacher(model, **kwargs)


def as_teacher(obj):
    """Coerce a `Model`, or anything already teacher-shaped, into a teacher."""
    if obj is None:
        raise ValueError(
            "synthesize needs teacher= — a loaded Model, or "
            "slm.synth.frontier('gpt-4o') for an OpenAI-compatible endpoint")
    if hasattr(obj, "chat") and hasattr(obj, "parallelism"):
        return obj
    if hasattr(obj, "chat"):
        return _ModelTeacher(obj)
    raise TypeError(
        f"{type(obj).__name__} is not a teacher — expected a loaded Model or an "
        "object with .chat(messages)")
