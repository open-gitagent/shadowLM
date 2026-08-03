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
_RETRY_CODES = frozenset({408, 429, 500, 502, 503, 504})


class OpenAIChatTeacher:
    """A frontier (or any OpenAI-compatible) model, over plain HTTP.

    Works against OpenAI, vLLM, Ollama, a ShadowLM capture proxy — anything
    serving `/chat/completions`. Retries the transient failures that make long
    synthesis runs fall over; everything else raises with the server's own words.
    """

    def __init__(self, model: str, *, base_url: str | None = None,
                 api_key: str | None = None, parallelism: int = 4,
                 timeout: float = 120.0) -> None:
        self.name = self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self.parallelism = max(1, parallelism)
        self.timeout = timeout

    def chat(self, messages: list[dict], *, temperature: float = 0.7,
             max_new_tokens: int = 1024, **_) -> str:
        body = json.dumps({"model": self.model, "messages": messages,
                           "temperature": temperature,
                           "max_tokens": max_new_tokens}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions",
                                     data=body, headers=headers, method="POST")
        delay = _RETRY_AFTER_S
        for attempt in range(_RETRIES):
            final = attempt == _RETRIES - 1
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read())
                return payload["choices"][0]["message"].get("content") or ""
            except urllib.error.HTTPError as e:
                if final or e.code not in _RETRY_CODES:
                    detail = e.read()[:200].decode("utf-8", "replace")
                    raise RuntimeError(
                        f"teacher {self.name!r} returned HTTP {e.code}: {detail}"
                    ) from None
            except (urllib.error.URLError, TimeoutError) as e:
                if final:
                    raise RuntimeError(
                        f"teacher {self.name!r} unreachable at {self.base_url}: {e}"
                    ) from None
            time.sleep(delay)
            delay *= 4


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
    """Wraps a teacher to count calls for the run report."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.name, self.parallelism = inner.name, inner.parallelism
        self.calls = 0
        self._lock = threading.Lock()

    def chat(self, messages, **kwargs) -> str:
        with self._lock:
            self.calls += 1
        return self._inner.chat(messages, **kwargs)


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
