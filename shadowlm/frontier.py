"""A frontier model behind any OpenAI-compatible API, as a ShadowLM model.

The studio talks to the user's frontier model in three places: as a baseline in
an evaluation (does the fine-tune match it?), as the LLM judge that scores
answers, and as the upstream an agent's traffic passes through while it is
captured. All three need only `chat()` or a raw forward, so this is a small
stdlib client — no SDK dependency, and the key never leaves the server.

    fm = FrontierModel("https://api.openai.com/v1", key, "gpt-4o-mini")
    slm.evaluate(fm, ds)                       # the frontier baseline
    slm.evaluate(model, ds, metric="judge", judge=fm)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import __version__


class FrontierError(RuntimeError):
    """The frontier API refused or failed a call; the message says why."""


class _Reply:
    """The slice of a chat reply evaluation and the judge read."""

    def __init__(self, message: dict) -> None:
        self.message = message
        self.content = message.get("content") or ""
        self.raw = self.content
        self.tool_calls = message.get("tool_calls") or []

    def to_message(self) -> dict:
        return self.message

    def __str__(self) -> str:
        return self.content


class FrontierModel:
    """`chat()` against an OpenAI-compatible `/chat/completions` endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str, *, timeout: float = 120.0) -> None:
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("the frontier base URL must start with https:// or http://")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.name = model
        self.timeout = timeout

    def forward(self, body: dict) -> dict:
        """POST a chat/completions body as-is (model forced to ours); the
        provider's JSON response comes back unchanged."""
        payload = {**body, "model": self.name, "stream": False}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}",
                     # some edges refuse urllib's default agent outright
                     "User-Agent": f"shadowlm/{__version__}"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            try:
                detail = json.loads(detail).get("error", {}).get("message", detail)
            except (ValueError, AttributeError):
                pass
            raise FrontierError(f"the frontier model answered {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            raise FrontierError(f"couldn't reach the frontier model: {getattr(e, 'reason', e)}") from None

    def chat(self, messages: list[dict], *, tools=None, temperature: float = 0.0,
             max_new_tokens: int = 512, top_p: float | None = None, **_) -> _Reply:
        body: dict = {"messages": messages, "temperature": temperature,
                      "max_tokens": max_new_tokens}
        if top_p is not None:
            body["top_p"] = top_p
        if tools:
            body["tools"] = tools
        out = self.forward(body)
        try:
            return _Reply(out["choices"][0]["message"])
        except (KeyError, IndexError, TypeError):
            raise FrontierError("the frontier model's reply had no message in it") from None
