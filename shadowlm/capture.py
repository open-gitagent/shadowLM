"""Harness capture — train any agent without opening the box.

Every LLM-based agent, whatever its internals, must talk to a model. That API
boundary is the one interface that always exists — so instead of porting a
harness into an RL framework, run it *unchanged* against a local
OpenAI-compatible proxy that serves your model and records every call:

    model = slm.load(...)
    with slm.capture(model) as proxy:                # http://127.0.0.1:8327/v1
        run_my_agent(base_url=proxy.base_url)        # any OpenAI-client harness
    trajectories = proxy.trajectories()              # reconstructed episodes

    for t in trajectories: t.reward = my_score(t)    # or judge_group(...)
    run = model.finetune(slm.TrajectoryGroup(trajectories), method="grpo")

Reconstruction is conservative and message-level: calls that extend a previous
call's message prefix (the normal multi-turn pattern — the harness appends the
assistant reply and asks again) are merged into one trajectory; unrelated calls
start new ones. Group calls explicitly with an `x-session-id` header when the
harness interleaves conversations.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .rl import Trajectory

DEFAULT_PORT = 8327


@dataclass
class _Call:
    session: str
    messages: list[dict]
    response: dict
    tools: list[dict] | None = None
    ts: float = field(default_factory=time.time)


def _is_prefix(shorter: list[dict], longer: list[dict]) -> bool:
    if len(shorter) > len(longer):
        return False
    return all(a == b for a, b in zip(shorter, longer))


class CaptureProxy:
    """An OpenAI-compatible /v1/chat/completions server over a loaded Model."""

    def __init__(self, model, *, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
        self.model = model
        self.host = host
        self.port = port
        self.calls: list[_Call] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    # ---- lifecycle ----------------------------------------------------------
    def start(self) -> "CaptureProxy":
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet — shadowLM owns the console
                pass

            def do_POST(self):  # noqa: N802 (http.server API)
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    self.send_error(404)
                    return
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                reply = proxy._serve(body, self.headers.get("x-session-id"))
                data = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self) -> "CaptureProxy":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---- serving ------------------------------------------------------------
    def _serve(self, body: dict, session: str | None) -> dict:
        messages = body.get("messages", [])
        tools = body.get("tools")
        reply = self.model.chat(
            messages,
            tools=tools,
            temperature=body.get("temperature", 0.7),
            top_p=body.get("top_p", 0.95),
            max_new_tokens=body.get("max_tokens") or body.get("max_completion_tokens") or 512,
        )
        message = reply.to_message()
        with self._lock:
            self.calls.append(_Call(session=session or "", messages=list(messages),
                                    response=message, tools=tools))
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model.name,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if reply.tool_calls else "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # ---- reconstruction -----------------------------------------------------
    def trajectories(self) -> list[Trajectory]:
        """Reconstruct captured calls into episodes.

        Within a session (explicit `x-session-id`, or implicit), a call whose
        messages extend the previous conversation replaces it — so the final
        trajectory holds the full multi-turn history. Calls that don't extend
        anything start a new trajectory.
        """
        with self._lock:
            calls = list(self.calls)
        episodes: dict[str, list[list]] = {}  # session -> list of [messages, tools]
        for call in calls:
            convo = call.messages + [call.response]
            buckets = episodes.setdefault(call.session, [])
            for bucket in reversed(buckets):
                if _is_prefix(bucket[0], call.messages) or _is_prefix(call.messages, bucket[0]):
                    bucket[0] = convo  # this call extends the episode
                    bucket[1] = call.tools or bucket[1]
                    break
            else:
                buckets.append([convo, call.tools])
        out = []
        for session, buckets in episodes.items():
            for convo, tools in buckets:
                out.append(Trajectory(messages=convo, tools=tools,
                                      metadata={"session": session}))
        return out

    def clear(self) -> None:
        with self._lock:
            self.calls.clear()


def capture(model, *, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> CaptureProxy:
    """Start a capture proxy over `model`. Use as a context manager or call
    .start()/.stop() yourself."""
    return CaptureProxy(model, host=host, port=port)
