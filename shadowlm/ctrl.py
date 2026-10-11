"""Ctrl agent: the cockpit's conversation, answered by Claude.

The cockpit narrates a project from its state (rule-based, always true). What
the person *types* comes here: Claude reads the project's state and the thread,
answers in a few plain sentences, and may propose one next step by calling a
tool. A tool call is never executed — the studio turns it into the same action
card the rules propose, and nothing runs until the person approves it there.

The key is the server's (`ANTHROPIC_API_KEY`); it never reaches the browser.
Without one the studio keeps its rule-based replies.

    agent = CtrlAgent.from_env()
    agent.reply(state, thread, "why did v2 score worse?")
    # -> {"reply": "...", "action": {"kind": "finetune", "args": {...}} | None}
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from . import __version__

DEFAULT_MODEL = "claude-sonnet-5-5"
_API = "https://api.anthropic.com/v1"
# what the cockpit can put on a card; one per reply at most
_METHODS = ["lora", "sdft", "more", "qlora", "dora"]

TOOLS = [
    {"name": "propose_finetune",
     "description": "Propose training the next version of this project's model. The studio starts from its own "
                    "recipe (base model, method and steps picked from the goal, data and hardware); set only what "
                    "should change from it.",
     "input_schema": {"type": "object", "properties": {
         "method": {"type": "string", "enum": _METHODS,
                    "description": "lora: learns a task's pattern; sdft: like lora but keeps more of the base "
                                   "model's general skills; more: each fact becomes a retrieval expert, answers "
                                   "come back exactly as written; qlora/dora: lora variants"},
         "max_steps": {"type": "integer", "minimum": 1, "maximum": 100000},
         "base_model": {"type": "string", "description": "a Hugging Face model id; omit to keep the recipe's"},
     }}},
    {"name": "propose_evaluate",
     "description": "Propose evaluating the current version on the project's questions, against its base model "
                    "(and the frontier model, when one is set up, which then also judges the answers).",
     "input_schema": {"type": "object", "properties": {"with_frontier": {"type": "boolean"}}}},
    {"name": "propose_write_examples",
     "description": "Propose having the frontier model write more training examples in the style of the project's "
                    "own; its examples stay in. Needs a frontier model and at least one example.",
     "input_schema": {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 1000}},
                      "required": ["n"]}},
    {"name": "propose_deploy",
     "description": "Propose deploying the current version behind an OpenAI-compatible endpoint.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "open_playground",
     "description": "Offer to open the current version in the Playground, to chat with it beside its base model.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "add_frontier_model",
     "description": "Offer to set up the frontier model (any OpenAI-compatible API) used as a baseline, judge and "
                    "example writer.",
     "input_schema": {"type": "object", "properties": {}}},
]
_KINDS = {"propose_finetune": "finetune", "propose_evaluate": "evaluate", "propose_write_examples": "synthesize",
          "propose_deploy": "deploy", "open_playground": "playground", "add_frontier_model": "add-frontier"}

SYSTEM = """You are Ctrl agent, the assistant inside OpenFinetuner's cockpit. The cockpit makes one fine-tuned model \
for one job: Data (examples) -> Fine-tune -> Evaluate -> Deploy, versions compared on evidence. The person may be a \
business user or an ML researcher; answer at their level, in plain words, without jargon they didn't use.

How you work:
- Reply in at most three short sentences; fewer is better. No headings, no lists unless they ask for steps, \
no emoji. Don't restate the numbers the cockpit already shows unless they asked for them.
- Use only the facts in the project state you are given. If something isn't there, say you can't see it. Never \
invent scores, step counts, model names or errors.
- You cannot run or set up anything. To suggest an action, call exactly one tool; the studio shows it as a card \
the person approves. Never say you started, ran, set up, queued or changed something: say what the card will do \
("Here's version 2 with SDFT at 300 steps; start it when you're ready").
- Propose an action when the person asks for one or it is clearly the next step; for a question, just answer.
- A tool can't do what the state rules out (no fine-tune finished -> no evaluate or deploy; no examples -> no \
fine-tune; no frontier model -> no writing examples, offer to add one instead). Say what has to happen first.
- When a run failed, explain the cause from its log lines in plain words and what would fix it.
- Thin data (under ~50 examples) usually needs more examples before more steps. SDFT keeps more of the base \
model's general skills than LoRA. MoRE is for facts that must come back word for word."""


class CtrlError(RuntimeError):
    """The model API refused or failed a call; the message says why."""


class CtrlAgent:
    """Claude behind the cockpit's conversation. Stdlib only, like the frontier client."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, base_url: str = _API, timeout: float = 60.0) -> None:
        if not api_key:
            raise ValueError("Ctrl agent needs an Anthropic API key")
        self.api_key, self.model, self.base_url, self.timeout = api_key, model, base_url.rstrip("/"), timeout

    @classmethod
    def from_env(cls) -> CtrlAgent | None:
        """The agent ANTHROPIC_API_KEY configures, or None (the studio then keeps its rules)."""
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            return None
        return cls(key, model=os.environ.get("SHADOWLM_CTRL_MODEL", "").strip() or DEFAULT_MODEL,
                   base_url=os.environ.get("SHADOWLM_CTRL_BASE_URL", "").strip() or _API)

    def info(self) -> dict:
        return {"model": self.model}

    def reply(self, state: dict, thread: list[dict], message: str) -> dict:
        """Answer `message`. Returns {"reply": text, "action": {"kind", "args"} | None}."""
        transcript = "\n".join(f"{'Person' if t.get('from') == 'you' else 'Ctrl agent'}: {t.get('text', '')}"
                               for t in thread[-16:] if t.get("text"))
        content = (f"Project state (JSON):\n{json.dumps(state, indent=1, default=str)[:24000]}\n\n"
                   f"Conversation so far:\n{transcript or '(none)'}\n\nThe person now says: {message}")
        body = {"model": self.model, "max_tokens": 1024, "system": SYSTEM, "tools": TOOLS,
                "messages": [{"role": "user", "content": content}]}
        out = self._post("/messages", body)
        text = " ".join(b.get("text", "").strip() for b in out.get("content", []) if b.get("type") == "text").strip()
        action = None
        for b in out.get("content", []):
            if b.get("type") == "tool_use" and b.get("name") in _KINDS:
                action = {"kind": _KINDS[b["name"]], "args": b.get("input") or {}}
                break
        return {"reply": text, "action": action}

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path, data=json.dumps(body).encode(), method="POST",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json", "user-agent": f"shadowlm/{__version__}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get("error", {}).get("message", "")
            except (ValueError, OSError):
                detail = ""
            raise CtrlError(f"the model API answered {e.code}{f': {detail}' if detail else ''}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise CtrlError(f"couldn't reach the model API: {getattr(e, 'reason', e)}") from None
