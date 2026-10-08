"""Quality control — validate, deduplicate, judge.

Synthetic data is cheap to make and easy to make badly, so nothing reaches a
dataset without passing through here. Every rejection is counted rather than
quietly dropped: the run report has to be able to tell you the truth about what
survived and why the rest didn't.
"""

from __future__ import annotations

import json
import re

from ..apo import _norm
from ..more_plus import _tokenize

_ROLES = frozenset({"system", "user", "assistant", "tool"})

# Two tells that a teacher phoned it in: an un-filled template slot, or the
# refusal boilerplate that would teach the student to refuse.
_JUNK = re.compile(
    r"\[(?:NAME|COMPANY|DATE|TODO|INSERT|PRODUCT|X)\]|as an AI language model",
    re.IGNORECASE)


def first_json_array(text: str) -> list | None:
    """The first valid JSON array in `text` — teachers wrap JSON in prose."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "[":
            try:
                obj, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, list):
                return obj
    return None


def validate(messages: list[dict], *, tools: list[dict] | None = None) -> list[str]:
    """Everything wrong with a generated conversation — empty list means usable.

    The last rule is the load-bearing one: the torch backend only takes the
    prompt-masking path when every row ends on an assistant turn, so a row that
    doesn't is worse than useless — it silently degrades the whole batch.
    """
    if not messages:
        return ["no messages"]
    problems = []
    roles = [m.get("role") for m in messages]
    unknown = set(roles) - _ROLES
    if unknown:
        problems.append(f"unknown roles {sorted(map(str, unknown))}")
    if roles.count("system") > 1 or ("system" in roles[1:]):
        problems.append("a system turn may appear only once, first")
    last = messages[-1]
    if last.get("role") != "assistant":
        problems.append("the conversation must end with an assistant turn")
    elif not (last.get("content") or last.get("tool_calls")):
        problems.append("the final assistant turn is empty")
    problems += _tool_problems(messages, tools)
    if _JUNK.search(" ".join(str(m.get("content") or "") for m in messages)):
        problems.append("contains a placeholder or assistant boilerplate")
    return problems


def _tool_problems(messages: list[dict], tools: list[dict] | None) -> list[str]:
    """Tool calls that don't parse, aren't declared, or are never answered."""
    declared = {(t.get("function") or {}).get("name") for t in (tools or [])}
    problems: list[str] = []
    unanswered: list[str] = []
    for msg in messages:
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = fn.get("name")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    json.loads(args)
                except json.JSONDecodeError:
                    problems.append(f"tool call {name!r} has unparseable arguments")
            elif not isinstance(args, dict):
                problems.append(f"tool call {name!r} has no arguments")
            if declared and name not in declared:
                problems.append(f"calls undeclared tool {name!r}")
            unanswered.append(call.get("id"))
        if msg.get("role") == "tool":
            call_id = msg.get("tool_call_id")
            if call_id in unanswered:
                unanswered.remove(call_id)
            else:
                problems.append(f"tool result for unknown call {call_id!r}")
    if unanswered:
        problems.append(f"{len(unanswered)} tool call(s) never answered")
    return problems


def jaccard(a: set, b: set) -> float:
    """Token-set overlap, 0–1. Used for both dedup and paraphrase diversity."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class Dedup:
    """Rejects repeats: exact on normalized text, near on query-side overlap.

    Mode collapse is *the* failure of LLM synthesis — a teacher asked for
    variety will happily rewrite one example five hundred times — so near-dup
    rejection is on by default. One pass over the accepted pool per candidate is
    fine at the scale a run produces; swap in minhash if that ever changes.
    """

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold
        self._exact: set[str] = set()
        self._tokens: list[set[str]] = []

    def seed(self, texts) -> None:
        """Pre-load texts to reject against — real episodes we must not clone."""
        for text in texts:
            self._exact.add(_norm(text))
            self._tokens.append(set(_tokenize(text)))

    def accept(self, text: str, *, key: str | None = None) -> bool:
        """True when `text` is new (and now remembered), False when it repeats.

        `key` is the query-side text near-duplicates are judged on; two rows with
        different answers to the same question are still a duplicate question.
        """
        exact = _norm(text)
        if exact in self._exact:
            return False
        tokens = set(_tokenize(key if key is not None else text))
        if tokens and any(jaccard(tokens, seen) >= self.threshold
                          for seen in self._tokens):
            return False
        self._exact.add(exact)
        self._tokens.append(tokens)
        return True
