"""SDFT (self-distillation fine-tuning) helpers shared by both backends.

The method (arXiv 2601.19897): sample a completion on-policy from the student,
then push its per-token distributions toward the same model reading the row's
golden response in-context — the demonstration-conditioned model is its own
teacher. These helpers build the two contexts and the logit alignment; they are
pure stdlib (no torch, no mlx) so both backends and the CPU tests share them.
"""

from __future__ import annotations

SDFT_TOP_P = 0.95  # nucleus cutoff for the on-policy rollouts (paper setting)

# The paper's teacher suffix, appended to the final user turn. The demonstration
# lands in-context so the teacher's next-token distributions carry it; the
# wording asks for a fresh response so the teacher doesn't just parrot it.
DEFAULT_TEACHER_TEMPLATE = (
    "\n\nThis is an example for a response to the question:\n{demonstration}"
    "\n\nNow answer with a response of your own, including the thinking process."
)


def _where(row_index: int | None) -> str:
    return "the row" if row_index is None else f"row {row_index}"


def split_demonstration(messages: list[dict], *, row_index: int | None = None,
                        ) -> tuple[list[dict], str]:
    """(student context, demonstration text) for one chat row.

    The student context is every message but the last; the demonstration is the
    final assistant turn's content. Empty demonstrations are returned, not
    raised — the training loops skip and count them.
    """
    if not messages or messages[-1].get("role") != "assistant":
        last = messages[-1].get("role") if messages else None
        raise ValueError(
            f"method='sdft' needs chat rows that end with the assistant "
            f"demonstration ({_where(row_index)} ends with role {last!r}) — "
            f"each row: [(system,) user, ..., assistant]"
        )
    demo = messages[-1].get("content") or ""
    if not isinstance(demo, str):
        raise ValueError(
            f"method='sdft' needs plain-text assistant content "
            f"({_where(row_index)} has {type(demo).__name__})"
        )
    return messages[:-1], demo


def teacher_messages(messages: list[dict], template: str | None = None, *,
                     row_index: int | None = None) -> list[dict]:
    """The teacher context: the student context with the rendered demonstration
    suffix appended to the last user turn. Returns copied dicts — never mutates
    the dataset row."""
    ctx, demo = split_demonstration(messages, row_index=row_index)
    tpl = DEFAULT_TEACHER_TEMPLATE if template is None else template
    if "{demonstration}" not in tpl:
        raise ValueError(
            "sdft_teacher_template must contain a '{demonstration}' placeholder"
        )
    last_user = next((i for i in range(len(ctx) - 1, -1, -1)
                      if ctx[i].get("role") == "user"), None)
    if last_user is None:
        raise ValueError(
            f"method='sdft' found no user turn to carry the demonstration "
            f"({_where(row_index)}) — text-format rows can't train with sdft; "
            f"use chat or instruction data"
        )
    out = [dict(m) for m in ctx]
    # .replace, not .format — braces in demonstrations or templates must survive.
    out[last_user]["content"] = (
        (out[last_user].get("content") or "") + tpl.replace("{demonstration}", demo)
    )
    return out


def completion_slice(prompt_len: int, total_len: int) -> slice:
    """Logit positions that predict the completion tokens.

    A causal LM's logits at position i predict token i+1: with completion
    tokens at input positions [prompt_len, total_len), the predicting logits
    sit at [prompt_len - 1, total_len - 1). Feed the model the sequence minus
    its final token and this slice selects exactly the completion's
    distributions — the same arithmetic for the student and teacher sequences,
    whose prompts differ in length but share the completion.
    """
    if not 1 <= prompt_len < total_len:
        raise ValueError(
            f"prompt_len must satisfy 1 <= prompt_len < total_len "
            f"(got prompt_len={prompt_len}, total_len={total_len})"
        )
    return slice(prompt_len - 1, total_len - 1)
