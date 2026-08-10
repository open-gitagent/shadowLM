"""SDPO (self-distillation policy optimization) helpers shared by both backends.

The method (arXiv 2601.20802): sample a group of rollouts per prompt, score
them with the caller's reward fns, then push each rollout's per-token
distributions toward the same model reading feedback in-context — a successful
sibling rollout as the "correct solution", and/or textual feedback the reward
fn returned. These helpers score groups and build the teacher's reprompt; they
are pure stdlib (no torch, no mlx) so both backends and the CPU tests share
them.
"""

from __future__ import annotations

# The reference implementation's reprompt wording (lasgroup/SDPO), verbatim.
# Sections drop out when absent.
SDPO_REPROMPT_TEMPLATE = (
    "{prompt}{solution}{feedback}\n\n"
    "Correctly solve the original question.\n"
)
SDPO_SOLUTION_TEMPLATE = (
    "\n"
    "Correct solution:\n\n"
    "{successful_previous_attempt}\n\n"
)
SDPO_FEEDBACK_TEMPLATE = (
    "\n"
    "The following is feedback from your unsuccessful earlier attempt:\n\n"
    "{feedback_raw}\n\n"
)


def score_group(outputs_per_fn: list, n: int, *,
                fn_names: list[str] | None = None,
                ) -> tuple[list[float], list[str | None]]:
    """Normalize one rollout group's reward-fn outputs.

    Each fn returns one element per completion: a float score, or a
    ``(score, feedback_text)`` pair to hand the self-teacher rich feedback.
    Scores sum across fns; non-empty feedback joins with a blank line.
    """
    scores = [0.0] * n
    notes: list[list[str]] = [[] for _ in range(n)]
    for k, out in enumerate(outputs_per_fn):
        name = (fn_names[k] if fn_names and k < len(fn_names)
                else f"reward_fns[{k}]")
        if not isinstance(out, (list, tuple)):
            raise ValueError(
                f"reward fn {name!r} returned {type(out).__name__} — expected "
                f"a list of {n} float scores or (score, feedback) pairs"
            )
        if len(out) != n:
            raise ValueError(
                f"reward fn {name!r} returned {len(out)} score(s) for {n} "
                "completion(s) — one per completion"
            )
        for i, el in enumerate(out):
            feedback = None
            if isinstance(el, (tuple, list)):
                if len(el) != 2 or not isinstance(el[1], str):
                    raise ValueError(
                        f"reward fn {name!r} element {i} is {el!r} — a pair "
                        "must be (score, feedback_text)"
                    )
                el, feedback = el
            if not isinstance(el, (int, float)):
                raise ValueError(
                    f"reward fn {name!r} element {i} is {type(el).__name__} — "
                    "expected a float score or a (score, feedback) pair"
                )
            scores[i] += float(el)
            if feedback and feedback.strip():
                notes[i].append(feedback)
    return scores, ["\n\n".join(f) if f else None for f in notes]


def pick_solution(index: int, scores: list[float], texts: list[str],
                  threshold: float) -> str | None:
    """The first *sibling* rollout scoring at/above the threshold, if any.

    A rollout never teaches itself its own success (the paper's runs set
    ``dont_reprompt_on_self_success``) — with a lone success in the group the
    failures learn from it while the success carries no signal that step.
    """
    for j, s in enumerate(scores):
        if j != index and s >= threshold:
            return texts[j]
    return None


def teacher_prompt(prompt: str, solution: str | None,
                   feedback: str | None) -> str | None:
    """The teacher's user-turn content, or None when there is nothing to
    condition on — no solution and no feedback means no signal, and the
    training loops skip the rollout entirely (the reference masks it out).
    """
    if solution is None and feedback is None:
        return None
    parts = {
        "{prompt}": prompt,
        "{solution}": "" if solution is None else SDPO_SOLUTION_TEMPLATE.replace(
            "{successful_previous_attempt}", solution),
        "{feedback}": "" if feedback is None else SDPO_FEEDBACK_TEMPLATE.replace(
            "{feedback_raw}", feedback),
    }
    # One left-to-right pass over the reprompt template: substituted text is
    # never re-scanned, so braces in prompts, solutions, or feedback survive.
    out, rest = [], SDPO_REPROMPT_TEMPLATE
    while rest:
        hits = [(rest.index(key), key) for key in parts if key in rest]
        if not hits:
            out.append(rest)
            break
        pos, key = min(hits)
        out.append(rest[:pos])
        out.append(parts[key])
        rest = rest[pos + len(key):]
    return "".join(out)
