"""Generation — scenarios first, then instances.

The anti-mode-collapse mechanism here is structural, not a clever prompt: we
never ask a teacher for "500 examples". We ask it to lay out a *taxonomy* of
distinct scenarios, then generate a few instances per scenario with rotated user
styles. A teacher asked for variety in one breath will rewrite one example five
hundred times; a teacher asked to fill a named slot won't.

Two axes compose here. Where the scenarios come from is the **seed** (a task
description, a document's facts, or real episodes to vary), and what gets
written per scenario is the **mode** (a conversation, a preference pair, or a
set of paraphrases for retrieval routing). Any seed works with any mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..apo import _norm
from ..models import _first_json_object
from ..more_plus import _tokenize
from ..rl import Trajectory
from .quality import first_json_array, jaccard, validate

# Asking for more than this in one array is where teachers start repeating
# themselves and truncating JSON.
_MAX_LEAVES_PER_CALL = 25

# Rotated so consecutive rows never read like the same person wrote them.
STYLES = (
    "terse, a single line",
    "verbose, with background detail the model must look past",
    "non-native English, slightly broken grammar",
    "lowercase and typo-prone",
    "frustrated — this is their second attempt at getting help",
    "formal and businesslike",
    "asks two things at once",
    "vague, so the model must ask or state a reasonable assumption",
    "pastes in a log line or error message",
    "polite and chatty before getting to the point",
    "keyword-style, barely a sentence",
    "confident but wrong about one premise",
)

# Rotated failure modes for the rejected side of a preference pair.
FLAWS = (
    "a confident factual error",
    "it ignores part of the instruction",
    "the right idea in the wrong format",
    "vague and padded with filler",
    "it answers a subtly different question",
)


@dataclass
class Leaf:
    """One scenario to write training data for."""

    scenario: str
    difficulty: str = "medium"
    angle: str = ""
    grounding: str | None = None          # source passage, for document seeds
    exemplars: list[Trajectory] = field(default_factory=list)


@dataclass
class Outcome:
    """What one generation attempt produced, and what it cost in rejections.

    An outcome is the unit of acceptance: its rows are kept or dropped together.
    That matters for paraphrases, where MoRE+ groups rows by fixed size and
    losing one row of a unit would misalign every unit after it.
    """

    trajectories: list[Trajectory] = field(default_factory=list)
    key: str = ""        # what "already have this" means for these rows
    attempted: int = 1   # candidate rows asked for
    invalid: int = 0     # candidates still broken after the corrective retry
    repaired: int = 0    # candidates the retry rescued


# ---- scenarios ---------------------------------------------------------------
_TAXONOMY = """You are designing a training curriculum for a small model that must learn this task:

{task}

Produce a JSON array of exactly {n} training scenarios. Each element:
{{"scenario": "<one concrete situation, specific not generic>",
  "difficulty": "easy" | "medium" | "hard",
  "angle": "<what makes this case distinct: an edge case, an ambiguity, an adversarial user, ...>"}}

Cover the full breadth of the task. No two scenarios may share an angle.{avoid}
Reply with JSON only."""

_FACTS = """List the distinct factual claims stated in this passage.

PASSAGE:
\"\"\"
{chunk}
\"\"\"

Reply with a JSON array of strings — one self-contained claim per element, and
only claims the passage actually makes. JSON only."""

_PATTERNS = """Below are real conversations from an agent.

{examples}

TASK CONTEXT: {task}

Describe {n} DIFFERENT situations the same agent would plausibly face, in the
same domain and register. Reply with a JSON array of objects:
{{"scenario": "...", "difficulty": "easy" | "medium" | "hard", "angle": "..."}}{avoid}
JSON only."""


def plan_leaves(seed, teacher, *, count: int, rng, avoid=()) -> list[Leaf]:
    """Expand a seed into `count` distinct scenarios to generate against."""
    if seed.kind == "document":
        return _document_leaves(seed, teacher, count=count)
    if seed.kind == "episodes":
        return _episode_leaves(seed, teacher, count=count, rng=rng, avoid=avoid)
    return _batched_leaves(
        teacher, lambda n, block: _TAXONOMY.format(
            task=seed.context(), n=n, avoid=block),
        count=count, avoid=avoid)


def _batched_leaves(teacher, prompt_for, *, count: int, avoid) -> list[Leaf]:
    """Ask for scenarios in batches, steering each round clear of the last."""
    leaves: list[Leaf] = []
    seen = list(avoid)
    while len(leaves) < count:
        batch = min(_MAX_LEAVES_PER_CALL, count - len(leaves))
        fresh = _parse_leaves(teacher.chat(
            [{"role": "user", "content": prompt_for(batch, _avoid_block(seen))}],
            temperature=0.9, max_new_tokens=160 * batch + 300))
        if not fresh:
            break  # the teacher stopped producing usable JSON — report what we have
        for leaf in fresh[:batch]:
            leaves.append(leaf)
            seen.append(leaf.scenario)
    return leaves


def _document_leaves(seed, teacher, *, count: int) -> list[Leaf]:
    """One leaf per fact stated in the document, carrying its source passage."""
    leaves: list[Leaf] = []
    for chunk in seed.chunks:
        if len(leaves) >= count:
            break
        raw = teacher.chat([{"role": "user", "content": _FACTS.format(chunk=chunk)}],
                           temperature=0.2, max_new_tokens=1200)
        for fact in first_json_array(str(raw)) or []:
            if isinstance(fact, str) and fact.strip():
                leaves.append(Leaf(scenario=fact.strip(), grounding=chunk))
    return leaves[:count]


def _episode_leaves(seed, teacher, *, count: int, rng, avoid) -> list[Leaf]:
    examples = _render_exemplars(seed.exemplars[:3])
    leaves = _batched_leaves(
        teacher, lambda n, block: _PATTERNS.format(
            examples=examples, task=seed.context(), n=n, avoid=block),
        count=count, avoid=avoid)
    for leaf in leaves:
        leaf.exemplars = [rng.choice(seed.exemplars)]
    return leaves


def _parse_leaves(raw) -> list[Leaf]:
    leaves = []
    for item in first_json_array(str(raw)) or []:
        if isinstance(item, str) and item.strip():
            leaves.append(Leaf(scenario=item.strip()))
        elif isinstance(item, dict) and item.get("scenario"):
            leaves.append(Leaf(scenario=str(item["scenario"]),
                               difficulty=str(item.get("difficulty", "medium")),
                               angle=str(item.get("angle", ""))))
    return leaves


def _avoid_block(scenarios) -> str:
    if not scenarios:
        return ""
    recent = "\n".join(f"- {s[:120]}" for s in scenarios[-30:])
    return f"\n\nAlready covered — do not repeat or rephrase these:\n{recent}\n"


# ---- instances ---------------------------------------------------------------
_TOOL_RULES = """- A tool call is {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"...","arguments":"<a JSON string>"}}]}.
- Every tool call must be followed by {"role":"tool","tool_call_id":"call_1","content":"<a plausible result>"}.
- Call only the tools listed above."""

_CONVERSATION = """Write ONE realistic training conversation for this task.

TASK: {task}
SCENARIO: {scenario}
DIFFICULTY: {difficulty}{angle}
USER STYLE: {style}
{tools}{grounding}{exemplars}
Output strict JSON only:
{{"messages": [{{"role": "system" | "user" | "assistant" | "tool", "content": "..."}}]}}

Rules:
- The conversation MUST end with an assistant turn.
- Write specific, realistic content — never placeholders like [NAME] or [DATE].
- The assistant answers the way the task demands, not as a generic chatbot.
{rules}"""

_QUESTION = """TASK: {task}
SCENARIO: {scenario}
USER STYLE: {style}

Write the single user message this scenario would produce. The message only —
no preamble, no quotes."""

_ANSWER = """{task}
{grounding}
Answer this as well as you possibly can.

QUESTION: {question}

Reply with the answer only."""

_FLAWED = """{task}

Write a plausible but FLAWED answer to the question below. The flaw: {flaw}.
Never mention or label the flaw — it must read as a sincere attempt.

QUESTION: {question}

Reply with the answer only."""

_PARAPHRASES = """A user could ask about this fact in many different ways. Write {k}
questions that should all retrieve it, varying vocabulary, question form and
specificity — include one keyword-style query with no sentence structure.

FACT: {fact}
{grounding}
Then write the single canonical answer, stated plainly and completely.

Reply with JSON only:
{{"questions": ["...", "..."], "answer": "..."}}"""


def conversation(leaf: Leaf, seed, teacher, *, tools=None, style: str) -> Outcome:
    """A full multi-turn episode for this scenario."""
    prompt = _CONVERSATION.format(
        task=seed.context(), scenario=leaf.scenario, difficulty=leaf.difficulty,
        angle=f"\nANGLE: {leaf.angle}" if leaf.angle else "", style=style,
        tools=_tools_block(tools), grounding=_grounding_block(leaf),
        exemplars=_exemplar_block(leaf), rules=_TOOL_RULES if tools else "")
    outcome = Outcome()
    messages = _ask_messages(teacher, prompt, tools=tools, outcome=outcome)
    if messages:
        traj = _trajectory(messages, leaf, teacher, tools=tools, style=style)
        # instances of one scenario are meant to differ, so identity is the
        # question asked — not the scenario they came from
        outcome.key = traj.first_user_content()
        outcome.trajectories.append(traj)
    return outcome


def preference(leaf: Leaf, seed, teacher, *, student=None, style: str,
               flaw: str) -> Outcome:
    """A `chosen` / `rejected` pair for the same question.

    With a `student`, the rejected side is the student's own answer — so DPO
    targets exactly the teacher/student gap, which is the whole point of
    shadowing. Without one, the teacher writes a deliberately flawed answer.
    """
    outcome = Outcome()
    question = str(teacher.chat(
        [{"role": "user", "content": _QUESTION.format(
            task=seed.context(), scenario=leaf.scenario, style=style)}],
        temperature=0.9, max_new_tokens=300)).strip()
    if not question:
        outcome.invalid += 1
        return outcome
    chosen = _ask_answer(teacher, _ANSWER.format(
        task=seed.context(), grounding=_grounding_block(leaf), question=question),
        temperature=0.3)
    rejected = (
        _ask_answer(student, _ANSWER.format(
            task=seed.context(), grounding="", question=question), temperature=0.9)
        if student is not None else
        _ask_answer(teacher, _FLAWED.format(
            task=seed.context(), flaw=flaw, question=question), temperature=0.9))
    if not chosen or not rejected or _norm(chosen) == _norm(rejected):
        outcome.invalid += 1
        return outcome
    traj = _trajectory(
        [{"role": "user", "content": question},
         {"role": "assistant", "content": chosen}],
        leaf, teacher, tools=None, style=style)
    traj.metadata["rejected"] = rejected
    traj.metadata["rejected_from"] = "student" if student is not None else "flaw"
    outcome.key = question
    outcome.trajectories.append(traj)
    return outcome


def paraphrases(leaf: Leaf, seed, teacher, *, k: int, style: str) -> Outcome:
    """`k` differently-worded questions sharing one canonical answer.

    MoRE and MoRE+ index the *user* turn — it is the retrieval key, and for
    MoRE+ the BM25 routing surrogate. One fact phrased one way yields an expert
    nobody can route to, so phrasing diversity is the objective here, not a
    nicety.
    """
    outcome = Outcome()
    raw = teacher.chat([{"role": "user", "content": _PARAPHRASES.format(
        k=k, fact=leaf.scenario, grounding=_grounding_block(leaf))}],
        temperature=0.9, max_new_tokens=200 * k + 400)
    parsed = _first_json_object(str(raw)) or {}
    questions = [str(q).strip() for q in parsed.get("questions") or []
                 if str(q).strip()]
    answer = str(parsed.get("answer") or "").strip()
    picked = _diverse(questions, k) if answer else []
    if len(picked) < k:
        # a unit must be exactly k rows — see Outcome — so a fact that can't be
        # phrased k genuinely different ways is dropped rather than shipped short
        outcome.invalid += 1
        return outcome
    for question in picked:
        outcome.trajectories.append(_trajectory(
            [{"role": "user", "content": question},
             {"role": "assistant", "content": answer}],
            leaf, teacher, tools=None, style=style))
    outcome.key = leaf.scenario  # the fact is the identity; phrasings are its rows
    outcome.attempted = len(outcome.trajectories)
    return outcome


def _diverse(questions: list[str], k: int, *, threshold: float = 0.6) -> list[str]:
    """Drop phrasings that echo one already kept — a duplicate paraphrase is
    routing dead weight, not extra coverage."""
    picked: list[str] = []
    seen: list[set[str]] = []
    for question in questions:
        tokens = set(_tokenize(question))
        if any(jaccard(tokens, other) > threshold for other in seen):
            continue
        picked.append(question)
        seen.append(tokens)
        if len(picked) == k:
            break
    return picked


def _ask_messages(teacher, prompt: str, *, tools, outcome: Outcome) -> list[dict] | None:
    """Ask for a conversation, validate it, allow exactly one corrective retry.

    One retry, not a loop: a teacher that can't emit valid JSON twice won't on
    the fifth attempt, and the scenario is cheap to abandon.
    """
    for attempt in (0, 1):
        raw = str(teacher.chat([{"role": "user", "content": prompt}],
                               temperature=0.9, max_new_tokens=1600))
        parsed = _first_json_object(raw)
        messages = parsed.get("messages") if isinstance(parsed, dict) else None
        problems = validate(messages or [], tools=tools)
        if not problems:
            outcome.repaired += attempt
            return messages
        if attempt:
            break
        prompt = (f"{prompt}\n\nYour previous attempt was rejected: "
                  f"{'; '.join(problems)}.\nEmit corrected JSON only.")
    outcome.invalid += 1
    return None


def _ask_answer(teacher, prompt: str, *, temperature: float) -> str:
    return str(teacher.chat([{"role": "user", "content": prompt}],
                            temperature=temperature, max_new_tokens=800)).strip()


def _trajectory(messages, leaf: Leaf, teacher, *, tools, style: str) -> Trajectory:
    """Wrap generated messages with the provenance every synthetic row carries."""
    return Trajectory(messages=messages, tools=tools, metadata={
        "source": "synth", "teacher": teacher.name, "style": style,
        "taxonomy_path": leaf.scenario, "difficulty": leaf.difficulty,
        "grounding": leaf.grounding,
    })


def _tools_block(tools) -> str:
    if not tools:
        return ""
    return f"\nTOOLS THE ASSISTANT MAY CALL:\n{json.dumps(tools, indent=2)}\n"


def _grounding_block(leaf: Leaf) -> str:
    if not leaf.grounding:
        return ""
    return ("\nSOURCE MATERIAL — every claim in the answer must be supported by "
            f"this passage:\n\"\"\"\n{leaf.grounding}\n\"\"\"\n")


def _exemplar_block(leaf: Leaf) -> str:
    if not leaf.exemplars:
        return ""
    return ("\nREAL EXAMPLES from this agent — match their voice and format, but "
            f"do not reuse their content:\n{_render_exemplars(leaf.exemplars)}\n")


def _render_exemplars(trajectories) -> str:
    blocks = []
    for i, traj in enumerate(trajectories, 1):
        turns = "\n".join(
            f"{m.get('role')}: {m.get('content') or json.dumps(m.get('tool_calls', ''))}"
            for m in traj.messages)
        blocks.append(f"### Example {i}\n{turns}")
    return "\n\n".join(blocks)
