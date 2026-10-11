"""The ShadowLM reference server — `python -m shadowlm.serve`.

Serves the remote protocol (see `shadowlm.remote`) backed by the *real* local
backend: jobs train on this machine's mlx/torch, metrics stream to clients,
trained adapters download as tar.gz. Run it on a GPU box and point any SDK at
it with `backend="remote"` + SHADOWLM_API_URL — no mock, the same training the
local backend does. ShadowLM Studio implements this same protocol at scale.

    python -m shadowlm.serve --port 8329 --backend auto
    SHADOWLM_API_KEY=secret python -m shadowlm.serve   # require Bearer auth

Pure stdlib (http.server + threads). One training job at a time — honest about
being the reference implementation, not the fleet tier.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import html
import io
import json
import os
import queue
import re
import tarfile
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import __version__
from .data import Dataset
from .training import Metric, TrainConfig

DEFAULT_PORT = 8329

# Request bodies land in memory whole (datasets, adapter tarballs), so the
# Content-Length header is a promise to allocate — cap it. Generous: the
# biggest legitimate body is a worker shipping an adapter home.
_MAX_BODY = 512 << 20


class _PayloadTooLarge(Exception):
    """A request body over `_MAX_BODY` — answered with a 413, never read."""


def _write_private(path: Path, text: str) -> None:
    """write_text, but owner-only — for files that hold credentials."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.chmod(path, 0o600)  # a pre-existing file keeps its old mode otherwise


# Ids that may appear in a filesystem path (dataset ids, job ids): one plain
# segment, nothing the path layer could interpret.
_PLAIN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


class LoginThrottle:
    """After `limit` consecutive failures from one address, /v1/login answers
    429 for `window` seconds — enough to make online password guessing
    pointless without ever locking out the console user for long."""

    def __init__(self, limit: int = 5, window: float = 60.0,
                 clock=time.monotonic) -> None:
        self._limit, self._window, self._clock = limit, window, clock
        self._strikes: dict[str, tuple[int, float]] = {}  # addr → (fails, last)
        self._lock = threading.Lock()

    def allowed(self, addr: str) -> bool:
        with self._lock:
            fails, last = self._strikes.get(addr, (0, 0.0))
            if fails < self._limit:
                return True
            if self._clock() - last >= self._window:
                self._strikes.pop(addr, None)  # lock expired
                return True
            return False

    def record(self, addr: str, *, ok: bool) -> None:
        with self._lock:
            if ok:
                self._strikes.pop(addr, None)
            else:
                fails, _ = self._strikes.get(addr, (0, 0.0))
                self._strikes[addr] = (fails + 1, self._clock())

# Shown only when running from a source checkout where the React app hasn't been
# built. Every pip install ships the compiled UI in _static, so users never see
# this — it's a hint for contributors, not a fallback UI.
_NO_BUILD_PAGE = """<!doctype html><meta charset="utf-8">
<title>ShadowLM</title>
<body style="font:15px ui-monospace,monospace;background:#16120e;color:#f2eae0;
 padding:64px;line-height:1.7">
<h1 style="color:#e5484d">slm&#9829; ShadowLM</h1>
<p>The API is running — but the studio UI hasn't been built in this checkout.</p>
<pre style="background:#221c16;border:1px solid #3a3129;border-radius:10px;
 padding:16px">cd frontend &amp;&amp; npm install &amp;&amp; npm run build</pre>
<p>…then reload. Or <code>shadowlm serve --dev</code> for hot reload.
 (pip installs ship the built UI — you only see this from source.)</p>
</body>"""

# ---- embedding: a host console may show the studio in a frame ---------------
# A host console such as opencontroller shows the studio inside its own, in a
# frame, speaking the "oc-embed/1" bridge (frontend/src/lib/embed.ts). Two
# kinds of origin may frame it: the built-in ones (opencontroller on
# studio-dev, and localhost on any port for development), plus whatever
# SHADOWLM_FRAME_ANCESTORS lists (space- or comma-separated origins). Every
# response sends that list as frame-ancestors, and the studio page carries it
# in a meta tag so, once framed, it talks to nothing else.
_FRAME_ANCESTORS_ENV = "SHADOWLM_FRAME_ANCESTORS"
_BUILTIN_FRAME_ANCESTORS = (
    "https://dev.opencontroller.sh",
    "http://localhost:*",
    "https://localhost:*",
)


def _parse_origins(raw: str) -> list[str]:
    """Plain http(s) origins from a space/comma list; anything else (a path,
    a query, credentials, another scheme) is dropped, so framing is never
    opened wider than meant."""
    out: list[str] = []
    for f in re.split(r"[\s,]+", raw or ""):
        if not f:
            continue
        u = urllib.parse.urlsplit(f)
        if (u.scheme not in ("http", "https") or not u.netloc or "@" in u.netloc
                or u.path not in ("", "/") or u.query or u.fragment):
            continue
        origin = f"{u.scheme}://{u.netloc.lower()}"
        if origin not in out:
            out.append(origin)
    return out


def frame_ancestors() -> list[str]:
    """Origins that may frame the studio: built-ins, then the env's."""
    out = list(_BUILTIN_FRAME_ANCESTORS)
    for o in _parse_origins(os.environ.get(_FRAME_ANCESTORS_ENV, "")):
        if o not in out:
            out.append(o)
    return out


def _with_embed_parents(page: bytes, origins: list[str]) -> bytes:
    """The studio page with the framing origins in a meta tag, for embed.ts."""
    meta = ('<meta name="shadowlm-embed-parents" content="'
            + html.escape(" ".join(origins)) + '">').encode()
    return page.replace(b"</head>", meta + b"</head>", 1)


@dataclass
class _Job:
    job_id: str
    base_model: str
    name: str = ""  # human label for the shadow; falls back to the id in the UI
    status: str = "pending"  # pending | running | succeeded | failed | stopped
    steps: list[dict] = field(default_factory=list)
    evals: list[dict] = field(default_factory=list)
    error: str | None = None
    checkpoint: str | None = None
    final_loss: float | None = None
    method: str | None = None
    worker: str | None = None  # trained on this machine; inference routes there
    created: float = 0.0
    logs: list[str] = field(default_factory=list)  # captured console lines
    live: str = ""  # the in-progress (post-\r) line, e.g. the progress bar
    cancel: threading.Event = field(default_factory=threading.Event)

    _LOG_CAP = 600  # keep the tail; banner + progress + done fit comfortably

    def record(self) -> dict:
        """The serializable job record persisted to disk (survives restarts)."""
        return {"job_id": self.job_id, "base_model": self.base_model,
                "name": self.name, "worker": self.worker,
                "status": self.status, "method": self.method,
                "error": self.error, "checkpoint": self.checkpoint,
                "final_loss": self.final_loss, "created": self.created,
                "steps": self.steps, "evals": self.evals,
                "logs": self.logs[-self._LOG_CAP:]}

    @classmethod
    def from_record(cls, d: dict) -> "_Job":
        job = cls(job_id=d["job_id"], base_model=d.get("base_model", "?"),
                  name=d.get("name", ""), worker=d.get("worker"),
                  status=d.get("status", "succeeded"), method=d.get("method"),
                  error=d.get("error"), checkpoint=d.get("checkpoint"),
                  final_loss=d.get("final_loss"), created=d.get("created", 0.0))
        job.steps = d.get("steps", [])
        job.evals = d.get("evals", [])
        job.logs = d.get("logs", [])
        if job.worker is None:
            # records written before the worker field existed: the worker
            # announced itself in the captured console — recover it from there
            for ln in job.logs:
                if ln.startswith("[worker:") and "] picked up job" in ln:
                    job.worker = ln[len("[worker:"):ln.index("]")]
                    break
        # a run that was mid-flight when the server died can't still be running
        if job.status in ("pending", "running"):
            job.status, job.error = "stopped", "interrupted by a server restart"
        return job


class _LogTee:
    """Capture a job's console output line by line while still echoing to the
    real terminal. Carriage-return updates (progress bars) overwrite the live
    line instead of flooding the log."""

    def __init__(self, real, job: "_Job") -> None:
        self._real = real
        self._job = job

    def write(self, s: str) -> int:
        if self._real:
            self._real.write(s); self._real.flush()
        for ch in s:
            if ch == "\n":
                self._job.logs.append(self._job.live)
                self._job.live = ""
            elif ch == "\r":
                self._job.live = ""  # overwrite the current line
            else:
                self._job.live += ch
        if len(self._job.logs) > self._job._LOG_CAP * 2:
            self._job.logs = self._job.logs[-self._job._LOG_CAP:]
        return len(s)

    def flush(self) -> None:
        if self._real:
            self._real.flush()


def _rebuild_config(d: dict) -> TrainConfig:
    """TrainConfig from a client dict — full to_dict() or a partial config."""
    from . import methods  # noqa: PLC0415

    d = dict(d)
    if isinstance(d.get("target_modules"), list):
        d["target_modules"] = tuple(d["target_modules"])
    if isinstance(d.get("report_to"), list):
        d["report_to"] = tuple(d["report_to"])
    known = {f.name for f in __import__("dataclasses").fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in d.items() if k in known})
    if cfg.learning_rate is None:  # the Model layer does this for local runs;
        # partial configs (e.g. from the dashboard) need it resolved here too
        cfg.learning_rate = methods.get(cfg.method).default_learning_rate
    return cfg


def _rebuild_dataset(d: dict | None) -> Dataset | None:
    if not d:
        return None
    return Dataset.from_list(d["rows"], format=d.get("format"))


# A small curated catalog for the dashboard's Models page. `gated` = needs an
# HF token; `dev` = the fast local-loop pick.
_MODEL_CATALOG = [
    {"id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "params": "0.5B",
     "note": "fastest dev loop (mlx, 4-bit)", "gated": False, "dev": True},
    {"id": "Qwen/Qwen2.5-0.5B-Instruct", "params": "0.5B",
     "note": "small + capable", "gated": False, "dev": False},
    {"id": "Qwen/Qwen2.5-1.5B-Instruct", "params": "1.5B",
     "note": "quality jump, still light", "gated": False, "dev": False},
    {"id": "Qwen/Qwen2.5-3B-Instruct", "params": "3B",
     "note": "serious task model", "gated": False, "dev": False},
    {"id": "HuggingFaceTB/SmolLM2-360M-Instruct", "params": "360M",
     "note": "tiny experiments", "gated": False, "dev": False},
    {"id": "meta-llama/Llama-3.2-1B-Instruct", "params": "1B",
     "note": "llama family", "gated": True, "dev": False},
    {"id": "meta-llama/Llama-3.2-3B-Instruct", "params": "3B",
     "note": "llama family", "gated": True, "dev": False},
    {"id": "google/gemma-2-2b-it", "params": "2B",
     "note": "gemma family", "gated": True, "dev": False},
    {"id": "Qwen/Qwen3-0.6B", "params": "0.6B",
     "note": "Qwen3, tiny", "gated": False, "dev": False},
    {"id": "Qwen/Qwen3-4B-Instruct-2507", "params": "4B",
     "note": "Qwen3 instruct", "gated": False, "dev": False},
    {"id": "Qwen/Qwen3-8B", "params": "8B",
     "note": "Qwen3, CUDA examples", "gated": False, "dev": False},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "params": "7B",
     "note": "mistral family", "gated": False, "dev": False},
    {"id": "microsoft/Phi-3.5-mini-instruct", "params": "3.8B",
     "note": "phi family", "gated": False, "dev": False},
    {"id": "google/gemma-3-4b-it", "params": "4B",
     "note": "gemma 3", "gated": True, "dev": False},
    {"id": "openai/gpt-oss-20b", "params": "20B",
     "note": "gpt-oss (MoE)", "gated": False, "dev": False},
    # more Qwen3 sizes + Qwen2.5 specialists
    {"id": "Qwen/Qwen3-1.7B", "params": "1.7B", "note": "Qwen3, small", "gated": False, "dev": False},
    {"id": "Qwen/Qwen3-14B", "params": "14B", "note": "Qwen3, large", "gated": False, "dev": False},
    {"id": "Qwen/Qwen3-30B-A3B", "params": "30B", "note": "Qwen3 MoE (3B active)", "gated": False, "dev": False},
    {"id": "Qwen/Qwen2.5-7B-Instruct", "params": "7B", "note": "Qwen2.5 workhorse", "gated": False, "dev": False},
    {"id": "Qwen/Qwen2.5-Coder-7B-Instruct", "params": "7B", "note": "code-specialized", "gated": False, "dev": False},
    {"id": "Qwen/Qwen2.5-Math-7B-Instruct", "params": "7B", "note": "math-specialized", "gated": False, "dev": False},
    # Llama
    {"id": "meta-llama/Llama-3.1-8B-Instruct", "params": "8B", "note": "llama 3.1", "gated": True, "dev": False},
    # Gemma
    {"id": "google/gemma-3-1b-it", "params": "1B", "note": "gemma 3, tiny", "gated": True, "dev": False},
    {"id": "google/gemma-2-9b-it", "params": "9B", "note": "gemma 2", "gated": True, "dev": False},
    # Mistral
    {"id": "mistralai/Mistral-Nemo-Instruct-2407", "params": "12B", "note": "mistral nemo", "gated": False, "dev": False},
    {"id": "mistralai/Ministral-8B-Instruct-2410", "params": "8B", "note": "ministral", "gated": False, "dev": False},
    # Phi
    {"id": "microsoft/Phi-4", "params": "14B", "note": "phi-4", "gated": False, "dev": False},
    # SmolLM
    {"id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "params": "1.7B", "note": "smol, capable", "gated": False, "dev": False},
    {"id": "HuggingFaceTB/SmolLM2-135M-Instruct", "params": "135M", "note": "smallest", "gated": False, "dev": False},
    # DeepSeek R1 distills (reasoning)
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "params": "7B", "note": "R1 distill (reasoning)", "gated": False, "dev": False},
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "params": "8B", "note": "R1 distill (reasoning)", "gated": False, "dev": False},
    # other open families
    {"id": "tiiuae/Falcon3-7B-Instruct", "params": "7B", "note": "falcon 3", "gated": False, "dev": False},
    {"id": "ibm-granite/granite-3.1-8b-instruct", "params": "8B", "note": "granite", "gated": False, "dev": False},
    {"id": "allenai/OLMo-2-1124-7B-Instruct", "params": "7B", "note": "fully-open OLMo", "gated": False, "dev": False},
    {"id": "HuggingFaceH4/zephyr-7b-beta", "params": "7B", "note": "zephyr", "gated": False, "dev": False},
]


# Sample JSONLs shipped in the wheel (shadowlm/_samples) — a fresh studio seeds
# these so the Datasets page isn't empty on first spin.
# Synthesis history is bounded on both axes: a run keeps its most recent log
# lines, and the store keeps its most recent runs. Neither grows without end.
_SYNTH_LOG_LINES = 200
_SYNTH_KEEP = 100

_SAMPLE_DIR = Path(__file__).resolve().parent / "_samples"
_SAMPLE_NAMES = {
    "chat": "ShadowLM Q&A · chat",
    "preference": "ShadowLM preferences · DPO",
    "domain": "ShadowLM domain text · CPT",
    "facts": "ShadowLM facts · MoRE",
    "shadowlm_qa": "ShadowLM about-itself · chat",
}

# Curated popular open datasets, listed alongside the samples so users have real
# data to pick from day one. Stored as HF references (resolved at train time):
# (repo, subset, split, format). Canonical upstreams only.
_CURATED_HF = [
    ("yahma/alpaca-cleaned", None, "train", "instruction"),
    ("mlabonne/FineTome-100k", None, "train", "sharegpt"),
    ("openai/gsm8k", "main", "train", "instruction"),
    ("HuggingFaceH4/ultrafeedback_binarized", None, "train_prefs", "preference"),
    ("HuggingFaceH4/no_robots", None, "train", "chat"),
    ("databricks/databricks-dolly-15k", None, "train", "instruction"),
    ("tatsu-lab/alpaca", None, "train", "instruction"),
    ("teknium/OpenHermes-2.5", None, "train", "sharegpt"),
    ("Open-Orca/OpenOrca", None, "train", "instruction"),
    ("garage-bAInd/Open-Platypus", None, "train", "instruction"),
    ("microsoft/orca-math-word-problems-200k", None, "train", "instruction"),
    ("allenai/tulu-3-sft-mixture", None, "train", "chat"),
    ("roneneldan/TinyStories", None, "train", "text"),
    ("Magpie-Align/Magpie-Air-300K-Filtered", None, "train", "chat"),
    ("HuggingFaceH4/Multilingual-Thinking", None, "train", "chat"),
    ("openbmb/UltraInteract_sft", None, "train", "instruction"),
    ("vicgalle/alpaca-gpt4", None, "train", "instruction"),
]


class DatasetStore:
    """Datasets in the dashboard — uploaded JSONL, or a HuggingFace reference.
    Both persist as a small JSON meta; survives restarts."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        if not any(self.root.glob("*.json")):
            self._seed()  # fresh studio → show the bundled samples + a starter catalog
        self._migrate_curated()  # backfill the Explore flag on pre-existing seeds

    def _migrate_curated(self) -> None:
        repos = {r[0] for r in _CURATED_HF}
        for p in self.root.glob("*.json"):
            try:
                m = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if m.get("source") == "hf" and m.get("repo") in repos and not m.get("curated"):
                m["curated"] = True
                p.write_text(json.dumps(m))

    def _seed(self) -> None:
        for p in sorted(_SAMPLE_DIR.glob("*.jsonl")):
            try:
                rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
                if rows:
                    self.save(_SAMPLE_NAMES.get(p.stem, p.stem), rows)
            except (OSError, ValueError):
                continue
        for repo, subset, split, fmt in _CURATED_HF:
            try:
                self.save_hf(repo, subset=subset, split=split, fmt=fmt, rows=None,
                             curated=True)
            except (OSError, ValueError):
                continue

    def save(self, name: str, rows: list[dict]) -> dict:
        ds = Dataset.from_list(rows)  # validates + detects format
        ds_id = uuid.uuid4().hex[:10]
        (self.root / f"{ds_id}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows))
        meta = {"dataset_id": ds_id, "name": name or f"dataset-{ds_id[:4]}",
                "source": "upload", "format": ds.format, "rows": len(rows),
                "created": int(time.time())}
        (self.root / f"{ds_id}.json").write_text(json.dumps(meta))
        return meta

    def save_hf(self, repo: str, *, subset: str | None, split: str,
                fmt: str, rows: int | None, eval_split: str | None = None,
                curated: bool = False) -> dict:
        """Register a HuggingFace dataset by reference (resolved at train time).
        ``curated`` marks the bundled starter catalog (shown under Explore)."""
        ds_id = uuid.uuid4().hex[:10]
        meta = {"dataset_id": ds_id, "name": repo, "source": "hf",
                "repo": repo, "subset": subset or "default", "split": split,
                "eval_split": eval_split or None, "curated": curated,
                "format": fmt, "rows": rows, "created": int(time.time())}
        (self.root / f"{ds_id}.json").write_text(json.dumps(meta))
        return meta

    def list(self) -> list[dict]:
        metas = []
        for p in sorted(self.root.glob("*.json")):
            try:
                metas.append(json.loads(p.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        return sorted(metas, key=lambda m: m.get("created", 0), reverse=True)

    def rows(self, ds_id: str) -> list[dict] | None:
        if not _PLAIN_ID.fullmatch(ds_id or ""):
            return None  # ids come off the URL — never let one shape a path
        p = self.root / f"{ds_id}.jsonl"
        if not p.exists():
            return None
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def meta(self, ds_id: str) -> dict | None:
        if not _PLAIN_ID.fullmatch(ds_id or ""):
            return None
        p = self.root / f"{ds_id}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def resolve(self, ds_id: str) -> Dataset:
        """A trainable Dataset — local rows, or pulled from HF on demand."""
        meta = self.meta(ds_id)
        if meta is None:
            raise ValueError(f"unknown dataset {ds_id!r}")
        if meta.get("source") == "hf":
            sub = meta["subset"] if meta["subset"] != "default" else None
            return Dataset.from_hf(meta["repo"], subset=sub, split=meta["split"])
        rows = self.rows(ds_id) or []
        return Dataset.from_list(rows)

    def resolve_eval(self, ds_id: str) -> Dataset | None:
        """The dataset's own eval split, if it declared one (HF only)."""
        meta = self.meta(ds_id)
        if not meta or meta.get("source") != "hf" or not meta.get("eval_split"):
            return None
        sub = meta["subset"] if meta["subset"] != "default" else None
        return Dataset.from_hf(meta["repo"], subset=sub, split=meta["eval_split"])

    def delete(self, ds_id: str) -> bool:
        if not _PLAIN_ID.fullmatch(ds_id or ""):
            return False
        found = False
        for suffix in (".json", ".jsonl"):
            p = self.root / f"{ds_id}{suffix}"
            if p.exists():
                p.unlink()
                found = True
        return found


# ---- projects: one fine-tune, as Business mode walks it ----------------------
# A project ties together the fine-tune a business user is making for one job:
# its data (a dataset), the training run, the evaluation that proves it. Its
# stage is read off those links by the UI; the store only keeps them. Goals:
# "knowledge" (teach it facts), "task" (teach it a job from examples),
# "takeover" (move a task off a frontier model, from captured agent traffic).
_PROJECT_GOALS = ("knowledge", "task", "takeover")
_PROJECT_LINKS = ("dataset_id", "run_id", "eval_id", "synth_id")  # synth_id: examples being written


class ProjectStore:
    """Fine-tune projects in one JSON file under the server's work dir."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        try:
            self._items: dict[str, dict] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            self._items = {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._items, indent=1))

    def list(self) -> list[dict]:
        with self._lock:
            return sorted(self._items.values(), key=lambda d: -d["updated"])

    def get(self, sid: str) -> dict | None:
        return self._items.get(sid)

    def create(self, name: str, goal: str) -> dict:
        name = (name or "").strip()
        if not name or len(name) > 80:
            raise ValueError("a project needs a name of 1-80 characters")
        if goal not in _PROJECT_GOALS:
            raise ValueError(f"goal must be one of {', '.join(_PROJECT_GOALS)}")
        now = int(time.time())
        item = {"project_id": uuid.uuid4().hex[:12], "name": name, "goal": goal,
                "dataset_id": None, "run_id": None, "eval_id": None, "synth_id": None, "history": [],
                "created": now, "updated": now}
        with self._lock:
            self._items[item["project_id"]] = item
            self._save()
        return item

    def update(self, sid: str, patch: dict) -> dict | None:
        with self._lock:
            item = self._items.get(sid)
            if item is None:
                return None
            if "name" in patch:
                name = (patch["name"] or "").strip()
                if not name or len(name) > 80:
                    raise ValueError("a project needs a name of 1-80 characters")
                item["name"] = name
            for k in _PROJECT_LINKS:
                if k in patch:
                    v = patch[k]
                    if v is not None and not _PLAIN_ID.fullmatch(str(v)):
                        raise ValueError(f"field {k!r} must be a plain id")
            # a new fine-tune makes a new version: the one it replaces, with its
            # evaluation, moves to history so versions can be compared
            new_run = patch.get("run_id", item.get("run_id"))
            if "run_id" in patch and item.get("run_id") and new_run != item["run_id"]:
                item.setdefault("history", []).append(
                    {"run_id": item["run_id"], "eval_id": item.get("eval_id"),
                     "dataset_id": item.get("run_dataset_id", item.get("dataset_id")),
                     "replaced": int(time.time())})
                item["eval_id"] = None
            for k in _PROJECT_LINKS:
                if k in patch:
                    item[k] = patch[k]
            # what the current version trained on: the data can change under it
            # (more examples written) and the cockpit offers a new version then
            if "run_id" in patch:
                item["run_dataset_id"] = item.get("dataset_id")
            item.setdefault("history", [])
            item["updated"] = int(time.time())
            self._save()
            return dict(item)

    def delete(self, sid: str) -> bool:
        with self._lock:
            if self._items.pop(sid, None) is None:
                return False
            self._save()
            return True


# ---- captures: an agent's traffic to its frontier model, recorded ------------
# A capture session is an OpenAI-compatible URL the user points their agent at
# (base URL /v1/capture/<id>): every call passes through to the frontier model
# configured in settings and is recorded, then reconstructed into episodes
# (capture.reconstruct) and saved as a dataset to fine-tune on. The agent is
# not modified. The id is the URL's only secret, so it is long and random, and
# a closed session stops forwarding.
_CAPTURE_MAX_CALLS = 5000


class CaptureStore:
    """Capture sessions, one JSON file each under <work>/captures/."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}
        for f in sorted(root.glob("*.json")):
            try:
                d = json.loads(f.read_text())
                self._items[d["capture_id"]] = d
            except (OSError, ValueError, KeyError):
                continue

    def _save(self, item: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _write_private(self.root / f"{item['capture_id']}.json", json.dumps(item))

    def create(self, name: str, project_id: str | None = None) -> dict:
        name = (name or "").strip()[:80] or "Agent capture"
        if project_id is not None and not _PLAIN_ID.fullmatch(str(project_id)):
            raise ValueError("project_id must be a plain id")
        item = {"capture_id": uuid.uuid4().hex + uuid.uuid4().hex[:8], "name": name,
                "project_id": project_id, "status": "open",
                "created": int(time.time()), "calls": []}
        with self._lock:
            self._items[item["capture_id"]] = item
            self._save(item)
        return item

    def get(self, cid: str) -> dict | None:
        return self._items.get(cid)

    def record(self, cid: str, call: dict) -> None:
        with self._lock:
            item = self._items[cid]
            item["calls"].append(call)
            if len(item["calls"]) >= _CAPTURE_MAX_CALLS:
                item["status"] = "closed"  # a cap, so a leaked URL can't run forever
            self._save(item)

    def close(self, cid: str) -> dict | None:
        with self._lock:
            item = self._items.get(cid)
            if item is None:
                return None
            item["status"] = "closed"
            self._save(item)
            return item

    def delete(self, cid: str) -> bool:
        with self._lock:
            if self._items.pop(cid, None) is None:
                return False
            (self.root / f"{cid}.json").unlink(missing_ok=True)
            return True

    def list(self) -> list[dict]:
        with self._lock:
            items = sorted(self._items.values(), key=lambda d: -d["created"])
        return [{k: v for k, v in d.items() if k != "calls"} | {"calls": len(d["calls"])}
                for d in items]


def _capture_trajectories(item: dict):
    from .capture import _Call, reconstruct  # noqa: PLC0415

    return reconstruct([_Call(session=c.get("session") or "", messages=c["messages"],
                              response=c["response"], tools=c.get("tools"), ts=c.get("ts", 0.0))
                        for c in item["calls"]])


# ---- deployments: a fine-tune behind an OpenAI-compatible endpoint -----------
# A deployment serves one fine-tune (a run, optionally at a checkpoint) at
# /openai/v1/chat/completions under its own name and key, so an agent switches
# to it by changing its base URL, key and model name. The key is shown once and
# stored hashed; deleting the deployment revokes it.
class DeploymentStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        try:
            self._items: dict[str, dict] = json.loads(path.read_text())
        except (OSError, ValueError):
            self._items = {}
        self._dirty_at = 0.0

    def _save(self) -> None:
        _write_private(self.path, json.dumps(self._items, indent=1))

    def list(self) -> list[dict]:
        with self._lock:
            return [{k: v for k, v in d.items() if k != "key_hash"}
                    for d in sorted(self._items.values(), key=lambda d: -d["created"])]

    def create(self, *, name: str, run_id: str, base_model: str,
               checkpoint: int | None, project_id: str | None) -> tuple[dict, str]:
        name = (name or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
            raise ValueError("a deployment name is 1-64 letters, digits, '.', '_' or '-', "
                             "starting with a letter or digit: it becomes the API's model name")
        with self._lock:
            if any(d["name"] == name for d in self._items.values()):
                raise ValueError(f"a deployment named {name!r} already exists")
            key = "slmd_" + uuid.uuid4().hex + uuid.uuid4().hex[:8]
            item = {"deployment_id": uuid.uuid4().hex[:12], "name": name, "run_id": run_id,
                    "base_model": base_model, "checkpoint": checkpoint,
                    "project_id": project_id, "key_prefix": key[:9],
                    "key_hash": hashlib.sha256(key.encode()).hexdigest(),
                    "created": int(time.time()), "requests": 0, "last_used": None}
            self._items[item["deployment_id"]] = item
            self._save()
        return {k: v for k, v in item.items() if k != "key_hash"}, key

    def by_key(self, key: str) -> dict | None:
        h = hashlib.sha256(key.encode()).hexdigest()
        with self._lock:
            return next((d for d in self._items.values()
                         if hmac.compare_digest(d["key_hash"], h)), None)

    def used(self, dep_id: str) -> None:
        with self._lock:
            d = self._items.get(dep_id)
            if d is None:
                return
            d["requests"] += 1
            d["last_used"] = int(time.time())
            if time.time() - self._dirty_at > 5:  # don't rewrite the file per request
                self._save()
                self._dirty_at = time.time()

    def delete(self, dep_id: str) -> bool:
        with self._lock:
            if self._items.pop(dep_id, None) is None:
                return False
            self._save()
            return True


# ---- evaluations: task quality, as a background job --------------------------
# An evaluation scores one or more targets (a base model, or a shadow at a
# checkpoint) on the same questions, so the studio can show shadow vs base side
# by side. It runs in the background — a held HTTP request would hit the
# 100-second proxy limit — and persists under <work>/evals/<id>.json.
_EVAL_METRICS = ("contains", "exact", "judge")  # judge: the frontier model scores
_EVAL_MAX_TARGETS = 4


@dataclass
class _Eval:
    eval_id: str
    name: str
    dataset_id: str
    metric: str
    targets: list  # [{label, model, adapter, checkpoint}]
    sample: int | None
    status: str = "pending"  # pending | running | succeeded | failed
    error: str | None = None
    results: list = field(default_factory=list)  # per target: {score, n, examples}
    created: int = 0
    finished: int | None = None

    def record(self) -> dict:
        return {k: getattr(self, k) for k in (
            "eval_id", "name", "dataset_id", "metric", "targets", "sample",
            "status", "error", "results", "created", "finished")}

    def summary(self) -> dict:
        """The record without per-example detail, for lists."""
        d = self.record()
        d["results"] = [{k: r.get(k) for k in ("score", "n")} for r in self.results]
        return d


@dataclass
class _ChatTask:
    task_id: str
    status: str = "running"  # running | succeeded | failed
    text: str | None = None
    error: str | None = None
    created: float = 0.0


@dataclass
class _Worker:
    """A machine that dialed in with `shadowlm worker` and takes jobs from us.

    Workers are keyed by name and live in memory only — a worker that never
    polls again is just stale (see `online`), and jobs assigned to it wait in
    its inbox until it comes back. ponytail: no offline eviction/reassignment —
    add when a worker dying mid-queue actually strands someone.
    """

    name: str
    backend: str = "?"
    device: str = "?"
    gpus: int = 0
    gpu_name: str = ""
    vram_gb: float = 0.0
    ram_gb: float = 0.0
    cores: int = 0
    models: list = field(default_factory=list)  # [{id, size_gb}] on that machine
    last_seen: float = 0.0
    inbox: "queue.Queue[str]" = field(default_factory=queue.Queue)

    def info(self) -> dict:
        return {"name": self.name, "backend": self.backend, "device": self.device,
                "gpus": self.gpus, "gpu_name": self.gpu_name,
                "vram_gb": self.vram_gb, "ram_gb": self.ram_gb,
                "cores": self.cores, "models": self.models,
                "last_seen": int(self.last_seen),
                "online": (time.time() - self.last_seen) < 90,
                "queued": self.inbox.qsize()}


def _gpu_count() -> int:
    """How many CUDA GPUs this box has (0 on mlx/CPU). Advertised in /v1/health.

    One job already spans all of them — the torch backend loads with
    device_map="auto" — so this is a capacity signal for clients choosing between
    servers, not a knob. Running one job *per* GPU would need cuda:N placement,
    which device_map deliberately does not do.
    """
    try:
        import torch  # noqa: PLC0415  (optional: mlx boxes have no torch)

        return torch.cuda.device_count()
    except Exception:  # noqa: BLE001 — no torch, no driver, no CUDA: all "0 GPUs"
        return 0


class Server:
    """Job store + the single training worker + an inference slot."""

    def __init__(self, *, backend: str, accelerator: str, device: str,
                 work_root: Path) -> None:
        self.backend_name = backend
        self.accelerator = accelerator
        self.device = device
        self.work_root = work_root
        self.jobs: dict[str, _Job] = {}
        self.workers: dict[str, _Worker] = {}  # remote executors, keyed by name
        self.worker_socks: dict[str, object] = {}  # name → live WSConn
        self._infer_waiters: dict[str, "queue.Queue[dict]"] = {}  # req id → reply
        self._prewarming: set = set()  # (model, adapter) loads in flight
        self._prewarm_errors: dict = {}  # key → (when, message) of a failed load
        self.datasets = DatasetStore(work_root / "datasets")
        self.projects = ProjectStore(work_root / "projects.json")
        # Ctrl agent answers the cockpit's conversation with Claude when the
        # server has ANTHROPIC_API_KEY; without it the studio keeps its rules
        from .ctrl import CtrlAgent  # noqa: PLC0415
        self.ctrl = CtrlAgent.from_env()
        self.evals: dict[str, _Eval] = {}
        self._evals_root = work_root / "evals"
        self._load_evals()
        self.chat_tasks: dict[str, _ChatTask] = {}
        self.captures = CaptureStore(work_root / "captures")
        self.deployments = DeploymentStore(work_root / "deployments.json")
        self.queue: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()          # job-store mutations
        self._model_lock = threading.Lock()    # one model computation at a time
        self._infer_cache: dict[tuple, object] = {}  # (model, adapter) → Model
        self._infer_cache_cap = 3  # compare mode alternates base ↔ adapter
        self._downloads: dict[str, dict] = {}  # model id → prefetch status
        self._synth: dict[str, dict] = {}      # synth id → status + log lines
        self._synth_cancel: dict[str, threading.Event] = {}  # live runs only
        self._settings_path = work_root / "settings.json"
        self._custom_path = work_root / "custom_models.json"
        self._tokens_path = work_root / "tokens.json"
        self._custom_models = self._load_custom_models()  # user-added HF repos
        self._machine_tokens = self._load_tokens()  # name → {hash, created}
        self._load_settings()
        self._load_jobs()
        self._load_synth()
        threading.Thread(target=self._worker, daemon=True).start()

    def capacity(self) -> dict:
        """What this box is and how busy it is — clients route on this."""
        with self._lock:
            jobs = list(self.jobs.values())
        with self._lock:
            workers = [w.info() for w in self.workers.values()]
        return {
            "gpus": _gpu_count(),
            "running": sum(j.status == "running" for j in jobs),
            "pending": sum(j.status == "pending" for j in jobs),
            "workers": sum(w["online"] for w in workers),
        }

    # ---- settings: HF token for gated/private models --------------------------
    def _load_settings(self) -> None:
        from . import hub  # noqa: PLC0415

        try:
            self._settings: dict = json.loads(self._settings_path.read_text())
        except (OSError, ValueError):
            self._settings = {}
        # a persisted token wins; otherwise honor one already in the environment
        if self._settings.get("hf_token") or os.environ.get("HF_TOKEN"):
            hub.set_token(self._settings.get("hf_token") or os.environ.get("HF_TOKEN"))

    def _save_settings(self) -> None:
        try:
            _write_private(self._settings_path, json.dumps(self._settings))
        except OSError:
            pass  # in-memory still works for this process

    def set_hf_token(self, token: str | None) -> None:
        from . import hub  # noqa: PLC0415

        hub.set_token(token)
        self._settings["hf_token"] = token or ""
        self._save_settings()

    # ---- frontier model: the user's OpenAI-compatible model, by its key -------
    # Used as an evaluation baseline, as the judge, and as the upstream an
    # agent's captured traffic passes through. The key is stored owner-only
    # and never sent back to a client.
    def set_frontier(self, base_url: str | None, api_key: str | None, model: str | None) -> None:
        from .frontier import FrontierModel  # noqa: PLC0415

        if not base_url:
            self._settings.pop("frontier", None)
        else:
            if not api_key or not model:
                raise ValueError("a frontier model needs a base URL, an API key and a model name")
            FrontierModel(base_url, api_key, model)  # validates the URL
            self._settings["frontier"] = {"base_url": base_url.rstrip("/"),
                                          "api_key": api_key, "model": model}
        self._save_settings()

    def frontier(self):
        """The configured frontier model, or None."""
        from .frontier import FrontierModel  # noqa: PLC0415

        f = self._settings.get("frontier")
        return FrontierModel(f["base_url"], f["api_key"], f["model"]) if f else None

    def ctrl_reply(self, project_id: str, body: dict) -> dict:
        """Answer one message in a project's cockpit conversation.

        The browser sends the state it shows (stations, versions, scores) and
        the thread; the server adds what only it has: the current run's last
        log lines when it failed, and a few of the project's examples.
        """
        from .ctrl import CtrlError  # noqa: PLC0415

        if self.ctrl is None:
            raise LookupError("Ctrl agent has no model: set ANTHROPIC_API_KEY on the server")
        project = self.projects.get(project_id)
        if project is None:
            raise KeyError(project_id)
        message = str(body.get("message") or "").strip()[:4000]
        if not message:
            raise ValueError("say something to Ctrl agent")
        state = {"project": {k: project.get(k) for k in ("name", "goal", "dataset_id", "run_id", "eval_id")},
                 "cockpit": body.get("state") or {}}
        job = self.jobs.get(project.get("run_id") or "")
        if job is not None and job.status in ("failed", "stopped"):
            state["run_log_tail"] = job.logs[-40:]
        if project.get("dataset_id"):
            try:
                ds = self.datasets.resolve(project["dataset_id"])
                state["example_rows"] = {"total": len(ds.rows),
                                         "first": json.dumps(ds.rows[:8], default=str)[:6000]}
            except Exception:  # noqa: BLE001 — examples are context, not required
                pass
        thread = [t for t in (body.get("thread") or []) if isinstance(t, dict)]
        try:
            return self.ctrl.reply(state, thread, message)
        except CtrlError as e:
            raise RuntimeError(str(e)) from None

    def frontier_info(self) -> dict | None:
        f = self._settings.get("frontier")
        return {"base_url": f["base_url"], "model": f["model"]} if f else None

    # ---- machine tokens: long-lived credentials for `shadowlm worker` --------
    # Named + individually revocable, hashed at rest in tokens.json — the same
    # single-box persistence every other piece of hub state uses.
    # ponytail: a JSON file, not a database — swap the store when a hub outgrows
    # one box, which is also when this whole tier hands off to Studio.
    def _load_tokens(self) -> dict:
        try:
            data = json.loads(self._tokens_path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_tokens(self) -> None:
        try:
            _write_private(self._tokens_path, json.dumps(self._machine_tokens, indent=1))
        except OSError:
            pass  # in-memory still works for this process

    def mint_machine_token(self, name: str) -> str:
        """A long-lived worker credential; the raw value is shown exactly once."""
        import secrets  # noqa: PLC0415

        raw = "slmk_" + secrets.token_urlsafe(32)
        with self._lock:
            self._machine_tokens[name] = {
                "hash": hashlib.sha256(raw.encode()).hexdigest(),
                "created": int(time.time())}
            self._save_tokens()
        return raw

    def revoke_machine_token(self, name: str) -> bool:
        with self._lock:
            if name not in self._machine_tokens:
                return False
            del self._machine_tokens[name]
            self._save_tokens()
        return True

    def valid_machine_token(self, raw: str) -> bool:
        if not raw.startswith("slmk_"):
            return False
        digest = hashlib.sha256(raw.encode()).hexdigest()
        return any(hmac.compare_digest(digest, t.get("hash", ""))
                   for t in self._machine_tokens.values())

    def machine_tokens(self) -> list[dict]:
        with self._lock:
            return [{"name": n, "created": t.get("created", 0)}
                    for n, t in sorted(self._machine_tokens.items())]

    # ---- custom models: user-added HF repos beyond the curated catalog -------
    def _load_custom_models(self) -> list:
        try:
            return json.loads(self._custom_path.read_text())
        except (OSError, ValueError):
            return []

    def _save_custom_models(self) -> None:
        try:
            self._custom_path.write_text(json.dumps(self._custom_models))
        except OSError:
            pass  # in-memory still works for this process

    def catalog(self) -> list:
        """Curated catalog + the user's added repos (added ones first)."""
        return [*self._custom_models, *_MODEL_CATALOG]

    def add_custom_model(self, model: str) -> list:
        model = (model or "").strip()
        with self._lock:
            known = {m["id"] for m in _MODEL_CATALOG} | {m["id"] for m in self._custom_models}
            if model and model not in known:
                self._custom_models.insert(0, {"id": model, "custom": True})
                self._save_custom_models()
            return list(self._custom_models)

    def remove_custom_model(self, model: str) -> list:
        with self._lock:
            self._custom_models = [m for m in self._custom_models if m["id"] != model]
            self._save_custom_models()
            return list(self._custom_models)

    # ---- model downloads: prefetch weights, report progress ------------------
    def start_download(self, model: str) -> dict:
        from . import hub  # noqa: PLC0415

        with self._lock:
            cur = self._downloads.get(model)
            if cur and cur.get("state") == "downloading":
                return cur
            if hub.is_cached(model):
                self._downloads[model] = {"state": "ready", "total": hub.cached_bytes(model)}
                return self._downloads[model]
            self._downloads[model] = {"state": "downloading", "total": 0, "error": None}

        def run() -> None:
            token = os.environ.get("HF_TOKEN")
            try:
                total = hub.repo_bytes(model, token)
                with self._lock:
                    self._downloads[model]["total"] = total
                hub.download(model, token)
                with self._lock:
                    self._downloads[model] = {"state": "ready",
                                              "total": total or hub.cached_bytes(model)}
            except Exception as e:  # noqa: BLE001 — surfaced to the UI
                with self._lock:
                    self._downloads[model] = {"state": "error",
                                              "error": f"{type(e).__name__}: {e}"}

        threading.Thread(target=run, daemon=True).start()
        return self._downloads[model]

    def download_status(self) -> dict:
        from . import hub  # noqa: PLC0415

        with self._lock:
            items = list(self._downloads.items())
        out = {}
        for model, st in items:
            d = dict(st)
            if d.get("state") == "downloading":  # live progress from disk
                d["downloaded"] = hub.cached_bytes(model)
                total = d.get("total") or 0
                d["pct"] = round(100 * d["downloaded"] / total, 1) if total else None
            out[model] = d
        return out

    # ---- synthesis: generate a dataset instead of uploading one --------------
    def start_synth(self, body: dict) -> dict:
        """Kick off a synthesis run in the background; poll `synth_status`.

        Deliberately not on the training queue — synthesis is teacher calls and
        would otherwise block the one training slot for the whole run.
        """
        from .synth import frontier, synthesize  # noqa: PLC0415

        spec = body.get("teacher") or {}
        synth_id = uuid.uuid4().hex[:10]
        name = body.get("name") or f"synth-{synth_id[:4]}"
        requested = int(body.get("n") or 100)
        budget = int(body.get("token_budget") or 0) or None
        # a project the examples are for: it shows the run while it writes, and
        # its data becomes the result when the run lands
        project_id = body.get("project_id")
        if project_id and not self.projects.get(project_id):
            raise ValueError(f"no project {project_id!r}")
        cancel = threading.Event()
        with self._lock:
            self._synth[synth_id] = {
                "synth_id": synth_id, "name": name, "status": "running",
                "kept": 0, "requested": requested, "phase": "starting",
                "done": 0, "total": 0, "tokens": 0, "logs": [],
                "created": int(time.time()),
            }
            self._synth_cancel[synth_id] = cancel
        self._persist_synth(synth_id)
        if project_id:
            self.projects.update(project_id, {"synth_id": synth_id})

        def update(**fields) -> None:
            with self._lock:
                self._synth[synth_id].update(**fields)

        def run() -> None:
            def progress(done: int, total: int, phase: str) -> None:
                with self._lock:
                    entry = self._synth[synth_id]
                    entry["phase"], entry["done"], entry["total"] = phase, done, total
                    # Only a completed round is worth a log line; ticking once
                    # per finished job would bury the report under hundreds.
                    if phase == "kept":
                        entry["kept"] = done
                        entry["logs"].append(f"[synth] {done}/{total} rows kept")
                        del entry["logs"][:-_SYNTH_LOG_LINES]

            try:
                # teacher and episodes resolve in here: loading a local model
                # (or pulling an HF dataset) can take minutes, and the POST
                # must return the synth_id immediately, not after the download
                if spec.get("kind") == "local":
                    from .models import load  # noqa: PLC0415
                    teacher = load(spec["model"], backend=self.backend_name)
                elif spec.get("kind") == "frontier":
                    # the frontier model saved in settings: its stored key, so
                    # the person never pastes it twice
                    f = self._settings.get("frontier")
                    if not f:
                        raise ValueError("no frontier model is set up; add one in settings")
                    teacher = frontier(f["model"], base_url=f["base_url"], api_key=f["api_key"])
                else:
                    # the key is used for this run and never written to disk
                    teacher = frontier(spec.get("model") or "gpt-4o",
                                       base_url=spec.get("base_url") or None,
                                       api_key=spec.get("api_key") or None)
                episodes = (self.datasets.resolve(body["dataset_id"])
                            if body.get("dataset_id") else None)
                result = synthesize(
                    teacher=teacher, task=body.get("task") or None,
                    document=body.get("document") or None, episodes=episodes,
                    n=requested, method=body.get("method") or None,
                    # 0 from the UI means "no gate", same as the CLI
                    min_score=body.get("min_score", 0.6) or None, verbose=False,
                    token_budget=budget, should_stop=cancel.is_set,
                    on_progress=progress)
                rows = result.rows()
                # include_seed: the dataset carries the examples it was written
                # from too, so the next fine-tune trains on old and new together
                if body.get("include_seed") and episodes is not None:
                    rows = list(episodes.rows) + list(rows)
                meta = self.datasets.save(name, rows)
                with self._lock:
                    entry = self._synth[synth_id]
                    # A cancelled run still saved its rows — report it as
                    # stopped, not succeeded, so the number means something.
                    entry.update(status="stopped" if cancel.is_set() else "succeeded",
                                 kept=result.report.kept,
                                 tokens=result.report.tokens,
                                 dataset_id=meta["dataset_id"])
                    entry["logs"].append(result.report.summary())
                    del entry["logs"][:-_SYNTH_LOG_LINES]
                if project_id and result.report.kept:
                    self.projects.update(project_id, {"dataset_id": meta["dataset_id"]})
            except Exception as e:  # noqa: BLE001 — surfaced to the UI
                update(status="stopped" if cancel.is_set() else "failed",
                       error=f"{type(e).__name__}: {e}")
            finally:
                self._persist_synth(synth_id)
                with self._lock:
                    self._synth_cancel.pop(synth_id, None)

        threading.Thread(target=run, daemon=True).start()
        return {"synth_id": synth_id}

    def cancel_synth(self, synth_id: str) -> bool:
        """Ask a running synthesis to stop. It keeps the rows it already has."""
        with self._lock:
            event = self._synth_cancel.get(synth_id)
        if event is None:
            return False
        event.set()
        return True

    def synth_status(self, synth_id: str | None = None) -> dict:
        with self._lock:
            if synth_id is not None:
                entry = self._synth.get(synth_id)
                return dict(entry) if entry else {}
            recent = sorted(self._synth.values(),
                            key=lambda e: e.get("created", 0), reverse=True)
            return {"jobs": [{k: v for k, v in e.items() if k != "logs"}
                             for e in recent]}

    def _persist_synth(self, synth_id: str) -> None:
        """Synth records outlive the process, like training jobs do."""
        with self._lock:
            entry = dict(self._synth.get(synth_id) or {})
        if not entry:
            return
        root = self.work_root / "synth"
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{synth_id}.json").write_text(json.dumps(entry))

    def _load_synth(self) -> None:
        """Reload past runs, newest first, capped so the store can't grow forever.

        Anything still marked running belongs to a process that no longer
        exists, so it is recorded as stopped rather than left spinning.
        """
        records = sorted(self.work_root.glob("synth/*.json"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        for path in records[:_SYNTH_KEEP]:
            try:
                entry = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if entry.get("status") == "running":
                entry["status"] = "stopped"
                entry["error"] = "the server restarted while this run was going"
            self._synth[entry["synth_id"]] = entry
        for path in records[_SYNTH_KEEP:]:
            path.unlink(missing_ok=True)

    # ---- persistence: jobs survive restarts ----------------------------------
    def _persist(self, job: _Job) -> None:
        out = self.work_root / job.job_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "job.json").write_text(json.dumps(job.record()))

    def _load_jobs(self) -> None:
        for rec in sorted(self.work_root.glob("*/job.json")):
            try:
                job = _Job.from_record(json.loads(rec.read_text()))
                self.jobs[job.job_id] = job
            except (json.JSONDecodeError, OSError, KeyError):
                continue

    # ---- training worker -----------------------------------------------------
    def _worker(self) -> None:
        import contextlib  # noqa: PLC0415
        import sys  # noqa: PLC0415

        from .ascii import _HEART, _NAME  # noqa: PLC0415
        from .backends import Callbacks, select_backend  # noqa: PLC0415
        from .models import _eval_holdout  # noqa: PLC0415

        while True:
            job_id = self.queue.get()
            job = self.jobs[job_id]
            if job.cancel.is_set():
                job.status = "stopped"
                self._persist(job)
                continue
            job.status = "running"
            job.logs = []  # fresh console for this run
            self._persist(job)
            tee = _LogTee(sys.__stdout__ or sys.stdout, job)
            try:
              with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                print(_HEART); print(_NAME)
                print("Starting training session...\n")
                payload = job._payload  # attached at submit time
                # "auto"/"15%"/0.15 → carve a hold-out; anything else is an
                # explicit eval set (or the dataset's own split)
                holdout = _eval_holdout(payload.get("eval_dataset"))
                # dataset by reference (dashboard — local or HF) or inline rows (SDK)
                if payload.get("dataset_id"):
                    ds = self.datasets.resolve(payload["dataset_id"])
                    payload["dataset"] = {"rows": ds.rows, "format": ds.format}
                    # the dataset's own eval split is used unless the user asked
                    # for an automatic hold-out
                    if holdout is None:
                        ev = self.datasets.resolve_eval(payload["dataset_id"])
                        if ev is not None:
                            payload["eval_dataset"] = {"rows": ev.rows, "format": ev.format}
                if holdout is not None:
                    full = _rebuild_dataset(payload["dataset"])
                    train, ev = full.split(test_size=holdout)
                    payload["dataset"] = {"rows": train.rows, "format": train.format}
                    payload["eval_dataset"] = {"rows": ev.rows, "format": ev.format}
                with self._model_lock:
                    # Playground models cached on the GPU would leave a big base
                    # too little room to train (an 8B in bf16 next to a cached
                    # 8B is an OOM on 48 GB), so a run starts on a clean card.
                    if self._infer_cache:
                        n, _ = self._drop_inference_models()
                        print(f"[{job_id[:8]}] unloaded {n} playground "
                              f"model{'s' if n != 1 else ''} to free the GPU "
                              "for training", flush=True)
                    be = select_backend(self.backend_name,
                                        accelerator=self.accelerator,
                                        device=self.device)
                    be.load(job.base_model,
                            load_in_4bit=payload["load_in_4bit"],
                            max_seq_length=payload["max_seq_length"])
                    out_dir = self.work_root / job_id
                    tag = job_id[:8]

                    # Write progress straight to the tee, not via print(): the
                    # backends run training inside quiet_backend(), which redirects
                    # sys.stdout to devnull — so a plain print() here would vanish.
                    def _on_step(m, _tee=tee, _tag=tag):
                        job.steps.append(m.to_dict())
                        bits = [f"step {m.step}"]
                        if m.loss is not None:
                            bits.append(f"loss {m.loss:.4f}")
                        if m.lr:
                            bits.append(f"lr {m.lr:.2e}")
                        if getattr(m, "tokens_per_s", None):
                            bits.append(f"{m.tokens_per_s:,.0f} tok/s")
                        _tee.write(f"[{_tag}] {' · '.join(bits)}\n")

                    def _on_eval(m, _tee=tee, _tag=tag):
                        job.evals.append(m.to_dict())
                        if m.loss is not None:
                            _tee.write(f"[{_tag}] eval · step {m.step} · loss {m.loss:.4f}\n")

                    callbacks = Callbacks(
                        on_step=_on_step,
                        on_eval=_on_eval,
                        on_log=lambda line: tee.write(f"[{tag}] {line}\n"),
                        should_stop=job.cancel.is_set,
                    )
                    result = be.finetune(
                        _rebuild_dataset(payload["dataset"]),
                        _rebuild_config(payload["config"]),
                        callbacks,
                        str(out_dir),
                        eval_dataset=_rebuild_dataset(payload["eval_dataset"]),
                    )
                job.checkpoint = result.checkpoint
                job.final_loss = result.final_loss
                job.status = "stopped" if job.cancel.is_set() else "succeeded"
            except KeyboardInterrupt:
                job.status = "stopped"
            except Exception as e:  # noqa: BLE001 — job isolation, error reported
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
                print(f"[{job_id[:8]}] FAILED: {job.error}", flush=True)
            if job.live:  # keep the last progress line in the persisted log
                job.logs.append(job.live)
                job.live = ""
            self._persist(job)  # terminal state + final metrics + logs → disk
            # The trained model would otherwise live on in this loop's locals
            # until the next job reassigns them, holding the GPU between runs
            # where Clean VRAM can't reach it: let it go now.
            be = result = callbacks = None  # noqa: F841
            self._release_gpu_cache()

    # ---- inference -------------------------------------------------------------
    def _infer_key(self, model: str, adapter: str | None,
                   checkpoint: int | None = None) -> tuple:
        """The inference-cache key: (model, resolved adapter path)."""
        from . import checkpoints as _ck  # noqa: PLC0415

        adapter_path = None
        if adapter:
            job = self.jobs.get(adapter)
            if job and job.checkpoint and Path(job.checkpoint).is_file():
                # a worker-uploaded tarball: trained elsewhere, in that
                # machine's format — this box must not try to load it
                raise RuntimeError(
                    f"this shadow was trained on another machine"
                    f"{f' ({job.worker})' if job.worker else ''} and its "
                    "adapter format matches that backend — chat with it while "
                    "that machine's `shadowlm worker` is connected")
            if job and job.checkpoint:
                adapter_path = (_ck.resolve(job.checkpoint, checkpoint)
                                if checkpoint is not None else job.checkpoint)
            else:
                adapter_path = adapter  # a server-local path the caller knows
        return (model, adapter_path)

    def _infer_model(self, model: str, adapter: str | None, checkpoint: int | None = None):
        """Load (and cache) a model for /generate and /chat.

        ``adapter`` is a run id (or a server-local adapter path the caller knows);
        ``checkpoint`` optionally picks a mid-run step saved via ``save_steps`` —
        the SDK resolves either backend's layout to a loadable path.
        """
        from . import checkpoints as _ck  # noqa: PLC0415
        from .models import load  # noqa: PLC0415

        key = self._infer_key(model, adapter, checkpoint)
        adapter_path = key[1]
        if key in self._infer_cache:
            return self._infer_cache[key]

        def _load():
            return load(model, backend=self.backend_name,
                        accelerator=self.accelerator, device=self.device,
                        adapter=adapter_path, verbose=False)

        try:
            m = _load()
        except Exception as e:
            s = str(e).lower()
            if not any(x in s for x in ("meta tensor", "out of memory")):
                raise
            # the GPU is full of previously cached models — accelerate starts
            # offloading to meta and adapter loads blow up. Evict everything,
            # free the allocator, and try once more with a clean card.
            print(f"[infer] VRAM pressure loading {model} "
                  f"({type(e).__name__}) — evicting "
                  f"{len(self._infer_cache)} cached models and retrying",
                  flush=True)
            import gc  # noqa: PLC0415

            self._infer_cache.clear()
            gc.collect()
            try:
                import torch  # noqa: PLC0415

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001 — the retry below is the real
                # recovery; but if the cache never cleared, the retry's OOM is a
                # consequence of this, so leave a trail.
                print(f"[serve] VRAM release failed ({type(e).__name__}: {e}) — "
                      "retrying the load anyway", flush=True)
            m = _load()
        if len(self._infer_cache) >= self._infer_cache_cap:
            self._infer_cache.pop(next(iter(self._infer_cache)))  # oldest out
        self._infer_cache[key] = m
        return m

    @staticmethod
    def _gpu_used_mb() -> int | None:
        """Device-wide VRAM in use (MiB), or None off-CUDA."""
        try:
            import torch  # noqa: PLC0415
            if not torch.cuda.is_available():
                return None
            free, total = torch.cuda.mem_get_info()
            return round((total - free) / 1024 / 1024)
        except Exception:  # noqa: BLE001
            return None

    def _drop_inference_models(self) -> tuple[int, str | None]:
        """Empty the inference cache and release the GPU allocator's cache.
        Caller holds ``_model_lock``. Returns (models dropped, release error)."""
        n = len(self._infer_cache)
        self._infer_cache.clear()
        return n, self._release_gpu_cache()

    @staticmethod
    def _release_gpu_cache() -> str | None:
        """Collect dropped models and hand the allocator's cache back to the
        GPU. Returns the release error, if any."""
        import gc  # noqa: PLC0415

        gc.collect()
        freed_error: str | None = None
        try:
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001 — report, don't fail the caller
            freed_error = f"{type(e).__name__}: {e}"
            print(f"[serve] VRAM release failed ({freed_error})", flush=True)
        gc.collect()
        return freed_error

    def clear_vram(self) -> dict:
        """Drop every cached inference model and release the GPU allocator's
        cache — frees VRAM held after inference/compare without restarting the
        server. Queued/running training is untouched (one job at a time)."""
        before = self._gpu_used_mb()
        with self._model_lock:
            n, freed_error = self._drop_inference_models()
        out = {"unloaded": n, "before_mb": before, "after_mb": self._gpu_used_mb()}
        if freed_error:  # don't report a clean sweep when the allocator threw
            out["error"] = freed_error
        return out

    # ---- operations ------------------------------------------------------------
    # ---- evaluations -------------------------------------------------------
    def _load_evals(self) -> None:
        for rec in sorted(self._evals_root.glob("*.json")):
            try:
                ev = _Eval(**json.loads(rec.read_text()))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            if ev.status in ("pending", "running"):  # the server restarted mid-run
                ev.status, ev.error = "failed", "the server restarted during this evaluation"
            self.evals[ev.eval_id] = ev

    def _persist_eval(self, ev: _Eval) -> None:
        self._evals_root.mkdir(parents=True, exist_ok=True)
        (self._evals_root / f"{ev.eval_id}.json").write_text(json.dumps(ev.record()))

    def start_eval(self, body: dict) -> _Eval:
        """Validate an evaluation request and run it in the background."""
        ds_id = body.get("dataset_id") or ""
        if self.datasets.meta(ds_id) is None:
            raise ValueError(f"unknown dataset {ds_id!r}")
        metric = body.get("metric") or "contains"
        if metric not in _EVAL_METRICS:
            raise ValueError(f"metric must be one of {', '.join(_EVAL_METRICS)}")
        targets = body.get("targets")
        if not isinstance(targets, list) or not 1 <= len(targets) <= _EVAL_MAX_TARGETS:
            raise ValueError(f"targets must be a list of 1-{_EVAL_MAX_TARGETS} models")
        fm = self.frontier()
        if metric == "judge" and fm is None:
            raise ValueError("judge scoring needs a frontier model; add one in settings")
        clean = []
        for t in targets:
            if isinstance(t, dict) and t.get("kind") == "frontier":
                if fm is None:
                    raise ValueError("a frontier baseline needs a frontier model; add one in settings")
                clean.append({"label": str(t.get("label") or "Frontier model")[:40],
                              "model": fm.name, "adapter": None, "checkpoint": None,
                              "kind": "frontier"})
                continue
            if not isinstance(t, dict) or not t.get("model"):
                raise ValueError("each target needs a 'model'")
            clean.append({"label": str(t.get("label") or t.get("adapter") or "base")[:40],
                          "model": str(t["model"]),
                          "adapter": t.get("adapter") or None,
                          "checkpoint": t.get("checkpoint")})
        sample = body.get("sample")
        if sample is not None and (not isinstance(sample, int) or sample < 1):
            raise ValueError("sample must be a positive integer")
        ev = _Eval(eval_id=uuid.uuid4().hex[:12],
                   name=str(body.get("name") or "")[:80],
                   dataset_id=ds_id, metric=metric, targets=clean,
                   sample=sample if sample is not None else 50,
                   created=int(time.time()))
        with self._lock:
            self.evals[ev.eval_id] = ev
        self._persist_eval(ev)
        threading.Thread(target=self._run_eval, args=(ev,), daemon=True).start()
        return ev

    def _eval_rows(self, ds_id: str) -> Dataset:
        """The questions an evaluation asks: the dataset's own eval split when it
        has one, else its rows (for a fact set, 'does it know these' is the test)."""
        return self.datasets.resolve_eval(ds_id) or self.datasets.resolve(ds_id)

    def _run_eval(self, ev: _Eval) -> None:
        from .eval import evaluate  # noqa: PLC0415

        ev.status = "running"
        self._persist_eval(ev)
        try:
            data = self._eval_rows(ev.dataset_id)
            judge = self.frontier() if ev.metric == "judge" else None
            for t in ev.targets:
                if t.get("kind") == "frontier":
                    fm = self.frontier()
                    if fm is None:
                        raise RuntimeError("the frontier model was removed from settings")
                    res = evaluate(fm, data, metric=ev.metric, judge=judge,
                                   sample=ev.sample, verbose=False)
                    ev.results.append(res.to_dict())
                    self._persist_eval(ev)
                    continue
                wjob = self.jobs.get(t["adapter"] or "")
                if wjob is not None and wjob.worker:
                    raise RuntimeError(
                        f"{t['label']} was trained on {wjob.worker}; evaluate it "
                        "there — this box can't load that machine's adapter format")
                with self._model_lock:
                    m = self._infer_model(t["model"], t["adapter"], t["checkpoint"])
                    res = evaluate(m, data, metric=ev.metric, judge=judge,
                                   sample=ev.sample, verbose=False)
                ev.results.append(res.to_dict())
                self._persist_eval(ev)  # each target's numbers land as they finish
            ev.status = "succeeded"
        except Exception as e:  # noqa: BLE001 — reported on the evaluation
            ev.status, ev.error = "failed", f"{type(e).__name__}: {e}"
            print(f"[eval {ev.eval_id[:8]}] FAILED: {ev.error}", flush=True)
        ev.finished = int(time.time())
        self._persist_eval(ev)

    # ---- chat as a background task -----------------------------------------
    def start_chat(self, body: dict) -> _ChatTask:
        """Answer a chat in the background: the first answer from a cold model
        can take longer than the proxy in front of the studio holds a request."""
        task = _ChatTask(task_id=uuid.uuid4().hex[:12], created=time.time())
        with self._lock:
            cutoff = time.time() - 3600  # finished answers are kept for an hour
            for tid in [k for k, t in self.chat_tasks.items()
                        if t.status != "running" and t.created < cutoff]:
                self.chat_tasks.pop(tid, None)
            self.chat_tasks[task.task_id] = task
        threading.Thread(target=self._run_chat, args=(task, body), daemon=True).start()
        return task

    def _run_chat(self, task: _ChatTask, b: dict) -> None:
        try:
            wjob = self.jobs.get(b.get("adapter") or "")
            if wjob is not None and wjob.worker:
                task.text = self.worker_infer(wjob, {
                    "type": "chat", "messages": b["messages"],
                    "max_new_tokens": b.get("max_new_tokens", 512),
                    "temperature": b.get("temperature", 0.7),
                    "top_p": b.get("top_p", 0.95)})
            else:
                with self._model_lock:
                    m = self._infer_model(b["model"], b.get("adapter"), b.get("checkpoint"))
                    reply = m.chat(b["messages"], tools=b.get("tools"),
                                   max_new_tokens=b.get("max_new_tokens", 512),
                                   temperature=b.get("temperature", 0.7),
                                   top_p=b.get("top_p", 0.95))
                task.text = reply.raw or reply.content
            task.status = "succeeded"
        except Exception as e:  # noqa: BLE001 — reported on the task
            task.status, task.error = "failed", f"{type(e).__name__}: {e}"

    def submit(self, payload: dict) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = _Job(job_id=job_id, base_model=payload["base_model"],
                   name=(payload.get("name") or "").strip(),
                   method=(payload.get("config") or {}).get("method"),
                   worker=(payload.get("worker") or None),
                   created=int(time.time()))
        job._payload = payload
        with self._lock:
            self.jobs[job_id] = job
        self._persist(job)  # visible (as pending) the instant it's queued
        # worker="<worker name>" routes the job to that machine's inbox instead
        # of this box's own training queue. Submitting ahead of the worker
        # connecting is fine — the inbox waits.
        if payload.get("worker"):
            self.worker_named(payload["worker"]).inbox.put(job_id)
        else:
            self.queue.put(job_id)
        return job_id

    # ---- remote workers --------------------------------------------------------
    def worker_named(self, name: str) -> _Worker:
        with self._lock:
            return self.workers.setdefault(name, _Worker(name=name))

    def register_worker(self, body: dict) -> _Worker:
        w = self.worker_named(body["name"])
        w.backend = body.get("backend", w.backend)
        w.device = body.get("device", w.device)
        w.gpus = int(body.get("gpus") or 0)
        w.gpu_name = str(body.get("gpu_name") or "")
        w.vram_gb = float(body.get("vram_gb") or 0)
        w.ram_gb = float(body.get("ram_gb") or 0)
        w.cores = int(body.get("cores") or 0)
        if isinstance(body.get("models"), list):
            w.models = body["models"][:50]
        w.last_seen = time.time()
        return w

    def next_job_for(self, name: str, *, wait_s: float = 20.0) -> _Job | None:
        """Long-poll pickup: the next runnable job in this worker's inbox."""
        w = self.worker_named(name)
        deadline = time.monotonic() + wait_s
        while True:
            w.last_seen = time.time()
            try:
                job_id = w.inbox.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                return None
            job = self.jobs.get(job_id)
            if job is None:
                continue
            if job.cancel.is_set():  # cancelled while waiting for pickup
                job.status = "stopped"
                self._persist(job)
                continue
            job.status = "running"
            job.logs = []
            self._persist(job)
            return job

    def serve_worker_socket(self, name: str, conn) -> None:
        """One worker's whole session, run on its connection's handler thread.

        Uplink: the worker streams `events` messages in. Downlink: a sender
        thread pushes `job` messages the moment something lands in the inbox,
        and `cancel` the moment the studio asks — that's the bidirectional wire.
        """
        first = conn.recv_json(timeout=30.0)
        if not isinstance(first, dict) or first.get("type") != "register":
            conn.close()
            return
        self.register_worker({**first, "name": name})
        with self._lock:
            self.worker_socks[name] = conn
        print(f"[hub] worker '{name}' connected "
              f"({first.get('backend', '?')} · {first.get('device', '?')})",
              flush=True)

        dead = threading.Event()

        def sender() -> None:
            while not dead.is_set():
                job = self.next_job_for(name, wait_s=1.0)
                if job is None:
                    continue
                # a send into a just-closed socket can "succeed" into the TCP
                # buffer — so re-check liveness after popping, and requeue
                # instead of dispatching into the void
                if dead.is_set():
                    requeue = True
                else:
                    try:
                        conn.send_json({"type": "job", "job": {
                            "job_id": job.job_id, **self.wire_payload(job)}})
                        requeue = False
                    except OSError:
                        requeue = True
                if requeue:
                    job.status = "pending"
                    self._persist(job)
                    self.worker_named(name).inbox.put(job.job_id)
                    return

        threading.Thread(target=sender, daemon=True).start()
        try:
            while True:
                try:
                    msg = conn.recv_json(timeout=90.0)  # workers ping every 45s
                except TimeoutError:
                    break  # silent too long: presume gone; it will reconnect
                if msg is None:
                    break
                self.worker_named(name).last_seen = time.time()
                if msg.get("type") == "events":
                    if (job := self.jobs.get(msg.get("job_id"))) is not None:
                        if self.ingest_events(job, msg)["cancel"]:
                            conn.send_json({"type": "cancel",
                                            "job_id": job.job_id})
                elif msg.get("type") == "infer_result":
                    if (q := self._infer_waiters.pop(msg.get("id", ""), None)):
                        q.put(msg)
        except (ConnectionError, OSError, json.JSONDecodeError):
            pass  # a dropped worker is normal — the studio just shows offline
        finally:
            dead.set()
            with self._lock:
                if self.worker_socks.get(name) is conn:
                    del self.worker_socks[name]
            conn.close()
            print(f"[hub] worker '{name}' disconnected", flush=True)

    def prewarm(self, model: str, adapter: str | None,
                checkpoint: int | None = None) -> dict:
        """Start loading a model for inference in the background; report state.

        Proxies (Cloudflare) cut requests around 100s, and a cold 8B load takes
        longer — so the UI fires this when a shadow is picked and polls until
        {ready: true}, keeping every chat request warm-fast.
        """
        job = self.jobs.get(adapter or "")
        if job is not None and job.worker:
            return {"ready": True}  # answers on its worker; nothing to load here
        key = self._infer_key(model, adapter, checkpoint)
        with self._lock:
            if key in self._infer_cache:
                return {"ready": True}
            if key in self._prewarming:
                return {"ready": False}
            # a load that just failed will fail again — report it instead of
            # respawning the same doomed load on every 3s poll
            when, msg = self._prewarm_errors.get(key, (0, ""))
            if msg and time.time() - when < 120:
                return {"ready": False, "error": msg}
            self._prewarm_errors.pop(key, None)
            self._prewarming.add(key)

        def load() -> None:
            try:
                with self._model_lock:
                    self._infer_model(model, adapter, checkpoint)
            except Exception as e:  # noqa: BLE001 — reported on the next poll
                print(f"[prewarm] {model} failed: {e}", flush=True)
                with self._lock:
                    self._prewarm_errors[key] = (time.time(),
                                                 f"{type(e).__name__}: {e}")
            finally:
                with self._lock:
                    self._prewarming.discard(key)

        threading.Thread(target=load, daemon=True).start()
        return {"ready": False}

    def worker_infer(self, job: _Job, req: dict, *, timeout: float = 240.0) -> str:
        """Run generate/chat on the machine that trained `job`, over its socket.

        An mlx-trained adapter can't load into this box's torch stack (and vice
        versa) — the weights live where they trained, so inference goes there.
        """
        conn = self.worker_socks.get(job.worker)
        if conn is None:
            raise RuntimeError(
                f"this shadow was trained on machine '{job.worker}', which is "
                f"offline — start `shadowlm worker` there to chat with it")
        rid = uuid.uuid4().hex
        q: "queue.Queue[dict]" = queue.Queue()
        self._infer_waiters[rid] = q
        try:
            conn.send_json({**req, "id": rid, "job_id": job.job_id,
                            "base_model": job.base_model})
            reply = q.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(
                f"machine '{job.worker}' didn't answer within {timeout:.0f}s"
            ) from None
        except OSError as e:
            raise RuntimeError(
                f"machine '{job.worker}' dropped its link mid-request ({e})"
            ) from None
        finally:
            self._infer_waiters.pop(rid, None)
        if reply.get("error"):
            raise RuntimeError(f"machine '{job.worker}': {reply['error']}")
        return reply.get("text", "")

    def push_cancel(self, job: _Job) -> None:
        """Tell the executing worker to stop, right now, over its socket."""
        name = (getattr(job, "_payload", None) or {}).get("worker")
        if not name:
            return  # hub-local job: its own training loop watches the flag
        conn = self.worker_socks.get(name)
        if conn is not None:
            try:
                conn.send_json({"type": "cancel", "job_id": job.job_id})
                return
            except OSError:
                pass  # fall through: the socket just died
        # no live socket to deliver to — terminalize here so the studio isn't
        # stuck showing "running" for a machine that's gone
        if job.status in ("pending", "running"):
            job.status = "stopped"
            self._persist(job)

    def wire_payload(self, job: _Job) -> dict:
        """The job as shipped to a worker: dataset refs resolved to inline rows
        (the worker has no access to this hub's dataset store)."""
        from .models import _eval_holdout  # noqa: PLC0415

        payload = dict(job._payload)
        if payload.get("dataset_id"):
            ds = self.datasets.resolve(payload["dataset_id"])
            payload["dataset"] = {"rows": ds.rows, "format": ds.format}
            if _eval_holdout(payload.get("eval_dataset")) is None:
                ev = self.datasets.resolve_eval(payload["dataset_id"])
                if ev is not None:
                    payload["eval_dataset"] = {"rows": ev.rows, "format": ev.format}
        return payload

    def ingest_events(self, job: _Job, body: dict) -> dict:
        """Fold a worker's pushed progress into the job record the studio reads."""
        job.steps.extend(body.get("steps") or [])
        job.evals.extend(body.get("evals") or [])
        job.logs.extend(body.get("logs") or [])
        if len(job.logs) > job._LOG_CAP * 2:
            job.logs = job.logs[-job._LOG_CAP:]
        if body.get("final_loss") is not None:
            job.final_loss = body["final_loss"]
        if body.get("status") in ("succeeded", "failed", "stopped"):
            job.status = body["status"]
            job.error = body.get("error")
            self._persist(job)
        return {"cancel": job.cancel.is_set()}  # the downlink: hub → worker

    def store_artifact(self, job: _Job, blob: bytes) -> None:
        """A worker shipped its trained adapter home — the weights live here now."""
        path = self.work_root / job.job_id / "artifact.tar.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        job.checkpoint = str(path)
        self._persist(job)

    def artifact(self, job: _Job) -> bytes:
        root = Path(job.checkpoint)
        if root.is_file():  # a worker-uploaded tarball: already in wire format
            return root.read_bytes()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    tar.add(p, arcname=str(p.relative_to(root)))
        return buf.getvalue()


class Auth:
    """Studio auth — username/password login that mints short-lived bearer tokens,
    plus the legacy static `SHADOWLM_API_KEY`. Both gate every `/v1` route.

    Tokens are HMAC-signed (`<exp>.<sig>`, key = sha256(password)) so they need no
    server-side session store and survive restarts; an attacker can't forge one
    without the password. Auth is OFF unless a password or api key is configured.
    """

    def __init__(self, *, user: str | None, password: str | None,
                 api_key: str | None, ttl: int = 12 * 3600) -> None:
        self.user = user or "admin"
        self.password = password or None
        self.api_key = api_key or None
        self.ttl = ttl

    @property
    def enabled(self) -> bool:
        return bool(self.password or self.api_key)

    @property
    def mode(self) -> str:
        return "password" if self.password else "apikey" if self.api_key else "none"

    def _secret(self) -> bytes:
        return hashlib.sha256((self.password or "").encode()).digest()

    def check_login(self, user: str, password: str) -> bool:
        if not self.password:
            return False
        return (hmac.compare_digest(user or "", self.user)
                & hmac.compare_digest(password or "", self.password))

    def issue_token(self) -> tuple[str, int]:
        exp = int(time.time()) + self.ttl
        sig = hmac.new(self._secret(), str(exp).encode(), hashlib.sha256).hexdigest()
        return f"{exp}.{sig}", exp

    def _valid_token(self, token: str) -> bool:
        if not self.password or "." not in token:
            return False
        exp_s, _, sig = token.partition(".")
        try:
            exp = int(exp_s)
        except ValueError:
            return False
        if exp < int(time.time()):
            return False
        good = hmac.new(self._secret(), str(exp).encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, good)

    def valid_bearer(self, token: str) -> bool:
        if self.api_key and hmac.compare_digest(token, self.api_key):
            return True
        return self._valid_token(token)


def make_handler(server: Server, auth: "Auth"):
    throttle = LoginThrottle()

    ancestors = frame_ancestors()
    frame_policy = "frame-ancestors 'self' " + " ".join(ancestors)

    class Handler(BaseHTTPRequestHandler):
        def end_headers(self):
            # Only the listed host consoles may frame anything this serves.
            self.send_header("Content-Security-Policy", frame_policy)
            super().end_headers()

        def log_message(self, fmt, *args):  # quiet; job logs print directly
            pass

        # ---- plumbing -------------------------------------------------------
        def _send(self, code: int, payload: dict | bytes,
                  ctype: str = "application/json") -> None:
            body = (payload if isinstance(payload, bytes)
                    else json.dumps(payload).encode())
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code: int, msg: str) -> None:
            self._send(code, {"error": msg})

        def _openai_error(self, code: int, msg: str, kind: str = "invalid_request_error") -> None:
            """An OpenAI-shaped error: callers here are OpenAI clients (an agent
            under capture, or one using a deployment), which handle a JSON error
            but read a dropped connection as a transport fault."""
            self._send(code, {"error": {"message": msg, "type": kind, "param": None, "code": None}})

        def _send_completion(self, reply: dict, stream: bool) -> None:
            if not stream:
                return self._send(200, reply)
            from .capture import CaptureProxy  # noqa: PLC0415

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for chunk in CaptureProxy._stream_chunks(reply):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def _capture_call(self, cid: str) -> None:
            """Pass one agent call through to the frontier model and record it."""
            item = server.captures.get(cid)
            if item is None:
                return self._openai_error(404, "no such capture session")
            if item["status"] != "open":
                return self._openai_error(403, "this capture session is closed", "permission_error")
            fm = server.frontier()
            if fm is None:
                return self._openai_error(503, "no frontier model is set up in the studio", "server_error")
            body = self._body()
            if not isinstance(body.get("messages"), list):
                return self._openai_error(400, "'messages' must be a list")
            from .frontier import FrontierError  # noqa: PLC0415
            try:
                out = fm.forward(body)
                message = out["choices"][0]["message"]
            except FrontierError as e:
                return self._openai_error(502, str(e), "server_error")
            except (KeyError, IndexError, TypeError):
                return self._openai_error(502, "the frontier model's reply had no message", "server_error")
            server.captures.record(cid, {
                "session": self.headers.get("x-session-id") or "",
                "messages": body["messages"], "response": message,
                "tools": body.get("tools"), "ts": time.time()})
            out.setdefault("id", f"chatcmpl-{uuid.uuid4().hex[:24]}")
            out.setdefault("created", int(time.time()))
            out.setdefault("model", fm.name)
            self._send_completion(out, bool(body.get("stream")))

        def _deployment_call(self) -> None:
            """Serve a deployed fine-tune, authenticated by its own key."""
            from .capture import _wire_message  # noqa: PLC0415

            got = self.headers.get("Authorization", "")
            dep = server.deployments.by_key(got[7:]) if got.startswith("Bearer ") else None
            if dep is None:
                return self._openai_error(401, "invalid deployment key", "authentication_error")
            body = self._body()
            if body.get("model") and body["model"] != dep["name"]:
                return self._openai_error(404, f"this key serves the model {dep['name']!r}")
            if not isinstance(body.get("messages"), list):
                return self._openai_error(400, "'messages' must be a list")
            with server._model_lock:
                m = server._infer_model(dep["base_model"], dep["run_id"], dep.get("checkpoint"))
                reply = m.chat(body["messages"], tools=body.get("tools"),
                               temperature=body.get("temperature", 0.7),
                               top_p=body.get("top_p", 0.95),
                               max_new_tokens=body.get("max_tokens") or body.get("max_completion_tokens") or 512)
            server.deployments.used(dep["deployment_id"])
            message = reply.to_message()
            self._send_completion({
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion",
                "created": int(time.time()), "model": dep["name"],
                "choices": [{"index": 0, "message": _wire_message(message),
                             "finish_reason": "tool_calls" if reply.tool_calls else "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }, bool(body.get("stream")))

        def _authed(self) -> bool:
            if not auth.enabled:
                return True
            got = self.headers.get("Authorization", "")
            if got.startswith("Bearer ") and (
                    auth.valid_bearer(got[7:])
                    or server.valid_machine_token(got[7:])):
                return True
            self._error(401, "authentication required")
            return False

        def _body(self) -> dict:
            return json.loads(self._read_body() or b"{}")

        def _read_body(self) -> bytes:
            """The request body, refused up front when the client claims more
            than `_MAX_BODY` — we read into memory, so the header is a promise
            we'd otherwise allocate sight unseen."""
            length = int(self.headers.get("Content-Length") or 0)
            if length > _MAX_BODY:
                self.close_connection = True  # unread body would poison keep-alive
                raise _PayloadTooLarge(
                    f"request body of {length} bytes exceeds the {_MAX_BODY} cap")
            return self.rfile.read(length)

        def _require(self, body: dict, *fields: str) -> bool:
            """422 naming the first missing/empty field. Without this the route
            table's blanket except turns a client typo into `500 KeyError`."""
            for f in fields:
                if not body.get(f):
                    self._error(422, f"missing field {f!r}")
                    return False
            return True

        def _job_or_404(self, job_id: str):
            job = server.jobs.get(job_id)
            if job is None:
                self._error(404, f"unknown job {job_id!r}")
            return job

        # ---- routes ---------------------------------------------------------
        def do_GET(self):  # noqa: N802
            parts = self.path.split("?")[0].strip("/").split("/")
            if parts == [""]:  # the React studio shell — no auth for the page;
                static_index = Path(__file__).parent / "_static" / "index.html"
                if static_index.exists():  # the API itself stays authed
                    self._send(200, _with_embed_parents(
                        static_index.read_bytes(), ancestors),
                               ctype="text/html; charset=utf-8")
                else:  # only on an unbuilt source checkout — build it or use --dev
                    self._send(200, _NO_BUILD_PAGE.encode(),
                               ctype="text/html; charset=utf-8")
                return
            if parts[0] == "assets" and len(parts) == 2:
                root = (Path(__file__).parent / "_static" / "assets").resolve()
                asset = (root / parts[1]).resolve()
                if asset.is_relative_to(root) and asset.is_file():
                    ctype = ("text/javascript" if asset.suffix == ".js"
                             else "text/css" if asset.suffix == ".css"
                             else "application/octet-stream")
                    self._send(200, asset.read_bytes(), ctype=ctype)
                else:
                    self._error(404, "no such asset")
                return
            if len(parts) == 1 and parts[0].endswith(".png") and "/" not in parts[0] \
                    and not parts[0].startswith("."):
                try:
                    from importlib import resources  # noqa: PLC0415
                    blob = (resources.files("shadowlm") / "_assets" / parts[0]).read_bytes()
                    self._send(200, blob, ctype="image/png")
                except (FileNotFoundError, ModuleNotFoundError):
                    self._error(404, "no such image bundled")
                return
            if parts == ["v1", "auth"]:  # public: lets the UI decide to show login
                self._send(200, {"auth_required": auth.enabled, "mode": auth.mode})
                return
            # OpenAI clients health-check /models; these answer on their own keys
            if len(parts) == 4 and parts[:2] == ["v1", "capture"] and parts[3] == "models":
                item, fm = server.captures.get(parts[2]), server.frontier()
                if item is None or fm is None:
                    return self._openai_error(404, "no such capture session")
                return self._send(200, {"object": "list", "data": [
                    {"id": fm.name, "object": "model", "created": item["created"], "owned_by": "frontier"}]})
            if parts == ["openai", "v1", "models"]:
                got = self.headers.get("Authorization", "")
                dep = server.deployments.by_key(got[7:]) if got.startswith("Bearer ") else None
                if dep is None:
                    return self._openai_error(401, "invalid deployment key", "authentication_error")
                return self._send(200, {"object": "list", "data": [
                    {"id": dep["name"], "object": "model", "created": dep["created"], "owned_by": "openfinetuner"}]})
            if not self._authed():
                return
            if parts == ["v1", "health"]:
                self._send(200, {"ok": True, "backend": server.backend_name,
                                 "version": __version__, **server.capacity()})
            elif parts == ["v1", "finetunes"]:
                with server._lock:
                    jobs = sorted(server.jobs.values(), key=lambda j: j.created)
                self._send(200, {"jobs": [{
                    "job_id": j.job_id, "base_model": j.base_model,
                    "name": j.name,
                    "status": j.status, "error": j.error,
                    "final_loss": j.final_loss, "steps": len(j.steps),
                    "method": j.method,
                } for j in reversed(jobs)]})  # newest first
            elif parts == ["v1", "projects"]:
                self._send(200, {"projects": server.projects.list()})
            elif len(parts) == 3 and parts[:2] == ["v1", "projects"]:
                item = server.projects.get(parts[2])
                if item is None:
                    return self._error(404, f"unknown project {parts[2]!r}")
                self._send(200, item)
            elif parts == ["v1", "evals"]:
                evs = sorted(server.evals.values(), key=lambda e: -e.created)
                self._send(200, {"evals": [e.summary() for e in evs]})
            elif len(parts) == 3 and parts[:2] == ["v1", "evals"]:
                ev = server.evals.get(parts[2])
                if ev is None:
                    return self._error(404, f"unknown evaluation {parts[2]!r}")
                self._send(200, ev.record())
            elif len(parts) == 3 and parts[:2] == ["v1", "tasks"]:
                task = server.chat_tasks.get(parts[2])
                if task is None:
                    return self._error(404, f"unknown task {parts[2]!r}")
                self._send(200, {"task_id": task.task_id, "status": task.status,
                                 "text": task.text, "error": task.error})
            elif parts == ["v1", "datasets"]:
                self._send(200, {"datasets": server.datasets.list()})
            elif len(parts) == 3 and parts[:2] == ["v1", "datasets"]:
                meta = server.datasets.meta(parts[2])
                if meta is None:
                    self._error(404, f"unknown dataset {parts[2]!r}")
                else:
                    rows = server.datasets.rows(parts[2]) or []
                    self._send(200, {**meta, "preview": rows[:8]})
            elif parts == ["v1", "models"]:
                from . import hub as _hub  # noqa: PLC0415
                recent = []
                with server._lock:
                    for j in server.jobs.values():
                        if j.base_model not in recent:
                            recent.append(j.base_model)
                catalog = [{**m, "cached": _hub.is_cached(m["id"])} for m in server.catalog()]
                self._send(200, {"catalog": catalog, "recent": recent,
                                 "server_backend": server.backend_name})
            elif parts == ["v1", "models", "downloads"]:
                self._send(200, {"downloads": server.download_status()})
            elif parts == ["v1", "synth"]:
                self._send(200, server.synth_status())
            elif len(parts) == 3 and parts[:2] == ["v1", "synth"]:
                status = server.synth_status(parts[2])
                if status:
                    self._send(200, status)
                else:
                    self._error(404, f"unknown synth run {parts[2]!r}")
            elif parts == ["v1", "vram"]:
                self._send(200, {"used_mb": server._gpu_used_mb(),
                                 "cached_models": len(server._infer_cache)})
            elif parts == ["v1", "settings"]:
                from . import hub as _hub  # noqa: PLC0415
                self._send(200, {"hf_token_set": _hub.has_token(),
                                 "frontier": server.frontier_info(),
                                 "ctrl": server.ctrl.info() if server.ctrl else None})
            elif parts == ["v1", "captures"]:
                self._send(200, {"captures": server.captures.list()})
            elif len(parts) == 3 and parts[:2] == ["v1", "captures"]:
                item = server.captures.get(parts[2])
                if item is None:
                    return self._error(404, f"unknown capture {parts[2]!r}")
                eps = _capture_trajectories(item)
                self._send(200, {**{k: v for k, v in item.items() if k != "calls"},
                                 "calls": len(item["calls"]), "episodes": len(eps),
                                 "preview": [{"question": t.first_user_content()[:400],
                                              "answer": t.final_content()[:400],
                                              "turns": sum(m.get("role") == "assistant" for m in t.messages)}
                                             for t in eps[-50:][::-1]]})
            elif parts == ["v1", "deployments"]:
                self._send(200, {"deployments": server.deployments.list()})
            elif parts == ["v1", "methods"]:
                from . import methods as _methods  # noqa: PLC0415
                self._send(200, {"methods": [{
                    "name": m.name, "description": m.description,
                    "default_lr": m.default_learning_rate, "trainer": m.trainer,
                    "adapter": m.adapter,
                } for m in (_methods.get(n) for n in _methods.available())]})
            elif len(parts) == 3 and parts[:2] == ["v1", "finetunes"]:
                if (job := self._job_or_404(parts[2])):
                    self._send(200, {
                        "status": job.status, "error": job.error,
                        "checkpoint": job.checkpoint,
                        "final_loss": job.final_loss,
                    })
            elif len(parts) == 4 and parts[:2] == ["v1", "finetunes"] \
                    and parts[3] == "metrics":
                if (job := self._job_or_404(parts[2])):
                    self._send(200, {"steps": job.steps, "evals": job.evals})
            elif parts == ["v1", "workers"]:
                with server._lock:
                    infos = [w.info() for w in server.workers.values()]
                self._send(200, {"workers": sorted(infos, key=lambda w: w["name"])})
            elif parts == ["v1", "tokens"]:
                self._send(200, {"tokens": server.machine_tokens()})
            elif len(parts) == 4 and parts[:2] == ["v1", "workers"] \
                    and parts[3] == "socket":
                from . import ws  # noqa: PLC0415

                if (self.headers.get("Upgrade") or "").lower() != "websocket":
                    return self._error(426, "this route speaks websocket only")
                key = self.headers.get("Sec-WebSocket-Key")
                if not key:
                    return self._error(400, "missing Sec-WebSocket-Key")
                self.send_response_only(101)
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", ws.accept_key(key))
                self.end_headers()
                self.close_connection = True
                # Our client sends nothing until it has read this 101, so no
                # frames can be stranded in rfile's buffer — the raw socket is
                # safe to hand over. This handler thread becomes the session.
                conn = ws.WSConn(self.connection, is_client=False)
                conn.trace = f"ws:{parts[2]}"  # every frame → the hub console
                server.serve_worker_socket(parts[2], conn)
            elif len(parts) == 4 and parts[:2] == ["v1", "finetunes"] \
                    and parts[3] == "logs":
                if (job := self._job_or_404(parts[2])):
                    lines = list(job.logs) + ([job.live] if job.live else [])
                    self._send(200, {"logs": lines})
            elif len(parts) == 4 and parts[:2] == ["v1", "finetunes"] \
                    and parts[3] == "checkpoints":
                if (job := self._job_or_404(parts[2])):
                    from . import checkpoints as _ck  # noqa: PLC0415
                    cks = _ck.list_checkpoints(job.checkpoint) if job.checkpoint else []
                    self._send(200, {"checkpoints": [c.to_dict() for c in cks]})
            elif len(parts) == 4 and parts[:2] == ["v1", "finetunes"] \
                    and parts[3] == "artifact":
                if (job := self._job_or_404(parts[2])):
                    if job.status != "succeeded" or not job.checkpoint:
                        self._error(409, f"job is {job.status}; no artifact")
                    else:
                        self._send(200, server.artifact(job),
                                   ctype="application/gzip")
            else:
                self._error(404, f"no route: GET {self.path}")

        def do_POST(self):  # noqa: N802
            parts = self.path.split("?")[0].strip("/").split("/")
            if parts == ["v1", "login"]:  # public: exchange credentials for a token
                try:
                    b = self._body()
                except _PayloadTooLarge as e:
                    return self._error(413, str(e))
                addr = self.client_address[0]
                if not throttle.allowed(addr):
                    return self._error(429, "too many failed logins; wait a minute")
                ok = auth.check_login(b.get("username", ""), b.get("password", ""))
                throttle.record(addr, ok=ok)
                if ok:
                    token, exp = auth.issue_token()
                    self._send(200, {"token": token, "user": auth.user, "expires": exp})
                else:
                    self._error(401, "invalid username or password")
                return
            # an agent under capture, or one using a deployment: own credentials
            if len(parts) == 5 and parts[:2] == ["v1", "capture"] and parts[3:] == ["chat", "completions"]:
                try:
                    return self._capture_call(parts[2])
                except _PayloadTooLarge as e:
                    return self._openai_error(413, str(e))
                except Exception as e:  # noqa: BLE001 — the agent gets an error, not a dead socket
                    return self._openai_error(500, f"{type(e).__name__}: {e}", "server_error")
            if parts == ["openai", "v1", "chat", "completions"]:
                try:
                    return self._deployment_call()
                except _PayloadTooLarge as e:
                    return self._openai_error(413, str(e))
                except Exception as e:  # noqa: BLE001
                    return self._openai_error(500, f"{type(e).__name__}: {e}", "server_error")
            if not self._authed():
                return
            try:
                if parts == ["v1", "finetunes"]:
                    body = self._body()
                    for req in ("base_model", "config"):
                        if not body.get(req):
                            return self._error(422, f"missing field {req!r}")
                    if not body.get("dataset") and not body.get("dataset_id"):
                        return self._error(422, "provide 'dataset' rows or a 'dataset_id'")
                    self._send(202, {"job_id": server.submit(body)})
                elif parts == ["v1", "projects"]:
                    b = self._body()
                    try:
                        self._send(201, server.projects.create(b.get("name"), b.get("goal")))
                    except ValueError as e:
                        self._error(422, str(e))
                elif len(parts) == 4 and parts[:2] == ["v1", "projects"] and parts[3] == "ctrl":
                    try:
                        self._send(200, server.ctrl_reply(parts[2], self._body()))
                    except KeyError:
                        self._error(404, f"unknown project {parts[2]!r}")
                    except LookupError as e:
                        self._error(503, str(e))
                    except ValueError as e:
                        self._error(422, str(e))
                    except RuntimeError as e:
                        self._error(502, str(e))
                elif parts == ["v1", "evals"]:
                    try:
                        self._send(202, server.start_eval(self._body()).record())
                    except ValueError as e:
                        self._error(422, str(e))
                elif parts == ["v1", "tasks", "chat"]:
                    b = self._body()
                    if not self._require(b, "messages", "model"):
                        return
                    if not isinstance(b["messages"], list):
                        return self._error(422, "field 'messages' must be a list")
                    self._send(202, {"task_id": server.start_chat(b).task_id})
                elif parts == ["v1", "models", "download"]:
                    b = self._body()
                    if not b.get("model"):
                        return self._error(422, "provide a 'model' id")
                    self._send(202, server.start_download(b["model"]))
                elif parts == ["v1", "synth"]:
                    b = self._body()
                    if not (b.get("task") or b.get("document") or b.get("dataset_id")):
                        return self._error(
                            422, "provide a 'task', a 'document', or a 'dataset_id'")
                    try:
                        self._send(202, server.start_synth(b))
                    except ValueError as e:
                        self._error(422, str(e))
                elif len(parts) == 4 and parts[:2] == ["v1", "synth"] \
                        and parts[3] == "cancel":
                    if server.cancel_synth(parts[2]):
                        self._send(200, {"ok": True})
                    else:
                        self._error(404, f"no synthesis running as {parts[2]!r}")
                elif parts == ["v1", "vram", "clear"]:
                    self._send(200, server.clear_vram())
                elif parts == ["v1", "models", "custom"]:
                    b = self._body()
                    model = (b.get("model") or "").strip()
                    if not model:
                        return self._error(422, "provide a 'model' id")
                    custom = (server.remove_custom_model(model) if b.get("remove")
                              else server.add_custom_model(model))
                    self._send(200, {"custom": custom})
                elif parts == ["v1", "settings"]:
                    b = self._body()
                    if "hf_token" in b:
                        server.set_hf_token((b.get("hf_token") or "").strip() or None)
                    if "frontier" in b:
                        f = b["frontier"] or {}
                        try:
                            server.set_frontier((f.get("base_url") or "").strip() or None,
                                                (f.get("api_key") or "").strip() or None,
                                                (f.get("model") or "").strip() or None)
                        except ValueError as e:
                            return self._error(422, str(e))
                    from . import hub as _hub  # noqa: PLC0415
                    self._send(200, {"hf_token_set": _hub.has_token(),
                                     "frontier": server.frontier_info(),
                                     "ctrl": server.ctrl.info() if server.ctrl else None})
                elif parts == ["v1", "captures"]:
                    b = self._body()
                    try:
                        item = server.captures.create(b.get("name"), b.get("project_id"))
                    except ValueError as e:
                        return self._error(422, str(e))
                    self._send(201, {**{k: v for k, v in item.items() if k != "calls"}, "calls": 0})
                elif len(parts) == 4 and parts[:2] == ["v1", "captures"] and parts[3] == "close":
                    item = server.captures.close(parts[2])
                    if item is None:
                        return self._error(404, f"unknown capture {parts[2]!r}")
                    self._send(200, {"ok": True})
                elif len(parts) == 4 and parts[:2] == ["v1", "captures"] and parts[3] == "dataset":
                    item = server.captures.get(parts[2])
                    if item is None:
                        return self._error(404, f"unknown capture {parts[2]!r}")
                    eps = _capture_trajectories(item)
                    if not eps:
                        return self._error(409, "nothing captured yet: run the agent against the capture URL first")
                    from . import traces as _traces  # noqa: PLC0415
                    ds = _traces.to_dataset(eps)
                    name = (self._body().get("name") or item["name"]).strip()[:80]
                    self._send(201, server.datasets.save(name, ds.rows))
                elif parts == ["v1", "datasets", "traces"]:
                    b = self._body()
                    if not b.get("traces"):
                        return self._error(422, "provide 'traces': OTLP JSON or a list of spans")
                    from . import traces as _traces  # noqa: PLC0415
                    try:
                        ds = _traces.to_dataset(b["traces"])
                    except (ValueError, TypeError, KeyError) as e:
                        return self._error(422, f"couldn't read those traces: {e}")
                    if not ds.rows:
                        return self._error(422, "no model calls found in those traces")
                    self._send(201, server.datasets.save((b.get("name") or "Imported traces").strip()[:80], ds.rows))
                elif parts == ["v1", "deployments"]:
                    b = self._body()
                    job = server.jobs.get(b.get("run_id") or "")
                    if job is None or job.status != "succeeded":
                        return self._error(422, "deploy a fine-tune whose run succeeded")
                    if job.worker:
                        return self._error(422, f"this fine-tune was trained on {job.worker}; "
                                                "it can only be served while that machine is connected")
                    try:
                        dep, key = server.deployments.create(
                            name=b.get("name"), run_id=job.job_id, base_model=job.base_model,
                            checkpoint=b.get("checkpoint"), project_id=b.get("project_id"))
                    except ValueError as e:
                        return self._error(422, str(e))
                    server.prewarm(job.base_model, job.job_id, b.get("checkpoint"))  # first call fast
                    self._send(201, {"deployment": dep, "key": key})
                elif parts == ["v1", "datasets", "hf-info"]:
                    b = self._body()
                    if not b.get("repo"):
                        return self._error(422, "provide a HuggingFace 'repo'")
                    self._send(200, Dataset.hf_info(b["repo"], subset=b.get("subset")))
                elif parts == ["v1", "datasets", "preview"]:
                    b = self._body()
                    if not b.get("repo"):
                        return self._error(422, "provide a HuggingFace 'repo'")
                    self._send(200, Dataset.hf_preview(
                        b["repo"], subset=b.get("subset"),
                        split=b.get("split", "train"), limit=b.get("limit", 8)))
                elif parts == ["v1", "datasets"]:
                    body = self._body()
                    if body.get("source") == "hf":
                        if not body.get("repo"):
                            return self._error(422, "provide a HuggingFace 'repo'")
                        self._send(201, server.datasets.save_hf(
                            body["repo"], subset=body.get("subset"),
                            split=body.get("split", "train"),
                            eval_split=body.get("eval_split"),
                            fmt=body.get("format", "?"), rows=body.get("rows")))
                    else:
                        rows = body.get("rows")
                        if not rows or not isinstance(rows, list):
                            return self._error(422, "provide 'rows': a list of JSON objects")
                        self._send(201, server.datasets.save(body.get("name", ""), rows))
                elif len(parts) == 4 and parts[:2] == ["v1", "finetunes"] \
                        and parts[3] == "cancel":
                    if (job := self._job_or_404(parts[2])):
                        job.cancel.set()
                        server.push_cancel(job)  # instant over the socket
                        self._send(200, {"ok": True})
                elif parts == ["v1", "tokens"]:
                    b = self._body()
                    tname = (b.get("name") or "").strip()
                    if not tname:
                        return self._error(422, "provide a token 'name'")
                    self._send(201, {"name": tname,
                                     "token": server.mint_machine_token(tname)})
                elif len(parts) == 6 and parts[:2] == ["v1", "workers"] \
                        and parts[3] == "jobs" and parts[5] == "artifact":
                    if (job := self._job_or_404(parts[4])):
                        server.store_artifact(job, self._read_body())
                        self._send(200, {"ok": True})
                elif parts == ["v1", "prewarm"]:
                    b = self._body()
                    if not self._require(b, "model"):
                        return
                    self._send(200, server.prewarm(
                        b["model"], b.get("adapter"), b.get("checkpoint")))
                elif parts == ["v1", "generate"]:
                    b = self._body()
                    if not self._require(b, "prompt", "model"):
                        return
                    # a worker-trained shadow answers on its own machine — the
                    # adapter format matches that backend, not necessarily ours
                    wjob = server.jobs.get(b.get("adapter") or "")
                    if wjob is not None and wjob.worker:
                        self._send(200, {"text": server.worker_infer(wjob, {
                            "type": "generate", "prompt": b["prompt"],
                            "max_new_tokens": b.get("max_new_tokens", 256),
                            "temperature": b.get("temperature", 0.7),
                            "top_p": b.get("top_p", 0.95)})})
                        return
                    with server._model_lock:
                        m = server._infer_model(b["model"], b.get("adapter"), b.get("checkpoint"))
                        text = m.generate(
                            b["prompt"],
                            max_new_tokens=b.get("max_new_tokens", 256),
                            temperature=b.get("temperature", 0.7),
                            top_p=b.get("top_p", 0.95))
                    self._send(200, {"text": text})
                elif parts == ["v1", "chat"]:
                    b = self._body()
                    if not self._require(b, "messages", "model"):
                        return
                    if not isinstance(b["messages"], list):
                        return self._error(422, "field 'messages' must be a list")
                    wjob = server.jobs.get(b.get("adapter") or "")
                    if wjob is not None and wjob.worker:
                        self._send(200, {"text": server.worker_infer(wjob, {
                            "type": "chat", "messages": b["messages"],
                            "max_new_tokens": b.get("max_new_tokens", 512),
                            "temperature": b.get("temperature", 0.7),
                            "top_p": b.get("top_p", 0.95)})})
                        return
                    with server._model_lock:
                        m = server._infer_model(b["model"], b.get("adapter"), b.get("checkpoint"))
                        reply = m.chat(
                            b["messages"], tools=b.get("tools"),
                            max_new_tokens=b.get("max_new_tokens", 512),
                            temperature=b.get("temperature", 0.7),
                            top_p=b.get("top_p", 0.95))
                    self._send(200, {"text": reply.raw or reply.content})
                else:
                    self._error(404, f"no route: POST {self.path}")
            except _PayloadTooLarge as e:
                self._error(413, str(e))
            except Exception as e:  # noqa: BLE001 — report, keep serving
                self._error(500, f"{type(e).__name__}: {e}")

        def do_PATCH(self):  # noqa: N802
            if not self._authed():
                return
            parts = self.path.split("?")[0].strip("/").split("/")
            try:
                if len(parts) == 3 and parts[:2] == ["v1", "projects"]:
                    try:
                        item = server.projects.update(parts[2], self._body())
                    except ValueError as e:
                        return self._error(422, str(e))
                    if item is None:
                        return self._error(404, f"unknown project {parts[2]!r}")
                    self._send(200, item)
                else:
                    self._error(404, f"no route: PATCH {self.path}")
            except _PayloadTooLarge as e:
                self._error(413, str(e))
            except Exception as e:  # noqa: BLE001 — report, keep serving
                self._error(500, f"{type(e).__name__}: {e}")

        def do_DELETE(self):  # noqa: N802
            if not self._authed():
                return
            parts = self.path.split("?")[0].strip("/").split("/")
            if len(parts) == 3 and parts[:2] == ["v1", "datasets"]:
                if server.datasets.delete(parts[2]):
                    self._send(200, {"ok": True})
                else:
                    self._error(404, f"unknown dataset {parts[2]!r}")
            elif len(parts) == 3 and parts[:2] == ["v1", "projects"]:
                if server.projects.delete(parts[2]):
                    self._send(200, {"ok": True})
                else:
                    self._error(404, f"unknown project {parts[2]!r}")
            elif len(parts) == 3 and parts[:2] == ["v1", "captures"]:
                if server.captures.delete(parts[2]):
                    self._send(200, {"ok": True})
                else:
                    self._error(404, f"unknown capture {parts[2]!r}")
            elif len(parts) == 3 and parts[:2] == ["v1", "deployments"]:
                if server.deployments.delete(parts[2]):
                    self._send(200, {"ok": True})
                else:
                    self._error(404, f"unknown deployment {parts[2]!r}")
            elif len(parts) == 3 and parts[:2] == ["v1", "evals"]:
                ev = server.evals.get(parts[2])
                if ev is None or ev.status in ("pending", "running"):
                    return self._error(404 if ev is None else 409,
                                       f"unknown evaluation {parts[2]!r}" if ev is None
                                       else "this evaluation is still running")
                server.evals.pop(parts[2], None)
                (server._evals_root / f"{parts[2]}.json").unlink(missing_ok=True)
                self._send(200, {"ok": True})
            elif len(parts) == 3 and parts[:2] == ["v1", "tokens"]:
                if server.revoke_machine_token(parts[2]):
                    self._send(200, {"ok": True})
                else:
                    self._error(404, f"unknown token {parts[2]!r}")
            else:
                self._error(404, f"no route: DELETE {self.path}")

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m shadowlm.serve",
        description="ShadowLM reference server — the remote protocol over the "
                    "local backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--backend", default="auto", help="auto | mlx | torch")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--work-dir", default=str(Path.home() / ".shadowlm" / "serve"))
    parser.add_argument("--dev", action="store_true",
                        help="also launch the Vite UI dev server (hot reload)")
    args = parser.parse_args(argv)

    auth = Auth(user=os.environ.get("SHADOWLM_USER", "admin"),
                password=os.environ.get("SHADOWLM_PASSWORD"),
                api_key=os.environ.get("SHADOWLM_API_KEY"))
    work_root = Path(args.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    server = Server(backend=args.backend, accelerator=args.accelerator,
                    device=args.device, work_root=work_root)
    httpd = ThreadingHTTPServer((args.host, args.port),
                                make_handler(server, auth))

    static = Path(__file__).parent / "_static" / "index.html"
    ui = ("React studio" if static.exists() else "built-in dashboard (no-build)")
    auth_desc = (f"login required (user '{auth.user}')" if auth.password
                 else "Bearer api-key auth" if auth.api_key
                 else "no auth (set SHADOWLM_PASSWORD to require login)")
    base = f"http://{args.host}:{args.port}"
    print(f"slm♥ ShadowLM server · {base} · backend={args.backend} · {auth_desc}",
          flush=True)
    print(f"     UI: {ui} + API on the same port — open {base}", flush=True)

    vite = _maybe_start_vite(args.port) if args.dev else None
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        if vite:
            vite.terminate()
    return 0


def _maybe_start_vite(api_port: int):
    """Launch `npm run dev` in frontend/ for hot-reload UI work, proxying its
    /v1 calls to this backend. Dev convenience only — prod serves _static."""
    import subprocess  # noqa: PLC0415

    frontend = Path(__file__).resolve().parent.parent / "frontend"
    if not (frontend / "package.json").exists():
        print("     --dev: no frontend/ here (source checkout only) — skipping",
              flush=True)
        return None
    env = {**os.environ, "SHADOWLM_DEV_API": f"http://127.0.0.1:{api_port}"}
    try:
        proc = subprocess.Popen(["npm", "run", "dev"], cwd=frontend, env=env)
    except FileNotFoundError:
        print("     --dev: npm not found — skipping the UI dev server", flush=True)
        return None
    print("     --dev: Vite hot-reload UI starting (see its URL above) — "
          "edit frontend/src and refresh", flush=True)
    return proc


if __name__ == "__main__":
    raise SystemExit(main())
