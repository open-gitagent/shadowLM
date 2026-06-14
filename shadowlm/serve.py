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
import io
import json
import os
import queue
import tarfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import __version__
from .data import Dataset
from .training import Metric, TrainConfig

DEFAULT_PORT = 8329

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
    created: float = 0.0
    logs: list[str] = field(default_factory=list)  # captured console lines
    live: str = ""  # the in-progress (post-\r) line, e.g. the progress bar
    cancel: threading.Event = field(default_factory=threading.Event)

    _LOG_CAP = 600  # keep the tail; banner + progress + done fit comfortably

    def record(self) -> dict:
        """The serializable job record persisted to disk (survives restarts)."""
        return {"job_id": self.job_id, "base_model": self.base_model,
                "name": self.name,
                "status": self.status, "method": self.method,
                "error": self.error, "checkpoint": self.checkpoint,
                "final_loss": self.final_loss, "created": self.created,
                "steps": self.steps, "evals": self.evals,
                "logs": self.logs[-self._LOG_CAP:]}

    @classmethod
    def from_record(cls, d: dict) -> "_Job":
        job = cls(job_id=d["job_id"], base_model=d.get("base_model", "?"),
                  name=d.get("name", ""),
                  status=d.get("status", "succeeded"), method=d.get("method"),
                  error=d.get("error"), checkpoint=d.get("checkpoint"),
                  final_loss=d.get("final_loss"), created=d.get("created", 0.0))
        job.steps = d.get("steps", [])
        job.evals = d.get("evals", [])
        job.logs = d.get("logs", [])
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
]


class DatasetStore:
    """Datasets in the dashboard — uploaded JSONL, or a HuggingFace reference.
    Both persist as a small JSON meta; survives restarts."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

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
                fmt: str, rows: int | None, eval_split: str | None = None) -> dict:
        """Register a HuggingFace dataset by reference (resolved at train time)."""
        ds_id = uuid.uuid4().hex[:10]
        meta = {"dataset_id": ds_id, "name": repo, "source": "hf",
                "repo": repo, "subset": subset or "default", "split": split,
                "eval_split": eval_split or None,
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
        p = self.root / f"{ds_id}.jsonl"
        if not p.exists():
            return None
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def meta(self, ds_id: str) -> dict | None:
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
        found = False
        for suffix in (".json", ".jsonl"):
            p = self.root / f"{ds_id}{suffix}"
            if p.exists():
                p.unlink()
                found = True
        return found


class Server:
    """Job store + the single training worker + an inference slot."""

    def __init__(self, *, backend: str, accelerator: str, device: str,
                 work_root: Path) -> None:
        self.backend_name = backend
        self.accelerator = accelerator
        self.device = device
        self.work_root = work_root
        self.jobs: dict[str, _Job] = {}
        self.datasets = DatasetStore(work_root / "datasets")
        self.queue: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()          # job-store mutations
        self._model_lock = threading.Lock()    # one model computation at a time
        self._infer_cache: dict[tuple, object] = {}  # (model, adapter) → Model
        self._infer_cache_cap = 3  # compare mode alternates base ↔ adapter
        self._downloads: dict[str, dict] = {}  # model id → prefetch status
        self._settings_path = work_root / "settings.json"
        self._load_settings()
        self._load_jobs()
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- settings: HF token for gated/private models --------------------------
    def _load_settings(self) -> None:
        from . import hub  # noqa: PLC0415

        try:
            saved = json.loads(self._settings_path.read_text())
        except (OSError, ValueError):
            saved = {}
        # a persisted token wins; otherwise honor one already in the environment
        if saved.get("hf_token") or os.environ.get("HF_TOKEN"):
            hub.set_token(saved.get("hf_token") or os.environ.get("HF_TOKEN"))

    def set_hf_token(self, token: str | None) -> None:
        from . import hub  # noqa: PLC0415

        hub.set_token(token)
        try:
            self._settings_path.write_text(json.dumps({"hf_token": token or ""}))
        except OSError:
            pass  # in-memory still works for this process

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
                    be = select_backend(self.backend_name,
                                        accelerator=self.accelerator,
                                        device=self.device)
                    be.load(job.base_model,
                            load_in_4bit=payload["load_in_4bit"],
                            max_seq_length=payload["max_seq_length"])
                    out_dir = self.work_root / job_id
                    callbacks = Callbacks(
                        on_step=lambda m: job.steps.append(m.to_dict()),
                        on_eval=lambda m: job.evals.append(m.to_dict()),
                        on_log=lambda line: print(f"[{job_id[:8]}] {line}", flush=True),
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

    # ---- inference -------------------------------------------------------------
    def _infer_model(self, model: str, adapter: str | None, checkpoint: int | None = None):
        """Load (and cache) a model for /generate and /chat.

        ``adapter`` is a run id (or a server-local adapter path the caller knows);
        ``checkpoint`` optionally picks a mid-run step saved via ``save_steps`` —
        the SDK resolves either backend's layout to a loadable path.
        """
        from . import checkpoints as _ck  # noqa: PLC0415
        from .models import load  # noqa: PLC0415

        adapter_path = None
        if adapter:
            job = self.jobs.get(adapter)
            if job and job.checkpoint:
                adapter_path = (_ck.resolve(job.checkpoint, checkpoint)
                                if checkpoint is not None else job.checkpoint)
            else:
                adapter_path = adapter  # a server-local path the caller knows
        key = (model, adapter_path)
        if key in self._infer_cache:
            return self._infer_cache[key]
        m = load(model, backend=self.backend_name, accelerator=self.accelerator,
                 device=self.device, adapter=adapter_path, verbose=False)
        if len(self._infer_cache) >= self._infer_cache_cap:
            self._infer_cache.pop(next(iter(self._infer_cache)))  # oldest out
        self._infer_cache[key] = m
        return m

    # ---- operations ------------------------------------------------------------
    def submit(self, payload: dict) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = _Job(job_id=job_id, base_model=payload["base_model"],
                   name=(payload.get("name") or "").strip(),
                   method=(payload.get("config") or {}).get("method"),
                   created=int(time.time()))
        job._payload = payload
        with self._lock:
            self.jobs[job_id] = job
        self._persist(job)  # visible (as pending) the instant it's queued
        self.queue.put(job_id)
        return job_id

    def artifact(self, job: _Job) -> bytes:
        buf = io.BytesIO()
        root = Path(job.checkpoint)
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    tar.add(p, arcname=str(p.relative_to(root)))
        return buf.getvalue()


def make_handler(server: Server, api_key: str | None):
    class Handler(BaseHTTPRequestHandler):
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

        def _authed(self) -> bool:
            if not api_key:
                return True
            got = self.headers.get("Authorization", "")
            if got == f"Bearer {api_key}":
                return True
            self._error(401, "missing or invalid API key")
            return False

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length) or b"{}")

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
                    self._send(200, static_index.read_bytes(),
                               ctype="text/html; charset=utf-8")
                else:  # only on an unbuilt source checkout — build it or use --dev
                    self._send(200, _NO_BUILD_PAGE.encode(),
                               ctype="text/html; charset=utf-8")
                return
            if parts[0] == "assets" and len(parts) == 2 and ".." not in parts[1]:
                asset = Path(__file__).parent / "_static" / "assets" / parts[1]
                if asset.is_file():
                    ctype = ("text/javascript" if asset.suffix == ".js"
                             else "text/css" if asset.suffix == ".css"
                             else "application/octet-stream")
                    self._send(200, asset.read_bytes(), ctype=ctype)
                else:
                    self._error(404, "no such asset")
                return
            if parts == ["logo.png"]:
                try:
                    from importlib import resources  # noqa: PLC0415
                    blob = (resources.files("shadowlm") / "_assets" / "logo.png").read_bytes()
                    self._send(200, blob, ctype="image/png")
                except (FileNotFoundError, ModuleNotFoundError):
                    self._error(404, "no logo bundled")
                return
            if not self._authed():
                return
            if parts == ["v1", "health"]:
                self._send(200, {"ok": True, "backend": server.backend_name,
                                 "version": __version__})
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
                catalog = [{**m, "cached": _hub.is_cached(m["id"])} for m in _MODEL_CATALOG]
                self._send(200, {"catalog": catalog, "recent": recent,
                                 "server_backend": server.backend_name})
            elif parts == ["v1", "models", "downloads"]:
                self._send(200, {"downloads": server.download_status()})
            elif parts == ["v1", "settings"]:
                from . import hub as _hub  # noqa: PLC0415
                self._send(200, {"hf_token_set": _hub.has_token()})
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
            if not self._authed():
                return
            parts = self.path.split("?")[0].strip("/").split("/")
            try:
                if parts == ["v1", "finetunes"]:
                    body = self._body()
                    for req in ("base_model", "config"):
                        if not body.get(req):
                            return self._error(422, f"missing field {req!r}")
                    if not body.get("dataset") and not body.get("dataset_id"):
                        return self._error(422, "provide 'dataset' rows or a 'dataset_id'")
                    self._send(202, {"job_id": server.submit(body)})
                elif parts == ["v1", "models", "download"]:
                    b = self._body()
                    if not b.get("model"):
                        return self._error(422, "provide a 'model' id")
                    self._send(202, server.start_download(b["model"]))
                elif parts == ["v1", "settings"]:
                    b = self._body()
                    server.set_hf_token((b.get("hf_token") or "").strip() or None)
                    from . import hub as _hub  # noqa: PLC0415
                    self._send(200, {"hf_token_set": _hub.has_token()})
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
                        self._send(200, {"ok": True})
                elif parts == ["v1", "generate"]:
                    b = self._body()
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

    api_key = os.environ.get("SHADOWLM_API_KEY")
    work_root = Path(args.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    server = Server(backend=args.backend, accelerator=args.accelerator,
                    device=args.device, work_root=work_root)
    httpd = ThreadingHTTPServer((args.host, args.port),
                                make_handler(server, api_key))

    static = Path(__file__).parent / "_static" / "index.html"
    ui = ("React studio" if static.exists() else "built-in dashboard (no-build)")
    auth = "Bearer auth ON" if api_key else "no auth (set SHADOWLM_API_KEY)"
    base = f"http://{args.host}:{args.port}"
    print(f"slm♥ ShadowLM server · {base} · backend={args.backend} · {auth}",
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
