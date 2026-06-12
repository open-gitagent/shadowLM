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


@dataclass
class _Job:
    job_id: str
    base_model: str
    status: str = "pending"  # pending | running | succeeded | failed | stopped
    steps: list[dict] = field(default_factory=list)
    evals: list[dict] = field(default_factory=list)
    error: str | None = None
    checkpoint: str | None = None
    final_loss: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)


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
    """Datasets uploaded through the dashboard — JSONL on disk, survives restarts."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, rows: list[dict]) -> dict:
        ds = Dataset.from_list(rows)  # validates + detects format
        ds_id = uuid.uuid4().hex[:10]
        (self.root / f"{ds_id}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows))
        meta = {"dataset_id": ds_id, "name": name or f"dataset-{ds_id[:4]}",
                "format": ds.format, "rows": len(rows),
                "created": int(time.time())}
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
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- training worker -----------------------------------------------------
    def _worker(self) -> None:
        from .backends import Callbacks, select_backend  # noqa: PLC0415

        while True:
            job_id = self.queue.get()
            job = self.jobs[job_id]
            if job.cancel.is_set():
                job.status = "stopped"
                continue
            job.status = "running"
            try:
                payload = job._payload  # attached at submit time
                # dataset by reference (dashboard) or inline rows (SDK)
                if payload.get("dataset_id"):
                    rows = self.datasets.rows(payload["dataset_id"])
                    if rows is None:
                        raise ValueError(f"unknown dataset {payload['dataset_id']!r}")
                    payload["dataset"] = {"rows": rows, "format": None}
                if payload.get("eval_dataset") == "auto":
                    full = _rebuild_dataset(payload["dataset"])
                    train, ev = full.split(test_size=0.1)
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

    # ---- inference -------------------------------------------------------------
    def _infer_model(self, model: str, adapter: str | None):
        """Load (and cache) a model for /generate and /chat."""
        from .models import load  # noqa: PLC0415

        adapter_path = None
        if adapter:
            job = self.jobs.get(adapter)
            if job and job.checkpoint:
                adapter_path = job.checkpoint
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
        job = _Job(job_id=job_id, base_model=payload["base_model"])
        job._payload = payload
        with self._lock:
            self.jobs[job_id] = job
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
            if parts == [""]:  # the dashboard — no auth for the shell page;
                from .webui import HTML  # noqa: PLC0415 — the API stays authed
                self._send(200, HTML.encode(), ctype="text/html; charset=utf-8")
                return
            if not self._authed():
                return
            if parts == ["v1", "health"]:
                self._send(200, {"ok": True, "backend": server.backend_name,
                                 "version": __version__})
            elif parts == ["v1", "finetunes"]:
                with server._lock:
                    jobs = [{
                        "job_id": j.job_id, "base_model": j.base_model,
                        "status": j.status, "error": j.error,
                        "final_loss": j.final_loss,
                        "steps": len(j.steps),
                        "method": (j._payload.get("config", {}) or {}).get("method"),
                    } for j in server.jobs.values()]
                self._send(200, {"jobs": jobs[::-1]})  # newest first
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
                recent = []
                with server._lock:
                    for j in server.jobs.values():
                        if j.base_model not in recent:
                            recent.append(j.base_model)
                self._send(200, {"catalog": _MODEL_CATALOG, "recent": recent,
                                 "server_backend": server.backend_name})
            elif parts == ["v1", "methods"]:
                from . import methods as _methods  # noqa: PLC0415
                self._send(200, {"methods": [{
                    "name": m.name, "description": m.description,
                    "default_lr": m.default_learning_rate, "trainer": m.trainer,
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
                elif parts == ["v1", "datasets"]:
                    body = self._body()
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
                        m = server._infer_model(b["model"], b.get("adapter"))
                        text = m.generate(
                            b["prompt"],
                            max_new_tokens=b.get("max_new_tokens", 256),
                            temperature=b.get("temperature", 0.7),
                            top_p=b.get("top_p", 0.95))
                    self._send(200, {"text": text})
                elif parts == ["v1", "chat"]:
                    b = self._body()
                    with server._model_lock:
                        m = server._infer_model(b["model"], b.get("adapter"))
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
    args = parser.parse_args(argv)

    api_key = os.environ.get("SHADOWLM_API_KEY")
    work_root = Path(args.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    server = Server(backend=args.backend, accelerator=args.accelerator,
                    device=args.device, work_root=work_root)
    httpd = ThreadingHTTPServer((args.host, args.port),
                                make_handler(server, api_key))
    auth = "Bearer auth ON" if api_key else "no auth (set SHADOWLM_API_KEY)"
    print(f"slm♥ ShadowLM server · http://{args.host}:{args.port} · "
          f"backend={args.backend} · {auth}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
