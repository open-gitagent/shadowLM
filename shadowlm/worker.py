"""`shadowlm worker` — hire this machine out to a ShadowLM hub.

The studio runs somewhere with a public face; this machine (a MacBook, an
office GPU box) sits behind NAT. So the worker dials *out* and keeps one
websocket open to the hub: jobs arrive on it the instant the studio dispatches
them, logs/metrics stream back up it, and a cancel from the studio lands
mid-step. The hub never needs a route to this machine — the socket *is* the
bidirectional wire.

    shadowlm worker --hub https://studio.example.com --name macbook

Pure stdlib + the local backend, like everything else. The adapter itself is
shipped home over plain HTTP at the end (one blob, no need for framing).
"""

from __future__ import annotations

import io
import platform
import queue
import tarfile
import threading
import time
from pathlib import Path

from . import ws
from .remote import RemoteClient

_PUSH_EVERY_S = 1.0
_PING_EVERY_S = 45.0  # keeps NAT mappings warm; hub gives up at 90s of silence
_FINAL_PUSH_TRIES = 150  # ~5 min of retries — a terminal status must land


class _Link:
    """The persistent socket to the hub: owns connect/re-connect, fans incoming
    messages out (jobs → queue, cancels → flags), and serializes sends."""

    def __init__(self, client: RemoteClient, name: str, register: dict) -> None:
        self._client, self._name, self._register = client, name, register
        self.jobs: "queue.Queue[dict]" = queue.Queue()
        self.cancelled: set[str] = set()  # job_ids the hub told us to stop
        self._conn: ws.WSConn | None = None
        threading.Thread(target=self._run, daemon=True).start()

    def send(self, obj: dict) -> None:
        conn = self._conn
        if conn is None:
            raise ConnectionError("hub link is down")
        conn.send_json(obj)

    @property
    def connected(self) -> bool:
        return self._conn is not None

    def _run(self) -> None:
        while True:
            try:
                conn = ws.connect(self._client.api_url,
                                  f"/v1/workers/{self._name}/socket",
                                  api_key=self._client.api_key)
                conn.send_json({"type": "register", **self._register})
                self._conn = conn
                print(f"[worker:{self._name}] connected to "
                      f"{self._client.api_url} — waiting for jobs", flush=True)
                while True:
                    try:
                        msg = conn.recv_json(timeout=_PING_EVERY_S)
                    except TimeoutError:
                        conn.ping()  # idle is fine; prove we're alive
                        continue
                    if msg is None:
                        break  # clean close from the hub
                    if msg.get("type") == "job":
                        self.jobs.put(msg["job"])
                    elif msg.get("type") == "cancel":
                        self.cancelled.add(msg.get("job_id", ""))
            except (OSError, ConnectionError) as e:
                print(f"[worker:{self._name}] hub link lost ({e}) — "
                      "reconnecting in 5s", flush=True)
            self._conn = None
            time.sleep(5)


class _Uplink:
    """Buffers one job's console lines + metrics; pushes over the link about
    once a second. Send failures buffer and ride the next push — only a
    terminal status insists (retries until the link is back)."""

    def __init__(self, link: _Link, job_id: str) -> None:
        self._link, self._job = link, job_id
        self._logs: list[str] = []
        self._steps: list[dict] = []
        self._evals: list[dict] = []
        self._last_push = 0.0

    @property
    def cancelled(self) -> bool:
        return self._job in self._link.cancelled

    def log(self, line: str) -> None:
        self._logs.append(line)
        self._maybe_push()

    def step(self, m) -> None:
        self._steps.append(m.to_dict())
        self._maybe_push()

    def eval(self, m) -> None:
        self._evals.append(m.to_dict())
        self._maybe_push()

    def _maybe_push(self) -> None:
        if time.monotonic() - self._last_push >= _PUSH_EVERY_S:
            self.push()

    def push(self, **final) -> None:
        events = {"type": "events", "job_id": self._job, "logs": self._logs,
                  "steps": self._steps, "evals": self._evals, **final}
        self._logs, self._steps, self._evals = [], [], []
        for attempt in range(_FINAL_PUSH_TRIES if final else 1):
            try:
                self._link.send(events)
                self._last_push = time.monotonic()
                return
            except (OSError, ConnectionError):
                if not final:  # buffer; the next cadence push carries it all
                    self._logs = events["logs"] + self._logs
                    self._steps = events["steps"] + self._steps
                    self._evals = events["evals"] + self._evals
                    self._last_push = time.monotonic()  # don't hot-loop retries
                    return
                time.sleep(2)  # terminal: wait for _Link to reconnect
        raise ConnectionError(f"could not report job {self._job} final status")


def _tar_dir(root: str | Path) -> bytes:
    buf = io.BytesIO()
    root = Path(root)
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                tar.add(p, arcname=str(p.relative_to(root)))
    return buf.getvalue()


def _run_job(link: _Link, client: RemoteClient, name: str, job: dict,
             work_root: Path, *, backend_factory=None) -> None:
    """Execute one hub job on the local backend, streaming everything up."""
    from .backends import Callbacks, select_backend  # noqa: PLC0415
    from .models import _eval_holdout  # noqa: PLC0415
    from .serve import _rebuild_config, _rebuild_dataset  # noqa: PLC0415

    job_id = job["job_id"]
    up = _Uplink(link, job_id)
    if up.cancelled:  # cancelled while queued locally
        up.push(status="stopped")
        return
    try:
        dataset = _rebuild_dataset(job["dataset"])
        eval_ds = None
        holdout = _eval_holdout(job.get("eval_dataset"))
        if holdout is not None:
            dataset, eval_ds = dataset.split(test_size=holdout)
        elif isinstance(job.get("eval_dataset"), dict):
            eval_ds = _rebuild_dataset(job["eval_dataset"])

        be = (backend_factory or (lambda: select_backend("auto")))()
        up.log(f"[worker:{name}] picked up job {job_id} · backend {be.name}")
        be.load(job["base_model"],
                load_in_4bit=job.get("load_in_4bit", False),
                max_seq_length=job.get("max_seq_length", 2048))
        out_dir = work_root / job_id
        result = be.finetune(
            dataset, _rebuild_config(job.get("config") or {}),
            Callbacks(on_step=up.step, on_eval=up.eval, on_log=up.log,
                      should_stop=lambda: up.cancelled),
            str(out_dir), eval_dataset=eval_ds)
        if up.cancelled:
            up.push(status="stopped")
            return
        client.upload_artifact(name, job_id, _tar_dir(result.checkpoint))
        up.push(status="succeeded", final_loss=result.final_loss)
    except KeyboardInterrupt:
        up.push(status="stopped")
    except Exception as e:  # noqa: BLE001 — job isolation: report, keep serving
        up.push(status="failed", error=f"{type(e).__name__}: {e}")


def run_worker(hub: str | None = None, *, name: str | None = None,
               api_key: str | None = None, work_root: str | Path | None = None,
               once: bool = False, backend_factory=None) -> None:
    """Keep one socket open to the hub and train whatever it sends, forever.

    `once=True` processes a single job then returns (tests, one-shot runs).
    """
    from .backends import select_backend  # noqa: PLC0415
    from .serve import _gpu_count  # noqa: PLC0415

    client = RemoteClient(hub, api_key)
    name = name or platform.node().split(".")[0] or "worker"
    work_root = Path(work_root or Path.home() / ".shadowlm" / "worker")
    work_root.mkdir(parents=True, exist_ok=True)

    be_name = select_backend("auto").name  # what this machine trains with
    link = _Link(client, name, {
        "backend": be_name,
        "device": f"{platform.system()}/{platform.machine()}",
        "gpus": _gpu_count()})

    while True:
        job = link.jobs.get()
        _run_job(link, client, name, job, work_root,
                 backend_factory=backend_factory)
        print(f"[worker:{name}] job {job['job_id']} finished — back to waiting",
              flush=True)
        if once:
            return
