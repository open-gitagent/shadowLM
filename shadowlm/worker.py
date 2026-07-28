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
import re
import tarfile
import threading
import time
from pathlib import Path

from . import ws
from .remote import RemoteClient

_PUSH_EVERY_S = 1.0
_PING_EVERY_S = 45.0  # keeps NAT mappings warm; hub gives up at 90s of silence
_FINAL_PUSH_TRIES = 150  # ~5 min of retries — a terminal status must land

_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _job_dir(work_root: Path, job_id: str) -> Path:
    """`work_root / job_id`, refusing any id that isn't a plain path segment.
    The id arrives over the hub's websocket — the worker trusts the hub to
    schedule work, not to name paths on this machine."""
    if not _JOB_ID.fullmatch(job_id or ""):
        raise ValueError(f"refusing job_id {job_id!r}: not a plain id")
    return work_root / job_id


class _Link:
    """The persistent socket to the hub: owns connect/re-connect, fans incoming
    messages out (jobs → queue, cancels → flags), and serializes sends."""

    def __init__(self, client: RemoteClient, name: str, register: dict,
                 work_root: Path) -> None:
        self._client, self._name, self._register = client, name, register
        self._work_root = work_root
        self.jobs: "queue.Queue[dict]" = queue.Queue()
        self.cancelled: set[str] = set()  # job_ids the hub told us to stop
        self._conn: ws.WSConn | None = None
        self._infer_cache: dict[tuple, object] = {}  # (model, adapter) → Model
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
                conn.trace = "hub"  # every frame → this terminal
                # models are rescanned per connect — a reconnect refreshes them
                conn.send_json({"type": "register", **self._register,
                                "models": _local_models()})
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
                    elif msg.get("type") in ("chat", "generate"):
                        # the hub is proxying playground traffic to the shadow
                        # trained here — answer off-thread, recv keeps flowing
                        threading.Thread(target=self._serve_infer,
                                         args=(msg,), daemon=True).start()
            except (OSError, ConnectionError) as e:
                print(f"[worker:{self._name}] hub link lost ({e}) — "
                      "reconnecting in 5s", flush=True)
            self._conn = None
            time.sleep(5)

    def _serve_infer(self, msg: dict) -> None:
        """Answer a hub-proxied chat/generate with the shadow trained here."""
        out = {"type": "infer_result", "id": msg.get("id", "")}
        try:
            from .models import load  # noqa: PLC0415

            adapter = _job_dir(self._work_root, msg.get("job_id") or "")
            if not adapter.is_dir():
                raise FileNotFoundError(
                    f"adapter for job {msg.get('job_id')} is not on this "
                    "machine anymore (looked in "
                    f"{self._work_root})")
            key = (msg["base_model"], adapter.name)
            model = self._infer_cache.get(key)
            if model is None:
                if len(self._infer_cache) >= 2:  # tiny cache: base ↔ shadow flips
                    self._infer_cache.clear()
                model = load(msg["base_model"], adapter=str(adapter))
                self._infer_cache[key] = model
            if msg["type"] == "chat":
                reply = model.chat(msg["messages"],
                                   max_new_tokens=msg.get("max_new_tokens", 512),
                                   temperature=msg.get("temperature", 0.7),
                                   top_p=msg.get("top_p", 0.95))
                out["text"] = getattr(reply, "raw", None) or str(reply)
            else:
                out["text"] = model.generate(
                    msg["prompt"],
                    max_new_tokens=msg.get("max_new_tokens", 256),
                    temperature=msg.get("temperature", 0.7),
                    top_p=msg.get("top_p", 0.95))
            print(f"[worker] answered a {msg['type']} for the studio "
                  f"({len(out['text'])} chars)", flush=True)
        except Exception as e:  # noqa: BLE001 — report to the hub, keep serving
            out["error"] = f"{type(e).__name__}: {e}"
        try:
            self.send(out)
        except (OSError, ConnectionError):
            pass  # link dropped; the hub's waiter times out and says so


class _Uplink:
    """Buffers one job's console lines + metrics; pushes over the link about
    once a second, and echoes everything to this terminal — the worker's
    console reads like a local training session, not a black box. Send
    failures buffer and ride the next push — only a terminal status insists
    (retries until the link is back)."""

    def __init__(self, link: _Link, job_id: str) -> None:
        self._link, self._job = link, job_id
        self._tag = job_id[:8]
        self._logs: list[str] = []
        self._steps: list[dict] = []
        self._evals: list[dict] = []
        self._last_push = 0.0

    @property
    def cancelled(self) -> bool:
        return self._job in self._link.cancelled

    def log(self, line: str) -> None:
        print(line, flush=True)
        self._logs.append(line)
        self._maybe_push()

    def step(self, m) -> None:
        bits = [f"step {m.step}"]
        if m.loss is not None:
            bits.append(f"loss {m.loss:.4f}")
        if m.lr:
            bits.append(f"lr {m.lr:.2e}")
        if getattr(m, "tokens_per_s", None):
            bits.append(f"{m.tokens_per_s:,.0f} tok/s")
        print(f"[{self._tag}] {' · '.join(bits)}", flush=True)
        self._steps.append(m.to_dict())
        self._maybe_push()

    def eval(self, m) -> None:
        if m.loss is not None:
            print(f"[{self._tag}] eval · step {m.step} · loss {m.loss:.4f}",
                  flush=True)
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


def _hardware() -> dict:
    """What this machine brings: GPU name + memory, cores, RAM.

    CUDA reports the card and its VRAM; Apple silicon reports the chip and its
    unified memory (the GPU addresses all of it); anything else reports cores
    and RAM. Best-effort — a field we can't read is just its zero value.
    """
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    hw = {"gpus": 0, "gpu_name": "", "vram_gb": 0.0, "ram_gb": 0.0,
          "cores": os.cpu_count() or 0}
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            hw["gpus"] = torch.cuda.device_count()
            hw["gpu_name"] = torch.cuda.get_device_name(0)
            hw["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception:  # noqa: BLE001 — no torch / no driver: not a GPU box
        pass
    if platform.system() == "Darwin":
        try:
            hw["ram_gb"] = round(int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True)) / 2**30)
            if platform.machine() == "arm64" and not hw["gpu_name"]:
                hw["gpus"] = 1  # the integrated GPU mlx trains on
                hw["gpu_name"] = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    text=True).strip()
                hw["vram_gb"] = float(hw["ram_gb"])  # unified memory
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    else:
        try:
            hw["ram_gb"] = round(
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)
        except (ValueError, OSError, AttributeError):
            pass
    return hw


def _local_models() -> list[dict]:
    """Base models already on this machine (the HF cache) — the studio shows
    them per-machine so you know what a device can train without downloading."""
    try:
        from huggingface_hub import scan_cache_dir  # noqa: PLC0415

        repos = [{"id": r.repo_id, "size_gb": round(r.size_on_disk / 2**30, 1)}
                 for r in scan_cache_dir().repos if r.repo_type == "model"]
        return sorted(repos, key=lambda m: -m["size_gb"])[:50]
    except Exception:  # noqa: BLE001 — no hub lib / no cache: just report none
        return []


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
    from .ascii import _HEART, _NAME  # noqa: PLC0415

    print(_HEART)
    print(_NAME)
    print("Starting training session...\n", flush=True)
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
        out_dir = _job_dir(work_root, job_id)
        result = be.finetune(
            dataset, _rebuild_config(job.get("config") or {}),
            Callbacks(on_step=up.step, on_eval=up.eval, on_log=up.log,
                      should_stop=lambda: up.cancelled),
            str(out_dir), eval_dataset=eval_ds)
        if up.cancelled:
            print(f"[{job_id[:8]}] stopped by the studio", flush=True)
            up.push(status="stopped")
            return
        client.upload_artifact(name, job_id, _tar_dir(result.checkpoint))
        up.push(status="succeeded", final_loss=result.final_loss)
    except KeyboardInterrupt:
        up.push(status="stopped")
    except Exception as e:  # noqa: BLE001 — job isolation: report, keep serving
        print(f"[{job_id[:8]}] FAILED: {type(e).__name__}: {e}", flush=True)
        up.push(status="failed", error=f"{type(e).__name__}: {e}")


def run_worker(hub: str | None = None, *, name: str | None = None,
               api_key: str | None = None, work_root: str | Path | None = None,
               once: bool = False, backend_factory=None) -> None:
    """Keep one socket open to the hub and train whatever it sends, forever.

    `once=True` processes a single job then returns (tests, one-shot runs).
    """
    from .backends import select_backend  # noqa: PLC0415

    client = RemoteClient(hub, api_key)
    name = name or platform.node().split(".")[0] or "worker"
    work_root = Path(work_root or Path.home() / ".shadowlm" / "worker")
    work_root.mkdir(parents=True, exist_ok=True)

    be_name = select_backend("auto").name  # what this machine trains with
    link = _Link(client, name, {
        "backend": be_name,
        "device": f"{platform.system()}/{platform.machine()}",
        **_hardware()}, work_root)

    while True:
        job = link.jobs.get()
        _run_job(link, client, name, job, work_root,
                 backend_factory=backend_factory)
        print(f"[worker:{name}] job {job['job_id']} finished — back to waiting",
              flush=True)
        if once:
            return
