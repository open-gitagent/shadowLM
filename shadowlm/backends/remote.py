"""RemoteBackend — train on a ShadowLM server, drive it like a local backend.

    model = slm.load("Qwen/Qwen2.5-0.5B-Instruct", backend="remote")
    run = model.finetune(ds, method="lora")     # same five lines, remote GPUs

The job runs wherever SHADOWLM_API_URL points — `python -m shadowlm.serve` on
a GPU box, or ShadowLM Studio. Metrics stream back into the normal callback
machinery, so the live progress bar, sparkline, and local run records work
exactly as they do for local training. Pure stdlib, like the core.
"""

from __future__ import annotations

import time

from ..data import Dataset
from ..remote import RemoteClient, RemoteError
from ..training import Metric, TrainConfig
from .base import Backend, Callbacks, FinetuneResult

_TERMINAL = {"succeeded", "failed", "stopped"}
_POLL_S = 1.0


def _serialize(ds: Dataset | None) -> dict | None:
    if ds is None:
        return None
    return {"format": ds.format, "rows": ds.rows}


class RemoteBackend(Backend):
    name = "remote"

    def __init__(self, *, device: str = "auto", accelerator: str = "auto",
                 api_url: str | None = None, api_key: str | None = None) -> None:
        super().__init__(device=device, accelerator=accelerator)
        self._client = RemoteClient(api_url, api_key)
        self.adapter: str | None = None
        self._last_job: str | None = None

    @classmethod
    def is_available(cls) -> bool:
        return True  # availability is the server's problem, checked in load()

    # ---- lifecycle -----------------------------------------------------------
    def load(self, name: str, *, load_in_4bit: bool, max_seq_length: int,
             adapter: str | None = None, api_url: str | None = None,
             api_key: str | None = None, **kwargs) -> None:
        if api_url or api_key:  # load-time override of env/defaults
            self._client = RemoteClient(api_url or self._client.api_url,
                                        api_key or self._client.api_key)
        health = self._client.health()  # fail fast: reachable + authorized
        self.model_name = name
        self.load_in_4bit = load_in_4bit
        self.max_seq_length = max_seq_length
        self.adapter = adapter  # a remote job id, or a server-known adapter ref
        self._server_backend = health.get("backend", "?")

    # ---- training ------------------------------------------------------------
    def finetune(self, dataset: Dataset, config: TrainConfig, callbacks: Callbacks,
                 output_dir: str, eval_dataset: Dataset | None = None,
                 reward_fns: list | None = None) -> FinetuneResult:
        if reward_fns:
            raise RuntimeError(
                "the remote backend can't ship Python reward functions — run "
                "method='grpo' with reward_fns locally, or train on judge-scored "
                "trajectory groups (those serialize)."
            )
        job_id = self._client.submit_finetune(
            base_model=self.model_name,
            config=config.to_dict(),
            dataset=_serialize(dataset),
            eval_dataset=_serialize(eval_dataset),
            load_in_4bit=self.load_in_4bit,
            max_seq_length=self.max_seq_length,
        )
        self._last_job = job_id
        callbacks.log(f"[remote:{self._client.api_url}] job {job_id} submitted "
                      f"(server backend: {self._server_backend})")

        seen_steps = seen_evals = 0
        cancelled = False
        status: dict = {}
        while True:
            if callbacks.stopped() and not cancelled:
                self._client.cancel(job_id)
                callbacks.log(f"[remote] job {job_id} cancelled")
                cancelled = True
            m = self._client.metrics(job_id)
            for d in m.get("steps", [])[seen_steps:]:
                callbacks.step(Metric(**d))
                seen_steps += 1
            for d in m.get("evals", [])[seen_evals:]:
                callbacks.eval(Metric(**d))
                seen_evals += 1
            status = self._client.job(job_id)
            if status["status"] in _TERMINAL:
                break
            time.sleep(_POLL_S)

        if status["status"] == "failed":
            raise RuntimeError(f"remote training failed: {status.get('error')}")
        if status["status"] == "stopped":
            raise KeyboardInterrupt  # the Model layer records a stopped run
        # Bring the trained adapter home so load(adapter=) and runs/plot work
        # exactly as for local training.
        self._client.download_artifact(job_id, output_dir)
        self.adapter = job_id  # subsequent generate/chat use the trained weights
        callbacks.log(f"[remote] artifact → {output_dir}")
        return FinetuneResult(checkpoint=output_dir,
                              final_loss=status.get("final_loss"))

    # ---- inference -------------------------------------------------------------
    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float,
                 top_p: float, **kwargs) -> str:
        return self._client.generate(
            model=self.model_name, adapter=self.adapter, prompt=prompt,
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)

    def chat(self, messages: list[dict], *, tools: list[dict] | None = None,
             max_new_tokens: int, temperature: float, top_p: float, **kwargs) -> str:
        return self._client.chat(
            model=self.model_name, adapter=self.adapter, messages=messages,
            tools=tools, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p)

    def save(self, path: str, *, fmt: str = "adapter") -> str:
        if fmt != "adapter":
            raise RuntimeError("the remote backend exports fmt='adapter' only "
                               "(merge locally from the downloaded adapter)")
        if not self._last_job:
            raise RemoteError("nothing trained yet — finetune first")
        return self._client.download_artifact(self._last_job, path)
