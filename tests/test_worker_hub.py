"""The hub↔worker websocket loop, end to end over a real socket — no model.

Worker-targeted jobs never touch the hub's own training queue, so a full
connect → register → dispatch → events → cancel → artifact round-trip runs on
CPU in milliseconds while exercising the actual RFC 6455 wire both ways.
"""

import socket
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from shadowlm import ws
from shadowlm.remote import RemoteClient
from shadowlm.serve import Auth, Server, make_handler
from shadowlm.worker import _Link, _run_job, run_worker

ROWS = [{"instruction": "2+2?", "output": "4"},
        {"instruction": "capital of France?", "output": "Paris"}]


# ---- the frames themselves ---------------------------------------------------
@pytest.mark.parametrize("size", [0, 5, 125, 126, 70_000])
@pytest.mark.parametrize("mask", [True, False])
def test_frame_roundtrip_all_length_encodings(size, mask):
    a, b = socket.socketpair()
    try:
        payload = bytes(i % 251 for i in range(size))
        # send from a thread: a 70KB frame overflows the socketpair buffer and
        # sendall would deadlock against recv on the same thread
        sender = threading.Thread(target=ws.send_frame, args=(a, payload),
                                  kwargs={"mask": mask})
        sender.start()
        opcode, got = ws.recv_frame(b)
        sender.join(timeout=5)
        assert (opcode, got) == (ws.OP_TEXT, payload)
    finally:
        a.close()
        b.close()


def test_handshake_accept_key_is_rfc6455_exact():
    # the worked example straight from the RFC
    assert ws.accept_key("dGhlIHNhbXBsZSBub25jZQ==") == \
        "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


# ---- the hub↔worker session ---------------------------------------------------
@pytest.fixture()
def hub(tmp_path):
    server = Server(backend="auto", accelerator="auto", device="auto",
                    work_root=tmp_path)
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(server, Auth(user="admin", password=None, api_key=None)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield server, RemoteClient(url), url
    httpd.shutdown()


def _submit(client, worker="mac"):
    return client.submit_finetune(
        base_model="tiny/model", config={"method": "lora", "max_steps": 2},
        dataset={"rows": ROWS, "format": "instruction"}, eval_dataset=None,
        load_in_4bit=False, max_seq_length=512, worker=worker)


def _wait_terminal(server, job_id, timeout=10.0):
    """The worker returns before the hub's recv thread ingests its final frame —
    give the ingest a moment instead of asserting against the race."""
    deadline = time.time() + timeout
    while server.jobs[job_id].status in ("pending", "running") \
            and time.time() < deadline:
        time.sleep(0.05)
    return server.jobs[job_id]


def _connect(url, name="mac"):
    conn = ws.connect(url, f"/v1/workers/{name}/socket")
    conn.send_json({"type": "register", "backend": "mlx",
                    "device": "Darwin/arm64", "gpus": 1,
                    "gpu_name": "Test M-chip", "vram_gb": 48.0,
                    "ram_gb": 48.0, "cores": 12})
    return conn


def test_register_dispatch_events_over_one_socket(hub):
    server, client, url = hub
    conn = _connect(url)
    try:
        deadline = time.time() + 5
        while not client.workers() and time.time() < deadline:
            time.sleep(0.05)
        assert client.workers()[0]["online"]

        job_id = _submit(client)
        msg = conn.recv_json(timeout=5.0)  # pushed, not polled
        assert msg["type"] == "job" and msg["job"]["job_id"] == job_id
        assert msg["job"]["dataset"]["rows"] == ROWS
        assert server.jobs[job_id].status == "running"

        conn.send_json({"type": "events", "job_id": job_id,
                        "logs": ["hello from the mac"],
                        "steps": [{"step": 1, "loss": 2.0}], "evals": []})
        conn.send_json({"type": "events", "job_id": job_id, "logs": [],
                        "steps": [], "evals": [],
                        "status": "succeeded", "final_loss": 1.5})
        deadline = time.time() + 5
        while server.jobs[job_id].status != "succeeded" and time.time() < deadline:
            time.sleep(0.05)
        j = server.jobs[job_id]
        assert (j.status, j.final_loss) == ("succeeded", 1.5)
        assert "hello from the mac" in j.logs
        assert j.steps == [{"step": 1, "loss": 2.0}]
    finally:
        conn.close()


def test_cancel_is_pushed_down_the_socket(hub):
    server, client, url = hub
    conn = _connect(url)
    try:
        job_id = _submit(client)
        assert conn.recv_json(timeout=5.0)["type"] == "job"
        client.cancel(job_id)
        msg = conn.recv_json(timeout=5.0)  # arrives unprompted — no poll
        assert msg == {"type": "cancel", "job_id": job_id}
    finally:
        conn.close()


def test_job_survives_a_dead_worker_socket(hub):
    """Dispatch to a vanished socket must re-queue, not strand the job."""
    server, client, url = hub
    conn = _connect(url)
    while not server.worker_socks:
        time.sleep(0.02)
    conn._sock.close()  # yank the transport out from under the hub
    job_id = _submit(client)
    deadline = time.time() + 10
    while time.time() < deadline:
        j = server.jobs[job_id]
        if j.status == "pending" and not server.worker_socks:
            break
        time.sleep(0.05)
    assert server.jobs[job_id].status == "pending"  # waiting for a reconnect


def test_full_worker_loop_with_stub_backend(hub, tmp_path):
    """run_worker(once=True) against the live hub: the real _Link, the real
    socket, a stub in place of the training backend."""
    server, client, url = hub

    class StubBackend:
        name = "stub"

        def load(self, name, **kw):
            pass

        def finetune(self, dataset, config, callbacks, output_dir, **kw):
            from pathlib import Path

            from shadowlm.backends.base import FinetuneResult
            from shadowlm.training import Metric

            assert len(dataset.rows) == 2
            callbacks.log("training away")
            callbacks.step(Metric(step=1, loss=3.0))
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "adapter.bin").write_bytes(b"w8s")
            return FinetuneResult(checkpoint=str(out), final_loss=3.0)

    job_id = _submit(client, worker="stubmac")
    t = threading.Thread(target=run_worker, daemon=True, kwargs=dict(
        hub=url, name="stubmac", work_root=tmp_path / "wk", once=True,
        backend_factory=StubBackend))
    t.start()
    t.join(timeout=15)
    assert not t.is_alive(), "worker never finished the job"

    j = _wait_terminal(server, job_id)
    assert j.status == "succeeded" and j.final_loss == 3.0
    assert any("training away" in ln for ln in j.logs)
    assert j.checkpoint and j.checkpoint.endswith("artifact.tar.gz")

    dest = tmp_path / "home"
    client.download_artifact(job_id, dest)
    assert (dest / "adapter.bin").read_bytes() == b"w8s"


def test_worker_failure_is_reported_not_swallowed(hub, tmp_path):
    server, client, url = hub

    class ExplodingBackend:
        name = "boom"

        def load(self, name, **kw):
            raise RuntimeError("no such model")

    job_id = _submit(client, worker="boommac")
    t = threading.Thread(target=run_worker, daemon=True, kwargs=dict(
        hub=url, name="boommac", work_root=tmp_path / "wk", once=True,
        backend_factory=ExplodingBackend))
    t.start()
    t.join(timeout=15)
    j = _wait_terminal(server, job_id)
    assert j.status == "failed" and "no such model" in j.error


def test_playground_chat_routes_to_the_training_machine(hub):
    """A worker-trained shadow answers on its worker: the hub proxies /v1/chat
    over the socket instead of loading a foreign-format adapter itself."""
    server, client, url = hub
    conn = _connect(url)
    try:
        job_id = _submit(client)
        assert conn.recv_json(timeout=5.0)["type"] == "job"

        def fake_mac_answers():
            req = conn.recv_json(timeout=10.0)  # the proxied chat arrives
            assert req["type"] == "chat" and req["job_id"] == job_id
            assert req["messages"] == [{"role": "user", "content": "hi"}]
            conn.send_json({"type": "infer_result", "id": req["id"],
                            "text": "hello from the mac"})

        t = threading.Thread(target=fake_mac_answers, daemon=True)
        t.start()
        out = client._request("POST", "/v1/chat", {
            "model": "tiny/model", "adapter": job_id,
            "messages": [{"role": "user", "content": "hi"}]})
        t.join(timeout=5)
        assert out == {"text": "hello from the mac"}
    finally:
        conn.close()


def test_chat_with_offline_machine_says_which_machine(hub):
    server, client, url = hub
    job_id = _submit(client, worker="gone-mac")  # never connects
    with pytest.raises(Exception, match="gone-mac.*offline|offline.*gone-mac"):
        client._request("POST", "/v1/chat", {
            "model": "tiny/model", "adapter": job_id,
            "messages": [{"role": "user", "content": "hi"}]})


def test_old_job_records_recover_their_worker_from_logs():
    """job.json written before the worker field existed: the name comes back
    from the '[worker:X] picked up job' console line."""
    from shadowlm.serve import _Job

    job = _Job.from_record({
        "job_id": "old123", "base_model": "m", "status": "succeeded",
        "logs": ["banner", "[worker:patel] picked up job old123 · backend mlx"]})
    assert job.worker == "patel"


def test_machine_tokens_mint_authenticate_revoke(tmp_path):
    """Against an auth-enabled hub: a minted machine token opens the worker
    socket, survives a hub restart (tokens.json), and dies on revoke."""
    server = Server(backend="auto", accelerator="auto", device="auto",
                    work_root=tmp_path)
    auth = Auth(user="admin", password=None, api_key="ADMIN_KEY")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(server, auth))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        admin = RemoteClient(url, "ADMIN_KEY")
        with pytest.raises(Exception, match="401|auth"):
            RemoteClient(url, "slmk_forged").health()  # unknown token: rejected

        out = admin._request("POST", "/v1/tokens", {"name": "macbook"})
        token = out["token"]
        assert token.startswith("slmk_")
        assert RemoteClient(url, token).health()["ok"]  # HTTP accepts it

        conn = ws.connect(url, "/v1/workers/macbook/socket", api_key=token)
        conn.send_json({"type": "register", "backend": "mlx",
                        "device": "t", "gpus": 0})  # socket accepts it
        deadline = time.time() + 5
        while not admin.workers() and time.time() < deadline:
            time.sleep(0.05)
        assert admin.workers()[0]["name"] == "macbook"
        conn.close()

        # raw value is never stored — only the hash is on disk
        assert token not in (tmp_path / "tokens.json").read_text()
        # a fresh Server over the same work_root still honors it (persistence)
        assert Server(backend="auto", accelerator="auto", device="auto",
                      work_root=tmp_path).valid_machine_token(token)

        admin._request("DELETE", "/v1/tokens/macbook")
        with pytest.raises(Exception, match="401|auth"):
            RemoteClient(url, token).health()  # revoked: rejected
    finally:
        httpd.shutdown()


def test_cancelled_before_pickup_never_dispatches(hub):
    server, client, url = hub
    job_id = _submit(client, worker="ghost")  # no such worker connected
    client.cancel(job_id)
    assert server.next_job_for("ghost", wait_s=0.01) is None
    assert server.jobs[job_id].status == "stopped"
