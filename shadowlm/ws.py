"""A minimal WebSocket (RFC 6455) — just enough for the hub↔worker wire.

Pure stdlib, like the rest of the transport. Text frames carry JSON messages;
ping/pong keeps NAT mappings alive and doubles as the liveness signal. We speak
only to ourselves, so the corners of the RFC we skip are marked:

ponytail: no fragmented frames (we always send fin=1 and our messages are
small), no extensions, no subprotocols — add if a third-party client ever
needs to connect.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import ssl
import struct
import threading
import urllib.parse

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"  # fixed by RFC 6455

OP_TEXT, OP_CLOSE, OP_PING, OP_PONG = 0x1, 0x8, 0x9, 0xA


def accept_key(client_key: str) -> str:
    """The Sec-WebSocket-Accept value proving the server speaks WebSocket."""
    digest = hashlib.sha1((client_key + _GUID).encode()).digest()  # noqa: S324 — RFC-mandated
    return base64.b64encode(digest).decode()


def _read_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("websocket peer closed")
        buf += chunk
    return buf


def send_frame(sock: socket.socket, payload: bytes, *, opcode: int = OP_TEXT,
               mask: bool = False) -> None:
    head = bytes([0x80 | opcode])  # fin=1
    n = len(payload)
    mask_bit = 0x80 if mask else 0
    if n < 126:
        head += bytes([mask_bit | n])
    elif n < 1 << 16:
        head += bytes([mask_bit | 126]) + struct.pack(">H", n)
    else:
        head += bytes([mask_bit | 127]) + struct.pack(">Q", n)
    if mask:  # clients MUST mask (RFC 6455 §5.3)
        key = os.urandom(4)
        payload = bytes(b ^ key[i % 4] for i, b in enumerate(payload))
        head += key
    sock.sendall(head + payload)


def recv_frame(sock: socket.socket) -> tuple[int, bytes]:
    b1, b2 = _read_exact(sock, 2)
    opcode = b1 & 0x0F
    masked, n = b2 & 0x80, b2 & 0x7F
    if n == 126:
        (n,) = struct.unpack(">H", _read_exact(sock, 2))
    elif n == 127:
        (n,) = struct.unpack(">Q", _read_exact(sock, 8))
    key = _read_exact(sock, 4) if masked else None
    payload = _read_exact(sock, n)
    if key:
        payload = bytes(b ^ key[i % 4] for i, b in enumerate(payload))
    return opcode, payload


def _describe(obj: dict) -> str:
    """One line per wire message — the socket narrates its own traffic."""
    kind = obj.get("type", "?")
    bits = [kind]
    if obj.get("job_id"):
        bits.append(str(obj["job_id"])[:8])
    if kind == "job" and isinstance(obj.get("job"), dict):
        j = obj["job"]
        bits.append(str(j.get("job_id", ""))[:8])
        rows = len((j.get("dataset") or {}).get("rows") or [])
        bits.append(f"{j.get('base_model', '?')} · {rows} rows")
    elif kind == "events":
        n_s, n_l = len(obj.get("steps") or []), len(obj.get("logs") or [])
        if n_s:
            bits.append(f"{n_s} steps")
        if n_l:
            bits.append(f"{n_l} log lines")
        if obj.get("status"):
            bits.append(f"status={obj['status']}")
    elif kind == "register":
        bits.append(f"{obj.get('backend', '?')} · {obj.get('gpu_name') or obj.get('device', '?')}"
                    f" · {len(obj.get('models') or [])} models")
    elif kind == "infer_result":
        bits.append(f"{len(obj.get('text') or '')} chars"
                    if not obj.get("error") else f"error: {obj['error'][:60]}")
    elif kind in ("chat", "generate"):
        bits.append(f"→ answer on this wire, id {str(obj.get('id', ''))[:8]}")
    return " · ".join(bits)


class WSConn:
    """A connected websocket: thread-safe JSON sends, ping/pong handled inline.

    `is_client` decides masking (clients mask, servers don't). Set `trace` to a
    label ("ws:patel") and every frame — messages, pings, close — prints as a
    one-liner, so the socket's whole conversation is visible in the console.
    """

    def __init__(self, sock: socket.socket, *, is_client: bool) -> None:
        self._sock = sock
        self._mask = is_client
        self._send_lock = threading.Lock()
        self.trace: str | None = None

    def _log(self, arrow: str, what: str) -> None:
        if self.trace:
            print(f"[{self.trace}] {arrow} {what}", flush=True)

    def send_json(self, obj: dict) -> None:
        self._log("→", _describe(obj))
        with self._send_lock:
            send_frame(self._sock, json.dumps(obj).encode(), mask=self._mask)

    def recv_json(self, *, timeout: float | None = None) -> dict | None:
        """The next JSON message; None on clean close. Answers pings itself.
        Raises `socket.timeout` if `timeout` elapses with no frame."""
        self._sock.settimeout(timeout)
        while True:
            opcode, payload = recv_frame(self._sock)
            if opcode == OP_TEXT:
                obj = json.loads(payload)
                self._log("←", _describe(obj))
                return obj
            if opcode == OP_PING:
                self._log("←", "ping (answered)")
                with self._send_lock:
                    send_frame(self._sock, payload, opcode=OP_PONG,
                               mask=self._mask)
            elif opcode == OP_CLOSE:
                self._log("←", "close")
                return None
            # OP_PONG and anything else: liveness noise, keep reading

    def ping(self) -> None:
        self._log("→", "ping")
        with self._send_lock:
            send_frame(self._sock, b"", opcode=OP_PING, mask=self._mask)

    def close(self) -> None:
        try:
            with self._send_lock:
                send_frame(self._sock, b"", opcode=OP_CLOSE, mask=self._mask)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass


def connect(url: str, path: str, *, api_key: str | None = None,
            timeout: float = 30.0) -> WSConn:
    """Open a client websocket to `path` on the server at `url` (http/https)."""
    u = urllib.parse.urlparse(url)
    host = u.hostname or "127.0.0.1"
    port = u.port or (443 if u.scheme == "https" else 80)
    sock = socket.create_connection((host, port), timeout=timeout)
    if u.scheme == "https":
        sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
    key = base64.b64encode(os.urandom(16)).decode()
    headers = [f"GET {path} HTTP/1.1", f"Host: {host}:{port}",
               "Upgrade: websocket", "Connection: Upgrade",
               f"Sec-WebSocket-Key: {key}", "Sec-WebSocket-Version: 13",
               "User-Agent: shadowlm-ws"]
    if api_key:
        headers.append(f"Authorization: Bearer {api_key}")
    sock.sendall(("\r\n".join(headers) + "\r\n\r\n").encode())

    # read the 101 response (headers end at the blank line)
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("server closed during websocket handshake")
        buf += chunk
    status = buf.split(b"\r\n", 1)[0].decode()
    if " 101 " not in f"{status} ":
        raise ConnectionError(f"websocket handshake refused: {status.strip()}")
    lower = buf.lower()
    expect = accept_key(key).encode().lower()
    if b"sec-websocket-accept: " + expect not in lower:
        raise ConnectionError("websocket handshake: bad Sec-WebSocket-Accept")
    return WSConn(sock, is_client=True)
