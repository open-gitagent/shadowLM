"""download_artifact must not let a served tarball write outside its dest dir.

The archive comes over plain HTTP from whatever SHADOWLM_API_URL names, so a
malicious or MITM'd server controls every member name. On Python < 3.12 the
`filter="data"` kwarg doesn't exist and the fallback used to extract unfiltered.
"""

from __future__ import annotations

import io
import tarfile

import pytest

from shadowlm.remote import RemoteClient, RemoteError


def _client_serving(blob: bytes) -> RemoteClient:
    client = RemoteClient(api_url="http://test.invalid")
    client._request = lambda *a, **k: blob  # type: ignore[method-assign]
    return client


def _tar_bytes(*members: tuple[tarfile.TarInfo, bytes | None]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for info, data in members:
            if data is not None:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            else:
                tar.addfile(info)
    return buf.getvalue()


def _file(name: str, data: bytes = b"x") -> tuple[tarfile.TarInfo, bytes]:
    return tarfile.TarInfo(name=name), data


def test_benign_archive_extracts(tmp_path):
    dest = tmp_path / "out"
    blob = _tar_bytes(_file("adapter_config.json", b"{}"),
                      _file("weights/adapter.safetensors", b"w"))
    _client_serving(blob).download_artifact("job1", dest)
    assert (dest / "adapter_config.json").read_bytes() == b"{}"
    assert (dest / "weights" / "adapter.safetensors").read_bytes() == b"w"


def test_dotdot_member_rejected(tmp_path):
    dest = tmp_path / "out"
    blob = _tar_bytes(_file("../evil.txt"))
    with pytest.raises(RemoteError):
        _client_serving(blob).download_artifact("job1", dest)
    assert not (tmp_path / "evil.txt").exists()


def test_absolute_member_rejected(tmp_path):
    dest = tmp_path / "out"
    victim = tmp_path / "victim.txt"
    blob = _tar_bytes(_file(str(victim)))
    with pytest.raises(RemoteError):
        _client_serving(blob).download_artifact("job1", dest)
    assert not victim.exists()


def test_symlink_escape_rejected(tmp_path):
    dest = tmp_path / "out"
    link = tarfile.TarInfo(name="link")
    link.type = tarfile.SYMTYPE
    link.linkname = "../../secret"
    blob = _tar_bytes((link, None))
    with pytest.raises(RemoteError):
        _client_serving(blob).download_artifact("job1", dest)


def test_special_member_rejected(tmp_path):
    dest = tmp_path / "out"
    fifo = tarfile.TarInfo(name="pipe")
    fifo.type = tarfile.FIFOTYPE
    blob = _tar_bytes((fifo, None))
    with pytest.raises(RemoteError):
        _client_serving(blob).download_artifact("job1", dest)
