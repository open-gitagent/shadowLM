"""Files holding credentials are written owner-only (0600), not umask-default.

settings.json carries the HF token in plaintext; tokens.json carries machine
tokens (hashed, but the names alone map the fleet). Neither belongs
world-readable on a shared box.
"""

from __future__ import annotations

import stat

from shadowlm.serve import Server


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_hf_token_file_is_owner_only(tmp_path):
    server = Server(backend="auto", accelerator="auto", device="auto",
                    work_root=tmp_path)
    server.set_hf_token("hf_secret")
    assert _mode(tmp_path / "settings.json") == 0o600


def test_machine_tokens_file_is_owner_only(tmp_path):
    server = Server(backend="auto", accelerator="auto", device="auto",
                    work_root=tmp_path)
    server.mint_machine_token("macbook")
    assert _mode(tmp_path / "tokens.json") == 0o600
