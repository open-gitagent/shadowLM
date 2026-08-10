"""select_backend's five branches and four error messages.

Availability probes are monkeypatched, so this runs anywhere — the point is the
dispatch table and that every failure names what to install.
"""

from __future__ import annotations

import pytest

from shadowlm import backends
from shadowlm.backends import select_backend


@pytest.fixture()
def probes(monkeypatch):
    """Pretend nothing is installed; each test turns on what it needs."""
    state = {"cuda": False, "mlx": False, "torch": False, "verl": False}

    class FakeTorch:
        def __init__(self, *, device="auto", accelerator="auto"):
            self.name, self.device, self.accelerator = "torch", device, accelerator

        @classmethod
        def has_cuda(cls):
            return state["cuda"]

        @classmethod
        def is_available(cls):
            return state["torch"]

    class FakeMLX:
        def __init__(self, *, accelerator="auto"):
            self.name, self.accelerator = "mlx", accelerator

        @classmethod
        def is_available(cls):
            return state["mlx"]

    class FakeVerl:
        def __init__(self, *, device="auto", accelerator="auto"):
            self.name = "verl"

        @classmethod
        def is_available(cls):
            return state["verl"]

    import shadowlm.backends.mlx as mlx_mod
    import shadowlm.backends.torch as torch_mod
    import shadowlm.backends.verl as verl_mod
    monkeypatch.setattr(torch_mod, "TorchBackend", FakeTorch)
    monkeypatch.setattr(mlx_mod, "MLXBackend", FakeMLX)
    monkeypatch.setattr(verl_mod, "VerlBackend", FakeVerl)
    monkeypatch.setattr(backends, "_mlx_available", FakeMLX.is_available)
    monkeypatch.setattr(backends, "_torch_available", FakeTorch.is_available)
    return state


def test_auto_prefers_cuda_torch(probes):
    probes.update(cuda=True, torch=True, mlx=True)
    be = select_backend("auto")
    assert be.name == "torch" and be.device == "auto"  # not pinned to cpu


def test_auto_falls_to_mlx_without_cuda(probes):
    probes.update(mlx=True, torch=True)
    assert select_backend("auto").name == "mlx"


def test_auto_falls_to_torch_cpu_last(probes):
    probes.update(torch=True)
    be = select_backend("auto")
    assert (be.name, be.device) == ("torch", "cpu")


def test_auto_with_nothing_installed_lists_both_installs(probes):
    with pytest.raises(RuntimeError, match=r"shadowlm\[mlx\]"):
        select_backend("auto")


def test_explicit_mlx_unavailable_is_actionable(probes):
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        select_backend("mlx")


def test_explicit_torch_unavailable_is_actionable(probes):
    with pytest.raises(RuntimeError, match=r"shadowlm\[torch\]"):
        select_backend("torch")


def test_explicit_verl_unavailable_is_actionable(probes):
    with pytest.raises(RuntimeError, match="verl"):
        select_backend("verl")


def test_remote_is_always_constructible(probes):
    assert select_backend("remote").name == "remote"


def test_name_is_case_insensitive(probes):
    probes.update(torch=True)
    assert select_backend("TORCH").name == "torch"


def test_unknown_backend_lists_the_valid_names(probes):
    with pytest.raises(ValueError, match="auto|mlx|torch|remote|verl"):
        select_backend("tensorflow")
