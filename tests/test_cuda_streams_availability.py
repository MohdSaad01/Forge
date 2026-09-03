"""Milestone 27: CUDA-unavailable behavior for the public stream API.

Split out from `tests/test_cuda_streams.py` deliberately: that file carries a
module-level `pytestmark` skipping every test when CUDA is unavailable (the
convention every `test_cuda_*.py` file uses), which would also skip this one
-- the one that specifically exercises the "CUDA is not available" path
itself via a monkeypatched `is_cuda_available`, and so must run on *any*
machine, CUDA or not, to mean anything. Mirrors
`tests/test_cuda_synchronize_availability.py` exactly.
"""

from __future__ import annotations

import pytest

import forge
from forge.exceptions import CUDAError


def test_stream_construction_raises_cuda_error_when_unavailable(monkeypatch):
    monkeypatch.setattr("forge.backend.cuda.backend.is_cuda_available", lambda: False)
    with pytest.raises(CUDAError):
        forge.cuda.Stream()


def test_current_stream_raises_cuda_error_when_unavailable(monkeypatch):
    monkeypatch.setattr("forge.backend.cuda.backend.is_cuda_available", lambda: False)
    with pytest.raises(CUDAError):
        forge.cuda.current_stream()


def test_set_stream_raises_cuda_error_when_unavailable(monkeypatch):
    monkeypatch.setattr("forge.backend.cuda.backend.is_cuda_available", lambda: False)
    with pytest.raises(CUDAError):
        forge.cuda.set_stream(None)


def test_stream_context_manager_raises_cuda_error_when_unavailable(monkeypatch):
    monkeypatch.setattr("forge.backend.cuda.backend.is_cuda_available", lambda: False)
    with pytest.raises(CUDAError):
        with forge.cuda.stream(None):
            pass


def test_import_forge_without_cuda_does_not_require_cuda(monkeypatch):
    """Importing `forge` (and `forge.cuda`) must never itself require a working CUDA backend.

    `forge` is already imported by the time this test runs (module import at
    the top of this file), so this proves the *attributes* are reachable and
    merely *calling* them is what requires CUDA -- matching every other
    CUDA-specific entry point in Forge.
    """
    for name in ("Stream", "current_stream", "set_stream", "stream"):
        assert hasattr(forge.cuda, name)
        assert callable(getattr(forge.cuda, name))
