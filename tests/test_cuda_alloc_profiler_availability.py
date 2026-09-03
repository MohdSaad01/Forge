"""Milestone 24: `forge.cuda.profiler` availability-error tests.

Split out from `tests/test_cuda_alloc_profiler.py` deliberately, for the same
reason `test_cuda_memory_availability.py` is split from `test_cuda_memory.py`:
that file's module-level `pytestmark` skips every test when CUDA is
unavailable, which would also skip these -- the ones that specifically
exercise the "CUDA is not available" path itself via a monkeypatched
`is_cuda_available`, and so must run on *any* machine, CUDA or not.
"""

from __future__ import annotations

import pytest

import forge
from forge.exceptions import CUDAError


def _unavailable(monkeypatch):
    monkeypatch.setattr("forge.backend.cuda.backend.is_cuda_available", lambda: False)


def test_start_raises_cuda_error_when_unavailable(monkeypatch):
    _unavailable(monkeypatch)
    with pytest.raises(CUDAError):
        forge.cuda.profiler.start()


def test_stop_raises_cuda_error_when_unavailable(monkeypatch):
    _unavailable(monkeypatch)
    with pytest.raises(CUDAError):
        forge.cuda.profiler.stop()


def test_reset_raises_cuda_error_when_unavailable(monkeypatch):
    _unavailable(monkeypatch)
    with pytest.raises(CUDAError):
        forge.cuda.profiler.reset()


def test_is_active_raises_cuda_error_when_unavailable(monkeypatch):
    _unavailable(monkeypatch)
    with pytest.raises(CUDAError):
        forge.cuda.profiler.is_active()


def test_events_raises_cuda_error_when_unavailable(monkeypatch):
    _unavailable(monkeypatch)
    with pytest.raises(CUDAError):
        forge.cuda.profiler.events()


def test_tag_raises_cuda_error_when_unavailable(monkeypatch):
    _unavailable(monkeypatch)
    with pytest.raises(CUDAError):
        with forge.cuda.profiler.tag("x"):
            pass


def test_profile_context_manager_raises_cuda_error_when_unavailable(monkeypatch):
    _unavailable(monkeypatch)
    with pytest.raises(CUDAError):
        with forge.cuda.profiler.profile():
            pass


def test_importing_forge_cuda_profiler_never_requires_cuda():
    """Import alone (no CUDA calls) must succeed on any machine -- exercised
    implicitly by every test above already importing `forge`, but asserted
    directly here per the milestone's CPU-only-environment requirement."""
    import forge.cuda.profiler  # noqa: F401
    import forge.backend.cuda.profiler  # noqa: F401
