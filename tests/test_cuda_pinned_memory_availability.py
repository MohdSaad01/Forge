"""Milestone 29: CUDA-unavailable behavior for the public pinned-memory API.

Split out from `tests/test_cuda_pinned_memory.py` deliberately, matching
every other `test_cuda_*_availability.py` file's convention (see
`tests/test_cuda_streams_availability.py`): must run on any machine, CUDA or
not, since it specifically exercises the "CUDA is not available" path via a
monkeypatched `is_cuda_available`.
"""

from __future__ import annotations

import pytest

import forge
from forge.exceptions import CUDAError


def test_pinned_memory_construction_raises_cuda_error_when_unavailable(monkeypatch):
    monkeypatch.setattr("forge.backend.cuda.backend.is_cuda_available", lambda: False)
    with pytest.raises(CUDAError):
        forge.cuda.PinnedMemory(1024)


def test_pinned_memory_stats_raises_cuda_error_when_unavailable(monkeypatch):
    monkeypatch.setattr("forge.backend.cuda.backend.is_cuda_available", lambda: False)
    with pytest.raises(CUDAError):
        forge.cuda.pinned_memory_stats()
