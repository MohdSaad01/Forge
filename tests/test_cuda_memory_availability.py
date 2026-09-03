"""Milestone 22: `forge.cuda` availability-error tests.

Split out from `tests/test_cuda_memory.py` deliberately: that file carries a
module-level `pytestmark` skipping every test when CUDA is unavailable (the
convention every `test_cuda_*.py` file uses), which would also skip these
two -- the ones that specifically exercise the "CUDA is not available" path
itself via a monkeypatched `is_cuda_available`, and so must run on *any*
machine, CUDA or not, to mean anything.
"""

from __future__ import annotations

import pytest

import forge
from forge.exceptions import CUDAError


def test_memory_stats_raises_cuda_error_when_unavailable(monkeypatch):
    monkeypatch.setattr("forge.backend.cuda.backend.is_cuda_available", lambda: False)
    with pytest.raises(CUDAError):
        forge.cuda.memory_stats()


def test_reset_peak_memory_stats_raises_cuda_error_when_unavailable(monkeypatch):
    monkeypatch.setattr("forge.backend.cuda.backend.is_cuda_available", lambda: False)
    with pytest.raises(CUDAError):
        forge.cuda.reset_peak_memory_stats()
