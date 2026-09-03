"""Milestone 25: `forge.cuda.empty_cache()` availability-error test.

Split out from `tests/test_cuda_allocator.py` deliberately, mirroring
`tests/test_cuda_memory_availability.py`'s own rationale exactly: that file
carries a module-level `pytestmark` skipping every test when CUDA is
unavailable, which would also skip this one -- the one that specifically
exercises the "CUDA is not available" path itself via a monkeypatched
`is_cuda_available`, and so must run on *any* machine, CUDA or not, to mean
anything.
"""

from __future__ import annotations

import pytest

import forge
from forge.exceptions import CUDAError


def test_empty_cache_raises_cuda_error_when_unavailable(monkeypatch):
    monkeypatch.setattr("forge.backend.cuda.backend.is_cuda_available", lambda: False)
    with pytest.raises(CUDAError):
        forge.cuda.empty_cache()
