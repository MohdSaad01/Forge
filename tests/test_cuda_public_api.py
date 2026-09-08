"""Milestone 61: `forge.cuda.is_cuda_available` public re-export.

Before this milestone, checking CUDA availability required reaching into
`forge.backend.cuda` (an internal-sounding path) even though every other
CUDA entry point a caller needs (`Stream`, `synchronize`, `memory_stats`,
...) already lives on the public `forge.cuda` package. This test protects
the re-export so it does not silently regress; it runs on any machine,
CUDA or not, since `is_cuda_available()` itself never raises.
"""

from __future__ import annotations

import forge


def test_is_cuda_available_is_exported_from_forge_cuda():
    assert "is_cuda_available" in forge.cuda.__all__
    assert forge.cuda.is_cuda_available is forge.backend.cuda.is_cuda_available


def test_is_cuda_available_returns_a_bool_without_raising():
    result = forge.cuda.is_cuda_available()
    assert isinstance(result, bool)
