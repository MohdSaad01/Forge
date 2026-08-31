"""Forge's CUDA backend (Milestone 8).

Not imported by `forge.backend` at package-import time -- only pulled in
lazily by `forge.backend.get_backend()` when a `"cuda"` device is actually
requested, so a machine with no CUDA toolchain never pays any cost (or risks
any import-time failure) for a CPU-only session. See
`docs/architecture/cuda-backend.md`.
"""

from .backend import CUDABackend, CUDAStorage, get_cuda_backend, is_cuda_available

__all__ = ["CUDABackend", "CUDAStorage", "get_cuda_backend", "is_cuda_available"]
