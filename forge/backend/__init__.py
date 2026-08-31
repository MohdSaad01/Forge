"""Backend dispatch: which implementation executes a tensor's operations.

CPU is always registered and eagerly constructed. CUDA (Milestone 8) is
registered by name but resolved lazily -- `forge.backend.cuda` (which pulls
in `ctypes` and, on first real use, invokes `nvcc`) is only imported the
first time a `"cuda"` device is actually requested, so a CPU-only
environment never pays any CUDA-related cost or import-time risk.
"""

from __future__ import annotations

from ..exceptions import CUDAError, UnsupportedDeviceError
from .base import Backend
from .cpu import CPUBackend
from .device import Device

_BACKENDS: dict[str, Backend] = {
    "cpu": CPUBackend(),
}


def get_backend(device: Device) -> Backend:
    backend = _BACKENDS.get(device.type)
    if backend is not None:
        return backend

    if device.type == "cuda":
        if device.index not in (None, 0):
            raise CUDAError(
                f"Forge's CUDA backend targets a single device (index 0) in this milestone; "
                f"got index {device.index}."
            )
        from .cuda.backend import get_cuda_backend

        return get_cuda_backend()
    raise UnsupportedDeviceError(f"No backend registered for device type '{device.type}'.")


__all__ = ["Backend", "CPUBackend", "Device", "get_backend"]
