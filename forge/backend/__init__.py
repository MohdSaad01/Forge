"""Backend dispatch: which implementation executes a tensor's operations.

Only CPU is registered in Milestone 1. Requesting a CUDA device parses fine
(``Device.parse``) but dispatching to it fails clearly here -- CUDA is not
executable yet, and Forge does not pretend otherwise.
"""

from __future__ import annotations

from ..exceptions import UnsupportedDeviceError
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
        raise UnsupportedDeviceError(
            "CUDA execution is not implemented yet (Milestone 1 is CPU-only). "
            "The device string 'cuda' can be named, but no CUDA backend exists to run on it."
        )
    raise UnsupportedDeviceError(f"No backend registered for device type '{device.type}'.")


__all__ = ["Backend", "CPUBackend", "Device", "get_backend"]
