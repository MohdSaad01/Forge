"""A minimal, transfer-specific completion handle (Milestone 29).

`PendingTransfer` is the smallest possible answer to Section 18/32 of the
milestone brief: an asynchronous D2H copy needs *some* way for a host-facing
API to know when it becomes safe to read the destination buffer, but the
brief explicitly rules out a general futures/promises subsystem. This wraps
exactly one internal `CUDAEvent` (`stream.py`, already used for allocator
pending-block safety and cross-stream dependencies -- no new synchronization
primitive introduced) and nothing else.

Not public API (`forge.cuda` exposes no `Transfer`/`Future` type -- Section
46). The only consumer is `Tensor._data`'s property getter
(`forge/tensor/tensor.py`), which calls `synchronize()` exactly once, the
first time a pending D2H tensor's storage is actually read from the host --
see that module's docstring for why a property is the right chokepoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .stream import CUDAEvent


class PendingTransfer:
    """Wraps one CUDA completion event recorded at the end of an async D2H copy."""

    __slots__ = ("_event", "_synced")

    def __init__(self, event: "CUDAEvent") -> None:
        self._event = event
        self._synced = False

    def is_ready(self) -> bool:
        """True if the transfer has already completed -- never blocks (`cudaEventQuery`)."""
        if self._synced:
            return True
        return self._event.query()

    def synchronize(self) -> None:
        """Block the host until this transfer completes. Safe to call more than once."""
        if not self._synced:
            self._event.synchronize()
            self._synced = True


__all__ = ["PendingTransfer"]
