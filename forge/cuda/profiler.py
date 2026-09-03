"""Public CUDA allocation-profiling API (Milestone 24).

```python
with forge.cuda.profiler.profile():
    train_step()
events = forge.cuda.profiler.events()   # -> tuple[AllocationEvent, ...]
```

or, equivalently, the explicit start/stop form:

```python
forge.cuda.profiler.reset()
forge.cuda.profiler.start()
with forge.cuda.profiler.tag("forward"):
    ...
forge.cuda.profiler.stop()
events = forge.cuda.profiler.events()
```

Thin wrappers around `forge.backend.cuda.profiler` (the actual collector),
mirroring `forge/cuda/__init__.py`'s existing `memory_stats()`/
`reset_peak_memory_stats()` pattern: this is a *diagnostic* facility,
independent of `forge.cuda.memory_stats()` (Milestone 22, current/peak
counters) and of any future caching allocator. It is explicit (nothing is
recorded unless `start()`/`profile()` was called), optional, and
low-overhead when not running (see `forge/backend/cuda/profiler.py`'s module
docstring for the exact disabled-path cost).

Importing `forge.cuda.profiler` never requires a CUDA-capable device --
only *calling* these functions does, raising `forge.CUDAError` otherwise,
matching every other CUDA-specific Forge entry point.
"""

from __future__ import annotations

from contextlib import contextmanager

from ..backend.cuda.profiler import AllocationEvent
from ..exceptions import CUDAError


def _require_cuda() -> None:
    from ..backend.cuda.backend import is_cuda_available

    if not is_cuda_available():
        raise CUDAError(
            "forge.cuda.profiler requires a working CUDA backend; CUDA is not available on this machine."
        )


def start() -> None:
    """Begin recording allocation/free events. Raises `forge.CUDAError` if CUDA is unavailable."""
    from ..backend.cuda.profiler import get_profiler

    _require_cuda()
    get_profiler().start()


def stop() -> None:
    """Stop recording. Previously recorded events are unaffected; call `events()` to read them."""
    from ..backend.cuda.profiler import get_profiler

    _require_cuda()
    get_profiler().stop()


def reset() -> None:
    """Discard all recorded events. Does not change whether the profiler is running."""
    from ..backend.cuda.profiler import get_profiler

    _require_cuda()
    get_profiler().reset()


def is_active() -> bool:
    """Whether the profiler is currently recording."""
    from ..backend.cuda.profiler import get_profiler

    _require_cuda()
    return get_profiler().active


def events() -> "tuple[AllocationEvent, ...]":
    """A snapshot of every allocation/free event recorded since the last `reset()`."""
    from ..backend.cuda.profiler import get_profiler

    _require_cuda()
    return get_profiler().events()


@contextmanager
def tag(name: str):
    """Label every allocation/free recorded during this block with `name` (e.g. "forward", "backward")."""
    from ..backend.cuda.profiler import get_profiler

    _require_cuda()
    with get_profiler().tag(name):
        yield


@contextmanager
def profile():
    """Convenience: `reset()` + `start()` on entry, `stop()` on exit. Call `events()` afterward to read results."""
    _require_cuda()
    from ..backend.cuda.profiler import get_profiler

    p = get_profiler()
    p.reset()
    p.start()
    try:
        yield
    finally:
        p.stop()


__all__ = ["AllocationEvent", "start", "stop", "reset", "is_active", "events", "tag", "profile"]
