"""Public CUDA API: streams (Milestone 27), synchronization (Milestone 26), memory statistics (Milestone 22; caching allocator in Milestone 25).

```python
forge.cuda.Stream()                   # a real CUDA stream (Milestone 27)
forge.cuda.current_stream()           # the stream Forge CUDA ops issued right now execute on
forge.cuda.set_stream(s)              # make `s` current, returning the previous one
with forge.cuda.stream(s): ...        # make `s` current for the block, then restore
forge.cuda.synchronize()              # block until all issued CUDA work on this device completes
forge.cuda.memory_stats()             # -> CUDAMemoryStats (active/reserved/cached/pending, see below)
forge.cuda.reset_peak_memory_stats()  # resets peak only, live allocations untouched
forge.cuda.empty_cache()              # returns every non-active (cached + pending) block to the driver
```

Thin, explicit wrappers around `forge.backend.cuda` (`synchronize()` around
`CUDABackend.synchronize()`; `Stream`/`current_stream`/`set_stream`/`stream`
around `forge.backend.cuda.stream`; the memory functions around
`forge.backend.cuda.allocator`, the caching allocator and its counters
sitting between `CUDABackend._alloc()`/`CUDAStorage.__del__()` and the real
`cudaMalloc`/`cudaFree` boundary -- see that module's docstring) --
mirroring how `forge.optim`/`forge.serialization` are public packages
fronting `forge.backend`-internal implementation. See
`docs/architecture/cuda-streams.md` for the full Milestone 27 stream/async
execution contract, `docs/architecture/cuda-backend.md`'s **CUDA Execution
and Synchronization Semantics (Milestone 26)** section for the execution
model streams build on, `docs/architecture/cuda-memory-allocator.md` for the
full allocator design, and `docs/architecture/cuda-backend.md`'s **CUDA
Memory Statistics** section for the memory-stats field-by-field semantics.

**Default stream compatibility mode.** Without an active `with forge.cuda.
stream(s):` block, `current_stream()` is `None`, meaning CUDA's own default
(null) stream -- Forge's exact Milestone 8-26 host-synchronous behavior,
unchanged: every CUDA-backed operation still synchronizes internally before
trusting its own result. Only *inside* a `with forge.cuda.stream(s):` block
does execution become asynchronous with respect to the host -- see
`docs/architecture/cuda-streams.md`'s **Default stream compatibility mode**
section for why this was chosen over making every operation asynchronous by
default.

`synchronize()` still exists for callers that need an explicit host-side
barrier of their own (bracketing a benchmark measurement, or simply wanting
a device-idle checkpoint) -- it remains unnecessary for correctness in
default-stream (host-synchronous) code, but is *required* before reading a
result from the host after issuing work inside a `with forge.cuda.stream(s):`
block (or use `s.synchronize()` for just that stream).

Importing `forge.cuda` itself never requires a CUDA-capable device or
`nvcc` -- it only imports pure-Python counters and stream/event wrappers
(see `forge/backend/cuda/__init__.py`'s module docstring) -- so `import
forge` remains CUDA-optional. Only *calling* `Stream()`/`current_stream()`/
`set_stream()`/`stream()`/`synchronize()`/`memory_stats()`/
`reset_peak_memory_stats()`/`empty_cache()` requires a working CUDA backend,
raising `forge.CUDAError` otherwise, matching every other CUDA-specific
entry point in Forge (e.g. `Tensor(..., device="cuda")` on a machine with no
GPU).
"""

from __future__ import annotations

from contextlib import contextmanager

from ..backend.cuda.allocator import CUDAMemoryStats
from ..backend.cuda.stream import CUDAStream
from ..exceptions import CUDAError
from . import profiler


def _require_cuda() -> None:
    from ..backend.cuda.backend import is_cuda_available

    if not is_cuda_available():
        raise CUDAError(
            "forge.cuda.Stream()/current_stream()/set_stream()/stream()/synchronize()/"
            "memory_stats()/reset_peak_memory_stats()/empty_cache() require a working CUDA "
            "backend; CUDA is not available on this machine."
        )


class Stream(CUDAStream):
    """A real CUDA stream. See `forge.backend.cuda.stream.CUDAStream` for the full contract.

    A thin subclass rather than a bare alias only so construction is gated
    by the same `_require_cuda()` check every other `forge.cuda` entry point
    uses (`CUDAStream.__init__` itself already raises `forge.CUDAError` via
    `get_cuda_backend()` once CUDA is genuinely unavailable; this makes that
    explicit and consistent here too, rather than relying on that as an
    implementation detail of stream construction specifically).
    """

    def __init__(self) -> None:
        _require_cuda()
        super().__init__()


# -- streams (Milestone 27) --------------------------------------------------


def current_stream() -> "Stream | None":
    """The stream Forge CUDA operations issued right now execute on.

    `None` means CUDA's default (null) stream -- Forge's host-synchronous
    compatibility mode (see this module's docstring). Raises
    `forge.CUDAError` if CUDA is not available on this machine.
    """
    from ..backend.cuda.stream import current_stream as _current_stream

    _require_cuda()
    return _current_stream()


def set_stream(stream: "Stream | None") -> "Stream | None":
    """Make `stream` the current Forge CUDA stream, returning whatever was current before.

    `stream=None` restores the default (host-synchronous) stream. Prefer
    `with forge.cuda.stream(s):` where the change should be scoped and
    automatically restored; use this only when a manual, unscoped switch is
    actually what's wanted. Raises `forge.CUDAError` if CUDA is not
    available on this machine.
    """
    from ..backend.cuda.stream import set_stream as _set_stream

    _require_cuda()
    return _set_stream(stream)


@contextmanager
def stream(stream_obj: "Stream"):
    """`with forge.cuda.stream(s): ...` -- makes `s` current for the block, then restores the prior stream.

    Every Forge CUDA operation issued inside the block (`Tensor` ops,
    `Module` forward/backward, `Optimizer.step()`, ...) executes on `s`
    asynchronously with respect to the host -- see this module's docstring
    and `docs/architecture/cuda-streams.md`. Raises `forge.CUDAError` if
    CUDA is not available on this machine.
    """
    from ..backend.cuda.stream import stream_context as _stream_context

    _require_cuda()
    with _stream_context(stream_obj) as s:
        yield s


def synchronize() -> None:
    """Block the host until all previously issued Forge CUDA work on this device has completed.

    A thin wrapper around `CUDABackend.synchronize()` (`forge/backend/cuda/
    backend.py`), itself a direct `cudaDeviceSynchronize()` call -- no dummy
    kernel, no sleep/poll loop. `cudaDeviceSynchronize()` waits for *every*
    stream on the device, not just the current one, so this remains a
    correct, if coarse, barrier regardless of how many `Stream`s exist or
    which one is current (Milestone 27) -- there is still no device or
    stream argument to pass, since Forge supports exactly one CUDA device.

    Calling this is never required for correctness in default-stream
    (host-synchronous) code: every CUDA-backed `Tensor`/`Module`/`Loss`/
    `Optimizer` operation already calls `cudaDeviceSynchronize()` internally
    before returning its result there. It *is* required before a host-side
    caller can safely read a result produced inside a `with forge.cuda.
    stream(s):` block (or use `s.synchronize()` to wait for only that
    stream) -- see this module's docstring and `docs/architecture/
    cuda-streams.md`. It also remains useful for bracketing a benchmark
    measurement (`benchmarks/timing.py`) or a manual device-idle checkpoint.
    Calling it repeatedly, or when no CUDA work is outstanding, is always
    safe (a completed/empty queue synchronizes trivially).

    Raises `forge.CUDAError` if CUDA is not available on this machine, or if
    the underlying `cudaDeviceSynchronize()` call itself reports an error
    (e.g. an asynchronous kernel-execution error from a previous launch,
    on any stream, that had not yet been observed).
    """
    from ..backend.cuda.backend import get_cuda_backend

    _require_cuda()
    get_cuda_backend().synchronize()


def memory_stats() -> CUDAMemoryStats:
    """Return a snapshot of Forge's CUDA allocation accounting.

    Raises `forge.CUDAError` if CUDA is not available on this machine.
    """
    from ..backend.cuda.allocator import memory_stats as _memory_stats

    _require_cuda()
    return _memory_stats()


def reset_peak_memory_stats() -> None:
    """Reset `peak_allocated_bytes`/`peak_reserved_bytes` to their current values; live allocations are untouched.

    Raises `forge.CUDAError` if CUDA is not available on this machine.
    """
    from ..backend.cuda.allocator import reset_peak_memory_stats as _reset_peak

    _require_cuda()
    _reset_peak()


def empty_cache() -> int:
    """Release every currently non-active CUDA allocation back to the driver.

    Live Tensor/Parameter/Adam-state storage is never touched -- only blocks
    a `CUDAStorage` has already released to the allocator (see
    `forge.backend.cuda.allocator`). Returns the number of blocks actually
    freed. Raises `forge.CUDAError` if CUDA is not available, or if a
    `cudaFree` call itself fails partway through (see `CUDACachingAllocator.
    empty_cache`'s docstring for the partial-failure behavior in that case).

    **Cost changed in Milestone 27.** A block released by a `CUDAStorage`
    last used on the CUDA default stream is freed immediately, exactly as
    before (M25/M26): it was already guaranteed complete by the time
    `__del__` ran, so no waiting is needed. A block released by a storage
    last used on an explicit `Stream`, however, may still be in flight --
    `empty_cache()` now waits for each such block's recorded completion
    event (`CUDAEvent.synchronize()`) before freeing it, so this call **can
    now block** the calling thread if asynchronous work has not finished.
    See `docs/architecture/cuda-streams.md`'s **`empty_cache()`** section.
    """
    from ..backend.cuda.allocator import empty_cache as _empty_cache
    from ..backend.cuda.backend import get_cuda_backend

    _require_cuda()
    backend = get_cuda_backend()
    return _empty_cache(backend._lib)


__all__ = [
    "Stream",
    "current_stream",
    "set_stream",
    "stream",
    "synchronize",
    "memory_stats",
    "reset_peak_memory_stats",
    "empty_cache",
    "CUDAMemoryStats",
    "profiler",
]
