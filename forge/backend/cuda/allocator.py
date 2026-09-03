"""Exact-size CUDA caching allocator (Milestone 25).

Milestone 24 measured Forge's *direct* allocation model (one `cudaMalloc` per
`CUDAStorage`, one `cudaFree` at its destruction) on the real 940MX and found
the driver-call overhead (~175-300 us/call) the same order of magnitude as an
entire MNIST training iteration, with 99%+ of allocated bytes an exact-size
repeat of an earlier request. `docs/architecture/cuda-memory-allocator.md`
concluded a caching allocator was justified; this module is that allocator.

```text
CUDAStorage creation
    -> CUDABackend._alloc(nbytes)
    -> allocate(lib, nbytes)
         exact-size free list has a block?
             yes -> pop it, no driver call (cache hit)
             no  -> cudaMalloc(max(nbytes, 1)) (cache miss)

CUDAStorage destruction
    -> release(nbytes, ptr)
         push (nbytes, ptr) onto the exact-size free list -- no cudaFree
```

The allocator retains only a raw `ctypes.c_void_p` per cached block, never a
`CUDAStorage`/`Tensor` -- exactly the same "never retain a Forge object"
discipline `profiler.py` already documents for `AllocationEvent.block_id`,
and for the same reason (retaining a live object here would keep otherwise-
unreachable Python state alive forever, independent of whether its device
memory is reused).

**Ownership invariant** (the one this whole module exists to preserve): a
device allocation is at any moment either an *active* block (owned by exactly
one live `CUDAStorage`) or a *cached free* block (owned by this allocator) --
never both, never neither while `Forge` still considers it live. A block
enters the cache only via `release()`, called only from `CUDAStorage.__del__`
after that storage has stopped considering the pointer its own; a block
leaves the cache only via `allocate()` (handed to a fresh `CUDAStorage`) or
`empty_cache()` (returned to the driver). No other code path touches
`_free_blocks`.

**Exact-size only** (per the M24 design's Candidate A, Section 10): a cached
block is reused only for a request of the identical byte count. No block
splitting, no coalescing, no size-class rounding -- see the module docstring
of `docs/architecture/cuda-memory-allocator.md` for why this is the
recommended design for Forge's workloads (a small, fixed, highly-repetitive
set of allocation sizes) rather than a general-purpose best-fit allocator.

**Synchronization**: every Forge CUDA operation already calls
`cudaDeviceSynchronize()` before trusting its own result (`CUDABackend.
_synchronize`, called at the end of every op in `backend.py`), so a
`CUDAStorage` becoming unreachable and "the GPU is done touching that memory"
already coincide today -- no in-flight kernel can still be reading/writing a
buffer at the moment its owning `CUDAStorage.__del__` runs. This caching
allocator inherits that safety property unchanged: it does not alter *when*
`__del__` runs, only what happens to the pointer afterward. It assumes this
synchronous execution model throughout; a future stream-aware Forge backend
would need to re-establish "safe to reuse" explicitly (e.g. stream-ordered
events) before immediate cache reuse across streams would be safe again --
out of scope here (see Section 30 of the milestone brief).

**Thread safety**: one `threading.Lock` guards `_free_blocks` and every
counter, matching `memory.py`'s pre-existing `_MemoryTracker` convention --
Forge remains single-threaded elsewhere; this is cheap insurance, not a
concurrency subsystem. The lock is released before any real CUDA driver call
(`cf_malloc`/`cf_free`) so a slow driver call never holds up unrelated
bookkeeping reads.

**Milestone 27 -- pending (stream-ordered) blocks.** The ownership invariant
above still holds, but a device allocation released by a `CUDAStorage` last
used on an explicit (non-default) `CUDAStream` can no longer become a *ready*
free block immediately -- Python object destruction no longer implies GPU
completion once asynchronous execution exists (see
`docs/architecture/cuda-streams.md`). Such a release instead becomes a
*pending* block: `CUDAStorage.__del__` calls `release_pending()`, which
records a real `CUDAEvent` (`stream.py`) on that storage's last-use stream
and stores `(event, ptr)` in `_pending_blocks[nbytes]`, still exclusively
owned by this allocator (Invariant 3) but not yet eligible for reuse. A
pending block becomes eligible the moment its event is observed complete
(`CUDAEvent.query()`) -- checked opportunistically on the next same-size
`allocate()` call, and forced (via `CUDAEvent.synchronize()`) only as a last
resort during OOM handling or `empty_cache()`. A block released by a
`CUDAStorage` last used on the *default* stream is unaffected: it still goes
through the original `release()` -> ready free list path unchanged, because
the M26 contract (every default-stream operation already synchronizes before
returning) still holds for that stream specifically.
"""

from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass

from ...exceptions import CUDAError
from . import profiler as _profiler
from . import stream as _stream


@dataclass(frozen=True)
class CUDAMemoryStats:
    """A snapshot of Forge's CUDA allocation accounting at one point in time.

    The four original Milestone 22 fields keep their original meaning and
    position (a caller doing `CUDAMemoryStats(allocated_bytes=..., peak_allocated_bytes=...,
    allocation_count=..., free_count=...)` still works):

    - `allocated_bytes` / `peak_allocated_bytes`: bytes owned by currently
      *live* `CUDAStorage` objects (and its historical peak) -- unchanged in
      meaning by caching, since a cache hit/miss both still correspond
      exactly to one `CUDAStorage` becoming active/inactive.
    - `allocation_count` / `free_count`: under Milestone 22's direct
      allocator these were 1:1 with real `cudaMalloc`/`cudaFree` calls: under
      caching, they still are -- they now count only *driver* calls (a cache
      hit/release causes neither), per `docs/architecture/cuda-memory-
      allocator.md` Section 14's explicit recommendation. `cuda_malloc_count`/
      `cuda_free_count` below are clearer aliases for the same two fields.

    Milestone 25 fields distinguish active memory from cached-but-unused
    memory:

    - `reserved_bytes`: total device memory this process currently holds via
      the allocator -- active, ready-cached, or pending (`active + cached +
      pending`).
    - `peak_reserved_bytes`: historical peak of `reserved_bytes`.
    - `cached_bytes`: bytes held in the *ready* free list -- immediately
      reusable for a same-size request with no driver call and no wait.
    - `cache_hit_count` / `cache_miss_count`: allocation *requests* (not
      driver calls) served from the ready cache or a reclaimed pending block
      vs. requiring a `cudaMalloc`.

    New Milestone 27 fields distinguish *ready* cached memory (immediately
    reusable) from *pending* memory (released, but the `CUDAStream` that
    last used it may not have finished -- see `allocator.py`'s module
    docstring):

    - `pending_bytes`: bytes released by a `CUDAStorage` last used on a
      non-default stream, not yet confirmed safe to reuse.
    - `pending_count`: number of individual pending blocks (across all
      sizes) making up `pending_bytes`.
    """

    allocated_bytes: int
    peak_allocated_bytes: int
    allocation_count: int
    free_count: int
    reserved_bytes: int = 0
    peak_reserved_bytes: int = 0
    cached_bytes: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    pending_bytes: int = 0
    pending_count: int = 0

    @property
    def cuda_malloc_count(self) -> int:
        """Clearer alias for `allocation_count` -- real driver `cudaMalloc` calls only."""
        return self.allocation_count

    @property
    def cuda_free_count(self) -> int:
        """Clearer alias for `free_count` -- real driver `cudaFree` calls only (via `empty_cache()`)."""
        return self.free_count

    def as_dict(self) -> "dict[str, int]":
        return {
            "allocated_bytes": self.allocated_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "allocation_count": self.allocation_count,
            "free_count": self.free_count,
            "reserved_bytes": self.reserved_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "cached_bytes": self.cached_bytes,
            "cache_hit_count": self.cache_hit_count,
            "cache_miss_count": self.cache_miss_count,
            "pending_bytes": self.pending_bytes,
            "pending_count": self.pending_count,
        }


# -- raw driver calls, bypassing the cache entirely -----------------------------
#
# Used internally for a cache miss / `empty_cache()`, and exposed (non-
# underscore) for benchmarks/tests that need a true "direct, uncached"
# baseline to compare the caching allocator against (Section 21 of the
# milestone brief) -- e.g. `benchmarks/allocator_bench.py`. Neither function
# touches `_free_blocks` or any `CUDAMemoryStats` counter; a caller using
# these directly is entirely off Forge's normal accounting, by design.


def _format_cuda_error(lib: "ctypes.CDLL", code: int, action: str) -> str:
    message = lib.cf_error_string(code)
    message = message.decode() if message is not None else "<no message>"
    return f"{action}: {message} (code {code})."


def _try_raw_malloc(lib: "ctypes.CDLL", nbytes: int) -> "tuple[ctypes.c_void_p | None, int]":
    ptr = ctypes.c_void_p()
    code = lib.cf_malloc(ctypes.byref(ptr), ctypes.c_size_t(max(nbytes, 1)))
    if code == 0:
        return ptr, 0
    return None, code


def raw_malloc(lib: "ctypes.CDLL", nbytes: int) -> "ctypes.c_void_p":
    """A direct, uncached `cudaMalloc(max(nbytes, 1))`. Raises `CUDAError` on failure."""
    ptr, code = _try_raw_malloc(lib, nbytes)
    if ptr is None:
        raise CUDAError(_format_cuda_error(lib, code, f"CUDA memory allocation of {nbytes} bytes failed"))
    return ptr


def raw_free(lib: "ctypes.CDLL", ptr: "ctypes.c_void_p") -> None:
    """A direct, uncached `cudaFree(ptr)`. Raises `CUDAError` on failure."""
    code = lib.cf_free(ptr)
    if code != 0:
        raise CUDAError(_format_cuda_error(lib, code, "cudaFree failed"))


# -- pending (stream-ordered) blocks (Milestone 27) -------------------------------


@dataclass
class _PendingBlock:
    """A released block whose safety-to-reuse depends on a recorded `CUDAEvent` completing."""

    event: "_stream.CUDAEvent"
    ptr: "ctypes.c_void_p"


# -- the caching allocator --------------------------------------------------------


class CUDACachingAllocator:
    """Process-wide exact-size CUDA block cache, guarded by one lock.

    `_free_blocks[nbytes]` is a list of cached, currently-unowned device
    pointers of exactly `nbytes` bytes. Forge supports exactly one CUDA
    device today (`CUDABackend.device_count` is probed but never selected
    among), so the cache is keyed by `nbytes` alone -- not `(device, nbytes)`
    -- matching the actually-exercised behavior rather than adding
    unreachable multi-device machinery ahead of need.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._free_blocks: "dict[int, list[ctypes.c_void_p]]" = {}
        self._pending_blocks: "dict[int, list[_PendingBlock]]" = {}
        self._active_bytes = 0
        self._peak_active_bytes = 0
        self._reserved_bytes = 0
        self._peak_reserved_bytes = 0
        self._cache_hit_count = 0
        self._cache_miss_count = 0
        self._cuda_malloc_count = 0
        self._cuda_free_count = 0
        self._pending_bytes = 0
        self._pending_count = 0

    # -- allocation ----------------------------------------------------------

    def allocate(self, lib: "ctypes.CDLL", nbytes: int) -> "ctypes.c_void_p":
        """Serve `nbytes` from the ready cache, a reclaimed pending block, or a real `cudaMalloc`."""
        with self._lock:
            blocks = self._free_blocks.get(nbytes)
            if blocks:
                ptr = blocks.pop()
                if not blocks:
                    del self._free_blocks[nbytes]
                self._active_bytes += nbytes
                self._cache_hit_count += 1
                if self._active_bytes > self._peak_active_bytes:
                    self._peak_active_bytes = self._active_bytes
                _profiler.get_profiler().record("alloc", nbytes, ptr.value or 0)
                return ptr

        # No ready block: opportunistically reclaim a pending block whose
        # event has already completed (Milestone 27) before paying for a
        # driver call -- still a cache hit, since it needed no `cudaMalloc`.
        reclaimed = self._try_reclaim_pending(nbytes)
        if reclaimed is not None:
            with self._lock:
                self._active_bytes += nbytes
                self._cache_hit_count += 1
                if self._active_bytes > self._peak_active_bytes:
                    self._peak_active_bytes = self._active_bytes
            _profiler.get_profiler().record("alloc", nbytes, reclaimed.value or 0)
            return reclaimed

        # Cache miss: no block held under the lock across the driver call.
        ptr = self._driver_malloc(lib, nbytes)
        with self._lock:
            self._active_bytes += nbytes
            self._reserved_bytes += nbytes
            self._cache_miss_count += 1
            self._cuda_malloc_count += 1
            if self._active_bytes > self._peak_active_bytes:
                self._peak_active_bytes = self._active_bytes
            if self._reserved_bytes > self._peak_reserved_bytes:
                self._peak_reserved_bytes = self._reserved_bytes
        _profiler.get_profiler().record("alloc", nbytes, ptr.value or 0)
        return ptr

    def _try_reclaim_pending(self, nbytes: int) -> "ctypes.c_void_p | None":
        """Return a pending block of exactly `nbytes` whose event has already completed, if any."""
        with self._lock:
            pending = self._pending_blocks.get(nbytes)
            if not pending:
                return None
            for i, block in enumerate(pending):
                if block.event.query():
                    pending.pop(i)
                    if not pending:
                        del self._pending_blocks[nbytes]
                    self._pending_bytes -= nbytes
                    self._pending_count -= 1
                    return block.ptr
            return None

    def _driver_malloc(self, lib: "ctypes.CDLL", nbytes: int) -> "ctypes.c_void_p":
        """`cudaMalloc`, with the M24 OOM policy extended for pending blocks (Milestone 27):

        purge the ready cache and retry; if that still fails, wait for and
        free every pending block (the last resort -- correctness over
        overlap under real memory pressure) and retry once more.
        """
        ptr, code = _try_raw_malloc(lib, nbytes)
        if ptr is not None:
            return ptr
        self._empty_ready(lib)
        ptr, code = _try_raw_malloc(lib, nbytes)
        if ptr is not None:
            return ptr
        self._drain_pending(lib)
        ptr, code = _try_raw_malloc(lib, nbytes)
        if ptr is not None:
            return ptr
        raise CUDAError(
            _format_cuda_error(lib, code, f"CUDA memory allocation of {nbytes} bytes failed (after cache purge)")
        )

    # -- release ---------------------------------------------------------------

    def release(self, nbytes: int, ptr: "ctypes.c_void_p") -> None:
        """Return `ptr` (an active block's `nbytes`) to the ready free list. No driver call.

        Only safe for a block last used on the CUDA default stream -- see
        `release_pending()` for a block last used on an explicit `CUDAStream`.
        `CUDAStorage.__del__` already guards against calling this twice for
        the same storage (it clears `self.ptr` before this call ever
        happens), so this should never see the same pointer released twice
        in ordinary use -- the check below exists to fail loudly, not
        silently corrupt `_free_blocks`, if that invariant is ever violated
        by a future internal bug (Section 7 of the milestone brief).
        """
        with self._lock:
            blocks = self._free_blocks.setdefault(nbytes, [])
            if any(b.value == ptr.value for b in blocks):
                raise RuntimeError(
                    f"CUDACachingAllocator.release() called with a pointer already cached at "
                    f"{nbytes} bytes -- this indicates a double-release of the same allocation "
                    "(an internal Forge bug), not a normal runtime condition."
                )
            self._active_bytes -= nbytes
            blocks.append(ptr)
        _profiler.get_profiler().record("free", nbytes, ptr.value or 0)

    def release_pending(
        self, lib: "ctypes.CDLL", nbytes: int, ptr: "ctypes.c_void_p", stream_handle: "ctypes.c_void_p | None"
    ) -> None:
        """Return `ptr` to the *pending* set, safe to reuse only once its last-use event completes.

        Records a fresh `CUDAEvent` on `stream_handle` right now -- since
        `CUDAStorage.__del__` calls this synchronously from Python, every
        operation that used `ptr` on that stream was necessarily already
        enqueued before this call, so the recorded event completing implies
        all of them have too (stream program-order, not wall-clock timing).
        See `allocator.py`'s module docstring.
        """
        event = _stream.CUDAEvent(lib)
        event.record(stream_handle)
        with self._lock:
            self._active_bytes -= nbytes
            self._pending_blocks.setdefault(nbytes, []).append(_PendingBlock(event, ptr))
            self._pending_bytes += nbytes
            self._pending_count += 1
        _profiler.get_profiler().record("free", nbytes, ptr.value or 0)

    # -- cache purge -------------------------------------------------------------

    def empty_cache(self, lib: "ctypes.CDLL") -> int:
        """Return every non-active block to the driver via `cudaFree`.

        Ready blocks are freed immediately, no waiting needed (identical to
        the M25/M26 behavior). Pending blocks (Milestone 27) are *waited on*
        first (`CUDAEvent.synchronize()`, one at a time) since they may still
        be in flight on their producing stream -- so, unlike the M25/M26
        version, this can now block the calling thread if pending work has
        not finished. Never touches active blocks. Returns the number of
        blocks actually freed (ready + pending).
        """
        return self._empty_ready(lib) + self._drain_pending(lib)

    def _empty_ready(self, lib: "ctypes.CDLL") -> int:
        """Free every currently *ready* cached block. No waiting -- see `empty_cache()`.

        Processes one block at a time, removing it from `_free_blocks` only
        after its `cudaFree` succeeds -- a failure partway through (see
        Section 9 of the milestone brief for why a large `cudaMalloc`
        failure, not `cudaFree`, is the hardware-observed hazard) leaves the
        remaining not-yet-freed blocks exactly as cached as they were, rather
        than silently losing track of them.
        """
        with self._lock:
            snapshot = [(nbytes, ptr) for nbytes, ptrs in self._free_blocks.items() for ptr in ptrs]
        freed = 0
        for nbytes, ptr in snapshot:
            raw_free(lib, ptr)  # raises CUDAError on failure; nothing swallowed
            with self._lock:
                blocks = self._free_blocks.get(nbytes)
                if blocks and ptr in blocks:
                    blocks.remove(ptr)
                    if not blocks:
                        del self._free_blocks[nbytes]
                self._reserved_bytes -= nbytes
                self._cuda_free_count += 1
            freed += 1
        return freed

    def _drain_pending(self, lib: "ctypes.CDLL") -> int:
        """Wait for and free every pending block. Can block -- see `empty_cache()`."""
        with self._lock:
            snapshot = [(nbytes, block) for nbytes, blocks in self._pending_blocks.items() for block in blocks]
            self._pending_blocks = {}
        freed = 0
        for nbytes, block in snapshot:
            block.event.synchronize()  # wait -- this is the "last resort" cost documented above
            raw_free(lib, block.ptr)
            with self._lock:
                self._reserved_bytes -= nbytes
                self._cuda_free_count += 1
                self._pending_bytes -= nbytes
                self._pending_count -= 1
            freed += 1
        return freed

    # -- statistics --------------------------------------------------------------

    def snapshot(self) -> CUDAMemoryStats:
        with self._lock:
            active = self._active_bytes
            pending = self._pending_bytes
            return CUDAMemoryStats(
                allocated_bytes=active,
                peak_allocated_bytes=self._peak_active_bytes,
                allocation_count=self._cuda_malloc_count,
                free_count=self._cuda_free_count,
                reserved_bytes=self._reserved_bytes,
                peak_reserved_bytes=self._peak_reserved_bytes,
                cached_bytes=self._reserved_bytes - active - pending,
                cache_hit_count=self._cache_hit_count,
                cache_miss_count=self._cache_miss_count,
                pending_bytes=pending,
                pending_count=self._pending_count,
            )

    def reset_peak(self) -> None:
        with self._lock:
            self._peak_active_bytes = self._active_bytes
            self._peak_reserved_bytes = self._reserved_bytes


_allocator = CUDACachingAllocator()


def get_allocator() -> CUDACachingAllocator:
    """Return the process-wide `CUDACachingAllocator` singleton."""
    return _allocator


def allocate(lib: "ctypes.CDLL", nbytes: int) -> "ctypes.c_void_p":
    """Internal -- called only by `CUDABackend._alloc`."""
    return _allocator.allocate(lib, nbytes)


def release(nbytes: int, ptr: "ctypes.c_void_p") -> None:
    """Internal -- called only by `CUDAStorage.__del__`, for a default-stream storage."""
    _allocator.release(nbytes, ptr)


def release_pending(
    lib: "ctypes.CDLL", nbytes: int, ptr: "ctypes.c_void_p", stream_handle: "ctypes.c_void_p | None"
) -> None:
    """Internal -- called only by `CUDAStorage.__del__`, for a storage last used on a `CUDAStream`."""
    _allocator.release_pending(lib, nbytes, ptr, stream_handle)


def empty_cache(lib: "ctypes.CDLL") -> int:
    """Return every cached (not active) block to the driver. See `forge.cuda.empty_cache`."""
    return _allocator.empty_cache(lib)


def memory_stats() -> CUDAMemoryStats:
    """Return a snapshot of the current CUDA allocation counters. See `forge.cuda.memory_stats`."""
    return _allocator.snapshot()


def reset_peak_memory_stats() -> None:
    """Reset both peak-active and peak-reserved to their current values. See `forge.cuda.reset_peak_memory_stats`."""
    _allocator.reset_peak()


__all__ = [
    "CUDAMemoryStats",
    "CUDACachingAllocator",
    "get_allocator",
    "allocate",
    "release",
    "release_pending",
    "empty_cache",
    "memory_stats",
    "reset_peak_memory_stats",
    "raw_malloc",
    "raw_free",
]
