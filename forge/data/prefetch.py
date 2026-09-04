"""Bounded asynchronous CUDA GPU-prefetch DataLoader wrapper (Milestone 30).

`CUDAPrefetchLoader` wraps an existing `DataLoader` (any object supporting
`__iter__`/`__len__`, in practice always a `forge.data.DataLoader`) and turns
its plain, synchronous, CPU-only iteration into a bounded asynchronous
pipeline:

```text
Dataset -> DataLoader batching (unchanged)   -- background CPU thread
             |
        pinned host batch                    -- main thread
             |
        transfer stream: cudaMemcpyAsync      -- main thread (submission only)
             |
        CUDA Tensor batch, last_stream=transfer_stream
             |
        handed to the caller                  -- Milestone 28 `_stream_guard`
                                                   makes the caller's first
                                                   kernel touching this batch
                                                   wait for the transfer,
                                                   GPU-side only
```

`DataLoader` itself is completely untouched -- this module only ever calls
its existing, public `__iter__()`/`__len__()`. Batching, shuffling, RNG,
`drop_last`, and dataset semantics are all exactly the M5 `DataLoader`
contract; this module adds nothing to that layer, matching `docs/architecture/
data-pipeline.md`'s existing "no multiprocessing workers, no asynchronous
prefetching happens *inside* DataLoader" boundary -- prefetching is a
strictly separate, opt-in wrapper (Section 3 of the milestone brief).

**Threading model.** A single bounded background `threading.Thread` ("the CPU
producer") does exactly one thing: repeatedly call `next()` on the wrapped
loader's own iterator and push each resulting *CPU* batch (unmodified --
still whatever `DataLoader.__iter__` already yields) into a bounded
`queue.Queue`. It never touches CUDA, a stream, or `forge.cuda` in any way.
This is deliberate: `forge.backend.cuda.stream`'s current-stream state is one
process-global variable, not thread-local (see that module's docstring) --
letting two threads independently call `forge.cuda.stream(...)`/
`Tensor.to(..., non_blocking=True)` concurrently would race on that global
and could silently launch a kernel or transfer on the wrong stream. Confining
all CUDA-stream-touching work (pinning a batch, submitting its async H2D,
and everything the caller subsequently does with the resulting CUDA Tensor)
to the single calling thread sidesteps that hazard entirely, while the
background thread's pure-Python/NumPy dataset+collate work still genuinely
overlaps with GPU execution: any blocking CUDA call the calling thread makes
(e.g. a default-stream kernel's implicit synchronize, or `Tensor.to("cpu")`
reading a pending result) is a `ctypes` call, which releases the GIL for its
duration -- exactly the window in which the background thread's Python
bytecode (dataset indexing, transform application, `np.stack`) actually
runs. See `docs/architecture/async-dataloader.md`'s **Threading Model**
section for the full reasoning, including why this was chosen over a
CUDA-touching background thread.

**RNG safety.** `DataLoader.__iter__()` is a generator; its one RNG draw (the
shuffle permutation, if `shuffle=True`) executes lazily, on the *first*
`next()` call, not at `iter()` time. To guarantee that draw can never race
against another thread reading the same process-global generator (e.g. a
`Dropout` layer drawing from `forge.random.default_generator()` during the
main thread's forward pass), this module always performs exactly one
`next()` call on the wrapped iterator itself, synchronously, on the calling
thread, *before* starting the background thread -- only the batches after
that point (which draw no further randomness; see `DataLoader._collate`)
are ever produced by the background thread. See Section 35 of the milestone
brief and `docs/architecture/async-dataloader.md`'s **RNG Semantics**
section.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Iterator

from .. import cuda as _cuda
from ..backend.device import Device
from ..exceptions import DataError
from ..tensor.tensor import Tensor

_SENTINEL = object()


class _ProducerError:
    """Wraps an exception raised inside the background CPU-producer thread."""

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


def _run_producer(source_iter: "Iterator[Any]", out_queue: "queue.Queue[Any]") -> None:
    """The background thread's entire body -- a free function, deliberately.

    Bound as `threading.Thread(target=_run_producer, args=(source_iter,
    self._queue))` rather than `target=self._produce` -- a bound method
    would hold a strong reference back to the owning `_PrefetchIterator`,
    which itself holds the `Thread` object (`self._thread`), forming a
    reference cycle. CPython's plain refcounting (which this codebase's
    lifetime tests rely on throughout -- see `docs/development/progress.md`'s
    CUDA test notes) cannot collect a cycle immediately on `break`/scope
    exit; only the periodic cyclic GC eventually would, leaving this
    iterator's background thread and queued CUDA/pinned batches alive for an
    unbounded, unpredictable interval after early termination (Section 30/64
    of the milestone brief). Taking only `source_iter`/`out_queue` as plain
    arguments -- never `self` -- keeps this function's only strong reference
    the live `Thread` itself, so `_PrefetchIterator.__del__` runs on
    ordinary refcounting the moment nothing else holds the iterator.
    """
    try:
        for batch in source_iter:
            out_queue.put(batch)
    except BaseException as exc:  # propagate Dataset/transform exceptions to the consumer
        out_queue.put(_ProducerError(exc))
        return
    out_queue.put(_SENTINEL)


def _pin_component(component: Tensor) -> Tensor:
    """A CPU `Tensor`, staged into a fresh `forge.cuda.PinnedMemory` buffer.

    Reuses Milestone 29's pinned-memory mechanism exactly as
    `benchmarks/async_transfer_bench.py`'s own `_pinned_tensor()` helper
    does -- one `PinnedMemory(nbytes)` allocation, one host-side copy into
    its zero-copy NumPy view, wrapped back into a `Tensor` (dtype-preserving,
    Section 23). No new pinned-memory lifetime system: the returned Tensor's
    `_pinned_owner` back-reference is exactly what keeps this buffer alive
    for as long as an in-flight transfer needs it (`PinnedMemory.free()`),
    per Section 12.
    """
    if component.device.type != "cpu":
        raise DataError(
            "CUDAPrefetchLoader requires every batch Tensor from the wrapped "
            f"DataLoader to already be on device 'cpu' (got '{component.device}')."
        )
    host = component.numpy()
    mem = _cuda.PinnedMemory(host.nbytes)
    pinned_view = mem.numpy(shape=host.shape, dtype=host.dtype)
    pinned_view[:] = host
    return Tensor(pinned_view, dtype=component.dtype, device="cpu")


def _map_batch(batch: Any, fn) -> Any:
    """Apply `fn` to every `Tensor` component of `batch`, preserving its structure.

    Supports exactly the structures `DataLoader._collate` produces: a single
    `Tensor`, or a flat `tuple`/`list` of `Tensor`s (Section 21) -- a non-
    `Tensor` component (e.g. a raw target, matching `Trainer._to_device_batch`'s
    own tolerance for a non-Tensor `y`) passes through unchanged. Anything
    else (a deeper nested structure, or an unrecognized top-level type) is
    explicitly rejected with `DataError` rather than silently mishandled.
    """
    if isinstance(batch, Tensor):
        return fn(batch)
    if isinstance(batch, (tuple, list)):
        mapped = [fn(c) if isinstance(c, Tensor) else c for c in batch]
        return type(batch)(mapped)
    raise DataError(
        "CUDAPrefetchLoader supports batches that are a Tensor or a tuple/list of "
        f"Tensors (matching DataLoader's own batch contract); got {type(batch).__name__}."
    )


class _ReadyBatch:
    """One pipeline slot: a CUDA batch plus the pinned host batch that must outlive its transfer.

    `_pinned` is never read again -- it exists purely for lifetime (Section
    12): dropping it immediately after submitting the async H2D copy would
    let its `PinnedMemory.__del__` run right away, which *blocks* the host
    until that same transfer completes (defeating the whole point of
    submitting it asynchronously). Keeping it alive inside this record until
    the *next* pipeline slot replaces it gives the transfer roughly one full
    batch's worth of real work to complete in the background first, so by
    the time this record is actually discarded the wait is normally already
    satisfied.
    """

    __slots__ = ("cuda_batch", "_pinned")

    def __init__(self, cuda_batch: Any, pinned_batch: Any) -> None:
        self.cuda_batch = cuda_batch
        self._pinned = pinned_batch


class _PrefetchIterator:
    """One epoch's worth of asynchronous prefetch state. See module docstring."""

    def __init__(self, owner: "CUDAPrefetchLoader") -> None:
        self._device = owner._device
        self._transfer_stream = owner._transfer_stream
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=owner.prefetch_size)
        self._thread: "threading.Thread | None" = None
        self._ready: "_ReadyBatch | None" = None

        source_iter: "Iterator[Any]" = iter(owner._loader)
        try:
            first_cpu_batch = next(source_iter)
        except StopIteration:
            return  # empty epoch -- `_ready`/`_thread` stay None, first __next__ stops immediately

        self._thread = threading.Thread(
            target=_run_producer,
            args=(source_iter, self._queue),
            daemon=True,
            name="forge-prefetch-cpu-producer",
        )
        self._thread.start()
        self._ready = self._make_ready(first_cpu_batch)

    # -- main-thread staging (the only thread that ever touches CUDA/streams) -

    def _make_ready(self, cpu_batch: Any) -> _ReadyBatch:
        pinned_batch = _map_batch(cpu_batch, _pin_component)
        with _cuda.stream(self._transfer_stream):
            cuda_batch = _map_batch(
                pinned_batch, lambda t: t.to(self._device, non_blocking=True)
            )
        return _ReadyBatch(cuda_batch, pinned_batch)

    # -- iterator protocol -----------------------------------------------------

    def __iter__(self) -> "_PrefetchIterator":
        return self

    def __next__(self) -> Any:
        if self._ready is None:
            self._stop()
            raise StopIteration
        result = self._ready.cuda_batch

        item = self._queue.get()
        if item is _SENTINEL:
            self._ready = None
            self._stop()
        elif isinstance(item, _ProducerError):
            self._ready = None
            self._stop()
            raise item.exc
        else:
            self._ready = self._make_ready(item)
        return result

    # -- lifecycle (Section 29/30/32/64) ---------------------------------------

    def _stop(self) -> None:
        """Terminate the background producer and drop any queued/in-flight resources.

        Safe to call more than once, and safe to call while the producer is
        blocked on `queue.put()` (a full queue, e.g. after early termination
        via `for batch in loader: break`) -- draining the queue unblocks it
        so it can observe there is nothing left to consume and exit. Never
        forcibly kills the thread (Python cannot); it only guarantees the
        thread is not left waiting forever on a queue nobody will ever read
        again, and that queued pinned/CUDA batches are released promptly
        rather than held until this iterator itself is garbage collected.
        """
        thread = self._thread
        self._thread = None
        self._ready = None
        if thread is None:
            return
        while thread.is_alive():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                thread.join(timeout=0.05)
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def __del__(self) -> None:
        try:
            self._stop()
        except Exception:
            pass  # interpreter shutdown may have already torn down module globals


class CUDAPrefetchLoader:
    """Wraps `loader` to yield CUDA-resident batches via a bounded asynchronous prefetch pipeline.

    ```python
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    prefetch_loader = CUDAPrefetchLoader(loader, device="cuda", prefetch_size=2)
    # or, equivalently:
    prefetch_loader = loader.prefetch(device="cuda", prefetch_size=2)

    for x, y in prefetch_loader:   # x, y are already CUDA Tensors
        ...
    ```

    `loader` is never modified and remains independently usable for plain
    synchronous iteration -- this is a wrapper, not a DataLoader subclass or
    reimplementation (Section 3). `device` must resolve to `'cuda'`
    (`device='cpu'` raises `DataError` immediately -- Section 24); CUDA
    itself is only required, and only probed, at construction (creating the
    dedicated transfer `Stream()` raises `forge.CUDAError` if unavailable --
    Section 25), never merely by importing `forge.data`/this module.

    `prefetch_size` (default `2`) bounds how many *CPU* batches the
    background producer may hold ready at once (Section 6/49): once that
    many are queued, `queue.Queue.put()` blocks the producer until the
    consumer catches up, which is the pipeline's entire backpressure
    mechanism -- no unbounded CPU, pinned, or GPU memory growth (Section
    15/47/48). Independent of that, exactly one batch's worth of GPU/pinned
    memory is ever in flight on the transfer side (Section 45's minimum
    "double buffering": the batch currently held by the caller, plus the one
    already submitted and staged as this epoch's next `__next__()` result).

    The dedicated transfer `Stream()` is created once, here, and reused for
    every batch of every epoch for as long as this `CUDAPrefetchLoader`
    exists (Section 28) -- `iter(prefetch_loader)` (called once per epoch by
    `for batch in prefetch_loader:`) only creates a fresh background
    thread/queue for that epoch (Section 33: no batch crosses an epoch
    boundary), never a fresh stream.
    """

    def __init__(self, loader: Any, device: "str | Device" = "cuda", prefetch_size: int = 2) -> None:
        if not hasattr(loader, "__iter__"):
            raise DataError(
                f"CUDAPrefetchLoader requires an iterable DataLoader-like object, got "
                f"{type(loader).__name__}."
            )
        target = Device.parse(device)
        if target.type != "cuda":
            raise DataError(
                f"CUDAPrefetchLoader requires device='cuda' (GPU prefetch has no CPU mode); "
                f"got '{target}'."
            )
        if isinstance(prefetch_size, bool) or not isinstance(prefetch_size, int) or prefetch_size < 1:
            raise DataError(f"prefetch_size must be a positive int, got {prefetch_size!r}.")

        self._loader = loader
        self._device = target
        self.prefetch_size = prefetch_size
        # Raises forge.CUDAError immediately if CUDA is unavailable -- the
        # one point this constructor actually touches CUDA (Section 25).
        self._transfer_stream = _cuda.Stream()

    def __iter__(self) -> _PrefetchIterator:
        return _PrefetchIterator(self)

    def __len__(self) -> int:
        return len(self._loader)

    def __repr__(self) -> str:
        return f"CUDAPrefetchLoader(device={self._device}, prefetch_size={self.prefetch_size})"


__all__ = ["CUDAPrefetchLoader"]
