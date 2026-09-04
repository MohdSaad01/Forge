# Asynchronous DataLoader GPU Prefetch (Milestone 30)

Builds a bounded, opt-in asynchronous CPU-batch-preparation + H2D-transfer +
GPU-compute pipeline entirely on top of Milestones 25-29's existing
primitives (the exact-size caching allocator, CUDA streams, cross-stream
dependencies, and pinned memory / async transfers) -- zero new
synchronization mechanism was introduced. See `docs/architecture/
cuda-transfers.md`'s Section 21 and `docs/architecture/cuda-streams.md`'s
Section 19, both of which anticipated this milestone.

## 1. DataLoader synchronous behavior (unchanged)

`forge/data/dataloader.py` was **not modified** beyond adding one
convenience method (`DataLoader.prefetch()`, Section 2). `DataLoader`
remains a plain, synchronous, CPU-only Python generator: shuffling, RNG
draws, `drop_last`, and NumPy-based batch assembly (`_collate`/`_stack`) are
byte-for-byte the Milestone 5 contract. `loader = DataLoader(dataset,
batch_size=64)` continues to work exactly as before, with no CUDA
dependency of any kind.

## 2. GPU prefetch API

A wrapper, not a `DataLoader` subclass or reimplementation:

```python
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Either of these is equivalent:
prefetch_loader = forge.data.CUDAPrefetchLoader(loader, device="cuda", prefetch_size=2)
prefetch_loader = loader.prefetch(device="cuda", prefetch_size=2)

for x, y in prefetch_loader:   # x, y are already CUDA Tensors
    ...
```

`device` must resolve to `'cuda'` -- `device='cpu'` raises `DataError`
immediately, before touching CUDA at all (there is no CPU "prefetch" mode).
Constructing a `CUDAPrefetchLoader` is the one point CUDA is actually
required/probed (creating its dedicated transfer `Stream()` raises
`forge.CUDAError` if unavailable) -- `import forge.data` itself never
requires CUDA.

## 3. Prefetch depth

`prefetch_size` (default `2`) bounds how many *CPU* batches the background
producer may hold ready in its `queue.Queue` at once (Section 6/14/49 of the
milestone brief) -- once full, `queue.put()` blocks the producer until the
consumer catches up. This is the pipeline's entire backpressure mechanism;
no unbounded CPU/pinned/GPU memory growth is possible. Independent of that
CPU-side depth, exactly one batch is ever staged ahead on the GPU side (the
batch the caller currently holds, plus the one already pinned and
H2D-submitted as the next `__next__()` result) -- "double buffering"
(Section 45), not a deeper GPU-side ring buffer; `prefetch_size` only
changes how far ahead CPU preparation is allowed to run.

## 4. Transfer stream

`CUDAPrefetchLoader.__init__` creates one real `forge.cuda.Stream()`,
reused for every batch of every epoch for the loader's entire lifetime
(never recreated per batch or per epoch -- Section 28). Only the calling
thread ever enters `with forge.cuda.stream(self._transfer_stream):` to
submit a batch's H2D copy.

## 5. Compute stream

Unaffected by default: a caller consuming a `CUDAPrefetchLoader` directly
(not via `Trainer`) does compute on whatever `forge.cuda.current_stream()`
already is -- correctness is unaffected either way (Section 9), but *real*
overlap requires that to be a genuine non-default stream, since CUDA's
legacy default/null stream synchronizes against every explicitly created
stream (see `benchmarks/async_transfer_bench.py`'s own overlap measurement,
which uses two explicit streams for exactly this reason). `Trainer(...,
prefetch=True)` handles this automatically: it creates one dedicated compute
`Stream`, lazily on first use, cached on `self._compute_stream` and reused
for the Trainer's entire lifetime (never recreated per epoch), current for
the duration of `fit()`'s/`evaluate()`'s batch loop only -- restoring
whatever stream was current before on exit, exactly like `forge.cuda.
stream()`'s own scoping contract. See `docs/architecture/cuda-streams.md`'s
Section 15 for how this supersedes "Trainer remains synchronous" for this
one opt-in path.

## 6. Pinned host batches

Every CPU batch the background producer yields is staged into a fresh
`forge.cuda.PinnedMemory` buffer before submission -- reusing Milestone 29's
mechanism exactly as `benchmarks/async_transfer_bench.py`'s own
`_pinned_tensor()` helper does (one `PinnedMemory(nbytes)` allocation, one
host-side copy into its zero-copy NumPy view, wrapped back into a `Tensor`).
No new pinned-memory lifetime system, and no pinned caching allocator
(explicitly out of scope -- Section 72).

## 7. Transfer→compute dependency

Zero new mechanism. `Tensor.to("cuda", non_blocking=True)`, submitted while
the transfer stream is current, produces a CUDA `Tensor` whose backing
`CUDAStorage.last_stream` is the transfer stream (`CUDAStorage.__init__`
already sets this unconditionally -- Milestone 27). The very first CUDA
kernel the caller's consuming code launches that touches this storage on a
*different* current stream invokes `CUDABackend._stream_guard` (Milestone
28), which inserts a GPU-side `cudaEventRecord`/`cudaStreamWaitEvent`
dependency automatically -- no explicit event, no `cudaDeviceSynchronize()`,
and no code in this milestone's own implementation ever waits for a
transfer directly. Verified directly: `tests/test_dataloader_prefetch.py::
test_consuming_prefetched_batches_never_calls_cuda_device_synchronize` and
`tests/test_trainer_prefetch.py::test_fit_never_calls_cuda_device_synchronize`
monkeypatch `cf_synchronize` and assert zero calls across a full prefetch
training run.

## 8. Buffer ownership

`_ReadyBatch` (`forge/data/prefetch.py`) is the explicit owner: it holds
both the CUDA batch handed to the caller and the pinned host batch that
produced it, for exactly one pipeline slot's lifetime. The pinned reference
is never read again -- it exists purely so its `PinnedMemory` is not
`__del__`'d (which would *block* the host, waiting on the transfer's
completion event, defeating the point of submitting it asynchronously)
immediately after submission. Dropping it only when the *next* slot replaces
it gives the transfer roughly one full batch's worth of real compute time to
complete in the background first.

## 9. Backpressure

`queue.Queue(maxsize=prefetch_size)` is the entire mechanism (Section 3
above) -- no separate rate limiter, no polling.

## 10. Epoch boundaries

`CUDAPrefetchLoader.__iter__()` (called once per epoch by `for batch in
prefetch_loader:`) creates an entirely fresh `_PrefetchIterator`: fresh
background thread, fresh bounded queue, fresh `iter(self._loader)` -- no
state survives across this boundary except the transfer stream itself
(Section 4). `tests/test_dataloader_prefetch.py::
test_no_batches_leak_across_epoch_boundary` reconstructs each epoch's
consumed index set and asserts it covers `0..n-1` exactly once, every epoch.

## 11. RNG semantics

`DataLoader.__iter__()` is a generator; its one RNG draw (the shuffle
permutation) executes lazily, on the *first* `next()` call, not at `iter()`
time. `_PrefetchIterator.__init__` always performs exactly one `next()` call
on the wrapped iterator itself, synchronously, on the calling thread,
*before* starting the background thread -- forcing that one draw to happen
safely, single-threaded. Only batches after that point (which draw no
further randomness -- `DataLoader._collate` touches no RNG) are ever
produced by the background thread. This matters because Forge's
process-global `forge.random.default_generator()` is not thread-safe, and a
model's `Dropout` layer draws from that same generator during the main
thread's forward pass -- without this ordering, a `shuffle=True` loader
combined with `Dropout` could race two threads on one `numpy.random.
Generator`. `tests/test_dataloader_prefetch.py`'s shuffle-ordering tests
(parametrized over `prefetch_size`) assert byte-for-byte identical batch
order between the synchronous and prefetch paths for a fixed seed.

## 12. Trainer integration

`Trainer(..., prefetch=True, prefetch_size=2)` (default `prefetch=False`,
requires `device='cuda'`) transparently wraps whatever loader `fit()`/
`evaluate()` receives in a cached `CUDAPrefetchLoader` (keyed by `id(loader)`
so the same loader object is wrapped, and its transfer stream created,
exactly once per `Trainer` -- not once per epoch). A loader that is already
a `CUDAPrefetchLoader` is used as-is. Because the wrapped batches arrive
already on `self.device`, `Trainer`'s existing `_to_device_batch()` ->
`x.to(self.device)` call becomes a no-op (same-device early return) with
zero code changes to that method. See `docs/architecture/training-engine.md`
for `Trainer`'s general device-placement contract, unaffected otherwise.

## 13. Memory impact

Measured directly (`tests/test_dataloader_prefetch.py::
test_repeated_epochs_do_not_grow_cuda_or_pinned_memory`, and
`benchmarks/async_dataloader_bench.py`'s MNIST section): CUDA active bytes
and pinned active bytes both return to their pre-loop baseline after
`forge.cuda.empty_cache()`, across 20+ repeated epochs -- no growth with
epoch count. In-flight GPU/pinned memory overhead is bounded by the
double-buffering design (Section 3): at most ~2 batches' worth of CUDA
memory and ~2 batches' worth of pinned host memory are ever live at once,
independent of `prefetch_size` (which only bounds *CPU*-side lookahead).

## 14. Error handling

A `Dataset`/transform exception raised inside the background CPU producer is
caught, wrapped, and re-raised on the consumer's next `__next__()` call
(never swallowed, never deadlocks) -- `tests/test_dataloader_prefetch.py::
test_dataset_exception_propagates_to_consumer`.

## 15. Cleanup semantics

`_PrefetchIterator._stop()` drains the queue (unblocking a producer stuck on
`queue.put()` after early termination) and joins the thread; called both on
normal exhaustion and from `__del__`. Deliberately built to be collectible
by **plain CPython refcounting**, matching this codebase's established
lifetime-testing convention (`docs/development/progress.md`'s CUDA test
notes) -- the background thread's target is a free function taking only
`(source_iter, queue)`, never a bound method closing over the iterator
itself, specifically to avoid a `self -> Thread -> bound-method -> self`
reference cycle that would otherwise require the cyclic GC (an unpredictable
delay) to ever run before resources are released. Verified with `gc.
disable()` in effect: `tests/test_dataloader_prefetch.py::
test_early_termination_stops_background_thread_via_refcounting`.

## 16. Performance expectations

`benchmarks/async_dataloader_bench.py` (`python -m
benchmarks.async_dataloader_bench`) measures both a synthetic workload (CPU
prep / GPU compute cost independently controllable) and the real M20 MNIST
CNN. See `docs/performance/benchmarking.md`'s **Milestone 30** section for
full numbers; summary from the 940MX:

- Negligible CPU work + light GPU work: prefetch is *slower* (~0.63x) --
  pinning/threading overhead with nothing to hide it behind is a real,
  expected cost, not a bug.
- Heavy CPU work + light GPU work: ~0.92x -- still no net win, because a
  light-GPU-work step spends little time blocked inside a synchronizing
  CUDA call (the one window the background thread's Python bytecode can run
  concurrently in, given the GIL), so the background thread gets little
  opportunity to make progress concurrently with the main thread either.
- Light CPU work + heavy GPU work: **~3.28x** -- real, substantial overlap,
  confirming the transfer-stream/compute-stream pipeline genuinely executes
  concurrently on the device.
- Real MNIST CNN (synthetic batches, `examples.mnist.model.build_model()`):
  **~1.2x** -- a modest but real, honestly-measured improvement for a
  realistic workload.

## 17. Current limitations

- The background CPU-producer thread does pure Python/NumPy work only
  (`Dataset.__getitem__` + `DataLoader._collate`) and never touches CUDA or
  any stream -- deliberate (Section 18 below), not an oversight.
- Real transfer/compute overlap (not just correctness) requires the
  consumer to run on a genuine non-default CUDA stream; plain iteration of
  a `CUDAPrefetchLoader` outside `Trainer` on the default stream remains
  fully correct but may show little or no measured overlap, since CUDA's
  legacy default stream synchronizes against any explicitly created stream.
- No pinned caching allocator (reuses Milestone 29's direct
  `cudaHostAlloc`/`cudaFreeHost`-per-allocation policy, unchanged).
- GPU-side lookahead depth is fixed at 1 (double buffering); only the
  CPU-side queue depth is configurable via `prefetch_size`.
- `DataLoader` generator resume state (shuffle order mid-epoch) is not, and
  has never been, part of Forge's checkpoint format (Milestone 18's
  existing limitation) -- prefetching does not change this either way.

## 18. Threading model: why a background thread only ever does CPU work

`forge/backend/cuda/stream.py`'s current-stream state is one process-global
variable, explicitly documented as relying on "Forge is single-threaded
elsewhere." A background thread that itself called `forge.cuda.stream(...)`/
`Tensor.to(..., non_blocking=True)` would race the main thread on that
global: if the background thread temporarily sets it to the transfer stream
while submitting a copy, and the main thread happens to launch a kernel in
that exact window, that kernel would silently execute on the wrong stream --
a correctness bug, not just a performance one. A coarse lock serializing all
CUDA-touching work between the two threads was considered and rejected: it
would need to be held for each kernel's launch *and* its automatic
default-stream host-synchronize (there is no finer chokepoint), which
serializes the two threads for the entire duration of any blocking CUDA
call -- eliminating the concurrency benefit a second thread would otherwise
provide, while keeping all of its complexity and risk.

The design actually used sidesteps this entirely: the background thread
does **only** the pure-Python/NumPy work `DataLoader.__iter__()` already
does (dataset indexing, transform application, `np.stack`) and never calls
into `forge.cuda` at all; **all** CUDA/stream-touching work (pinning,
submitting the async H2D copy, and everything the caller does afterward)
happens on the single calling thread. Genuine overlap is still achieved
two ways: (1) real CUDA-stream-level concurrency between the transfer
stream and a non-default compute stream, entirely independent of Python
threading (Section 16's 3.28x synthetic result); and (2) whenever the
calling thread is blocked inside a `ctypes` call to a blocking CUDA
operation (e.g. a default-stream kernel's implicit `cudaDeviceSynchronize`,
or `Tensor.to("cpu")` reading a pending result), `ctypes` releases the GIL
for that call's duration -- exactly the window in which the background
thread's Python bytecode can run. No thread-local stream state, and no
locking, needed.
