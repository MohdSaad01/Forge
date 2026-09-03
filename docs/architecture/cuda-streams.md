# CUDA Streams and Asynchronous Execution (Milestone 27)

Milestone 26 formally established Forge's Milestone 8-26 CUDA execution
model: every kernel launch and `cudaMemcpy` runs on CUDA's default (null)
stream, and every `CUDABackend` method calls `cudaDeviceSynchronize()` before
returning its result, so Forge's CUDA execution is host-synchronous *per
operation* despite the underlying CUDA calls being individually
asynchronous. That document's **Future Stream-Aware Design** section listed
exactly what introducing streams would require. This milestone is that work.

## 1. Stream abstraction

`forge.backend.cuda.stream.CUDAStream` (public alias: `forge.cuda.Stream`)
wraps one real `cudaStreamCreate`d handle:

```python
s = forge.cuda.Stream()   # cudaStreamCreate() -- a real, distinct CUDA stream
s.synchronize()           # cudaStreamSynchronize(s) -- waits for only this stream
s.destroy()               # cudaStreamDestroy(s) -- explicit; also runs on __del__
```

`Stream()` construction and destruction are real CUDA runtime calls
(`cf_stream_create`/`cf_stream_destroy`, `kernels.cu`) -- never simulated,
and never the default-stream handle wearing a `Stream` costume. Multiple
`Stream()` calls produce distinct, independently identifiable handles
(`s.handle`, a `ctypes.c_void_p`; `repr(s)` shows it), verified directly on
the 940MX by `tests/test_cuda_streams.py::test_two_streams_have_distinct_handles`.

No public CUDA event API, stream priorities, stream pools, or CUDA Graphs
are exposed -- exactly the milestone brief's scope limit (Section 6/45).

## 2. Default stream compatibility mode

```python
forge.cuda.current_stream()   # -> Stream | None
```

`None` means CUDA's own default (null) stream -- and it is the state Forge
starts in and returns to after every `with forge.cuda.stream(s):` block
exits. **This preserves the exact Milestone 8-26 contract unchanged**: with
no active stream context, every `CUDABackend` operation still calls
`cudaDeviceSynchronize()` before returning (see `_maybe_synchronize` below),
so existing code that has never heard of `forge.cuda.Stream` behaves
identically to before this milestone -- verified directly by re-running the
entire pre-existing CUDA test suite (`tests/test_cuda_*.py`, 380 tests)
completely unmodified against this milestone's code; all 380 pass.

**Why this design, not "everything asynchronous by default."** The
alternative the brief explicitly floats (Section 9) -- making every CUDA
operation asynchronous by default and providing an *opt-in* synchronous
compatibility mode -- was rejected because it would silently invalidate an
assumption the entire pre-existing test suite depends on: many tests read a
CUDA result back to the host (`.to("cpu")`, `float(loss.to("cpu").numpy())`)
immediately after an operation with no explicit synchronization call, relying
exactly on the M26 "every op already synchronized" guarantee. Flipping the
default would require auditing and touching dozens of pre-existing tests
(and worse, any downstream user code) for a milestone whose own stated goal
is incremental, controlled scope. Making the *default* stream keep its
historical host-synchronous behavior, and asynchronous execution only
*opt-in* via an explicit `with forge.cuda.stream(s):` block, achieves
Milestone 27's real goal -- real asynchronous execution exists and is
demonstrably used -- with zero risk to existing correctness.

## 3. Current stream and the stream context

```python
forge.cuda.current_stream()      # the stream new CUDA ops execute on right now
forge.cuda.set_stream(s)         # make `s` current, returns the previous one
with forge.cuda.stream(s):       # make `s` current for the block, then restore
    ...
```

Implemented in `forge/backend/cuda/stream.py` as one process-global variable
(`_current_stream`), not a per-thread/contextvar mechanism -- matching the
"Forge is single-threaded elsewhere" convention `allocator.py`'s own
docstring already states. `with forge.cuda.stream(s):` is a plain
try/finally around `set_stream()`, so nested contexts restore correctly even
through an exception (`tests/test_cuda_streams.py::
test_stream_context_restores_previous_stream_even_on_exception`) and nesting
composes as expected (`with stream(a): with stream(b): ...` restores to `a`
then to `None`).

No Tensor operation takes a stream argument. Every `CUDABackend` method
reads `current_stream()` itself at the start of the call
(`CUDABackend._stream_handle()`/`_maybe_synchronize()`/`_stream_guard()`),
so `Tensor`/`Module`/`Optimizer`/autograd code is completely unaware of
streams -- the milestone brief's explicit "do not require users to pass a
stream argument to every Tensor operation."

## 4. Asynchronous execution model

Every kernel-launching `CUDABackend` method now follows:

```text
kernel launch (onto _stream_handle(), current_stream()'s raw handle or NULL)
    -> cudaGetLastError()
    -> CUDABackend._check() raises CUDAError if nonzero
    -> _maybe_synchronize(action)
         current_stream() is None?  -> cudaDeviceSynchronize()  (M26 behavior)
         current_stream() is a Stream? -> nothing (async)
```

Inside a `with forge.cuda.stream(s):` block, the per-operation
`cudaDeviceSynchronize()` is skipped entirely -- this is the entire point of
Milestone 27. `kernels.cu`'s `*_LAUNCHER` macros were extended with a
trailing `void* stream` parameter (`kernel<<<blocks, threads, 0,
(cudaStream_t)stream>>>(...)`, `stream=NULL` meaning the default stream --
identical launch configuration to before this milestone when no explicit
`Stream` is current). All ~40 kernel-launching `CUDABackend` methods were
updated identically: `self._stream_handle()` appended as the trailing
`ctypes` argument, `self._synchronize(action)` replaced with
`self._maybe_synchronize(action)`. **Every operation in Section 32 of the
milestone brief respects the current stream** -- `add`/`sub`/`mul`
(exact-shape and both broadcast kinds), `matmul`, `sum` (full and axis=1),
`reshape`, `relu`, `exp`/`log`, every backward kernel, `Conv2d`/`MaxPool2d`
(forward and backward), `Dropout`, `SGD`/`Adam` -- no exceptions.

`forge.cuda.synchronize()` remains an unconditional, real
`cudaDeviceSynchronize()`: it waits for *every* stream on the device, not
just the current one, so it remains a correct (if coarse) barrier regardless
of how many `Stream`s exist. `stream.synchronize()` (`cudaStreamSynchronize`)
waits for only that one stream.

## 5. Memory copy semantics

`cf_memcpy_h2d`/`cf_memcpy_d2h`/`cf_memcpy_d2d` (`kernels.cu`) are
**unchanged** -- still plain, synchronous `cudaMemcpy`, never
`cudaMemcpyAsync`, no stream parameter added. The milestone brief explicitly
sanctions this (Section 33: "do not introduce `cudaMemcpyAsync` unless
required for the stream implementation"; it was not required). Under CUDA's
legacy default-stream semantics (`build.py` does not pass
`--default-stream per-thread` to `nvcc`), a plain `cudaMemcpy` on the
implicit null stream is *itself* an implicit whole-device barrier: it waits
for every other stream's prior work before it starts, and every other
stream's later work waits for it to finish. This makes `reshape()` (a
device-to-device copy) and `from_array()`'s CUDA-to-CUDA clone path
unconditionally safe regardless of the current stream, with no code change
needed for correctness -- `CUDABackend._stream_guard()` is still called on
their input storages (via `reshape`/`from_array`), but only for a clear
Forge-level error message and to keep `last_stream` bookkeeping accurate,
not because the underlying copy would otherwise be unsafe.

`to_numpy()` (the D2H direction, backing `Tensor.to("cpu")`) additionally
calls `storage.last_stream.synchronize()` *explicitly*, before the
`cudaMemcpy`, whenever `last_stream` is a real (non-default) stream -- this
keeps `to_numpy()`'s "always host-blocking, always correct" contract
self-evident in the Python source rather than relying only on the implicit
legacy-stream ordering described above, and it waits for only that one
storage's producing stream, not the whole device.

**No nonblocking `.to()` was introduced.** `Tensor.to("cuda")`/`.to("cpu")`
remain fully host-blocking, exactly as before this milestone (Section 24 of
the brief explicitly rules nonblocking transfers out of scope).

## 6. Tensor / storage lifetime vs. GPU execution lifetime

This is the reason Milestone 27 exists (Section 19 of the brief). Before
this milestone, `del tensor` running `CUDAStorage.__del__` *implied* "the
GPU has finished using this memory," because the M26 contract guaranteed
every operation had already synchronized by the time Python could see the
result. That implication is no longer true for a storage produced or last
used inside a `with forge.cuda.stream(s):` block: `del` can run while `s`'s
kernel is still executing on the device.

`CUDAStorage` (`forge/backend/cuda/backend.py`) gains one field:
`last_stream` -- the `Stream` this storage was last touched by (as an input
*or* output of any kernel-launching operation), or `None` if it has only
ever been touched on the default stream. This is deliberately the *only*
piece of stream history tracked (per the brief's "do not attach a full
stream history to every Tensor"): not a list of streams, not a per-operation
event log.

- Set automatically in `CUDAStorage.__init__` to whatever
  `current_stream()` is at construction time -- correct because every
  `CUDAStorage` is constructed as the immediate result of an operation that
  just launched on the current stream.
- Refreshed on every *input* storage a kernel-launching operation touches,
  by `CUDABackend._stream_guard()` (called from `_require_compute_dtype()`,
  which nearly every kernel-launching method already calls with its full
  storage argument list -- see Section 7 below) and explicitly by
  `reshape()`/`from_array()`'s CUDA-to-CUDA branch.
- Read by `CUDAStorage.__del__` to decide how to release the block (Section
  8) and by `to_numpy()` to decide whether an explicit stream-specific
  synchronize is needed before a D2H read (Section 5).

`CUDAStorage.last_stream` holds a *strong reference* to the `Stream` object,
not just its raw handle -- deliberately. It keeps the stream's underlying
CUDA resource alive at least until this storage's own `__del__` has recorded
a completion event on it (Section 8); once an event has been recorded,
`cudaEventQuery`/`cudaEventSynchronize` remain valid to call even after the
stream itself is later destroyed (documented CUDA runtime behavior -- see
`stream.py`'s `CUDAStream` docstring), so nothing needs to keep the stream
object alive past that point.

## 7. Cross-stream dependency policy: fail clearly

Section 20/21 of the milestone brief explicitly sanctions this: "if
cross-stream execution is not supported for arbitrary Tensor dependencies
yet, fail clearly rather than silently producing incorrect results." Forge
does the former, not full `cudaStreamWaitEvent`-based cross-stream
dependency resolution (explicitly out of scope -- "do not attempt arbitrary
graph-level multi-stream scheduling").

`CUDABackend._stream_guard(storages, op)` runs before every kernel launch
(via `_require_compute_dtype`, and explicitly in `reshape`/`from_array`):
for each input storage, if `storage.last_stream` is not `None` and is not
identical to the stream this operation is about to launch on, it raises
`forge.CUDAError` immediately, naming both streams. A storage whose
`last_stream` is `None` (only ever touched on the default stream) is always
safe to read from any stream -- the M26 contract already guarantees that
work completed, synchronously, before Python could see the storage at all.

```text
Stream A: x = a + b            # x.last_stream = A
Stream A: y = x + x            # OK -- x.last_stream is A, current stream is A
Stream B: z = x + x            # CUDAError -- x.last_stream is A, current stream is B
(no stream) w = x + x          # CUDAError -- x.last_stream is A (real), current is None
```

Verified directly: `tests/test_cuda_stream_allocator.py::
test_using_a_tensor_across_two_different_explicit_streams_raises_clearly`,
`tests/test_cuda_stream_autograd.py::
test_backward_on_a_different_stream_than_forward_raises_cuda_error` (forward
on stream A, `backward()` attempted under stream B).

**Working across streams remains possible** -- just not implicitly. Call
`stream.synchronize()` (or `forge.cuda.synchronize()`) between uses on
different streams to reset `last_stream` back to a state (default,
effectively) where any stream can safely consume it. Explicit synchronization,
not implicit dependency tracking, is Forge's Milestone 27 answer to
cross-stream reads -- exactly the smallest correct mechanism the brief calls
for (Section 12).

## 8. Allocator changes: pending blocks

The M25/M26 allocator's central assumption -- "release implies safe to
reuse immediately" -- is no longer universally true. `forge/backend/cuda/
allocator.py`'s `CUDACachingAllocator` now tracks three states per block:

- **Active**: owned by a live `CUDAStorage` (`_active_bytes`).
- **Ready** (`_free_blocks`): released by a storage whose `last_stream` was
  `None` (default stream) -- safe to hand to a new `CUDAStorage` immediately,
  no driver call, exactly the M25/M26 behavior, completely unchanged.
- **Pending** (`_pending_blocks`, new): released by a storage whose
  `last_stream` was a real `Stream` -- not yet known to be safe.

`CUDAStorage.__del__` dispatches on `last_stream`:

```text
last_stream is None    -> allocator.release(nbytes, ptr)           # unchanged M25/M26
last_stream is a Stream -> allocator.release_pending(lib, nbytes, ptr, last_stream.handle)
```

`release_pending()` creates a real `CUDAEvent` (`forge/backend/cuda/
stream.py`, internal-only -- no public event API) and calls
`cudaEventRecord(event, last_stream)` **right now**, synchronously, from
Python. This is correct despite `__del__` running arbitrarily long after the
kernels that used this storage were launched: because CUDA streams execute
in strict program (FIFO) order, and every operation that touched this
storage was necessarily enqueued on `last_stream` *before* `__del__` could
possibly run (Python is single-threaded; nothing else could have released
this storage earlier), recording the event now and waiting for *it* to
complete is equivalent to waiting for every earlier operation on that stream
to complete too.

The block becomes reusable the moment `CUDAEvent.query()` reports true
(`cudaEventQuery() == cudaSuccess`) -- checked opportunistically inside
`CUDACachingAllocator.allocate()` on the next same-size request
(`_try_reclaim_pending`), *after* the ready free list is checked and
*before* falling through to a real `cudaMalloc`. A pending block is
otherwise left alone: Forge never calls `cudaDeviceSynchronize()` or forces
an event to complete merely to make a block reusable sooner (that would
defeat the entire point of asynchronous execution) -- confirmed directly by
`tests/test_cuda_stream_allocator.py::
test_same_stream_rapid_release_and_reallocate_never_corrupts_data`, which
shows the allocator *may* fall through to a fresh `cudaMalloc` rather than
reuse a not-yet-complete pending block, and that this is still always
correct.

## 9. Same-stream reuse

Releasing and reallocating the same byte count on the *same* stream is
always eventually reusable (once that stream's own earlier work completes)
and always correct in the meantime, because CUDA's per-stream program order
guarantees the new allocation's writes are enqueued strictly after the old
one's last use, regardless of which physical block backs either one. Two
outcomes were both observed directly on the 940MX
(`tests/test_cuda_stream_allocator.py`):

- If the pending event has already completed by the time the same-size
  request arrives, `_try_reclaim_pending` reuses the exact block (a cache
  hit, no driver call).
- If it has not, `allocate()` falls through to a fresh `cudaMalloc` (a cache
  miss) rather than wait -- still fully correct, simply a missed reuse
  opportunity (see Section 40's "maximum reuse is not mandatory,
  correctness is" from the brief).

## 10. Cross-stream reuse

Stream A releases a block; Stream B immediately requests the same byte
count. The allocator must never (and, verified directly, does not) hand
Stream B a block Stream A might still be reading or writing:
`_try_reclaim_pending` only ever returns a pending block whose *own*
recorded event has completed -- it has no notion of "which stream is
asking," so a still-in-flight block from Stream A is simply invisible to
Stream B's request too, and Stream B instead gets a fresh `cudaMalloc` (or a
genuinely-safe ready/pending block of the right size). Verified directly:
`tests/test_cuda_stream_allocator.py::
test_cross_stream_reuse_never_hands_out_a_still_in_flight_block` runs real,
data-dependent computation on both streams concurrently and confirms neither
stream's values were ever corrupted by the other's allocation activity.

## 11. `empty_cache()` under asynchronous execution

```text
empty_cache()
    -> _empty_ready(lib)    # ready blocks: cudaFree immediately, no waiting (unchanged M25/M26)
    -> _drain_pending(lib)  # pending blocks: CUDAEvent.synchronize() (wait), then cudaFree
```

**Cost changed.** A ready block is freed exactly as before -- no waiting
needed, since it was already guaranteed complete when it entered the ready
list. A pending block, however, may genuinely still be in flight, so
`_drain_pending` calls `CUDAEvent.synchronize()` (a real, blocking
`cudaEventSynchronize()`) for each one before `cudaFree`ing it. This means
**`forge.cuda.empty_cache()` can now block the calling thread** if
asynchronous work has not finished -- a real, documented behavior change
from M25/M26 (where `empty_cache()` never needed to wait for anything).
This is the brief's own explicitly sanctioned policy (Section 18: "pending
blocks -> wait until safe -> `cudaFree`"), chosen because `empty_cache()` is
an occasional, caller-invoked operation where paying a wait to guarantee
"every non-active byte was actually returned to the driver" is a reasonable
trade -- unlike per-operation synchronization, which would defeat
asynchronous execution entirely if reintroduced here. Verified directly:
`tests/test_cuda_stream_allocator.py::
test_empty_cache_drains_pending_blocks_and_returns_them_to_the_driver`.

## 12. Memory statistics

`CUDAMemoryStats` (`forge/backend/cuda/allocator.py`) gains two fields:

- `pending_bytes` -- bytes released by a storage last used on a non-default
  stream, not yet confirmed safe to reuse.
- `pending_count` -- number of individual pending blocks (all sizes).

`cached_bytes` now means specifically *ready* (immediately reusable, no
wait) bytes -- `reserved_bytes - allocated_bytes - pending_bytes` (was
`reserved_bytes - allocated_bytes` before pending blocks existed; the two
definitions agree whenever `pending_bytes == 0`, i.e. every pre-Milestone-27
scenario). `reserved_bytes` is unchanged in meaning: `active + ready cached +
pending` -- verified by `tests/test_cuda_stream_allocator.py::
test_memory_stats_reserved_equals_active_plus_cached_plus_pending`. Every
M22/M25 field (`allocated_bytes`, `peak_allocated_bytes`, `allocation_count`
/`cuda_malloc_count`, `free_count`/`cuda_free_count`, `cache_hit_count`,
`cache_miss_count`) keeps its exact prior meaning -- a reclaimed pending
block counts as a cache hit (Section 8), same as a ready-cache hit always
has.

## 13. Autograd semantics

`Tensor.backward()` drives `forge.autograd.engine.run_backward()`, which is
backend-agnostic and calls each `Node.backward_fn` (a closure over
`CUDABackend.*_backward` methods). Since every `CUDABackend` method reads
`current_stream()` itself (Section 3), **no change was needed anywhere in
`forge/tensor/tensor.py` or `forge/autograd/engine.py`** for backward
computation to correctly execute on whatever stream is current at the
moment each op runs -- ambient/global stream state, not something threaded
through function signatures, generalizes automatically. A full `with
forge.cuda.stream(s): output = model(x); loss = criterion(output, target);
loss.backward()` block therefore runs every forward *and* backward op on
`s`, correctly ordered by `s`'s own program order.

Cross-stream backward (forward issued under stream A, `backward()` called
under stream B) fails clearly via `_stream_guard` (Section 7) -- verified by
`tests/test_cuda_stream_autograd.py::
test_backward_on_a_different_stream_than_forward_raises_cuda_error`.

## 14. Optimizer semantics

`SGD.step()`/`Adam.step()` call `CUDABackend.sgd_step()`/`adam_step()`
directly -- ordinary stream-aware `CUDABackend` methods, no changes needed
in `forge/optim/`. A parameter update issued under `with forge.cuda.
stream(s):` is stream-ordered against every subsequent op issued on `s`: a
forward pass immediately after `optimizer.step()` on the same stream
correctly observes the updated parameter values, with no explicit
synchronization needed in between (CUDA's own per-stream ordering
guarantees this) -- verified directly by `tests/test_cuda_stream_autograd.py::
test_optimizer_step_then_forward_on_same_stream_sees_updated_parameters`.

## 15. Trainer semantics

**Decision: Option A (Trainer remains synchronous).** `forge/training/
trainer.py` was **not modified** for this milestone, and does not use
`forge.cuda.stream()` internally. Since `current_stream()` defaults to
`None` (Section 2) everywhere `Trainer` runs, every `Module`/`Loss`/
`Optimizer` call it makes remains exactly as host-synchronous as it always
was (M26's contract, unchanged), and `Trainer.fit()`/`evaluate()` continue
to return only after all CUDA work issued during that call has completed --
trivially satisfied, not newly implemented. Overlapping DataLoader/transfer/
compute inside `Trainer` is explicitly out of scope for this milestone
(Section 23/45 of the brief); a future milestone that wants `Trainer` to use
an explicit stream internally can build on this contract directly.

## 16. Persistence / checkpoint semantics

`save_model()`/`save_checkpoint()` (`forge/serialization/`) read every
parameter via `Backend.to_numpy()`. Since `to_numpy()` now synchronizes a
storage's own `last_stream` before its D2H copy (Section 5), no caller-side
`stream.synchronize()`/`forge.cuda.synchronize()` is required before saving
a model whose parameters were last updated inside a `with forge.cuda.
stream(s):` block -- verified directly (no explicit sync in either test) by
`tests/test_cuda_stream_autograd.py::
test_save_model_after_async_work_on_a_stream_round_trips_correctly` and
`::test_save_checkpoint_after_async_work_on_a_stream_round_trips_correctly`.
No serialization code changed.

## 17. Multi-stream overlap results

`benchmarks/stream_bench.py` (`python -m benchmarks.stream_bench`)
demonstrates real overlap on the 940MX -- full methodology and numbers live
in `docs/performance/benchmarking.md`'s **Milestone 27** section (not
repeated here to avoid two copies of the same table drifting apart).
Summary: a default-stream baseline (no explicit streams, every op
synchronizes) took a median 87.05 ms for two independent chained-add
workloads; issuing the same workloads on two real streams but synchronizing
between them (no overlap, only the per-op synchronization removed) took
41.62 ms; issuing them concurrently on both streams with synchronization
only at the end took 36.55 ms -- a further, real ~1.14x speedup
attributable specifically to overlapping kernel execution across the
device's 3 SMs. A large-single-kernel configuration measured only ~1.01x
overlap, since one such kernel already occupies the whole device -- both
outcomes are honestly reported (see Section 29 of the milestone brief: "do
not expect dramatic overlap on every kernel/GPU").

## 18. Current limitations

- No cross-stream dependency resolution (Section 7) -- an explicit
  `stream.synchronize()` is required to hand a tensor from one stream to
  another.
- No public CUDA event API, stream priorities, stream pools, CUDA Graphs,
  or nonblocking `.to()` -- all explicitly out of scope (Section 45).
- `Trainer` does not use streams internally (Section 15) -- DataLoader/
  transfer/compute overlap remains a future milestone.
- The allocator's pending-block scan (`_try_reclaim_pending`) is a linear
  scan over pending blocks of the requested size -- fine at the "small
  number of streams" (2-8) scale the brief targets, not designed for a large
  number of concurrent streams.
- `cf_memcpy_h2d`/`_d2h`/`_d2d` remain synchronous, legacy-default-stream
  `cudaMemcpy` (Section 5) -- correct, but each one is a coarse whole-device
  ordering point; a future milestone introducing pinned memory and
  `cudaMemcpyAsync` could remove that cost.

## 19. Future stream/event design

Explicitly out of scope for this milestone, listed so a future milestone
does not have to rediscover it:

- **General cross-stream dependencies**: `cudaStreamWaitEvent`-based
  automatic dependency insertion (rather than "fail clearly, synchronize
  explicitly") so a tensor could move between streams without an explicit
  barrier.
- **`Trainer`-internal stream use**: a designated training stream with
  synchronization only at public method boundaries (Option B from Section
  23 of the brief), enabling overlap between data loading/transfer and
  compute.
- **Nonblocking `.to()`**: pinned host memory + `cudaMemcpyAsync`, with an
  explicit story for when the destination tensor's data is guaranteed
  ready.
- **A public CUDA event API**: `forge.cuda.Event` for user-level fine-grained
  synchronization, rather than the allocator-internal-only `CUDAEvent` this
  milestone introduces.
- **Stream priorities / pools**: per-thread or scheduler-managed stream
  pools, useful once Forge has a caller that actually needs more than a
  handful of manually-created streams.

None of this is implemented now.
