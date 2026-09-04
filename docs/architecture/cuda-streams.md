# CUDA Streams and Asynchronous Execution (Milestone 27; cross-stream dependencies in Milestone 28)

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

## 7. Cross-stream dependency policy: automatic GPU-side dependencies (Milestone 28)

**Superseded from Milestone 27.** M27 chose to raise `forge.CUDAError`
immediately on any cross-stream Tensor use ("fail clearly rather than
silently produce incorrect results" -- that milestone's Section 20/21).
Milestone 28's brief explicitly removes that limitation: "Forge should be
able to determine when a CUDA Tensor was last produced/used on another
stream and establish the necessary dependency automatically." This section
describes the current (M28) behavior; see **Milestone 28: Automatic
Cross-Stream Dependencies** below for the full design discussion.

`CUDABackend._stream_guard(storages, op)` still runs before every kernel
launch (via `_require_compute_dtype`, and explicitly in `reshape`/
`from_array`) -- the same single chokepoint as M27, unchanged in *where* it
runs, changed in *what it does*. For each input storage whose `last_stream`
is not `None` and not identical to the stream this operation is about to
launch on, it now inserts a real, GPU-side dependency -- `cudaEventRecord`
on the producing stream, then `cudaStreamWaitEvent` on the consuming stream
-- instead of raising:

```text
Stream A: x = a + b            # x.last_stream = A
Stream A: y = x + x            # same-stream fast path -- no event, no wait
Stream B: z = x + x            # cudaEventRecord(A) + cudaStreamWaitEvent(B) -- then OK
(no stream) w = x + x          # cudaEventRecord(A) + cudaStreamWaitEvent(NULL) -- then OK
```

A storage whose `last_stream` is `None` (only ever touched on the default
stream) still never needs a dependency -- the M26 contract already
guarantees that work completed, synchronously, before Python could see the
storage at all.

Verified directly: `tests/test_cuda_stream_allocator.py::
test_using_a_tensor_across_two_different_explicit_streams_establishes_a_dependency`,
`tests/test_cuda_stream_autograd.py::
test_backward_on_a_different_stream_than_forward_matches_same_stream_reference`
(forward on stream A, `backward()` on stream B -- now produces the same
gradients as an identical run kept on one stream, not a `CUDAError`), and
the whole of `tests/test_cuda_stream_dependencies.py`.

**Explicit synchronization remains available, just no longer required.**
`stream.synchronize()`/`forge.cuda.synchronize()` between cross-stream uses
still work exactly as before (M26/M27) -- they simply become optional for
correctness in the cases `_stream_guard` now covers automatically.

## 8. Allocator changes: pending blocks (Milestone 27; unaffected by Milestone 28)

Milestone 28's cross-stream dependencies changed *when* `_stream_guard`
raises versus proceeds, not the allocator's own pending-block model below --
`CUDAStorage.__del__`, `release()`/`release_pending()`, and
`_try_reclaim_pending()` are byte-for-byte unchanged from M27. The one
interaction worth stating explicitly: `_stream_guard` now updates a
*read-only* input's `last_stream` to the consuming stream too (not just a
freshly written output's), exactly as M27 already did -- so a storage
consumed cross-stream still ends up with `last_stream` naming whichever
stream will *actually* next touch it, which is exactly what
`release_pending()` needs to record the correct event when that storage is
later released. See **Milestone 28: Automatic Cross-Stream Dependencies**
below, **Storage lifetime and the allocator**, for why this remains safe.

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

**Milestone 28:** cross-stream backward (forward issued under stream A,
`backward()` called under stream B) is now automatically safe via
`_stream_guard` (Section 7) -- every backward kernel reads activations/
weights last touched on stream A from stream B, and `_stream_guard`
establishes the needed dependency for each one, with no code change to
`forge/autograd/engine.py` or `forge/tensor/tensor.py` (the same "ambient
stream state, not threaded through function signatures" property that made
same-stream backward work automatically in M27 generalizes to cross-stream
backward automatically too). Verified by `tests/test_cuda_stream_autograd.py::
test_backward_on_a_different_stream_than_forward_matches_same_stream_reference`,
which checks the resulting gradients exactly match an identical run kept on
one stream (previously, M27's identically-named test asserted a
`CUDAError`).

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

**Milestone 28:** an optimizer step consuming a gradient produced on a
*different* stream, or reading/writing a parameter last touched on a
different stream, is also safe -- `sgd_step`/`adam_step` call
`_require_compute_dtype` (hence `_stream_guard`) exactly like every other
`CUDABackend` method, so cross-stream dependencies are established
automatically with no optimizer-specific code. Verified by
`tests/test_cuda_stream_autograd.py::
test_optimizer_step_on_a_different_stream_than_backward_matches_same_stream_reference`
and `::test_parameter_read_and_update_across_streams_are_never_racy_in_either_order`
-- the latter alternates a parameter *read* (forward) and *write*
(optimizer step) across two streams in both orders with no explicit
synchronization anywhere in between, and matches an identical sequence kept
on one stream exactly (Section 39 of the M28 brief: "one of the most
important correctness tests in the milestone").

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

- No public CUDA event API, stream priorities, stream pools, CUDA Graphs,
  or nonblocking `.to()` -- all explicitly out of scope (Milestone 27's
  Section 45; reaffirmed as out of scope, or optional-and-not-taken, by the
  Milestone 28 brief's Sections 7/52).
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
- **Milestone 28**: `_stream_guard`'s dependency insertion is a fresh
  `CUDAEvent` per distinct producer stream on *every* cross-stream op --
  correct and, per the M28 brief's own Section 6, deliberately not pooled
  ("first implement correctness ... optimize only if profiling justifies
  it"); `benchmarks/stream_dependency_bench.py`'s "event creation" number
  (~2.6 us on the 940MX) is the isolated per-event cost this would target if
  a future milestone's profiling shows it matters. Only one piece of
  producer-stream history (`CUDAStorage.last_stream`) is tracked, per the
  brief's "do not attach a full stream history" constraint -- see
  **Milestone 28: Automatic Cross-Stream Dependencies**, **Why one
  `last_stream` field is still enough**, below for why this remains
  sufficient even for multi-consumer/multi-producer graphs.

## 19. Future stream/event design

Explicitly out of scope through Milestone 28, listed so a future milestone
does not have to rediscover it:

- **`Trainer`-internal stream use**: a designated training stream with
  synchronization only at public method boundaries (Option B from Section
  23 of the M27 brief), enabling overlap between data loading/transfer and
  compute.
- **Nonblocking `.to()`**: pinned host memory + `cudaMemcpyAsync`, with an
  explicit story for when the destination tensor's data is guaranteed
  ready.
- **A public CUDA event API**: `forge.cuda.Event` for user-level fine-grained
  synchronization, rather than the internal-only `CUDAEvent` Forge has
  through Milestone 28. The M28 brief explicitly left this optional
  (Section 7); it was not introduced, since `_stream_guard`'s automatic
  dependency insertion already meets the milestone's actual acceptance
  criterion (cross-stream correctness) with no user-facing API needed.
- **Stream priorities / pools**: per-thread or scheduler-managed stream
  pools, useful once Forge has a caller that actually needs more than a
  handful of manually-created streams.
- **Event pooling**: see Section 18's Milestone 28 bullet above.

None of this is implemented now.

## 20. Milestone 28: Automatic Cross-Stream Dependencies

### Mechanism

`CUDABackend._stream_guard(storages, op)` (`forge/backend/cuda/backend.py`),
the single chokepoint every kernel-launching method already ran through in
M27 (Section 7), now does this instead of raising:

```text
for each distinct producer stream P among storages' last_stream (P != current):
    event = CUDAEvent(lib)         # cudaEventCreateWithFlags(cudaEventDisableTiming)
    event.record(P.handle)         # cudaEventRecord(event, P)
    current.wait_event(event)      # cudaStreamWaitEvent(current, event, 0) -- or
                                    # wait_event_on_default_stream() if current is NULL
    # event falls out of scope here -- see "Event lifetime" below
for each storage:
    storage.last_stream = current
```

`CUDAStream.wait_event()` and the free function
`stream.wait_event_on_default_stream()` (`forge/backend/cuda/stream.py`) are
the two thin wrappers around the new `cf_stream_wait_event` export
(`kernels.cu`, a direct `cudaStreamWaitEvent` call) -- the first for an
explicit current stream, the second for the one stream Forge doesn't wrap in
a `CUDAStream` object (`current_stream() is None`, meaning CUDA's default/
null stream). Both compile to the identical underlying CUDA call; `stream=
NULL` waiting on an event is a valid, documented `cudaStreamWaitEvent` usage.

### All four directions

```text
default  -> explicit    : cudaEventRecord(NULL) + cudaStreamWaitEvent(B)
explicit -> default      : cudaEventRecord(A) + cudaStreamWaitEvent(NULL)
explicit A -> explicit B : cudaEventRecord(A) + cudaStreamWaitEvent(B)
explicit B -> explicit A : cudaEventRecord(B) + cudaStreamWaitEvent(A)
```

All four are the *same* code path in `_stream_guard` -- there is no
default-stream special case, deliberately: Section 14 of the milestone brief
warns against "assum[ing] CUDA's legacy/default-stream ordering automatically
solves all cases," so Forge does not rely on the (real, but driver-mode-
dependent) implicit whole-device ordering the legacy default stream provides
for `cudaMemcpy` (Section 5) -- it establishes an explicit, defined
dependency here regardless of which side is the default stream. All four
directions are verified directly on the 940MX by
`tests/test_cuda_stream_dependencies.py::test_all_four_stream_directions_produce_correct_results`
(parametrized over all four).

### Why one `last_stream` field is still enough

The milestone brief's Section 18/19 raise a natural worry: with only one
`last_stream` field per storage (no producer-stream *history*), can a
multi-consumer or multi-producer graph still be handled correctly? Yes,
without any extra metadata, for a subtle but load-bearing reason:

`_stream_guard` updates `last_stream` to the *current* stream for every
storage it touches -- inputs (reads) exactly as much as freshly constructed
outputs, unchanged from M27 (Section 6 there). Consider `x` produced on
stream P, then read by consumer stream A (`_stream_guard` makes A wait for
P, then sets `x.last_stream = A`), then read again later by consumer stream
B. `x.last_stream` is now `A`, not `P` -- so B's `_stream_guard` call
establishes a dependency on *A*, not P. This is still correct, because A's
own command queue already contains "wait for P's event" (enqueued before
A's read of `x`), so waiting for an event recorded on A *after* that read
transitively implies P's work completed too, by CUDA's own per-stream FIFO
program-order guarantee -- not by re-deriving P from anywhere. The
dependency chain is carried entirely by the stream's own queue order, not by
any Forge-side bookkeeping of "who *originally* produced this." This is
conservative (B may wait slightly longer than the true minimum -- for all of
A's read to finish, not just P's write) but always correct, and costs
nothing extra to implement: the exact same one-field, "last toucher" model
M27 already used for allocator safety turns out to already be sufficient
for the dependency-insertion problem too, with no additional history
required. Verified directly:
`tests/test_cuda_stream_dependencies.py::test_multi_consumer_streams_each_see_correct_producer_data`
(two independent consumer streams reading one producer's tensor) and
`::test_multi_producer_streams_each_get_their_own_dependency` (`C = A + B`,
`A` and `B` each on their own stream -- `_stream_guard` iterates every
input storage, so both producers get a dependency, satisfying Invariant 4
independently of the single-field design above).

### Deduplication

`_stream_guard` collects the *distinct* producer streams among an
operation's storages into a `set` before creating any event -- two inputs
last touched on the same other stream cost exactly one event and one wait,
not two (Section 20 of the milestone brief). Verified:
`tests/test_cuda_stream_dependencies.py::test_two_inputs_from_the_same_producer_stream_dedupe_to_one_dependency`.
Redundant *repeated* events across separate operations (e.g. the same
producer/consumer pair used twice in a row) are not deduplicated -- Section
6's "correctness first, pool/optimize only if justified" applies here too;
seep Section 18's Milestone 28 limitations bullet.

### No host blocking

`cudaStreamWaitEvent` only ever inserts a GPU-side ordering point into the
consuming stream's own command queue; the call itself returns to Python
immediately regardless of whether the event has completed -- this is why
`_stream_guard` can run inline on every operation without turning
asynchronous execution back into host-synchronous execution. Verified
directly (not merely assumed): `tests/test_cuda_stream_dependencies.py::
test_cross_stream_dependency_between_two_explicit_streams_never_calls_device_synchronize`
spies on the real `cf_synchronize` (`cudaDeviceSynchronize`) entry point and
confirms it is called zero times while establishing and using a cross-stream
dependency between two explicit streams.

### Storage lifetime and the allocator

Because `_stream_guard` still updates every touched storage's `last_stream`
to the current stream (see "Why one `last_stream` field is still enough"
above), M27's allocator integration (Section 8) needs no change: whichever
stream a storage's `__del__` finds in `last_stream` is genuinely the last
stream that will touch it, cross-stream reads included, so
`release_pending()`'s event-recording remains correct unmodified. Verified:
`tests/test_cuda_stream_dependencies.py::
test_empty_cache_remains_safe_after_cross_stream_dependencies` and
`::test_repeated_cross_stream_dependencies_do_not_grow_cuda_allocation` (100
repeated cross-stream release/reallocate cycles, steady-state
`allocated_bytes` unchanged -- no leak).

### Persistence / checkpointing

Unaffected: `to_numpy()` already synchronizes a storage's own `last_stream`
before its D2H copy (Section 5/6), and `last_stream` remains accurate under
cross-stream reads (above) -- so `save_model()`/`save_checkpoint()` remain
correct with no code change, exactly as M27 already established. Not
re-tested here (M27's `tests/test_cuda_stream_autograd.py::
test_save_model_after_async_work_on_a_stream_round_trips_correctly` and
`::test_save_checkpoint_after_async_work_on_a_stream_round_trips_correctly`
already cover the "read after async work" case this reasoning depends on;
Milestone 28 does not change what stream a parameter's `last_stream` ends up
naming when only one stream ever touches it).

### Benchmark results (940MX, real hardware; `python -m benchmarks.stream_dependency_bench`)

| Measurement | Median |
|---|---:|
| same-stream baseline (no dependency) | 56.43 us/op |
| cross-stream dependency (1 producer) | 79.04 us/op |
| multi-input dependency (2 producers, incl. producing both operands) | 459.33 us/op |
| event creation + destruction (isolated) | 2.63 us/event |
| cross-stream allocator reuse (release + realloc, same stream) | 197.58 us/cycle |

Cross-stream overhead over the same-stream fast path measured ~1.4x on a
4,096-element elementwise add (79.04 / 56.43 us) -- the added
`cudaEventRecord` + `cudaStreamWaitEvent` pair, real but small relative to
kernel-launch overhead at this size. The same-stream fast path itself
remains within run-to-run noise of Milestone 27's own numbers (both ~50-60
us range for a comparably sized chained add in `stream_bench.py`), confirming
Section 12's "should not turn a single-stream workload into an event-heavy
workload" requirement.

The existing M20 CNN/MNIST workload (`python -m benchmarks.mnist_bench`,
default-stream mode -- it never calls `forge.cuda.stream()`, so it never
exercises the cross-stream path added this milestone) measured 19.07 ms/
iteration (3,310 samples/sec) on the 940MX after this milestone's changes,
within the M26 baseline's own documented 18.56-19.53 ms range (see
**Milestone 27**'s benchmarking section above) and better than either of
M27's own two runs (23.03 ms, 27.47 ms) -- consistent with the
already-documented WDDM driver-scheduling variance across runs, not a
regression. This confirms Section 12/26 of the M28 brief: default-stream
(M26-compatible) performance is unaffected, since every storage's
`last_stream` stays `None` throughout default-stream execution and
`_stream_guard`'s producer-collection loop degenerates to the same no-op it
already was in M27.
