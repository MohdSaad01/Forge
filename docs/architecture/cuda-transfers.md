# Pinned Memory and Asynchronous CUDA Transfers (Milestone 29)

Milestone 27/28 made *compute* asynchronous (real streams, automatic
cross-stream dependencies). Host<->device *transfers* stayed exactly as
Milestone 26 left them: `cf_memcpy_h2d`/`cf_memcpy_d2h` are plain,
synchronous `cudaMemcpy` calls on ordinary pageable NumPy memory, and
`Tensor.to()` is always fully host-blocking. This milestone adds the
lower-level primitives real asynchronous transfer requires: pinned
(page-locked) host memory, `cudaMemcpyAsync`, and an explicit opt-in
`non_blocking=True` on `Tensor.to()`.

## 1. Pageable vs. pinned host memory

Ordinary NumPy memory is *pageable* -- the OS can move it between physical
RAM pages at any time. `cudaMemcpy` can still copy to/from it, but the CUDA
driver must first stage it through an internal pinned bounce buffer, which
serializes the copy with respect to the host regardless of which `cudaMemcpy`
variant is called. *Pinned* (page-locked) memory, allocated via
`cudaHostAlloc`, cannot be paged out -- the GPU's DMA engine can transfer
it directly, which is what makes a `cudaMemcpyAsync` call against it actually
return to the host before the transfer completes, and actually overlap with
other GPU/host work.

`forge.cuda.PinnedMemory` (`forge/backend/cuda/pinned.py`) wraps a real
`cudaHostAlloc`d buffer -- never ordinary memory pretending to be pinned:

```python
mem = forge.cuda.PinnedMemory(nbytes)
array = mem.numpy(shape=(1024,), dtype=np.float32)  # zero-copy NumPy view
mem.free()  # explicit release; also runs on __del__
```

## 2. Pinned allocation lifecycle

Direct, uncached `cudaHostAlloc`/`cudaFreeHost` per allocation -- deliberately
not a caching allocator (Section 25 of the milestone brief: "start with
direct lifecycle... do not build a pinned caching allocator unless profiling
demonstrates it is necessary"; profiling was not asked for or performed).

**Invariant 1 (pinned lifetime):** pinned memory must never be freed while an
asynchronous CUDA operation may still reference it. `PinnedMemory._mark_pending(event)`
records a completion `CUDAEvent` for every in-flight transfer that touches
the buffer (called by `CUDABackend.from_array_async`/`to_numpy_async`);
`free()` (and `__del__`) waits for every such event (`CUDAEvent.synchronize()`)
*before* the real `cudaFreeHost` call. This can block the calling thread if a
transfer has not finished -- an explicit, documented cost, exactly like
`empty_cache()`'s pending-block drain (Milestone 27).

## 3. NumPy interoperability

`PinnedMemory.numpy(shape, dtype)` returns a `_PinnedArray` -- a thin
`np.ndarray` subclass that is a real, zero-copy view over the pinned buffer,
carrying a strong `_pinned_owner` back-reference to the `PinnedMemory`
instance. This one attribute is the entire lifetime mechanism: as long as
some array (or a `Tensor` built from it) is reachable, ordinary CPython
refcounting keeps the owning `PinnedMemory` alive too -- no `gc.collect()`,
no manual bookkeeping, matching this codebase's established lifetime-testing
convention.

`forge/backend/cpu.py`'s `CPUBackend.from_array` was given one narrow
exception to its normal "always copy" contract: a `_PinnedArray` of matching
dtype is returned as-is, not copied. Without this, `Tensor(mem.numpy(...),
device="cpu")` -- the natural, sanctioned way to build a pinned-backed CPU
Tensor -- would silently lose the pinned buffer, since `np.array(data,
dtype=dtype)` always allocates a fresh (pageable) copy. Every other input to
`from_array` is completely unaffected -- this check is `False` for any
ordinary array.

## 4. Pinned memory statistics

`forge.cuda.pinned_memory_stats() -> PinnedMemoryStats` (`pinned_active_bytes`,
`pinned_peak_bytes`, `pinned_allocation_count`, `pinned_free_count`) is a
separate small dataclass from `forge.cuda.memory_stats()`'s `CUDAMemoryStats`
-- pinned host bytes are not folded into `reserved_bytes`/`cached_bytes`/
`pending_bytes`, which remain exclusively device-memory concepts (Section 55
of the milestone brief). `forge.cuda.empty_cache()` remains device-allocator
-only, unchanged -- it does not touch pinned allocations (Section 56); no
`empty_pinned_cache()` was added, since there is no pinned cache to empty.

## 5. `cudaMemcpyAsync` bindings

`kernels.cu` gained `cf_host_alloc`/`cf_host_free` (`cudaHostAlloc`/
`cudaFreeHost`, `cudaHostAllocDefault`) and `cf_memcpy_h2d_async`/
`cf_memcpy_d2h_async` (`cudaMemcpyAsync`, taking an explicit `cudaStream_t`
-- `NULL` meaning the default stream, the same convention every
kernel-launcher already uses). `cf_memcpy_d2d` was left untouched (Section
22: D2D async transfer was explicitly not prioritized this milestone; the
existing device-to-device copy path, used only by `reshape()`/`from_array()`'s
CUDA-to-CUDA clone, stays a plain synchronous `cudaMemcpy`, still correct
under CUDA's legacy-default-stream implicit ordering exactly as documented in
`docs/architecture/cuda-streams.md`'s **Memory copy semantics**).

## 6. H2D transfer semantics

`CUDABackend.from_array_async(host_array, dtype)` (`backend.py`):

```text
host_array must carry `_pinned_owner` (a real PinnedMemory-backed array)
    -> else: CUDAError (Section 13, Option A -- see below)
cf_memcpy_h2d_async(dest, host_array, nbytes, current_stream_handle)
    -> CUDAEvent recorded on current_stream_handle, marked pending on the
       source PinnedMemory (Invariant 1)
CUDAStorage(dest, ...) constructed
    -> CUDAStorage.__init__ already sets last_stream = current_stream()
       unconditionally -- no new code needed for this
```

No `cudaDeviceSynchronize()` anywhere in this path -- the host returns as
soon as the copy is *submitted*.

## 7. D2H transfer semantics

`CUDABackend.to_numpy_async(storage)` (`backend.py`):

```text
_stream_guard((storage,), ...)
    -> if storage.last_stream != current stream: cudaEventRecord + cudaStreamWaitEvent
       (the *existing* Milestone 28 cross-stream dependency mechanism -- no
       new code needed for this either)
allocate a fresh PinnedMemory(storage.nbytes) as the destination
cf_memcpy_d2h_async(pinned_dest, storage, nbytes, current_stream_handle)
    -> CUDAEvent recorded; marked pending on the destination PinnedMemory
return (host_array, PendingTransfer(event))
```

The returned `host_array` may not be fully written when this call returns --
see **Host-Read Synchronization** below for how that is made safe.

## 8. `non_blocking` and the pageable-memory policy (Section 12/13)

`Tensor.to(device, non_blocking=False)`. `False` (default) is byte-for-byte
the pre-Milestone-29 fully host-synchronous contract -- unchanged.
`non_blocking=True`:

- Only valid for a `cpu<->cuda` direction; anything else raises
  `UnsupportedDeviceError`.
- **CPU -> CUDA**: requires the source data to already be pinned
  (`_pinned_owner` set). **Policy: Option A (fail clearly)** -- ordinary
  pageable memory raises `CUDAError` rather than being silently staged
  through a hidden pinned buffer. Rejected alternatives: Option B (silent
  fallback to synchronous) would make `non_blocking=True` a no-op lie for
  the common case; Option C (transparent staging) means a hidden allocation
  and a hidden extra host-to-host copy on every call, exactly what Section
  14 forbids ("do not automatically allocate pinned staging buffers... that
  would introduce hidden allocations, hidden copies, lifetime complexity").
  Option A is the smallest, most honest choice: the caller explicitly
  allocates a `PinnedMemory`, once, and reuses it -- exactly the pattern a
  real training loop's DataLoader would want anyway.
- **CUDA -> CPU**: always succeeds. Forge allocates the pinned destination
  buffer itself (this is not "hidden staging" in the Section 14 sense --
  it is the unavoidable *destination* of the copy the caller explicitly
  requested, not a substitute for pageable input the caller already had).

## 9. Current-stream semantics

Both async transfer paths read `CUDABackend._stream_handle()` -- the same
"ambient current stream" mechanism every other `CUDABackend` method already
uses (`forge/backend/cuda/stream.py`). No transfer takes an explicit stream
argument; `with forge.cuda.stream(s): x.to("cuda", non_blocking=True)`
submits on `s`, exactly like any other Tensor operation issued in that block.

## 10. Cross-stream transfer dependencies

**H2D -> compute (Section 16/17).** `CUDAStorage.__init__` unconditionally
sets `last_stream = current_stream()`, so a storage produced by an async H2D
transfer on stream A already carries `last_stream = A` with zero new code.
When that storage is later read from stream B, `CUDABackend._stream_guard`
(Milestone 28, unmodified) inserts the usual `cudaEventRecord(A) +
cudaStreamWaitEvent(B)` before the consuming kernel launches. This is the
entire mechanism -- Milestone 29 added no new dependency-insertion code path
for this direction.

**Compute -> D2H (Section 23/24).** `to_numpy_async` calls `_stream_guard((storage,),
...)` *before* submitting the D2H copy -- if `storage.last_stream` differs
from the stream the D2H copy is about to run on, the existing dependency
mechanism makes the D2H-issuing stream wait for the producer first. Verified
directly: `tests/test_cuda_transfer_dependencies.py::
test_compute_on_one_stream_then_d2h_on_another_needs_no_explicit_sync`, plus
a `cf_synchronize`-spy test in the same file confirming zero
`cudaDeviceSynchronize()` calls on this path.

No new event/dependency mechanism was introduced anywhere in this milestone
-- both directions reuse Milestone 28's `_stream_guard` chokepoint exactly.

## 11. Tensor stream ownership

Unchanged from Milestone 27/28: a `CUDAStorage`'s `last_stream` field is the
single piece of stream-provenance metadata Forge tracks, and it is set
identically whether the storage came from a kernel launch or an async H2D
transfer (`CUDAStorage.__init__`, always).

## 12. Host-read synchronization (Section 18/19)

The hardest design question this milestone answers: how does a CPU `Tensor`
whose data was submitted for an async D2H copy know, transparently, that a
host read must wait first?

**Design chosen: a synchronizing `_data` property** (`forge/tensor/tensor.py`).
`Tensor._data` was converted from a plain attribute into a property backed by
`Tensor._storage`, gated by `Tensor._pending` (`None` for every tensor except
one still awaiting an async D2H's completion):

```python
@property
def _data(self):
    if self._pending is not None:
        self._pending.synchronize()   # forge.backend.cuda.transfer.PendingTransfer
        self._pending = None
        self._storage = np.array(self._storage, copy=True)  # detach from pinned memory -- see below
    return self._storage
```

This is the *single* chokepoint every existing Tensor method already passes
through -- `.numpy()`, `__repr__`, every forward/backward op, `backward()`,
persistence's `Backend.to_numpy(param._data)` -- so none of them needed to
learn that pending transfers exist. The wait is targeted (only this
tensor's own `PendingTransfer`, via `CUDAEvent.synchronize()` -- never a
device-wide synchronization) and paid at most once. `Tensor.to(...,
non_blocking=True)`'s D2H branch is the only code that ever sets `_pending`.

**Why detach from pinned memory on first sync, not just wait.** Without the
`np.array(..., copy=True)` step, `self._storage` would remain a
`_PinnedArray`. NumPy ufuncs propagate subclass identity through
`__array_finalize__`, so *every* array later derived from this tensor
(`cpu_t + other`, slicing, reshaping...) would also carry a `_pinned_owner`
reference, keeping a potentially large pinned allocation alive indefinitely
-- an unbounded-retention hazard directly at odds with Section 27's leak
requirement. Copying once, immediately after the transfer is confirmed
complete, severs that chain: the tensor's *permanent* representation is
ordinary pageable memory, and pinned memory is exactly what Section 21/45
say it should be -- a transfer-time optimization, never permanent Tensor
state. The extra host-to-host `memcpy` this costs is negligible next to the
PCIe transfer it follows.

**One known limitation:** `Tensor.shape` is `self._data.shape` (there is no
separately cached shape), so even a pure shape inspection on a still-pending
D2H tensor forces a synchronization today. Every case that actually needs
the underlying values already needs this same sync moments later (shape
checks inside `_binary_op` precede the backend call that would sync anyway),
so this is conservative, not incorrect -- but a caller wanting to inspect a
pending transfer's shape without blocking cannot do so in this milestone.

## 13. Transfer completion (Section 32)

`forge.backend.cuda.transfer.PendingTransfer` -- the smallest possible
completion handle: one internal `CUDAEvent`, `is_ready()` (`cudaEventQuery`,
never blocks) and `synchronize()` (`cudaEventSynchronize`, idempotent). Not
public API (`forge.cuda` exposes no `Transfer`/`Future` type, per Section 46)
-- its only consumer is `Tensor._data`'s property getter.

## 14. Allocator integration (Section 34/35)

Device-side H2D destination memory is allocated through the existing
Milestone 25 caching allocator (`CUDABackend._alloc`), completely
unmodified. Because the resulting `CUDAStorage.last_stream` is set exactly
like any other storage, releasing it before the transfer completes goes
through the *existing* pending-block path (`CUDAStorage.__del__` ->
`allocator.release_pending()`), unmodified since Milestone 27 -- a block an
in-flight H2D copy is still writing cannot be reused until its recorded
event completes. **Mandatory race test** (Section 35), verified directly:
`tests/test_cuda_transfer_allocator.py::
test_async_h2d_release_never_hands_the_still_in_flight_block_to_another_stream`.

## 15. `empty_cache()` semantics

Unchanged: device-allocator-only. A pending block from an in-flight async
transfer is drained exactly like a pending block from an in-flight kernel
(`CUDACachingAllocator._drain_pending`, `CUDAEvent.synchronize()` then
`cudaFree`) -- no transfer-specific code was needed here either.

## 16. Autograd / optimizer / Trainer

**Autograd (Section 21):** unaffected. An async H2D result is an ordinary
`requires_grad=False` leaf `CUDAStorage` (device transfer has never been
differentiable in Forge), safe to use as a constant operand in any
differentiable CUDA computation -- `_stream_guard` handles the cross-stream
dependency exactly as it does for any other input. Verified:
`tests/test_cuda_transfer_dependencies.py::
test_autograd_works_with_an_asynchronously_transferred_constant_input`.

**Optimizer:** unaffected -- `sgd_step`/`adam_step` already route through
`_stream_guard`; nothing here changes.

**Trainer (Section 42):** not redesigned, per the brief. `forge/training/
trainer.py` continues to use fully synchronous `.to()` calls; this milestone
only establishes that a `non_blocking=True`-transferred `Tensor` behaves
correctly if handed to existing CUDA computation, which the tests above
confirm.

## 17. Persistence / checkpoint safety (Section 43/44)

Unaffected. `Module.to(device)` (`Tensor._move_storage_`) remains fully
synchronous, so a `Parameter`'s data is never itself the product of an async
transfer; `save_model()`/`save_checkpoint()` read parameters via `Backend.
to_numpy()` exactly as before. Verified with a realistic scenario --
training on an asynchronously (pinned) transferred *input* immediately
followed by `save_model()`, no explicit `forge.cuda.synchronize()` call from
the test itself: `tests/test_cuda_transfer_persistence.py`.

## 18. Error semantics (Section 30)

Every new native call (`cf_host_alloc`, `cf_host_free`, `cf_memcpy_h2d_async`,
`cf_memcpy_d2h_async`) returns a real `cudaError_t`, checked by
`CUDABackend._check()`/`PinnedMemory`'s `raw_host_alloc`/`raw_host_free`
exactly like every pre-existing native call -- no return value is ignored.
Submission-time failures (a bad `cudaMemcpyAsync` argument, an
already-failed context) surface immediately as `CUDAError` at the call
site, before the host ever proceeds; execution-time failures on the device
surface at the next synchronization boundary (`CUDAEvent.synchronize()`),
exactly like any other asynchronous CUDA operation.

## 19. Benchmark methodology and results

`benchmarks/async_transfer_bench.py` (`python -m benchmarks.async_transfer_bench`),
following `benchmarks/timing.py`'s synchronize-bracketed methodology. Real
940MX numbers (driver 582.53; representative of several runs -- WDDM
scheduling introduces the same run-to-run variance already documented for
Milestone 27/28):

**Pinned (async, synchronized) vs. pageable (synchronous) H2D:**

| Size | Bytes | Pageable | Pinned (submit + sync) |
|---|---:|---:|---:|
| tiny | 4,096 | ~0.08-0.14 ms | ~0.09-0.15 ms |
| small | 400,000 | ~0.44-0.69 ms | ~0.33-0.57 ms |
| medium | 4,000,000 | ~4.76-5.03 ms (0.80-0.84 GB/s) | ~2.51-2.71 ms (1.48-1.59 GB/s) |

At the tiny scale, fixed per-call overhead (Python + driver call latency)
dominates and pinned shows no advantage; at 4 MB, avoiding the driver's
internal pageable-to-pinned staging copy makes the pinned path consistently
~1.8-1.9x faster. Neither number should be read as PCIe bus saturation --
the 940MX's practical achievable bandwidth here is well under theoretical
PCIe Gen3 x4/x8 limits, consistent with a laptop-class part and shared
system memory bandwidth (i5-7200U, 8 GB RAM).

**Async submission latency vs. synchronized completion (4 MB H2D):**

| Measurement | Median |
|---|---:|
| submission only (`cudaMemcpyAsync` queued, host returns) | ~0.04-0.09 ms |
| full completion (submission + `forge.cuda.synchronize()`) | ~2.46-2.65 ms |

Submission returns roughly 30-60x faster than completion -- direct
confirmation that `non_blocking=True` does not secretly synchronize before
returning (Section 31's "async D2H must not lie," applied identically to
H2D here).

**H2D transfer / compute overlap** (8 MB H2D transfer on one stream,
concurrently with 400 chained 20,000-element adds on another):

| Measurement | Median |
|---|---:|
| sequential (transfer, then compute) | ~22.6-37.1 ms |
| concurrent (both issued together) | ~21.8-30.0 ms |
| speedup | 0.97x-1.24x across runs |

**D2H transfer / compute overlap** (same shapes, D2H instead of H2D):

| Measurement | Median |
|---|---:|
| sequential | ~25.4-26.1 ms |
| concurrent | ~27.9-30.3 ms |
| speedup | 0.86x-0.91x across runs |

H2D overlap is real but modest and noisy on this hardware -- consistent
with Milestone 27's own finding that a 3-SM Maxwell part leaves little room
for independent-kernel overlap once one workload is already large enough to
matter, compounded here by WDDM scheduling variance. D2H overlap measured
*below* 1.0x in every run: both the D2H copy and the elementwise-add compute
are memory-bandwidth-bound, and on this GPU they appear to contend for the
same memory controller/PCIe path rather than overlap productively -- an
honestly reported real result, not tuned to look better (per Section 37's
explicit instruction), not a Forge correctness issue (every value produced
in the concurrent case was still verified bit-exact against the sequential
reference by the correctness tests in `tests/test_cuda_transfer_dependencies.py`
and `tests/test_cuda_async_transfer.py`).

## 20. Current limitations

- No pinned caching allocator -- direct `cudaHostAlloc`/`cudaFreeHost` per
  allocation (Section 25); revisit only if profiling justifies it.
- No hidden pageable-to-pinned staging for H2D (Section 14/Option A above)
  -- a pageable `non_blocking=True` source raises `CUDAError` rather than
  silently degrading to a slower path.
- D2D transfers remain synchronous, unchanged (Section 22) -- `reshape()`
  and `from_array()`'s CUDA-to-CUDA clone path were not touched.
- No asynchronous DataLoader, prefetch worker, or automatic GPU prefetch
  (Section 41/45) -- this milestone provides only the primitives a future
  milestone's DataLoader integration would consume. **Milestone 30**
  consumes exactly these primitives (`PinnedMemory`, `from_array_async`/
  `to_numpy_async` via `Tensor.to(..., non_blocking=True)`, and the M28
  `_stream_guard` cross-stream dependency) to build that DataLoader
  integration with zero new transfer/synchronization mechanism -- see
  `docs/architecture/async-dataloader.md`.
- `Tensor.shape` on a still-pending D2H tensor forces synchronization (see
  **Host-Read Synchronization** above) -- a known, documented, low-impact
  limitation, not a correctness issue.
- One hardware-observed quirk, unrelated to Forge's own logic: on the
  940MX/driver 582.53 combination, an out-of-memory `cudaHostAlloc` request
  (even a "merely" 64 GiB one) has been observed to leave the process's
  CUDA context unable to serve small subsequent `cudaMalloc` requests. The
  regression test for pinned-allocation failure handling
  (`tests/test_cuda_pinned_memory.py::
  test_absurdly_large_pinned_allocation_raises_cuda_error_and_touches_no_counters`)
  runs in an isolated subprocess specifically to contain this, while still
  exercising the real, hardware-verified failure path (never simulated).

## 21. Future async data pipeline design

Explicitly out of scope for this milestone, listed so a future one does not
have to rediscover it:

- **Async DataLoader / prefetch workers**: a background thread or process
  that stages the next batch into `PinnedMemory` while the current batch
  trains, then submits `non_blocking=True` H2D transfers -- the primitives
  this milestone adds are exactly what such a design would consume.
  **Implemented in Milestone 30** (`forge.data.CUDAPrefetchLoader`) -- see
  `docs/architecture/async-dataloader.md`. Its background CPU-producer
  thread deliberately never touches CUDA/streams itself (only the calling
  thread does), sidestepping the thread-safety hazard the next bullet in
  this section already anticipated.
- **`Trainer`-internal transfer/compute overlap**: `Trainer` issuing its own
  H2D transfer on a dedicated stream ahead of the compute stream needing it,
  synchronizing only at public method boundaries (mirrors Milestone 27's
  already-deferred "Trainer-internal stream use"). **Implemented in
  Milestone 30** as `Trainer(..., prefetch=True)`, which also gives
  `Trainer` a dedicated, persistent compute `Stream` for the duration of
  the prefetch-enabled training loop -- see `docs/architecture/
  async-dataloader.md`'s **Trainer Integration**/**Compute Stream**
  sections and `docs/architecture/cuda-streams.md`'s note on this
  superseding the "Trainer remains synchronous" default for that opt-in
  path only.
- **A pinned caching allocator**: only if profiling of a real prefetch
  pipeline shows repeated `cudaHostAlloc`/`cudaFreeHost` calls are a
  measurable bottleneck.
- **D2D async transfer**: `cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice,
  stream)` for `reshape()`/multi-GPU data movement, if a future milestone
  needs device-to-device copies to respect the current stream explicitly.
