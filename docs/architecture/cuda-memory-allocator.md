# CUDA Memory Allocation Profiling & Caching Allocator Design (Milestone 24)

Milestone 22 instrumented Forge's real `cudaMalloc`/`cudaFree` boundary
(`forge.cuda.memory_stats()`); Milestone 23 fixed the one Forge-owned
reference cycle that made those counters unreliable. Both left the
allocation *strategy* itself untouched: **one direct `cudaMalloc` per
`CUDAStorage`, one direct `cudaFree` at its destruction** -- no pool, no
cache, no reuse.

Milestone 24 asks whether that strategy is actually a bottleneck, using real
measurements on the development hardware, and proposes (without
implementing) a caching-allocator design for a future milestone if the
evidence justifies it.

```text
CUDAStorage
    |
cudaMalloc
    |
CUDA memory
    |
CUDAStorage destruction
    |
cudaFree
```

This diagram is **unchanged** by Milestone 24. No pooling, caching, block
splitting/coalescing, deferred frees, or CUDA memory pools were added to the
runtime allocation path. Every number below was measured against Forge's
existing direct allocator.

## 1. Current direct-allocation model

`CUDABackend._alloc()` (`forge/backend/cuda/backend.py`) is the single choke
point for every CUDA allocation in Forge -- every operation that produces a
new `CUDAStorage` calls it exactly once per output buffer. `CUDAStorage.
__del__()` is the single choke point for every CUDA free. Milestone 22's
counters and Milestone 24's profiler are both instrumented at exactly these
two call sites (see "Allocation Event Data" below) -- no operation-specific
instrumentation exists anywhere else in the ~40 `CUDABackend` methods.

There is no allocation-size rounding, no minimum block size (other than
clamping a zero-byte request up to 1 byte, since `cudaMalloc(0)` is
undefined), and no reuse of any kind: two same-sized, back-to-back
allocate/free cycles issue two independent `cudaMalloc`/`cudaFree` pairs to
the driver.

## 2. Allocation profiling infrastructure

A new, optional, low-overhead diagnostic facility -- deliberately distinct
from `memory_stats()` (current/peak state) and never touching it.

**`forge/backend/cuda/profiler.py`** -- `CUDAMemoryProfiler`, a process-wide
singleton (`get_profiler()`) holding a list of `AllocationEvent`s:

```python
@dataclass(frozen=True)
class AllocationEvent:
    kind: str            # "alloc" | "free"
    nbytes: int
    timestamp: float     # time.perf_counter()
    block_id: int        # the allocated pointer's raw integer value
    category: str | None # the innermost active tag, if any
```

Instrumented at the same two call sites as `memory.py`'s counters:
`CUDABackend._alloc()` and `CUDAStorage.__del__()` now also pass the
allocated pointer's integer value (`ptr.value`) through to
`forge.backend.cuda.memory.record_alloc`/`record_free`, which forward
`(kind, nbytes, block_id)` to the profiler. **No `CUDAStorage`, `Tensor`, or
other Forge object is ever passed or retained** -- only three primitives per
event. `block_id` exists purely to correlate an "alloc" with its later
"free" in offline analysis; it is never dereferenced.

**Disabled-path cost**: `CUDAMemoryProfiler.record()` checks one `bool`
attribute (`self._active`) and returns immediately if the profiler was never
started -- no `AllocationEvent` construction, no `time.perf_counter()` call,
no list append, no lock acquisition. This satisfies the "low overhead when
disabled" requirement without any `if profiling_enabled:` check needed at
either of the two call sites themselves (they call unconditionally; the
profiler decides).

**Public API** (`forge/cuda/profiler.py`, mirroring `forge.cuda.memory_stats()`'s
existing thin-wrapper pattern):

```python
forge.cuda.profiler.start() / .stop() / .reset() / .is_active() / .events()
with forge.cuda.profiler.tag("forward"):
    ...
with forge.cuda.profiler.profile():   # reset + start, stop on exit
    train_step()
```

Categorization is opt-in via `tag()` (a small stack, pushed/popped around a
code region) rather than instrumenting every `CUDABackend` method
individually -- `benchmarks/alloc_profile.py` tags `"transfer"`, `"forward"`,
`"loss"`, `"backward"`, `"optimizer"` around the corresponding phases of one
training step.

**Offline analysis** (`benchmarks/alloc_analysis.py`) is a set of pure
functions over an `AllocationEvent` trace -- no CUDA calls, nothing mutated:
`size_distribution`, `lifetime_distribution` (with `pair_lifetimes`'
FIFO-per-`block_id` matching), `persistent_vs_temporary`,
`reuse_opportunity`, and `simulate_caching_allocator` (Section 8 below).

## 3. Measured allocation behavior: M20 MNIST workload

Methodology: the real M20 CNN (`examples.mnist.model.build_model()`,
~27.6k parameters), batch size 64, `CrossEntropyLoss` + `Adam`. 5 warmup
iterations (untracked, absorbing lazy kernel-module/compile-cache effects,
per `benchmarks/timing.py`'s established convention), then 30 steady-state
iterations with the profiler active and phase-tagged. Measured on the
development GPU: **NVIDIA GeForce 940MX, Compute Capability 5.0, CUDA 12.6,
driver 582.53** (Windows, WDDM).

| Metric | Value |
|---|---|
| Mean full-iteration wall-clock time (unprofiled pass) | 25.21 ms |
| Allocations per iteration | 64.0 |
| Frees per iteration | 64.0 |
| Allocated bytes per iteration (mean) | 9,475,776 bytes (~9.0 MiB) |
| Peak allocated bytes (any instant) | 7,789,424 bytes (~7.4 MiB) |
| Real persistent CUDA memory (before/after/after-gc, `memory_stats()`) | 440,992 bytes, flat |

The 440,992-byte figure is `forge.cuda.memory_stats().allocated_bytes`
captured immediately before and after the 30-iteration steady-state window
(with an intervening `gc.collect()`, per Milestone 23's established
methodology) -- it never changes, confirming the model's Parameters plus
Adam's `m`/`v` are allocated once and never regrow, consistent with
`tests/test_cuda_memory.py`'s existing steady-state tests.

**Per-iteration allocation traffic by phase** (from `tag()` categories):

| Phase | Allocations/iter | Bytes/iter |
|---|---:|---:|
| transfer (`.to("cuda")` for the input batch) | 1 | 200,704 |
| forward | 12 | 4,365,312 |
| loss | 12 | 13,836 |
| backward | 39 | 4,895,924 |
| optimizer (`Adam.step()`) | 0 | 0 |

`optimizer` recording zero allocations matches Section 2's expectation and
`docs/architecture/optimization.md`'s new **Allocation behavior (Milestone
24)** note. `backward` is the largest single contributor to allocation
*count* (39 of 64, ~61%) -- consistent with Milestone 21's finding that
`Conv2d`'s backward kernels dominate CUDA *time* too, though here it is
allocation-call volume, not per-call kernel cost, driving the number.

## 4. Allocation-size distribution

Across the 1,920 total "alloc" events in the 30-iteration MNIST trace:

| Bucket | Count |
|---|---:|
| < 1 KB | 690 |
| 1-4 KB | 450 |
| 4-16 KB | 30 |
| 16-64 KB | 180 |
| 64-256 KB | 270 |
| 256 KB-1 MB | 180 |
| 1-4 MB | 120 |
| 4-16 MB | 0 |
| 16-64 MB | 0 |
| 64+ MB | 0 |

Only **14 distinct exact byte sizes** occur across all 1,920 allocations
(`reuse_opportunity`'s `distinct_sizes`). The most common sizes are small,
fixed-shape gradient/bias-adjacent buffers (2,560 bytes x450, 256 bytes x390,
4 bytes x180 -- scalar losses/reductions) and the CNN's larger activation
tensors (1,384,448 bytes x120, matching a `(64, 8, 26, 26)` float32
activation exactly). This is a workload of **small-to-medium, highly
repetitive, fixed-shape buffers** -- not a workload dominated by rare,
large, irregularly-sized allocations. That shape distribution is the single
most important input to Sections 8-10's allocator recommendation below.

## 5. Allocation-lifetime distribution

Pairing each alloc with its next same-`block_id` free (`pair_lifetimes`):

| Bucket | Count |
|---|---:|
| < 1 ms | 724 |
| 1-10 ms | 264 |
| 10-100 ms | 924 |
| 100 ms-1 s | 0 |
| 1-10 s | 0 |
| > 10 s | 0 |
| still live at trace end | 8 |

Median lifetime: 7.4 ms; mean: 9.6 ms; max: 27.6 ms -- every measured
lifetime falls within roughly one training-iteration duration (25.2 ms).
**No allocation in this trace lives longer than a single iteration** --
Forge's temporaries are exactly the same-operation/forward-lived/
backward-lived/iteration-lived categories the milestone brief anticipated,
never a slow multi-iteration leak.

**A trace-boundary artifact, documented explicitly**: the 8 "still live"
allocations at the end of the window are not persistent model/optimizer
state (that was allocated during warmup, before the profiler started, so it
never appears as an "alloc" event in this trace at all). They are exactly
the 8 gradient tensors (one per model Parameter -- 2 Conv2d layers x
{weight, bias} + 2 Linear layers x {weight, bias}) produced by the final
recorded iteration's `backward()` call, which would ordinarily be freed by
the *next* iteration's `zero_grad()` -- a call that falls just outside the
30-iteration window. Their total size (110,248 bytes) matches the model's
total parameter-gradient footprint exactly. This is a real limitation of
trace-based persistent/temporary classification over a bounded window (see
Section 6) -- true persistent memory is measured via `memory_stats()`
around the window instead, not by "never freed within the trace."

## 6. Persistent vs. temporary memory

Two independent measurements, deliberately cross-checked:

- **True persistent footprint** (`memory_stats()`, unaffected by trace
  windowing): 440,992 bytes -- the CNN's ~27.6k parameters plus Adam's `m`/`v`
  for each, flat across the entire steady-state window.
- **Trace-based temporary churn** (sum of allocated bytes for blocks that
  *are* freed within the 30-iteration window): 284,163,032 bytes across
  1,912 alloc/free pairs -- i.e., ~9.47 MiB of temporary allocation traffic
  *per iteration*, entirely intermediate/activation/gradient buffers that
  are allocated and freed again before the next iteration begins.

At any single instant, peak active memory (7.79 MiB) is dominated by
transient forward/backward activations, not by the 431 KiB of persistent
parameter/optimizer state -- for this small CNN, **temporary memory is
roughly 18x the persistent footprint at peak**, and the *cumulative* churn
across a training run vastly exceeds the persistent footprint. This is
exactly the situation a caching allocator is designed for: a large volume
of same-shape, short-lived, per-iteration allocation traffic sitting on top
of a small, stable persistent base.

## 7. Reuse opportunity

`reuse_opportunity()` asks: of all allocation requests after the first
occurrence of their exact byte size, what fraction of *bytes* do they
represent? (A descriptive statistic about the trace's shape repetition --
not a claim about achievable allocator speedup; see Section 8 for an actual
simulated policy.)

- 1,920 total allocation requests across the trace, spanning only 14
  distinct exact byte sizes.
- **99.1% of allocated bytes**, and **99.3% of allocation *count***, are a
  repeat of a size seen earlier in the trace.

This is about as strong a same-shape-repetition signal as a workload can
produce: a fixed batch size, a fixed model architecture, and no
control-flow-dependent shape changes across iterations (true of every
Forge model -- Forge has no dynamic/data-dependent shapes) mean nearly every
temporary tensor recurs at an identical byte size every single iteration.

## 8. Offline caching-allocator simulation

**Simulation only** -- replayed against the real trace above,
`simulate_caching_allocator()` never runs inside Forge's actual allocator.
Two policies:

### Exact-size cache (Candidate A)
A freed block is reused only for a request of the identical size.

| Metric | Value |
|---|---:|
| Total requests | 1,920 |
| Simulated `cudaMalloc` calls | **42** |
| Cache hit rate | 97.8% |
| Peak reserved bytes | 9,132,312 |
| Peak active bytes | 7,348,432 |
| Internal fragmentation | 0 bytes (by construction) |

Instead of 1,920 real `cudaMalloc`/`cudaFree` pairs, a simple exact-size
cache would have driven only **42** real driver calls for the entire
30-iteration run -- roughly 1.4 real allocations per iteration, all during
early warmup convergence, dropping to zero new driver calls once every
recurring size has a cached block.

### Size-class cache (Candidate B)
Requests are rounded up to `SIZE_BUCKETS`' boundaries before the cache
lookup, so different-but-nearby sizes can still hit.

| Metric | Value |
|---|---:|
| Simulated `cudaMalloc` calls | 36 |
| Cache hit rate | 98.1% |
| Peak reserved bytes | 23,445,504 (~22.4 MiB) |
| Cumulative internal fragmentation | 484,556,160 bytes |

Size-class rounding buys a marginally higher hit rate (98.1% vs. 97.8%) and
6 fewer real `cudaMalloc` calls, at the cost of **2.6x more reserved VRAM**
(22.4 vs. 8.7 MiB) and substantial internal fragmentation -- on a 2 GiB
card, that reserved-memory cost is a real, not theoretical, concern.

**Conclusion from the simulation**: given only 14 distinct sizes and a
99%+ exact-size repetition rate, exact-size caching alone already captures
essentially all of the available reuse; size-class rounding's extra
flexibility is not needed here and actively costs VRAM. This directly
informs Section 10's recommendation.

## 9. cudaMalloc / cudaFree overhead

**Methodology**: `cudaMalloc`/`cudaFree` are classic, host-blocking CUDA
Runtime API calls -- unlike a kernel launch, there is no asynchronous queue
to be misled by (`benchmarks/timing.py`'s synchronize-bracketing concern for
kernels does not apply here). `CUDABackend._alloc()` and the raw `cf_free`
export were each timed directly with `time.perf_counter()`, 10 warmup + 200
measured calls per size, entirely isolated from any concurrent kernel
activity (`benchmarks/alloc_profile.py::_measure_alloc_free_overhead`).

| Scale | Bytes | Mean `cudaMalloc` | Mean `cudaFree` |
|---|---:|---:|---:|
| tiny | 4,096 | 177.3 us | 245.5 us |
| small | 65,536 | 204.8 us | 294.8 us |
| medium | 1,048,576 | 173.3 us | 256.0 us |

Two findings:

1. **Cost is essentially size-independent** across three orders of
   magnitude of request size (4 KB to 1 MB) -- consistent with `cudaMalloc`/
   `cudaFree` cost being dominated by driver/page-table/WDDM bookkeeping
   overhead, not by the (zero, for an allocation) bytes moved.
2. **~175-300 microseconds per call is high** relative to this machine's
   own kernel execution times (M11/M21: sub-millisecond to low-millisecond
   for most operations at these scales) -- almost certainly reflecting this
   specific environment's WDDM (Windows Display Driver Model) overhead,
   known to add meaningfully more per-call latency to CUDA Runtime API
   calls than a Linux/TCC driver stack would. This is an environment-
   specific number, not a general CUDA claim (see Section 25's hardware
   caveats) -- but it is *this* development environment's real number, and
   this milestone's job is to decide based on it.

### Comparison to the MNIST training step
64 allocations + 64 frees occur per steady-state MNIST iteration (Section
3). Multiplying the isolated per-call costs above by that count gives an
**order-of-magnitude estimate**: 64 x (~190 us alloc + ~265 us free) ≈ 29 ms
-- comparable to, and by this crude estimate slightly exceeding, the
measured 25.2 ms mean full-iteration wall-clock time.

This number **must be read as an upper bound, not a precise attribution**,
for exactly the reason Section 10 of the milestone brief warns about:
the isolated timing above measures `cudaMalloc`/`cudaFree` with *no*
concurrent kernel activity, while the real training step interleaves
allocation calls with kernel launches that a WDDM driver can batch/overlap
at the command-buffer level. The true in-workload overhead is very likely
lower than this naive product. What the estimate does establish robustly,
without needing that precision: **allocation/free overhead is the same
order of magnitude as the entire training step on this hardware**, not a
one- or two-order-of-magnitude-smaller rounding error. That qualitative
conclusion is what Section 11 (and the final Decision) rests on -- not the
specific 29 ms figure.

## 10. Candidate allocator designs

| Design | Reuse trigger | Fragmentation | Fit for this trace |
|---|---|---|---|
| **A. Exact-size cache** | Identical byte size | None | Excellent -- 97.8% hit rate, 14 distinct sizes, zero waste (Section 8) |
| **B. Size-class cache** | Rounded-up size bucket | Internal (unused bytes within a bucket) | Marginal hit-rate gain over A, at 2.6x reserved VRAM on a 2 GiB card (Section 8) -- not favorable here |
| **C. Best-fit block reuse** | Smallest sufficient free block | Internal (over-sized reuse) + search cost | Not simulated numerically; with only 14 distinct sizes and 97.8% exact-size hits already, best-fit's extra search flexibility has little left to find -- the trace does not exhibit the highly irregular size spread that would justify its added bookkeeping |
| **D. Split/coalesce allocator** | Splits large free blocks; merges adjacent free blocks | External (address-space gaps) minimized | Solves a problem (external fragmentation from varied, non-recurring sizes) this workload does not have -- Forge has no dynamic shapes, so allocation sizes are a small, fixed, recurring set, not an unbounded stream needing general-purpose splitting/coalescing |

Given the measured size distribution (Section 4: 14 distinct sizes, ~99%
exact repetition) and the simulation (Section 8: exact-size already
captures 97.8% of possible reuse at zero fragmentation cost), the smallest
design that captures nearly all of the available benefit is **Candidate A,
possibly extended with a small number of size classes only for the long
tail of one-off sizes** -- not a general-purpose best-fit or split/coalesce
allocator. Forge's workloads (fixed model architecture, fixed batch shape,
no dynamic control flow) structurally do not produce the irregular,
unbounded-size-space allocation pattern that justifies C or D's complexity.

## 11. Recommended allocator architecture (for a future milestone)

**Not implemented in Milestone 24.** If a future milestone proceeds, this is
the smallest design consistent with the evidence above.

### Ownership
The allocator (a new `CUDACachingAllocator`, analogous in spirit to
`CUDABackend`'s existing singleton) owns a pool of freed-but-cached device
pointers, keyed by size. `CUDAStorage` continues to hold exactly one device
pointer and never gains awareness of whether that pointer came from a fresh
`cudaMalloc` or a cache hit -- the allocator, not the storage, decides.

### Storage relationship
`CUDAStorage` gains no new fields. Internally, `CUDABackend._alloc()`
delegates to `allocator.allocate(nbytes)` instead of calling `cf_malloc`
directly; `CUDAStorage.__del__()` calls `allocator.free(ptr, nbytes)`
instead of `cf_free` directly. `Tensor -> CUDAStorage` identity and
ownership (Section 17 of the brief) are completely unaffected: two live
`CUDAStorage` objects still never alias the same pointer while both are
alive -- the allocator only reuses a pointer *after* its previous
`CUDAStorage` owner has called `free()` on it, i.e., after that
`CUDAStorage` no longer exists.

### Allocation
```text
Tensor op needs device memory
    -> CUDABackend._alloc(nbytes)
    -> allocator.allocate(nbytes)
         -> exact-size free list has a block? pop and return it
         -> otherwise: cudaMalloc(nbytes), return it
    -> wrapped in a fresh CUDAStorage
```

### Free
```text
CUDAStorage.__del__()
    -> allocator.free(ptr, nbytes)
         -> push (ptr, nbytes) onto the exact-size free list
         -> (real cudaFree only ever happens during an explicit cache purge -- see Failure below)
```

### Reuse
An exact-size free list (`dict[nbytes, list[ptr]]`), per Section 10's
recommendation -- no best-fit search, no splitting. A request for a size
with no cached block falls through to a real `cudaMalloc`.

### Fragmentation
By construction, an exact-size cache has **zero internal fragmentation**
(a reused block is always exactly the requested size) and no block
splitting/coalescing means **no external fragmentation logic is needed
either** -- the tradeoff is that a size seen only once never benefits from
caching (it simply behaves like today's direct allocator). Given Section 4
and 7's measurements, this tradeoff costs almost nothing in Forge's actual
workloads.

### Statistics
Extends `CUDAMemoryStats` (Section 13 below) with `reserved_bytes` (real
VRAM currently held by the allocator, cached or active) and `cached_bytes`
(`reserved_bytes - active_bytes`) alongside the existing `allocated_bytes`
(renamed conceptually to "active," if desired, for consistency with the
reserved/cached split) and `peak_*` variants of each.

### Failure (OOM)
```text
allocate(nbytes)
    -> exact-size free list has a block? use it
    -> otherwise: cudaMalloc(nbytes)
         -> success: use it
         -> failure: purge the entire free-list cache (real cudaFree for every cached block)
                     -> retry cudaMalloc(nbytes)
                        -> success: use it
                        -> failure: raise CUDAError (unchanged from today)
```
A cache purge affects `reserved_bytes`/`cached_bytes` (both drop toward the
purged amount) but never `active_bytes` -- no live `CUDAStorage` is ever
touched by a purge, since only *freed* (cached) blocks are held.

### Cleanup
An explicit `forge.cuda.empty_cache()` (mirroring the purge-on-OOM path but
callable directly) would let a caller reclaim cached-but-unused VRAM
proactively, e.g. between an evaluation pass and the next training phase.

### Thread safety
One `threading.Lock` around the free-list dict, matching `_MemoryTracker`'s
existing convention (Forge is single-threaded elsewhere; this is cheap
insurance, not a concurrency subsystem).

### Multi-device
Forge supports exactly one GPU today (`CUDABackend.device_count` is probed
but never used to select among devices). The allocator would key its
free-list dict by `(device_index, nbytes)` from the start, even though
`device_index` is always `0` today -- cheap to add now, expensive to
retrofit later, and it changes no observable behavior while Forge remains
single-GPU.

## 12. Invariants a future allocator must preserve

1. **Storage identity** (Section 17 of the brief): two live `CUDAStorage`
   objects never alias the same device pointer. A cached block may be
   reused, but only strictly after its previous `CUDAStorage` has been
   destroyed.
2. **No reuse before the owning `CUDAStorage` is gone** (Section 18): a
   block enters the free cache only from `CUDAStorage.__del__()` -- never
   speculatively, never based on "probably unused." This is automatically
   satisfied by routing `free()` through the existing `__del__` call site
   rather than introducing a second path.
3. **Persistent state is never reclaimed as "just another cached block"**
   (Section 19): Parameters and Adam `m`/`v` are never freed until their
   owning `Module`/`Optimizer` is destroyed -- the allocator only ever sees
   a `free()` call for them at that point, same as today. Nothing about
   caching changes when Forge decides a `CUDAStorage` is dead; it only
   changes what happens to the pointer *after* that decision.
4. **Real allocation/free events remain observable**: `forge.cuda.
   memory_stats()` and `forge.cuda.profiler` must still describe real
   `cudaMalloc`/`cudaFree` boundaries (now: allocator-cache-miss/purge
   events) plus new reserved/cached figures -- not just active bytes, so a
   caching allocator does not make Milestone 22/24's diagnostics stop
   answering "is memory actually growing."

## 13. CUDA asynchrony implications

Every Forge CUDA operation already calls an explicit `cf_synchronize()`
(`CUDABackend._synchronize`) before trusting its own result (see
`docs/architecture/cuda-backend.md`'s **Operation set**) -- this is the
existing mechanism that makes "a Python `CUDAStorage` becomes unreachable"
and "the GPU has finished using that memory" coincide today: by the time any
operation's result is handed back to Python, every kernel that touched its
inputs has already completed, because the *previous* operation's own
synchronization already drained the queue. There is no outstanding
asynchronous kernel that could still be reading/writing a buffer at the
moment its `CUDAStorage.__del__()` runs.

**A future caching allocator inherits this safety property directly and for
free**, precisely because it changes nothing about *when* `__del__` runs or
what synchronization already exists -- it only changes what happens to the
pointer afterward (cached vs. immediately `cudaFree`d). If Forge ever
introduces genuinely asynchronous/stream-ordered execution (multiple
in-flight kernels, no synchronize-per-op), this safety property would need
to be re-established explicitly (e.g. stream-ordered "safe to reuse" events)
before a caching allocator could reuse a just-freed block without risking a
still-in-flight kernel touching stale memory. **Milestone 24 does not
introduce CUDA streams or async execution** -- this is documented as a
condition for later, not solved now.

## 14. Memory-statistics API evaluation

M22's `CUDAMemoryStats` (`allocated_bytes`, `peak_allocated_bytes`,
`allocation_count`, `free_count`) remains sufficient for the direct
allocator Forge has today, and Milestone 24 does not extend it -- the new
allocation-*behavior* questions this milestone asks are answered by the
separate, optional `forge.cuda.profiler` facility instead (Section 2),
exactly because Section 22 of the brief asks for evaluation, not
unconditional expansion: "Do not necessarily add these to the production
API in M24 unless the profiling implementation genuinely needs them" -- it
did not.

**If Section 11's caching allocator is later implemented**, `CUDAMemoryStats`
should grow:

```text
allocated_bytes        (kept; "active" bytes under a caching allocator)
reserved_bytes         (new: allocated_bytes + cached_bytes -- real VRAM held)
cached_bytes           (new: reserved_bytes - allocated_bytes)
peak_allocated_bytes   (kept)
peak_reserved_bytes    (new)
allocation_count       (kept; would then mean "real cudaMalloc calls" -- cache hits would not increment it)
free_count             (kept; would then mean "real cudaFree calls" -- cache returns would not increment it)
cache_hit_count        (new)
cache_miss_count       (new)
```

This mirrors the offline simulation's own output fields (Section 8) --
deliberately, so the eventual real-allocator statistics and this
milestone's simulation results remain directly comparable.

## 15. Risks of a caching allocator

- **Reserved VRAM never shrinks automatically** on the 940MX's 2 GiB card --
  a workload with a brief large-batch phase followed by a smaller one would
  hold the large phase's cached blocks indefinitely without an explicit
  `empty_cache()` call (or an OOM-triggered purge), unlike today's allocator
  which returns memory to the driver the instant a `CUDAStorage` is freed.
- **Every invariant in Section 12 is a correctness requirement, not a
  performance one** -- a bug in the free-list bookkeeping (e.g. returning a
  block to the cache before its true last use) would silently corrupt
  results rather than crash, since CUDA does not fault on a
  use-after-logical-free the way host memory sometimes does.
- **Added code and testing surface** for a framework whose current
  correctness story (Milestones 8-23) benefits from the direct allocator's
  simplicity -- every `CUDAStorage`'s lifetime today is exactly "one
  `cudaMalloc` call, one `cudaFree` call," which made Milestone 22/23's
  audit tractable. A cache reintroduces "was this pointer actually just
  reused" as a question future audits must account for.
- **Benefit is workload-shaped**: Section 4/7's near-100% exact-size
  repetition is a direct consequence of Forge having no dynamic shapes.
  A hypothetical future workload with genuinely variable tensor shapes
  (e.g. variable-length sequences) would see far less reuse from an
  exact-size cache, changing the cost/benefit calculus measured here.

## 16. Conditions under which caching should be implemented

Implement Section 11's design when:
1. A real training workload's profiled allocation/free overhead (Section 9)
   is confirmed, via more precise interleaved (not isolated) measurement, to
   be a genuinely significant fraction (not just same-order-of-magnitude) of
   iteration time -- e.g. a future in-workload instrumentation that times
   `cf_malloc`/`cf_free` calls *as they occur inside* a real training step,
   rather than this milestone's isolated microbenchmark.
2. The target workload's allocation-size distribution continues to show the
   high repetition this MNIST CNN shows (Section 4/7) -- verified per new
   model architecture, not assumed to generalize.
3. VRAM headroom exists to tolerate reserved-but-cached memory not shrinking
   automatically (Section 15) -- particularly relevant on the 940MX's 2 GiB
   card, where a future larger model could make "cached memory the OS can't
   reclaim" a real constraint the direct allocator does not have.

Do not implement it merely because "`cudaMalloc` is generally considered
expensive" in the abstract -- Section 9's numbers are this environment's
own measurement, not a borrowed rule of thumb, and any future decision
should be similarly re-measured against whatever workload motivates it.

## Decision

```text
Caching allocator: JUSTIFIED (evidence-backed, not yet implemented)
```

Every measurement in this document points the same direction:
- 99%+ of allocated bytes are exact-size repeats across iterations (Section 7).
- An offline simulation shows a trivial exact-size cache would have reduced
  1,920 real `cudaMalloc`/`cudaFree` pairs to 42 real calls for an entire
  30-iteration run (Section 8).
- Directly measured `cudaMalloc`/`cudaFree` host-API cost on this hardware
  (~175-300 us/call) is the same order of magnitude as an entire training
  iteration (~25 ms) once multiplied by this workload's 64 allocations +
  64 frees per iteration (Section 9).
- No allocation in the trace lives longer than one iteration (Section 5),
  and true persistent memory is a small, flat 431 KiB against a 7.4 MiB
  transient peak (Section 6) -- exactly the shape of workload a cache
  benefits, with almost nothing for it to hold onto incorrectly.

This is a **recommendation for a future milestone**, not a mandate to
implement immediately: Section 16's conditions (particularly more precise
in-workload overhead attribution, beyond this milestone's isolated
microbenchmark) should be checked first, and Section 15's risks (chiefly,
VRAM that does not shrink automatically on a 2 GiB card) weighed against the
measured benefit at implementation time. Per the milestone's own
instructions, **no caching allocator was implemented in Milestone 24**.

## Implementation (Milestone 25)

Milestone 25 implements Section 11's design essentially as specified here:
`forge/backend/cuda/allocator.py`'s `CUDACachingAllocator`, an exact-size
`dict[nbytes, list[ptr]]` free-block cache sitting between `CUDABackend.
_alloc()`/`CUDAStorage.__del__()` and the real `cudaMalloc`/`cudaFree`
boundary. See `docs/architecture/cuda-backend.md`'s **CUDA Caching Allocator
(Milestone 25)** section for the full design writeup (ownership boundary,
OOM handling, statistics fields, `empty_cache()`, synchronization
assumptions, limitations) -- this section records the **measured results**
against this document's own predictions, on the same hardware (940MX, driver
582.53, CUDA 12.6).

### Allocation microbenchmark (Section 21) vs. Section 9's overhead numbers

`benchmarks/allocator_bench.py`, direct (`raw_malloc`/`raw_free`, bypassing
the cache) vs. cached (`allocate`/`release`, after a one-call warmup), 200
measured cycles per size:

| Size | Direct (mean) | Cached (mean) | Speedup | Cache hits / misses |
|---|---:|---:|---:|---|
| 4,096 B | 559-701 us | 3.9-5.8 us | 122-144x | 200 / 0 |
| 65,536 B | 10.0-10.5 us | 3.6-4.9 us | 2.1-2.8x | 200 / 0 |
| 1,048,576 B | 604-686 us | 2.0-2.5 us | 243-336x | 200 / 0 |

(Two runs each; ranges cover both.) The 65,536-byte "direct" figure is
consistently ~10 us here -- an order of magnitude below the 4 KB/1 MB direct
figures (~600-700 us, consistent with Section 9's original ~175-300 us/call
estimate at a different sample size/methodology) -- reproducibly, across
repeated runs. This looks like a WDDM/driver-level small-pool effect specific
to this exact byte count on this hardware, not something Forge's allocator
controls or explains; reported honestly rather than smoothed over, per this
document's own "environment-specific number, not a general CUDA claim"
convention (Section 9). It does not change the qualitative conclusion: the
cached path never issues a driver call after warmup (0 misses across 600
requests at 3 sizes in the multi-size interleaving check,
`bench_multi_size()`), so its cost floor is a few microseconds of pure Python
bookkeeping regardless of size, while every direct path pays a real driver
round-trip every time.

### M20 MNIST workload (Section 19-20) vs. Section 3's baseline

Real `examples.mnist.model.build_model()`, batch 64, `CrossEntropyLoss` +
`Adam`, 5 untracked warmup iterations then 30 measured (`benchmarks/mnist_
bench.py`, same configuration Section 3 used):

| Metric | M24 (direct, Section 3) | M25 (caching allocator) |
|---|---:|---:|
| Mean iteration time | 25.21 ms | 19.56-19.68 ms (two runs) |
| Real `cudaMalloc` calls, steady-state window | 64/iteration (1,920 over 30 iters) | **0** (cache already warm from the 5 untracked warmup iterations) |
| Cache hit count, steady-state window | n/a | 1,920 (100% of requests) |
| Real `cudaMalloc` calls, cold process start through 35 iterations (warmup + steady) | ~2,240 (direct model, extrapolated) | **66** |
| Persistent (`allocated_bytes`) before/after | 440,992 B, flat | 440,992 B, flat (unchanged) |
| Peak active bytes | 7,789,424 B | 7,789,424 B (unchanged -- caching does not change *active* peak) |
| Reserved bytes (steady state) | n/a (no concept under direct allocation) | 9,573,304 B |
| Cached bytes (steady state) | n/a | 9,132,312 B |

Mean iteration time drops **~22%** (25.21 ms -> ~19.6 ms), consistent across
two independent runs (stdev ~1.8 ms each) -- large enough relative to
Milestone 21's documented WDDM clock/thermal variance to be a real effect,
not noise. Real driver `cudaMalloc` calls collapse from a projected ~2,240
(one per allocation, direct model, over a cold-start 35-iteration run) to
**66** -- a **97.1% reduction** -- closely matching, and in the measured
steady-state window *exceeding*, Section 8's offline simulation's predicted
97.8% hit rate / 42-call figure. The steady-state window alone shows exactly
**zero** new driver calls: because this measurement's own 5 warmup iterations
already ran through the same live caching allocator (unlike Section 8's
simulation, which modeled a cache starting cold at the beginning of its
30-iteration trace), every one of the 14 distinct sizes is already cached by
the time the timed window begins.

### Conclusion

Every prediction in this document's **Decision** held on real hardware: the
exact-size cache captures essentially all of this workload's reuse
opportunity, real driver-call volume collapses by two orders of magnitude,
and a real (not merely projected) ~22% mean-iteration-time improvement was
measured and reproduced. `peak_allocated_bytes` (active memory) is unchanged
by caching, as designed -- only `reserved_bytes`/`cached_bytes` grow, and
`forge.cuda.empty_cache()` reclaims them on demand. See `docs/architecture/
cuda-backend.md` for the full API/statistics/ownership writeup and `docs/
performance/benchmarking.md` for the benchmark-harness-level methodology
notes.

## Milestone 26: Synchronization Contract (Formalized)

Section 13 above (**CUDA asynchrony implications**) stated the reuse-safety
assumption this allocator depends on informally, as a condition for the
M25 implementation to lean on. Milestone 26 audited every CUDA-touching code
path in Forge end-to-end (kernel launches, memory copies, autograd,
optimizers, `Trainer`, persistence) specifically to confirm that assumption
against the actual CUDA API semantics rather than restating it, and adds
`forge.cuda.synchronize()` as Forge's public synchronization primitive. The
full audit lives in `docs/architecture/cuda-backend.md`'s **CUDA Execution
and Synchronization Semantics (Milestone 26)** section; this is the
allocator-specific consequence, stated as a formal contract:

> **Current M26 contract.** Forge issues all CUDA work -- every kernel
> launch (`kernels.cu` creates no stream, so every launch runs on CUDA's
> default stream) and every memory copy (`cf_memcpy_h2d`/`_d2h`/`_d2d`, all
> plain synchronous `cudaMemcpy`, never `cudaMemcpyAsync`) -- in program
> order on that one stream. Every `CUDABackend` operation calls
> `cudaDeviceSynchronize()` (`CUDABackend._synchronize`) before returning its
> result to Python. Consequently, a `CUDAStorage` never becomes unreachable
> (triggering `__del__` -> `CUDACachingAllocator.release()`) while any kernel
> that reads or writes its memory is still in flight -- the operation that
> last touched that memory already synchronized before control returned to
> Python, and no Forge CUDA work is ever issued out of that order. The
> allocator may therefore hand a cached block to a new `CUDAStorage` (a
> cache hit, in `CUDACachingAllocator.allocate()`) with **no additional
> synchronization of its own** -- the safety property is established by the
> per-operation synchronization every `CUDABackend` method already performs,
> not by anything the allocator itself does.

This is not a new design decision -- `CUDACachingAllocator.allocate()`/
`release()` are byte-for-byte unchanged by Milestone 26. It is the same
"exact-size cache, no reuse-time synchronization" design M25 already shipped,
now backed by an explicit, verified statement of *why* it is correct rather
than an inherited assumption. `empty_cache()` inherits the identical
reasoning: a block can only ever be in the free list after its last use
already synchronized, so freeing it back to the driver needs no
synchronization step of its own either.

**Verification**: `tests/test_cuda_synchronize.py::
test_allocator_reuse_does_not_corrupt_data` and `::
test_allocator_reuse_across_many_alloc_release_cycles_stays_correct`
allocate, use, and release a block, then allocate a same-size block
(confirmed via `CUDAMemoryStats.cache_hit_count` to actually reuse the freed
one) and prove its readback is exactly correct -- with no `forge.cuda.
synchronize()` call anywhere between the release and the reuse. `::
test_empty_cache_after_recent_work_is_safe_and_preserves_correctness` proves
the same for `empty_cache()` specifically.

**What would have to change for this to break**: only the introduction of
multiple CUDA streams (explicitly out of scope for Milestone 26 and every
prior milestone) -- see `docs/architecture/cuda-backend.md`'s **Future
Stream-Aware Design** subsection for exactly what a stream-aware allocator
would need (per-block last-use-stream tracking, CUDA events, deferred/
stream-ordered freeing) before this contract's "no additional
synchronization on reuse" claim could be restated safely under multiple
streams.

## Milestone 27: Pending Blocks (Streams Implemented)

Milestone 27 is exactly the "what would have to change" scenario named
above: it introduces real CUDA streams (`forge.cuda.Stream`), and this
allocator is no longer allowed to assume every release is already safe to
reuse. The **M26 contract restated above still holds, unchanged, for any
block released by a `CUDAStorage` whose `last_stream` is `None`** (the
default-stream compatibility mode, Forge's unchanged historical behavior --
see `docs/architecture/cuda-streams.md`'s **Default stream compatibility
mode** section). It does *not* hold for a block released by a storage last
used on an explicit `Stream`; those blocks now go through a new path:

> **M27 contract, pending blocks.** `CUDACachingAllocator` tracks a third
> block state, *pending* (alongside *active* and *ready*): a block released
> by a `CUDAStorage` whose `last_stream` was a real `Stream`, not yet known
> to be safe to reuse. `CUDAStorage.__del__` routes such a release through
> `release_pending()`, which records a real `CUDAEvent` on that storage's
> `last_stream` *at release time* -- correct because CUDA streams execute in
> strict per-stream program order, so every operation that touched this
> storage was necessarily enqueued before this release could run. The block
> becomes eligible for reuse (`CUDACachingAllocator.allocate()`'s
> `_try_reclaim_pending`) the moment that event is observed complete
> (`CUDAEvent.query()`), checked opportunistically on the next same-size
> request, after the ready free list and before a real `cudaMalloc`. Forge
> never forces an event to complete early (e.g. via
> `cudaDeviceSynchronize()`) merely to make a block reusable sooner --
> that would defeat asynchronous execution.

Full design, invariants, and hardware verification: `docs/architecture/
cuda-streams.md`'s **Allocator Changes**/**Pending Allocation Model**/
**Same-Stream Reuse**/**Cross-Stream Reuse**/**Memory Statistics** sections.
`CUDAMemoryStats` gains `pending_bytes`/`pending_count`; `cached_bytes` now
means specifically *ready* bytes (`reserved_bytes - allocated_bytes -
pending_bytes`); `empty_cache()` now *waits* (`CUDAEvent.synchronize()`) for
each pending block before freeing it, a real, documented cost change from
M25/M26 where `empty_cache()` never needed to wait for anything.

Every M25/M26 test in `tests/test_cuda_allocator.py` continues to pass
unmodified: they exercise only the default-stream path, which is byte-for-
byte unchanged. New Milestone 27 tests live in `tests/
test_cuda_stream_allocator.py`.

## Milestone 28: Unaffected by Cross-Stream Dependencies

Milestone 28 replaced M27's "fail clearly" cross-stream Tensor policy with
automatic `cudaStreamWaitEvent`-based dependencies (`docs/architecture/
cuda-streams.md`'s **Milestone 28** section) -- but changed nothing in this
allocator. `CUDAStorage.__del__`, `release()`/`release_pending()`, and
`_try_reclaim_pending()` above are byte-for-byte the same code as M27. The
reason this remains safe: `CUDABackend._stream_guard` (`backend.py`) still
updates every storage's `last_stream` to the *current* stream after
establishing whatever dependency was needed, exactly as it always updated
`last_stream` for every touched storage in M27 -- so `last_stream` continues
to mean "the stream that will next touch this storage," cross-stream reads
included, which is precisely what `release_pending()` needs to record the
correct event. Verified directly:
`tests/test_cuda_stream_dependencies.py::
test_empty_cache_remains_safe_after_cross_stream_dependencies` and
`::test_repeated_cross_stream_dependencies_do_not_grow_cuda_allocation` (100
repeated cross-stream release/reallocate cycles with no growth in
`allocated_bytes`).

## Milestone 29: Unaffected by Pinned Memory / Async Transfers

An asynchronous H2D transfer's destination device memory is allocated
through this exact allocator (`CUDABackend._alloc`, unmodified) and its
resulting `CUDAStorage.last_stream` is set by the same unconditional
`CUDAStorage.__init__` logic every other storage uses -- so releasing it
before the transfer completes goes through the *existing* `release_pending()`
path with zero new allocator code. See `docs/architecture/cuda-transfers.md`'s
**Allocator Integration** section and the mandatory race test,
`tests/test_cuda_transfer_allocator.py::
test_async_h2d_release_never_hands_the_still_in_flight_block_to_another_stream`.

Pinned *host* memory is tracked entirely separately (`forge/backend/cuda/
pinned.py`'s own small counter, not this module) and is never cached --
direct `cudaHostAlloc`/`cudaFreeHost` per allocation, deliberately simpler
than this module's ready/pending free-list model (Section 25 of the
Milestone 29 brief: build a pinned caching allocator only if profiling
demonstrates it is necessary). `forge.cuda.empty_cache()` remains
device-allocator-only and does not touch pinned allocations.
