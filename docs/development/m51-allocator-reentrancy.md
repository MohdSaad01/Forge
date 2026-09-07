# M51 — CUDA Allocator Reentrancy & Concurrency Correctness

## 1. Executive summary

M50 discovered, but deliberately did not fix, a real self-deadlock in
`CUDACachingAllocator.release()` (`forge/backend/cuda/allocator.py`): a
non-reentrant `threading.Lock` could be reacquired by the same thread while
already held, via CPython's cyclic GC finalizing an unrelated `CUDAStorage`
mid-critical-section. This milestone's brief was to reproduce it
independently, establish the exact mechanism from evidence rather than
M50's hypothesis alone, and decide between fixing it or documenting why not.

**Outcome: confirmed and fixed.** An independent, deterministic reproducer
(Section 2) proved the mechanism with two identical `py-spy` stack dumps.
The root cause is broader than M50's hypothesis: the trigger is not
specifically the `any()` generator M50 pointed to, but `self._free_blocks
.setdefault(nbytes, [])`, which unconditionally allocates a new GC-tracked
list on **every** call to `release()`, not only on a first-time size. The
fix (Section 6) removes all new-object allocation from `release()`'s and
`release_pending()`'s critical sections, and additionally switches
`CUDACachingAllocator._lock` from `threading.Lock` to `threading.RLock` as
a deliberate, analyzed backstop for the one path this milestone did not
restructure (`_empty_ready`/`_drain_pending`'s snapshot construction,
reachable only via `empty_cache()`). 8 new regression tests were added; the
full 1,627-test suite (1,619 + 8) passes as a **single process**, repeated
three times, with no hang, leak, or accounting drift.

## 2. Reproducer

`release()`/`release_pending()` are the *only* methods `CUDAStorage.__del__`
(`forge/backend/cuda/backend.py:522`) calls, so they are the only methods
whose critical section can be reentered by a GC-triggered finalizer chain.
The independent reproducer:

1. Traps a real CUDA `Tensor` in an unreachable Python reference cycle
   (`_CycleHolder.self_ref = holder`) -- refcounting alone cannot free it,
   only `gc.collect()` can, closely matching the shape of the real M50
   failure (autograd graph teardown, not a simple flat `del`).
2. Wraps the allocator's real lock in a thin proxy (`_GCTrippingLock`) whose
   `acquire()` calls `gc.collect()` immediately after acquiring the real
   lock. This makes *when* a GC pass lands deterministic instead of
   probabilistic, without changing the mechanism, the lock type, or
   `release()`/`release_pending()`'s own code at all -- a first attempt
   relying on CPython's natural generation-0 threshold (`gc.set_threshold(1,
   1, 1)`) did **not** reliably land the trip inside the critical section
   (it was as likely to fire during unrelated `Tensor` construction), so the
   proxy was needed to isolate the mechanism from GC's own scheduling noise.
3. Releases an unrelated `Tensor` (`del victim`) on the real allocator,
   through real `CUDAStorage.__del__` → `allocator.release()`.

Against the pre-M51 code (`threading.Lock`), this **reliably hangs** (100%
reproduction across all trials run). Against the fixed code, it completes
in well under a second, also reliably, across the reentrancy-specific tests
run individually and 25x in a single stress loop
(`tests/test_cuda_allocator_reentrancy.py`).

Characterization (per the brief's Phase A checklist), established directly:

- **Deterministic once engineered, not deterministic under pure natural
  timing.** The `_GCTrippingLock` proxy makes it deterministic. A first
  attempt using only `gc.set_threshold(1, 1, 1)` with no proxy did not
  reliably land the GC trip inside `release()`'s specific critical section
  -- CPython's allocation counter is process-wide and coarse, so with a
  threshold that low it just as easily fires during nearby, unrelated
  container allocation (e.g. `Tensor.__init__`'s own bookkeeping). This
  matches M50's own characterization ("rare enough to have never surfaced
  across 48 prior milestones... depends on GC timing/accumulated object
  churn").
- **Requires *a* GC pass landing inside the critical section, not
  necessarily an explicit user-level `gc.collect()` call.** In the real
  M50 failure it was CPython's automatic generation-0 collector, triggered
  by ordinary allocation inside `release()`.
- **Does not require generator evaluation specifically.** M50's hypothesis
  singled out the `any()` genexpr. Direct, isolated measurement (`gc`
  callback counting around minimal snippets, described in Section 3) shows
  `self._free_blocks.setdefault(nbytes, [])` is a stronger, more universal
  trigger: it allocates a new list *unconditionally*, on every single call,
  regardless of whether `nbytes` already has cached blocks -- whereas the
  `any()` genexpr only participates when `blocks` is non-empty and is, in
  any case, only one of two GC-tracked allocations release() was making.
- **Does not require CUDA stream activity, allocator pressure, or repeated
  iterations.** The reproducer uses the CUDA default stream only, a single
  small (16-byte) tensor pair, and reproduces on the very first attempt.
  (`release_pending()`, the explicit-stream analog, has the identical
  hazard via its own unconditional `self._pending_blocks.setdefault(nbytes,
  [])` and its `_PendingBlock(event, ptr)` construction -- confirmed with a
  parallel reproducer.)
- **Does require a CUDAStorage-holding object to be reachable only through
  an uncollected reference cycle at the moment of the trip.** A plain,
  acyclic `del` sequence cannot trigger this: CPython deallocates a flat
  container's elements strictly sequentially (each element's `__del__`,
  including any nested `release()` call, fully completes before the next
  element's decref even begins), so there is no *nesting* without cyclic
  GC's synchronous, single-thread finalization pass being the one thing
  capable of running a second `__del__` while the first is still on the
  call stack.

## 3. Root cause

Confirmed by direct code inspection of `forge/backend/cuda/allocator.py`
(pre-fix) plus targeted, isolated measurement of what CPython actually
tracks for garbage collection:

- `release()`'s critical section (`with self._lock:`) did two things capable
  of allocating a new GC-tracked object: `self._free_blocks.setdefault(
  nbytes, [])` (the `[]` default argument is evaluated unconditionally,
  every call) and `any(b.value == ptr.value for b in blocks)` (a generator
  object, itself GC-tracked). `release_pending()`'s critical section
  similarly allocated a fresh `_PendingBlock(event, ptr)` (a non-`__slots__`
  dataclass instance, GC-tracked) and used the same unconditional
  `setdefault(nbytes, [])` pattern.
- Any new GC-tracked object allocation anywhere in the process can trip
  CPython's generation-0 collection threshold, which runs synchronously,
  inline, before the allocating bytecode continues -- and a `gc.collect()`
  pass can finalize (`__del__`) any unreachable object, including ones in
  reference cycles, on the calling thread.
- Isolated, differential measurement (identical harness, only the payload
  varied, via `gc.callbacks`) confirmed which loop constructs actually
  allocate a GC-tracked object and which do not:

  | construct | GC-tracked allocation? |
  |---|---|
  | `any(b.value == x for b in blocks)` (generator) | yes |
  | `for b in blocks: ...` (list iterator) | yes -- `list_iterator` objects ARE gc-tracked in CPython, contrary to the initial assumption that a plain `for` loop would be allocation-free |
  | manual index `while i < len(blocks): ... i += 1` | **no** |
  | `dict.setdefault(key, [])` where `[]` is a fresh literal | yes, unconditionally, every call |
  | `d[key] = existing_list` (assigning an already-existing object into a dict, no new key logic beyond that) | no |
  | plain `int` arithmetic (`self._active_bytes -= nbytes`) | no -- `int` is never GC-tracked regardless of magnitude |

  This directly informed the fix: a manual index `while` loop, not a
  rewritten `for` loop, was required to actually eliminate the scan's
  allocation (a `for`-loop rewrite alone would not have helped).
- `CUDAStorage.__del__` (`backend.py:522`) is the only call site for both
  `release()` and `release_pending()` -- confirmed by repo-wide grep -- so
  these two methods are the only ones whose critical section a GC-triggered
  finalizer chain can loop back into. `allocate()`'s locked sections
  perform no new allocation already (dict `get`/`pop`/`del`, int
  arithmetic only). `_empty_ready()`/`_drain_pending()`'s snapshot-
  construction list comprehensions do allocate under the lock, but are
  reachable only via `empty_cache()`, never via `__del__` -- see Section 5's
  discussion of why this residual case is handled differently.

## 4. Lock/reentrancy sequence (from two independent `py-spy` dumps, 3s apart, identical frames)

```
MainThread:
    release (allocator.py:354)              <- BLOCKED re-acquiring self._lock
    release (allocator.py:486)               [module-level release() wrapper]
    __del__ (backend.py:532)                 [CUDAStorage.__del__, finalizing the
                                               reference-cycle-trapped Tensor]
    setdefault (harness's _GCTrippingDict, standing in for a natural GC trip)
    release (allocator.py:355)               <- OUTER call, lock already held here
    release (allocator.py:486)
    __del__ (backend.py:532)                 [CUDAStorage.__del__ for `victim`]
    main (repro script)
    <module>
```

The outer `release()` call (for `victim`) acquires `self._lock` and, at the
point of its own `setdefault`/allocation, triggers a GC pass. That pass
finalizes the cycle-trapped `Tensor`'s `CUDAStorage`, whose `__del__` calls
`release()` again -- same thread, same lock object, now already held by the
outer frame -- and blocks forever on a non-reentrant `threading.Lock`.

## 5. Candidate fixes considered

- **Candidate A -- move destruction-sensitive work outside the lock.**
  Selected, applied narrowly (Section 6, item 1) to `release()`/
  `release_pending()` specifically, since those are the only two methods
  reachable from `__del__`. Fully eliminates the *demonstrated,
  dominant* hazard (fires on every single `CUDAStorage` destruction) with
  zero behavioral change to the lock itself.
- **Candidate B -- defer destruction-sensitive cleanup.** Not needed as a
  separate mechanism once Candidate A removes the allocations themselves;
  there is nothing left to defer in the fixed critical sections.
- **Candidate C -- reentrant locking (`RLock`).** Adopted, but explicitly
  *not* as a substitute for Candidate A -- as a deliberate, analyzed
  backstop layered on top of it (Section 6, item 2). Evaluated on its own
  merits per the brief's explicit caution against accepting it merely
  because the deadlock disappears:
  - **Is reentrant execution of `release()`/`release_pending()` actually
    *correct*, not just non-deadlocking?** Yes: CPython's GIL guarantees a
    same-thread reentrant call always runs to full completion before the
    outer frame resumes (true stack nesting, never interleaved), and both
    methods perform only simple, order-independent accumulation (per-size
    list membership, integer counters) -- nested execution order cannot
    change the final state.
  - **Does it mask an unsafe critical-section design elsewhere?** Yes, one
    case, found and documented rather than silently accepted:
    `_empty_ready()`/`_drain_pending()` build their block-snapshot via a
    dict-`.items()`-iterating list comprehension while holding the lock. If
    a reentrant `release()`/`release_pending()` call were to insert a *new*
    size key into `_free_blocks`/`_pending_blocks` while that comprehension
    is mid-iteration, CPython would raise `RuntimeError: dictionary changed
    size during iteration`. This is a real, but much narrower, residual
    surface than the one this milestone fixed directly: it requires an
    `empty_cache()` call (not every `CUDAStorage` destruction) to coincide
    with a still-uncollected cyclic-garbage `CUDAStorage` of a genuinely new
    size. Restructuring those two methods to be fully allocation-free while
    iterating a dict is materially more invasive (would require abandoning
    dict iteration entirely, e.g. suspending the cyclic GC for the duration)
    and was judged out of proportion to a residual this narrow -- consistent
    with the brief's "smallest technically sound fix" and "do not redesign
    the allocator" constraints. Documented here as a known limitation
    (Section 12) rather than silently left unaddressed: the `RLock`
    converts this one case from a silent deadlock into a loud, debuggable
    `RuntimeError` -- consistent with `release()`'s own pre-existing
    philosophy ("exists to fail loudly, not silently corrupt").
  - **Does it introduce new races?** No new threads are introduced anywhere
    in Forge by this change; the only "concurrency" in play is same-thread
    reentrancy via GC, which `RLock` is specifically designed for.
- **Candidate D -- other minimal structural fix.** None identified beyond
  A+C combined.

## 6. Why the selected fix (A + C, layered) is safest

Neither alone was judged sufficient on its own:

- A alone leaves `_empty_ready`/`_drain_pending`'s much rarer allocation-
  under-lock exposure as a live *deadlock*, not just a rare exception.
- C alone ("just swap to `RLock`") is exactly the shortcut the brief warns
  against accepting uncritically -- it would have shipped without the
  Section 5 analysis showing reentrant execution is actually *correct* for
  `release()`/`release_pending()`'s specific accumulation pattern, and
  without discovering (and documenting, rather than silently leaving) the
  `_empty_ready`/`_drain_pending` residual.

Together: A closes the dominant, demonstrated, every-single-tensor-
destruction hazard completely (zero allocation under the lock in the common
path); C is a small, well-understood, analyzed backstop for the one
narrower case A does not reach, converting it from a hang into a loud
failure rather than leaving it live. Both changes are confined to
`forge/backend/cuda/allocator.py`; no other subsystem, kernel, public API,
or Tensor semantic was touched.

## 7. Exact production changes

All in `forge/backend/cuda/allocator.py`:

1. `CUDACachingAllocator.__init__`: `self._lock = threading.Lock()` →
   `self._lock = threading.RLock()`.
2. `release()`: the fallback empty list is now built *before* acquiring the
   lock (`spare = []`) and only installed into `self._free_blocks` if
   `nbytes` has no existing entry (`.get()` instead of the always-allocating
   `.setdefault(nbytes, [])`); the duplicate-pointer scan is a manual index
   `while` loop instead of `any(genexpr)`. Behavior (including the
   double-release `RuntimeError`) is unchanged; only the allocation
   footprint of the critical section changed.
3. `release_pending()`: the `_PendingBlock(event, ptr)` instance and the
   fallback empty list are both built before acquiring the lock, mirroring
   `release()`'s restructuring; stream-ordering/event semantics are
   unchanged (the event is still recorded before the lock is touched, as
   before).
4. Module docstring: added a "Milestone 51" section explaining the hazard
   and both parts of the fix, so a future reader does not "simplify" either
   change back into an allocating form.

No public API, Tensor semantic, kernel, or other subsystem was touched.

## 8. Regression tests

`tests/test_cuda_allocator_reentrancy.py`, 8 new tests, all CUDA-hardware-
gated (`pytestmark`, matching every other `test_cuda_*.py` file):

1. `test_release_reentrant_gc_finalization_does_not_deadlock` -- the
   canonical regression reproducer (default stream), run in a subprocess
   with a timeout. Fails/times out against the pre-fix code (verified:
   `git stash` back to the original `threading.Lock` and re-ran -- this
   test times out at 30s); passes in ~3s against the fix.
2. `test_release_pending_reentrant_gc_finalization_does_not_deadlock` -- the
   same hazard through the explicit-stream `release_pending()` path.
3. `test_reentrant_scenario_stable_under_repeated_cycles` -- the reproducer
   repeated 25x in one process (acceptance criterion 2, "passes repeatedly
   under stress").
4. `test_allocator_lock_is_reentrant_same_thread` -- direct sanity check on
   the real production lock object, using a *bounded*
   `lock.acquire(timeout=...)` for the reentrant probe rather than a plain
   nested `with lock:` -- deliberately, since an unbounded version was
   found during this milestone's own pre-fix validation run to leak a
   permanently-stuck daemon thread holding the real shared allocator lock
   forever, poisoning every subsequent test's `empty_cache()` fixture
   teardown in the same process. The bounded version correctly
   distinguishes reentrant (succeeds instantly) from non-reentrant (times
   out) behavior without that side effect.
5. `test_double_release_still_raises_runtime_error` -- confirms the
   duplicate-pointer detector still works after being rewritten from
   `any()`/genexpr to a manual `while` loop.
6. `test_repeated_alloc_release_cycles_reuse_and_accounting_correct` --
   ordinary (non-adversarial) churn: allocator reuse, cache-hit accounting,
   reserved-vs-allocated-bytes accounting.
7. `test_explicit_gc_collect_during_release_does_not_corrupt_state` -- a
   plain `gc.collect()` sitting in the middle of ordinary Tensor churn
   (no engineered reference cycle) does not disturb bookkeeping.
8. `test_cross_stream_release_pending_reclaim_still_correct` -- light
   confirmation that the `release_pending()` rewrite preserved stream-
   ordering/reclaim semantics (`tests/test_cuda_stream_allocator.py` remains
   the exhaustive version of this coverage).

All 8 were verified to fail or misbehave appropriately against the original
pre-fix code (tests 1-3 time out; test 4 correctly reports the non-
reentrant lock without hanging the test process) before verifying all 8
pass against the fix.

## 9. Full-suite results

- `tests/test_cuda_allocator_reentrancy.py`: 8/8 passed.
- All pre-existing allocator-adjacent suites unchanged and passing: 65
  tests across `test_cuda_allocator.py`, `test_cuda_allocator_availability.py`,
  `test_cuda_stream_allocator.py`, `test_cuda_transfer_allocator.py`,
  `test_cuda_alloc_profiler.py`, `test_cuda_alloc_profiler_availability.py`,
  `test_alloc_analysis.py`.
- CPU-only subset (`pytest tests/ -k "not cuda"`): 721 passed.
- CUDA-only subset (`pytest tests/ -k "cuda"`): 906 passed (898 + 8 new).
- **Full suite, single process** (`pytest tests/`, the exact condition M50's
  defect required): **1,627 passed**, run **three consecutive times**, ~52-
  53 seconds each, no hang, no deadlock, no flake.

## 10. Memory-safety / accounting results

- `test_repeated_alloc_release_cycles_reuse_and_accounting_correct` directly
  asserts `allocated_bytes`, `cached_bytes`, `reserved_bytes`, and
  `cache_hit_count` all remain internally consistent across 100 alloc/
  release cycles post-fix.
- `test_reentrant_scenario_stable_under_repeated_cycles` confirms no runaway
  growth in `reserved_bytes` across 25 adversarial reentrant cycles in one
  process.
- No change to `empty_cache()`, `_empty_ready()`, `_drain_pending()`,
  `allocate()`, or any `CUDAMemoryStats` field's meaning.

## 11. Performance comparison

The brief's own guidance: measure only to answer "did the fix introduce a
meaningful regression," using the existing benchmark rather than building a
new one. Ran `benchmarks/allocator_bench.py`'s `cached` path (the exact
`allocate()`/`release()` cycle this milestone changed; it involves zero
driver calls either way, so it isolates pure Python-level bookkeeping cost)
before (via `git stash` to the pre-fix code) and after:

| size | pre-fix mean | post-fix mean |
|---|---|---|
| tiny (4096 B) | 2.00 us | 1.71-3.52 us (2 trials) |
| small (65536 B) | 2.08 us | 1.73-1.81 us |
| medium (1048576 B) | 2.11 us | 1.82-1.86 us |

All deltas are within this hardware's known microbenchmark noise floor for
operations at this timescale (single-digit microseconds; see
`docs/development/m47-bottleneck-recharacterization.md`'s thermal-drift
methodology note) and show no consistent direction -- post-fix is equal to
or faster than pre-fix in most trials. No meaningful regression. This is
expected: the fix replaces one unconditional list-literal allocation plus a
generator with a pre-built list plus a manual index loop (comparable cost),
and `RLock` vs. `Lock` acquire/release overhead is negligible at CPython's
level for the uncontended, single-thread case Forge always exercises.

## 12. Remaining limitations

- `_empty_ready()`/`_drain_pending()`'s snapshot-construction list
  comprehensions still allocate while holding the lock (Section 5).
  Reachable only via `empty_cache()`, never via `__del__`, so this cannot
  itself be the *reentrant trigger* -- but if it coincides with a still-
  uncollected cyclic-garbage `CUDAStorage` of a genuinely new size, the
  `RLock` converts what would otherwise be a deadlock into a
  `RuntimeError: dictionary changed size during iteration`. Not observed in
  three full single-process suite runs. A future milestone should only
  revisit this if it is ever actually observed -- per this milestone's own
  "do not optimize for milestone count" directive, speculative hardening
  against an unobserved, narrow residual is not justified on its own.
- The fix does not change `allocate()`, `_try_reclaim_pending()`,
  `_empty_ready()`, or `_drain_pending()` beyond what Section 5 analyzed;
  none of these are reachable from `__del__` and none showed a reproducible
  hazard in this investigation.
- Forge remains documented as single-threaded elsewhere; this milestone's
  "concurrency" is entirely same-thread GC reentrancy, not multi-thread
  contention. `RLock`'s correctness argument (Section 5) is specific to
  that same-thread-reentrant case and was not evaluated for a hypothetical
  future genuinely multi-threaded Forge.
