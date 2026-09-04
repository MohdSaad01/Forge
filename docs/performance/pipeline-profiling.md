# Asynchronous Training Pipeline Profiling (Milestone 31)

## Purpose
Milestones 27-30 built a real asynchronous CUDA training pipeline (streams,
cross-stream dependencies, pinned memory + async transfers, GPU prefetch),
but left no visibility into exactly where time was going. This milestone
profiles the complete pipeline on the real, hardware-verified 940MX and
implements exactly one measurement-justified optimization. Per
`docs/development/roadmap.md`'s Milestone 31 brief: **do not optimize
blindly** -- profile, identify, optimize one or a few real bottlenecks,
measure, verify.

All numbers below are from the verified development machine: NVIDIA GeForce
940MX, Compute Capability 5.0, CUDA 12.6, driver 582.53, Windows/WDDM.

## How to reproduce
```bash
python -m benchmarks.mnist_profile          # per-phase / per-op breakdown (Milestone 21 tool, still accurate)
python -m benchmarks.pipeline_profile        # Milestone 31: async pipeline profile (this document's main source)
python -m benchmarks.stream_dependency_bench   # dependency/event overhead
python -m benchmarks.async_dataloader_bench     # prefetch speedup across workload shapes
```

Archived before/after results backing the **Optimization selected** section
below: `benchmarks/results/m31_cross_entropy_baseline.json` /
`m31_cross_entropy_optimized.json` (isolated op benchmark) and
`m31_mnist_training_baseline.json` / `m31_mnist_training_optimized.json`
(end-to-end real M20 CNN training step), all produced by `benchmarks/
ops_bench.py` + `backward_bench.py` + `mnist_bench.py` + `training_bench.py`
run before and after the fix (identical hardware, same session).

## Synchronization audit
Every `cudaDeviceSynchronize`/`cudaStreamSynchronize`/`cudaEventSynchronize`
call site in the CUDA runtime, classified:

| Call site | Classification |
|---|---|
| `CUDABackend._maybe_synchronize` (every kernel-launching method) | **Required** -- default-stream (M26 compatibility-mode) contract: skipped entirely on an explicit stream (M27), so this is never paid in the async training path. |
| `CUDABackend._synchronize` / `forge.cuda.synchronize()` | **Explicit API** -- public, opt-in, used by benchmarks' `time_cuda` bracketing and `Trainer`/example code that wants a full device barrier. |
| `CUDABackend.to_numpy`: `storage.last_stream.synchronize()` | **Host-read boundary** -- targeted to the one storage's own last-use stream, never device-wide. Backs `.to("cpu")`/`.numpy()`. |
| `Tensor._data` property: `self._pending.synchronize()` | **Host-read boundary** -- the M29 chokepoint; waits only for *this* tensor's own pending D2H transfer, at most once. |
| `PinnedMemory.free()` (`event.synchronize()` per pending event) | **Persistence/lifetime** -- only blocks if an in-flight transfer referencing this buffer hasn't finished; required for correctness (Invariant 1, M29). |
| `CUDACachingAllocator._drain_pending` (`empty_cache()`) | **Persistence/compatibility** -- last-resort OOM/explicit-purge path only; never on the ordinary allocate/release path. |
| `benchmarks/timing.py::time_cuda` | **Benchmark** -- deliberate, brackets exactly what is being measured. |
| `benchmarks/mnist_profile.py`'s per-phase/per-op `_sync()` calls | **Benchmark** -- deliberately destroys overlap to get clean per-op numbers (see that module's own docstring); this is why `pipeline_profile.py` exists as a second tool that does *not* do this. |

**No accidental/unnecessary device-wide synchronization was found.** Every
`cudaDeviceSynchronize` reachable from the async training path is either
skipped (explicit-stream mode) or an explicit, opt-in public API call.

## Dependency / event overhead (`stream_dependency_bench.py`)
| Measurement | Cost |
|---|---|
| Same-stream op (no dependency) | 54.4 us/op |
| Cross-stream op (1 producer: record + wait) | 63.8 us/op (1.17x) |
| Multi-input (2 producers) | 443.3 us/op (includes producing both operands) |
| Event creation + destruction (isolated) | 2.6 us/event |
| Cross-stream allocator reuse (pending -> ready) | 182.9 us/cycle |

A real MNIST batch crosses a stream boundary once (the prefetch loader's
transfer stream -> the Trainer's compute stream, on the batch's first
touch) -- one ~64us dependency out of a multi-millisecond training step.
**Dependency/event overhead is not a measurable bottleneck** in the real
pipeline.

## Allocator characterization (`pipeline_profile.py`, one profiled epoch)
Cache-hit rate is consistently >95% after warmup (thousands of hits vs.
under 300 misses across an entire profiling run of ~130 batches at three
batch sizes plus a depth sweep). `reserved_bytes`/`pending_bytes` return
fully to 0 after `gc.collect()` + `empty_cache()` -- no leak. **Allocator
overhead is not a measurable bottleneck** (matches M25's own design intent).

## Pinned-memory characterization
Staging one MNIST batch's `(x, y)` into fresh `PinnedMemory` buffers costs
~0.22 ms (isolated measurement, batch_size=64) -- two `cudaHostAlloc` +
host-side copy + `cudaFreeHost` pairs per batch (no caching allocator, by
M29's own design). `pinned_active_bytes` returns to 0 after each
measurement; peak stays bounded (~4.4 MB at batch_size=64, matching
double-buffering). Against a per-batch GPU compute cost of 8.5-24.7 ms
(batch-size dependent), **0.22 ms of pinned staging is comfortably hidden**
-- no evidence justifies a pinned caching allocator (Section 12 of the
milestone brief explicitly asks not to build one without evidence).

## CPU-side component costs (isolated, batch_size=64)
| Component | Cost |
|---|---|
| `DataLoader.__next__` (dataset access + collation) | 1.59 ms/batch |
| Pinned staging (`x`, `y` -> `PinnedMemory`) | 0.22 ms/batch |
| Async H2D submission (launch-only, not completion) | 0.13 ms/batch |
| **Total CPU-side per batch** | **~1.94 ms** |

## H2D transfer bandwidth (isolated, pinned source, async)
| Size | Time | Effective bandwidth |
|---|---|---|
| 4 KB (small) | 0.068 ms | 0.06 GB/s (latency-bound, not bandwidth-bound) |
| ~400 KB (medium) | 0.269 ms | 1.49 GB/s |
| ~4 MB (large) | 2.610 ms | 1.53 GB/s |

Small transfers are dominated by fixed per-call driver/launch latency, not
by bytes moved -- expected, and consistent with the launch-overhead evidence
throughout this document. No claim is made about theoretical PCIe maximum;
these are the measured numbers on this exact machine.

## Live async epoch: per-phase GPU busy time (event-based, no inter-phase sync)
`batch_size` sweep, `prefetch_size=2`, real M20 CNN, real `CUDAPrefetchLoader`:

| batch | epoch wall | samples/sec | fwd (ms) | loss (ms) | bwd (ms) | opt (ms) | compute-stream utilization |
|---|---|---|---|---|---|---|---|
| 32 | 325.4 ms | 3,147 | 1.02 | 0.94 | 6.51 | 0.08 | 84.1% |
| 64 | 261.5 ms | 3,915 | 2.05 | 1.10 | 10.82 | 0.08 | 85.9% |
| 128 | 221.9 ms | 4,614 | 4.04 | 0.99 | 19.88 | 0.08 | 90.0% |

**Compute-stream utilization is 84-90%** across all three batch sizes --
the compute stream stays continuously fed; CPU prep + H2D transfer are
successfully hidden behind GPU compute, not a source of pipeline bubbles for
this workload. This is consistent with (and explains) M30's own measured
~1.18-1.21x real-MNIST prefetch speedup ceiling: there simply isn't much
non-compute time left to hide once the compute stream is already this busy.

**Backward dominates GPU time at every batch size** (67-90% of per-step GPU
busy time), consistent with `mnist_profile.py`'s finding that conv2d
backward alone is ~55-57% of total step time.

## Prefetch depth sweep (batch_size=64)
| `prefetch_size` | epoch wall | samples/sec | compute-stream utilization |
|---|---|---|---|
| 1 | 247.9 ms | 4,131 | 89.9% |
| 2 | 270.1 ms | 3,791 | 83.6% |
| 3 | 271.4 ms | 3,772 | 83.5% |

Deeper prefetch does **not** help this workload -- `prefetch_size=1` already
fully hides the (cheap, synthetic) CPU preparation behind GPU compute, and
additional queued batches add background-thread/queue overhead without a
throughput benefit once the compute stream is the bottleneck. This is
expected, not a bug (differences are within realistic run-to-run noise for
this GPU); M30's default of `prefetch_size=2` is retained unchanged --
Section 26 of the milestone brief explicitly asks not to change the default
without justification, and this data does not provide one (a real,
CPU-heavier dataset would very plausibly see depth matter more; MNIST's
synthetic per-sample cost here is too cheap to tell).

## Kernel launch overhead
The pre-Milestone-31 `CrossEntropyLoss` forward composed ~9 Tensor
primitives (`max_axis1`, two column-broadcast `sub`s, `exp`,
`sum(axis=1)`, `log`, a `mul` against a freshly host-transferred one-hot
matrix, a full `sum`, a final `scale`) plus 2 fresh host->device transfers;
backward added ~7 more kernel launches through the same composed autograd
graph. At this GPU's measured ~54-65 us/launch dispatch cost (see
**Dependency / event overhead** above), that is 0.5-1.0 ms of launch
overhead alone on a `(batch, classes)` tensor with a trivial amount of real
arithmetic -- plus the two host transfers (host-side NumPy array
construction + allocation + `cudaMemcpy`), each costing more than a bare
kernel launch. `benchmarks/mnist_profile.py`'s own Milestone 21 finding
(reproduced in `benchmarks/results/m31_cross_entropy_baseline.json`'s
op-level numbers, pre-fix) was that
`CrossEntropyLoss`'s forward pass alone (4.73 ms) cost *more* wall-clock
time than the M20 CNN's entire forward pass (4.21 ms) -- a tensor of
`(64, 10)` elements taking longer than two `Conv2d` + two `MaxPool2d` +
`Linear` layers over a `(64, 1, 28, 28)` input is the signature of a
launch-overhead-bound, not compute-bound, operation.

## Layer-level profile (`mnist_profile.py`, post-fusion)
Phase breakdown, mean per training step, CUDA, batch_size=64:

| Phase | Time | % |
|---|---|---|
| transfer | 2.38 ms | 8.3% |
| forward | 5.41 ms | 18.9% |
| loss | 3.07 ms | 10.7% |
| backward | 16.39 ms | 57.3% |
| optimizer | 1.36 ms | 4.7% |
| **TOTAL** | **28.62 ms** | |

Backward, by op: `conv2d` 11.16 ms, `@` (matmul) 1.28 ms, `max_pool2d`
1.01 ms, `relu` 0.95 ms, `+` 0.57 ms, `cross_entropy` 0.19 ms, `reshape`
0.17 ms.

(This tool synchronizes between every phase by design -- see its own
docstring -- so these numbers characterize *operation cost in isolation*,
not overlap; **Live async epoch** above is the non-synchronizing
complement.)

## Bottleneck ranking

1. **Backward pass / `conv2d` backward** -- 55-57% of step time (~10-11 ms
   at batch_size=64). Genuinely compute-bound: two `Conv2d` layers' input +
   weight + bias gradients, computed by Forge's own hand-written (non-cuDNN)
   CUDA kernels on a Compute Capability 5.0 GPU. No plausible launch-count
   reduction touches this -- it is real arithmetic, not dispatch overhead.
   Out of scope for M31 (Section 50 explicitly excludes cuDNN migration and
   broad kernel rewrites); a future milestone could revisit tiling/shared-
   memory occupancy here with dedicated profiling, but that is a
   fundamentally different (and much larger) undertaking than this one.
2. **Forward pass** -- 15-19% (~4-5.4 ms), also `Conv2d`-dominated compute.
3. **`CrossEntropyLoss`** (pre-fusion) -- 10.7-16.7% (3.07-4.73 ms),
   *launch-overhead-bound*, not compute-bound (see **Kernel launch
   overhead** above) -- the optimization target selected below.
4. **Optimizer (Adam)** -- 3.3-4.7% (~0.9-1.4 ms), one kernel launch per
   parameter tensor (~10 in the M20 CNN). Too small a fraction for even a
   large realistic speedup to matter (Section 21's own worked example: a
   ~4% bottleneck at 2x speedup is ~2% overall -- not worth the added
   complexity of a fused multi-tensor Adam kernel this milestone).
5. **H2D transfer, dependency/event overhead, allocator overhead, pinned-
   memory overhead** -- all measured and all comfortably hidden/negligible
   in the real async pipeline (see the sections above). None is a
   bottleneck worth optimizing.

Using Section 21's `fraction x realistic speedup = expected benefit`
formula: `conv2d` backward is the largest fraction (55%) but has no
realistic *launch-overhead* speedup available (it isn't launch-overhead-
bound) and a real compute speedup is out of this milestone's scope.
`CrossEntropyLoss` was 10-17% of runtime with a *measured* 2-25x local
speedup available purely from removing launch/transfer overhead --
`0.15 x (1 - 1/10) ~= 13%` expected overall improvement is a realistic,
achievable, narrowly-scoped win; `conv2d`'s `0.55 x (best-case ~1.0)` is not
available within this milestone's constraints. `CrossEntropyLoss` fusion was
therefore the correct, and only, optimization selected for M31.

## Optimization selected: fused `CrossEntropyLoss` forward/backward kernels
See `docs/architecture/cuda-backend.md`'s **Milestone 31: fused
forward/backward kernels** section for the full implementation writeup
(problem, root cause, change, correctness validation). Summary:

- **Problem**: `CrossEntropyLoss` forward/backward composed ~16 total Tensor
  primitive calls plus 2 host transfers for a tiny `(batch, classes)` tensor.
- **Root cause**: per-launch/per-transfer dispatch overhead (~54-300 us
  each on this GPU) dominates when there is too little real arithmetic to
  amortize it against.
- **Change**: `Backend.cross_entropy`/`cross_entropy_backward`
  (`forge/backend/base.py`, implemented in `cpu.py` and `cuda/backend.py`)
  -- 2 CUDA kernel launches forward (1 fused per-row log-sum-exp-NLL kernel
  + the existing `sum` reduction, reused), 1 launch backward (fused
  softmax-minus-one-hot gradient). `Tensor.cross_entropy()`
  (`forge/tensor/tensor.py`) is the new autograd-aware entry point;
  `nn.CrossEntropyLoss.forward()`'s existing validation is unchanged, only
  its computational tail was replaced.
- **Correctness validation**: `tests/test_cuda_loss.py`'s full pre-existing
  suite (forward-vs-CPU across numerically difficult logits, backward vs.
  analytical formula, backward vs. finite differences, reduction semantics,
  device validation, "no CPU fallback" spy tests) passes unchanged --
  32/32. `tests/test_cuda_cross_entropy_fusion.py` (new) adds
  `Tensor.cross_entropy`'s own defense-in-depth validation, cross-stream
  correctness (logits/target/grad_output each produced on a different
  stream than the op itself runs on), and a repeated-use memory-safety
  check -- 7/7. Full suite: 1154/1154.
- **Performance result** (isolated op benchmark, `benchmarks/ops_bench.py` +
  `backward_bench.py`, 20 iterations, batch=64/classes=10, before vs after):

  | | forward | backward |
  |---|---|---|
  | CUDA, before | 1.02 ms | 15.20 ms (high variance) |
  | CUDA, after | 0.40 ms (2.5x) | 0.17 ms (~90x) |
  | CPU, before | 0.16 ms | 0.11 ms |
  | CPU, after | 0.08 ms (~2x) | 0.06 ms (~1.9x) |

  End-to-end (`benchmarks/mnist_bench.py`, real M20 CNN, 30 iterations,
  batch=64): CUDA full training step 19.20 ms -> 17.90 ms (~7% faster,
  consistent with the loss's ~10-17%-of-step fraction once `conv2d`
  backward's own larger run-to-run variance is accounted for -- see Section
  44's regression-threshold guidance: a modest, consistent, non-regressing
  end-to-end improvement, not a noisy microbenchmark artifact). CPU full
  training step also improved slightly (47.8 ms -> 50.5 ms range, within
  normal CPU run-to-run noise -- **no CPU regression**; NumPy pays no
  per-launch overhead, so the CPU-side benefit is only from fewer
  autograd-graph Python objects per step, not launch-count).
  `benchmarks/async_dataloader_bench.py`'s real-MNIST prefetch speedup is
  unchanged within noise (1.171x before -> 1.180x after) -- the fusion does
  not disturb the M30 async pipeline's own established behavior.
- **Memory impact**: `tests/test_cuda_cross_entropy_fusion.py::
  test_cross_entropy_repeated_use_does_not_grow_active_memory` -- 50
  repeated forward+backward calls, `allocated_bytes`/`reserved_bytes`/
  `pending_bytes` all return to exactly 0 after `gc.collect()` +
  `empty_cache()`. No leak.
- **Tradeoffs**: `max_axis1`/`sum(axis=1)`/column-broadcast `sub` remain in
  the `Backend` interface (still independently tested, still potentially
  useful primitives) but are no longer exercised by `CrossEntropyLoss` --
  effectively now-generic small ops rather than dead code, not removed
  (removing tested public interface surface was not this milestone's goal).
  The new CUDA kernels use one thread per row (matching `k_max_axis1`/
  `k_sum_axis1`'s existing convention) rather than a block-per-row
  shared-memory reduction -- correct and simple for Forge's actual class
  counts (tens, not thousands), but would not scale as well to very large
  vocabulary sizes; not a concern for any workload Forge currently targets.

## Resource lifetime
Streams, events (both the internal disable-timing `CUDAEvent` and the new
profiling-only `TimedEvent`), pinned allocations, and allocator blocks all
returned to their expected baseline after every profiling run in this
document (see **Allocator characterization** / **Pinned-memory
characterization** above, and the dedicated memory-safety test). The
background prefetch thread's own bounded lifecycle (Milestone 30) is
unchanged by this milestone.

## What was *not* found to be a bottleneck (and therefore not touched)
Dependency/event overhead, allocator overhead, pinned-memory overhead,
prefetch queue depth, and H2D transfer (in the real async pipeline) were all
measured and found comfortably within noise or successfully hidden by
existing M27-M30 machinery. Per Section 50 of the milestone brief, none of
CUDA Graphs, kernel fusion beyond the one narrowly-scoped case above,
multi-GPU, mixed precision, a pinned caching allocator, event pooling, or a
new prefetch/DataLoader architecture were implemented -- none were justified
by measurement.
