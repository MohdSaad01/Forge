"""Milestone 51 tests: `CUDACachingAllocator` reentrant-release safety.

M50's full 1,619-test single-process suite run hit a real, hardware-confirmed
self-deadlock in `CUDACachingAllocator.release()` (see
`docs/development/m51-allocator-reentrancy.md` for the full investigation):
CPython's cyclic GC can synchronously finalize an unrelated, reference-cycle-
trapped `CUDAStorage` while `release()`/`release_pending()` still hold
`self._lock` for an *earlier* `CUDAStorage.__del__` call -- and that
finalization calls back into the same allocator, on the same thread, via the
same lock. Against the original plain `threading.Lock` this deadlocks
outright.

Every reentrancy-triggering test here runs in a **subprocess with a
timeout**, matching the isolation rationale `tests/test_cuda_allocator.py`
and `tests/test_cuda_pinned_memory.py` already use for other hazard-adjacent
tests: a regression here should show up as a clean subprocess failure/
timeout, never as a hung `pytest` process. Each subprocess script wraps the
*real* allocator's real `self._lock` in `_GCTrippingLock`, which forces a
`gc.collect()` immediately after every acquire -- this fixes *when* a GC
pass happens (deterministic, not a probabilistic wait on natural GC
thresholds) without changing the mechanism at all: it still exercises the
real, unmodified `CUDACachingAllocator.release()`/`release_pending()`,
the real lock object currently installed on the process-wide allocator, and
real `CUDAStorage`/`Tensor` objects on real hardware.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import allocator as cuda_allocator
from forge.backend.cuda.backend import is_cuda_available

_REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


@pytest.fixture(autouse=True)
def _clean_cache():
    """Every test wants a known-empty starting cache; matches sibling allocator test files."""
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


def _run_subprocess(script: str, timeout: float = 30.0) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=str(_REPO_ROOT), timeout=timeout
    )


_HARNESS = '''
import gc
import forge
from forge import Tensor
from forge.backend.cuda import allocator as cuda_allocator


class _GCTrippingLock:
    """Wraps the allocator's real lock; every acquire() runs gc.collect()
    immediately after acquiring -- forces the M50/M51 hazard (a GC pass
    finalizing unrelated cyclic-garbage CUDAStorage while the allocator lock
    is held) to fire deterministically. Only *when* a collection happens is
    forced; release()/release_pending() below are real, unmodified code."""

    def __init__(self, real_lock):
        self._real = real_lock

    def acquire(self, *a, **kw):
        r = self._real.acquire(*a, **kw)
        gc.collect()
        return r

    def release(self, *a, **kw):
        return self._real.release(*a, **kw)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


class _CycleHolder:
    __slots__ = ("self_ref", "tensor")


def trap_tensor_in_cycle(tensor):
    """Puts `tensor` into an unreachable reference cycle -- refcounting alone
    cannot free it; only a gc.collect() pass can, matching the real M50
    failure's autograd-graph-teardown shape more closely than a plain
    `del`."""
    holder = _CycleHolder()
    holder.self_ref = holder
    holder.tensor = tensor
    del holder


def install_gc_tripping_lock():
    allocator = cuda_allocator.get_allocator()
    allocator._lock = _GCTrippingLock(allocator._lock)
    return allocator
'''


# -- 1. The canonical regression reproducer: default-stream release() ------------


def test_release_reentrant_gc_finalization_does_not_deadlock():
    """Fails (times out) against the pre-M51 `threading.Lock`; completes cleanly
    against the M51 `threading.RLock` + allocation-free critical section."""
    script = _HARNESS + """
import numpy as np

forge.cuda.empty_cache()
install_gc_tripping_lock()

trap_tensor_in_cycle(Tensor(np.ones((4,), dtype=np.float32), device="cuda"))

victim = Tensor(np.ones((4,), dtype=np.float32), device="cuda")
del victim  # triggers release() -> gc.collect() while locked -> reentrant release()

print("OK")
"""
    result = _run_subprocess(script)
    assert result.returncode == 0, f"stdout:\\n{result.stdout}\\nstderr:\\n{result.stderr}"
    assert "OK" in result.stdout


# -- 2. The same hazard through release_pending() (explicit stream) --------------


def test_release_pending_reentrant_gc_finalization_does_not_deadlock():
    """Same hazard, but through `release_pending()` -- covers explicit-stream
    behavior specifically, since a default-stream release() never reaches it."""
    script = _HARNESS + """
import numpy as np

forge.cuda.empty_cache()
install_gc_tripping_lock()

s_a = forge.cuda.Stream()
with forge.cuda.stream(s_a):
    # Constructed and handed straight to trap_tensor_in_cycle with no outer
    # name bound to it -- an intermediate `trapped = Tensor(...)` variable
    # would itself keep it reachable by plain refcounting, defeating the
    # need for a gc.collect() pass to free it at all.
    trap_tensor_in_cycle(Tensor(np.ones((4,), dtype=np.float32), device="cuda"))

s_b = forge.cuda.Stream()
with forge.cuda.stream(s_b):
    victim = Tensor(np.ones((4,), dtype=np.float32), device="cuda")
del victim  # triggers release_pending() -> gc.collect() while locked -> reentrant call

s_a.synchronize()
s_b.synchronize()
print("OK")
"""
    result = _run_subprocess(script)
    assert result.returncode == 0, f"stdout:\\n{result.stdout}\\nstderr:\\n{result.stderr}"
    assert "OK" in result.stdout


# -- 3. Stability under repetition (stress) ---------------------------------------


def test_reentrant_scenario_stable_under_repeated_cycles():
    """The exact reproducer, repeated 25x in one process/one allocator instance --
    acceptance criterion 2 ("passes repeatedly under stress")."""
    script = _HARNESS + """
import numpy as np

forge.cuda.empty_cache()
install_gc_tripping_lock()

for i in range(25):
    trap_tensor_in_cycle(Tensor(np.ones((4,), dtype=np.float32), device="cuda"))
    victim = Tensor(np.ones((4,), dtype=np.float32), device="cuda")
    del victim

print("OK", forge.cuda.memory_stats().reserved_bytes)
"""
    result = _run_subprocess(script, timeout=60)
    assert result.returncode == 0, f"stdout:\\n{result.stdout}\\nstderr:\\n{result.stderr}"
    assert "OK" in result.stdout


# -- 4. The allocator's lock is genuinely reentrant on the same thread -----------


def test_allocator_lock_is_reentrant_same_thread():
    """Direct sanity check on the production lock object.

    Uses a *bounded* `lock.acquire(timeout=...)` for the inner (reentrant)
    acquire rather than a plain `with lock:` -- against a non-reentrant
    `threading.Lock`, a plain nested `with lock:` would block the worker
    thread forever, and since this test deliberately uses the *real*,
    shared, process-wide allocator lock (not a private copy), a permanently
    stuck thread would keep holding it and silently deadlock every other
    test that touches the allocator afterward. The bounded acquire lets the
    worker always terminate cleanly, and still correctly distinguishes
    reentrant (`RLock`, succeeds instantly) from non-reentrant (`Lock`,
    times out) behavior.
    """
    lock = cuda_allocator.get_allocator()._lock
    result: dict = {}

    def worker():
        outer_acquired = lock.acquire(timeout=5)
        try:
            result["outer_acquired"] = outer_acquired
            if outer_acquired:
                inner_acquired = lock.acquire(timeout=5)  # reentrant attempt, bounded
                result["inner_acquired"] = inner_acquired
                if inner_acquired:
                    lock.release()
        finally:
            if outer_acquired:
                lock.release()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=15)
    assert not t.is_alive(), "worker thread did not terminate within its own bounded acquire timeouts"
    assert result.get("outer_acquired") is True
    assert result.get("inner_acquired") is True, "reentrant acquire on CUDACachingAllocator._lock blocked -- lock regressed to non-reentrant"


# -- 5. The duplicate-release detector still works (manual while-loop rewrite) ---


def test_double_release_still_raises_runtime_error():
    """`release()`'s duplicate-pointer scan was rewritten from `any()`/genexpr to
    a manual index `while` loop (M51, to avoid allocating a GC-tracked
    iterator while the lock is held) -- confirm it still detects a genuine
    double-release exactly as before."""
    from forge.backend.cuda.backend import get_cuda_backend

    backend_lib = get_cuda_backend()._lib
    ptr = cuda_allocator.allocate(backend_lib, 4096)
    cuda_allocator.release(4096, ptr)
    with pytest.raises(RuntimeError, match="double-release"):
        cuda_allocator.release(4096, ptr)


# -- 6. Repeated alloc/release cycles: reuse and accounting stay correct --------


def test_repeated_alloc_release_cycles_reuse_and_accounting_correct():
    """Covers "repeated allocation/free cycles", "allocator reuse", and
    "reserved vs allocated memory accounting" from the M51 regression-test
    checklist -- ordinary (non-adversarial) churn through the rewritten
    `release()`."""
    from forge.backend.cuda.backend import get_cuda_backend

    lib = get_cuda_backend()._lib
    before = cuda_allocator.memory_stats()

    ptrs = []
    for _ in range(50):
        ptrs.append(cuda_allocator.allocate(lib, 2048))
    for ptr in ptrs:
        cuda_allocator.release(2048, ptr)

    after_first_pass = cuda_allocator.memory_stats()
    assert after_first_pass.allocated_bytes == before.allocated_bytes
    assert after_first_pass.cached_bytes - before.cached_bytes == 50 * 2048

    hits_before = after_first_pass.cache_hit_count
    for _ in range(50):
        ptr = cuda_allocator.allocate(lib, 2048)
        cuda_allocator.release(2048, ptr)
    after_second_pass = cuda_allocator.memory_stats()

    assert after_second_pass.cache_hit_count - hits_before == 50
    assert after_second_pass.allocated_bytes == before.allocated_bytes
    assert after_second_pass.reserved_bytes == after_first_pass.reserved_bytes  # no growth from reuse


# -- 7. Explicit gc.collect() during ordinary allocator activity stays correct --


def test_explicit_gc_collect_during_release_does_not_corrupt_state():
    """Not adversarial -- just confirms a plain `gc.collect()` call sitting in
    the middle of normal Tensor churn (as any real training loop might
    trigger) never corrupts allocator bookkeeping, independent of the
    reentrancy-specific tests above."""
    import gc

    before = cuda_allocator.memory_stats()
    tensors = [Tensor(np.ones((16,), dtype=np.float32), device="cuda") for _ in range(10)]
    gc.collect()
    del tensors
    gc.collect()

    after = cuda_allocator.memory_stats()
    assert after.allocated_bytes == before.allocated_bytes
    assert after.cached_bytes - before.cached_bytes == 16 * 4 * 10


# -- 8. Cross-stream release/reclaim remains correct post-fix --------------------


def test_cross_stream_release_pending_reclaim_still_correct():
    """Light confirmation that the M51 rewrite of `release_pending()` (moving
    the `_PendingBlock` construction before the lock) did not change
    stream-ordering semantics; `tests/test_cuda_stream_allocator.py` is the
    exhaustive version of this coverage."""
    shape = (2048,)
    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        a = Tensor(np.full(shape, 5.0, dtype=np.float32), device="cuda")
    del a
    s.synchronize()

    stats_before = cuda_allocator.memory_stats()
    with forge.cuda.stream(s):
        b = Tensor(np.full(shape, 7.0, dtype=np.float32), device="cuda")
        readback = b.to("cpu")
    stats_after = cuda_allocator.memory_stats()

    np.testing.assert_array_equal(readback.numpy(), np.full(shape, 7.0, dtype=np.float32))
    assert stats_after.cache_hit_count >= stats_before.cache_hit_count
