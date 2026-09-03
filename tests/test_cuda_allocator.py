"""Milestone 25 tests: the exact-size CUDA caching allocator.

Requires an actual working CUDA backend; skipped cleanly otherwise via the
module-level `pytestmark`, matching every other `test_cuda_*.py` file. See
`docs/architecture/cuda-memory-allocator.md` for the design these tests
verify, and `tests/test_cuda_memory.py`/`tests/test_cuda_lifetime.py` for the
pre-existing Milestone 22/23 lifecycle tests this allocator must not break
(all pass unchanged against this module, aside from two `allocation_count`/
`free_count` assertions updated for the new "real driver call only" meaning
-- see that file's `_empty_cache_between_tests` fixture).

Most tests here exercise `forge.backend.cuda.allocator` directly (`allocate`/
`release`/`empty_cache`, plus the module-level singleton via `get_allocator`)
rather than only through `Tensor`/`CUDAStorage`, so pointer reuse and cache
bookkeeping can be asserted precisely -- per the milestone brief's own
caution, pointer-equality assertions are used only where the allocator's own
cache-hit accounting is asserted alongside them, not as the sole evidence.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import allocator as cuda_allocator
from forge.backend.cuda.backend import get_cuda_backend, is_cuda_available

_REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


@pytest.fixture(autouse=True)
def _clean_cache():
    """Every test in this module wants a known-empty starting cache -- see
    `tests/test_cuda_memory.py`'s identical fixture for why this must run
    both before (isolate from earlier tests/modules) and after (isolate
    later ones)."""
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


def _lib():
    return get_cuda_backend()._lib


# -- 1. Cache hit reuses the exact pointer ----------------------------------------


def test_allocate_then_release_then_allocate_same_size_reuses_pointer():
    lib = _lib()
    before = cuda_allocator.memory_stats()

    ptr_a = cuda_allocator.allocate(lib, 4096)
    after_alloc = cuda_allocator.memory_stats()
    assert after_alloc.cache_miss_count - before.cache_miss_count == 1
    assert after_alloc.allocated_bytes - before.allocated_bytes == 4096

    cuda_allocator.release(4096, ptr_a)
    after_release = cuda_allocator.memory_stats()
    assert after_release.allocated_bytes == before.allocated_bytes
    assert after_release.cached_bytes - before.cached_bytes == 4096

    ptr_b = cuda_allocator.allocate(lib, 4096)
    after_hit = cuda_allocator.memory_stats()
    assert after_hit.cache_hit_count - before.cache_hit_count == 1
    assert after_hit.cache_miss_count == after_alloc.cache_miss_count  # no new driver call
    assert ptr_b.value == ptr_a.value  # the actual pointer was reused, not just "a" pointer

    cuda_allocator.release(4096, ptr_b)


# -- 2. Exact-size only: no split, no reuse across sizes ---------------------------


def test_different_size_request_is_a_cache_miss_not_a_split():
    lib = _lib()
    ptr_a = cuda_allocator.allocate(lib, 4096)
    cuda_allocator.release(4096, ptr_a)

    before = cuda_allocator.memory_stats()
    ptr_b = cuda_allocator.allocate(lib, 2048)  # smaller -- must NOT reuse/split the 4096 block
    after = cuda_allocator.memory_stats()

    assert after.cache_miss_count - before.cache_miss_count == 1
    assert after.cache_hit_count == before.cache_hit_count
    assert ptr_b.value != ptr_a.value
    assert after.cached_bytes == before.cached_bytes == 4096  # the 4096 block is still cached, untouched

    cuda_allocator.release(2048, ptr_b)


def test_distinct_sizes_all_remain_cached_independently_no_coalescing():
    """Documents the M25 fragmentation limitation (Section 14 of the milestone
    brief): cached blocks of different sizes are never merged/coalesced --
    a request that could in principle be satisfied by combining two smaller
    cached blocks always falls through to a fresh `cudaMalloc` instead."""
    lib = _lib()
    sizes = (1_048_576, 2_097_152, 4_194_304)  # 1 MiB, 2 MiB, 4 MiB
    ptrs = [cuda_allocator.allocate(lib, n) for n in sizes]
    for n, p in zip(sizes, ptrs):
        cuda_allocator.release(n, p)

    before = cuda_allocator.memory_stats()
    assert before.cached_bytes == sum(sizes)

    # A 3 MiB request could in principle be served by combining the 1 MiB and
    # 2 MiB blocks -- the exact-size allocator does not attempt this.
    ptr_new = cuda_allocator.allocate(lib, 3_145_728)
    after = cuda_allocator.memory_stats()
    assert after.cache_miss_count - before.cache_miss_count == 1
    assert after.cached_bytes == sum(sizes)  # the three original blocks are still exactly as cached
    cuda_allocator.release(3_145_728, ptr_new)


# -- 3. empty_cache() ---------------------------------------------------------------


def test_empty_cache_frees_cached_blocks_and_returns_count():
    lib = _lib()
    ptr_a = cuda_allocator.allocate(lib, 8192)
    ptr_b = cuda_allocator.allocate(lib, 16384)
    cuda_allocator.release(8192, ptr_a)
    cuda_allocator.release(16384, ptr_b)

    before = cuda_allocator.memory_stats()
    assert before.cached_bytes == 8192 + 16384

    freed = forge.cuda.empty_cache()
    after = forge.cuda.memory_stats()
    assert freed == 2
    assert after.cached_bytes == 0
    assert after.reserved_bytes == after.allocated_bytes
    assert after.free_count - before.free_count == 2


def test_empty_cache_never_touches_active_tensors():
    keep = Tensor(np.arange(100, dtype=np.float32), device="cuda")
    temp = Tensor(np.arange(200, dtype=np.float32), device="cuda")
    del temp

    before = forge.cuda.memory_stats()
    assert before.cached_bytes > 0  # `temp`'s block is cached

    forge.cuda.empty_cache()
    after = forge.cuda.memory_stats()
    assert after.cached_bytes == 0
    assert after.allocated_bytes == before.allocated_bytes  # `keep` untouched

    # `keep` must still be fully usable after the purge.
    np.testing.assert_array_equal(keep.to("cpu").numpy(), np.arange(100, dtype=np.float32))
    del keep


def test_empty_cache_on_a_fresh_cache_is_a_safe_no_op():
    forge.cuda.empty_cache()  # already emptied by the autouse fixture
    freed = forge.cuda.empty_cache()
    assert freed == 0


# -- 4. Ownership invariant: never active and cached simultaneously ------------------


def test_allocated_and_cached_bytes_never_double_count_the_same_block():
    lib = _lib()
    ptr = cuda_allocator.allocate(lib, 4096)
    mid = cuda_allocator.memory_stats()
    assert mid.allocated_bytes >= 4096
    assert mid.cached_bytes == 0  # nothing cached yet -- the block is only active

    cuda_allocator.release(4096, ptr)
    after = cuda_allocator.memory_stats()
    assert after.cached_bytes >= 4096
    # The just-released block no longer contributes to `allocated_bytes`.
    assert after.allocated_bytes == mid.allocated_bytes - 4096


# -- 5. Double-release protection --------------------------------------------------


def test_releasing_the_same_pointer_twice_raises_instead_of_corrupting_cache():
    lib = _lib()
    ptr = cuda_allocator.allocate(lib, 4096)
    cuda_allocator.release(4096, ptr)
    with pytest.raises(RuntimeError):
        cuda_allocator.release(4096, ptr)  # the same block, released a second time -- an internal bug


def test_cuda_storage_del_guards_against_double_free():
    """`CUDAStorage.__del__` clears `self.ptr` before handing the pointer to
    the allocator -- calling `__del__` a second time on the same object
    (never happens under normal refcounting, but is a cheap, explicit
    invariant to hold) must be a safe no-op, not a double-release."""
    t = Tensor(np.zeros(10, dtype=np.float32), device="cuda")
    storage = t._data
    del t
    storage.__del__()  # explicit second call -- must not raise or double-count


# -- 6. Repeated same-shape workload reaches a high cache-hit steady state -----------


def test_repeated_same_shape_allocation_reaches_high_hit_rate():
    lib = _lib()
    nbytes = 65536
    iterations = 50

    for _ in range(5):  # warmup: primes the cache with one block of this size
        ptr = cuda_allocator.allocate(lib, nbytes)
        cuda_allocator.release(nbytes, ptr)

    before = cuda_allocator.memory_stats()
    for _ in range(iterations):
        ptr = cuda_allocator.allocate(lib, nbytes)
        cuda_allocator.release(nbytes, ptr)
    after = cuda_allocator.memory_stats()

    assert after.cache_miss_count == before.cache_miss_count  # steady state: zero new driver calls
    assert after.cache_hit_count - before.cache_hit_count == iterations


def test_repeated_tensor_creation_of_same_shape_stabilizes_driver_allocations():
    shape_bytes = 32 * 4  # np.float32, 32 elements

    for _ in range(5):
        t = Tensor(np.zeros(32, dtype=np.float32), device="cuda")
        del t
    before = forge.cuda.memory_stats()

    for _ in range(20):
        t = Tensor(np.zeros(32, dtype=np.float32), device="cuda")
        del t
    after = forge.cuda.memory_stats()

    assert after.allocation_count == before.allocation_count  # no new real cudaMalloc calls
    assert after.cache_hit_count - before.cache_hit_count == 20
    assert after.cached_bytes >= shape_bytes


# -- 7. Correctness: a reused cached block never leaks its previous occupant's data --


def test_reused_block_contains_only_the_new_tensor_data_not_stale_data():
    a = Tensor(np.full(1000, 7.0, dtype=np.float32), device="cuda")
    del a  # the block enters the cache still holding 7.0s

    b = Tensor(np.full(1000, 3.0, dtype=np.float32), device="cuda")  # same nbytes -- likely a cache hit
    np.testing.assert_array_equal(b.to("cpu").numpy(), np.full(1000, 3.0, dtype=np.float32))
    del b


# -- 8. No CPU fallback --------------------------------------------------------------


def test_cache_hit_path_never_touches_cpu_backend(monkeypatch):
    """A cache hit must be served entirely from `forge.backend.cuda.allocator`'s
    own bookkeeping -- monkeypatching CPUBackend's construction to explode
    proves no CPU code path is reachable from it."""
    from forge.backend.cpu import CPUBackend

    def _explode(*args, **kwargs):
        raise AssertionError("CPUBackend must never be invoked by a CUDA cache hit")

    a = Tensor(np.zeros(500, dtype=np.float32), device="cuda")
    del a  # primes the cache

    monkeypatch.setattr(CPUBackend, "from_array", _explode)
    b = Tensor(np.zeros(500, dtype=np.float32), device="cuda")  # must be servable as a cache hit
    assert b.device.type == "cuda"
    del b


def test_cached_and_reused_pointers_are_real_device_pointers():
    lib = _lib()
    ptr_a = cuda_allocator.allocate(lib, 4096)
    cuda_allocator.release(4096, ptr_a)
    ptr_b = cuda_allocator.allocate(lib, 4096)
    assert ptr_b.value is not None and ptr_b.value != 0
    cuda_allocator.release(4096, ptr_b)


# -- 9. Persistence / checkpoint interaction ------------------------------------------


def test_empty_cache_does_not_disturb_a_loaded_checkpoint(tmp_path):
    from forge.nn import Linear
    from forge.optim import Adam
    from forge.serialization import load_checkpoint, save_checkpoint

    forge.random.seed(0)
    model = Linear(4, 3).to("cuda")
    optimizer = Adam(model.parameters(), lr=0.01)
    x = Tensor(np.random.default_rng(0).standard_normal((5, 4)).astype(np.float32), device="cuda")
    for _ in range(2):
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()

    path = tmp_path / "m25.ckpt"
    save_checkpoint(str(path), model, optimizer, epoch=0, global_step=2)

    forge.cuda.empty_cache()
    checkpoint = load_checkpoint(str(path))
    forge.cuda.empty_cache()

    out = checkpoint.model(x)
    assert out.shape == (5, 3)
    assert np.isfinite(out.to("cpu").numpy()).all()


# -- 10. OOM handling (isolated subprocess -- see M22's process-poisoning finding) ----


def test_oom_purges_cache_before_final_failure():
    """Runs in a **subprocess**, deliberately -- mirrors
    `tests/test_cuda_memory.py::test_failed_allocation_does_not_corrupt_statistics`'s
    isolation rationale exactly: a `cudaMalloc` request large enough to fail
    can poison the CUDA context for the rest of that process on this
    hardware (940MX, driver 582.53), so this provokes it in a throwaway
    child process instead of the shared `pytest` process.
    """
    script = """
import numpy as np
import forge
from forge import Tensor
from forge.exceptions import CUDAError
from forge.backend.cuda.backend import get_cuda_backend

t = Tensor(np.zeros(10_000, dtype=np.float32), device="cuda")
del t  # a cached block now sits in the allocator
before = forge.cuda.memory_stats()
assert before.cached_bytes > 0, "expected a cached block before the OOM attempt"

backend = get_cuda_backend()
try:
    backend._alloc(2**34)  # 16 GiB -- far beyond the 940MX's 2 GiB VRAM
    raise SystemExit("expected CUDAError was not raised")
except CUDAError:
    pass

after = forge.cuda.memory_stats()
# The M24-designed OOM policy purges the cache before its final retry/raise
# (see forge/backend/cuda/allocator.py's `_driver_malloc`) -- so by the time
# CUDAError propagates, the cache that was populated above must be empty.
assert after.cached_bytes == 0, f"cache was not purged by OOM handling: {after}"
assert after.allocated_bytes == before.allocated_bytes, "OOM must never touch active memory"
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=str(_REPO_ROOT), timeout=60
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "OK" in result.stdout
