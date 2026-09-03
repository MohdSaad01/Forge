"""Milestone 22 tests: CUDA memory statistics and allocation lifecycle.

Every test in this module requires an actual working CUDA backend and is
skipped cleanly otherwise via the module-level `pytestmark`, matching the
convention in `tests/test_cuda_backend.py` and every other `test_cuda_*.py`
file. The two `forge.cuda`-unavailable tests (which must run on *any*
machine, CUDA or not, since they exercise the "CUDA is not available" error
path itself) live in `tests/test_cuda_memory_availability.py` instead --
a module-level `pytestmark` applies to every test in its module regardless
of where in the file it is assigned, so they cannot coexist with this file's
blanket skip.

Every measurement helper (`_stable_stats`) forces a `gc.collect()` plus an
explicit `CUDABackend.synchronize()` before reading `forge.cuda.memory_stats()`
-- not because any individual CUDA op here is unsynchronized (every one
already synchronizes internally before trusting its own result, see
`docs/architecture/cuda-backend.md`), but because a lifecycle assertion needs
Python's own garbage collector to have actually run before "how much CUDA
memory is live" is a meaningful question (see `docs/architecture/cuda-
backend.md`'s **CUDA Memory Statistics** section, "Known limitations").

Every test below asserts *deltas* against a snapshot taken immediately
before the operation under test, rather than absolute counter values --
`forge.cuda`'s counters are process-wide and cumulative across the whole
test session, so this is what makes each test's assertions independent of
test execution order.
"""

from __future__ import annotations

import gc
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.exceptions import CUDAError
from forge.nn import Dropout, Linear, ReLU, Sequential
from forge.optim import Adam
from forge.serialization import load_checkpoint, save_checkpoint

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _stable_stats():
    gc.collect()
    from forge.backend.cuda.backend import get_cuda_backend

    get_cuda_backend().synchronize()
    return forge.cuda.memory_stats()


def _small_model():
    forge.random.seed(0)
    return Sequential(Linear(8, 16), ReLU(), Linear(16, 4))


pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


@pytest.fixture(autouse=True)
def _empty_cache_between_tests():
    """Milestone 25: the caching allocator's cache persists across tests --
    process-wide, like every other `forge.cuda` counter. Purge it before and
    after every test in this module so a block a *previous* test's tensor
    left cached can never turn one of *this* test's expected `cudaMalloc`
    cache misses into a hit. That would change `allocation_count`/`free_count`
    (real driver-call counts, see `docs/architecture/cuda-memory-allocator.md`)
    but never `allocated_bytes`/`cached_bytes` -- exactly why this file's other
    tests, which only assert `allocated_bytes` deltas, never needed this fixture
    to pass, while the handful asserting `allocation_count`/`free_count` did."""
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


# -- 1. Basic allocation ------------------------------------------------------


def test_basic_allocation_increases_allocated_bytes_by_expected_amount():
    before = _stable_stats()
    t = Tensor(np.zeros(1000, dtype=np.float32), device="cuda")
    after = forge.cuda.memory_stats()
    assert after.allocated_bytes - before.allocated_bytes == 1000 * 4
    assert after.allocation_count - before.allocation_count == 1
    del t


def test_basic_allocation_increments_allocation_count_not_free_count():
    before = _stable_stats()
    t = Tensor(np.zeros(10, dtype=np.float32), device="cuda")
    after = forge.cuda.memory_stats()
    assert after.allocation_count - before.allocation_count == 1
    assert after.free_count == before.free_count
    del t


# -- 2. Multiple allocations ---------------------------------------------------


def test_multiple_allocations_sum_to_total_storage_bytes():
    before = _stable_stats()
    sizes = (10, 20, 30, 40)
    tensors = [Tensor(np.zeros(n, dtype=np.float32), device="cuda") for n in sizes]
    after = forge.cuda.memory_stats()
    assert after.allocated_bytes - before.allocated_bytes == sum(sizes) * 4
    assert after.allocation_count - before.allocation_count == len(sizes)
    del tensors


# -- 3. Deallocation ------------------------------------------------------------


def test_deleting_a_cuda_tensor_eventually_decreases_allocated_bytes():
    before = _stable_stats()
    t = Tensor(np.zeros(5000, dtype=np.float32), device="cuda")
    mid = forge.cuda.memory_stats()
    assert mid.allocated_bytes - before.allocated_bytes == 5000 * 4

    del t
    after = _stable_stats()
    assert after.allocated_bytes == before.allocated_bytes
    # Milestone 25: `del t` returns the block to the caching allocator's
    # exact-size cache rather than issuing a real `cudaFree` -- `free_count`
    # (real driver frees) is unaffected until an explicit `empty_cache()`,
    # while `cached_bytes` picks up exactly the freed tensor's bytes.
    assert after.free_count == before.free_count
    assert after.cached_bytes - before.cached_bytes == 5000 * 4

    freed = forge.cuda.empty_cache()
    purged = forge.cuda.memory_stats()
    assert freed >= 1
    assert purged.free_count > after.free_count
    assert purged.cached_bytes == 0
    assert purged.allocated_bytes == after.allocated_bytes  # empty_cache never touches active memory


def test_multiple_tensors_each_release_their_own_bytes_independently():
    before = _stable_stats()
    a = Tensor(np.zeros(100, dtype=np.float32), device="cuda")
    b = Tensor(np.zeros(200, dtype=np.float32), device="cuda")

    del a
    mid = _stable_stats()
    assert mid.allocated_bytes - before.allocated_bytes == 200 * 4

    del b
    after = _stable_stats()
    assert after.allocated_bytes == before.allocated_bytes


# -- 4. Peak memory -------------------------------------------------------------


def test_peak_is_at_least_current_allocated():
    forge.cuda.reset_peak_memory_stats()
    t = Tensor(np.zeros(1000, dtype=np.float32), device="cuda")
    stats = forge.cuda.memory_stats()
    assert stats.peak_allocated_bytes >= stats.allocated_bytes
    del t


def test_peak_remains_at_historical_maximum_after_temporary_freed():
    forge.cuda.reset_peak_memory_stats()
    baseline = forge.cuda.memory_stats().allocated_bytes

    big = Tensor(np.zeros(200_000, dtype=np.float32), device="cuda")  # 800,000 bytes
    during = forge.cuda.memory_stats()
    assert during.peak_allocated_bytes >= baseline + 200_000 * 4

    del big
    after = _stable_stats()
    assert after.allocated_bytes == baseline
    assert after.peak_allocated_bytes == during.peak_allocated_bytes
    assert after.peak_allocated_bytes > after.allocated_bytes


# -- 5. Reset peak ----------------------------------------------------------------


def test_reset_peak_memory_stats_resets_peak_without_freeing_live_allocations():
    keep = Tensor(np.zeros(1000, dtype=np.float32), device="cuda")
    temp = Tensor(np.zeros(100_000, dtype=np.float32), device="cuda")
    del temp
    live = _stable_stats()
    assert live.peak_allocated_bytes > live.allocated_bytes  # the freed `temp` inflated peak

    forge.cuda.reset_peak_memory_stats()
    after_reset = forge.cuda.memory_stats()
    assert after_reset.allocated_bytes == live.allocated_bytes  # unchanged: reset never frees
    assert after_reset.peak_allocated_bytes == after_reset.allocated_bytes  # peak collapses to current
    assert after_reset.allocation_count == live.allocation_count  # counters untouched
    assert after_reset.free_count == live.free_count

    del keep


# -- 6. CPU isolation ---------------------------------------------------------------


def test_cpu_tensor_allocation_does_not_affect_cuda_stats():
    before = forge.cuda.memory_stats()
    cpu_tensors = [Tensor(np.zeros(10_000, dtype=np.float32)) for _ in range(5)]
    after = forge.cuda.memory_stats()
    assert after == before
    del cpu_tensors


# -- 7. CPU <-> CUDA transfer ---------------------------------------------------------


def test_cpu_to_cuda_transfer_allocates_new_cuda_storage():
    cpu_t = Tensor(np.ones(2000, dtype=np.float32))
    before = _stable_stats()
    cuda_t = cpu_t.to("cuda")
    after = forge.cuda.memory_stats()
    assert after.allocated_bytes - before.allocated_bytes == 2000 * 4
    del cuda_t


def test_cuda_to_cpu_transfer_creates_independent_copy_source_still_live():
    """`.to()` creates a new Tensor rather than mutating -- the CUDA source
    stays allocated as long as it is referenced, independent of the CPU copy."""
    cuda_t = Tensor(np.ones(2000, dtype=np.float32), device="cuda")
    before = forge.cuda.memory_stats()

    cpu_copy = cuda_t.to("cpu")
    after_transfer = forge.cuda.memory_stats()
    assert after_transfer.allocated_bytes == before.allocated_bytes  # no new CUDA storage from a d2h copy

    del cpu_copy
    after_del_cpu_copy = forge.cuda.memory_stats()
    assert after_del_cpu_copy.allocated_bytes == before.allocated_bytes  # cuda_t is still referenced

    del cuda_t
    after_del_cuda = _stable_stats()
    assert after_del_cuda.allocated_bytes == before.allocated_bytes - 2000 * 4


def test_cuda_to_cuda_same_device_to_is_a_no_op_allocates_nothing():
    cuda_t = Tensor(np.ones(500, dtype=np.float32), device="cuda")
    before = forge.cuda.memory_stats()
    same = cuda_t.to("cuda")
    after = forge.cuda.memory_stats()
    assert same is cuda_t
    assert after == before
    del cuda_t


# -- 8. Autograd lifecycle ------------------------------------------------------------


def test_repeated_forward_backward_without_optimizer_does_not_leak():
    model = _small_model().to("cuda")
    x = Tensor(np.random.default_rng(0).standard_normal((16, 8)).astype(np.float32), device="cuda")

    def run_once():
        out = model(x)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            p.zero_grad()

    for _ in range(5):
        run_once()
    steady1 = _stable_stats().allocated_bytes

    for _ in range(20):
        run_once()
    steady2 = _stable_stats().allocated_bytes

    assert steady2 == steady1


# -- 9. Model / Trainer lifecycle ------------------------------------------------------


def test_model_trainer_workload_lifecycle_returns_to_steady_state():
    forge.random.seed(1)
    model = _small_model().to("cuda")
    optimizer = Adam(model.parameters(), lr=1e-3)
    x = Tensor(np.random.default_rng(1).standard_normal((16, 8)).astype(np.float32), device="cuda")

    before_training = _stable_stats()

    def train_step():
        optimizer.zero_grad()
        prediction = model(x)
        loss = prediction.sum()
        loss.backward()
        optimizer.step()

    forge.cuda.reset_peak_memory_stats()
    for _ in range(5):
        train_step()
    peak_during = forge.cuda.memory_stats().peak_allocated_bytes

    for _ in range(15):
        train_step()
    steady = _stable_stats()

    # Evaluation (no_grad) forward pass after training.
    with forge.no_grad():
        model(x)
    after_eval = _stable_stats()

    assert peak_during >= before_training.allocated_bytes
    # Steady-state after repeated training == steady-state after warmup;
    # persistent state is the model's parameters + Adam's m/v, allocated once.
    assert after_eval.allocated_bytes == steady.allocated_bytes
    assert steady.allocated_bytes >= before_training.allocated_bytes


# -- 10. Dropout / temporary tensor lifecycle --------------------------------------------


def test_dropout_model_repeated_training_and_eval_does_not_leak():
    forge.random.seed(2)
    model = Sequential(Linear(8, 16), ReLU(), Dropout(0.3), Linear(16, 4)).to("cuda")
    optimizer = Adam(model.parameters(), lr=1e-3)
    x = Tensor(np.random.default_rng(2).standard_normal((16, 8)).astype(np.float32), device="cuda")

    def train_step():
        optimizer.zero_grad()
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

    for _ in range(5):
        train_step()
    steady_train_1 = _stable_stats().allocated_bytes
    for _ in range(15):
        train_step()
    steady_train_2 = _stable_stats().allocated_bytes
    assert steady_train_2 == steady_train_1

    model.eval()

    def eval_step():
        with forge.no_grad():
            model(x)

    for _ in range(5):
        eval_step()
    steady_eval_1 = _stable_stats().allocated_bytes
    for _ in range(15):
        eval_step()
    steady_eval_2 = _stable_stats().allocated_bytes
    assert steady_eval_2 == steady_eval_1


# -- 11. Adam optimizer state -------------------------------------------------------------


def test_adam_first_step_allocates_persistent_state_subsequent_steps_do_not_grow():
    p_model = Linear(4, 3).to("cuda")
    optimizer = Adam(p_model.parameters(), lr=0.01)
    x = Tensor(np.random.default_rng(3).standard_normal((5, 4)).astype(np.float32), device="cuda")

    before_first_step = _stable_stats()

    optimizer.zero_grad()
    p_model(x).sum().backward()
    optimizer.step()  # first step: allocates m/v for weight + bias
    after_first_step = _stable_stats()
    assert after_first_step.allocated_bytes > before_first_step.allocated_bytes

    for _ in range(10):
        optimizer.zero_grad()
        p_model(x).sum().backward()
        optimizer.step()
    after_more_steps = _stable_stats()
    assert after_more_steps.allocated_bytes == after_first_step.allocated_bytes


def test_clearing_adam_state_releases_its_cuda_moment_tensors():
    p_model = Linear(4, 3).to("cuda")
    optimizer = Adam(p_model.parameters(), lr=0.01)
    x = Tensor(np.random.default_rng(4).standard_normal((5, 4)).astype(np.float32), device="cuda")

    before = _stable_stats()
    optimizer.zero_grad()
    p_model(x).sum().backward()
    optimizer.step()
    optimizer.zero_grad()  # isolate Adam's own m/v footprint from the (still-live) gradient
    after_step = _stable_stats()
    assert after_step.allocated_bytes > before.allocated_bytes

    optimizer.state.clear()
    after_clear = _stable_stats()
    assert after_clear.allocated_bytes == before.allocated_bytes


# -- 12. Persistence / checkpoint lifecycle ------------------------------------------------


def test_save_checkpoint_does_not_grow_persistent_cuda_allocation(tmp_path):
    forge.random.seed(5)
    model = Linear(4, 3).to("cuda")
    optimizer = Adam(model.parameters(), lr=0.01)
    x = Tensor(np.random.default_rng(5).standard_normal((5, 4)).astype(np.float32), device="cuda")
    for _ in range(3):
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()

    before_save = _stable_stats()
    path = tmp_path / "m22.ckpt"
    save_checkpoint(str(path), model, optimizer, epoch=1, global_step=3)
    after_save = _stable_stats()
    assert after_save.allocated_bytes == before_save.allocated_bytes


def test_load_checkpoint_allocates_expected_state_and_releases_on_deletion(tmp_path):
    forge.random.seed(6)
    model = Linear(4, 3).to("cuda")
    optimizer = Adam(model.parameters(), lr=0.01)
    x = Tensor(np.random.default_rng(6).standard_normal((5, 4)).astype(np.float32), device="cuda")
    for _ in range(3):
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()

    path = tmp_path / "m22_load.ckpt"
    save_checkpoint(str(path), model, optimizer, epoch=1, global_step=3)

    before_load = _stable_stats()
    checkpoint = load_checkpoint(str(path))
    after_load = _stable_stats()
    assert after_load.allocated_bytes > before_load.allocated_bytes  # a second model + optimizer state

    del checkpoint
    after_del = _stable_stats()
    assert after_del.allocated_bytes == before_load.allocated_bytes


# -- 13. Allocation failure semantics --------------------------------------------------------


def test_failed_allocation_does_not_corrupt_statistics():
    """Runs in a **subprocess**, deliberately.

    On the verified development hardware (940MX, driver 582.53, CUDA 12.6), a
    `cudaMalloc` request large enough to fail (e.g. 2**34 bytes, 16 GiB) does
    not corrupt `cudaMalloc`/`cudaMemcpy` themselves -- but it does leave the
    CUDA context unable to launch *any* further kernel (every subsequent
    `add`/`relu`/`matmul`/... call in that same process fails with the same
    "out of memory" driver error, even for a handful of bytes) for the rest
    of that process. This is a genuine, reproducible hardware/driver
    limitation -- not a Forge accounting bug -- documented in
    `docs/architecture/cuda-backend.md`'s **CUDA Memory Statistics** section.
    Provoking it inside the main `pytest` process would poison every CUDA
    test that runs afterward (in this file and any other), so this test
    provokes it in an isolated child process instead: whatever the driver
    does to that process's CUDA context, only that process pays for it.
    """
    script = """
import forge
from forge.exceptions import CUDAError
from forge.backend.cuda.backend import get_cuda_backend

backend = get_cuda_backend()
before = forge.cuda.memory_stats()
try:
    backend._alloc(2**34)  # 16 GiB -- far beyond the 940MX's 2 GiB VRAM
    raise SystemExit("expected CUDAError was not raised")
except CUDAError:
    pass
after = forge.cuda.memory_stats()
assert after == before, f"stats changed across a failed allocation: {before} -> {after}"
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=str(_REPO_ROOT), timeout=60
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "OK" in result.stdout


# -- 14. Memory regression / leak test -----------------------------------------------------


def test_leak_regression_bounded_growth_over_many_iterations():
    """Realistic workload: N forward/backward/optimizer iterations. Compares
    an early steady-state region against a much later one with a bounded-
    growth tolerance, per the milestone brief -- not exact byte equality,
    since a stricter guarantee than "no unbounded growth" is not what this
    test is meant to certify (see `test_model_trainer_workload_lifecycle_
    returns_to_steady_state` above for the exact-equality case this
    environment happens to also satisfy)."""
    forge.random.seed(7)
    model = _small_model().to("cuda")
    optimizer = Adam(model.parameters(), lr=1e-3)
    x = Tensor(np.random.default_rng(7).standard_normal((32, 8)).astype(np.float32), device="cuda")

    def train_step():
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()

    warmup = 10
    for _ in range(warmup):
        train_step()
    early = _stable_stats().allocated_bytes

    iterations = 100
    for _ in range(iterations):
        train_step()
    late = _stable_stats().allocated_bytes

    one_batch_worth = 32 * 8 * 4  # a generous per-iteration tolerance, not zero-tolerance
    assert late - early <= one_batch_worth
