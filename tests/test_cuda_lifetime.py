"""Milestone 23 tests: CUDA memory lifecycle without relying on `gc.collect()`.

Requires an actual working CUDA backend; skipped cleanly otherwise via the
module-level `pytestmark`, matching every other `test_cuda_*.py` file.

Before M23's fix, `CUDAStorage` release depended on cyclic GC: a repeated
CUDA training loop grew `forge.cuda.memory_stats().allocated_bytes` across
iterations unless `gc.collect()` was called, because the graph's Tensors
(and therefore their `CUDAStorage`) were kept alive by a Python reference
cycle in `forge.autograd.engine._topological_order` (see
`docs/architecture/autograd.md`). `tests/test_cuda_memory.py`'s existing
lifecycle tests all call `gc.collect()` in their `_stable_stats()` helper and
so cannot, by construction, distinguish "reclaimed by refcounting" from
"reclaimed by cyclic GC" -- that is exactly what this file adds: the same
kind of workload, asserted stable *without* ever calling `gc.collect()`.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.nn import Dropout, Linear, ReLU, Sequential
from forge.optim import Adam

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


@pytest.fixture(autouse=True)
def _gc_disabled():
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        gc.collect()
        if was_enabled:
            gc.enable()


def _small_model():
    forge.random.seed(0)
    return Sequential(Linear(8, 16), ReLU(), Linear(16, 4))


def test_repeated_forward_backward_does_not_grow_cuda_allocation_without_gc_collect():
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
    steady1 = forge.cuda.memory_stats().allocated_bytes

    for _ in range(30):
        run_once()
    steady2 = forge.cuda.memory_stats().allocated_bytes

    assert steady2 == steady1


def test_training_loop_does_not_grow_cuda_allocation_without_gc_collect():
    forge.random.seed(1)
    model = _small_model().to("cuda")
    optimizer = Adam(model.parameters(), lr=1e-3)
    x = Tensor(np.random.default_rng(1).standard_normal((16, 8)).astype(np.float32), device="cuda")

    def train_step():
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()

    for _ in range(5):
        train_step()
    steady1 = forge.cuda.memory_stats().allocated_bytes

    for _ in range(30):
        train_step()
    steady2 = forge.cuda.memory_stats().allocated_bytes

    assert steady2 == steady1


def test_dropout_training_loop_does_not_grow_cuda_allocation_without_gc_collect():
    forge.random.seed(2)
    model = Sequential(Linear(8, 16), ReLU(), Dropout(0.3), Linear(16, 4)).to("cuda")
    optimizer = Adam(model.parameters(), lr=1e-3)
    x = Tensor(np.random.default_rng(2).standard_normal((16, 8)).astype(np.float32), device="cuda")

    def train_step():
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()

    for _ in range(5):
        train_step()
    steady1 = forge.cuda.memory_stats().allocated_bytes

    for _ in range(30):
        train_step()
    steady2 = forge.cuda.memory_stats().allocated_bytes

    assert steady2 == steady1


def test_gc_collect_after_long_run_frees_no_cuda_memory():
    """Once a training loop has run for a while with `gc.collect()` never
    called, running one now must free zero CUDA bytes -- i.e. nothing was
    "waiting" on cyclic GC to release device memory. This is the precise
    CUDA-visible counterpart of `tests/test_lifetime.py`'s CPU tests: it
    does not assert `gc.collect()` finds zero garbage overall (see the note
    below), only that whatever it finds holds no live CUDA allocation.
    """
    forge.random.seed(3)
    model = _small_model().to("cuda")
    optimizer = Adam(model.parameters(), lr=1e-3)
    x = Tensor(np.random.default_rng(3).standard_normal((16, 8)).astype(np.float32), device="cuda")

    def train_step():
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()

    for _ in range(30):
        train_step()

    before = forge.cuda.memory_stats().allocated_bytes
    gc.collect()
    after = forge.cuda.memory_stats().allocated_bytes

    assert after == before


def test_no_forge_objects_survive_a_backward_call_without_gc():
    """Isolates the M23 fix's actual claim from an unrelated finding.

    `backward()` on a CUDA graph was found, during this milestone's
    investigation, to leave behind a small (`ctypes.c_void_p`, `dict`) pair
    of cyclic garbage on every call -- a long-standing CPython `_ctypes`
    argument-marshaling artifact (not a Forge object, and not something
    this milestone's scope covers: see `docs/architecture/autograd.md`'s
    **Graph teardown and object lifetime** section, "Known limitations").
    It holds no CUDA allocation and is unrelated to the Tensor/Node cycle
    M23 fixes. This test confirms that distinction directly: whatever
    `gc.collect()` still finds after a CUDA backward call contains no
    Forge-owned object.
    """
    import forge.nn as nn
    import forge.tensor as tensor_mod
    from forge.autograd.engine import Node
    from forge.optim.optimizer import Optimizer

    model = _small_model().to("cuda")
    x = Tensor(np.random.default_rng(6).standard_normal((16, 8)).astype(np.float32), device="cuda")

    for _ in range(3):
        model(x).sum().backward()
    gc.collect()

    for _ in range(10):
        model(x).sum().backward()

    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        gc.collect()
        forge_types = (tensor_mod.Tensor, Node, nn.Module, nn.Parameter, Optimizer)
        forge_garbage = [obj for obj in gc.garbage if isinstance(obj, forge_types)]
        assert forge_garbage == []
    finally:
        gc.garbage.clear()
        gc.set_debug(0)
