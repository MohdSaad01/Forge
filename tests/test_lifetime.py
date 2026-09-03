"""Milestone 23 tests: Tensor/autograd/Module/Optimizer reference-lifetime.

Covers the M22-discovered finding -- Forge's autograd graph contained a
genuine Python reference cycle (a recursive nested closure in
`forge.autograd.engine._topological_order` referencing itself, see
`docs/architecture/autograd.md`'s **Graph teardown and object lifetime**
section) -- and its fix: every scenario below asserts that a temporary
computation graph's Tensors become unreachable through plain reference
counting alone, with cyclic GC *disabled*, rather than requiring
`gc.collect()` to reclaim them.

These tests are deliberately CPU-only (per the milestone brief's "CPU-first
diagnosis": the cycle is a Python object-ownership issue, not a CUDA one).
The corresponding CUDA memory-lifecycle regression lives in
`tests/test_cuda_lifetime.py`.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.autograd.engine import Node, _topological_order
from forge.data import DataLoader, TensorDataset
from forge.nn import Conv2d, Dropout, Flatten, Linear, MaxPool2d, ReLU, Sequential
from forge.nn.loss import CrossEntropyLoss, MSELoss
from forge.optim import SGD, Adam
from forge.training import Trainer


@pytest.fixture(autouse=True)
def _gc_disabled():
    """Every test in this module runs with cyclic GC off, then restores it.

    The property under test is specifically "plain reference counting is
    enough" -- enabling `gc` would let a leftover cycle get silently
    reclaimed by the collector and mask a regression.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        gc.collect()
        if was_enabled:
            gc.enable()


def _dead(ref: "weakref.ReferenceType") -> bool:
    return ref() is None


# -- 1. _topological_order itself creates no self-referential closure -------


def test_topological_order_visit_is_not_a_recursive_closure():
    """Regression test for the exact M22 cycle: a nested `def visit(...)`
    that calls itself by name captures a cell referencing its own function
    object. `_topological_order` must not contain such a nested function at
    all (see its docstring)."""
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = (x * x).sum()

    order = _topological_order(y)
    assert [t.numpy().tolist() if t.shape else t.numpy().item() for t in order] == [
        [1.0, 2.0],
        [1.0, 4.0],
        5.0,
    ]

    # No nested function (e.g. the old recursive "visit" closure) should
    # still be alive from that call -- there is nothing left to be
    # self-referential. `_topological_order` itself (the module-level
    # function, reachable via the module and thus expected to be "alive")
    # is excluded; only a `<locals>`-qualified nested function would
    # indicate the cycle is back.
    leftover_closures = [
        obj
        for obj in gc.get_objects()
        if callable(obj) and "_topological_order.<locals>" in getattr(obj, "__qualname__", "")
    ]
    assert leftover_closures == []


# -- 2. Simple autograd: x -> op -> backward ---------------------------------


def test_simple_op_backward_releases_graph_without_gc():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * x
    node_ref = weakref.ref(y.grad_fn)
    y_ref = weakref.ref(y)

    z = y.sum()
    z.backward()

    assert x.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), [2.0, 4.0, 6.0])

    del y, z
    assert _dead(y_ref), "non-leaf output Tensor survived backward + deref without gc.collect()"
    assert _dead(node_ref), "grad_fn Node survived backward + deref without gc.collect()"


# -- 3. Multi-use graph: x feeds two branches that recombine -----------------


def test_multi_use_graph_releases_all_intermediate_tensors_without_gc():
    x = Tensor([2.0, 3.0], requires_grad=True)
    a = x * 2.0
    b = x * 3.0
    combined = (a + b).sum()

    a_ref, b_ref, combined_ref = weakref.ref(a), weakref.ref(b), weakref.ref(combined)

    combined.backward()
    np.testing.assert_allclose(x.grad.numpy(), [5.0, 5.0])

    del a, b, combined
    assert _dead(a_ref)
    assert _dead(b_ref)
    assert _dead(combined_ref)


# -- 4. Linear -> ReLU -> Linear -> loss -------------------------------------


class _MLP(forge.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 8)
        self.relu = ReLU()
        self.fc2 = Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_mlp_forward_backward_leaves_no_uncollected_tensor_growth():
    forge.random.seed(0)
    model = _MLP()
    loss_fn = MSELoss()
    x = Tensor([[1.0, 2.0, 3.0, 4.0]])
    target = Tensor([[0.0, 1.0]])

    def step():
        prediction = model(x)
        loss = loss_fn(prediction, target)
        loss.backward()
        for p in model.parameters():
            p.zero_grad()

    step()  # warmup: absorb any first-call one-time state
    before = len(gc.get_objects())
    for _ in range(25):
        step()
    after = len(gc.get_objects())

    # Every iteration builds and discards an identically-shaped graph;
    # without the fix this grows by ~10 uncollected Tensors/iteration.
    assert after - before == 0


# -- 5. CNN (Conv2d / MaxPool2d / Flatten / Linear) --------------------------


def test_cnn_forward_backward_leaves_no_uncollected_tensor_growth():
    forge.random.seed(1)
    model = Sequential(
        Conv2d(1, 4, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2d(2),
        Flatten(),
        Linear(4 * 4 * 4, 3),
    )
    loss_fn = CrossEntropyLoss()
    x = Tensor(np.random.default_rng(1).standard_normal((2, 1, 8, 8)).astype(np.float32))
    target = np.array([0, 2])

    def step():
        prediction = model(x)
        loss = loss_fn(prediction, target)
        loss.backward()
        for p in model.parameters():
            p.zero_grad()

    step()
    before = len(gc.get_objects())
    for _ in range(10):
        step()
    after = len(gc.get_objects())
    assert after - before == 0


# -- 6. Dropout (training mode) ----------------------------------------------


def test_dropout_training_forward_backward_leaves_no_uncollected_growth():
    forge.random.seed(2)
    model = Sequential(Linear(6, 10), ReLU(), Dropout(0.4), Linear(10, 3))
    model.train()
    loss_fn = MSELoss()
    x = Tensor(np.random.default_rng(2).standard_normal((4, 6)).astype(np.float32))
    target = Tensor(np.random.default_rng(3).standard_normal((4, 3)).astype(np.float32))

    def step():
        prediction = model(x)
        loss = loss_fn(prediction, target)
        loss.backward()
        for p in model.parameters():
            p.zero_grad()

    step()
    before = len(gc.get_objects())
    for _ in range(15):
        step()
    after = len(gc.get_objects())
    assert after - before == 0


# -- 7. Adam: persistent state stays, temporaries do not ---------------------


def test_adam_repeated_steps_keep_only_persistent_state_alive():
    forge.random.seed(3)
    model = Linear(4, 3)
    optimizer = Adam(model.parameters(), lr=0.01)
    x = Tensor(np.random.default_rng(3).standard_normal((5, 4)).astype(np.float32))
    target = Tensor(np.random.default_rng(4).standard_normal((5, 3)).astype(np.float32))
    loss_fn = MSELoss()

    def step():
        optimizer.zero_grad()
        loss = loss_fn(model(x), target)
        loss.backward()
        optimizer.step()

    step()  # allocates Adam's m/v state
    assert set(optimizer.state.keys()) == set(model.parameters())
    before = len(gc.get_objects())

    for _ in range(20):
        step()
    after = len(gc.get_objects())

    assert after - before == 0
    # Adam state must still be exactly the persistent moment buffers, not
    # something that grew or was replaced by leftover graph state.
    assert set(optimizer.state.keys()) == set(model.parameters())
    for state in optimizer.state.values():
        assert state.m.shape == (4, 3) or state.m.shape == (3,)


# -- 8. Trainer: repeated fit() epochs ---------------------------------------


def _tiny_loader():
    forge.random.seed(4)
    rng = np.random.default_rng(4)
    X = rng.uniform(-1, 1, size=(16, 3))
    y = (X[:, 0] - X[:, 1] + 0.5 * X[:, 2]).reshape(-1, 1)
    dataset = TensorDataset(Tensor(X.astype(np.float32)), Tensor(y.astype(np.float32)))
    return DataLoader(dataset, batch_size=4, shuffle=False)


def test_trainer_repeated_epochs_do_not_accumulate_uncollected_objects():
    forge.random.seed(4)
    model = Linear(3, 1)
    trainer = Trainer(model=model, loss_fn=MSELoss(), optimizer=SGD(model.parameters(), lr=0.01), verbose=False)
    loader = _tiny_loader()

    trainer.fit(loader, epochs=1)  # warmup epoch
    before = len(gc.get_objects())
    trainer.fit(loader, epochs=5)
    after = len(gc.get_objects())

    # Five epochs of the same loader build and discard the same shape of
    # graph each batch; growth should not scale with epoch count.
    assert after - before == 0


# -- 9. no_grad mode: no graph is built, nothing to release -------------------


def test_no_grad_forward_builds_no_graph_objects():
    x = Tensor([1.0, 2.0], requires_grad=True)
    with forge.no_grad():
        y = x * x
    assert y.grad_fn is None
    assert y.requires_grad is False


# -- 10. GC-disabled regression across a mixed workload -----------------------


def test_mixed_workload_gc_disabled_regression():
    """The milestone's headline property: repeated, varied Forge workloads
    do not accumulate Forge-created unreachable cycles that only cyclic GC
    can reclaim. Runs several of the scenarios above back to back and checks
    that live object count returns to (and stays at) a steady state without
    ever calling `gc.collect()` mid-workload."""
    forge.random.seed(5)
    model = Sequential(Linear(5, 12), ReLU(), Dropout(0.3), Linear(12, 4))
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = CrossEntropyLoss()
    x = Tensor(np.random.default_rng(5).standard_normal((8, 5)).astype(np.float32))
    target = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    def step():
        optimizer.zero_grad()
        loss = loss_fn(model(x), target)
        loss.backward()
        optimizer.step()

    for _ in range(5):
        step()
    steady1 = len(gc.get_objects())

    for _ in range(50):
        step()
    steady2 = len(gc.get_objects())

    assert steady2 == steady1

    # Confirm this isn't vacuous: gc.collect() finds nothing new to do,
    # i.e. the steady state genuinely has no dangling Forge cycles left for
    # it to clean up.
    collected = gc.collect()
    assert collected == 0
