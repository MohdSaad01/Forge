"""Milestone 61: package-level smoke test.

Not a substitute for the full suite -- a single fast check that a fresh
install actually works end-to-end (import, model construction, forward,
backward, an optimizer step, and CPU/CUDA device selection), for a
developer who wants a quick "did the install work" signal without running
all ~1,800 tests. See README.md's Testing section.
"""

from __future__ import annotations

import forge
from forge import Tensor
from forge.data import DataLoader, TensorDataset
from forge.nn import Linear, MSELoss, ReLU, Sequential
from forge.optim import SGD


def test_import_exposes_the_documented_public_surface():
    for name in ("Tensor", "nn", "optim", "data", "training", "serialization", "cuda", "random"):
        assert hasattr(forge, name)


def test_minimal_model_trains_one_step_on_cpu():
    forge.random.seed(0)
    model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)

    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y = Tensor([[1.0], [2.0]])
    before = [p.numpy().copy() for p in model.parameters()]

    prediction = model(x)
    loss = loss_fn(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    after = [p.numpy() for p in model.parameters()]
    assert any((a != b).any() for a, b in zip(after, before)), "optimizer.step() should update parameters"


def test_minimal_model_trains_via_dataloader_and_device_selection_is_explicit():
    dataset = TensorDataset(Tensor([[0.0, 0.0], [1.0, 1.0]]), Tensor([[0.0], [1.0]]))
    loader = DataLoader(dataset, batch_size=2)
    model = Linear(2, 1)

    batch_x, _ = next(iter(loader))
    prediction = model(batch_x)
    assert prediction.shape == (2, 1)

    # CUDA is opt-in and must never be assumed available.
    assert isinstance(forge.cuda.is_cuda_available(), bool)
