"""Milestone 29 tests: persistence/checkpointing remain safe around async transfers (Section 43/44).

Every test requires an actual working CUDA backend and is skipped cleanly
otherwise via the module-level `pytestmark`. Parameters themselves are never
produced by an async transfer (`Module.to(device)` uses `Tensor.
_move_storage_`, always fully synchronous -- unchanged by this milestone), so
`save_model()`/`save_checkpoint()` need no code changes; these tests confirm
that training on an asynchronously (pinned, `non_blocking=True`) transferred
*input* does not leave any parameter's data unsafe to serialize.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.nn import Linear, Sequential
from forge.optim import SGD
from forge.serialization import load_model, save_model


pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


@pytest.fixture(autouse=True)
def _clean_cache():
    """Return device memory to the driver between tests -- see `test_cuda_stream_allocator.py`."""
    forge.cuda.empty_cache()
    yield
    forge.cuda.empty_cache()


def _pinned_input(shape, seed: int) -> Tensor:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(shape).astype(np.float32)
    mem = forge.cuda.PinnedMemory(values.nbytes)
    array = mem.numpy(shape=shape, dtype=np.float32)
    array[:] = values
    return Tensor(array, device="cpu")


def test_save_model_after_training_on_an_asynchronously_transferred_input(tmp_path):
    forge.random.seed(29)
    model = Sequential(Linear(4, 8), Linear(8, 2)).to("cuda")
    optimizer = SGD(model.parameters(), lr=0.1)
    x_cpu = _pinned_input((5, 4), seed=29)

    x = x_cpu.to("cuda", non_blocking=True)  # default stream -- returns to Python before the copy necessarily finishes
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    path = tmp_path / "model_m29.forge"
    save_model(model, str(path))  # must observe fully-written parameter data, no incomplete transfer

    loaded = load_model(str(path), device="cuda")
    for original, restored in zip(model.parameters(), loaded.parameters()):
        np.testing.assert_allclose(original.to("cpu").numpy(), restored.to("cpu").numpy())
