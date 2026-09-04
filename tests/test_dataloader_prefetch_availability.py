"""Milestone 30: CUDA-unavailable / CPU-only behavior for `CUDAPrefetchLoader`.

Split out from `tests/test_dataloader_prefetch.py` deliberately, matching
every other `test_*_availability.py` file's convention in this repo (see
`tests/test_cuda_pinned_memory_availability.py`): must run on any machine,
CUDA or not, since it specifically exercises the "CUDA is not available"
path (via a monkeypatched `is_cuda_available`) and the "importing forge.data
never requires CUDA" contract (Section 25/67 of the milestone brief).
"""

from __future__ import annotations

import numpy as np
import pytest

from forge import Tensor
from forge.exceptions import CUDAError, DataError


def test_importing_forge_data_never_requires_cuda():
    """Re-import path exercised fresh -- importing forge.data/forge.data.prefetch must never probe hardware."""
    import forge.data
    import forge.data.prefetch

    assert hasattr(forge.data, "CUDAPrefetchLoader")
    assert hasattr(forge.data, "DataLoader")


def test_plain_dataloader_fully_functional_without_cuda():
    from forge.data import DataLoader, TensorDataset

    x = Tensor(np.arange(12, dtype=np.float32).reshape(6, 2))
    y = Tensor(np.arange(6, dtype=np.float32))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    batches = list(loader)
    assert len(batches) == 3


def test_prefetch_loader_construction_raises_cuda_error_when_unavailable(monkeypatch):
    monkeypatch.setattr("forge.backend.cuda.backend.is_cuda_available", lambda: False)
    from forge.data import DataLoader, TensorDataset
    from forge.data.prefetch import CUDAPrefetchLoader

    x = Tensor(np.arange(8, dtype=np.float32).reshape(4, 2))
    y = Tensor(np.arange(4, dtype=np.float32))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    with pytest.raises(CUDAError):
        CUDAPrefetchLoader(loader, device="cuda")


def test_dataloader_prefetch_method_raises_cuda_error_when_unavailable(monkeypatch):
    monkeypatch.setattr("forge.backend.cuda.backend.is_cuda_available", lambda: False)
    from forge.data import DataLoader, TensorDataset

    x = Tensor(np.arange(8, dtype=np.float32).reshape(4, 2))
    y = Tensor(np.arange(4, dtype=np.float32))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    with pytest.raises(CUDAError):
        loader.prefetch(device="cuda")


def test_prefetch_loader_rejects_cpu_device_even_without_cuda(monkeypatch):
    """device='cpu' is rejected before ever touching CUDA -- must fail with DataError, not CUDAError."""
    monkeypatch.setattr("forge.backend.cuda.backend.is_cuda_available", lambda: False)
    from forge.data import DataLoader, TensorDataset
    from forge.data.prefetch import CUDAPrefetchLoader

    x = Tensor(np.arange(8, dtype=np.float32).reshape(4, 2))
    y = Tensor(np.arange(4, dtype=np.float32))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    with pytest.raises(DataError):
        CUDAPrefetchLoader(loader, device="cpu")


def test_trainer_prefetch_true_rejects_cpu_device():
    from forge.nn import Linear, MSELoss
    from forge.optim import SGD
    from forge.training import Trainer
    from forge.exceptions import TrainerError

    model = Linear(4, 1)
    with pytest.raises(TrainerError):
        Trainer(model, MSELoss(), SGD(model.parameters(), lr=0.01), device="cpu", prefetch=True)
