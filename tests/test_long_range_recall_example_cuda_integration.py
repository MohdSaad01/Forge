"""Milestone 67 integration tests: the `examples/long_range_recall` pipeline on CUDA.

Mirrors `tests/test_long_range_recall_example_integration.py` (same fast
synthetic settings) but trains with `device="cuda"`, additionally verifying
CUDA residency of parameters/gradients and CPU/CUDA loss parity for both
cells. Skips cleanly when CUDA is unavailable; hardware-verified on the
development machine's GeForce 940MX (CC 5.0) per
`docs/development/development-environment.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.backend.cuda import is_cuda_available
from forge.data import DataLoader
from forge.nn import CrossEntropyLoss
from forge.optim import Adam

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.long_range_recall.dataset import VOCAB_SIZE, build_dataset  # noqa: E402
from examples.long_range_recall.model import build_model  # noqa: E402
from examples.long_range_recall.train import train_one_epoch  # noqa: E402


@pytest.mark.parametrize("cell_type", ["rnn", "lstm"])
def test_full_pipeline_trains_and_learns_on_cuda(cell_type):
    forge.random.seed(0)
    dataset = build_dataset(n_sequences=256, seq_len=8, seed=0)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, generator=np.random.default_rng(0))
    model = build_model(cell_type, VOCAB_SIZE, hidden_size=16, device="cuda")
    optimizer = Adam(model.parameters(), lr=2e-2)
    loss_fn = CrossEntropyLoss()

    losses = [train_one_epoch(model, loader, optimizer, loss_fn, "cuda") for _ in range(15)]

    chance_baseline = np.log(VOCAB_SIZE)
    assert losses[-1] < chance_baseline * 0.3
    assert losses[-1] < losses[0]

    for name, param in model.named_parameters():
        assert param.device.type == "cuda", f"parameter '{name}' is not CUDA-resident"
        assert param.grad is not None, f"parameter '{name}' has no gradient after training"
        assert param.grad.device.type == "cuda", f"parameter '{name}' gradient is not CUDA-resident"


@pytest.mark.parametrize("cell_type", ["rnn", "lstm"])
def test_cpu_and_cuda_first_epoch_loss_match(cell_type):
    """Same seed/data on CPU vs. CUDA should match closely (parity check)."""
    forge.random.seed(1)
    dataset = build_dataset(n_sequences=128, seq_len=8, seed=1)

    def run(device: str) -> float:
        forge.random.seed(1)
        loader = DataLoader(dataset, batch_size=16, shuffle=True, generator=np.random.default_rng(1))
        model = build_model(cell_type, VOCAB_SIZE, hidden_size=12, device=device)
        optimizer = Adam(model.parameters(), lr=1e-2)
        return train_one_epoch(model, loader, optimizer, CrossEntropyLoss(), device)

    cpu_loss = run("cpu")
    cuda_loss = run("cuda")
    np.testing.assert_allclose(cuda_loss, cpu_loss, rtol=1e-3, atol=1e-3)
