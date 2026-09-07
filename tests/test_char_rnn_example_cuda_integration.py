"""Milestone 50 integration tests: the `examples/char_rnn` pipeline on CUDA.

Mirrors `tests/test_char_rnn_example_integration.py` (same tiny synthetic
corpus) but trains with `device="cuda"`, additionally verifying CUDA
residency of parameters and gradients through a full multi-timestep
training step. Skips cleanly when CUDA is unavailable; hardware-verified on
the development machine's GeForce 940MX (CC 5.0) per
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

from examples.char_rnn.dataset import build_dataset  # noqa: E402
from examples.char_rnn.model import build_model  # noqa: E402
from examples.char_rnn.train import train_one_epoch  # noqa: E402

_TINY_CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "a tensor holds numbers and remembers how they were made. "
    "the quick brown fox runs past the lazy dog again. "
) * 6


def test_full_pipeline_trains_and_learns_on_cuda():
    forge.random.seed(0)
    dataset, vocab = build_dataset(_TINY_CORPUS, seq_len=16)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=np.random.default_rng(0))
    model = build_model(vocab.size, hidden_size=24, device="cuda")
    optimizer = Adam(model.parameters(), lr=5e-2)
    loss_fn = CrossEntropyLoss()

    losses = [train_one_epoch(model, loader, optimizer, loss_fn, "cuda") for _ in range(12)]

    uniform_baseline = np.log(vocab.size)
    assert losses[-1] < uniform_baseline * 0.5
    assert losses[-1] < losses[0]

    for name, param in model.named_parameters():
        assert param.device.type == "cuda", f"parameter '{name}' is not CUDA-resident"
        assert param.grad is not None, f"parameter '{name}' has no gradient after training"
        assert param.grad.device.type == "cuda", f"parameter '{name}' gradient is not CUDA-resident"


def test_cpu_and_cuda_first_epoch_loss_match():
    """Same seed/data on CPU vs. CUDA should match closely (parity check)."""
    forge.random.seed(1)
    dataset, vocab = build_dataset(_TINY_CORPUS, seq_len=16)

    def run(device: str) -> float:
        forge.random.seed(1)
        loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=np.random.default_rng(1))
        model = build_model(vocab.size, hidden_size=16, device=device)
        optimizer = Adam(model.parameters(), lr=1e-2)
        return train_one_epoch(model, loader, optimizer, CrossEntropyLoss(), device)

    cpu_loss = run("cpu")
    cuda_loss = run("cuda")
    np.testing.assert_allclose(cuda_loss, cpu_loss, rtol=1e-3, atol=1e-3)


def test_training_with_explicit_compute_stream_matches_default_stream():
    """Milestone 55: `compute_stream=forge.cuda.Stream()` must produce loss
    values matching the default (implicit, per-launch-synchronizing) stream
    -- see `train_one_epoch`'s own docstring and `docs/development/
    m55-post-m54-assessment.md` for why this optional parameter exists
    (skips the default stream's per-kernel-launch blocking synchronize,
    found to dominate this loop's wall-clock time on the reference 940MX)."""
    import forge.cuda as cuda

    dataset, vocab = build_dataset(_TINY_CORPUS, seq_len=16)

    def run(compute_stream) -> float:
        forge.random.seed(2)
        loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=np.random.default_rng(2))
        model = build_model(vocab.size, hidden_size=16, device="cuda")
        optimizer = Adam(model.parameters(), lr=1e-2)
        return train_one_epoch(model, loader, optimizer, CrossEntropyLoss(), "cuda", compute_stream)

    default_loss = run(None)
    stream_loss = run(cuda.Stream())
    np.testing.assert_allclose(stream_loss, default_loss, rtol=1e-4, atol=1e-5)


def test_full_pipeline_trains_and_learns_on_cuda_with_compute_stream():
    """The explicit-stream path must still train correctly end to end, not
    just match a single epoch's loss -- mirrors
    `test_full_pipeline_trains_and_learns_on_cuda` above with a
    `compute_stream` supplied."""
    import forge.cuda as cuda

    forge.random.seed(0)
    dataset, vocab = build_dataset(_TINY_CORPUS, seq_len=16)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=np.random.default_rng(0))
    model = build_model(vocab.size, hidden_size=24, device="cuda")
    optimizer = Adam(model.parameters(), lr=5e-2)
    loss_fn = CrossEntropyLoss()
    compute_stream = cuda.Stream()

    losses = [
        train_one_epoch(model, loader, optimizer, loss_fn, "cuda", compute_stream) for _ in range(12)
    ]

    uniform_baseline = np.log(vocab.size)
    assert losses[-1] < uniform_baseline * 0.5
    assert losses[-1] < losses[0]
