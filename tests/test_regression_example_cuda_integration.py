"""Milestone 60 integration tests: the `examples/regression` pipeline on CUDA.

Mirrors `tests/test_regression_example_integration.py` (same small
synthetic dataset) but drives `Trainer(..., device="cuda")`, additionally
verifying CUDA residency of parameters/gradients/Adam state and CPU/CUDA
prediction parity -- the acceptance criteria specific to Milestone 60's CUDA
path. Skips cleanly when CUDA is unavailable; hardware-verified on the
development machine's GeForce 940MX (CC 5.0) per
`docs/development/development-environment.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import CUDAStorage, is_cuda_available
from forge.data import DataLoader
from forge.nn import MSELoss
from forge.optim import Adam
from forge.serialization import load_checkpoint, load_model, save_model
from forge.training import MeanAbsoluteError, Trainer

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.regression.dataset import N_FEATURES, make_datasets  # noqa: E402
from examples.regression.experiment import load_run_record  # noqa: E402
from examples.regression.model import build_model  # noqa: E402
from examples.regression.train import main as train_main  # noqa: E402


def test_full_pipeline_trains_and_beats_baseline_on_cuda():
    forge.random.seed(0)
    train_ds, val_ds, _, stats = make_datasets(300, 60, 60, seed=1)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=np.random.default_rng(2))
    val_loader = DataLoader(val_ds, batch_size=16)

    model = build_model().to("cuda")
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cuda", metrics=[MeanAbsoluteError()], verbose=False)

    history = trainer.fit(train_loader, epochs=25, validation_loader=val_loader)

    assert history[-1].train_loss < history[0].train_loss * 0.2
    eval_result = trainer.evaluate(val_loader)
    assert eval_result.loss < stats["y_train_var"] * 0.3


def test_cuda_residency_of_parameters_gradients_and_adam_state():
    forge.random.seed(5)
    train_ds, _, _, _ = make_datasets(64, 10, 10, seed=6)
    loader = DataLoader(train_ds, batch_size=8, shuffle=False)

    model = build_model().to("cuda")
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cuda", verbose=False)
    trainer.fit(loader, epochs=1)

    for name, param in model.named_parameters():
        assert isinstance(param._data, CUDAStorage), f"parameter '{name}' is not CUDA-resident"
        assert param.grad is not None, f"parameter '{name}' has no gradient"
        assert isinstance(param.grad._data, CUDAStorage), f"gradient for '{name}' is not CUDA-resident"
        state = optimizer.state[param]
        assert isinstance(state.m, CUDAStorage), f"Adam 'm' for '{name}' is not CUDA-resident"
        assert isinstance(state.v, CUDAStorage), f"Adam 'v' for '{name}' is not CUDA-resident"


def test_checkpoint_save_and_resume_on_cuda(tmp_path):
    forge.random.seed(10)
    train_ds, _, _, _ = make_datasets(80, 10, 10, seed=11)
    loader = DataLoader(train_ds, batch_size=16, shuffle=False)

    model = build_model().to("cuda")
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cuda", verbose=False)
    trainer.fit(loader, epochs=2)

    checkpoint_path = tmp_path / "regression_tiny_cuda.ckpt"
    trainer.save_checkpoint(str(checkpoint_path))

    checkpoint = load_checkpoint(str(checkpoint_path), device="cuda")
    assert checkpoint.model.device.type == "cuda"
    for _, param in checkpoint.model.named_parameters():
        assert isinstance(param._data, CUDAStorage)

    resumed_trainer = Trainer(checkpoint.model, MSELoss(), checkpoint.optimizer, device="cuda", verbose=False)
    resumed_trainer.resume(checkpoint)
    resumed_trainer.fit(loader, epochs=1)
    assert resumed_trainer.epoch == 3
    assert resumed_trainer.model.device.type == "cuda"


def test_model_persistence_preserves_predictions_on_cuda(tmp_path):
    forge.random.seed(20)
    train_ds, _, _, _ = make_datasets(48, 10, 10, seed=21)
    loader = DataLoader(train_ds, batch_size=8, shuffle=False)

    model = build_model().to("cuda")
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cuda", verbose=False)
    trainer.fit(loader, epochs=2)

    query = Tensor(np.random.default_rng(22).standard_normal((4, N_FEATURES)).astype(np.float32), device="cuda")
    with forge.no_grad():
        pre_save = model(query).to("cpu").numpy()

    model_path = tmp_path / "regression_tiny_cuda.forge"
    save_model(model, str(model_path))
    reloaded = load_model(str(model_path), device="cuda")
    for _, param in reloaded.named_parameters():
        assert isinstance(param._data, CUDAStorage)

    with forge.no_grad():
        post_load = reloaded(query).to("cpu").numpy()

    np.testing.assert_allclose(pre_save, post_load, atol=1e-5)


def test_cpu_and_cuda_prediction_parity_after_training():
    """Same seed/data trained identically on CPU vs. CUDA should predict closely (parity check)."""
    train_ds, _, _, _ = make_datasets(200, 4, 4, seed=1)
    query = Tensor(np.random.default_rng(99).standard_normal((8, N_FEATURES)).astype(np.float32))

    def run(device: str) -> np.ndarray:
        forge.random.seed(1)
        loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=np.random.default_rng(1))
        model = build_model().to(device)
        optimizer = Adam(model.parameters(), lr=5e-3)
        trainer = Trainer(model, MSELoss(), optimizer, device=device, verbose=False)
        trainer.fit(loader, epochs=1)
        with forge.no_grad():
            return model(query.to(device)).to("cpu").numpy()

    cpu_pred = run("cpu")
    cuda_pred = run("cuda")
    np.testing.assert_allclose(cuda_pred, cpu_pred, rtol=1e-3, atol=1e-3)


# -- Milestone 65: reproducible-training workflow, on CUDA ---------------------


def _cuda_params(model_path):
    model = load_model(str(model_path), device="cuda")
    return {name: p.to("cpu").numpy().copy() for name, p in model.named_parameters()}


def test_full_training_matches_partial_then_resume_with_shuffle_on_cuda(tmp_path):
    """CUDA counterpart of `tests/test_regression_reproducible_training.py`'s
    same-named CPU test -- the DataLoader shuffle-generator save/restore this
    milestone added is backend-agnostic (it operates entirely on NumPy
    generators the example owns, never on device-resident state), so it
    should close the same resume gap on CUDA. Hardware-verified on the
    reference GeForce 940MX.
    """
    small = ["--n-train", "80", "--n-val", "20", "--n-test", "20", "--batch-size", "8", "--device", "cuda"]
    base = small + ["--seed", "55", "--lr", "5e-3"]
    out_full = tmp_path / "full"
    out_partial = tmp_path / "partial"

    train_main(base + ["--epochs", "5", "--output-dir", str(out_full)])
    train_main(base + ["--epochs", "3", "--output-dir", str(out_partial)])
    checkpoint_path = out_partial / "regression_checkpoint.forge"
    train_main(base + ["--epochs", "2", "--resume", str(checkpoint_path), "--output-dir", str(out_partial)])

    full_params = _cuda_params(out_full / "regression_model.forge")
    partial_params = _cuda_params(out_partial / "regression_model.forge")
    for name in full_params:
        np.testing.assert_allclose(
            full_params[name], partial_params[name], atol=1e-5,
            err_msg=f"parameter '{name}' diverged between full and resumed CUDA training",
        )

    record = load_run_record(out_partial / "regression_history.json")
    assert [e["epoch"] for e in record["history"]] == [1, 2, 3, 4, 5]
