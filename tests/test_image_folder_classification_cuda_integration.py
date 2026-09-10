"""Milestone 69 integration tests: the `examples/image_folder_classification`
pipeline on CUDA.

Mirrors `tests/test_image_folder_classification_integration.py` (same
`generate_dataset()` real-image-file dataset, same small sizes) but drives
`Trainer(..., device="cuda")`, additionally verifying CUDA residency of
parameters/gradients/Adam state -- `forge.data.ImageFolder` itself always
produces plain CPU Tensors (`forge.data` remains CPU-only, per
`docs/architecture/data-system.md`); it is `Trainer` that moves each batch
to CUDA, exactly as it already does for every other Forge dataset. Skips
cleanly when CUDA is unavailable; hardware-verified on the development
machine's GeForce 940MX (CC 5.0) per
`docs/development/development-environment.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.backend.cuda import CUDAStorage, is_cuda_available
from forge.data import Compose, DataLoader, ImageFolder, Lambda, random_split
from forge.nn import CrossEntropyLoss
from forge.optim import Adam
from forge.serialization import load_checkpoint, load_model, save_model
from forge.training import Accuracy, Trainer, predict

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.image_folder_classification.generate_dataset import CLASSES, generate_dataset  # noqa: E402
from examples.image_folder_classification.model import build_model  # noqa: E402

_NUM_CLASSES = len(CLASSES)
_IMAGE_SIZE = 32


def _build_transform():
    return Compose([Lambda(lambda x: x * (1.0 / 255.0))])


def _make_image_folder(root: Path, samples_per_class: int, seed: int) -> ImageFolder:
    generate_dataset(root, samples_per_class=samples_per_class, image_size=_IMAGE_SIZE, seed=seed)
    return ImageFolder(str(root), transform=_build_transform())


def test_full_pipeline_trains_and_learns_on_cuda(tmp_path):
    forge.random.seed(0)
    full_ds = _make_image_folder(tmp_path / "data", samples_per_class=120, seed=0)
    n_train = int(len(full_ds) * 0.8)
    n_test = len(full_ds) - n_train
    train_ds, test_ds = random_split(full_ds, [n_train, n_test], generator=np.random.default_rng(0))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=np.random.default_rng(1))
    test_loader = DataLoader(test_ds, batch_size=32)

    model = build_model(num_classes=_NUM_CLASSES).to("cuda")
    optimizer = Adam(model.parameters(), lr=4e-4)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cuda", metrics=[Accuracy()], verbose=False)

    history = trainer.fit(train_loader, epochs=25, validation_loader=test_loader)

    assert history[-1].train_loss < history[0].train_loss * 0.5
    eval_result = trainer.evaluate(test_loader)
    assert eval_result.metrics["accuracy"] >= 0.5


def test_cuda_residency_of_parameters_gradients_and_adam_state(tmp_path):
    forge.random.seed(5)
    full_ds = _make_image_folder(tmp_path / "data", samples_per_class=20, seed=6)
    loader = DataLoader(full_ds, batch_size=16, shuffle=False)

    model = build_model(num_classes=_NUM_CLASSES).to("cuda")
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cuda", verbose=False)
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
    full_ds = _make_image_folder(tmp_path / "data", samples_per_class=20, seed=11)
    loader = DataLoader(full_ds, batch_size=16, shuffle=False)

    model = build_model(num_classes=_NUM_CLASSES).to("cuda")
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cuda", verbose=False)
    trainer.fit(loader, epochs=2)

    checkpoint_path = tmp_path / "image_folder_tiny_cuda.ckpt"
    trainer.save_checkpoint(str(checkpoint_path))

    checkpoint = load_checkpoint(str(checkpoint_path), device="cuda")
    assert checkpoint.model.device.type == "cuda"
    for _, param in checkpoint.model.named_parameters():
        assert isinstance(param._data, CUDAStorage)

    resumed_trainer = Trainer(checkpoint.model, CrossEntropyLoss(), checkpoint.optimizer, device="cuda", verbose=False)
    resumed_trainer.resume(checkpoint)
    resumed_trainer.fit(loader, epochs=1)
    assert resumed_trainer.epoch == 3
    assert resumed_trainer.model.device.type == "cuda"


def test_model_persistence_preserves_predictions_on_cuda(tmp_path):
    forge.random.seed(20)
    full_ds = _make_image_folder(tmp_path / "data", samples_per_class=20, seed=21)
    loader = DataLoader(full_ds, batch_size=16, shuffle=True, generator=np.random.default_rng(22))

    model = build_model(num_classes=_NUM_CLASSES).to("cuda")
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cuda", verbose=False)
    trainer.fit(loader, epochs=2)

    query_x, _ = full_ds[0]
    query_batch = query_x.to("cuda").reshape(1, *query_x.shape)
    pre_save = predict(model, query_batch).to("cpu").numpy()

    model_path = tmp_path / "image_folder_tiny_cuda.forge"
    save_model(model, str(model_path))
    reloaded = load_model(str(model_path), device="cuda")
    for _, param in reloaded.named_parameters():
        assert isinstance(param._data, CUDAStorage)

    post_load = predict(reloaded, query_batch).to("cpu").numpy()
    np.testing.assert_allclose(pre_save, post_load, atol=1e-5)
