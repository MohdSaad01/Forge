"""Milestone 63 integration tests: the `examples/autoencoder` pipeline, CPU-only.

Exercises the exact pipeline `examples/autoencoder/train.py` runs --
`ConvAutoencoder` (`Conv2d`/`MaxPool2d` encoder -> `Linear` bottleneck ->
`Linear`/`UpsampleNearest2d`/`Conv2d` decoder) -> `DataLoader` -> `Trainer`
-> `MSELoss` -> `Adam` -- plus checkpoint save/resume (including resume
equivalence against continuous training), model save/load, and CLI
inspection, against a tiny synthetic `(N, 1, 28, 28)` stand-in dataset so
this suite never needs the real MNIST download -- matching
`tests/test_mnist_example_integration.py`'s precedent exactly (this
example's own real dataset loading path, `examples.autoencoder.dataset.
AutoencoderDataset`, is instead covered directly below against tiny
in-memory IDX files, the one piece of example-specific glue not already
covered by `MNISTDataset`'s own -- untested-by-suite, per that same
precedent -- real-file-loading path).
"""

from __future__ import annotations

import gzip
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge import Tensor, no_grad
from forge.cli.main import main as cli_main
from forge.data import Compose, DataLoader, Lambda, Normalize, TensorDataset
from forge.nn import MSELoss
from forge.optim import Adam
from forge.serialization import load_checkpoint, load_model, load_preprocessing, save_model
from forge.training import Trainer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.autoencoder.dataset import AutoencoderDataset  # noqa: E402
from examples.autoencoder.model import ConvAutoencoder, build_model  # noqa: E402
from examples.autoencoder.train import (  # noqa: E402
    build_transform,
    latent_nearest_neighbor_label_agreement,
    trivial_baseline_mse,
)

_NUM_CLASSES = 10


def _make_synthetic_images(n: int, seed: int) -> "tuple[np.ndarray, np.ndarray]":
    """A tiny `(N, 1, 28, 28)` float32 stand-in for real MNIST, plus a `(N,)` int64 label.

    Same "class L -> a bright vertical stripe at columns [2L, 2L+2)" motif
    `tests/test_mnist_example_integration.py` uses -- reused here because it
    gives each synthetic sample genuine class-correlated visual structure
    (useful both for the reconstruction test and the latent-nearest-neighbor
    test below), not because classification is this example's task.
    """
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((n, 1, 28, 28)) * 0.05).astype(np.float32)
    Y = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        label = i % _NUM_CLASSES
        Y[i] = label
        col = label * 2
        stripe = 1.0 + rng.standard_normal((28, 2)).astype(np.float32) * 0.05
        X[i, 0, :, col : col + 2] = stripe
    X = np.clip(X, 0.0, 1.0)
    return X, Y


def _make_reconstruction_dataset(n: int, seed: int) -> TensorDataset:
    X, _ = _make_synthetic_images(n, seed)
    images = Tensor(X)
    return TensorDataset(images, images)


# -- shape / architecture -----------------------------------------------------


def test_model_forward_shape_round_trips_through_the_bottleneck():
    model = build_model(latent_dim=8)
    x = Tensor(np.zeros((5, 1, 28, 28), dtype=np.float32))
    y = model(x)
    assert y.shape == x.shape


def test_encode_produces_the_configured_latent_dimensionality():
    model = build_model(latent_dim=8)
    x = Tensor(np.zeros((5, 1, 28, 28), dtype=np.float32))
    z = model.encode(x)
    assert z.shape == (5, 8)


# -- CPU training: loss decreases and beats the trivial baseline --------------


def test_full_pipeline_trains_and_beats_baseline_on_cpu():
    forge.random.seed(0)
    train_ds = _make_reconstruction_dataset(200, seed=1)
    test_ds = _make_reconstruction_dataset(60, seed=2)
    train_loader = DataLoader(train_ds, batch_size=20, shuffle=True, generator=np.random.default_rng(3))
    test_loader = DataLoader(test_ds, batch_size=20)

    train_images = np.stack([train_ds[i][0].numpy() for i in range(len(train_ds))])
    test_images = np.stack([test_ds[i][0].numpy() for i in range(len(test_ds))])
    baseline_mse = float(((test_images - train_images.mean(axis=0)) ** 2).mean())

    model = build_model(latent_dim=8)
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cpu", verbose=False)

    initial_params = [p.numpy().copy() for p in model.parameters()]
    history = trainer.fit(train_loader, epochs=15, validation_loader=test_loader)

    assert history[-1].train_loss < history[0].train_loss * 0.5
    eval_result = trainer.evaluate(test_loader)
    assert eval_result.loss < baseline_mse * 0.5

    for before, after in zip(initial_params, model.parameters()):
        assert not np.allclose(before, after.numpy()), "a parameter did not change during training"


# -- checkpoint save/resume ----------------------------------------------------


def test_checkpoint_save_and_resume_restores_state_and_continues_training(tmp_path):
    forge.random.seed(10)
    train_loader = DataLoader(_make_reconstruction_dataset(64, seed=11), batch_size=16, shuffle=False)

    model = build_model(latent_dim=8)
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cpu", verbose=False)
    trainer.fit(train_loader, epochs=2)

    checkpoint_path = tmp_path / "autoencoder_tiny.ckpt"
    trainer.save_checkpoint(str(checkpoint_path))

    checkpoint = load_checkpoint(str(checkpoint_path))
    assert checkpoint.epoch == trainer.epoch == 2
    assert checkpoint.global_step == trainer.global_step
    assert isinstance(checkpoint.model, ConvAutoencoder)

    original_by_name = dict(trainer.model.named_parameters())
    for name, param in checkpoint.model.named_parameters():
        np.testing.assert_allclose(param.numpy(), original_by_name[name].numpy(), atol=1e-6)
        assert param.device == original_by_name[name].device

    for name, param in checkpoint.model.named_parameters():
        original_param = original_by_name[name]
        original_state = trainer.optimizer.state[original_param]
        restored_state = checkpoint.optimizer.state[param]
        assert restored_state.step == original_state.step
        np.testing.assert_allclose(restored_state.m, original_state.m, atol=1e-6)
        np.testing.assert_allclose(restored_state.v, original_state.v, atol=1e-6)

    resumed_trainer = Trainer(checkpoint.model, MSELoss(), checkpoint.optimizer, device="cpu", verbose=False)
    resumed_trainer.resume(checkpoint)
    resumed_trainer.fit(train_loader, epochs=1)
    assert resumed_trainer.epoch == 3
    assert resumed_trainer.global_step == trainer.global_step + len(train_loader)
    assert resumed_trainer.model.device.type == "cpu"


def test_resume_equivalence_matches_continuous_training(tmp_path):
    """`N` epochs -> checkpoint -> reload -> `M` epochs == continuous `N+M` epochs."""
    n_epochs, m_epochs = 2, 2

    def _make_trainer(seed: int) -> "tuple[Trainer, DataLoader]":
        forge.random.seed(seed)
        loader = DataLoader(_make_reconstruction_dataset(48, seed=seed + 1), batch_size=8, shuffle=False)
        model = build_model(latent_dim=8)
        optimizer = Adam(model.parameters(), lr=5e-3)
        return Trainer(model, MSELoss(), optimizer, device="cpu", verbose=False), loader

    continuous_trainer, continuous_loader = _make_trainer(seed=42)
    continuous_trainer.fit(continuous_loader, epochs=n_epochs + m_epochs)

    checkpointed_trainer, checkpointed_loader = _make_trainer(seed=42)
    checkpointed_trainer.fit(checkpointed_loader, epochs=n_epochs)
    checkpoint_path = tmp_path / "resume_equivalence.ckpt"
    checkpointed_trainer.save_checkpoint(str(checkpoint_path))

    checkpoint = load_checkpoint(str(checkpoint_path))
    resumed_trainer = Trainer(checkpoint.model, MSELoss(), checkpoint.optimizer, device="cpu", verbose=False)
    resumed_trainer.resume(checkpoint)
    resumed_trainer.fit(checkpointed_loader, epochs=m_epochs)

    continuous_by_name = dict(continuous_trainer.model.named_parameters())
    for name, param in resumed_trainer.model.named_parameters():
        np.testing.assert_allclose(
            param.numpy(), continuous_by_name[name].numpy(), atol=1e-5,
            err_msg=f"resumed vs. continuous training diverged for parameter '{name}'",
        )


# -- model persistence ---------------------------------------------------------


def test_model_persistence_preserves_predictions(tmp_path):
    forge.random.seed(20)
    train_loader = DataLoader(
        _make_reconstruction_dataset(48, seed=21), batch_size=16, shuffle=True, generator=np.random.default_rng(22)
    )

    model = build_model(latent_dim=8)
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cpu", verbose=False)
    trainer.fit(train_loader, epochs=3)

    query = Tensor(np.random.default_rng(23).random((4, 1, 28, 28)).astype(np.float32))
    with no_grad():
        pre_save = model(query).numpy()

    model_path = tmp_path / "autoencoder_tiny.forge"
    save_model(model, str(model_path))
    reloaded = load_model(str(model_path))
    assert isinstance(reloaded, ConvAutoencoder)
    assert reloaded.latent_dim == model.latent_dim

    with no_grad():
        post_load = reloaded(query).numpy()

    np.testing.assert_allclose(pre_save, post_load, atol=1e-6)


# -- CLI inspection on generated artifacts -------------------------------------


def test_cli_inspects_generated_autoencoder_model_and_checkpoint(tmp_path, capsys):
    forge.random.seed(30)
    train_loader = DataLoader(_make_reconstruction_dataset(32, seed=31), batch_size=8, shuffle=False)

    model = build_model(latent_dim=8)
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cpu", verbose=False)
    trainer.fit(train_loader, epochs=1)

    model_path = tmp_path / "autoencoder_tiny.forge"
    checkpoint_path = tmp_path / "autoencoder_tiny.ckpt"
    save_model(model, str(model_path))
    trainer.save_checkpoint(str(checkpoint_path))

    code = cli_main(["model", "inspect", str(model_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "ConvAutoencoder" in out

    code = cli_main(["checkpoint", "inspect", str(checkpoint_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Optimizer: Adam" in out
    assert "Epoch: 1" in out


# -- latent nearest-neighbor evaluation (algorithm correctness) ----------------


class _StubDataset:
    """A minimal `(image, image)` + `label_at()` stand-in for `latent_nearest_neighbor_label_agreement`."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self._images = images
        self._labels = labels

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int):
        image = Tensor(self._images[index])
        return image, image

    def label_at(self, index: int) -> int:
        return int(self._labels[index])


class _StubModel:
    """A model stand-in whose `encode()` looks up a fixed, hand-crafted latent per sample.

    `latent_nearest_neighbor_label_agreement` draws its subset via
    `rng.choice(..., replace=False)`, which reorders (or, at `n_samples ==
    len(dataset)`, fully permutes) the samples it fetches -- so `encode()`
    cannot just return a fixed array in original-index order, it must return
    the *correct* latent for whichever images it was actually handed. Each
    stub image encodes its own dataset index in pixel `[0, 0]` (see
    `_StubDataset` construction below) so `encode()` can look the right
    latent back up regardless of shuffle order.
    """

    def __init__(self, latents: np.ndarray):
        self._latents = latents

    def encode(self, x: Tensor) -> Tensor:
        indices = x.numpy()[:, 0, 0, 0].round().astype(int)
        return Tensor(self._latents[indices])


def _index_tagged_images(n: int) -> np.ndarray:
    """`(n, 1, 28, 28)` images whose pixel `[0, 0]` holds their own index -- see `_StubModel`."""
    images = np.zeros((n, 1, 28, 28), dtype=np.float32)
    images[:, 0, 0, 0] = np.arange(n)
    return images


def test_latent_nearest_neighbor_agreement_is_perfect_for_well_separated_clusters():
    # Two tight clusters, one per label, far apart -- every point's true
    # nearest neighbor (excluding itself) must share its own label.
    labels = np.array([0, 0, 0, 1, 1, 1])
    latents = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [10.0, 10.0], [10.1, 10.0], [10.0, 10.1]], dtype=np.float32
    )
    dataset = _StubDataset(_index_tagged_images(6), labels)
    model = _StubModel(latents)

    agreement = latent_nearest_neighbor_label_agreement(model, dataset, device="cpu", n_samples=6, seed=0)
    assert agreement == pytest.approx(1.0)


def test_latent_nearest_neighbor_agreement_is_zero_for_perfectly_interleaved_pairs():
    # Each same-label pair is placed maximally far from its partner and
    # right next to the *other* label's partner, forcing every nearest
    # neighbor to have the opposite label.
    labels = np.array([0, 1, 0, 1])
    latents = np.array([[0.0], [0.01], [10.0], [10.01]], dtype=np.float32)
    dataset = _StubDataset(_index_tagged_images(4), labels)
    model = _StubModel(latents)

    agreement = latent_nearest_neighbor_label_agreement(model, dataset, device="cpu", n_samples=4, seed=0)
    assert agreement == pytest.approx(0.0)


# -- AutoencoderDataset: real MNIST-IDX-format wrapping, no download ----------


def _write_idx_images(path: Path, images: np.ndarray) -> None:
    n, rows, cols = images.shape
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, n, rows, cols))
        f.write(images.astype(np.uint8).tobytes())


def _write_idx_labels(path: Path, labels: np.ndarray) -> None:
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">II", 2049, len(labels)))
        f.write(labels.astype(np.uint8).tobytes())


@pytest.fixture
def fake_mnist_root(tmp_path) -> Path:
    """A tiny, real-format (gzip IDX) MNIST-like dataset -- no network access."""
    rng = np.random.default_rng(0)
    n = 6
    images = rng.integers(0, 256, size=(n, 28, 28), dtype=np.uint8)
    labels = np.arange(n, dtype=np.uint8) % _NUM_CLASSES

    _write_idx_images(tmp_path / "train-images-idx3-ubyte.gz", images)
    _write_idx_labels(tmp_path / "train-labels-idx1-ubyte.gz", labels)
    _write_idx_images(tmp_path / "t10k-images-idx3-ubyte.gz", images)
    _write_idx_labels(tmp_path / "t10k-labels-idx1-ubyte.gz", labels)
    return tmp_path


def test_autoencoder_dataset_returns_identical_image_as_both_features_and_target(fake_mnist_root):
    ds = AutoencoderDataset(str(fake_mnist_root), train=True)
    x, y = ds[0]
    assert x.shape == (1, 28, 28)
    np.testing.assert_array_equal(x.numpy(), y.numpy())


def test_autoencoder_dataset_label_at_matches_the_underlying_mnist_label(fake_mnist_root):
    ds = AutoencoderDataset(str(fake_mnist_root), train=True)
    for i in range(len(ds)):
        assert ds.label_at(i) == i % _NUM_CLASSES


def test_autoencoder_dataset_applies_transform_only_to_the_returned_image(fake_mnist_root):
    ds = AutoencoderDataset(str(fake_mnist_root), train=True, transform=lambda t: t * 0.0)
    x, y = ds[0]
    np.testing.assert_array_equal(x.numpy(), np.zeros((1, 28, 28), dtype=np.float32))
    np.testing.assert_array_equal(y.numpy(), np.zeros((1, 28, 28), dtype=np.float32))


# -- Milestone 77: persistable preprocessing (no more Lambda) ------------------


def test_build_transform_is_bit_exact_with_the_old_lambda_based_pipeline():
    """`build_transform()`'s `Normalize(mean=0, std=255)` must compute
    exactly what the pre-Milestone-77 `Lambda(lambda x: x * (1/255))` step
    did -- a persistence-motivated rewrite, not a behavior change."""
    old_pipeline = Compose([Lambda(lambda x: x * (1.0 / 255.0))])
    x = Tensor(np.random.default_rng(0).uniform(0, 255, size=(1, 28, 28)).astype(np.float32))
    np.testing.assert_array_equal(build_transform()(x).numpy(), old_pipeline(x).numpy())


def test_model_persistence_with_preprocessing_round_trips(tmp_path):
    """Milestone 77: `save_model(..., preprocessing=...)` now works for the
    autoencoder (previously blocked by `build_transform()`'s `Lambda` step),
    and the reconstructed transform can prepare a brand-new raw image."""
    forge.random.seed(20)
    train_loader = DataLoader(
        _make_reconstruction_dataset(48, seed=21), batch_size=16, shuffle=True, generator=np.random.default_rng(22)
    )

    model = build_model(latent_dim=8)
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, MSELoss(), optimizer, device="cpu", verbose=False)
    trainer.fit(train_loader, epochs=1)

    model_path = tmp_path / "autoencoder_with_preprocessing.forge"
    save_model(model, str(model_path), preprocessing=build_transform())

    reloaded_preprocessing = load_preprocessing(str(model_path))
    reloaded_model = load_model(str(model_path))

    raw = Tensor(np.random.default_rng(23).uniform(0, 255, size=(1, 28, 28)).astype(np.float32))
    prepared = reloaded_preprocessing(raw).reshape(1, 1, 28, 28)
    with no_grad():
        reconstruction = reloaded_model(prepared)
    assert reconstruction.shape == prepared.shape
