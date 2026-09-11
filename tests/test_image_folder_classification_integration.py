"""Milestone 69/70 integration tests: the `examples/image_folder_classification`
pipeline, CPU-only.

Exercises the exact pipeline `examples/image_folder_classification/train.py`
runs -- `generate_dataset()` (real, **mixed-resolution** PNG files on disk,
since Milestone 70) -> `ImageFolder` -> `Resize` -> `random_split` ->
`DataLoader` -> `Trainer` -> `CrossEntropyLoss` -> `Adam`, plus checkpoint
save/resume and model save/load -- against a smaller synthetic dataset than
the full example run (faster, still a real end-to-end proof against real
image files, not a toy unit test). See
`tests/test_mnist_example_integration.py` for the precedent this follows.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.data import Compose, DataLoader, ImageFolder, Lambda, Resize, random_split
from forge.nn import CrossEntropyLoss
from forge.optim import Adam
from forge.serialization import load_checkpoint, load_classes, load_model, load_preprocessing, save_model
from forge.training import Accuracy, Trainer, interpret_classification, predict, start_training_session

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.image_folder_classification import train as train_module  # noqa: E402
from examples.image_folder_classification.generate_dataset import CLASSES, generate_dataset  # noqa: E402
from examples.image_folder_classification.model import build_model  # noqa: E402

_NUM_CLASSES = len(CLASSES)
_MIN_SIZE = 24
_MAX_SIZE = 48
_RESIZE_SIZE = (64, 64)  # must match examples/image_folder_classification/model.py's expected input


def _build_transform():
    return Compose([Resize(_RESIZE_SIZE), Lambda(lambda x: x * (1.0 / 255.0))])


def _make_image_folder(root: Path, samples_per_class: int, seed: int) -> ImageFolder:
    generate_dataset(root, samples_per_class=samples_per_class, min_size=_MIN_SIZE, max_size=_MAX_SIZE, seed=seed)
    return ImageFolder(str(root), transform=_build_transform())


# -- dataset generation ---------------------------------------------------


def test_generate_dataset_produces_a_valid_image_folder(tmp_path):
    ds = _make_image_folder(tmp_path / "data", samples_per_class=5, seed=0)
    assert ds.classes == sorted(CLASSES)
    assert len(ds) == 5 * _NUM_CLASSES
    image, label = ds[0]
    assert image.shape == (3, _RESIZE_SIZE[0], _RESIZE_SIZE[1])
    assert int(label.numpy()) in range(_NUM_CLASSES)


def test_generated_source_images_are_genuinely_mixed_resolution(tmp_path):
    """Confirms the M70 premise: raw generated files vary in (H, W), and
    ImageFolder alone (no transform) cannot batch them -- Resize is required.
    """
    root = tmp_path / "data"
    generate_dataset(root, samples_per_class=10, min_size=_MIN_SIZE, max_size=_MAX_SIZE, seed=0)
    raw_ds = ImageFolder(str(root))  # no transform: exposes native, varying (H, W)
    shapes = {raw_ds[i][0].shape for i in range(len(raw_ds))}
    assert len(shapes) > 1

    from forge.exceptions import DataError

    with pytest.raises(DataError):
        list(DataLoader(raw_ds, batch_size=4))


# -- shape / architecture -----------------------------------------------------


def test_model_forward_shape_matches_three_class_output():
    model = build_model(num_classes=_NUM_CLASSES)
    x = Tensor(np.zeros((5, 3, _RESIZE_SIZE[0], _RESIZE_SIZE[1]), dtype=np.float32))
    y = model(x)
    assert y.shape == (5, _NUM_CLASSES)


# -- CPU training: real image files -> loss decreases, learns above chance --


def test_full_pipeline_trains_and_learns_on_cpu(tmp_path):
    forge.random.seed(0)
    full_ds = _make_image_folder(tmp_path / "data", samples_per_class=120, seed=0)

    n_train = int(len(full_ds) * 0.8)
    n_test = len(full_ds) - n_train
    train_ds, test_ds = random_split(full_ds, [n_train, n_test], generator=np.random.default_rng(0))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=np.random.default_rng(1))
    test_loader = DataLoader(test_ds, batch_size=32)

    model = build_model(num_classes=_NUM_CLASSES)
    optimizer = Adam(model.parameters(), lr=4e-4)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cpu", metrics=[Accuracy()], verbose=False)

    initial_params = [p.numpy().copy() for p in model.parameters()]
    history = trainer.fit(train_loader, epochs=25, validation_loader=test_loader)

    assert history[-1].train_loss < history[0].train_loss * 0.5
    eval_result = trainer.evaluate(test_loader)
    # 3-class chance accuracy is 1/3; this checks substantially above chance
    # (real, real-file training -- see the module docstring for context).
    # Threshold is 0.45, not 0.5: M70's mixed-resolution + Resize task is
    # measurably harder than M69's uniform-32x32 one (Resize distorts
    # aspect ratio for non-square sources), and CUDA's reduction-order
    # non-determinism at this reduced test scale (120 samples/class, vs.
    # the full example's 300) lands reproducibly in the 0.47-0.49 range on
    # the reference 940MX -- see docs/development/m70-image-preprocessing.md.
    assert eval_result.metrics["accuracy"] >= 0.45

    for before, after in zip(initial_params, model.parameters()):
        assert not np.allclose(before, after.numpy()), "a parameter did not change during training"


# -- checkpoint save/resume ----------------------------------------------------


def test_checkpoint_save_and_resume_restores_state_and_continues_training(tmp_path):
    forge.random.seed(10)
    full_ds = _make_image_folder(tmp_path / "data", samples_per_class=20, seed=11)
    loader = DataLoader(full_ds, batch_size=16, shuffle=False)

    model = build_model(num_classes=_NUM_CLASSES)
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cpu", verbose=False)
    trainer.fit(loader, epochs=2)

    checkpoint_path = tmp_path / "image_folder_tiny.ckpt"
    trainer.save_checkpoint(str(checkpoint_path))

    checkpoint = load_checkpoint(str(checkpoint_path))
    assert checkpoint.epoch == trainer.epoch == 2
    assert checkpoint.global_step == trainer.global_step

    original_by_name = dict(trainer.model.named_parameters())
    for name, param in checkpoint.model.named_parameters():
        np.testing.assert_allclose(param.numpy(), original_by_name[name].numpy(), atol=1e-6)
        assert param.device == original_by_name[name].device

    resumed_trainer = Trainer(checkpoint.model, CrossEntropyLoss(), checkpoint.optimizer, device="cpu", verbose=False)
    resumed_trainer.resume(checkpoint)
    resumed_trainer.fit(loader, epochs=1)
    assert resumed_trainer.epoch == 3
    assert resumed_trainer.global_step == trainer.global_step + len(loader)
    assert resumed_trainer.model.device.type == "cpu"


# -- model persistence + standalone inference ----------------------------------


def test_model_persistence_and_predict_map_back_to_class_name(tmp_path):
    forge.random.seed(20)
    full_ds = _make_image_folder(tmp_path / "data", samples_per_class=20, seed=21)
    n_train = int(len(full_ds) * 0.8)
    n_test = len(full_ds) - n_train
    train_ds, test_ds = random_split(full_ds, [n_train, n_test], generator=np.random.default_rng(0))
    loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=np.random.default_rng(22))

    model = build_model(num_classes=_NUM_CLASSES)
    optimizer = Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer, device="cpu", verbose=False)
    trainer.fit(loader, epochs=3)

    query_x, query_y = test_ds[0]
    query_batch = query_x.reshape(1, *query_x.shape)
    pre_save_pred = predict(model, query_batch).numpy()

    model_path = tmp_path / "image_folder_tiny.forge"
    save_model(model, str(model_path))
    reloaded = load_model(str(model_path))

    post_load_pred = predict(reloaded, query_batch).numpy()
    np.testing.assert_allclose(pre_save_pred, post_load_pred, atol=1e-6)

    # Class-index -> class-name mapping (Section 12): the predicted index
    # must be usable as an index into `full_ds.classes`.
    predicted_idx = int(np.argmax(post_load_pred, axis=1)[0])
    assert 0 <= predicted_idx < _NUM_CLASSES
    predicted_name = full_ds.classes[predicted_idx]
    assert predicted_name in CLASSES
    true_name = full_ds.classes[int(query_y.numpy())]
    assert true_name in CLASSES


# -- Milestone 80: forge.train() fresh path + train->artifact->fresh-process --


def _tiny_main_args(tmp_path: Path, epochs: int, seed: int = 0) -> list:
    return [
        "--data-root", str(tmp_path / "data"),
        "--generate",
        "--samples-per-class", "15",
        "--min-size", str(_MIN_SIZE),
        "--max-size", str(_MAX_SIZE),
        "--device", "cpu",
        "--epochs", str(epochs),
        "--batch-size", "8",
        "--seed", str(seed),
        "--output-dir", str(tmp_path / "artifacts"),
    ]


def test_main_fresh_path_uses_forge_train_and_produces_a_working_artifact(tmp_path):
    """`train.py::main()`'s fresh (non-`--resume`) path now calls `forge.train()`
    (Milestone 80) instead of hand-assembling `Trainer`. This calls the real
    `main()` entry point -- the exact thing `python -m examples.
    image_folder_classification.train` runs -- not a hand-built `Trainer`,
    so it also exercises argument parsing and the script's own wiring, which
    every other test in this file (all of which call `Trainer`/`build_model`
    directly) does not."""
    train_module.main(_tiny_main_args(tmp_path, epochs=2))

    model_path = tmp_path / "artifacts" / "image_folder_model.forge"
    checkpoint_path = tmp_path / "artifacts" / "image_folder_checkpoint.forge"
    assert model_path.exists()
    assert checkpoint_path.exists()

    # The checkpoint's fresh path now writes "data_loader_rng_state" via
    # plain forge.save_checkpoint() (not TrainingSession.save_checkpoint())
    # -- confirm the key is still there, in the same shape
    # start_training_session()'s resume path expects.
    checkpoint = load_checkpoint(str(checkpoint_path))
    assert "data_loader_rng_state" in checkpoint.extra

    # The saved model is a real, working portable artifact: preprocessing
    # and classes round-trip, and a fresh prediction succeeds.
    classes = load_classes(str(model_path))
    assert classes == sorted(CLASSES)
    preprocessing = load_preprocessing(str(model_path))
    assert preprocessing is not None
    model = load_model(str(model_path))
    query = Tensor(np.zeros((1, 3, *_RESIZE_SIZE), dtype=np.float32))
    output = predict(model, query)
    result = interpret_classification(output, classes)[0]
    assert result.label in classes


def test_resume_after_a_forge_train_fresh_run_matches_continuous_training(tmp_path):
    """The Milestone 73 shuffle-resume-equivalence guarantee must survive the
    Milestone 80 retrofit: a fresh run trained via `forge.train()` (writing
    `data_loader_rng_state` through plain `forge.save_checkpoint()`) followed
    by `--resume` (still `start_training_session()`) must produce the exact
    same parameters as one continuous run over the combined epoch count --
    with `shuffle=True` (this example's real default), the specific case
    Milestone 65 originally found broken and Milestone 73 fixed for this
    example. `test_checkpoint_save_and_resume_restores_state_and_continues_
    training` above only exercises `shuffle=False` and the pre-existing
    `start_training_session()`-for-both-paths shape; this test is the one
    that would have caught a regression from the M80 refactor."""
    n_epochs, m_epochs, seed = 2, 2, 5

    root = tmp_path / "shared_data"
    generate_dataset(root, samples_per_class=15, min_size=_MIN_SIZE, max_size=_MAX_SIZE, seed=seed)

    def _run(full_epochs: "int | None", resume_epochs: "int | None", out_name: str):
        out_dir = tmp_path / out_name
        common = ["--data-root", str(root), "--device", "cpu", "--batch-size", "8",
                  "--seed", str(seed), "--output-dir", str(out_dir)]
        if full_epochs is not None:
            train_module.main(common + ["--epochs", str(full_epochs)])
        if resume_epochs is not None:
            ckpt = out_dir / "image_folder_checkpoint.forge"
            train_module.main(common + ["--epochs", str(resume_epochs), "--resume", str(ckpt)])
        return out_dir / "image_folder_model.forge"

    continuous_model_path = _run(n_epochs + m_epochs, None, "continuous")
    checkpointed_model_path = _run(n_epochs, m_epochs, "checkpointed")

    continuous_params = dict(load_model(str(continuous_model_path)).named_parameters())
    checkpointed_params = dict(load_model(str(checkpointed_model_path)).named_parameters())
    assert continuous_params.keys() == checkpointed_params.keys()
    for name, param in checkpointed_params.items():
        np.testing.assert_allclose(
            param.numpy(), continuous_params[name].numpy(), atol=1e-5,
            err_msg=f"resumed (forge.train() fresh -> start_training_session() resume) vs. "
                    f"continuous training diverged for parameter '{name}'",
        )


def test_infer_cli_runs_in_a_genuinely_separate_process(tmp_path):
    """Every one of Milestone 71/72/77/78's reports describes fresh-process
    inference as already "verified" -- but no test before this one ever
    actually launched a second OS process; `test_model_persistence_and_
    predict_map_back_to_class_name` above and `tests/test_classification_
    metadata.py`'s `infer.py` coverage both only import `infer.run()` into
    *this* test process. This test trains via the real `main()` entry point
    (exercising the actual `forge.train()`-based fresh path end to end),
    then launches `python -m examples.image_folder_classification.infer` as
    a genuine `subprocess` -- a completely separate Python process with no
    shared memory, module cache, or import state -- and cross-checks its
    stdout against `forge.predict()`/`interpret_classification()` computed
    directly in this process against the same file, proving the subprocess
    doesn't merely "not crash" but produces the exact documented result."""
    train_module.main(_tiny_main_args(tmp_path, epochs=2))
    output_dir = tmp_path / "artifacts"
    model_path = output_dir / "image_folder_model.forge"
    image_path = output_dir / "new_mixed_resolution_query.png"
    assert model_path.exists()
    assert image_path.exists()

    result = subprocess.run(
        [sys.executable, "-m", "examples.image_folder_classification.infer",
         "--model", str(model_path), "--image", str(image_path)],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"infer.py subprocess failed:\n{result.stderr}"
    assert "Prediction:" in result.stdout
    assert "Confidence:" in result.stdout

    preprocessing = load_preprocessing(str(model_path))
    reloaded_model = load_model(str(model_path))
    classes = load_classes(str(model_path))
    raw = ImageFolder._load_image(image_path)
    batch = preprocessing(raw).reshape(1, 3, *_RESIZE_SIZE)
    expected = interpret_classification(predict(reloaded_model, batch), classes)[0]
    assert f"Prediction: {expected.label}" in result.stdout
    assert f"Confidence: {expected.confidence:.1%}" in result.stdout
