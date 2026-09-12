"""Milestone 83 tests: the end-to-end regression artifact workflow --

```text
examples.regression.dataset/model -> forge.train_and_save() -> verified .forge
    artifact -> fresh process -> forge.predict_tensor_artifact()
    -> numerical regression prediction
```

This is the literal M83 acceptance test (see the milestone brief's Section
11/19): a genuinely separate OS process, given only the `.forge` file this
test's own process trained and saved, must produce a numerical prediction
that agrees with this process's own prediction on the same raw input --
proving the artifact (model + persisted `Normalize` preprocessing) is
actually portable, not merely "same process, reloaded." Mirrors
`tests/test_artifact_inference.py::
test_predict_artifact_works_from_a_genuinely_separate_process`'s structure
for the image-classification artifact shape.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.training import predict_tensor_artifact

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.regression.dataset import N_FEATURES, generate_raw  # noqa: E402
from examples.regression.train import main as train_main  # noqa: E402

_SMALL = ["--n-train", "150", "--n-val", "30", "--n-test", "30", "--batch-size", "16"]


def test_train_and_save_writes_a_regression_artifact_with_persisted_preprocessing(tmp_path):
    """The fresh (non-`--resume`) path trains through `forge.train_and_save()`
    and must save the fitted `Normalize` transform alongside the model --
    without it, a fresh process holding only the `.forge` file would have no
    way to standardize a brand-new raw feature vector (see
    `forge/training/inference.py::predict_tensor_artifact()`).
    """
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL + ["--epochs", "2", "--seed", "1", "--output-dir", str(out_dir)])

    model_path = out_dir / "regression_model.forge"
    assert model_path.is_file()

    from forge.serialization import load_classes, load_preprocessing

    preprocessing = load_preprocessing(str(model_path))
    assert preprocessing is not None, "regression artifact must carry its fitted Normalize transform"
    # A regression model has no class vocabulary -- must not be fabricated.
    assert load_classes(str(model_path)) is None


def test_predict_tensor_artifact_agrees_with_a_genuinely_separate_process(tmp_path):
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL + ["--epochs", "3", "--seed", "42", "--output-dir", str(out_dir)])
    model_path = out_dir / "regression_model.forge"

    # A brand-new raw feature vector, never part of train/val/test (seed
    # deliberately outside the training run's own seed-derived data stream).
    raw_x, _ = generate_raw(1, seed=99999)
    raw_x_literal = raw_x.tolist()

    expected = predict_tensor_artifact(str(model_path), raw_x)
    expected_value = float(expected.numpy()[0, 0])

    script = (
        "import numpy as np\n"
        "import forge\n"
        f"raw_x = np.array({raw_x_literal!r}, dtype=np.float32)\n"
        f"prediction = forge.predict_tensor_artifact({str(model_path)!r}, raw_x)\n"
        "print(f'prediction={prediction.numpy()[0, 0]:.8f}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"
    assert f"prediction={expected_value:.8f}" in result.stdout


def test_resume_branch_also_saves_preprocessing_and_stays_predictable(tmp_path):
    """`--resume` uses `save_and_verify()` directly (not `train_and_save()`)
    -- must not regress to omitting `preprocessing=` just because it takes a
    different code path than the fresh branch.
    """
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL + ["--epochs", "2", "--seed", "3", "--output-dir", str(out_dir)])
    checkpoint_path = out_dir / "regression_checkpoint.forge"
    train_main(_SMALL + ["--epochs", "1", "--seed", "3", "--resume", str(checkpoint_path), "--output-dir", str(out_dir)])

    model_path = out_dir / "regression_model.forge"
    from forge.serialization import load_preprocessing

    assert load_preprocessing(str(model_path)) is not None

    raw_x, _ = generate_raw(1, seed=54321)
    prediction = predict_tensor_artifact(str(model_path), raw_x)
    assert prediction.shape == (1, 1)


def test_predict_tensor_artifact_standardizes_raw_input_the_same_way_training_did(tmp_path):
    """Numerical-equivalence check against a hand-computed reference: a raw
    feature vector run through `predict_tensor_artifact()` must equal the
    same vector standardized with the training split's own recorded
    mean/std, then passed through the reloaded model directly.
    """
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL + ["--epochs", "2", "--seed", "7", "--output-dir", str(out_dir)])
    model_path = out_dir / "regression_model.forge"

    from examples.regression.dataset import make_datasets
    from forge.serialization import load_model

    _, _, _, stats = make_datasets(150, 30, 30, seed=7)
    raw_x, _ = generate_raw(4, seed=2024)

    reloaded = load_model(str(model_path))
    standardized = (raw_x - stats["mean"]) / stats["std"]
    with forge.no_grad():
        manual = reloaded(forge.Tensor(standardized.astype(np.float32))).numpy()

    via_api = predict_tensor_artifact(str(model_path), raw_x).numpy()
    np.testing.assert_allclose(manual, via_api, atol=1e-5)
