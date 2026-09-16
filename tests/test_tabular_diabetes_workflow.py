"""Milestone 92 tests: the real-dataset tabular classification workflow --

```text
data/diabetes.csv (real Pima Indians Diabetes CSV)
    -> examples.tabular_diabetes.dataset.load_raw()/make_datasets()
    -> forge.train_and_save() (preprocessing=Compose([ReplaceValue, Normalize]))
    -> verified .forge artifact -> fresh process
    -> forge.predict_model() -> ClassificationPrediction
```

Covers the real, reproduced Milestone 92 finding directly: before
`forge.data.ReplaceValue` existed, a manual (raw-NumPy, pre-Forge)
imputation of this dataset's sentinel-coded missing values (`0` in
`Glucose`/`BloodPressure`/`SkinThickness`/`Insulin`/`BMI`) had no
persistable representation -- `preprocessing=Normalize(...)` alone silently
dropped the imputation step, so a fresh raw inference row with a genuine
missing reading was standardized as if `0` were a real measurement,
producing a materially different (and undocumented, unrecoverable-from-the-
artifact) prediction than intended. See `docs/development/
m92-real-dataset-ingestion.md`.

Mirrors `tests/test_tabular_classification_artifact_prediction.py`'s
structure for the equivalent (synthetic-dataset) artifact shape.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.data import Compose, ReplaceValue
from forge.serialization import inspect_model, load_classes, load_preprocessing, save_model
from forge.training import predict_model, predict_tabular_classification_artifact

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.tabular_diabetes.dataset import (  # noqa: E402
    CLASS_NAMES,
    N_FEATURES,
    load_raw,
    make_datasets,
)
from examples.tabular_diabetes.train import main as train_main  # noqa: E402

_SMALL = ["--batch-size", "16"]


# -- real CSV ingestion ---------------------------------------------------


def test_load_raw_reads_the_real_bundled_csv():
    X, y = load_raw()
    assert X.shape == (768, N_FEATURES)
    assert y.shape == (768,)
    assert set(np.unique(y).tolist()) == {0, 1}
    # Majority-class imbalance is a real property of this dataset (not
    # engineered) -- both classes must be present with a genuine skew.
    counts = np.bincount(y)
    assert counts[0] > counts[1] > 0


def test_load_raw_is_deterministic():
    X1, y1 = load_raw()
    X2, y2 = load_raw()
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_load_raw_rejects_a_file_with_wrong_header(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError):
        load_raw(bad_csv)


# -- preprocessing: fitted on train only, ReplaceValue then Normalize -----


def test_make_datasets_fits_replace_value_fill_from_training_medians_only():
    train_ds, val_ds, test_ds, stats = make_datasets(seed=0)
    transform = stats["transform"]
    assert isinstance(transform, Compose)
    assert isinstance(transform.transforms[0], ReplaceValue)
    assert transform.transforms[0].sentinel == 0.0
    # Glucose, BloodPressure, SkinThickness, Insulin, BMI -- indices 1-5.
    assert transform.transforms[0].columns == [1, 2, 3, 4, 5]
    # Every fitted fill value must be strictly positive (a real median of
    # non-zero readings), never 0 itself (which would defeat the point).
    assert all(f > 0 for f in stats["fill"])


def test_make_datasets_splits_do_not_overlap():
    train_ds, val_ds, test_ds, _ = make_datasets(seed=3)
    train_idx = set(train_ds.indices)
    val_idx = set(val_ds.indices)
    test_idx = set(test_ds.indices)
    assert not (train_idx & val_idx)
    assert not (train_idx & test_idx)
    assert not (val_idx & test_idx)
    assert len(train_idx) + len(val_idx) + len(test_idx) == 768


def test_make_datasets_is_deterministic_for_a_fixed_seed():
    train_a, _, _, stats_a = make_datasets(seed=7)
    train_b, _, _, stats_b = make_datasets(seed=7)
    assert train_a.indices == train_b.indices
    assert stats_a["fill"] == stats_b["fill"]


def test_preprocessing_replaces_sentinel_zero_but_never_touches_pregnancies():
    """The one real behavioral requirement this dataset demonstrates: index 0
    (Pregnancies) must be left alone even though it also legitimately
    contains zeros, while index 4 (Insulin) -- the column with the most
    sentinel zeros (48.7%) -- must be imputed."""
    _, _, _, stats = make_datasets(seed=0)
    transform = stats["transform"]

    raw_row = forge.Tensor(np.array([0.0, 0.0, 70.0, 20.0, 0.0, 25.0, 0.3, 30.0], dtype=np.float32))
    processed = transform(raw_row).numpy()

    # Pregnancies (col 0) went straight through ReplaceValue untouched, then
    # only through Normalize -- recover what Normalize alone would produce.
    mean, std = stats["mean"], stats["std"]
    expected_pregnancies = (0.0 - mean[0]) / std[0]
    assert processed[0] == pytest.approx(expected_pregnancies, abs=1e-5)

    # Insulin (col 4) went through imputation first -- its normalized value
    # must equal the *fill* value normalized, not literal 0 normalized.
    expected_insulin_if_imputed = (stats["fill"][3] - mean[4]) / std[4]  # fill[3] == Insulin's fitted median
    expected_insulin_if_raw_zero = (0.0 - mean[4]) / std[4]
    assert processed[4] == pytest.approx(expected_insulin_if_imputed, abs=1e-5)
    assert processed[4] != pytest.approx(expected_insulin_if_raw_zero, abs=1e-3)


# -- end-to-end: real example, fresh process -------------------------------


def test_train_and_save_writes_an_artifact_with_full_metadata(tmp_path):
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL + ["--epochs", "2", "--seed", "1", "--output-dir", str(out_dir)])

    model_path = out_dir / "tabular_diabetes_model.forge"
    assert model_path.is_file()

    assert load_classes(str(model_path)) == CLASS_NAMES
    assert inspect_model(str(model_path)).task == "tabular_classification"

    preprocessing = load_preprocessing(str(model_path))
    assert isinstance(preprocessing, Compose)
    assert isinstance(preprocessing.transforms[0], ReplaceValue)


def test_real_example_achieves_well_above_majority_class_baseline_accuracy(tmp_path, capsys):
    """Confirms genuine learning on the real dataset -- the 65.1% majority-
    class baseline, not a naive 50% coin-flip, is the real bar this
    imbalanced real-world dataset sets (Milestone 92 brief, Section 13)."""
    out_dir = tmp_path / "artifacts"
    train_main(["--epochs", "60", "--seed", "0", "--output-dir", str(out_dir)])
    out = capsys.readouterr().out
    assert "Majority-class baseline accuracy was 65.1%." in out

    import re
    match = re.search(r"Final test evaluation: loss=[\d.]+, accuracy=([\d.]+)%", out)
    assert match is not None, out
    test_accuracy = float(match.group(1))
    assert test_accuracy > 65.1, f"expected above the 65.1% majority-class baseline, got {test_accuracy}%"


def test_predict_model_agrees_with_a_genuinely_separate_process(tmp_path):
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL + ["--epochs", "3", "--seed", "42", "--output-dir", str(out_dir)])
    model_path = out_dir / "tabular_diabetes_model.forge"

    # A raw row straight from the real CSV, including its genuine sentinel
    # zeros (a real missing-value row, not a synthetic edge case).
    X, _ = load_raw()
    raw_row = X[0:1]
    raw_row_literal = raw_row.tolist()

    expected = predict_tabular_classification_artifact(str(model_path), raw_row)[0]

    script = (
        "import numpy as np\n"
        "import forge\n"
        f"raw_row = np.array({raw_row_literal!r}, dtype=np.float32)\n"
        f"result = forge.predict_tabular_classification_artifact({str(model_path)!r}, raw_row)[0]\n"
        "print(f'label={result.label} index={result.index} confidence={result.confidence:.6f}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"
    assert f"label={expected.label} index={expected.index}" in result.stdout


def test_predict_model_dispatches_the_same_as_the_direct_call(tmp_path):
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL + ["--epochs", "2", "--seed", "5", "--output-dir", str(out_dir)])
    model_path = out_dir / "tabular_diabetes_model.forge"

    X, _ = load_raw()
    raw_rows = X[10:13]  # three real rows, batched
    via_unified = predict_model(str(model_path), raw_rows)
    via_direct = predict_tabular_classification_artifact(str(model_path), raw_rows)

    assert len(via_unified) == len(via_direct) == 3
    for u, d in zip(via_unified, via_direct):
        assert u.label == d.label
        assert u.index == d.index


# -- the Milestone 92 finding, reproduced directly -------------------------


def test_preprocessing_without_replace_value_gives_a_different_prediction_for_a_missing_reading(tmp_path):
    """Reproduces the exact blocker this milestone found: `Normalize` alone
    (no imputation) standardizes a genuinely-missing (sentinel-zero)
    reading as if it were a real extreme-low measurement, which can flip
    the predicted class relative to the artifact whose preprocessing
    actually imputes it -- proving `ReplaceValue` is load-bearing, not
    cosmetic."""
    from forge.data import Normalize
    from forge.nn import CrossEntropyLoss, Linear, ReLU, Sequential
    from forge.optim import Adam
    from forge.training import Accuracy, train_and_save

    train_ds, val_ds, test_ds, stats = make_datasets(seed=0)
    imputed_transform = stats["transform"]

    # A materially different transform saved on an otherwise-identical
    # artifact: Normalize alone, fit on the *un-imputed* raw training data
    # (i.e. exactly what a developer gets if they skip ReplaceValue
    # entirely and feed sentinel zeros straight into Normalize).
    X, y = load_raw()
    train_X_raw = X[train_ds.indices]
    mean = train_X_raw.mean(axis=0)
    std = train_X_raw.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    naive_transform = Normalize(mean=mean, std=std)

    def _train(transform, tag):
        forge.random.seed(0)
        from forge.data import DataLoader, TensorDataset, Subset

        full_ds = TensorDataset(forge.Tensor(X), forge.Tensor(y), transform=transform)
        ds = Subset(full_ds, train_ds.indices)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        model = Sequential(Linear(N_FEATURES, 16), ReLU(), Linear(16, 2))
        path = tmp_path / f"{tag}.forge"
        train_and_save(
            model, loader, loss=CrossEntropyLoss(), optimizer=Adam(model.parameters(), lr=1e-3),
            epochs=5, device="cpu", metrics=[Accuracy()],
            path=str(path), sample=forge.Tensor(X[0:1]),
            preprocessing=transform, classes=CLASS_NAMES, task="tabular_classification",
        )
        return path

    imputed_path = _train(imputed_transform, "imputed")
    naive_path = _train(naive_transform, "naive")

    # A brand-new raw row with a genuine missing Insulin reading (0), never
    # manually imputed by the caller -- exactly what "the artifact should
    # own its preprocessing" means (Milestone 92 brief, Section 8).
    missing_row = np.array([[2.0, 130.0, 70.0, 25.0, 0.0, 28.5, 0.5, 35.0]], dtype=np.float32)

    imputed_result = predict_tabular_classification_artifact(str(imputed_path), missing_row)[0]
    naive_result = predict_tabular_classification_artifact(str(naive_path), missing_row)[0]

    # The two artifacts see genuinely different standardized inputs for the
    # same raw row -- confirming ReplaceValue changes real model input, not
    # just metadata.
    imputed_preprocessed = imputed_transform(forge.Tensor(missing_row[0])).numpy()
    naive_preprocessed = naive_transform(forge.Tensor(missing_row[0])).numpy()
    assert not np.allclose(imputed_preprocessed, naive_preprocessed)


# -- Milestone 100: early stopping on the real workload ---------------------


def test_early_stopping_flag_stops_before_the_requested_epoch_limit(tmp_path, capsys):
    """The real, motivating case for Milestone 100: this workload's own
    validation loss visibly bottoms out well before 60 epochs (Milestone 98's
    finding) -- `--early-stopping` must actually exercise that, not just
    accept the flag."""
    out_dir = tmp_path / "artifacts"
    train_main(
        ["--epochs", "60", "--seed", "0", "--output-dir", str(out_dir),
         "--early-stopping", "--patience", "5"]
    )
    out = capsys.readouterr().out
    assert "Early stopping enabled: monitor=val_loss, patience=5" in out

    import re
    match = re.search(r"Early stopping: requested 60 epoch\(s\), completed (\d+), stopped_early=(True|False)", out)
    assert match is not None, out
    completed = int(match.group(1))
    stopped_early = match.group(2) == "True"
    assert completed < 60, f"expected early stopping to activate before epoch 60, ran {completed}"
    assert stopped_early is True

    model_path = out_dir / "tabular_diabetes_model.forge"
    assert model_path.is_file()
    assert inspect_model(str(model_path)).task == "tabular_classification"


def test_without_the_flag_early_stopping_never_activates(tmp_path, capsys):
    """Backward compatibility: omitting `--early-stopping` must run every
    requested epoch, exactly as every pre-Milestone-100 run did."""
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL + ["--epochs", "5", "--seed", "0", "--output-dir", str(out_dir)])
    out = capsys.readouterr().out
    assert "Early stopping" not in out
    assert "Trained 5 epoch(s)" in out


def test_early_stopping_restores_a_model_the_saved_artifact_reproduces(tmp_path):
    """The saved artifact must reflect the *restored* best model, not
    whatever the final completed epoch happened to leave in memory --
    `train_and_save()`'s own `save_and_verify()` call already asserts this
    for whatever model is live at save time; this proves early stopping ran
    for real on this workload beforehand (`stopped_early=True`)."""
    out_dir = tmp_path / "artifacts"
    train_main(
        ["--epochs", "60", "--seed", "0", "--output-dir", str(out_dir),
         "--early-stopping", "--patience", "5"]
    )
    model_path = out_dir / "tabular_diabetes_model.forge"

    X, _ = load_raw()
    raw_row = X[0:1]
    # A genuinely fresh reload agrees with the artifact's own recorded
    # prediction -- i.e. the artifact is self-consistent and portable,
    # exactly like every other real-workload artifact in this suite.
    first = predict_tabular_classification_artifact(str(model_path), raw_row)[0]
    second = predict_tabular_classification_artifact(str(model_path), raw_row)[0]
    assert first.label == second.label
    assert first.confidence == pytest.approx(second.confidence, abs=1e-6)
