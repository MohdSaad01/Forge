"""Milestone 97 tests: artifact-centric post-training evaluation --

```text
tabular_diabetes_model.forge (from Milestone 92's train.py)
    + unseen labeled CSV (dataset.make_datasets(seed=0)'s own held-out test split)
    -> evaluate.py's evaluate_artifact()
    -> forge.load_model() + forge.load_preprocessing() + forge.predict()
       + forge.training.Accuracy
    -> accuracy + majority-class baseline
```

Covers the real, reproduced Milestone 97 finding directly: `Trainer.
evaluate()` -- the obviously-named existing evaluation entry point -- has no
hook for a saved artifact's persisted preprocessing and silently returns a
badly wrong, worse-than-baseline accuracy on this dataset's raw rows,
because it predates `preprocessing=`/`load_preprocessing()` (Milestone 71).
`evaluate.py::evaluate_artifact()` is the correct composition instead
(`load_model()` + `load_preprocessing()` + `forge.predict()` + a
`forge.training.Metric`), requiring no Forge framework changes -- only a
documentation clarification on `Trainer.evaluate()`'s own docstring.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forge
from forge.data import DataLoader, TensorDataset
from forge.exceptions import DataError, TrainerError
from forge.nn import CrossEntropyLoss
from forge.optim import Adam
from forge.training import Accuracy, Trainer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.tabular_diabetes.dataset import load_raw, make_datasets  # noqa: E402
from examples.tabular_diabetes.evaluate import evaluate_artifact, load_labeled_csv, main  # noqa: E402
from examples.tabular_diabetes.train import main as train_main  # noqa: E402

_SMALL = ["--batch-size", "16"]
_HOLDOUT_CSV = _REPO_ROOT / "examples" / "tabular_diabetes" / "data" / "diabetes_holdout_eval.csv"


def _write_csv(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"f{i}" for i in range(X.shape[1])] + ["Outcome"])
        for row, label in zip(X, y):
            writer.writerow(list(row) + [int(label)])


@pytest.fixture()
def small_artifact(tmp_path):
    out_dir = tmp_path / "artifacts"
    train_main(_SMALL + ["--epochs", "3", "--seed", "11", "--output-dir", str(out_dir)])
    return out_dir / "tabular_diabetes_model.forge"


@pytest.fixture()
def small_eval_csv(tmp_path):
    X, y = load_raw()
    path = tmp_path / "eval.csv"
    _write_csv(path, X[:40], y[:40])
    return path


# -- the committed held-out CSV itself -------------------------------------


def test_holdout_csv_is_the_seed_zero_test_split():
    _, _, test_ds, _ = make_datasets(seed=0)
    X, y = load_raw()
    expected_X = X[test_ds.indices]
    expected_y = y[test_ds.indices]

    actual_X, actual_y = load_labeled_csv(str(_HOLDOUT_CSV))
    assert actual_X.shape == expected_X.shape
    np.testing.assert_allclose(actual_X, expected_X, atol=1e-4)
    np.testing.assert_array_equal(actual_y, expected_y)


# -- evaluate_artifact(): correctness ---------------------------------------


def test_evaluate_artifact_returns_a_useful_summary(small_artifact, small_eval_csv):
    summary = evaluate_artifact(str(small_artifact), str(small_eval_csv))
    assert summary.samples == 40
    assert 0.0 <= summary.accuracy <= 1.0
    assert 0.0 <= summary.baseline_accuracy <= 1.0
    assert summary.improvement_pp == pytest.approx((summary.accuracy - summary.baseline_accuracy) * 100.0)


def test_evaluate_artifact_reuses_persisted_preprocessing(small_artifact, small_eval_csv):
    """Reproduces the exact finding this milestone made: evaluating with the
    artifact's own persisted preprocessing must differ from evaluating on
    the same raw rows with no preprocessing applied at all -- otherwise
    preprocessing is not actually load-bearing for this composition."""
    with_preprocessing = evaluate_artifact(str(small_artifact), str(small_eval_csv))

    X, y = load_labeled_csv(str(small_eval_csv))
    model = forge.load_model(str(small_artifact))
    raw_output = forge.predict(model, forge.Tensor(X))  # no preprocessing applied
    metric = Accuracy()
    metric.update(raw_output, y)
    without_preprocessing_accuracy = metric.compute()

    assert with_preprocessing.accuracy != pytest.approx(without_preprocessing_accuracy)


def test_evaluate_artifact_computes_majority_class_baseline_from_the_eval_data(small_artifact, small_eval_csv):
    X, y = load_labeled_csv(str(small_eval_csv))
    expected_baseline = float((y == int(np.bincount(y).argmax())).mean())
    summary = evaluate_artifact(str(small_artifact), str(small_eval_csv))
    assert summary.baseline_accuracy == pytest.approx(expected_baseline)


# -- no mutation / immutability / determinism --------------------------------


def test_evaluate_artifact_does_not_modify_the_artifact_file(small_artifact, small_eval_csv):
    before = hashlib.sha256(small_artifact.read_bytes()).hexdigest()
    evaluate_artifact(str(small_artifact), str(small_eval_csv))
    after = hashlib.sha256(small_artifact.read_bytes()).hexdigest()
    assert before == after


def test_evaluate_artifact_does_not_change_model_parameters(small_artifact, small_eval_csv):
    before = [p.numpy().copy() for p in forge.load_model(str(small_artifact)).parameters()]
    evaluate_artifact(str(small_artifact), str(small_eval_csv))
    after = [p.numpy().copy() for p in forge.load_model(str(small_artifact)).parameters()]
    for b, a in zip(before, after):
        np.testing.assert_array_equal(b, a)


def test_evaluate_artifact_is_deterministic_across_repeated_calls(small_artifact, small_eval_csv):
    first = evaluate_artifact(str(small_artifact), str(small_eval_csv))
    second = evaluate_artifact(str(small_artifact), str(small_eval_csv))
    third = evaluate_artifact(str(small_artifact), str(small_eval_csv))
    assert first.accuracy == second.accuracy == third.accuracy
    assert first.baseline_accuracy == second.baseline_accuracy == third.baseline_accuracy


# -- fresh process -----------------------------------------------------------


def test_evaluate_artifact_agrees_with_a_genuinely_separate_process(small_artifact, small_eval_csv):
    expected = evaluate_artifact(str(small_artifact), str(small_eval_csv))

    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(_REPO_ROOT)!r})\n"
        "from examples.tabular_diabetes.evaluate import evaluate_artifact\n"
        f"summary = evaluate_artifact({str(small_artifact)!r}, {str(small_eval_csv)!r})\n"
        "print(f'accuracy={summary.accuracy:.6f} baseline={summary.baseline_accuracy:.6f}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"
    assert f"accuracy={expected.accuracy:.6f}" in result.stdout
    assert f"baseline={expected.baseline_accuracy:.6f}" in result.stdout


# -- invalid inputs -----------------------------------------------------------


def test_evaluate_artifact_rejects_wrong_feature_count(small_artifact, tmp_path):
    bad_csv = tmp_path / "bad.csv"
    _write_csv(bad_csv, np.zeros((3, 5), dtype=np.float32), np.zeros(3, dtype=np.int64))
    with pytest.raises(DataError):
        evaluate_artifact(str(small_artifact), str(bad_csv))


def test_evaluate_artifact_rejects_mismatched_batch_via_metric(small_artifact, small_eval_csv):
    # A CSV with zero rows: predict() runs but Accuracy.compute() must refuse
    # a no-samples-seen result rather than fabricate one.
    empty_csv = small_eval_csv.parent / "empty.csv"
    X, y = load_labeled_csv(str(small_eval_csv))
    _write_csv(empty_csv, X[:0], y[:0])
    with pytest.raises(TrainerError):
        evaluate_artifact(str(small_artifact), str(empty_csv))


# -- CLI -----------------------------------------------------------------


def test_main_prints_a_useful_summary(small_artifact, small_eval_csv, capsys):
    main(["--model", str(small_artifact), "--data", str(small_eval_csv)])
    out = capsys.readouterr().out
    assert "Accuracy:" in out
    assert "Baseline accuracy:" in out
    assert "Improvement:" in out


# -- production acceptance: full training run, real held-out data -----------


def test_full_workflow_beats_majority_baseline_on_the_committed_holdout_set(tmp_path):
    out_dir = tmp_path / "artifacts"
    train_main(["--epochs", "60", "--seed", "0", "--output-dir", str(out_dir)])
    model_path = out_dir / "tabular_diabetes_model.forge"

    summary = evaluate_artifact(str(model_path), str(_HOLDOUT_CSV))
    assert summary.samples == 154
    assert summary.baseline_accuracy == pytest.approx(0.6233766233766234, abs=1e-6)
    assert summary.accuracy > summary.baseline_accuracy


# -- the Milestone 97 finding, reproduced directly ---------------------------


def test_trainer_evaluate_without_preprocessing_is_silently_worse_than_baseline(small_artifact, small_eval_csv):
    """Regression-proofs the exact claim in `Trainer.evaluate()`'s own
    updated docstring: feeding raw (unpreprocessed) rows through
    `Trainer.evaluate()` produces no error, but a materially different
    (here: worse) accuracy than `evaluate_artifact()`'s own, correctly-
    preprocessed result."""
    correct = evaluate_artifact(str(small_artifact), str(small_eval_csv))

    X, y = load_labeled_csv(str(small_eval_csv))
    model = forge.load_model(str(small_artifact))
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters())  # constructed only because Trainer requires one; never stepped
    trainer = Trainer(model, loss_fn, optimizer, metrics=[Accuracy()], verbose=False)
    loader = DataLoader(TensorDataset(forge.Tensor(X), forge.Tensor(y)), batch_size=16)
    result = trainer.evaluate(loader)  # no preprocessing applied -- exactly the documented trap

    assert result.metrics["accuracy"] != pytest.approx(correct.accuracy)
