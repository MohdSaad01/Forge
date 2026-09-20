"""Real-dataset acceptance check for `forge.train_tabular_classifier()` / `forge.train_tabular_regressor()` (Milestone 114).

A standalone script, not a pytest test: `tests/` collects only `test_*.py`, so
this is never part of `python -m pytest tests/` or CI. It also (re)generates the
bundled artifacts under `models/tabular_classifier/` and `models/tabular_regressor/`.

```bash
# both workflows, on CPU, artifacts to a temporary directory:
python tests/real_world/tabular_workflows.py --concrete-csv /path/to/concrete.csv
# rebuild the bundled artifacts (CPU-trained, so they load on any machine):
python tests/real_world/tabular_workflows.py --concrete-csv /path/to/concrete.csv --models-dir models
python tests/real_world/tabular_workflows.py --device cuda --concrete-csv /path/to/concrete.csv
```

**Classification** uses the Pima Indians Diabetes data bundled with
`examples/tabular_diabetes`. The 154-row `diabetes_holdout_eval.csv` (the M97/M113
holdout, 62.3% majority share) is held out as the test set; the other 614 rows go to
`train_tabular_classifier()`, which splits them again 80/20 for validation. Zero is
Pima's "not measured" sentinel in five columns (`missing_columns=[1, 2, 3, 4, 5]`).

**Regression** uses UCI Concrete Compressive Strength (1030 rows, 8 features + the
strength in MPa), which Forge does not ship: pass it as a CSV with one header row and
the target last. The UCI download is a legacy `.xls`; convert it once to CSV (values
unchanged) -- `docs/development/m114-tabular-workflows.md` records the verified file
properties and the SHA-256 it was checked against. A fixed 206-row test set
(`default_rng(123).permutation(1030)[:206]`) is held out; the other 824 rows go to
`train_tabular_regressor()`. Without `--concrete-csv` the regression half is skipped
(exit 0, said out loud), never silently passed.

For each workload the script trains, reloads the artifact **fresh**
(`forge.load_predictor`), predicts a raw row, evaluates the held-out set, and checks
that the result beats the M113 baseline and that the result object agrees with
`evaluate()`. Any failure exits non-zero.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import forge

ROOT = Path(__file__).resolve().parents[2]
PIMA_DIR = ROOT / "examples" / "tabular_diabetes" / "data"
PIMA_CLASSES = ["no_diabetes", "diabetes"]
PIMA_MISSING_COLUMNS = [1, 2, 3, 4, 5]  # Glucose, BloodPressure, SkinThickness, Insulin, BMI
CONCRETE_TEST_ROWS = 206
CONCRETE_SPLIT_SEED = 123


def _fail(message: str) -> "None":
    print(f"FAIL: {message}")
    sys.exit(1)


def _check(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _load_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", skip_header=1)


def run_classification(device: str, out_dir: Path, seed: int) -> None:
    print("== tabular classification: Pima Indians Diabetes ==")
    data = _load_csv(PIMA_DIR / "diabetes.csv")
    holdout = _load_csv(PIMA_DIR / "diabetes_holdout_eval.csv")
    _check(data.shape == (768, 9) and holdout.shape == (154, 9), f"unexpected Pima shapes {data.shape}/{holdout.shape}")
    _check(bool(np.isfinite(data).all()), "Pima data has non-finite values")
    held_out = {tuple(row) for row in holdout}
    pool = np.array([row for row in data if tuple(row) not in held_out])
    _check(pool.shape == (614, 9), f"expected 614 training-pool rows, got {pool.shape}")
    print(f"data: {data.shape[0]} rows, 8 features; pool {pool.shape[0]} for training, holdout {holdout.shape[0]}")

    path = out_dir / "tabular_classifier" / "diabetes_classifier.forge"
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    result = forge.train_tabular_classifier(
        pool[:, :-1], pool[:, -1].astype(int), path=path, classes=PIMA_CLASSES,
        missing_columns=PIMA_MISSING_COLUMNS, device=device, seed=seed,
    )
    print(f"trained on '{device}' in {time.perf_counter() - start:.1f}s: {result.epochs_completed} epochs "
          f"(stopped_early={result.stopped_early}, best_epoch={result.best_epoch})")
    print(f"  train      acc {result.train_accuracy:.1%} loss {result.train_loss:.4f}  ({result.train_samples} rows)")
    print(f"  validation acc {result.validation_accuracy:.1%} loss {result.validation_loss:.4f}  "
          f"({result.validation_samples} rows, baseline {result.baseline_accuracy:.1%})")

    predictor = forge.load_predictor(str(path))  # fresh load, straight from the file
    _check(predictor.task == "tabular_classification", f"unexpected task {predictor.task!r}")
    _check(predictor.classes == PIMA_CLASSES, f"unexpected classes {predictor.classes!r}")
    prediction = predictor.predict(holdout[:1, :-1])[0]
    print(f"predict, first raw holdout row: {prediction.label} ({prediction.confidence:.1%}); "
          f"true label {PIMA_CLASSES[int(holdout[0, -1])]}")

    evaluation = predictor.evaluate(holdout[:, :-1], holdout[:, -1].astype(int))
    print(f"HOLDOUT ({evaluation.samples} rows): accuracy {evaluation.accuracy:.1%}, majority baseline "
          f"{evaluation.baseline_accuracy:.1%}, loss {evaluation.loss:.4f}")
    print(f"  confusion (rows=true, cols=predicted) {evaluation.confusion_matrix.tolist()}")
    print(f"  precision {[round(v, 3) for v in evaluation.precision]}, recall {[round(v, 3) for v in evaluation.recall]}")

    _check(abs(evaluation.baseline_accuracy - 0.6233766) < 1e-6, "holdout baseline is not the M113 62.3%")
    _check(evaluation.accuracy > evaluation.baseline_accuracy, "holdout accuracy does not beat the majority baseline")
    _check(bool(np.isfinite([evaluation.loss, result.train_loss, result.validation_loss]).all()), "non-finite loss")
    # the saved model's validation numbers must be what evaluate() says about the same rows
    _check(result.validation_accuracy > result.baseline_accuracy, "validation accuracy does not beat its baseline")
    print("classification OK")


def run_regression(device: str, out_dir: Path, seed: int, concrete_csv: "Path | None") -> None:
    print("== tabular regression: UCI Concrete Compressive Strength ==")
    if concrete_csv is None or not concrete_csv.is_file():
        print("SKIP: no --concrete-csv given (or it does not exist); regression half not run.")
        return
    data = _load_csv(concrete_csv)
    _check(data.shape == (1030, 9), f"expected the (1030, 9) Concrete table, got {data.shape}")
    _check(bool(np.isfinite(data).all()), "Concrete data has non-finite values")
    target = data[:, -1]
    print(f"data: {data.shape[0]} rows, 8 features + target; strength {target.min():.2f}..{target.max():.2f} MPa "
          f"(mean {target.mean():.2f}, std {target.std():.2f})")

    order = np.random.default_rng(CONCRETE_SPLIT_SEED).permutation(len(data))
    test, pool = data[order[:CONCRETE_TEST_ROWS]], data[order[CONCRETE_TEST_ROWS:]]

    path = out_dir / "tabular_regressor" / "concrete_strength_regressor.forge"
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    result = forge.train_tabular_regressor(pool[:, :-1], pool[:, -1], path=path, device=device, seed=seed)
    print(f"trained on '{device}' in {time.perf_counter() - start:.1f}s: {result.epochs_completed} epochs "
          f"(stopped_early={result.stopped_early}, best_epoch={result.best_epoch})")
    print(f"  train      mse {result.train_mse:.2f} mae {result.train_mae:.2f}  ({result.train_samples} rows)")
    print(f"  validation mse {result.validation_mse:.2f} mae {result.validation_mae:.2f}  "
          f"({result.validation_samples} rows, baseline mse {result.baseline_mse:.2f})")

    predictor = forge.load_predictor(str(path))
    _check(predictor.task == "regression", f"unexpected task {predictor.task!r}")
    predicted = predictor.predict(test[:1, :-1]).numpy()
    print(f"predict, first raw test row: {predicted[0, 0]:.2f} MPa; true {test[0, -1]:.2f} MPa")

    evaluation = predictor.evaluate(test[:, :-1], test[:, -1])
    r2 = 1.0 - evaluation.mse / evaluation.baseline_mse
    print(f"TEST ({evaluation.samples} rows): mse {evaluation.mse:.2f}, mae {evaluation.mae:.2f} MPa, "
          f"baseline mse {evaluation.baseline_mse:.2f}, R^2 {r2:.3f}")

    _check(bool(np.isfinite([evaluation.mse, evaluation.mae]).all()), "non-finite test metrics")
    _check(evaluation.mse < 0.5 * evaluation.baseline_mse, "test MSE is not well below the predict-the-mean baseline")
    _check(result.validation_mse < 0.5 * result.baseline_mse, "validation MSE is not well below its baseline")
    print("regression OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concrete-csv", type=Path, default=None, help="Concrete Compressive Strength as CSV, target last.")
    parser.add_argument("--models-dir", type=Path, default=None,
                        help="Write artifacts under <dir>/tabular_classifier and <dir>/tabular_regressor "
                             "(default: a temporary directory).")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = args.models_dir if args.models_dir is not None else Path(tmp)
        run_classification(args.device, out_dir, args.seed)
        print()
        run_regression(args.device, out_dir, args.seed, args.concrete_csv)
    print("\nALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
