"""Real-data acceptance script for the CSV tabular workflow (Milestone 118): CSV vs NumPy, arrays to CLI.

```text
python tests/real_world/csv_workflows.py                          # Pima only (bundled)
python tests/real_world/csv_workflows.py --cadata /path/cadata.txt   # + California housing
python tests/real_world/csv_workflows.py --device cuda --cadata ...
```

For each real dataset it shows that CSV is only a boundary: `forge.data.load_csv()` gives arrays *bit-identical* to
the NumPy reference, the unchanged `train_tabular_*()` functions then produce *identical* parameters and results, and
`forge model evaluate MODEL DATA.csv --target COLUMN` prints the same JSON as the `.npy` form of the same rows.

- **Pima diabetes** -- bundled (`examples/tabular_diabetes/data`): the 614-row training pool (the rows of
  `diabetes.csv` that are not in the 154-row `diabetes_holdout_eval.csv`) trains a classifier with
  `missing_columns=[1..5]`; the holdout CSV is evaluated through the CLI.
- **California housing** -- StatLib `cadata.txt` (`https://lib.stat.cmu.edu/datasets/houses.zip`, 20,640 rows, target
  median house value in dollars, *first* column): converted to a controlled CSV (header names, `repr(float)` values, no
  value changed), 3,000 training rows / 2,000 held-out rows from `default_rng(123).permutation` (the M115/M116/M117
  split), trained with `target_transform="standardize"`, evaluated in native dollars. Skipped, saying so, without
  `--cadata` (the file is not shipped).

Exit status 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

import forge
from forge.cli.main import main as cli_main

ROOT = Path(__file__).resolve().parents[2]
PIMA_DIR = ROOT / "examples" / "tabular_diabetes" / "data"
PIMA_CLASSES = ["no_diabetes", "diabetes"]
HOUSING_COLUMNS = [
    "median_house_value", "median_income", "housing_median_age", "total_rooms", "total_bedrooms",
    "population", "households", "latitude", "longitude",
]


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def _check(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _parameters_sha(path: Path) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(forge.load_model(str(path)).named_parameters(), key=lambda kv: kv[0]):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(parameter.numpy()).tobytes())
    return digest.hexdigest()


def _cli_json(*argv) -> dict:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(["model", "evaluate", *map(str, argv), "--json"])
    _check(code == 0, f"forge model evaluate {' '.join(map(str, argv))} exited {code}: {err.getvalue()}")
    return json.loads(out.getvalue())


def _write_csv(path: Path, header: "list[str]", rows) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([v if isinstance(v, str) else repr(float(v)) for v in row])


def _same_training(reference, from_csv, reference_path: Path, csv_path: Path, label: str) -> None:
    import dataclasses

    for field in dataclasses.fields(reference):
        if field.name in ("history", "model", "artifact_path", "target_transform"):
            continue
        _check(getattr(reference, field.name) == getattr(from_csv, field.name), f"{label}: result field {field.name} differs")
    _check(_parameters_sha(reference_path) == _parameters_sha(csv_path), f"{label}: trained parameters differ")
    print(f"  {label}: every result field and every trained parameter identical (SHA-256 {_parameters_sha(csv_path)[:16]}...)")


def run_pima(device: str, out_dir: Path, seed: int) -> None:
    print("== Pima diabetes: CSV -> train_tabular_classifier() -> forge model evaluate ==")
    full = PIMA_DIR / "diabetes.csv"
    holdout_path = PIMA_DIR / "diabetes_holdout_eval.csv"
    reference = np.genfromtxt(full, delimiter=",", skip_header=1)
    holdout = np.genfromtxt(holdout_path, delimiter=",", skip_header=1)
    held = {tuple(r) for r in holdout}
    with open(full, newline="") as fh:
        table = list(csv.reader(fh))
    pool_rows = [r for r in table[1:] if tuple(float(v) for v in r) not in held]
    pool_csv = out_dir / "pima_pool.csv"
    with open(pool_csv, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(table[0])
        writer.writerows(pool_rows)
    pool_ref = np.array([[float(v) for v in r] for r in pool_rows])
    _check(pool_ref.shape == (614, 9), f"expected a 614-row pool, got {pool_ref.shape}")

    start = time.perf_counter()
    X_all, y_all = forge.data.load_csv(full, target="Outcome", labels=True)
    parse_ms = (time.perf_counter() - start) * 1000
    _check(np.array_equal(X_all, reference[:, :-1]) and np.array_equal(y_all, reference[:, -1].astype(int)),
           "Pima CSV arrays differ from the genfromtxt reference")
    print(f"  diabetes.csv {full.stat().st_size / 1e3:.1f} kB, {X_all.shape[0]} rows parsed in {parse_ms:.1f} ms; "
          "X and y bit-identical to np.genfromtxt")

    kwargs = dict(classes=PIMA_CLASSES, missing_columns=[1, 2, 3, 4, 5], device=device, seed=seed)
    ref_path, csv_path = out_dir / "pima_numpy.forge", out_dir / "pima_csv.forge"
    reference_run = forge.train_tabular_classifier(pool_ref[:, :-1], pool_ref[:, -1].astype(int), path=ref_path, **kwargs)
    X, y = forge.data.load_csv(pool_csv, target="Outcome", labels=True)
    start = time.perf_counter()
    csv_run = forge.train_tabular_classifier(X, y, path=csv_path, **kwargs)
    print(f"  trained on '{device}' in {time.perf_counter() - start:.1f}s: {csv_run.epochs_completed} epochs, validation "
          f"accuracy {csv_run.validation_accuracy:.1%} vs {csv_run.baseline_accuracy:.1%} majority baseline")
    if device == "cpu":
        _same_training(reference_run, csv_run, ref_path, csv_path, "CSV vs NumPy training")
    _check(csv_run.validation_accuracy > csv_run.baseline_accuracy, "validation accuracy does not beat the baseline")

    np.save(out_dir / "hold_X.npy", holdout[:, :-1])
    np.save(out_dir / "hold_y.npy", holdout[:, -1].astype(int))
    from_csv = _cli_json(csv_path, holdout_path, "--target", "Outcome", "--device", device)
    from_npy = _cli_json(csv_path, out_dir / "hold_X.npy", out_dir / "hold_y.npy", "--device", device)
    _check(from_csv == from_npy, "CLI evaluation of the CSV differs from the .npy evaluation")
    api = forge.load_predictor(str(csv_path), device=device).evaluate(holdout[:, :-1], holdout[:, -1].astype(int))
    _check(from_csv["confusion_matrix"] == api.confusion_matrix.tolist(), "CLI confusion matrix differs from the API's")
    _check(abs(from_csv["baseline_accuracy"] - 96 / 154) < 1e-9, "holdout baseline is not the M113 62.3%")
    print(f"  holdout ({from_csv['samples']} rows) through the CLI: accuracy {from_csv['accuracy']:.1%}, baseline "
          f"{from_csv['baseline_accuracy']:.1%}; JSON identical for DATA.csv --target and X.npy y.npy")
    print("Pima OK")


def run_housing(device: str, out_dir: Path, seed: int, cadata: "Path | None") -> None:
    print("== California housing: CSV -> train_tabular_regressor(target_transform='standardize') -> forge model evaluate ==")
    if cadata is None or not cadata.is_file():
        print("SKIP: no --cadata given (or it does not exist); housing half not run.")
        return
    rows = [line.split() for line in cadata.read_text(encoding="latin-1").splitlines()]
    data = np.array([[float(t) for t in r] for r in rows if len(r) == 9 and all(t[0] in "+-.0123456789" for t in r)])
    _check(data.shape == (20640, 9), f"expected 20,640 x 9 rows, got {data.shape}")
    full_csv = out_dir / "housing_full.csv"
    _write_csv(full_csv, HOUSING_COLUMNS, data)

    start = time.perf_counter()
    X_all, y_all = forge.data.load_csv(full_csv, target="median_house_value")
    parse_ms = (time.perf_counter() - start) * 1000
    _check(np.array_equal(X_all, data[:, 1:]) and np.array_equal(y_all, data[:, 0]), "housing CSV arrays differ from cadata")
    print(f"  {full_csv.stat().st_size / 1e6:.2f} MB, {X_all.shape[0]} rows parsed in {parse_ms:.0f} ms; X and y bit-identical "
          "to the cadata.txt values (the target is the FIRST column and is selected by name)")

    order = np.random.default_rng(123).permutation(len(data))
    train, test = data[order[:3000]], data[order[3000:5000]]
    train_csv, test_csv = out_dir / "housing_train.csv", out_dir / "housing_test.csv"
    _write_csv(train_csv, HOUSING_COLUMNS, train)
    _write_csv(test_csv, HOUSING_COLUMNS, test)

    kwargs = dict(target_transform="standardize", device=device, seed=seed)
    ref_path, csv_path = out_dir / "housing_numpy.forge", out_dir / "housing_csv.forge"
    reference_run = forge.train_tabular_regressor(train[:, 1:], train[:, 0], path=ref_path, **kwargs)
    X, y = forge.data.load_csv(train_csv, target="median_house_value")
    start = time.perf_counter()
    csv_run = forge.train_tabular_regressor(X, y, path=csv_path, **kwargs)
    print(f"  trained on '{device}' in {time.perf_counter() - start:.1f}s: {csv_run.epochs_completed} epochs, "
          f"target_transform={csv_run.target_transform!r}")
    if device == "cpu":
        _same_training(reference_run, csv_run, ref_path, csv_path, "CSV vs NumPy training")

    np.save(out_dir / "test_X.npy", test[:, 1:])
    np.save(out_dir / "test_y.npy", test[:, 0])
    from_csv = _cli_json(csv_path, test_csv, "--target", "median_house_value", "--device", device)
    from_npy = _cli_json(csv_path, out_dir / "test_X.npy", out_dir / "test_y.npy", "--device", device)
    _check(from_csv == from_npy, "CLI evaluation of the CSV differs from the .npy evaluation")
    api = forge.load_predictor(str(csv_path), device=device).evaluate(test[:, 1:], test[:, 0])
    _check(from_csv["mse"] == api.mse and from_csv["mae"] == api.mae, "CLI numbers differ from the API's")
    _check(from_csv["baseline_mse"] > 1e9 and from_csv["mse"] > 1e3, "metrics are not in native dollars")
    r2 = 1.0 - from_csv["mse"] / from_csv["baseline_mse"]
    _check(r2 > 0.70, f"R^2 {r2:.3f} is far below the M116 figure (~0.75)")
    print(f"  held-out ({from_csv['samples']} rows), native dollars: MSE {from_csv['mse']:.4g}, MAE {from_csv['mae']:.0f}, "
          f"baseline MSE {from_csv['baseline_mse']:.4g}, R^2 {r2:.3f}; JSON identical for DATA.csv --target and X.npy y.npy")
    print("housing OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV workflow acceptance on real data (Milestone 118).")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cadata", type=Path, default=None, help="StatLib California housing cadata.txt (not shipped).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(f"forge {forge.__version__} from {forge.__file__}")
    with tempfile.TemporaryDirectory() as tmp:
        run_pima(args.device, Path(tmp), args.seed)
        run_housing(args.device, Path(tmp), args.seed, args.cadata)
    print("ALL OK")


if __name__ == "__main__":
    main()
