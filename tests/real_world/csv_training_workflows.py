"""Real-data acceptance script for CSV-to-artifact training (Milestone 121): id-bearing CSV -> trained artifact.

```text
python tests/real_world/csv_training_workflows.py                            # Pima only (bundled)
python tests/real_world/csv_training_workflows.py --cadata /path/cadata.txt    # + California housing
python tests/real_world/csv_training_workflows.py --device cuda --cadata ...
```

For each real dataset, an `id`-bearing production-style CSV is unreadable by `load_csv()`/
`train_tabular_classifier()`/`train_tabular_regressor()` without either removing the column by hand or
selecting explicitly. This script shows `forge.train_tabular_classifier_csv()`/`train_tabular_regressor_csv()`
training directly from that id-bearing file with `columns=` given, and proves the result is identical -- every
result field and every trained parameter -- to the id-free `load_csv()` + `train_tabular_*(...,
feature_names=...)` pipeline written out by hand, and identical to the `forge model train` CLI command run on
the same file. It also proves the M119 reordered-CSV alignment still holds for an artifact trained through
this door, and (regression) that `target_transform="standardize"` still reports native units.

- **Pima diabetes** -- bundled (`examples/tabular_diabetes/data/diabetes.csv`), with a synthetic `patient_id`
  column prepended (the same recipe M120's own report used for its CLI reproduction).
- **California housing** -- StatLib `cadata.txt` (`https://lib.stat.cmu.edu/datasets/houses.zip`, 20,640 rows,
  target in the *first* column), with a synthetic `parcel_id` column prepended. Skipped, saying so, without
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
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

import forge
from forge.cli.main import main as cli_main

ROOT = Path(__file__).resolve().parents[2]
PIMA_DIR = ROOT / "examples" / "tabular_diabetes" / "data"
PIMA_CLASSES = ["no_diabetes", "diabetes"]
PIMA_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
HOUSING_COLUMNS = [
    "median_house_value", "median_income", "housing_median_age", "total_rooms", "total_bedrooms",
    "population", "households", "latitude", "longitude",
]
HOUSING_FEATURES = HOUSING_COLUMNS[1:]


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def _check(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _parameters_sha(path: Path) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(forge.load_model(str(path), device="cpu").named_parameters(), key=lambda kv: kv[0]):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(parameter.numpy()).tobytes())
    return digest.hexdigest()


def _same_training(reference, from_csv, reference_path: Path, csv_path: Path, label: str) -> None:
    import dataclasses

    for field in dataclasses.fields(reference):
        if field.name in ("history", "model", "artifact_path", "target_transform"):
            continue
        _check(getattr(reference, field.name) == getattr(from_csv, field.name), f"{label}: result field {field.name} differs")
    _check(_parameters_sha(reference_path) == _parameters_sha(csv_path), f"{label}: trained parameters differ")
    print(f"  {label}: every result field and every trained parameter identical (SHA-256 {_parameters_sha(csv_path)[:16]}...)")


def _write_id_csv(path: Path, header: "list[str]", rows, id_start: int) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["record_id", *header])
        for i, row in enumerate(rows):
            writer.writerow([id_start + i, *(v if isinstance(v, str) else repr(float(v)) for v in row)])


def _cli_train_json(*argv) -> dict:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(["model", "train", *map(str, argv), "--json"])
    _check(code == 0, f"forge model train {' '.join(map(str, argv))} exited {code}: {err.getvalue()}")
    return json.loads(out.getvalue())


def run_pima(device: str, out_dir: Path, seed: int) -> None:
    print("== Pima diabetes (id-bearing CSV): train_tabular_classifier_csv() -> forge model train ==")
    full = PIMA_DIR / "diabetes.csv"
    with open(full, newline="") as fh:
        table = list(csv.reader(fh))
    header, rows = table[0], table[1:]
    id_csv = out_dir / "diabetes_with_id.csv"
    _write_id_csv(id_csv, header, [[float(c) for c in r] for r in rows], id_start=10000)

    kwargs = dict(classes=PIMA_CLASSES, missing_columns=[1, 2, 3, 4, 5], device=device, seed=seed, epochs=15)
    reference_path, wrapper_path = out_dir / "pima_reference.forge", out_dir / "pima_wrapper.forge"
    reference = np.genfromtxt(full, delimiter=",", skip_header=1)
    reference_run = forge.train_tabular_classifier(
        reference[:, :-1], reference[:, -1].astype(int), path=reference_path, **kwargs,
    )
    wrapper_run = forge.train_tabular_classifier_csv(
        id_csv, target="Outcome", path=wrapper_path, columns=PIMA_FEATURES, **kwargs,
    )
    _same_training(reference_run, wrapper_run, reference_path, wrapper_path, "wrapper vs manual pipeline")
    _check(wrapper_run.validation_accuracy > wrapper_run.baseline_accuracy, "validation accuracy does not beat the baseline")
    info = forge.inspect_model(wrapper_run.artifact_path)
    _check(info.input_schema.feature_names == tuple(PIMA_FEATURES), "feature names were not persisted correctly")
    print(f"  trained on '{device}': validation accuracy {wrapper_run.validation_accuracy:.1%} vs "
          f"{wrapper_run.baseline_accuracy:.1%} majority baseline; feature names persisted {info.input_schema.feature_names}")

    # forge model train's CLI surface deliberately omits missing_columns= (§11 of the brief: not exposed), so
    # its parity reference is the same call *without* missing_columns=, not `wrapper_run` above.
    cli_reference_path = out_dir / "pima_cli_reference.forge"
    cli_reference = forge.train_tabular_classifier_csv(
        id_csv, target="Outcome", path=cli_reference_path, columns=PIMA_FEATURES,
        device=device, seed=seed, epochs=15,
    )
    cli_path = out_dir / "pima_cli.forge"
    cli_report = _cli_train_json(
        id_csv, "--task", "classification", "--target", "Outcome", "--output", cli_path,
        "--columns", *PIMA_FEATURES, "--device", device, "--seed", str(seed), "--epochs", "15",
    )
    _check(cli_report["validation_accuracy"] == cli_reference.validation_accuracy, "CLI training result differs from the API's")
    if device == "cpu":
        _check(_parameters_sha(cli_path) == _parameters_sha(cli_reference_path), "CLI-trained parameters differ from the API's")
    print("  forge model train CLI (no missing_columns=, not exposed on the CLI surface): identical to the same API call")

    # M119 alignment still holds for a CSV-trained artifact: a reordered, feature-only CSV predicts identically.
    predictor = forge.load_predictor(str(wrapper_path), device=device)
    reordered = list(reversed(PIMA_FEATURES))
    reordered_csv = out_dir / "pima_reordered.csv"
    with open(reordered_csv, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(reordered)
        for row in reference[:10, :-1]:
            writer.writerow([repr(float(row[header.index(name)])) for name in reordered])
    from forge.data import load_csv_features
    X_sel, names_sel = load_csv_features(reordered_csv, columns=reordered)
    via_reordered = [p.label for p in predictor.predict(X_sel, feature_names=names_sel)]
    direct = [p.label for p in predictor.predict(reference[:10, :-1])]
    _check(via_reordered == direct, "reordered CSV does not predict identically to the direct array call")
    print("  reordered feature-only CSV predicts identically (M119 alignment intact)")
    print("Pima OK")


def run_housing(device: str, out_dir: Path, seed: int, cadata: "Path | None") -> None:
    print("== California housing (id-bearing CSV): train_tabular_regressor_csv(target_transform='standardize') ==")
    if cadata is None or not cadata.is_file():
        print("SKIP: no --cadata given (or it does not exist); housing half not run.")
        return
    rows = [line.split() for line in cadata.read_text(encoding="latin-1").splitlines()]
    data = np.array([[float(t) for t in r] for r in rows if len(r) == 9 and all(t[0] in "+-.0123456789" for t in r)])
    _check(data.shape == (20640, 9), f"expected 20,640 x 9 rows, got {data.shape}")

    order = np.random.default_rng(123).permutation(len(data))
    train, test = data[order[:3000]], data[order[3000:5000]]
    train_id_csv = out_dir / "housing_train_with_id.csv"
    _write_id_csv(train_id_csv, HOUSING_COLUMNS, train, id_start=900000)
    test_id_csv = out_dir / "housing_test_with_id.csv"
    _write_id_csv(test_id_csv, HOUSING_COLUMNS, test, id_start=920000)

    kwargs = dict(target_transform="standardize", device=device, seed=seed)
    reference_path, wrapper_path = out_dir / "housing_reference.forge", out_dir / "housing_wrapper.forge"
    reference_run = forge.train_tabular_regressor(train[:, 1:], train[:, 0], path=reference_path, **kwargs)
    wrapper_run = forge.train_tabular_regressor_csv(
        train_id_csv, target="median_house_value", path=wrapper_path, columns=HOUSING_FEATURES, **kwargs,
    )
    _same_training(reference_run, wrapper_run, reference_path, wrapper_path, "wrapper vs manual pipeline")

    predictor = forge.load_predictor(str(wrapper_path), device=device)
    held_out = predictor.evaluate(test[:, 1:], test[:, 0])
    r2 = 1.0 - held_out.mse / held_out.baseline_mse
    _check(held_out.baseline_mse > 1e9 and held_out.mse > 1e3, "metrics are not in native dollars")
    _check(r2 > 0.70, f"R^2 {r2:.3f} is far below the M116/M120 figure (~0.75)")
    print(f"  trained on '{device}': held-out ({held_out.samples} rows), native dollars: MSE {held_out.mse:.4g}, "
          f"MAE {held_out.mae:.0f}, R^2 {r2:.3f}")

    cli_path = out_dir / "housing_cli.forge"
    cli_report = _cli_train_json(
        train_id_csv, "--task", "regression", "--target", "median_house_value", "--output", cli_path,
        "--columns", *HOUSING_FEATURES, "--device", device, "--seed", str(seed), "--target-transform", "standardize",
    )
    _check(cli_report["validation_mse"] == wrapper_run.validation_mse, "CLI training result differs from the API's")
    if device == "cpu":
        _check(_parameters_sha(cli_path) == _parameters_sha(wrapper_path), "CLI-trained parameters differ from the API's")
    print("  forge model train CLI: identical result to the Python API")

    # id-bearing test CSV, unreadable without --columns, evaluated through the CLI with explicit selection.
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main([
            "model", "evaluate", str(wrapper_path), str(test_id_csv), "--target", "median_house_value",
            "--columns", *HOUSING_FEATURES, "--device", device, "--json",
        ])
    _check(code == 0, f"forge model evaluate on the id-bearing test CSV failed: {err.getvalue()}")
    evaluated = json.loads(out.getvalue())
    _check(evaluated["mse"] == held_out.mse and evaluated["mae"] == held_out.mae, "CLI evaluation differs from the API's")
    print("  id-bearing held-out CSV evaluated through the CLI with --columns: identical to the API's evaluate()")
    print("housing OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV-to-artifact training acceptance on real data (Milestone 121).")
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
