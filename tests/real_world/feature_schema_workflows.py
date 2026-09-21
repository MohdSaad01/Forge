"""Real-data acceptance script for persisted tabular feature schemas (Milestone 119): same columns, different order.

```text
python tests/real_world/feature_schema_workflows.py                          # Pima only (bundled)
python tests/real_world/feature_schema_workflows.py --cadata /path/cadata.txt   # + California housing
python tests/real_world/feature_schema_workflows.py --device cuda --cadata ...
```

The M118 limitation, on real data and a real artifact: an artifact that records only a feature *count* accepts a
CSV whose columns are in another order and silently scores it wrongly. This script trains the same model twice --
once from arrays (unnamed, as before M119) and once with the CSV header's names -- and feeds both the same
holdout rows in different column layouts through the real CLI:

| holdout CSV                                        | unnamed artifact (M118)  | named artifact (M119)           |
|----------------------------------------------------|--------------------------|---------------------------------|
| correct order                                      | scores                   | scores (identical)              |
| two columns swapped / fully shuffled               | scores, **wrongly**      | scores exactly like correct     |
| a feature renamed to a non-feature name            | scores                   | rejected, names the columns     |
| `id` column + all features                         | rejected by width alone  | rejected, names `id`            |
| `id` column replacing the last feature (width ok)  | scores, **wrongly**      | rejected, names both columns    |
| a feature's case / whitespace changed              | scores                   | rejected, says why              |

It also proves names change nothing else (identical parameters), that the bundled pre-M119 artifact still
evaluates unchanged, and measures what alignment costs. Exit status 0 only if every check passes.
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
BUNDLED = ROOT / "models" / "tabular_classifier" / "diabetes_classifier.forge"
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


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(forge.load_model(str(path)).named_parameters(), key=lambda kv: kv[0]):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(parameter.numpy()).tobytes())
    return digest.hexdigest()


def _cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


def _evaluate(model: Path, data: Path, target: str, device: str):
    code, out, err = _cli("model", "evaluate", model, data, "--target", target, "--device", device, "--json")
    return code, (json.loads(out) if code == 0 else None), err.strip()


def _write(path: Path, header, rows) -> Path:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _variants(source: Path, target: str, out_dir: Path, tag: str) -> "dict[str, Path]":
    """The same rows of `source` in different column layouts. Values are never changed, only moved/renamed/added."""
    with open(source, newline="") as fh:
        table = list(csv.reader(fh))
    header, body = table[0], table[1:]
    features = [h for h in header if h != target]
    t = header.index(target)
    idx = {h: i for i, h in enumerate(header)}

    def make(name, new_features, extra_first=None, drop_last_feature=False):
        cols = list(new_features)
        if drop_last_feature:
            cols = cols[:-1]
        rows = []
        for n, r in enumerate(body):
            row = ([str(n)] if extra_first else []) + [r[idx[c]] if c in idx else r[idx[features[-1]]] for c in cols]
            rows.append(row + [r[t]])
        return _write(out_dir / f"{tag}_{name}.csv", (["id"] if extra_first else []) + cols + [target], rows)

    variants = {"correct": make("correct", features)}
    variants["swapped"] = make("swapped", [features[1], features[0], *features[2:]])
    shuffled = np.random.default_rng(7).permutation(len(features)).tolist()
    variants["shuffled"] = make("shuffled", [features[i] for i in shuffled])
    variants["id_plus_all"] = make("id_plus_all", features, extra_first=True)
    variants["id_replaces_last"] = make("id_replaces_last", features, extra_first=True, drop_last_feature=True)
    # a feature renamed to something that is not a feature of the model (values untouched)
    renamed = _write(out_dir / f"{tag}_renamed.csv", [*features[:-1], "Weight", target], [
        [r[idx[c]] for c in features] + [r[t]] for r in body])
    variants["renamed"] = renamed
    variants["case"] = _write(out_dir / f"{tag}_case.csv", [features[0].upper(), *features[1:], target], [
        [r[idx[c]] for c in features] + [r[t]] for r in body])
    return variants


def _report(title: str, unnamed_model: Path, named_model: Path, files: "dict[str, Path]", target: str, device: str,
            metric: str) -> None:
    print(f"  {'holdout CSV':18s} | {'unnamed artifact (M118)':40s} | named artifact (M119)")
    results = {}
    for name, path in files.items():
        _, u, u_err = _evaluate(unnamed_model, path, target, device)
        _, n, n_err = _evaluate(named_model, path, target, device)
        results[name] = (u, u_err, n, n_err)

        def cell(value, err):
            if value is not None:
                return f"{metric}={value[metric]:.6g}"
            return "REJECTED: " + err.split("--", 1)[-1].strip().replace("Error: ", "")[:70]

        print(f"  {name:18s} | {cell(u, u_err):40s} | {cell(n, n_err)[:70]}")
    correct_u, _, correct_n, _ = results["correct"]
    _check(correct_u is not None and correct_n is not None, f"{title}: the correctly ordered CSV must score on both artifacts")
    _check(correct_u == correct_n, f"{title}: unnamed and named artifacts (same training) disagree on the correct CSV")
    for name in ("swapped", "shuffled"):
        u, _, n, _ = results[name]
        _check(u is not None and u != correct_u, f"{title}: the unnamed artifact should silently score '{name}' differently (the M118 defect)")
        _check(n == correct_n, f"{title}: the named artifact must score '{name}' exactly like the correct order")
    _check(results["id_replaces_last"][0] is not None and results["id_replaces_last"][0] != correct_u,
           f"{title}: the unnamed artifact should silently accept an id replacing the last feature (width still matches)")
    _check(results["renamed"][0] == correct_u, f"{title}: the unnamed artifact should accept a renamed column (it has no names)")
    for name, needles in {
        "id_plus_all": ("unexpected", "'id'"), "id_replaces_last": ("missing", "unexpected", "'id'"),
        "renamed": ("missing", "'Weight'"), "case": ("differ only in case/whitespace",),
    }.items():
        _, _, n, n_err = results[name]
        _check(n is None and all(s in n_err for s in needles), f"{title}: the named artifact must reject '{name}' naming the columns: {n_err}")
    _check(results["id_plus_all"][0] is None, f"{title}: the width check alone should reject id + all features on the unnamed artifact")


def _timing(named_model: Path, unnamed_model: Path, X: np.ndarray, y: np.ndarray, names, device: str, labels: bool) -> None:
    perm = list(range(X.shape[1]))[::-1]
    moved, moved_names = X[:, perm], [names[i] for i in perm]
    named = forge.load_predictor(str(named_model), device=device)
    plain = forge.load_predictor(str(unnamed_model), device=device)

    def best(fn, repeats=7):
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            times.append(time.perf_counter() - start)
        return min(times) * 1000

    t_plain = best(lambda: plain.evaluate(X, y))
    t_ident = best(lambda: named.evaluate(X, y, feature_names=names))
    t_moved = best(lambda: named.evaluate(moved, y, feature_names=moved_names))
    from forge.training.inference import _column_order
    t_order = best(lambda: _column_order(moved_names, named.input_schema, len(names), "bench"), repeats=200)
    print(f"  {len(X)} rows x {X.shape[1]} features, evaluate(): unnamed {t_plain:.2f} ms | named, already in order "
          f"{t_ident:.2f} ms | named, reversed {t_moved:.2f} ms | name matching alone {t_order * 1000:.1f} us")
    _check(t_moved < t_plain * 1.5 + 5, "alignment overhead is not negligible next to inference")


def run_pima(device: str, out_dir: Path, seed: int) -> None:
    print("== Pima diabetes: named CSV vs unnamed arrays, same trained model ==")
    full = PIMA_DIR / "diabetes.csv"
    holdout_path = PIMA_DIR / "diabetes_holdout_eval.csv"
    holdout = np.genfromtxt(holdout_path, delimiter=",", skip_header=1)
    held = {tuple(r) for r in holdout}
    with open(full, newline="") as fh:
        table = list(csv.reader(fh))
    pool_rows = [r for r in table[1:] if tuple(float(v) for v in r) not in held]
    pool_csv = _write(out_dir / "pima_pool.csv", table[0], pool_rows)
    _check(len(pool_rows) == 614, f"expected a 614-row pool, got {len(pool_rows)}")

    X, y, names = forge.data.load_csv(pool_csv, target="Outcome", labels=True, return_feature_names=True)
    print(f"  header names: {names}")
    kwargs = dict(classes=PIMA_CLASSES, missing_columns=[1, 2, 3, 4, 5], device=device, seed=seed)
    named_path, unnamed_path = out_dir / "pima_named.forge", out_dir / "pima_unnamed.forge"
    named_run = forge.train_tabular_classifier(X, y, path=named_path, feature_names=names, **kwargs)
    unnamed_run = forge.train_tabular_classifier(X, y, path=unnamed_path, **kwargs)
    schema = forge.inspect_model(str(named_path)).input_schema
    _check(schema.feature_names == tuple(names) and forge.inspect_model(str(unnamed_path)).input_schema.feature_names is None,
           "named artifact must record the header names, unnamed must record none")
    if device == "cpu":
        _check(_sha(named_path) == _sha(unnamed_path), "names changed the trained parameters")
        _check(named_run.validation_accuracy == unnamed_run.validation_accuracy, "names changed the validation accuracy")
        print(f"  names change nothing else: identical parameters (SHA-256 {_sha(named_path)[:16]}...), validation accuracy "
              f"{named_run.validation_accuracy:.1%} vs {named_run.baseline_accuracy:.1%} baseline")

    print("  holdout evaluated through `forge model evaluate` (accuracy):")
    files = _variants(holdout_path, "Outcome", out_dir, "pima")
    _report("Pima", unnamed_path, named_path, files, "Outcome", device, "accuracy")

    # predict: a feature-only CSV, correct vs reordered, same answer
    with open(holdout_path, newline="") as fh:
        rows = list(csv.reader(fh))
    hdr, body = rows[0], rows[1:6]
    swap = [1, 0, *range(2, 8)]
    p_ok = _write(out_dir / "pred_ok.csv", hdr[:8], [r[:8] for r in body])
    p_sw = _write(out_dir / "pred_sw.csv", [hdr[i] for i in swap], [[r[i] for i in swap] for r in body])
    a, b = _cli("model", "predict", named_path, p_ok, "--device", device, "--json"), _cli("model", "predict", named_path, p_sw, "--device", device, "--json")
    _check(a[0] == b[0] == 0 and a[1] == b[1] and a[2] == b[2] == "", "predict: reordered CSV must print exactly what the ordered CSV prints")
    with_target = _write(out_dir / "pred_target.csv", hdr, [r for r in rows[1:6]])
    c = _cli("model", "predict", named_path, with_target, "--device", device, "--json")
    _check(c[0] == 1 and "'Outcome'" in c[2] and c[1] == "", "predict: a target column left in the file must be reported as unexpected")
    print("  `forge model predict`: reordered feature CSV prints exactly what the ordered one prints; a left-in "
          "'Outcome' column is rejected as unexpected")

    # the bundled artifact predates M119: evaluates unchanged (the CSV path warns that order cannot be verified)
    code, out, err = _cli("model", "evaluate", BUNDLED, holdout_path, "--target", "Outcome", "--device", device, "--json")
    _check(code == 0 and err.startswith("Warning: ") and "records no feature names" in err, f"bundled pre-M119 artifact: {code} {err!r}")
    Xh, yh = holdout[:, :-1], holdout[:, -1].astype(int)
    api = forge.load_predictor(str(BUNDLED), device=device).evaluate(Xh, yh)
    _check(json.loads(out)["confusion_matrix"] == api.confusion_matrix.tolist(), "bundled artifact: CSV result differs from the API on arrays")
    print(f"  the bundled pre-M119 artifact still evaluates the holdout CSV unchanged (accuracy {json.loads(out)['accuracy']:.1%}); "
          "stderr: one 'Warning:' line, stdout still pure JSON")

    Xt, yt, nt = forge.data.load_csv(holdout_path, target="Outcome", labels=True, return_feature_names=True)
    print("  performance:")
    _timing(named_path, unnamed_path, np.tile(Xt, (20, 1)), np.tile(yt, 20), nt, device, labels=True)
    print("Pima OK")


def run_housing(device: str, out_dir: Path, seed: int, cadata: "Path | None") -> None:
    print("== California housing: named CSV -> train_tabular_regressor(target_transform='standardize') ==")
    if cadata is None or not cadata.is_file():
        print("SKIP: no --cadata given (or it does not exist); housing half not run.")
        return
    rows = [line.split() for line in cadata.read_text(encoding="latin-1").splitlines()]
    data = np.array([[float(t) for t in r] for r in rows if len(r) == 9 and all(t[0] in "+-.0123456789" for t in r)])
    _check(data.shape == (20640, 9), f"expected 20,640 x 9 rows, got {data.shape}")
    order = np.random.default_rng(123).permutation(len(data))
    train, test = data[order[:3000]], data[order[3000:5000]]
    train_csv = _write(out_dir / "housing_train.csv", HOUSING_COLUMNS, [[repr(float(v)) for v in r] for r in train])
    test_csv = _write(out_dir / "housing_test.csv", HOUSING_COLUMNS, [[repr(float(v)) for v in r] for r in test])

    X, y, names = forge.data.load_csv(train_csv, target="median_house_value", return_feature_names=True)
    _check(names == HOUSING_COLUMNS[1:], f"the target column (first in this file) leaked into the names: {names}")
    kwargs = dict(target_transform="standardize", device=device, seed=seed)
    named_path, unnamed_path = out_dir / "housing_named.forge", out_dir / "housing_unnamed.forge"
    forge.train_tabular_regressor(X, y, path=named_path, feature_names=names, **kwargs)
    forge.train_tabular_regressor(X, y, path=unnamed_path, **kwargs)
    if device == "cpu":
        _check(_sha(named_path) == _sha(unnamed_path), "names changed the trained parameters")
    _check(forge.inspect_model(str(named_path)).target_transform is not None, "the M116 target transform must survive alongside the names")
    print(f"  header names (target excluded, first column of the file): {names}")
    print("  held-out 2,000 rows evaluated through `forge model evaluate` (MSE, native dollars^2):")
    files = _variants(test_csv, "median_house_value", out_dir, "housing")
    _report("California", unnamed_path, named_path, files, "median_house_value", device, "mse")
    code, correct, _ = _evaluate(named_path, files["correct"], "median_house_value", device)
    _check(code == 0 and correct["baseline_mse"] > 1e9 and 1.0 - correct["mse"] / correct["baseline_mse"] > 0.70,
           "the named regression artifact must still score in native dollars with R^2 > 0.70")
    print(f"  native units intact: R^2 {1.0 - correct['mse'] / correct['baseline_mse']:.3f}, MAE {correct['mae']:.0f} dollars")
    print("  performance (20,640 rows):")
    _timing(named_path, unnamed_path, data[:, 1:], data[:, 0], names, device, labels=False)
    print("housing OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Persisted tabular feature schema acceptance on real data (Milestone 119).")
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
