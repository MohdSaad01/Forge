"""Milestone 93 production packaging smoke test.

Every other test in this suite imports `forge` from the source tree (or an
editable install) -- none of them prove the actual *distribution* is
consumable. This file does: it builds Forge's real wheel via `python -m
build`, installs it into a fresh virtual environment (never `pip install
-e .`), and exercises the installed package -- import, CLI `--version`/
`--help`, artifact inspect/predict, and error behavior -- via genuine
`subprocess` calls run from a directory outside the repository. See
`docs/development/m93-production-packaging.md`.

Module-scoped fixtures build the wheel and create the venv once and reuse
them across all tests in this file, since each step (build, venv creation,
dependency install) is the same regardless of which behavior is under test.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest

import numpy as np
from PIL import Image

import forge
from forge.nn import Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory):
    """Build Forge's actual wheel via `python -m build`, skip cleanly if unavailable."""
    if subprocess.run([sys.executable, "-m", "build", "--version"], capture_output=True).returncode != 0:
        pytest.skip("the 'build' package is not installed -- run `pip install build` to enable this test")

    dist_dir = tmp_path_factory.mktemp("packaging_dist")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir), str(REPO_ROOT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"wheel build failed:\n{result.stdout}\n{result.stderr}"
    wheels = list(dist_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one built wheel, got {wheels}"
    return wheels[0]


@pytest.fixture(scope="module")
def clean_install(tmp_path_factory, built_wheel):
    """Install the built wheel into a fresh venv; return that venv's python executable."""
    env_dir = tmp_path_factory.mktemp("packaging_venv") / "env"
    venv.create(env_dir, with_pip=True)
    python = env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    result = subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", str(built_wheel)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"clean-env install failed:\n{result.stdout}\n{result.stderr}"
    return python


@pytest.fixture()
def outside_repo_dir(tmp_path):
    """A working directory that is genuinely outside the Forge repository."""
    d = tmp_path / "outside"
    d.mkdir()
    return d


def _run(python: Path, args: list, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([str(python), *args], cwd=str(cwd), capture_output=True, text=True)


# -- import / identity -----------------------------------------------------


def test_installed_forge_imports_from_outside_repo(clean_install, outside_repo_dir):
    result = _run(
        clean_install,
        ["-c", "import forge; print(forge.__file__); print(forge.__version__)"],
        outside_repo_dir,
    )
    assert result.returncode == 0, result.stderr
    resolved_path, version = result.stdout.strip().splitlines()
    assert str(REPO_ROOT) not in resolved_path, "installed forge resolved to the source repository, not site-packages"
    assert version == forge.__version__


# -- CLI ---------------------------------------------------------------


def test_installed_cli_version_matches_package_version(clean_install, outside_repo_dir):
    result = _run(clean_install, ["-m", "forge", "--version"], outside_repo_dir)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"forge {forge.__version__}"


def test_installed_cli_help_lists_commands(clean_install, outside_repo_dir):
    result = _run(clean_install, ["-m", "forge", "--help"], outside_repo_dir)
    assert result.returncode == 0, result.stderr
    assert "model" in result.stdout and "checkpoint" in result.stdout


# -- representative artifact workflow ---------------------------------------


def test_installed_forge_inspects_and_predicts_real_artifact(clean_install, outside_repo_dir):
    # The artifact is built with the dev-tree forge already imported into this
    # test process; only the *consuming* side (inspect/predict) runs against
    # the installed distribution, in a fresh subprocess, outside the repo --
    # exactly the shape of a developer receiving a `.forge` file from someone
    # else.
    forge.random.seed(0)
    model = Sequential(Linear(3, 4), ReLU(), Linear(4, 2))
    model_path = outside_repo_dir / "model.forge"
    forge.save_model(model, str(model_path), task="regression")

    result = _run(clean_install, ["-m", "forge", "model", "inspect", "model.forge"], outside_repo_dir)
    assert result.returncode == 0, result.stderr
    assert "Task: regression" in result.stdout

    input_path = outside_repo_dir / "input.json"
    input_path.write_text("[0.1, 0.2, 0.3]")
    result = _run(clean_install, ["-m", "forge", "model", "predict", "model.forge", "input.json"], outside_repo_dir)
    assert result.returncode == 0, result.stderr
    assert "Prediction" in result.stdout


# -- grayscale image classification (Milestone 94) --------------------------


def test_installed_forge_predicts_grayscale_image_classification_artifact(clean_install, outside_repo_dir):
    """Milestone 94's own installed-package acceptance criterion: a real
    grayscale-model artifact + a genuinely grayscale PNG must predict
    correctly through the installed distribution, from a fresh process,
    outside the repository -- the same real production boundary M93
    established, now exercised for the grayscale channel-matching fix
    specifically (the exact shape of the M93-discovered MNIST failure)."""
    from forge.data.transforms import Compose, Normalize, Resize

    forge.random.seed(0)
    model = Sequential(
        Conv2d(1, 4, kernel_size=3, padding=1), ReLU(), MaxPool2d(kernel_size=2),
        Flatten(), Linear(4 * 4 * 4, 2),
    )
    preprocessing = Compose([Resize((8, 8)), Normalize(mean=0.0, std=255.0)])
    model_path = outside_repo_dir / "grayscale_model.forge"
    forge.save_model(
        model, str(model_path), preprocessing=preprocessing,
        classes=["zero", "one"], task="classification",
    )

    image_path = outside_repo_dir / "query.png"
    arr = np.full((8, 8), 100, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(image_path)  # genuinely grayscale

    result = _run(
        clean_install, ["-m", "forge", "model", "predict", "grayscale_model.forge", "query.png"], outside_repo_dir,
    )
    assert result.returncode == 0, result.stderr
    assert "Prediction:" in result.stdout
    assert "Confidence:" in result.stdout


# -- error behavior -----------------------------------------------------


def test_installed_forge_missing_artifact_error(clean_install, outside_repo_dir):
    result = _run(clean_install, ["-m", "forge", "model", "inspect", "does_not_exist.forge"], outside_repo_dir)
    assert result.returncode != 0
    assert "Error" in result.stdout + result.stderr


def test_installed_forge_invalid_json_input_error(clean_install, outside_repo_dir):
    forge.random.seed(0)
    model = Sequential(Linear(3, 4), ReLU(), Linear(4, 2))
    model_path = outside_repo_dir / "model2.forge"
    forge.save_model(model, str(model_path), task="regression")

    bad_input = outside_repo_dir / "bad_input.json"
    bad_input.write_text("{not valid json")
    result = _run(clean_install, ["-m", "forge", "model", "predict", "model2.forge", "bad_input.json"], outside_repo_dir)
    assert result.returncode != 0
    assert "Error" in result.stdout + result.stderr


# -- CUDA availability (no accidental CPU fallback) --------------------------


def test_installed_forge_cuda_availability_matches_source_tree(clean_install, outside_repo_dir):
    """The installed package must see the same CUDA availability as the dev tree.

    Guards against exactly the Milestone 93 regression found and fixed: CUDA
    kernel source (`forge/backend/cuda/kernels.cu`) missing from the built
    distribution, which made `is_cuda_available()` silently return `False`
    after install instead of compiling and reporting real hardware status.
    """
    result = _run(clean_install, ["-c", "import forge; print(forge.cuda.is_cuda_available())"], outside_repo_dir)
    assert result.returncode == 0, result.stderr
    installed_cuda_available = result.stdout.strip() == "True"
    assert installed_cuda_available == forge.cuda.is_cuda_available()


# -- `forge model evaluate` from the installed wheel (Milestone 117) -----------------------------------------
#
# The artifacts and data are produced by the dev-tree forge in this process (a developer *receives* them);
# everything that evaluates runs in the clean venv, in a fresh process, from a directory outside the repository.
# `_REFERENCE_EVALUATOR` is deliberately a separate, standalone consumer program using only `forge.load_predictor()`
# and `.evaluate()` -- the oracle the CLI's output is compared against, field by field.

_REFERENCE_EVALUATOR = '''
import json, sys
import numpy as np
import forge

predictor = forge.load_predictor(sys.argv[1])
if predictor.task == "classification":
    result = predictor.evaluate(sys.argv[2])
else:
    result = predictor.evaluate(np.load(sys.argv[2]), np.load(sys.argv[3]))
out = {}
for name, value in vars(result).items():
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, tuple):
        value = list(value)
    out[name] = value
print(json.dumps(out))
'''


def _forge_console_script(python: Path) -> Path:
    return python.parent / ("forge.exe" if sys.platform == "win32" else "forge")


def _evaluate_outside_repo(clean_install, cwd: Path, *args, cli_flags=("--json",)):
    """Run the installed `forge model evaluate` and the reference evaluator; return `(cli_json, reference_json)`."""
    cli = subprocess.run(
        [str(_forge_console_script(clean_install)), "model", "evaluate", *map(str, args), *cli_flags],
        cwd=str(cwd), capture_output=True, text=True,
    )
    assert cli.returncode == 0, cli.stderr
    reference_script = cwd / "reference_evaluate.py"
    reference_script.write_text(_REFERENCE_EVALUATOR)
    reference = _run(clean_install, [str(reference_script), *map(str, args)], cwd)
    assert reference.returncode == 0, reference.stderr
    return json.loads(cli.stdout), json.loads(reference.stdout)


def _assert_same_evaluation(cli: dict, reference: dict) -> None:
    assert cli.keys() == reference.keys()
    for key, expected in reference.items():
        if isinstance(expected, float):
            assert cli[key] == pytest.approx(expected, rel=1e-9), key
        elif isinstance(expected, list) and expected and isinstance(expected[0], float):
            assert cli[key] == pytest.approx(expected, rel=1e-9), key
        else:
            assert cli[key] == expected, key


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_installed_forge_is_what_evaluates(clean_install, outside_repo_dir):
    result = _run(clean_install, ["-c", "import forge; print(forge.__file__)"], outside_repo_dir)
    assert result.returncode == 0, result.stderr
    assert "site-packages" in result.stdout and str(REPO_ROOT) not in result.stdout
    assert _forge_console_script(clean_install).is_file()


def test_installed_cli_evaluates_a_tabular_classification_artifact(clean_install, outside_repo_dir):
    from forge.data.transforms import Normalize

    forge.random.seed(0)
    model_path = outside_repo_dir / "diabetes_like.forge"
    forge.save_model(
        Sequential(Linear(3, 6), ReLU(), Linear(6, 2)), str(model_path), task="tabular_classification",
        preprocessing=Normalize(mean=np.array([1.0, 2.0, 3.0], dtype=np.float32), std=np.array([2.0, 1.0, 4.0], dtype=np.float32)),
        classes=["no_disease", "disease"],
    )
    rng = np.random.default_rng(0)
    np.save(outside_repo_dir / "X.npy", rng.normal(size=(40, 3)).astype(np.float32) * 3 + 1)
    np.save(outside_repo_dir / "y.npy", np.array(["disease", "no_disease"] * 20))
    before = _sha256(model_path)

    cli, reference = _evaluate_outside_repo(clean_install, outside_repo_dir, "diabetes_like.forge", "X.npy", "y.npy")

    assert cli["task"] == "tabular_classification" and cli["samples"] == 40 and cli["classes"] == ["no_disease", "disease"]
    _assert_same_evaluation(cli, reference)
    assert _sha256(model_path) == before

    text = subprocess.run(
        [str(_forge_console_script(clean_install)), "model", "evaluate", "diabetes_like.forge", "X.npy", "y.npy"],
        cwd=str(outside_repo_dir), capture_output=True, text=True,
    )
    assert text.returncode == 0 and "Baseline accuracy:" in text.stdout and "Confusion matrix" in text.stdout


def test_installed_cli_evaluates_a_standardized_regression_artifact_in_native_units(clean_install, outside_repo_dir):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3)) * np.array([1000.0, 0.01, 5.0]) + np.array([500.0, 1.0, 20.0])
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    y = 2.0e5 + 1.0e5 * (z[:, 0] - 0.5 * z[:, 1]) + 3.0e3 * rng.normal(size=300)
    model_path = outside_repo_dir / "housing.forge"
    forge.train_tabular_regressor(X, y, path=model_path, seed=3, epochs=150, target_transform="standardize")
    np.save(outside_repo_dir / "X.npy", X[:100])
    np.save(outside_repo_dir / "y.npy", y[:100])
    before = _sha256(model_path)

    cli, reference = _evaluate_outside_repo(clean_install, outside_repo_dir, "housing.forge", "X.npy", "y.npy")

    _assert_same_evaluation(cli, reference)
    assert cli["baseline_mse"] > 1e9 and cli["mse"] > 1e3     # native units; z-space would be < ~10
    assert cli["mse"] < 0.2 * cli["baseline_mse"]
    assert _sha256(model_path) == before


def test_installed_cli_evaluates_an_image_classification_artifact(clean_install, outside_repo_dir):
    from forge.data.transforms import Compose, Normalize, Resize

    forge.random.seed(0)
    model = Sequential(
        Conv2d(3, 4, kernel_size=3, padding=1), ReLU(), MaxPool2d(kernel_size=2), Flatten(), Linear(4 * 4 * 4, 2),
    )
    model_path = outside_repo_dir / "pets.forge"
    forge.save_model(
        model, str(model_path), preprocessing=Compose([Resize((8, 8)), Normalize(mean=0.0, std=255.0)]),
        classes=["dog", "cat"], task="classification",     # artifact order is the reverse of folder-name order
    )
    for class_name, fills in {"cat": (30, 60, 90), "dog": (160, 200, 240, 250)}.items():
        (outside_repo_dir / "held_out" / class_name).mkdir(parents=True)
        for i, fill in enumerate(fills):
            Image.fromarray(np.full((8, 8, 3), fill, dtype=np.uint8), mode="RGB").save(
                outside_repo_dir / "held_out" / class_name / f"{i}.png"
            )
    before = _sha256(model_path)

    cli, reference = _evaluate_outside_repo(clean_install, outside_repo_dir, "pets.forge", "held_out")

    assert cli["classes"] == ["dog", "cat"] and cli["support"] == [4, 3] and cli["samples"] == 7
    _assert_same_evaluation(cli, reference)
    assert _sha256(model_path) == before


def test_installed_cli_evaluate_reports_a_bad_invocation_without_a_traceback(clean_install, outside_repo_dir):
    for args in (["does_not_exist.forge", "X.npy", "y.npy"], ["--json"]):
        result = subprocess.run(
            [str(_forge_console_script(clean_install)), "model", "evaluate", *args],
            cwd=str(outside_repo_dir), capture_output=True, text=True,
        )
        assert result.returncode != 0 and "Traceback" not in result.stderr
    missing = subprocess.run(
        [str(_forge_console_script(clean_install)), "model", "evaluate", "does_not_exist.forge", "X.npy", "y.npy", "--json"],
        cwd=str(outside_repo_dir), capture_output=True, text=True,
    )
    assert missing.returncode == 1 and missing.stdout == "" and "artifact not found" in missing.stderr


# -- CSV evaluation and training from the installed wheel (Milestone 118) ------------------------------------
#
# Same arrangement as above: artifacts and data are produced by the dev tree (a developer *receives* them); every
# run that reads a CSV happens in the clean venv (which has numpy + Pillow and nothing else -- asserted below, so
# "no pandas" is a fact about the environment, not a hope), from a directory outside the repository, in a fresh
# process. `_CSV_REFERENCE_EVALUATOR` is the oracle and deliberately does *not* use `forge.data.load_csv()`: it parses
# the file with the standard `csv` module, so the CLI's CSV reading is compared against an independent reading of the
# same bytes.

_CSV_REFERENCE_EVALUATOR = '''
import csv, json, sys
import numpy as np
import forge

model, path, target = sys.argv[1:4]
with open(path, newline="") as fh:
    rows = list(csv.reader(fh))
header, body = rows[0], rows[1:]
column = header.index(target)
features = np.array([[float(v) for i, v in enumerate(r) if i != column] for r in body])
raw_target = [r[column] for r in body]
predictor = forge.load_predictor(model)
if predictor.task == "regression":
    y = np.array([float(v) for v in raw_target])
else:
    y = np.array([int(float(v)) for v in raw_target])
result = predictor.evaluate(features, y)
out = {}
for name, value in vars(result).items():
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, tuple):
        value = list(value)
    out[name] = value
print(json.dumps(out))
'''

_CSV_TRAINING_CONSUMER = '''
import json, sys
import forge

assert "site-packages" in forge.__file__, forge.__file__
try:
    import pandas
except ImportError:
    pass
else:
    raise SystemExit("pandas is importable in the consumer environment")

X, y = forge.data.load_csv(sys.argv[1], target="Outcome", labels=True)
result = forge.train_tabular_classifier(
    X, y, path=sys.argv[2], classes=["no_diabetes", "diabetes"], missing_columns=[1, 2, 3, 4, 5], seed=0, epochs=15,
)
holdout_X, holdout_y = forge.data.load_csv(sys.argv[3], target="Outcome", labels=True)
held_out = forge.load_predictor(sys.argv[2]).evaluate(holdout_X, holdout_y)
assert "pandas" not in sys.modules
print(json.dumps({
    "file": forge.__file__, "samples": result.samples, "classes": result.classes,
    "validation_accuracy": result.validation_accuracy, "validation_loss": result.validation_loss,
    "holdout_accuracy": held_out.accuracy, "holdout_samples": held_out.samples,
}))
'''

_PIMA_DIR = REPO_ROOT / "examples" / "tabular_diabetes" / "data"


def _csv_evaluate_outside_repo(clean_install, cwd: Path, model: str, csv_name: str, target: str):
    """`(cli_json, reference_json)` for `forge model evaluate MODEL CSV --target TARGET --json` in the clean venv."""
    cli = subprocess.run(
        [str(_forge_console_script(clean_install)), "model", "evaluate", model, csv_name, "--target", target, "--json"],
        cwd=str(cwd), capture_output=True, text=True,
    )
    assert cli.returncode == 0, cli.stderr
    script = cwd / "reference_csv_evaluate.py"
    script.write_text(_CSV_REFERENCE_EVALUATOR)
    reference = _run(clean_install, [str(script), model, csv_name, target], cwd)
    assert reference.returncode == 0, reference.stderr
    return json.loads(cli.stdout), json.loads(reference.stdout)


def test_the_consumer_environment_has_no_dataframe_library(clean_install, outside_repo_dir):
    probe = _run(clean_install, ["-c", "import pandas"], outside_repo_dir)
    assert probe.returncode != 0 and "ModuleNotFoundError" in probe.stderr
    listing = _run(clean_install, ["-m", "pip", "list", "--format=freeze"], outside_repo_dir).stdout.lower()
    assert "pandas" not in listing and "numpy==" in listing and "pillow==" in listing


def test_installed_cli_evaluates_a_csv_with_a_pre_existing_real_artifact(clean_install, outside_repo_dir):
    """The bundled Pima classifier (saved long before M118) on the bundled 154-row holdout CSV: real data."""
    shutil.copy(REPO_ROOT / "models" / "tabular_classifier" / "diabetes_classifier.forge", outside_repo_dir / "pima.forge")
    shutil.copy(_PIMA_DIR / "diabetes_holdout_eval.csv", outside_repo_dir / "holdout.csv")
    before = (_sha256(outside_repo_dir / "pima.forge"), _sha256(outside_repo_dir / "holdout.csv"))

    cli, reference = _csv_evaluate_outside_repo(clean_install, outside_repo_dir, "pima.forge", "holdout.csv", "Outcome")

    assert cli["task"] == "tabular_classification" and cli["samples"] == 154
    assert cli["baseline_accuracy"] == pytest.approx(96 / 154) and cli["accuracy"] > cli["baseline_accuracy"]
    _assert_same_evaluation(cli, reference)
    assert (_sha256(outside_repo_dir / "pima.forge"), _sha256(outside_repo_dir / "holdout.csv")) == before

    text = subprocess.run(
        [str(_forge_console_script(clean_install)), "model", "evaluate", "pima.forge", "holdout.csv", "--target", "Outcome"],
        cwd=str(outside_repo_dir), capture_output=True, text=True,
    )
    assert text.returncode == 0 and "Baseline accuracy:" in text.stdout and "Confusion matrix" in text.stdout


def test_installed_cli_evaluates_a_standardized_regression_csv_in_native_units(clean_install, outside_repo_dir):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3)) * np.array([1000.0, 0.01, 5.0]) + np.array([500.0, 1.0, 20.0])
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    y = 2.0e5 + 1.0e5 * (z[:, 0] - 0.5 * z[:, 1]) + 3.0e3 * rng.normal(size=300)
    forge.train_tabular_regressor(X, y, path=outside_repo_dir / "housing.forge", seed=3, epochs=150, target_transform="standardize")
    with open(outside_repo_dir / "held_out.csv", "w", newline="") as fh:            # target first, as in the StatLib file
        fh.write("median_house_value,income,age,rooms\n")
        for target, row in zip(y[:100], X[:100]):
            fh.write(",".join(repr(float(v)) for v in (target, *row)) + "\n")
    before = _sha256(outside_repo_dir / "housing.forge")

    cli, reference = _csv_evaluate_outside_repo(clean_install, outside_repo_dir, "housing.forge", "held_out.csv", "median_house_value")

    _assert_same_evaluation(cli, reference)
    assert cli["baseline_mse"] > 1e9 and cli["mse"] > 1e3                            # native units; z-space would be < ~10
    assert cli["mse"] < 0.2 * cli["baseline_mse"]
    assert _sha256(outside_repo_dir / "housing.forge") == before


def test_installed_forge_trains_from_a_csv_without_any_conversion_layer(clean_install, outside_repo_dir):
    """`load_csv()` + the unchanged `train_tabular_classifier()` in the clean venv, then a fresh-process CLI evaluation."""
    shutil.copy(_PIMA_DIR / "diabetes.csv", outside_repo_dir / "diabetes.csv")
    shutil.copy(_PIMA_DIR / "diabetes_holdout_eval.csv", outside_repo_dir / "holdout.csv")
    script = outside_repo_dir / "consume.py"
    script.write_text(_CSV_TRAINING_CONSUMER)

    ran = _run(clean_install, [str(script), "diabetes.csv", "trained.forge", "holdout.csv"], outside_repo_dir)
    assert ran.returncode == 0, ran.stderr
    report = json.loads(ran.stdout)
    assert "site-packages" in report["file"] and str(REPO_ROOT) not in report["file"]
    assert (report["samples"], report["classes"], report["holdout_samples"]) == (768, ["no_diabetes", "diabetes"], 154)

    # The same training on plain NumPy arrays, in this (dev-tree) process, gives the same numbers.
    data = np.genfromtxt(_PIMA_DIR / "diabetes.csv", delimiter=",", skip_header=1)
    reference = forge.train_tabular_classifier(
        data[:, :-1], data[:, -1].astype(int), path=outside_repo_dir / "reference.forge",
        classes=["no_diabetes", "diabetes"], missing_columns=[1, 2, 3, 4, 5], seed=0, epochs=15,
    )
    assert report["validation_accuracy"] == reference.validation_accuracy
    assert report["validation_loss"] == pytest.approx(reference.validation_loss, rel=1e-6)

    cli, oracle = _csv_evaluate_outside_repo(clean_install, outside_repo_dir, "trained.forge", "holdout.csv", "Outcome")
    _assert_same_evaluation(cli, oracle)
    assert cli["accuracy"] == report["holdout_accuracy"]


def test_installed_cli_reports_a_bad_csv_without_a_traceback(clean_install, outside_repo_dir):
    shutil.copy(REPO_ROOT / "models" / "tabular_classifier" / "diabetes_classifier.forge", outside_repo_dir / "pima.forge")
    (outside_repo_dir / "bad.csv").write_text("a,b,Outcome\n1,two,0\n")
    (outside_repo_dir / "ok.csv").write_text("a,Outcome\n1,0\n")
    cases = [
        (["pima.forge", "bad.csv", "--target", "Outcome"], "line 2, column 'b': 'two' is not a number"),
        (["pima.forge", "ok.csv", "--target", "nope"], "target column 'nope' is not in the header"),
        (["pima.forge", "missing.csv", "--target", "Outcome"], "CSV file not found"),
        (["pima.forge", "ok.csv"], "needs --target COLUMN"),
    ]
    for args, expected in cases:
        for flags in ([], ["--json"]):
            result = subprocess.run(
                [str(_forge_console_script(clean_install)), "model", "evaluate", *args, *flags],
                cwd=str(outside_repo_dir), capture_output=True, text=True,
            )
            assert result.returncode == 1 and result.stdout == "", (args, result.stdout)
            assert result.stderr.startswith("Error: ") and "Traceback" not in result.stderr, result.stderr
            assert expected in result.stderr, result.stderr


# -- persisted tabular feature names from the installed wheel (Milestone 119) ---------------------------------
#
# The consumer trains from a CSV *with* its header names, in a fresh process of the clean venv, then a fresh CLI
# process per invocation checks named CSVs against the artifact: reordered columns are aligned, an unknown, extra
# (`id`) or misspelled column is rejected. Everything runs from a directory outside the repository.

_NAMED_TRAINING_CONSUMER = '''
import json, sys
import forge

assert "site-packages" in forge.__file__, forge.__file__
X, y, names = forge.data.load_csv(sys.argv[1], target="Outcome", labels=True, return_feature_names=True)
forge.train_tabular_classifier(
    X, y, path=sys.argv[2], classes=["no_diabetes", "diabetes"], missing_columns=[1, 2, 3, 4, 5], seed=0, epochs=15,
    feature_names=names,
)
schema = forge.inspect_model(sys.argv[2]).input_schema
print(json.dumps({"file": forge.__file__, "names": names, "recorded": list(schema.feature_names)}))
'''

_PIMA_FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]


def _write_csv_columns(path: Path, source: Path, columns: list, *, id_first: bool = False, keep_target: bool = True) -> Path:
    """`source`'s rows with `columns` (source header names, or `(new_name, source_name)`): values moved, never changed."""
    import csv

    with open(source, newline="") as fh:
        table = list(csv.reader(fh))
    header = table[0]
    picked = [(c, c) if isinstance(c, str) else c for c in columns]
    out_header = (["id"] if id_first else []) + [new for new, _ in picked] + (["Outcome"] if keep_target else [])
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(out_header)
        for n, row in enumerate(table[1:]):
            values = [row[header.index(old)] for _, old in picked]
            writer.writerow(([str(n)] if id_first else []) + values + ([row[header.index("Outcome")]] if keep_target else []))
    return path


def _installed_cli(clean_install, cwd: Path, *args):
    return subprocess.run(
        [str(_forge_console_script(clean_install)), *map(str, args)], cwd=str(cwd), capture_output=True, text=True,
    )


@pytest.fixture()
def named_pima(clean_install, outside_repo_dir):
    """A named Pima artifact trained in the clean venv, and the holdout in several column layouts."""
    shutil.copy(_PIMA_DIR / "diabetes.csv", outside_repo_dir / "diabetes.csv")
    holdout, features = _PIMA_DIR / "diabetes_holdout_eval.csv", _PIMA_FEATURES
    swapped = [features[1], features[0], *features[2:]]
    make = lambda name, *args, **kwargs: _write_csv_columns(outside_repo_dir / f"{name}.csv", holdout, *args, **kwargs)  # noqa: E731
    files = {
        "correct": make("correct", features),
        "reordered": make("reordered", swapped),
        "renamed": make("renamed", [*features[:-1], ("Weight", "Age")]),
        "extra_id": make("extra_id", features, id_first=True),
        "id_replaces_last": make("id_replaces_last", features[:-1], id_first=True),
        "case": make("case", [("age", "Age"), *features[:-1]]),
        "features_only": make("features_only", features, keep_target=False),
        "features_only_reordered": make("features_only_reordered", swapped, keep_target=False),
    }
    script = outside_repo_dir / "consume_named.py"
    script.write_text(_NAMED_TRAINING_CONSUMER)
    ran = _run(clean_install, [str(script), "diabetes.csv", "named.forge"], outside_repo_dir)
    assert ran.returncode == 0, ran.stderr
    report = json.loads(ran.stdout)
    assert "site-packages" in report["file"] and str(REPO_ROOT) not in report["file"]
    assert report["recorded"] == report["names"] == features
    return files


def test_installed_forge_records_the_csv_header_names_in_the_artifact(clean_install, outside_repo_dir, named_pima):
    inspected = _installed_cli(clean_install, outside_repo_dir, "model", "inspect", "named.forge", "--json")
    assert inspected.returncode == 0, inspected.stderr
    payload = json.loads(inspected.stdout)
    assert payload["input_feature_count"] == 8 and payload["input_feature_names"] == _PIMA_FEATURES
    assert payload["format_version"] == 2                                         # no format bump for names
    text = _installed_cli(clean_install, outside_repo_dir, "model", "inspect", "named.forge")
    assert "Feature names: Pregnancies, Glucose, BloodPressure" in text.stdout


def test_installed_cli_aligns_a_reordered_csv_and_scores_it_like_the_ordered_one(clean_install, outside_repo_dir, named_pima):
    args = ["--target", "Outcome", "--json"]
    correct = _installed_cli(clean_install, outside_repo_dir, "model", "evaluate", "named.forge", named_pima["correct"].name, *args)
    reordered = _installed_cli(clean_install, outside_repo_dir, "model", "evaluate", "named.forge", named_pima["reordered"].name, *args)
    assert correct.returncode == reordered.returncode == 0, reordered.stderr
    assert correct.stdout == reordered.stdout and correct.stderr == reordered.stderr == ""
    assert json.loads(correct.stdout)["accuracy"] > json.loads(correct.stdout)["baseline_accuracy"]


@pytest.mark.parametrize("name, needles", [
    ("renamed", ("missing", "['Age']", "unexpected", "['Weight']")),
    ("extra_id", ("unexpected", "['id']", "never drops")),
    ("id_replaces_last", ("missing", "['Age']", "unexpected", "['id']")),
    ("case", ("'age' vs 'Age' differ only in case/whitespace",)),
])
def test_installed_cli_rejects_a_csv_that_is_not_the_artifacts_columns(clean_install, outside_repo_dir, named_pima, name, needles):
    for flags in ([], ["--json"]):
        result = _installed_cli(clean_install, outside_repo_dir, "model", "evaluate", "named.forge", named_pima[name].name,
                                "--target", "Outcome", *flags)
        assert result.returncode == 1 and result.stdout == "", result.stdout
        assert result.stderr.startswith("Error: ") and "Traceback" not in result.stderr and result.stderr.count("\n") == 1
        for needle in needles:
            assert needle in result.stderr, result.stderr


def test_installed_cli_predicts_from_a_feature_only_csv_with_the_same_rule(clean_install, outside_repo_dir, named_pima):
    ordered = _installed_cli(clean_install, outside_repo_dir, "model", "predict", "named.forge", named_pima["features_only"].name, "--json")
    reordered = _installed_cli(clean_install, outside_repo_dir, "model", "predict", "named.forge",
                               named_pima["features_only_reordered"].name, "--json")
    assert ordered.returncode == reordered.returncode == 0, reordered.stderr
    assert ordered.stdout == reordered.stdout and len(json.loads(ordered.stdout)["predictions"]) == 154
    # a target column left in the file is an extra column, not silently dropped
    with_target = _installed_cli(clean_install, outside_repo_dir, "model", "predict", "named.forge", named_pima["correct"].name)
    assert with_target.returncode == 1 and with_target.stdout == "" and "['Outcome']" in with_target.stderr


def test_installed_cli_takes_an_unnamed_pre_m119_artifact_with_one_warning_line(clean_install, outside_repo_dir):
    """The bundled Pima classifier predates names: unchanged M118 behaviour, plus an honest stderr note."""
    shutil.copy(REPO_ROOT / "models" / "tabular_classifier" / "diabetes_classifier.forge", outside_repo_dir / "pima.forge")
    shutil.copy(_PIMA_DIR / "diabetes_holdout_eval.csv", outside_repo_dir / "holdout.csv")
    cli, reference = _csv_evaluate_outside_repo(clean_install, outside_repo_dir, "pima.forge", "holdout.csv", "Outcome")
    _assert_same_evaluation(cli, reference)
    raw = _installed_cli(clean_install, outside_repo_dir, "model", "evaluate", "pima.forge", "holdout.csv", "--target", "Outcome", "--json")
    assert raw.stderr.startswith("Warning: ") and raw.stderr.count("\n") == 1 and "records no feature names" in raw.stderr


# -- explicit column selection and feature-name retrofit from the installed wheel (Milestone 120) -------------
#
# Reuses `named_pima` (a real Pima artifact trained in the clean venv, plus the holdout in several column
# layouts, including `extra_id` -- an id column the M119 tests above already prove is rejected without
# `--columns`) rather than building a second fixture: `--columns` is a strictly narrower reading of the same
# CSV, so the id-column CSV M119 already wrote is exactly the right input to prove it against.


def test_installed_cli_evaluate_columns_excludes_the_id_and_matches_the_correct_csv(clean_install, outside_repo_dir, named_pima):
    correct = _installed_cli(clean_install, outside_repo_dir, "model", "evaluate", "named.forge",
                              named_pima["correct"].name, "--target", "Outcome", "--json")
    with_id = _installed_cli(clean_install, outside_repo_dir, "model", "evaluate", "named.forge",
                              named_pima["extra_id"].name, "--target", "Outcome", "--columns", *_PIMA_FEATURES, "--json")
    assert correct.returncode == with_id.returncode == 0, with_id.stderr
    assert correct.stdout == with_id.stdout and correct.stderr == with_id.stderr == ""


def test_installed_cli_predict_columns_excludes_the_id(clean_install, outside_repo_dir, named_pima):
    feature_only_with_id = _write_csv_columns(
        outside_repo_dir / "predict_with_id.csv", _PIMA_DIR / "diabetes_holdout_eval.csv", _PIMA_FEATURES,
        id_first=True, keep_target=False,
    )
    without_columns = _installed_cli(clean_install, outside_repo_dir, "model", "predict", "named.forge", feature_only_with_id.name, "--json")
    assert without_columns.returncode == 1 and "'id'" in without_columns.stderr

    with_columns = _installed_cli(clean_install, outside_repo_dir, "model", "predict", "named.forge", feature_only_with_id.name,
                                   "--columns", *_PIMA_FEATURES, "--json")
    ordered = _installed_cli(clean_install, outside_repo_dir, "model", "predict", "named.forge",
                              named_pima["features_only"].name, "--json")
    assert with_columns.returncode == ordered.returncode == 0, with_columns.stderr
    assert with_columns.stdout == ordered.stdout


def test_installed_cli_convert_feature_names_retrofit_changes_no_weights(clean_install, outside_repo_dir):
    """`convert --feature-names`: a metadata-only edit, proven by parameter-file hash equality inside the real wheel."""
    import zipfile

    shutil.copy(_PIMA_DIR / "diabetes.csv", outside_repo_dir / "diabetes.csv")
    script = outside_repo_dir / "consume_unnamed.py"
    script.write_text(_CSV_TRAINING_CONSUMER)
    ran = _run(clean_install, [str(script), "diabetes.csv", "unnamed.forge", "diabetes.csv"], outside_repo_dir)
    assert ran.returncode == 0, ran.stderr

    convert = _installed_cli(
        clean_install, outside_repo_dir, "model", "convert", "unnamed.forge", "--device", "cpu",
        "--output", "retrofit.forge", "--feature-names", *_PIMA_FEATURES,
    )
    assert convert.returncode == 0, convert.stderr

    def param_hashes(path):
        with zipfile.ZipFile(str(outside_repo_dir / path)) as zf:
            return {n: hashlib.sha256(zf.read(n)).hexdigest() for n in zf.namelist() if n != "metadata.json"}

    assert param_hashes("unnamed.forge") == param_hashes("retrofit.forge")

    inspected = _installed_cli(clean_install, outside_repo_dir, "model", "inspect", "retrofit.forge", "--json")
    assert json.loads(inspected.stdout)["input_feature_names"] == _PIMA_FEATURES

    # a wrong count is refused, and writes nothing
    bad = _installed_cli(
        clean_install, outside_repo_dir, "model", "convert", "unnamed.forge", "--device", "cpu",
        "--output", "bad.forge", "--feature-names", "a", "b",
    )
    assert bad.returncode == 1 and "input feature" in bad.stderr
    assert not (outside_repo_dir / "bad.forge").exists()


# -- CSV-to-artifact training from the installed wheel (Milestone 121) ----------------------------------------
#
# `train_tabular_classifier_csv()`/`train_tabular_regressor_csv()` and `forge model train` are new in this
# milestone; nothing above exercises them. Same arrangement as every other CSV test in this file: a real,
# id-bearing Pima CSV (an actual production-shaped file, not a rewritten one), a fresh consumer process in the
# clean venv (numpy + Pillow only, no pandas), run from a directory outside the repository.

_CSV_TRAINING_WRAPPER_CONSUMER = '''
import json, sys
import forge

assert "site-packages" in forge.__file__, forge.__file__
result = forge.train_tabular_classifier_csv(
    sys.argv[1], target="Outcome", path=sys.argv[2],
    columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
             "DiabetesPedigreeFunction", "Age"],
    classes=["no_diabetes", "diabetes"], missing_columns=[1, 2, 3, 4, 5], seed=0, epochs=15,
)
schema = forge.inspect_model(sys.argv[2]).input_schema
print(json.dumps({
    "file": forge.__file__, "features": result.features, "samples": result.samples,
    "classes": result.classes, "validation_accuracy": result.validation_accuracy,
    "validation_loss": result.validation_loss, "recorded_names": list(schema.feature_names),
}))
'''


@pytest.fixture()
def pima_with_id_csv(outside_repo_dir):
    """A real, id-bearing production-shaped Pima CSV -- the exact friction `columns=`/`--columns` close."""
    with open(_PIMA_DIR / "diabetes.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    out = outside_repo_dir / "diabetes_with_id.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["patient_id", *header])
        for i, row in enumerate(body):
            writer.writerow([10000 + i, *row])
    return out


def test_installed_forge_trains_directly_from_an_id_bearing_csv_with_the_new_wrapper(clean_install, outside_repo_dir, pima_with_id_csv):
    """`train_tabular_classifier_csv()` -- the M121 function itself -- trains straight from the id-bearing file."""
    script = outside_repo_dir / "consume_wrapper.py"
    script.write_text(_CSV_TRAINING_WRAPPER_CONSUMER)
    ran = _run(clean_install, [str(script), str(pima_with_id_csv), "wrapper.forge"], outside_repo_dir)
    assert ran.returncode == 0, ran.stderr
    report = json.loads(ran.stdout)
    assert "site-packages" in report["file"] and str(REPO_ROOT) not in report["file"]
    assert report["features"] == 8 and report["samples"] == 768
    assert report["classes"] == ["no_diabetes", "diabetes"]
    assert report["recorded_names"] == _PIMA_FEATURES

    # Identical to the same training done the M118/M120 way (load_csv + the unchanged array trainer) on the
    # id-free file, run in this (dev-tree) process.
    data = np.genfromtxt(_PIMA_DIR / "diabetes.csv", delimiter=",", skip_header=1)
    reference = forge.train_tabular_classifier(
        data[:, :-1], data[:, -1].astype(int), path=outside_repo_dir / "reference.forge",
        classes=["no_diabetes", "diabetes"], missing_columns=[1, 2, 3, 4, 5], seed=0, epochs=15,
    )
    assert report["validation_accuracy"] == reference.validation_accuracy
    assert report["validation_loss"] == pytest.approx(reference.validation_loss, rel=1e-6)


def test_installed_cli_model_train_excludes_the_id_and_matches_the_python_wrapper(clean_install, outside_repo_dir, pima_with_id_csv):
    """`forge model train` on the exact same id-bearing file the previous test used, compared to that function's own call."""
    script = outside_repo_dir / "consume_wrapper_reference.py"
    script.write_text('''
import json, sys
import forge
result = forge.train_tabular_classifier_csv(
    sys.argv[1], target="Outcome", path=sys.argv[2],
    columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
             "DiabetesPedigreeFunction", "Age"],
    seed=0, epochs=15,
)
print(json.dumps({"validation_accuracy": result.validation_accuracy, "features": result.features}))
''')
    reference = _run(clean_install, [str(script), str(pima_with_id_csv), "cli_reference.forge"], outside_repo_dir)
    assert reference.returncode == 0, reference.stderr
    reference_report = json.loads(reference.stdout)

    cli = _installed_cli(
        clean_install, outside_repo_dir, "model", "train", str(pima_with_id_csv), "--task", "classification",
        "--target", "Outcome", "--output", "cli.forge", "--columns", *_PIMA_FEATURES, "--seed", "0",
        "--epochs", "15", "--json",
    )
    assert cli.returncode == 0, cli.stderr
    cli_report = json.loads(cli.stdout)
    assert cli_report["validation_accuracy"] == reference_report["validation_accuracy"]
    assert cli_report["features"] == reference_report["features"] == 8

    # Fresh-process, third invocation: the CLI-trained artifact is a standalone, portable file.
    predictor_check = _run(
        clean_install,
        ["-c", "import forge; p = forge.load_predictor('cli.forge'); print(p.classes)"],
        outside_repo_dir,
    )
    assert predictor_check.returncode == 0 and "0" in predictor_check.stdout and "1" in predictor_check.stdout


_CSV_REGRESSOR_WRAPPER_CONSUMER = '''
import json, sys
import forge

result = forge.train_tabular_regressor_csv(
    sys.argv[1], target="median_house_value", path=sys.argv[2],
    columns=["income", "age", "rooms"], target_transform="standardize", seed=3, epochs=150,
)
print(json.dumps({
    "features": result.features, "validation_mse": result.validation_mse,
    "baseline_mse": result.baseline_mse, "has_target_transform": result.target_transform is not None,
}))
'''


def test_installed_forge_trains_a_standardized_regressor_from_an_id_bearing_csv(clean_install, outside_repo_dir):
    """`train_tabular_regressor_csv(..., target_transform='standardize')` from a realistic-scale, id-bearing CSV."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3)) * np.array([1000.0, 0.01, 5.0]) + np.array([500.0, 1.0, 20.0])
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    y = 2.0e5 + 1.0e5 * (z[:, 0] - 0.5 * z[:, 1]) + 3.0e3 * rng.normal(size=300)
    csv_path = outside_repo_dir / "housing_with_id.csv"
    with open(csv_path, "w", newline="") as fh:
        fh.write("parcel_id,median_house_value,income,age,rooms\n")
        for i, (target, row) in enumerate(zip(y, X)):
            fh.write(",".join([str(900000 + i), repr(float(target)), *(repr(float(v)) for v in row)]) + "\n")

    script = outside_repo_dir / "consume_regressor.py"
    script.write_text(_CSV_REGRESSOR_WRAPPER_CONSUMER)
    ran = _run(clean_install, [str(script), str(csv_path), "housing_wrapper.forge"], outside_repo_dir)
    assert ran.returncode == 0, ran.stderr
    report = json.loads(ran.stdout)
    assert report["features"] == 3 and report["has_target_transform"] is True
    assert report["baseline_mse"] > 1e9 and report["validation_mse"] > 1e3       # native dollars, not z-scores
    assert report["validation_mse"] < 0.2 * report["baseline_mse"]

    cli = _installed_cli(
        clean_install, outside_repo_dir, "model", "train", str(csv_path), "--task", "regression",
        "--target", "median_house_value", "--output", "housing_cli.forge", "--columns", "income", "age", "rooms",
        "--target-transform", "standardize", "--seed", "3", "--epochs", "150", "--json",
    )
    assert cli.returncode == 0, cli.stderr
    cli_report = json.loads(cli.stdout)
    assert cli_report["validation_mse"] == report["validation_mse"]


def test_installed_cli_model_train_reports_a_bad_invocation_without_a_traceback(clean_install, outside_repo_dir, pima_with_id_csv):
    cases = [
        (["missing.csv", "--task", "classification", "--target", "Outcome", "--output", "m.forge"], "input file not found"),
        ([str(pima_with_id_csv), "--task", "classification", "--target", "nope", "--output", "m.forge"], "not in the header"),
        ([str(pima_with_id_csv), "--task", "classification", "--target", "Outcome", "--output", "m.forge",
          "--columns", "does_not_exist"], "not in the header"),
    ]
    for args, expected in cases:
        result = _installed_cli(clean_install, outside_repo_dir, "model", "train", *args)
        assert result.returncode == 1 and result.stdout == ""
        assert result.stderr.startswith("Error: ") and "Traceback" not in result.stderr
        assert expected in result.stderr, result.stderr


# -- image-classification training from the installed wheel (Milestone 122) -----------------------------------
#
# `forge model train DIR --task image-classification` is new in this milestone -- the CLI door onto the
# already-installed-wheel-tested `forge.train_image_classifier()`. Same arrangement as every training test
# above: a fresh consumer process in the clean venv, run from a directory outside the repository, compared
# against the identical Python-API call for cross-API parity (the brief's own requirement: "the CLI must be
# an adapter, not a second implementation").

_IMAGE_TRAIN_KWARGS = dict(epochs=2, batch_size=4, seed=0, device="cpu")


def _make_image_classification_dir(root: Path) -> Path:
    data = root / "pets"
    for cls, base_fill in [("cat", 40), ("dog", 210)]:
        class_dir = data / cls
        class_dir.mkdir(parents=True)
        for i in range(8):
            Image.fromarray(np.full((16, 16, 3), (base_fill + i) % 256, dtype=np.uint8), mode="RGB").save(
                class_dir / f"{i}.png"
            )
    return data


def test_installed_cli_trains_an_image_classifier_and_matches_the_python_api(clean_install, outside_repo_dir):
    data_dir = _make_image_classification_dir(outside_repo_dir)

    script = outside_repo_dir / "consume_image_wrapper.py"
    script.write_text('''
import json, sys
import forge
result = forge.train_image_classifier(
    sys.argv[1], path=sys.argv[2], epochs=2, batch_size=4, seed=0, device="cpu",
    verbose=False,
)
print(json.dumps({
    "file": forge.__file__, "classes": result.classes, "dataset_size": result.dataset_size,
    "train_size": result.train_size, "val_size": result.val_size,
    "validation_accuracy": result.val_metrics.get("accuracy"),
}))
''')
    reference = _run(clean_install, [str(script), str(data_dir), "reference.forge"], outside_repo_dir)
    assert reference.returncode == 0, reference.stderr
    reference_report = json.loads(reference.stdout)
    assert "site-packages" in reference_report["file"] and str(REPO_ROOT) not in reference_report["file"]

    cli = _installed_cli(
        clean_install, outside_repo_dir, "model", "train", str(data_dir), "--task", "image-classification",
        "--output", "cli.forge", "--epochs", "2", "--batch-size", "4", "--seed", "0", "--device", "cpu", "--json",
    )
    assert cli.returncode == 0, cli.stderr
    cli_report = json.loads(cli.stdout)

    assert cli_report["task"] == "classification"
    assert cli_report["classes"] == reference_report["classes"] == ["cat", "dog"]
    assert cli_report["dataset_size"] == reference_report["dataset_size"] == 16
    assert cli_report["train_size"] == reference_report["train_size"]
    assert cli_report["val_size"] == reference_report["val_size"]
    assert cli_report["validation_accuracy"] == pytest.approx(reference_report["validation_accuracy"])

    # Fresh-process, third invocation: inspect/predict/evaluate on the CLI-trained artifact, no CLI reference at all.
    inspected = _installed_cli(clean_install, outside_repo_dir, "model", "inspect", "cli.forge", "--json")
    assert inspected.returncode == 0
    inspected_payload = json.loads(inspected.stdout)
    assert inspected_payload["task"] == "classification" and inspected_payload["classes"] == ["cat", "dog"]

    sample_image = next((data_dir / "cat").glob("*.png"))
    predicted = _installed_cli(clean_install, outside_repo_dir, "model", "predict", "cli.forge", str(sample_image), "--json")
    assert predicted.returncode == 0
    assert json.loads(predicted.stdout)["class"] in ["cat", "dog"]

    evaluated = _installed_cli(clean_install, outside_repo_dir, "model", "evaluate", "cli.forge", str(data_dir), "--json")
    assert evaluated.returncode == 0
    evaluated_payload = json.loads(evaluated.stdout)
    assert evaluated_payload["task"] == "classification" and evaluated_payload["samples"] == 16


def test_installed_cli_model_train_rejects_bad_image_classification_invocations_without_a_traceback(
    clean_install, outside_repo_dir,
):
    data_dir = _make_image_classification_dir(outside_repo_dir)
    csv_path = outside_repo_dir / "not_images.csv"
    csv_path.write_text("f0,f1,target\n0.1,0.2,0\n")

    cases = [
        ([str(csv_path), "--task", "image-classification", "--output", "m.forge"], "is not a directory"),
        ([str(data_dir), "--task", "image-classification", "--target", "label", "--output", "m.forge"], "--target"),
        ([str(data_dir), "--task", "image-classification", "--columns", "a", "--output", "m.forge"], "--columns"),
        ([str(data_dir), "--task", "classification", "--target", "label", "--output", "m.forge"], "image-classification"),
    ]
    for args, expected in cases:
        result = _installed_cli(clean_install, outside_repo_dir, "model", "train", *args)
        assert result.returncode == 1 and result.stdout == ""
        assert result.stderr.startswith("Error: ") and "Traceback" not in result.stderr
        assert expected in result.stderr, result.stderr
