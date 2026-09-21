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
