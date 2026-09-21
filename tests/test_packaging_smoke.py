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
