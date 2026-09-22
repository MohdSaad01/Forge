"""Milestone 122 tests: `forge model train DIR --task image-classification --output PATH`.

`forge model train` (Milestone 121) trained only from a `.csv` file, through the two tabular
Python APIs. Milestone 122 gives it a third route: `--task image-classification` dispatches the
identical `data`/`--output`/`--device`/`--epochs`/`--batch-size`/`--learning-rate`/`--seed`/`--json`
surface to the existing `forge.train_image_classifier()` (`forge/cli/model.py::_cmd_train_image`),
exactly as `--task classification`/`--task regression` already dispatch to
`train_tabular_classifier_csv()`/`train_tabular_regressor_csv()` (`_cmd_train_tabular`, unchanged).
No new training implementation is introduced; these tests exercise CLI routing, validation and
`--json`, not training numerics (already covered by `tests/test_image_classifier.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

import forge
import forge.cli.model as cli_model
from forge.cli.main import main


def _make_image(path: Path, size=(64, 64), color=(10, 20, 30)) -> None:
    Image.new("RGB", size, color).save(path)


def _make_two_class_root(tmp_path: Path, per_class: int = 8, size=(64, 64)) -> Path:
    root = tmp_path / "data"
    for cls, base_color in [("cat", (200, 0, 0)), ("dog", (0, 200, 0))]:
        class_dir = root / cls
        class_dir.mkdir(parents=True)
        for i in range(per_class):
            _make_image(class_dir / f"{cls}_{i}.png", size=size, color=(base_color[0], base_color[1], i * 5))
    return root


def run(argv, capsys):
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


TRAIN_ARGS = ["--epochs", "2", "--batch-size", "4", "--seed", "0"]


# --------------------------------------------------------------------------------------- happy path


def test_cli_trains_an_image_classifier_from_a_directory(tmp_path, capsys):
    root = _make_two_class_root(tmp_path)
    output = tmp_path / "pets.forge"
    code, out, err = run(
        ["model", "train", str(root), "--task", "image-classification", "--output", str(output), *TRAIN_ARGS],
        capsys,
    )
    assert code == 0 and err == ""
    assert output.is_file()
    assert "Trained classification model" in out
    assert "Classes: cat, dog" in out
    assert "Epochs completed: 2" in out
    assert "Validation accuracy:" in out

    info = forge.inspect_model(str(output))
    assert info.task == "classification"
    assert info.classes == ["cat", "dog"]


def test_cli_trains_an_image_classifier_and_prints_json(tmp_path, capsys):
    root = _make_two_class_root(tmp_path)
    output = tmp_path / "pets.forge"
    code, out, err = run(
        ["model", "train", str(root), "--task", "image-classification", "--output", str(output),
         *TRAIN_ARGS, "--json"],
        capsys,
    )
    assert code == 0 and err == ""
    payload = json.loads(out)
    assert payload["task"] == "classification"
    assert payload["artifact_path"] == str(output)
    assert payload["classes"] == ["cat", "dog"]
    assert payload["dataset_size"] == 16
    assert payload["train_size"] + payload["val_size"] == payload["dataset_size"]
    assert payload["epochs_completed"] == 2
    assert 0.0 <= payload["validation_accuracy"] <= 1.0
    assert payload["skipped_images"] == 0


# --------------------------------------------------------------------------------------- routing / mutation kills


def test_cli_image_classification_never_calls_the_tabular_trainers(tmp_path, monkeypatch, capsys):
    """Kills a mutation that routes --task image-classification to the tabular trainer(s)."""
    root = _make_two_class_root(tmp_path)
    output = tmp_path / "pets.forge"

    def fail(*args, **kwargs):
        raise AssertionError("tabular trainer must not be called for --task image-classification")

    monkeypatch.setattr(cli_model, "train_tabular_classifier_csv", fail)
    monkeypatch.setattr(cli_model, "train_tabular_regressor_csv", fail)

    code, out, err = run(
        ["model", "train", str(root), "--task", "image-classification", "--output", str(output), *TRAIN_ARGS],
        capsys,
    )
    assert code == 0 and err == ""


def test_cli_tabular_classification_never_calls_the_image_trainer(tmp_path, monkeypatch, capsys):
    """Symmetric check: --task classification must not reach train_image_classifier()."""
    def fail(*args, **kwargs):
        raise AssertionError("image trainer must not be called for --task classification")

    monkeypatch.setattr(cli_model, "train_image_classifier", fail)

    path = tmp_path / "data.csv"
    path.write_text("f0,f1,target\n0.1,0.2,0\n0.3,0.1,1\n0.2,0.4,0\n0.5,0.2,1\n")
    output = tmp_path / "m.forge"
    code, out, err = run(
        ["model", "train", str(path), "--task", "classification", "--target", "target",
         "--output", str(output), "--epochs", "3"],
        capsys,
    )
    assert code == 0 and err == ""


def test_cli_forwards_image_training_arguments_to_train_image_classifier(tmp_path, monkeypatch, capsys):
    """Kills mutations that ignore --device/--output/--epochs/--batch-size/--learning-rate/--seed."""
    root = _make_two_class_root(tmp_path)
    output = tmp_path / "pets.forge"
    real = cli_model.train_image_classifier
    captured = {}

    def spy(data_dir, *, path, **kwargs):
        captured["data_dir"] = data_dir
        captured["path"] = path
        captured.update(kwargs)
        return real(data_dir, path=path, **kwargs)

    monkeypatch.setattr(cli_model, "train_image_classifier", spy)

    code, out, err = run(
        ["model", "train", str(root), "--task", "image-classification", "--output", str(output),
         "--epochs", "2", "--batch-size", "4", "--learning-rate", "0.0005", "--seed", "7", "--device", "cpu"],
        capsys,
    )
    assert code == 0 and err == ""
    assert captured["data_dir"] == str(root)
    assert captured["path"] == str(output)
    assert captured["epochs"] == 2
    assert captured["batch_size"] == 4
    assert captured["learning_rate"] == 0.0005
    assert captured["seed"] == 7
    assert captured["device"] == "cpu"
    assert captured["verbose"] is False


def test_cli_image_classification_omitted_flags_use_the_python_apis_own_defaults(tmp_path, monkeypatch, capsys):
    root = _make_two_class_root(tmp_path, per_class=4)
    output = tmp_path / "pets.forge"
    real = cli_model.train_image_classifier
    captured = {}

    def spy(data_dir, *, path, **kwargs):
        captured.update(kwargs)
        # Override epochs down from the real default (5) purely to keep this test fast; every
        # other omitted kwarg is left exactly as the CLI forwarded it (none, here).
        kwargs["epochs"] = 1
        return real(data_dir, path=path, **kwargs)

    monkeypatch.setattr(cli_model, "train_image_classifier", spy)
    code, out, err = run(
        ["model", "train", str(root), "--task", "image-classification", "--output", str(output)], capsys,
    )
    assert code == 0 and err == ""
    # None of the optional flags were passed, so none should have been forwarded as kwargs --
    # an omitted CLI flag must reach train_image_classifier() as "not given", not as a
    # second, CLI-specific default value.
    assert captured == {"verbose": False}


# --------------------------------------------------------------------------------------- validation / invalid combos


def test_cli_image_classification_rejects_target(tmp_path, capsys):
    root = _make_two_class_root(tmp_path)
    code, out, err = run(
        ["model", "train", str(root), "--task", "image-classification", "--target", "label",
         "--output", str(tmp_path / "m.forge")],
        capsys,
    )
    assert code == 1 and out == ""
    assert err.startswith("Error: ") and "--target" in err and "image-classification" in err


def test_cli_image_classification_rejects_columns(tmp_path, capsys):
    root = _make_two_class_root(tmp_path)
    code, out, err = run(
        ["model", "train", str(root), "--task", "image-classification", "--columns", "a", "b",
         "--output", str(tmp_path / "m.forge")],
        capsys,
    )
    assert code == 1 and out == ""
    assert err.startswith("Error: ") and "--columns" in err and "image-classification" in err


def test_cli_image_classification_rejects_target_transform(tmp_path, capsys):
    root = _make_two_class_root(tmp_path)
    code, out, err = run(
        ["model", "train", str(root), "--task", "image-classification", "--target-transform", "standardize",
         "--output", str(tmp_path / "m.forge")],
        capsys,
    )
    assert code == 1 and out == ""
    assert err.startswith("Error: ") and "--target-transform applies only to --task regression" in err


def test_cli_image_classification_rejects_a_csv_file_as_input(tmp_path, capsys):
    path = tmp_path / "data.csv"
    path.write_text("f0,f1,target\n0.1,0.2,0\n")
    code, out, err = run(
        ["model", "train", str(path), "--task", "image-classification", "--output", str(tmp_path / "m.forge")],
        capsys,
    )
    assert code == 1 and out == ""
    assert err.startswith("Error: ") and "is not a directory" in err


def test_cli_image_classification_rejects_a_missing_directory(tmp_path, capsys):
    code, out, err = run(
        ["model", "train", str(tmp_path / "nope"), "--task", "image-classification",
         "--output", str(tmp_path / "m.forge")],
        capsys,
    )
    assert code == 1 and out == ""
    assert err.startswith("Error: ") and "is not a directory" in err


def test_cli_tabular_classification_rejects_a_directory_as_input(tmp_path, capsys):
    root = _make_two_class_root(tmp_path)
    code, out, err = run(
        ["model", "train", str(root), "--task", "classification", "--target", "target",
         "--output", str(tmp_path / "m.forge")],
        capsys,
    )
    assert code == 1 and out == ""
    assert err.startswith("Error: ") and "image-classification" in err


def test_cli_tabular_regression_rejects_a_directory_as_input(tmp_path, capsys):
    root = _make_two_class_root(tmp_path)
    code, out, err = run(
        ["model", "train", str(root), "--task", "regression", "--target", "target",
         "--output", str(tmp_path / "m.forge")],
        capsys,
    )
    assert code == 1 and out == ""
    assert err.startswith("Error: ") and "image-classification" in err


def test_cli_image_classification_output_directory_must_already_exist(tmp_path, capsys):
    root = _make_two_class_root(tmp_path)
    code, out, err = run(
        ["model", "train", str(root), "--task", "image-classification",
         "--output", str(tmp_path / "missing_dir" / "m.forge")],
        capsys,
    )
    assert code == 1 and out == ""
    assert err.startswith("Error: ") and "directory" in err


def test_cli_invalid_task_choice_still_an_argparse_error(tmp_path):
    root = _make_two_class_root(tmp_path)
    with pytest.raises(SystemExit):
        main(["model", "train", str(root), "--task", "nonsense", "--output", "m.forge"])


# --------------------------------------------------------------------------------------- artifact integration


def test_a_cli_trained_image_artifact_is_consumable_from_a_fresh_predictor(tmp_path, capsys):
    root = _make_two_class_root(tmp_path)
    output = tmp_path / "pets.forge"
    code, _, _ = run(
        ["model", "train", str(root), "--task", "image-classification", "--output", str(output), *TRAIN_ARGS],
        capsys,
    )
    assert code == 0

    predictor = forge.load_predictor(str(output))
    assert predictor.classes == ["cat", "dog"]

    sample_image = next((root / "cat").glob("*.png"))
    predict_code, predict_out, predict_err = run(
        ["model", "predict", str(output), str(sample_image)], capsys,
    )
    assert predict_code == 0 and predict_err == ""
    assert "Prediction:" in predict_out

    evaluate_code, evaluate_out, evaluate_err = run(
        ["model", "evaluate", str(output), str(root)], capsys,
    )
    assert evaluate_code == 0 and evaluate_err == ""
    assert "Accuracy:" in evaluate_out


def test_cli_image_classification_matches_the_python_api_for_the_same_seed(tmp_path):
    """The CLI must be an adapter, not a second implementation: identical inputs/seed -> identical artifact."""
    root = _make_two_class_root(tmp_path)
    cli_output = tmp_path / "cli.forge"
    api_output = tmp_path / "api.forge"

    code = main([
        "model", "train", str(root), "--task", "image-classification", "--output", str(cli_output), *TRAIN_ARGS,
    ])
    assert code == 0

    forge.train_image_classifier(
        str(root), path=str(api_output), epochs=2, batch_size=4, seed=0, device="cpu", verbose=False,
    )

    cli_model_obj = forge.load_model(str(cli_output))
    api_model_obj = forge.load_model(str(api_output))
    cli_params = [p.numpy() for p in cli_model_obj.parameters()]
    api_params = [p.numpy() for p in api_model_obj.parameters()]
    assert len(cli_params) == len(api_params)
    for a, b in zip(cli_params, api_params):
        assert (a == b).all()

    assert forge.inspect_model(str(cli_output)).classes == forge.inspect_model(str(api_output)).classes
