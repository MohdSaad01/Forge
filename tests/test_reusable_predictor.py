"""Milestone 102 tests: `forge.load_predictor()`/`forge.ArtifactPredictor`,
the reusable in-process inference workflow over a `.forge` artifact.

Before this milestone, `forge.predict_model()` (and each task-specific
`predict_*_artifact()` function beneath it) reopened and reconstructed the
entire artifact -- `inspect_model()`, `load_model()`, `load_preprocessing()`,
`load_classes()` -- on every single call, which is wasteful for an
application making many predictions from the same artifact.
`forge.load_predictor(path)` performs that loading exactly once and returns
an `ArtifactPredictor` whose `predict()` reruns only the genuinely per-call
work, by delegating to the same artifact-independent core functions the
existing one-shot functions themselves call.

Covers: loading semantics (success, and every failure condition
`predict_model()`/the task-specific functions already raise, now surfaced
once at load time rather than per call), no repeated artifact
reconstruction (instrumented call counts, not just timing), prediction
parity against the existing one-shot functions for every supported task
(classification/regression/segmentation/sequence/tabular_classification),
multiple predictions through one predictor instance, `InputSchema`
(Milestone 101) validation reuse, metadata exposure, artifact immutability,
and a genuine fresh-process run. See
`forge/training/inference.py::ArtifactPredictor`/`load_predictor()`.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import forge
from forge.data.transforms import Compose, Normalize, Resize
from forge.exceptions import DataError, PersistenceError
from forge.nn import Conv2d, Flatten, Linear, MaxPool2d, Module, RNNCell, ReLU, Sequential
from forge.serialization import register_module, save_model
from forge.training import (
    ArtifactPredictor,
    ClassificationPrediction,
    load_predictor,
    predict_artifact,
    predict_image_artifact,
    predict_model,
    predict_sequence_artifact,
    predict_tabular_classification_artifact,
    predict_tensor_artifact,
)
from forge.training.inference import load_predictor as load_predictor_direct

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESIZE_SIZE = (8, 8)
_N_FEATURES = 4
_VOCAB = list("abcde ")


# -- shared fixtures, one per real supported artifact shape -------------------


def _classification_transform():
    return Compose([Resize(_RESIZE_SIZE), Normalize(mean=0.0, std=255.0)])


def _classification_model(num_classes=2):
    return Sequential(
        Conv2d(3, 4, kernel_size=3, padding=1), ReLU(), MaxPool2d(kernel_size=2),
        Flatten(), Linear(4 * 4 * 4, num_classes),
    )


def _regression_transform():
    return Normalize(
        mean=np.array([1.0, -1.0, 0.5, 0.0], dtype=np.float32),
        std=np.array([2.0, 3.0, 1.0, 4.0], dtype=np.float32),
    )


def _regression_model():
    return Sequential(Linear(_N_FEATURES, 8), ReLU(), Linear(8, 1))


def _segmentation_transform():
    return Compose([Normalize(mean=0.0, std=255.0)])


def _segmentation_model():
    return Sequential(Conv2d(3, 4, kernel_size=3, padding=1), ReLU(), Conv2d(4, 1, kernel_size=3, padding=1))


def _tabular_classification_model(num_classes=3):
    return Sequential(Linear(_N_FEATURES, 8), ReLU(), Linear(8, num_classes))


def _make_image(path: Path, size=(8, 8), fill=100) -> None:
    arr = np.full((size[1], size[0], 3), fill, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _saved_classification_model(tmp_path, *, classes=("cat", "dog"), seed=0, name="classification.forge") -> Path:
    forge.random.seed(seed)
    model = _classification_model(num_classes=len(classes))
    path = tmp_path / name
    save_model(model, str(path), preprocessing=_classification_transform(), classes=list(classes), task="classification")
    return path


def _saved_regression_model(tmp_path, *, seed=0, name="regression.forge") -> Path:
    forge.random.seed(seed)
    model = _regression_model()
    path = tmp_path / name
    save_model(model, str(path), preprocessing=_regression_transform(), task="regression")
    return path


def _saved_segmentation_model(tmp_path, *, seed=0, name="segmentation.forge") -> Path:
    forge.random.seed(seed)
    model = _segmentation_model()
    path = tmp_path / name
    save_model(model, str(path), preprocessing=_segmentation_transform(), task="segmentation")
    return path


def _saved_tabular_classification_model(
    tmp_path, *, classes=("normal", "warning", "critical"), seed=0, name="tabular_classification.forge",
) -> Path:
    forge.random.seed(seed)
    model = _tabular_classification_model(num_classes=len(classes))
    path = tmp_path / name
    save_model(
        model, str(path), preprocessing=_regression_transform(), classes=list(classes),
        task="tabular_classification",
    )
    return path


class TinyStepModel(Module):
    """Mirrors `tests/test_sequence_artifact_prediction.py::TinyStepModel`
    exactly (duplicated, not imported, so this file has no cross-file test
    dependency -- the same convention that file itself documents)."""

    def __init__(self, vocab_size=6, hidden_size=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.cell = RNNCell(vocab_size, hidden_size)
        self.output = Linear(hidden_size, vocab_size)

    def step(self, x, h):
        h = self.cell(x, h)
        return self.output(h), h

    def init_hidden(self, batch_size, device="cpu"):
        return self.cell.init_hidden(batch_size, device=device)


register_module(
    "TinyStepModelM102",
    TinyStepModel,
    get_config=lambda m: {"vocab_size": m.vocab_size, "hidden_size": m.hidden_size},
)


def _saved_sequence_model(tmp_path, *, vocab=None, seed=0, name="sequence.forge") -> Path:
    vocab = list(_VOCAB) if vocab is None else vocab
    forge.random.seed(seed)
    model = TinyStepModel(vocab_size=len(vocab))
    path = tmp_path / name
    save_model(model, str(path), classes=vocab, task="sequence")
    return path


# -- basic API shape ----------------------------------------------------------


def test_load_predictor_is_reexported_consistently():
    assert forge.load_predictor is load_predictor
    assert forge.training.load_predictor is load_predictor
    assert load_predictor is load_predictor_direct
    assert forge.ArtifactPredictor is ArtifactPredictor


def test_artifact_predictor_repr_includes_path_and_task(tmp_path):
    model_path = _saved_regression_model(tmp_path)
    predictor = load_predictor(str(model_path))
    text = repr(predictor)
    assert repr(str(model_path)) in text
    assert "regression" in text


# -- loading: success, one per task shape -------------------------------------


def test_load_predictor_loads_classification_artifact(tmp_path):
    model_path = _saved_classification_model(tmp_path)
    predictor = load_predictor(str(model_path))
    assert isinstance(predictor, ArtifactPredictor)
    assert predictor.task == "classification"
    assert predictor.classes == ["cat", "dog"]
    assert predictor.input_schema is None


def test_load_predictor_loads_regression_artifact(tmp_path):
    model_path = _saved_regression_model(tmp_path)
    predictor = load_predictor(str(model_path))
    assert predictor.task == "regression"
    assert predictor.classes is None
    assert predictor.input_schema.feature_count == _N_FEATURES


def test_load_predictor_loads_segmentation_artifact(tmp_path):
    model_path = _saved_segmentation_model(tmp_path)
    predictor = load_predictor(str(model_path))
    assert predictor.task == "segmentation"
    assert predictor.input_schema is None


def test_load_predictor_loads_tabular_classification_artifact(tmp_path):
    model_path = _saved_tabular_classification_model(tmp_path)
    predictor = load_predictor(str(model_path))
    assert predictor.task == "tabular_classification"
    assert predictor.classes == ["normal", "warning", "critical"]
    assert predictor.input_schema.feature_count == _N_FEATURES


def test_load_predictor_loads_sequence_artifact(tmp_path):
    model_path = _saved_sequence_model(tmp_path)
    predictor = load_predictor(str(model_path))
    assert predictor.task == "sequence"
    assert predictor.classes == _VOCAB
    assert predictor.input_schema is None


def test_load_predictor_model_property_is_the_loaded_module(tmp_path):
    model_path = _saved_regression_model(tmp_path)
    predictor = load_predictor(str(model_path))
    assert isinstance(predictor.model, Module)


# -- loading: failure conditions ----------------------------------------------


def test_load_predictor_missing_file_raises_persistence_error(tmp_path):
    with pytest.raises(PersistenceError):
        load_predictor(str(tmp_path / "nope.forge"))


def test_load_predictor_corrupt_file_raises_persistence_error(tmp_path):
    bad_path = tmp_path / "corrupt.forge"
    bad_path.write_bytes(b"not a real archive")
    with pytest.raises(PersistenceError):
        load_predictor(str(bad_path))


def test_load_predictor_raises_clearly_for_an_undeterminable_artifact(tmp_path):
    forge.random.seed(0)
    model = Sequential(ReLU())
    model_path = tmp_path / "model.forge"
    save_model(model, str(model_path))
    with pytest.raises(PersistenceError, match="could not determine"):
        load_predictor(str(model_path))


def test_load_predictor_classification_without_preprocessing_raises_persistence_error(tmp_path):
    forge.random.seed(0)
    model = _classification_model()
    model_path = tmp_path / "model.forge"
    save_model(model, str(model_path), classes=["cat", "dog"], task="classification")
    with pytest.raises(PersistenceError, match="no automatic way to prepare"):
        load_predictor(str(model_path))


def test_load_predictor_segmentation_without_preprocessing_raises_persistence_error(tmp_path):
    forge.random.seed(0)
    model = _segmentation_model()
    model_path = tmp_path / "model.forge"
    save_model(model, str(model_path), task="segmentation")
    with pytest.raises(PersistenceError, match="no automatic way to prepare"):
        load_predictor(str(model_path))


def test_load_predictor_sequence_without_stepwise_protocol_raises_persistence_error(tmp_path):
    forge.random.seed(0)
    model = Sequential(Linear(len(_VOCAB), len(_VOCAB)), ReLU())
    model_path = tmp_path / "not_stepwise.forge"
    save_model(model, str(model_path), classes=list(_VOCAB), task="sequence")
    with pytest.raises(PersistenceError, match="stepwise-recurrence protocol"):
        load_predictor(str(model_path))


# -- reuse: no repeated artifact reconstruction --------------------------------


def test_predictor_loads_the_artifact_exactly_once(tmp_path, monkeypatch):
    """The core Milestone 102 requirement: `predict()` must not reopen or
    reconstruct the artifact. Instrument the loading primitives directly
    (call counts), not just timing."""
    import forge.serialization.model as model_module

    model_path = _saved_tabular_classification_model(tmp_path)
    batch = np.random.default_rng(0).standard_normal((1, _N_FEATURES)).astype(np.float32)

    counts = {"inspect_model": 0, "load_model": 0, "load_preprocessing": 0, "load_classes": 0}
    for name in counts:
        original = getattr(model_module, name)

        def _counted(*args, __name=name, __original=original, **kwargs):
            counts[__name] += 1
            return __original(*args, **kwargs)

        monkeypatch.setattr(model_module, name, _counted)

    predictor = load_predictor(str(model_path))
    assert counts == {"inspect_model": 1, "load_model": 1, "load_preprocessing": 1, "load_classes": 1}

    for _ in range(5):
        predictor.predict(batch)

    # None of the four loading primitives fire again for any of the five predictions.
    assert counts == {"inspect_model": 1, "load_model": 1, "load_preprocessing": 1, "load_classes": 1}


def test_predictor_validation_reuses_cached_input_schema_no_reinspection(tmp_path, monkeypatch):
    import forge.serialization.model as model_module

    model_path = _saved_regression_model(tmp_path)
    predictor = load_predictor(str(model_path))

    calls = {"inspect_model": 0}
    original = model_module.inspect_model

    def _counted(*args, **kwargs):
        calls["inspect_model"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model_module, "inspect_model", _counted)

    good = np.zeros((1, _N_FEATURES), dtype=np.float32)
    bad = np.zeros((1, _N_FEATURES + 1), dtype=np.float32)

    predictor.predict(good)
    with pytest.raises(DataError):
        predictor.predict(bad)
    predictor.predict(good)

    assert calls["inspect_model"] == 0


# -- prediction parity with the existing one-shot functions --------------------


def test_predictor_predict_matches_predict_artifact_bit_for_bit(tmp_path):
    model_path = _saved_classification_model(tmp_path)
    image_path = tmp_path / "query.png"
    _make_image(image_path, fill=200)

    predictor = load_predictor(str(model_path))
    via_predictor = predictor.predict(str(image_path))
    direct = predict_artifact(str(model_path), str(image_path))

    assert via_predictor.label == direct.label
    assert via_predictor.index == direct.index
    assert via_predictor.confidence == pytest.approx(direct.confidence, abs=1e-6)


def test_predictor_predict_matches_predict_tensor_artifact_bit_for_bit(tmp_path):
    model_path = _saved_regression_model(tmp_path)
    batch = np.random.default_rng(2).standard_normal((5, _N_FEATURES)).astype(np.float32)

    predictor = load_predictor(str(model_path))
    via_predictor = predictor.predict(batch)
    direct = predict_tensor_artifact(str(model_path), batch)

    np.testing.assert_array_equal(via_predictor.numpy(), direct.numpy())


def test_predictor_predict_matches_predict_image_artifact_bit_for_bit(tmp_path):
    model_path = _saved_segmentation_model(tmp_path)
    image_path = tmp_path / "query.png"
    _make_image(image_path, fill=210)

    predictor = load_predictor(str(model_path))
    via_predictor = predictor.predict(str(image_path))
    direct = predict_image_artifact(str(model_path), str(image_path))

    np.testing.assert_array_equal(via_predictor.numpy(), direct.numpy())


def test_predictor_predict_matches_predict_tabular_classification_artifact_bit_for_bit(tmp_path):
    model_path = _saved_tabular_classification_model(tmp_path)
    batch = np.random.default_rng(6).standard_normal((4, _N_FEATURES)).astype(np.float32)

    predictor = load_predictor(str(model_path))
    via_predictor = predictor.predict(batch)
    direct = predict_tabular_classification_artifact(str(model_path), batch)

    assert len(via_predictor) == len(direct) == 4
    for u, d in zip(via_predictor, direct):
        assert u.label == d.label
        assert u.index == d.index
        assert u.confidence == pytest.approx(d.confidence, abs=1e-6)


def test_predictor_predict_matches_predict_sequence_artifact_with_same_rng(tmp_path):
    model_path = _saved_sequence_model(tmp_path)
    predictor = load_predictor(str(model_path))

    via_predictor = predictor.predict(["a", "b"], length=15, rng=np.random.default_rng(7))
    direct = predict_sequence_artifact(str(model_path), ["a", "b"], length=15, rng=np.random.default_rng(7))

    assert via_predictor == direct


# -- multiple predictions through one predictor instance -----------------------


def test_predictor_multiple_predictions_agree_with_independent_one_shot_calls(tmp_path):
    model_path = _saved_tabular_classification_model(tmp_path)
    rows = np.random.default_rng(9).standard_normal((3, _N_FEATURES)).astype(np.float32)

    predictor = load_predictor(str(model_path))
    for i in range(3):
        row = rows[i : i + 1]
        via_predictor = predictor.predict(row)[0]
        expected = predict_model(str(model_path), row)[0]
        assert via_predictor.label == expected.label
        assert via_predictor.index == expected.index
        assert via_predictor.confidence == pytest.approx(expected.confidence, abs=1e-6)


def test_predictor_multiple_image_predictions_agree_with_independent_one_shot_calls(tmp_path):
    model_path = _saved_classification_model(tmp_path)
    fills = [50, 120, 200]

    predictor = load_predictor(str(model_path))
    for fill in fills:
        image_path = tmp_path / f"query_{fill}.png"
        _make_image(image_path, fill=fill)
        via_predictor = predictor.predict(str(image_path))
        expected = predict_model(str(model_path), str(image_path))
        assert via_predictor.label == expected.label
        assert via_predictor.index == expected.index


# -- input validation reuse (Milestone 101 integration) ------------------------


def test_predictor_rejects_wrong_feature_count_regression(tmp_path):
    model_path = _saved_regression_model(tmp_path)
    predictor = load_predictor(str(model_path))
    with pytest.raises(DataError):
        predictor.predict(np.zeros((1, _N_FEATURES - 1), dtype=np.float32))


def test_predictor_rejects_wrong_feature_count_tabular_classification(tmp_path):
    model_path = _saved_tabular_classification_model(tmp_path)
    predictor = load_predictor(str(model_path))
    with pytest.raises(DataError):
        predictor.predict(np.zeros((1, _N_FEATURES + 1), dtype=np.float32))


def test_predictor_accepts_valid_feature_count_after_a_rejected_call(tmp_path):
    """Validation failure must not leave the predictor in a broken state --
    the same cached model/preprocessing keep working for the next, valid call."""
    model_path = _saved_regression_model(tmp_path)
    predictor = load_predictor(str(model_path))
    with pytest.raises(DataError):
        predictor.predict(np.zeros((1, _N_FEATURES + 3), dtype=np.float32))
    result = predictor.predict(np.zeros((1, _N_FEATURES), dtype=np.float32))
    assert isinstance(result, forge.Tensor)


# -- error handling: dispatch / input type -------------------------------------


def test_predictor_predict_requires_length_for_sequence_artifact(tmp_path):
    model_path = _saved_sequence_model(tmp_path)
    predictor = load_predictor(str(model_path))
    with pytest.raises(DataError, match="length="):
        predictor.predict(["a", "b"])


def test_predictor_predict_rejects_a_tensor_for_a_classification_artifact(tmp_path):
    model_path = _saved_classification_model(tmp_path)
    predictor = load_predictor(str(model_path))
    with pytest.raises(DataError):
        predictor.predict(forge.Tensor(np.zeros((3, 8, 8), dtype=np.float32)))


def test_predictor_predict_rejects_an_image_path_for_a_regression_artifact(tmp_path):
    model_path = _saved_regression_model(tmp_path)
    image_path = tmp_path / "query.png"
    _make_image(image_path)
    predictor = load_predictor(str(model_path))
    with pytest.raises(DataError):
        predictor.predict(str(image_path))


def test_predictor_predict_rejects_empty_sequence_seed(tmp_path):
    model_path = _saved_sequence_model(tmp_path)
    predictor = load_predictor(str(model_path))
    with pytest.raises(DataError):
        predictor.predict([], length=5)


def test_predictor_predict_rejects_unknown_sequence_token(tmp_path):
    model_path = _saved_sequence_model(tmp_path)
    predictor = load_predictor(str(model_path))
    with pytest.raises(DataError, match="not in"):
        predictor.predict(["a", "Z"], length=5)


# -- artifact immutability -----------------------------------------------------


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_predictor_repeated_predictions_do_not_modify_the_artifact_file(tmp_path):
    model_path = _saved_tabular_classification_model(tmp_path)
    before = _hash(model_path)

    predictor = load_predictor(str(model_path))
    mid = _hash(model_path)
    batch = np.random.default_rng(0).standard_normal((2, _N_FEATURES)).astype(np.float32)
    for _ in range(5):
        predictor.predict(batch)
    after = _hash(model_path)

    assert before == mid == after


# -- fresh process --------------------------------------------------------------


def test_predictor_works_from_a_genuinely_separate_process_with_multiple_predictions(tmp_path):
    model_path = _saved_tabular_classification_model(tmp_path, seed=21)
    rows = np.random.default_rng(22).standard_normal((3, _N_FEATURES)).astype(np.float32)

    script = (
        "import numpy as np\n"
        "import forge\n"
        f"rows = np.array({rows.tolist()!r}, dtype=np.float32)\n"
        f"predictor = forge.load_predictor({str(model_path)!r})\n"
        "for i in range(rows.shape[0]):\n"
        "    result = predictor.predict(rows[i : i + 1])[0]\n"
        "    print(f'row {i}: label={result.label} index={result.index} confidence={result.confidence:.6f}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"

    predictor = load_predictor(str(model_path))
    for i in range(rows.shape[0]):
        expected = predictor.predict(rows[i : i + 1])[0]
        assert (
            f"row {i}: label={expected.label} index={expected.index}" in result.stdout
        )
