"""Shared fixtures for the Milestone 119 feature-schema tests (not a test module: nothing here is collected).

A tiny deterministic tabular problem whose four columns have very different scales and roles, so that
*a column in the wrong position visibly changes the prediction* -- after the artifact's persisted
`Normalize`, which is per-position, a swapped `income`/`age` is not a rounding difference. Every test
that claims "reordered input scores like correctly ordered input" is therefore non-vacuous, and the
tests also assert the precondition (`assert_order_matters`) so a fixture change cannot silently make
them vacuous.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import forge

FEATURES = ["income", "age", "rooms", "distance"]
CLASSES = ["low", "high"]
_SCALE = np.array([10.0, 3.0, 1.0, 50.0])
_CENTER = np.array([40.0, 30.0, 5.0, 100.0])


def make_data(n: int = 160, seed: int = 0):
    """`X` `(n, 4)` float64 in FEATURES order, regression target `y_reg`, class labels `y_cls` (0/1)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 4))
    X = z * _SCALE + _CENTER
    y_reg = 3.0 * z[:, 0] - 2.0 * z[:, 1] + 0.5 * z[:, 2] + 0.05 * rng.normal(size=n)
    y_cls = (z[:, 0] + z[:, 1] - 0.5 * z[:, 3] > 0).astype(np.int64)
    return X, y_reg, y_cls


def train_regressor(path, *, names="default", seed: int = 0, epochs: int = 12, **kwargs):
    X, y_reg, _ = make_data()
    names = FEATURES if names == "default" else names
    return forge.train_tabular_regressor(
        X, y_reg, path=str(path), seed=seed, epochs=epochs, patience=None, feature_names=names, **kwargs,
    )


def train_classifier(path, *, names="default", seed: int = 0, epochs: int = 12, **kwargs):
    X, _, y_cls = make_data()
    names = FEATURES if names == "default" else names
    return forge.train_tabular_classifier(
        X, y_cls, path=str(path), classes=CLASSES, seed=seed, epochs=epochs, patience=None,
        feature_names=names, **kwargs,
    )


def permute(X: np.ndarray, names: "list[str]", new_names: "list[str]") -> np.ndarray:
    """`X` (columns named `names`) re-laid-out so its columns are named `new_names` -- same values, moved."""
    return X[:, [names.index(n) for n in new_names]]


def rows(prediction):
    """A prediction as comparable plain data: regression -> ndarray, classification -> [(label, confidence)]."""
    if hasattr(prediction, "numpy"):
        return prediction.numpy()
    return [(p.label, p.confidence) for p in prediction]


def same_prediction(a, b) -> bool:
    a, b = rows(a), rows(b)
    return np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b


def assert_order_matters(predictor, X, names, new_names) -> None:
    """Precondition: WITHOUT names, feeding the permuted columns changes the prediction (so alignment is doing work)."""
    moved = permute(X, names, new_names)
    assert not same_prediction(predictor.predict(X), predictor.predict(moved)), (
        "fixture is degenerate: permuting the columns did not change the prediction"
    )


def write_named_csv(path: Path, header: "list[str]", X: np.ndarray, target: "tuple[str, np.ndarray] | None" = None) -> Path:
    """`X`'s columns under `header`, plus an optional `(name, values)` target column appended last."""
    names = list(header) + ([target[0]] if target else [])
    lines = [",".join(names)]
    for i in range(len(X)):
        cells = [repr(float(v)) for v in X[i]]
        if target:
            cells.append(str(target[1][i]) if target[1].dtype.kind in "iu" else repr(float(target[1][i])))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n")
    return path


def result_fields_equal(a, b) -> bool:
    """Every field of two evaluation-result dataclasses equal (arrays compared by value)."""
    import dataclasses

    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, (np.ndarray, list, tuple)) and not np.array_equal(np.asarray(x), np.asarray(y)):
            return False
        if not isinstance(x, (np.ndarray, list, tuple)) and x != y:
            return False
    return True


def metadata_of(path) -> dict:
    """The raw `metadata.json` of an artifact -- what is actually on disk, not what the API reports."""
    import json
    import zipfile

    with zipfile.ZipFile(str(path)) as zf:
        return json.loads(zf.read("metadata.json"))


def tamper(path, mutate) -> Path:
    """A copy of the artifact at `path` with its metadata edited by `mutate` -- a hand-edited / corrupted file."""
    import json
    import zipfile

    path = Path(path)
    with zipfile.ZipFile(str(path)) as zf:
        entries = {name: zf.read(name) for name in zf.namelist()}
    metadata = json.loads(entries["metadata.json"])
    mutate(metadata)                                    # edits in place; its return value (dict.update -> None, pop -> value) is ignored
    entries["metadata.json"] = json.dumps(metadata).encode("utf-8")
    out = path.parent / f"tampered_{len(list(path.parent.iterdir()))}_{path.name}"
    with zipfile.ZipFile(str(out), "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return out
