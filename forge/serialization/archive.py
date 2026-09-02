"""The on-disk archive format: a ZIP containing `metadata.json` plus one
`.npy` file per array entry.

Chosen over a single custom binary format because both pieces stay
independently inspectable with only the standard library (`unzip`, a text
editor, `numpy.load`) and neither requires executing anything to read:
`json.loads` cannot produce code, and every array is loaded via
`numpy.load(..., allow_pickle=False)`, which refuses to unpickle an object
array. Saving is atomic -- the archive is written to a temporary file in the
destination directory and only `os.replace`d into place once writing
succeeds, so a failed save never leaves a partially written file.

Shared by both `forge.serialization.model` (`save_model`/`load_model`) and
`forge.serialization.checkpoint` (`save_checkpoint`/`load_checkpoint`):
`write_archive`/`read_archive` know nothing about parameters, optimizers, or
any particular metadata shape -- `arrays` is keyed by an arbitrary
caller-chosen path (no leading/trailing slash, no `.npy` suffix -- this
function adds it), letting each caller lay out its own directory structure
inside the one shared ZIP container (e.g. `model.py` uses flat
`parameters/<dotted.name>` keys; `checkpoint.py` additionally uses
`optimizer/state/<dotted.name>/<field>` keys in the same archive).
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from typing import Any

import numpy as np

from ..exceptions import PersistenceError

METADATA_ENTRY = "metadata.json"
PARAMETERS_DIR = "parameters"


def write_archive(path: str, metadata: dict, arrays: "dict[str, np.ndarray]") -> None:
    """Write `metadata` as JSON plus one `.npy` per `arrays` entry to `path`.

    Each `arrays` key is used verbatim as the entry's path inside the ZIP
    (with `.npy` appended) -- callers namespace their own arrays by choosing
    keys, e.g. `"parameters/fc1.weight"` or `"optimizer/state/fc1.weight/m"`.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    if not os.path.isdir(directory):
        raise PersistenceError(
            f"Cannot save to '{path}': directory '{directory}' does not exist."
        )

    fd, tmp_path = tempfile.mkstemp(prefix=".forge-save-", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(METADATA_ENTRY, json.dumps(metadata, indent=2, sort_keys=True))
            for name, array in arrays.items():
                buffer = io.BytesIO()
                np.save(buffer, array, allow_pickle=False)
                zf.writestr(f"{name}.npy", buffer.getvalue())
        os.replace(tmp_path, path)
    except OSError as exc:
        raise PersistenceError(f"Failed to save to '{path}': {exc}") from exc
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def read_archive(path: str, *, kind: str = "model") -> "tuple[Any, dict[str, np.ndarray]]":
    """Read `path` back into `(metadata, arrays)`, the inverse of `write_archive`.

    `arrays` is keyed by the same full entry path each was written under
    (minus the `.npy` suffix). `kind` is only used to phrase error messages
    (`"model"` or `"checkpoint"`).
    """
    if not os.path.isfile(path):
        raise PersistenceError(f"Cannot load {kind}: file not found at '{path}'.")

    try:
        zf = zipfile.ZipFile(path, mode="r")
    except zipfile.BadZipFile as exc:
        raise PersistenceError(
            f"Cannot load {kind} from '{path}': not a valid Forge {kind} file (corrupt archive)."
        ) from exc

    with zf:
        try:
            raw_metadata = zf.read(METADATA_ENTRY)
        except KeyError as exc:
            raise PersistenceError(
                f"Cannot load {kind} from '{path}': missing '{METADATA_ENTRY}' entry."
            ) from exc
        try:
            metadata = json.loads(raw_metadata)
        except json.JSONDecodeError as exc:
            raise PersistenceError(
                f"Cannot load {kind} from '{path}': malformed metadata ({exc})."
            ) from exc

        arrays: "dict[str, np.ndarray]" = {}
        for entry in zf.namelist():
            if entry == METADATA_ENTRY or not entry.endswith(".npy"):
                continue
            name = entry[: -len(".npy")]
            try:
                data = zf.read(entry)
                arrays[name] = np.load(io.BytesIO(data), allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise PersistenceError(
                    f"Cannot load {kind} from '{path}': corrupt array data for '{name}' ({exc})."
                ) from exc

    return metadata, arrays


__all__ = ["write_archive", "read_archive", "METADATA_ENTRY", "PARAMETERS_DIR"]
