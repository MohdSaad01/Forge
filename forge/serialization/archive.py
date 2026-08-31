"""The on-disk archive format: a ZIP containing `metadata.json` plus one
`.npy` file per parameter.

Chosen over a single custom binary format because both pieces stay
independently inspectable with only the standard library (`unzip`, a text
editor, `numpy.load`) and neither requires executing anything to read:
`json.loads` cannot produce code, and every array is loaded via
`numpy.load(..., allow_pickle=False)`, which refuses to unpickle an object
array. Saving is atomic -- the archive is written to a temporary file in the
destination directory and only `os.replace`d into place once writing
succeeds, so a failed save never leaves a partially written model file.
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
    directory = os.path.dirname(os.path.abspath(path)) or "."
    if not os.path.isdir(directory):
        raise PersistenceError(
            f"Cannot save model to '{path}': directory '{directory}' does not exist."
        )

    fd, tmp_path = tempfile.mkstemp(prefix=".forge-save-", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(METADATA_ENTRY, json.dumps(metadata, indent=2, sort_keys=True))
            for name, array in arrays.items():
                buffer = io.BytesIO()
                np.save(buffer, array, allow_pickle=False)
                zf.writestr(f"{PARAMETERS_DIR}/{name}.npy", buffer.getvalue())
        os.replace(tmp_path, path)
    except OSError as exc:
        raise PersistenceError(f"Failed to save model to '{path}': {exc}") from exc
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def read_archive(path: str) -> "tuple[Any, dict[str, np.ndarray]]":
    if not os.path.isfile(path):
        raise PersistenceError(f"Cannot load model: file not found at '{path}'.")

    try:
        zf = zipfile.ZipFile(path, mode="r")
    except zipfile.BadZipFile as exc:
        raise PersistenceError(
            f"Cannot load model from '{path}': not a valid Forge model file (corrupt archive)."
        ) from exc

    with zf:
        try:
            raw_metadata = zf.read(METADATA_ENTRY)
        except KeyError as exc:
            raise PersistenceError(
                f"Cannot load model from '{path}': missing '{METADATA_ENTRY}' entry."
            ) from exc
        try:
            metadata = json.loads(raw_metadata)
        except json.JSONDecodeError as exc:
            raise PersistenceError(
                f"Cannot load model from '{path}': malformed metadata ({exc})."
            ) from exc

        arrays: "dict[str, np.ndarray]" = {}
        prefix = f"{PARAMETERS_DIR}/"
        for entry in zf.namelist():
            if entry.startswith(prefix) and entry.endswith(".npy"):
                dotted_name = entry[len(prefix) : -len(".npy")]
                try:
                    data = zf.read(entry)
                    arrays[dotted_name] = np.load(io.BytesIO(data), allow_pickle=False)
                except (OSError, ValueError) as exc:
                    raise PersistenceError(
                        f"Cannot load model from '{path}': corrupt parameter data for "
                        f"'{dotted_name}' ({exc})."
                    ) from exc

    return metadata, arrays


__all__ = ["write_archive", "read_archive", "METADATA_ENTRY", "PARAMETERS_DIR"]
