"""Reproducible-run bookkeeping for `examples/regression/train.py` (Milestone 65).

Not a general experiment-tracking framework -- three small, JSON-only pieces
built entirely from the standard library (`json`, `dataclasses.asdict`) and
Forge's *existing* public API. Milestone 65's baseline investigation
(`docs/development/m65-reproducible-training.md`) found every piece needed
for a reproducible-training workflow already present:

- `forge.training.EpochResult`/`EvaluationResult` are already plain,
  JSON-safe `@dataclass(frozen=True)` records -- `dataclasses.asdict()`
  converts either with no `forge.training` change required.
- `forge.serialization.save_checkpoint(..., extra=...)` already accepts an
  arbitrary caller-defined JSON-safe dict, persisted and restored across
  save/load -- exactly the mechanism `train.py` uses to save/restore the
  `DataLoader` shuffle generator's state (`numpy.random.Generator.
  bit_generator.state` is itself already a JSON-safe dict), closing the one
  genuine reproducibility gap the investigation found (see the module
  docstring of `train.py`'s **Determinism** section).

This module only adds: a small JSON "run record" sidecar file (config +
cumulative per-epoch history + final evaluation + total runtime) written
next to each run's checkpoint, and a plain-dict comparison between two such
records. No database, no visualization, no generic experiment-metadata
schema beyond what this one example needs.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from forge.training import EvaluationResult, TrainingHistory

RUN_RECORD_FORMAT_VERSION = 1


def history_to_dicts(history: TrainingHistory) -> "list[dict[str, Any]]":
    """Every `EpochResult` in `history` as a JSON-safe dict, in epoch order."""
    return [asdict(record) for record in history]


def evaluation_to_dict(result: EvaluationResult) -> "dict[str, Any]":
    """`EvaluationResult` as a JSON-safe dict."""
    return asdict(result)


def new_run_record(config: "dict[str, Any]") -> "dict[str, Any]":
    """A fresh run record for a from-scratch run, with `config` fixed at creation.

    `config` is never overwritten by a later resumed invocation (see
    `extend_run_record`) -- it records the configuration that produced this
    run's initialization and dataset, which `--epochs`/`--resume` alone
    don't change. The authoritative total epoch count is always
    `record["history"][-1]["epoch"]`, not any one invocation's `--epochs`.
    """
    now = time.time()
    return {
        "forge_run_record_format_version": RUN_RECORD_FORMAT_VERSION,
        "config": config,
        "history": [],
        "final_eval": None,
        "total_duration_seconds": 0.0,
        "created_at": now,
        "updated_at": now,
    }


def load_run_record(path: "str | Path") -> "dict[str, Any] | None":
    """Load a run record previously written by `save_run_record`, or `None` if `path` doesn't exist."""
    p = Path(path)
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as f:
        record = json.load(f)
    version = record.get("forge_run_record_format_version")
    if version != RUN_RECORD_FORMAT_VERSION:
        raise ValueError(
            f"Cannot load run record from '{path}': unsupported format version {version!r} "
            f"(this build supports version {RUN_RECORD_FORMAT_VERSION})."
        )
    return record


def save_run_record(path: "str | Path", record: "dict[str, Any]") -> None:
    """Write `record` to `path` as indented, sort-keyed JSON (diff-friendly, human-readable)."""
    record["updated_at"] = time.time()
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)


def extend_run_record(
    record: "dict[str, Any]",
    *,
    history: TrainingHistory,
    final_eval: "EvaluationResult | None",
    duration_seconds: float,
) -> "dict[str, Any]":
    """Append one `fit()` call's epochs (and optionally a final evaluation) onto `record`, in place.

    Called once per `train.py` invocation, whether from-scratch or resumed
    -- a resumed run's epochs are appended after whatever the checkpoint's
    originating run(s) already recorded, so `record["history"]` stays
    continuous across a resume exactly like `Trainer.epoch`/`global_step`
    already do.
    """
    record["history"].extend(history_to_dicts(history))
    if final_eval is not None:
        record["final_eval"] = evaluation_to_dict(final_eval)
    record["total_duration_seconds"] = record.get("total_duration_seconds", 0.0) + duration_seconds
    return record


def compare_runs(a: "dict[str, Any]", b: "dict[str, Any]") -> "dict[str, Any]":
    """A plain-dict diff of two run records: config/epoch/loss/metric/runtime deltas.

    Reports only *differing* config keys (a key present with the same value
    in both is omitted). No visualization -- this is the data `compare.py`
    formats as text.
    """
    config_a = a.get("config", {})
    config_b = b.get("config", {})
    config_diff = {
        key: (config_a.get(key), config_b.get(key))
        for key in sorted(set(config_a) | set(config_b))
        if config_a.get(key) != config_b.get(key)
    }

    history_a = a.get("history", [])
    history_b = b.get("history", [])
    final_eval_a = a.get("final_eval")
    final_eval_b = b.get("final_eval")

    return {
        "config_diff": config_diff,
        "epochs": {
            "a": history_a[-1]["epoch"] if history_a else 0,
            "b": history_b[-1]["epoch"] if history_b else 0,
        },
        "final_train_loss": {
            "a": history_a[-1]["train_loss"] if history_a else None,
            "b": history_b[-1]["train_loss"] if history_b else None,
        },
        "final_eval_loss": {
            "a": final_eval_a["loss"] if final_eval_a else None,
            "b": final_eval_b["loss"] if final_eval_b else None,
        },
        "final_eval_metrics": {
            "a": final_eval_a["metrics"] if final_eval_a else {},
            "b": final_eval_b["metrics"] if final_eval_b else {},
        },
        "total_duration_seconds": {
            "a": a.get("total_duration_seconds"),
            "b": b.get("total_duration_seconds"),
        },
    }


__all__ = [
    "RUN_RECORD_FORMAT_VERSION",
    "history_to_dicts",
    "evaluation_to_dict",
    "new_run_record",
    "load_run_record",
    "save_run_record",
    "extend_run_record",
    "compare_runs",
]
