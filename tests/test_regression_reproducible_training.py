"""Milestone 65 tests: the end-to-end reproducible-training workflow added to
`examples/regression/train.py`, driven exactly the way a user would (through
`main(argv)`, the same entry point `python -m examples.regression.train`
uses) rather than by reconstructing the pipeline from library calls, since
what is under test here is the CLI wiring itself -- argument parsing, the
`--resume` branch's `DataLoader` RNG restore, and run-record persistence --
not `Trainer`/`DataLoader` in isolation (those are already covered by
`tests/test_regression_example_integration.py`).

Covers Milestone 65's three "real validation" experiments (see
`docs/development/m65-reproducible-training.md` Section 12):
(A) same config from scratch twice -> identical initialization/training/
final parameters, (B) full training vs. partial-training-then-resume ->
identical final parameters, specifically with `shuffle=True` (the actual
`train.py` default) -- this is the case Milestone 65's baseline
investigation found broken before this milestone's `data_loader_rng_state`
fix, and (C) two different configurations -> the run record actually
distinguishes them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from forge.serialization import load_checkpoint, load_model  # noqa: E402

from examples.regression.compare import compare_runs  # noqa: E402
from examples.regression.experiment import load_run_record  # noqa: E402
from examples.regression.train import main as train_main  # noqa: E402

_SMALL = ["--n-train", "80", "--n-val", "20", "--n-test", "20", "--batch-size", "8"]


def _params(model_path: Path) -> "dict[str, np.ndarray]":
    model = load_model(str(model_path))
    return {name: p.numpy().copy() for name, p in model.named_parameters()}


def _assert_params_equal(a: "dict[str, np.ndarray]", b: "dict[str, np.ndarray]", *, atol: float) -> None:
    assert a.keys() == b.keys()
    for name in a:
        np.testing.assert_allclose(a[name], b[name], atol=atol, err_msg=f"parameter '{name}' diverged")


# -- (A) reproducibility: identical config from scratch, twice -----------------


def test_same_config_from_scratch_twice_is_bitwise_reproducible(tmp_path):
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    common = _SMALL + ["--epochs", "3", "--seed", "123", "--lr", "5e-3"]

    train_main(common + ["--output-dir", str(out_a)])
    train_main(common + ["--output-dir", str(out_b)])

    record_a = load_run_record(out_a / "regression_history.json")
    record_b = load_run_record(out_b / "regression_history.json")

    # Every deterministic field matches exactly; only wall-clock `duration`
    # (inherently non-deterministic timing, not a claimed-reproducible field)
    # is allowed to differ.
    for epoch_a, epoch_b in zip(record_a["history"], record_b["history"]):
        for key in ("epoch", "train_loss", "train_metrics", "val_loss", "val_metrics", "samples"):
            assert epoch_a[key] == epoch_b[key], f"epoch {epoch_a['epoch']} field '{key}' diverged"
    assert record_a["final_eval"]["loss"] == record_b["final_eval"]["loss"]
    assert record_a["final_eval"]["metrics"] == record_b["final_eval"]["metrics"]

    _assert_params_equal(
        _params(out_a / "regression_model.forge"), _params(out_b / "regression_model.forge"), atol=0.0
    )


# -- (B) resume equivalence with the actual shuffle=True default ---------------


def test_full_training_matches_partial_then_resume_with_shuffle(tmp_path):
    """The gap Milestone 65 closed: `train.py` always shuffles (no `--no-shuffle`
    flag), so before this milestone, resuming re-seeded a *fresh* DataLoader
    generator from `--seed` instead of continuing its stream -- a resumed run
    saw a different batch order than the interrupted run's continuation would
    have, and diverged from continuous training. This test is the permanent
    regression test for that fix.
    """
    out_full = tmp_path / "full"
    out_partial = tmp_path / "partial"
    base = _SMALL + ["--seed", "55", "--lr", "5e-3"]

    train_main(base + ["--epochs", "5", "--output-dir", str(out_full)])

    train_main(base + ["--epochs", "3", "--output-dir", str(out_partial)])
    checkpoint_path = out_partial / "regression_checkpoint.forge"
    train_main(
        base + ["--epochs", "2", "--resume", str(checkpoint_path), "--output-dir", str(out_partial)]
    )

    _assert_params_equal(
        _params(out_full / "regression_model.forge"),
        _params(out_partial / "regression_model.forge"),
        atol=1e-6,
    )

    # The run record stays continuous across the resume: 5 epochs total,
    # numbered 1..5, not restarted at 1 for the resumed invocation.
    record = load_run_record(out_partial / "regression_history.json")
    assert [e["epoch"] for e in record["history"]] == [1, 2, 3, 4, 5]


def test_checkpoint_extra_carries_dataloader_rng_state(tmp_path):
    out = tmp_path / "run"
    train_main(_SMALL + ["--epochs", "2", "--seed", "9", "--output-dir", str(out)])
    checkpoint = load_checkpoint(str(out / "regression_checkpoint.forge"))
    assert "data_loader_rng_state" in checkpoint.extra
    state = checkpoint.extra["data_loader_rng_state"]
    assert state["bit_generator"] == "PCG64"
    json.dumps(state)  # must be JSON-safe (it round-tripped through one already, but assert explicitly)


# -- (C) different configurations are distinguishable via the run record -------


def test_different_learning_rates_produce_distinguishable_run_records(tmp_path):
    out_lo = tmp_path / "lr_lo"
    out_hi = tmp_path / "lr_hi"
    base = _SMALL + ["--epochs", "3", "--seed", "1"]

    train_main(base + ["--lr", "1e-4", "--output-dir", str(out_lo)])
    train_main(base + ["--lr", "1e-1", "--output-dir", str(out_hi)])

    record_lo = load_run_record(out_lo / "regression_history.json")
    record_hi = load_run_record(out_hi / "regression_history.json")
    diff = compare_runs(record_lo, record_hi)

    assert diff["config_diff"]["lr"] == (1e-4, 1e-1)
    # A learning rate two orders of magnitude apart should also produce a
    # measurably different final loss -- the recorded metadata is not just
    # inert bookkeeping, it tracks a real behavioral difference.
    assert diff["final_train_loss"]["a"] != diff["final_train_loss"]["b"]


def test_resuming_without_an_existing_run_record_starts_a_fresh_one(tmp_path, capsys):
    """If the JSON sidecar is missing (e.g. deleted) but the checkpoint isn't,
    --resume still works -- it just starts a new run record from the resumed
    invocation's own config rather than raising, a documented limitation
    (see docs/development/m65-reproducible-training.md's Limitations section).
    """
    out = tmp_path / "run"
    train_main(_SMALL + ["--epochs", "2", "--seed", "3", "--output-dir", str(out)])
    (out / "regression_history.json").unlink()

    train_main(_SMALL + ["--epochs", "1", "--seed", "3", "--resume", str(out / "regression_checkpoint.forge"), "--output-dir", str(out)])

    record = load_run_record(out / "regression_history.json")
    assert record is not None
    assert [e["epoch"] for e in record["history"]] == [3]
