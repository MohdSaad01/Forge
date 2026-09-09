"""Milestone 65 tests: `examples/regression/experiment.py`'s run-record and
comparison helpers, and `examples/regression/compare.py`'s CLI.

These exercise only the small JSON-only bookkeeping M65 added on top of
Forge's existing `TrainingHistory`/`EvaluationResult`/checkpoint `extra`
mechanisms -- no training is run here (see
`tests/test_regression_reproducible_training.py` for the end-to-end
reproducibility/resume tests that actually train). Mirrors
`tests/test_regression_example_integration.py`'s sys.path bootstrap for
importing the `examples` package.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from forge.training.trainer import EpochResult, EvaluationResult, TrainingHistory  # noqa: E402

from examples.regression import compare as compare_cli  # noqa: E402
from examples.regression.experiment import (  # noqa: E402
    compare_runs,
    evaluation_to_dict,
    extend_run_record,
    history_to_dicts,
    load_run_record,
    new_run_record,
    save_run_record,
)


def _epoch(epoch: int, train_loss: float) -> EpochResult:
    return EpochResult(
        epoch=epoch,
        train_loss=train_loss,
        train_metrics={"mae": train_loss / 2},
        val_loss=train_loss * 0.9,
        val_metrics={"mae": train_loss * 0.4},
        duration=0.01,
        samples=32,
        device="cpu",
    )


def _history(losses: "list[float]") -> TrainingHistory:
    history = TrainingHistory()
    for i, loss in enumerate(losses, start=1):
        history.append(_epoch(i, loss))
    return history


def _eval(loss: float) -> EvaluationResult:
    return EvaluationResult(loss=loss, metrics={"mae": loss * 0.5}, samples=50, duration=0.005, device="cpu")


# -- history/evaluation -> JSON-safe dicts -------------------------------------


def test_history_to_dicts_is_json_safe_and_preserves_epoch_order():
    history = _history([10.0, 8.0, 6.0])
    dicts = history_to_dicts(history)
    assert [d["epoch"] for d in dicts] == [1, 2, 3]
    assert [d["train_loss"] for d in dicts] == [10.0, 8.0, 6.0]
    json.dumps(dicts)  # must not raise


def test_evaluation_to_dict_is_json_safe():
    result = _eval(1.5)
    d = evaluation_to_dict(result)
    assert d == {"loss": 1.5, "metrics": {"mae": 0.75}, "samples": 50, "duration": 0.005, "device": "cpu"}
    json.dumps(d)


# -- run record round trip -----------------------------------------------------


def test_new_run_record_has_expected_shape():
    record = new_run_record({"lr": 0.001, "seed": 0})
    assert record["config"] == {"lr": 0.001, "seed": 0}
    assert record["history"] == []
    assert record["final_eval"] is None
    assert record["total_duration_seconds"] == 0.0


def test_save_and_load_run_record_round_trips(tmp_path):
    record = new_run_record({"lr": 0.001})
    extend_run_record(record, history=_history([5.0, 4.0]), final_eval=_eval(3.0), duration_seconds=1.25)

    path = tmp_path / "history.json"
    save_run_record(path, record)
    loaded = load_run_record(path)

    assert loaded["config"] == {"lr": 0.001}
    assert [e["epoch"] for e in loaded["history"]] == [1, 2]
    assert loaded["final_eval"]["loss"] == 3.0
    assert loaded["total_duration_seconds"] == 1.25


def test_load_run_record_returns_none_for_missing_file(tmp_path):
    assert load_run_record(tmp_path / "does_not_exist.json") is None


def test_load_run_record_rejects_unsupported_format_version(tmp_path):
    path = tmp_path / "bad_version.json"
    path.write_text(json.dumps({"forge_run_record_format_version": 999}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported format version"):
        load_run_record(path)


def test_extend_run_record_accumulates_history_across_multiple_calls_and_keeps_original_config():
    record = new_run_record({"lr": 0.001, "epochs": 3})
    extend_run_record(record, history=_history([5.0, 4.0, 3.0]), final_eval=_eval(2.9), duration_seconds=1.0)
    # A "resumed" invocation extends the same record with more epochs; config
    # is not overwritten by the resumed run's own (different) --epochs value.
    extend_run_record(record, history=_history([2.0, 1.0]), final_eval=_eval(0.9), duration_seconds=0.5)

    assert record["config"] == {"lr": 0.001, "epochs": 3}
    assert [e["epoch"] for e in record["history"]] == [1, 2, 3, 1, 2]
    assert record["final_eval"]["loss"] == 0.9
    assert record["total_duration_seconds"] == 1.5


# -- compare_runs ----------------------------------------------------------------


def _record_with(config, losses, eval_loss, duration) -> dict:
    record = new_run_record(config)
    extend_run_record(record, history=_history(losses), final_eval=_eval(eval_loss), duration_seconds=duration)
    return record


def test_compare_runs_reports_no_config_diff_for_identical_configs():
    a = _record_with({"lr": 0.001, "seed": 0}, [10.0, 8.0], 7.5, 1.0)
    b = _record_with({"lr": 0.001, "seed": 0}, [10.0, 8.0], 7.5, 1.0)
    diff = compare_runs(a, b)
    assert diff["config_diff"] == {}
    assert diff["epochs"] == {"a": 2, "b": 2}
    assert diff["final_eval_loss"] == {"a": 7.5, "b": 7.5}


def test_compare_runs_detects_differing_config_keys():
    a = _record_with({"lr": 0.001, "seed": 0, "batch_size": 32}, [10.0], 9.0, 1.0)
    b = _record_with({"lr": 0.005, "seed": 0, "batch_size": 32}, [10.0], 9.0, 1.0)
    diff = compare_runs(a, b)
    assert diff["config_diff"] == {"lr": (0.001, 0.005)}


def test_compare_runs_reports_final_metrics_and_epoch_count_differences():
    a = _record_with({"lr": 0.001}, [10.0, 8.0, 6.0], 5.5, 3.0)
    b = _record_with({"lr": 0.001}, [10.0, 8.0], 8.5, 1.0)
    diff = compare_runs(a, b)
    assert diff["epochs"] == {"a": 3, "b": 2}
    assert diff["final_train_loss"] == {"a": 6.0, "b": 8.0}
    assert diff["final_eval_loss"] == {"a": 5.5, "b": 8.5}
    assert diff["total_duration_seconds"] == {"a": 3.0, "b": 1.0}


def test_compare_runs_handles_a_run_with_no_history_yet():
    a = new_run_record({"lr": 0.001})
    b = _record_with({"lr": 0.001}, [10.0], 9.0, 1.0)
    diff = compare_runs(a, b)
    assert diff["epochs"] == {"a": 0, "b": 1}
    assert diff["final_train_loss"] == {"a": None, "b": 10.0}
    assert diff["final_eval_loss"] == {"a": None, "b": 9.0}


# -- compare.py CLI ----------------------------------------------------------------


def test_compare_cli_reports_config_and_metric_differences(tmp_path, capsys):
    record_a = _record_with({"lr": 0.001}, [10.0, 5.0], 4.5, 1.0)
    record_b = _record_with({"lr": 0.01}, [10.0, 2.0], 1.5, 1.0)
    path_a, path_b = tmp_path / "a.json", tmp_path / "b.json"
    save_run_record(path_a, record_a)
    save_run_record(path_b, record_b)

    code = compare_cli.main([str(path_a), str(path_b)])
    assert code == 0
    out = capsys.readouterr().out
    assert "lr: A=0.001  B=0.01" in out
    assert "Final eval loss:    A=4.5  B=1.5" in out


def test_compare_cli_reports_identical_configuration():
    record_a = _record_with({"lr": 0.001}, [10.0], 9.0, 1.0)
    diff = compare_runs(record_a, record_a)
    report = compare_cli.format_report(diff, "a.json", "a.json")
    assert "Configuration: identical." in report


def test_compare_cli_returns_error_for_missing_run_record(tmp_path, capsys):
    existing = tmp_path / "exists.json"
    save_run_record(existing, new_run_record({"lr": 0.001}))
    missing = tmp_path / "missing.json"

    code = compare_cli.main([str(existing), str(missing)])
    assert code == 1
    err = capsys.readouterr().err
    assert "missing.json" in err
