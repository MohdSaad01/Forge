"""Lightweight tests of the benchmark harness itself (Milestone 11).

Per `docs/development/*` milestone brief: "Do NOT put timing thresholds
into normal unit tests. Timing varies across machines." These tests check
the harness's *mechanics* (iteration/warmup counts, aggregation math,
result structure, JSON round-tripping) with trivial, near-instant
callables -- never a real benchmark workload, and never an assertion about
how fast anything is. Running `python -m benchmarks` itself is a separate,
explicit action (see `benchmarks/run.py`), not something this test suite
triggers.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.results import BenchmarkResult, render_table, save_json
from benchmarks.sizes import DEFAULT_ITERATIONS, DEFAULT_WARMUP, ELEMENTWISE_SIZES, MATMUL_DIMS, TRANSFER_SIZES
from benchmarks.timing import Timing, time_calls, time_cpu, time_cuda


# -- Timing --------------------------------------------------------------


def test_time_cpu_runs_warmup_plus_iterations():
    calls = []
    time_cpu(lambda: calls.append(1), warmup=3, iterations=5)
    assert len(calls) == 3 + 5


def test_time_cpu_records_one_duration_per_iteration():
    timing = time_cpu(lambda: None, warmup=2, iterations=7)
    assert timing.warmup == 2
    assert timing.iterations == 7
    assert len(timing.durations) == 7
    assert all(d >= 0 for d in timing.durations)


def test_timing_aggregation_matches_known_values():
    timing = Timing(durations=(1.0, 2.0, 3.0), warmup=0, iterations=3)
    assert timing.mean == pytest.approx(2.0)
    assert timing.median == pytest.approx(2.0)
    assert timing.min == pytest.approx(1.0)
    assert timing.max == pytest.approx(3.0)
    assert timing.stdev == pytest.approx(1.0)


def test_timing_stdev_is_zero_for_a_single_iteration():
    timing = Timing(durations=(0.5,), warmup=0, iterations=1)
    assert timing.stdev == 0.0


def test_timing_rejects_mismatched_duration_count():
    with pytest.raises(ValueError):
        Timing(durations=(1.0, 2.0), warmup=0, iterations=3)


def test_time_calls_runs_each_distinct_callable_exactly_once():
    calls_made = []
    fns = [(lambda i=i: calls_made.append(i)) for i in range(6)]
    timing = time_calls(fns, warmup=2, iterations=4, device="cpu")
    assert calls_made == list(range(6))
    assert timing.warmup == 2
    assert timing.iterations == 4
    assert len(timing.durations) == 4


def test_time_calls_rejects_wrong_call_count():
    fns = [lambda: None for _ in range(3)]
    with pytest.raises(ValueError):
        time_calls(fns, warmup=2, iterations=4, device="cpu")


def test_time_cuda_raises_clearly_when_cuda_is_unavailable(monkeypatch):
    from forge.backend.cuda import backend as cuda_backend_module
    from forge.exceptions import CUDAError

    def _raise(*args, **kwargs):
        raise CUDAError("no CUDA for this test")

    monkeypatch.setattr(cuda_backend_module, "get_cuda_backend", _raise)
    with pytest.raises(CUDAError):
        time_cuda(lambda: None, warmup=1, iterations=1)


# -- Results ---------------------------------------------------------------


def _dummy_result(**overrides) -> BenchmarkResult:
    timing = Timing(durations=(0.001, 0.002, 0.0015), warmup=1, iterations=3)
    base = dict(
        category="forward", operation="add", device="cpu", scale="tiny",
        shape="(4,)", dtype="float32", timing=timing,
    )
    base.update(overrides)
    return BenchmarkResult.from_timing(**base)


def test_benchmark_result_from_timing_carries_aggregates():
    result = _dummy_result()
    assert result.mean_seconds == pytest.approx(0.0015)
    assert result.iterations == 3
    assert result.warmup == 1


def test_render_table_handles_empty_and_nonempty_results():
    assert "no results" in render_table([])
    table = render_table([_dummy_result(), _dummy_result(operation="matmul")])
    assert "add" in table
    assert "matmul" in table
    assert "forward" in table


def test_save_json_round_trips(tmp_path):
    results = [_dummy_result(), _dummy_result(device="cuda", operation="matmul")]
    output_path = tmp_path / "results.json"
    save_json(results, {"platform": "test"}, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["environment"] == {"platform": "test"}
    assert len(payload["results"]) == 2
    assert payload["results"][0]["operation"] == "add"
    assert payload["results"][1]["device"] == "cuda"


# -- Sizes -------------------------------------------------------------------


def test_size_configs_are_positive_and_ordered():
    for sizes in (MATMUL_DIMS, ELEMENTWISE_SIZES, TRANSFER_SIZES):
        values = list(sizes.values())
        assert all(v > 0 for v in values)
        assert values == sorted(values), "sizes should increase tiny -> small -> medium"


def test_elementwise_sizes_are_matmul_dims_squared():
    for scale, dim in MATMUL_DIMS.items():
        assert ELEMENTWISE_SIZES[scale] == dim * dim


def test_default_warmup_and_iterations_are_sane():
    assert DEFAULT_WARMUP > 0
    assert DEFAULT_ITERATIONS > 0


# -- Import boundary -----------------------------------------------------


def test_forge_does_not_expose_or_depend_on_benchmarks():
    """Benchmarks must not be required for package import or normal usage."""
    import forge

    assert not hasattr(forge, "benchmarks")
    assert "benchmarks" not in forge.__all__
