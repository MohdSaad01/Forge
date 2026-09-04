"""Tests for `benchmarks/roofline.py`'s pure FLOP/byte/AI/classification
model (Milestone 35).

Every function here is deterministic math with no CUDA/timing dependency,
so -- unlike `tests/test_cuda_*.py` -- this file carries no CUDA
skip-marker and runs on any machine. Mirrors `tests/test_benchmarks.py`'s
convention: exercise the harness's own logic, never assert a timing
threshold.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.roofline import (
    ADAM_FLOPS_PER_ELEMENT,
    Ceilings,
    Classification,
    arithmetic_intensity,
    bytes_adam,
    bytes_conv2d_dinput,
    bytes_conv2d_dweight,
    bytes_conv2d_forward,
    bytes_cross_entropy_backward,
    bytes_cross_entropy_forward,
    bytes_dropout,
    bytes_elementwise_binary,
    bytes_elementwise_unary,
    bytes_maxpool_backward,
    bytes_maxpool_forward,
    bytes_matmul_minimum,
    bytes_reduction,
    bytes_sgd,
    classify,
    flops_adam,
    flops_conv2d_dinput,
    flops_conv2d_dweight,
    flops_conv2d_forward,
    flops_cross_entropy_backward,
    flops_cross_entropy_forward,
    flops_dropout,
    flops_elementwise,
    flops_matmul,
    flops_maxpool,
    flops_reduction,
    flops_relu,
    flops_sgd,
    load_ceilings,
)


# -- FLOP counting ---------------------------------------------------------


def test_flops_elementwise_is_one_per_element():
    assert flops_elementwise(1000) == 1000


def test_flops_relu_and_maxpool_are_zero():
    assert flops_relu(1000) == 0
    assert flops_maxpool(500, 2, 2) == 0


def test_flops_reduction_is_n_minus_one():
    assert flops_reduction(10) == 9
    assert flops_reduction(0) == 0
    assert flops_reduction(1) == 0  # never negative


def test_flops_matmul_matches_2mnk():
    assert flops_matmul(M=4, N=8, K=16) == 2 * 4 * 8 * 16


def test_flops_conv2d_forward_matches_hand_computation():
    # 1 image, 2 output channels, 3x3 output, 1 input channel, 3x3 kernel:
    # 2 * 1*2*3*3 * 1*3*3 = 2*18*9 = 324
    assert flops_conv2d_forward(N=1, Cout=2, Hout=3, Wout=3, Cin=1, KH=3, KW=3) == 324


def test_flops_conv2d_dinput_and_dweight_match_forward_mac_count():
    fwd = flops_conv2d_forward(N=2, Cout=4, Hout=5, Wout=5, Cin=3, KH=3, KW=3)
    dinput = flops_conv2d_dinput(N=2, Cin=3, H=5, W=5, Cout=4, KH=3, KW=3)
    dweight = flops_conv2d_dweight(Cout=4, Cin=3, KH=3, KW=3, N=2, Hout=5, Wout=5)
    assert dinput == fwd
    assert dweight == fwd


def test_flops_sgd_is_two_per_element():
    assert flops_sgd(100) == 200


def test_flops_adam_uses_documented_constant():
    assert flops_adam(100) == 100 * ADAM_FLOPS_PER_ELEMENT
    assert ADAM_FLOPS_PER_ELEMENT == 14


def test_flops_dropout_is_one_per_element():
    assert flops_dropout(50) == 50


def test_flops_cross_entropy_scale_with_batch_and_classes():
    fwd = flops_cross_entropy_forward(batch=8, classes=10)
    bwd = flops_cross_entropy_backward(batch=8, classes=10)
    assert fwd == 8 * 3 * 10
    assert bwd == 8 * (5 * 10 - 1)
    assert bwd > fwd  # backward does strictly more work per the documented convention


# -- byte-traffic counting --------------------------------------------------


def test_bytes_elementwise_binary_vs_unary():
    assert bytes_elementwise_binary(100, itemsize=4) == 3 * 100 * 4
    assert bytes_elementwise_unary(100, itemsize=4) == 2 * 100 * 4


def test_bytes_reduction_reads_n_writes_one():
    assert bytes_reduction(100, itemsize=4) == 101 * 4


def test_bytes_matmul_minimum_matches_hand_computation():
    # A: 4x16, B: 16x8, C: 4x8 -> (64 + 128 + 32) * 4 = 896
    assert bytes_matmul_minimum(M=4, N=8, K=16, itemsize=4) == (4 * 16 + 16 * 8 + 4 * 8) * 4


def test_bytes_conv2d_forward_reads_input_and_weight_writes_output():
    expected = (1 * 1 * 5 * 5 + 2 * 1 * 3 * 3 + 1 * 2 * 3 * 3) * 4
    assert bytes_conv2d_forward(N=1, Cin=1, H=5, W=5, Cout=2, KH=3, KW=3, Hout=3, Wout=3, itemsize=4) == expected


def test_bytes_conv2d_dinput_and_dweight_are_distinct_but_same_total_elements():
    dinput = bytes_conv2d_dinput(N=1, Cin=1, H=5, W=5, Cout=2, KH=3, KW=3, Hout=3, Wout=3)
    dweight = bytes_conv2d_dweight(N=1, Cin=1, H=5, W=5, Cout=2, KH=3, KW=3, Hout=3, Wout=3)
    # Same three tensors (input, grad_output, weight-sized buffer) touched either way.
    assert dinput == dweight


def test_bytes_maxpool_forward_counts_window_overlap():
    # 1 image, 1 channel, 2x2 output, 2x2 kernel -> 4 outputs * 4 window reads + 4 writes
    nbytes = bytes_maxpool_forward(N=1, C=1, Hout=2, Wout=2, KH=2, KW=2, itemsize=4)
    assert nbytes == (4 * 4 + 4) * 4


def test_bytes_maxpool_backward_reads_grad_and_input_writes_grad_input():
    nbytes = bytes_maxpool_backward(N=1, C=1, H=4, W=4, Hout=2, Wout=2, itemsize=4)
    assert nbytes == (4 + 16 + 16) * 4


def test_bytes_dropout_reads_and_writes_once():
    assert bytes_dropout(100, itemsize=4) == 2 * 100 * 4


def test_bytes_sgd_touches_param_twice_grad_once():
    assert bytes_sgd(100, itemsize=4) == 3 * 100 * 4


def test_bytes_adam_touches_seven_arrays_worth():
    assert bytes_adam(100, itemsize=4) == 7 * 100 * 4


def test_bytes_cross_entropy_backward_moves_more_than_forward():
    fwd = bytes_cross_entropy_forward(batch=8, classes=10)
    bwd = bytes_cross_entropy_backward(batch=8, classes=10)
    assert bwd > fwd


# -- arithmetic intensity + classification -----------------------------------


def test_arithmetic_intensity_basic_ratio():
    assert arithmetic_intensity(flops=100, nbytes=50) == 2.0


def test_arithmetic_intensity_zero_bytes_is_zero_not_inf():
    assert arithmetic_intensity(flops=100, nbytes=0) == 0.0


def test_ceilings_ridge_point_is_compute_over_bandwidth():
    ceilings = Ceilings(compute_gflops=1000.0, bandwidth_gbps=100.0)
    assert ceilings.ridge_point == pytest.approx(10.0)


def test_roofline_ceiling_is_the_min_of_both_lines():
    ceilings = Ceilings(compute_gflops=1000.0, bandwidth_gbps=100.0)
    # Left of the ridge point (AI=1 < 10): bandwidth-limited.
    assert ceilings.roofline_ceiling_gflops(ai=1.0) == pytest.approx(100.0)
    # Right of the ridge point (AI=100 > 10): compute-limited.
    assert ceilings.roofline_ceiling_gflops(ai=100.0) == pytest.approx(1000.0)


def test_classify_low_ai_near_ceiling_is_memory_bandwidth_bound():
    ceilings = Ceilings(compute_gflops=1000.0, bandwidth_gbps=100.0)
    ai = 1.0  # ceiling = 100 GFLOP/s
    result = classify(achieved_gflops=95.0, elapsed_seconds=1e-3, ai=ai, ceilings=ceilings)
    assert result.label == "memory_bandwidth_bound"
    assert result.fraction_of_ceiling == pytest.approx(0.95)


def test_classify_high_ai_near_ceiling_is_compute_bound():
    ceilings = Ceilings(compute_gflops=1000.0, bandwidth_gbps=100.0)
    ai = 100.0  # ceiling = 1000 GFLOP/s
    result = classify(achieved_gflops=900.0, elapsed_seconds=1e-3, ai=ai, ceilings=ceilings)
    assert result.label == "compute_bound"


def test_classify_tiny_elapsed_far_below_ceiling_is_latency_bound():
    ceilings = Ceilings(compute_gflops=1000.0, bandwidth_gbps=100.0)
    result = classify(achieved_gflops=0.5, elapsed_seconds=5e-6, ai=1.0, ceilings=ceilings)
    assert result.label == "latency_launch_bound"


def test_classify_far_below_ceiling_but_not_tiny_is_ambiguous():
    ceilings = Ceilings(compute_gflops=1000.0, bandwidth_gbps=100.0)
    result = classify(achieved_gflops=5.0, elapsed_seconds=5e-3, ai=1.0, ceilings=ceilings)
    assert result.label == "mixed_or_ambiguous"


def test_classify_returns_frozen_dataclass_with_note():
    ceilings = Ceilings(compute_gflops=1000.0, bandwidth_gbps=100.0)
    result = classify(achieved_gflops=95.0, elapsed_seconds=1e-3, ai=1.0, ceilings=ceilings)
    assert isinstance(result, Classification)
    assert result.note  # non-empty explanation string
    with pytest.raises(Exception):
        result.label = "compute_bound"  # frozen -- reassignment must fail


def test_classify_zero_flop_op_falls_back_to_bandwidth_utilization():
    ceilings = Ceilings(compute_gflops=1000.0, bandwidth_gbps=100.0)
    result = classify(achieved_gflops=0.0, elapsed_seconds=1e-3, ai=0.0, ceilings=ceilings, achieved_gbps=95.0)
    assert result.label == "memory_bandwidth_bound"
    assert result.fraction_of_ceiling == pytest.approx(0.95)
    assert result.roofline_ceiling_gflops == pytest.approx(100.0)  # the bandwidth ceiling itself


def test_classify_zero_flop_op_tiny_elapsed_is_latency_bound():
    ceilings = Ceilings(compute_gflops=1000.0, bandwidth_gbps=100.0)
    result = classify(achieved_gflops=0.0, elapsed_seconds=5e-6, ai=0.0, ceilings=ceilings, achieved_gbps=5.0)
    assert result.label == "latency_launch_bound"


def test_classify_zero_flop_op_without_gbps_uses_flop_based_path():
    # No achieved_gbps given: falls through to the (degenerate, ceiling=0) FLOP-based branch.
    ceilings = Ceilings(compute_gflops=1000.0, bandwidth_gbps=100.0)
    result = classify(achieved_gflops=0.0, elapsed_seconds=1e-3, ai=0.0, ceilings=ceilings)
    assert result.roofline_ceiling_gflops == 0.0


def test_load_ceilings_reads_practical_values_from_json(tmp_path):
    path = tmp_path / "m35_hardware.json"
    path.write_text(
        json.dumps({"ceilings": {"practical_compute_gflops": 42.0, "practical_bandwidth_gbps": 7.0}}),
        encoding="utf-8",
    )
    ceilings = load_ceilings(path)
    assert ceilings.compute_gflops == 42.0
    assert ceilings.bandwidth_gbps == 7.0
