"""Roofline-style modeling primitives (Milestone 35).

A small, pure (no CUDA, no timing) library of FLOP-counting and
byte-traffic conventions, plus arithmetic-intensity and bottleneck
classification helpers, used by every `benchmarks/m35_*.py` script so every
op is measured against the *same* conventions and the *same* measured
hardware ceilings (Section 2's `MEASURE -> MODEL -> CLASSIFY -> RANK ->
DECIDE` pipeline; Section 7's roofline model; Section 8/9's FLOP/byte
counting conventions).

## FLOP counting convention (Section 8)

One multiply-add (`a*b+c`) is counted as **2 FLOPs** (1 multiply + 1 add) --
the standard convention used throughout (GEMM, convolution). A bare
elementwise multiply/add/sub is **1 FLOP/element**. Comparison-only ops
(`relu`, `MaxPool2d`) are **0 true FLOPs** -- reported separately as
elements/sec, never inflated into a fake FLOP count. Transcendental ops
(`exp`, `log`) are counted as **1 FLOP-equivalent/element** for arithmetic-
intensity purposes only -- documented explicitly here because a real `exp`
costs meaningfully more hardware cycles per element than an add, so its
GFLOP/s number is *not* comparable in an absolute sense to `add`'s GFLOP/s;
it is only meaningful for classifying `exp` against its own roofline point.

## Byte traffic convention (Section 9)

Every `bytes_*` function returns **minimum logical traffic**: each logical
operand read once, each logical result written once, at the operation's
mathematical definition -- e.g. GEMM's minimum logical traffic is `(M*K +
K*N + M*N) * itemsize`, ignoring the tiled kernel's actual cache/shared-
memory reuse pattern (which reads each tile multiple times from global
memory in a naive analysis, and far fewer times thanks to the M11 shared-
memory tiling). This is the standard roofline-model convention (the same
one used for reporting cuBLAS/cuDNN arithmetic intensity) and is used here
for computing arithmetic intensity and classifying kernels, never presented
as a measurement of actual DRAM traffic (Section 9: "do not claim exact
DRAM traffic unless it is actually measured" -- none of this is measured,
all of it is modeled).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

FMA_FLOPS = 2  # 1 multiply + 1 add, per Section 8's stated convention.

# A kernel whose total measured GPU time falls under this threshold is
# presumptively launch/latency-dominated rather than throughput-dominated --
# a documented heuristic (not a measured hardware fact) used only to steer
# `classify()`'s "latency/launch-bound" branch. Chosen from this codebase's
# own observed CUDA kernel-launch overhead scale (`docs/performance/
# conv2d-backward-profiling.md` and `pipeline-profiling.md` both observe
# sub-20us launch/sync overhead on the 940MX for the smallest kernels).
LATENCY_BOUND_THRESHOLD_SECONDS = 20e-6

# A kernel is considered "near its roofline ceiling" above this fraction --
# used only to decide compute-bound vs memory-bandwidth-bound in `classify`.
# Also a documented heuristic, not a hardware constant.
NEAR_CEILING_FRACTION = 0.30


# -- FLOP counting -------------------------------------------------------


def flops_elementwise(n: int) -> int:
    """1 FLOP/element -- add/sub/mul (and, by the documented FLOP-equivalent
    convention above, exp/log)."""
    return n


def flops_relu(n: int) -> int:
    """0 true FLOPs -- a comparison+select, not an arithmetic operation."""
    return 0


def flops_reduction(n: int) -> int:
    """A full `n`-element reduction: `n-1` adds (tree or serial, same count)."""
    return max(n - 1, 0)


def flops_cross_entropy_forward(batch: int, classes: int) -> int:
    """Log-sum-exp + gather, per row: max-reduce (`classes-1`) + `exp`
    (`classes`, FLOP-equivalent) + sum-reduce (`classes-1`) + `log` (1) +
    final subtract (1) = `3*classes` FLOP-equivalents, documented above."""
    return batch * (3 * classes)


def flops_cross_entropy_backward(batch: int, classes: int) -> int:
    """`softmax(x) - one_hot(target)`, scaled by `grad_output`: `exp`
    (`classes`) + sum-reduce (`classes-1`) + a divide+subtract+multiply per
    element (`3*classes`) = `5*classes - 1` FLOP-equivalents/row."""
    return batch * (5 * classes - 1)


def flops_matmul(M: int, N: int, K: int) -> int:
    """`[M,K] @ [K,N]`: `2*M*N*K` FLOPs (Section 8's stated convention)."""
    return FMA_FLOPS * M * N * K


def flops_conv2d_forward(N: int, Cout: int, Hout: int, Wout: int, Cin: int, KH: int, KW: int) -> int:
    """One multiply-add per (output element, weight tap): `2*N*Cout*Hout*Wout*Cin*KH*KW`."""
    return FMA_FLOPS * N * Cout * Hout * Wout * Cin * KH * KW


def flops_conv2d_dinput(N: int, Cin: int, H: int, W: int, Cout: int, KH: int, KW: int) -> int:
    """Same total multiply-add count as the forward pass, accumulated into `grad_input`."""
    return FMA_FLOPS * N * Cin * H * W * Cout * KH * KW


def flops_conv2d_dweight(Cout: int, Cin: int, KH: int, KW: int, N: int, Hout: int, Wout: int) -> int:
    """Same total multiply-add count as the forward pass, accumulated into `grad_weight`."""
    return FMA_FLOPS * Cout * Cin * KH * KW * N * Hout * Wout


def flops_maxpool(n_out: int, kh: int, kw: int) -> int:
    """0 true FLOPs -- `kh*kw - 1` comparisons per output element, not arithmetic."""
    return 0


def flops_dropout(n: int) -> int:
    """1 FLOP/element (the mask multiply); mask generation itself is 0 FLOPs (a hash+compare)."""
    return n


def flops_sgd(n: int) -> int:
    """`param -= lr * grad`: 1 multiply + 1 subtract = 2 FLOPs/element."""
    return 2 * n


# Documented per-element FLOP count for one Adam step (Section 20), derived
# from `k_adam_step` (`kernels.cu`): m-update (`b1*m + (1-b1)*g`, 3 FLOPs:
# 2 multiplies + 1 add) + v-update (`b2*v + (1-b2)*g*g`, 4 FLOPs: 1 square +
# 2 multiplies + 1 add) + bias-correction/parameter update (`m_hat = m/bc1`,
# `v_hat = v/bc2`, `sqrt(v_hat)`, `+eps`, `lr*m_hat`, `/denom`, `param -=`,
# 7 FLOPs) = 14 FLOPs/element total. A documented convention, not a literal
# hardware instruction count (fused multiply-adds/reciprocal-sqrt may issue
# differently in practice).
ADAM_FLOPS_PER_ELEMENT = 14


def flops_adam(n: int) -> int:
    return ADAM_FLOPS_PER_ELEMENT * n


# -- byte-traffic counting (minimum logical traffic, Section 9) ---------


def bytes_elementwise_binary(n: int, itemsize: int = 4) -> int:
    """add/sub/mul: read a, read b, write out."""
    return 3 * n * itemsize


def bytes_elementwise_unary(n: int, itemsize: int = 4) -> int:
    """relu/exp/log/neg: read a, write out."""
    return 2 * n * itemsize


def bytes_reduction(n: int, itemsize: int = 4) -> int:
    """Full reduction: read n elements, write 1 scalar."""
    return (n + 1) * itemsize


def bytes_cross_entropy_forward(batch: int, classes: int, itemsize: int = 4) -> int:
    """Read logits (`batch*classes`) + targets (`batch`, counted at `itemsize`
    for simplicity even though targets are int64), write per-row loss (`batch`)."""
    return (batch * classes + 2 * batch) * itemsize


def bytes_cross_entropy_backward(batch: int, classes: int, itemsize: int = 4) -> int:
    """Read logits + grad_output + targets, write grad_input (`batch*classes`)."""
    return (2 * batch * classes + 2 * batch) * itemsize


def bytes_matmul_minimum(M: int, N: int, K: int, itemsize: int = 4) -> int:
    """Minimum logical traffic: read A (`M*K`) once, B (`K*N`) once, write C (`M*N`) once.

    Deliberately *not* a model of the tiled kernel's actual global-memory
    traffic (which re-reads each operand element roughly `dim/TILE` times in
    a naive un-cached analysis, and far less thanks to shared-memory tiling)
    -- see this module's docstring.
    """
    return (M * K + K * N + M * N) * itemsize


def bytes_conv2d_forward(
    N: int, Cin: int, H: int, W: int, Cout: int, KH: int, KW: int, Hout: int, Wout: int, itemsize: int = 4
) -> int:
    """Minimum logical traffic: read input once, read weight once, write output once."""
    return (N * Cin * H * W + Cout * Cin * KH * KW + N * Cout * Hout * Wout) * itemsize


def bytes_conv2d_dinput(
    N: int, Cin: int, H: int, W: int, Cout: int, KH: int, KW: int, Hout: int, Wout: int, itemsize: int = 4
) -> int:
    """Minimum logical traffic: read grad_output once, read weight once, write grad_input once."""
    return (N * Cout * Hout * Wout + Cout * Cin * KH * KW + N * Cin * H * W) * itemsize


def bytes_conv2d_dweight(
    N: int, Cin: int, H: int, W: int, Cout: int, KH: int, KW: int, Hout: int, Wout: int, itemsize: int = 4
) -> int:
    """Minimum logical traffic: read input once, read grad_output once, write grad_weight once."""
    return (N * Cin * H * W + N * Cout * Hout * Wout + Cout * Cin * KH * KW) * itemsize


def bytes_maxpool_forward(N: int, C: int, Hout: int, Wout: int, KH: int, KW: int, itemsize: int = 4) -> int:
    """Logical traffic *with* pooling-window overlap counted: each output
    element's kernel window is read in full (`KH*KW` reads), plus one write."""
    n_out = N * C * Hout * Wout
    return (n_out * KH * KW + n_out) * itemsize


def bytes_maxpool_backward(N: int, C: int, H: int, W: int, Hout: int, Wout: int, itemsize: int = 4) -> int:
    """Read grad_output + input (for the max-index recompute), write grad_input."""
    return (N * C * Hout * Wout + N * C * H * W + N * C * H * W) * itemsize


def bytes_dropout(n: int, itemsize: int = 4) -> int:
    """Mask multiply: read input, write output (mask itself generated in-register)."""
    return 2 * n * itemsize


def bytes_sgd(n: int, itemsize: int = 4) -> int:
    """Read param, read grad, write param (in-place read-modify-write)."""
    return 3 * n * itemsize


def bytes_adam(n: int, itemsize: int = 4) -> int:
    """Read param, grad, m, v (4n); write param, m, v (3n) -- 7n total."""
    return 7 * n * itemsize


def arithmetic_intensity(flops: float, nbytes: float) -> float:
    """FLOPs / bytes moved. `0.0` if no bytes are moved (undefined AI)."""
    if nbytes <= 0:
        return 0.0
    return flops / nbytes


# -- roofline ceilings + classification -----------------------------------


@dataclass(frozen=True)
class Ceilings:
    """Practical (measured) hardware ceilings, per `m35_hardware.json` (Section 6)."""

    compute_gflops: float  # practical achievable GFLOP/s, from a large GEMM
    bandwidth_gbps: float  # practical achievable GB/s, from a large streaming add

    @property
    def ridge_point(self) -> float:
        """AI (FLOPs/byte) at which the two ceilings intersect on a roofline plot."""
        if self.bandwidth_gbps <= 0:
            return float("inf")
        return (self.compute_gflops * 1e9) / (self.bandwidth_gbps * 1e9)

    def roofline_ceiling_gflops(self, ai: float) -> float:
        """`min(compute_ceiling, bandwidth_ceiling * AI)` (Section 7)."""
        return min(self.compute_gflops, self.bandwidth_gbps * ai)


def load_ceilings(path: "str | Path") -> Ceilings:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    ceilings = payload["ceilings"]
    return Ceilings(
        compute_gflops=ceilings["practical_compute_gflops"],
        bandwidth_gbps=ceilings["practical_bandwidth_gbps"],
    )


@dataclass(frozen=True)
class Classification:
    label: str  # "compute_bound" | "memory_bandwidth_bound" | "latency_launch_bound" | "mixed_or_ambiguous"
    roofline_ceiling_gflops: float
    fraction_of_ceiling: float  # achieved_gflops / roofline_ceiling_gflops, clamped at a sane floor
    note: str


def classify(
    achieved_gflops: float,
    elapsed_seconds: float,
    ai: float,
    ceilings: Ceilings,
    achieved_gbps: "float | None" = None,
) -> Classification:
    """Section 27's four-way bottleneck classification, as an explicit,
    documented heuristic (never claimed as exact hardware utilization,
    Section 28) applied consistently everywhere via this one function.

    A zero-FLOP op (`relu`, `MaxPool2d`, dropout mask generation --
    comparison/hash-only, per this module's documented convention) has
    `ai == 0` by construction, which degenerates the FLOP-based ceiling to
    0 and makes `fraction_of_ceiling` meaningless. For that case, pass
    `achieved_gbps` and classification falls back to bandwidth utilization
    (`achieved_gbps / ceilings.bandwidth_gbps`) directly -- the only ceiling
    that means anything for a pure-data-movement op.
    """
    if ai <= 0 and achieved_gbps is not None:
        ceiling = ceilings.bandwidth_gbps
        fraction = (achieved_gbps / ceiling) if ceiling > 0 else 0.0
        if elapsed_seconds < LATENCY_BOUND_THRESHOLD_SECONDS and fraction < NEAR_CEILING_FRACTION:
            return Classification(
                "latency_launch_bound", ceiling, fraction,
                f"0-FLOP op: elapsed {elapsed_seconds * 1e6:.2f}us < "
                f"{LATENCY_BOUND_THRESHOLD_SECONDS * 1e6:.0f}us launch-overhead heuristic, and only "
                f"{fraction * 100:.1f}% of practical bandwidth achieved.",
            )
        if fraction >= NEAR_CEILING_FRACTION:
            return Classification(
                "memory_bandwidth_bound", ceiling, fraction,
                f"0-FLOP op (no arithmetic-intensity ceiling applies): {fraction * 100:.1f}% of practical "
                "bandwidth achieved -- data movement, not compute, is this op's only cost.",
            )
        return Classification(
            "mixed_or_ambiguous", ceiling, fraction,
            f"0-FLOP op: only {fraction * 100:.1f}% of practical bandwidth achieved, with elapsed time "
            f"({elapsed_seconds * 1e6:.2f}us) above the latency-bound heuristic -- no single cause is "
            "unambiguous from this measurement alone.",
        )

    ceiling = ceilings.roofline_ceiling_gflops(ai) if ai > 0 else ceilings.bandwidth_gbps * 0.0
    fraction = (achieved_gflops / ceiling) if ceiling > 0 else 0.0

    if elapsed_seconds < LATENCY_BOUND_THRESHOLD_SECONDS and fraction < NEAR_CEILING_FRACTION:
        return Classification(
            "latency_launch_bound", ceiling, fraction,
            f"elapsed {elapsed_seconds * 1e6:.2f}us < {LATENCY_BOUND_THRESHOLD_SECONDS * 1e6:.0f}us "
            f"launch-overhead heuristic, and only {fraction * 100:.1f}% of the roofline ceiling.",
        )

    if fraction >= NEAR_CEILING_FRACTION:
        if ai > 0 and ai < ceilings.ridge_point:
            return Classification(
                "memory_bandwidth_bound", ceiling, fraction,
                f"AI={ai:.3f} is left of the ridge point ({ceilings.ridge_point:.3f}) -- "
                f"bandwidth, not compute, sets this op's ceiling.",
            )
        return Classification(
            "compute_bound", ceiling, fraction,
            f"AI={ai:.3f} is at/right of the ridge point ({ceilings.ridge_point:.3f}) -- "
            f"compute, not bandwidth, sets this op's ceiling.",
        )

    return Classification(
        "mixed_or_ambiguous", ceiling, fraction,
        f"Only {fraction * 100:.1f}% of the {ceiling:.2f} GFLOP/s roofline ceiling achieved, "
        f"with elapsed time ({elapsed_seconds * 1e6:.2f}us) above the latency-bound heuristic -- "
        "no single cause is unambiguous from this measurement alone.",
    )


__all__ = [
    "FMA_FLOPS",
    "LATENCY_BOUND_THRESHOLD_SECONDS",
    "NEAR_CEILING_FRACTION",
    "ADAM_FLOPS_PER_ELEMENT",
    "flops_elementwise",
    "flops_relu",
    "flops_reduction",
    "flops_cross_entropy_forward",
    "flops_cross_entropy_backward",
    "flops_matmul",
    "flops_conv2d_forward",
    "flops_conv2d_dinput",
    "flops_conv2d_dweight",
    "flops_maxpool",
    "flops_dropout",
    "flops_sgd",
    "flops_adam",
    "bytes_elementwise_binary",
    "bytes_elementwise_unary",
    "bytes_reduction",
    "bytes_cross_entropy_forward",
    "bytes_cross_entropy_backward",
    "bytes_matmul_minimum",
    "bytes_conv2d_forward",
    "bytes_conv2d_dinput",
    "bytes_conv2d_dweight",
    "bytes_maxpool_forward",
    "bytes_maxpool_backward",
    "bytes_dropout",
    "bytes_sgd",
    "bytes_adam",
    "arithmetic_intensity",
    "Ceilings",
    "load_ceilings",
    "Classification",
    "classify",
]
