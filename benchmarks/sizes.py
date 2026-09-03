"""Reproducible benchmark size configurations (Milestone 11).

Three scales, per the milestone brief's own example dimensions and chosen to
stay well inside the 940MX's 2 GB VRAM and the development machine's 8 GB
system RAM (see `docs/development/development-environment.md`):

- "tiny":   32x32   (1,024 elements)
- "small":  128x128 (16,384 elements)
- "medium": 512x512 (262,144 elements)

`matmul` uses these directly as square `(N, N)` operands. The other
forward/backward tensor ops (`add`/`sub`/`mul`/`relu`/`sum`/`reshape`) use a
flat vector of the same *element count*, so results are directly comparable
across categories at a given scale rather than using an unrelated size
convention per operation.

A 512x512 float64 matmul operand is 512*512*8 = 2 MiB; the CUDA training
benchmark below never allocates more than a few MiB total -- nowhere close
to the 2 GB VRAM budget.
"""

from __future__ import annotations

MATMUL_DIMS: "dict[str, int]" = {
    "tiny": 32,
    "small": 128,
    "medium": 512,
}

ELEMENTWISE_SIZES: "dict[str, int]" = {name: dim * dim for name, dim in MATMUL_DIMS.items()}

# Transfer cost is naturally a function of bytes moved, not "matrix
# dimension" -- these are float32 element counts chosen to span a wide
# range (4 KB / ~400 KB / ~4 MB) while staying trivially small relative to
# available RAM/VRAM.
TRANSFER_SIZES: "dict[str, int]" = {
    "tiny": 1_024,
    "small": 100_000,
    "medium": 1_000_000,
}

# A single small model (Linear -> ReLU -> Linear), sized for a training
# loop that runs in well under a second per device on this hardware.
TRAINING_CONFIG: "dict[str, int]" = {
    "batch_size": 64,
    "in_features": 64,
    "hidden_features": 128,
    "out_features": 10,
    "iterations": 50,
    "warmup_iterations": 5,
}

# Conv2d forward/backward benchmark configurations (Milestone 15). Kept
# deliberately small -- correctness, not performance, is this milestone's
# objective (the 940MX's naive, unoptimized kernels are not expected to beat
# NumPy/BLAS here) -- while still spanning three scales, staying well inside
# the 940MX's 2 GB VRAM budget.
CONV2D_CONFIGS: "dict[str, dict[str, int]]" = {
    "tiny": {"N": 4, "Cin": 3, "Cout": 8, "H": 16, "W": 16, "K": 3},
    "small": {"N": 8, "Cin": 8, "Cout": 16, "H": 32, "W": 32, "K": 3},
    "medium": {"N": 8, "Cin": 16, "Cout": 32, "H": 64, "W": 64, "K": 3},
}

# Default warmup/iteration counts for forward/backward/transfer op
# benchmarks. See docs/performance/benchmarking.md for why warmup exists
# (separating launch/lazy-init/compilation/cache effects from steady-state
# execution) and why 20 measured iterations was chosen (enough for a
# stable mean/stdev on this hardware without making the full suite slow).
DEFAULT_WARMUP = 5
DEFAULT_ITERATIONS = 20

# -- Milestone 21 additions ---------------------------------------------------
#
# MaxPool2d reuses CONV2D_CONFIGS' spatial dimensions directly (pooling a
# (N, Cout, H, W)-shaped activation is exactly what follows a Conv2d in the
# M20 CNN), pooled with a fixed 2x2/stride-2 window -- the same window the
# M20 architecture uses at both pooling stages.
POOL2D_KERNEL = 2

# CrossEntropyLoss/MSELoss forward+backward configs: (batch, features/classes)
# at three scales, staying consistent with ELEMENTWISE_SIZES' element-count
# philosophy (loss inputs are small (batch, classes) tensors, not big enough
# to need a fourth scale of their own).
LOSS_CONFIGS: "dict[str, dict[str, int]]" = {
    "tiny": {"batch": 8, "classes": 10},
    "small": {"batch": 64, "classes": 10},
    "medium": {"batch": 256, "classes": 10},
}

# Dropout forward/backward reuses ELEMENTWISE_SIZES directly -- it is an
# elementwise mask-multiply, the same shape family as add/sub/mul/relu.
DROPOUT_P = 0.5

# Adam step benchmark: parameter-tensor element counts at three scales,
# matching a small Linear layer's weight matrix up to a Conv2d-sized one.
ADAM_PARAM_SIZES: "dict[str, int]" = {
    "tiny": 1_024,
    "small": 16_384,
    "medium": 262_144,
}

# End-to-end MNIST training-throughput benchmark (Section 14): the real M20
# CNN (`examples.mnist.model.build_model()`), a synthetic MNIST-shaped batch
# (no dataset download needed, matching `tests/test_mnist_example_integration.py`'s
# own synthetic-data convention), run for a fixed number of steps. Kept small
# enough to complete quickly on both the i5-7200U and the 940MX.
MNIST_TRAINING_CONFIG: "dict[str, int]" = {
    "batch_size": 64,
    "iterations": 30,
    "warmup_iterations": 5,
}

# MNIST workload profiling (Section 4): more iterations than the throughput
# benchmark above since per-phase/per-layer breakdowns are noisier
# individually and benefit from a larger sample.
MNIST_PROFILE_CONFIG: "dict[str, int]" = {
    "batch_size": 64,
    "iterations": 30,
    "warmup_iterations": 5,
}
