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

# Default warmup/iteration counts for forward/backward/transfer op
# benchmarks. See docs/performance/benchmarking.md for why warmup exists
# (separating launch/lazy-init/compilation/cache effects from steady-state
# execution) and why 20 measured iterations was chosen (enough for a
# stable mean/stdev on this hardware without making the full suite slow).
DEFAULT_WARMUP = 5
DEFAULT_ITERATIONS = 20
