"""`forge benchmark` -- a thin pass-through to Forge's existing benchmark suite.

`benchmarks/` (top-level, outside the `forge` package) is Forge's existing
benchmark subsystem (`docs/performance/benchmarking.md`); it is deliberately
excluded from `forge`'s own package installation and never imported by
`import forge` (see `benchmarks/__init__.py`). This command exists only to
make `python -m benchmarks` reachable through `forge`'s own entry point when
run from within the Forge repository -- it imports `benchmarks.run.main`
lazily (only when this command actually runs) and forwards every argument to
it unparsed, so `forge`'s argument parsing never re-implements `--categories`/
`--output`/benchmark selection: `benchmarks/run.py`'s own `argparse` remains
the one place that logic lives.
"""

from __future__ import annotations

import argparse

from .errors import CLIError


def add_parser(subparsers: "argparse._SubParsersAction") -> None:
    # Deliberately no arguments of its own (not even `-h`/`--help`, hence
    # `add_help=False`): every argument typed after `benchmark` is left for
    # `main()` to collect via `parse_known_args()` and hand to
    # `benchmarks.run.main()` unparsed -- see `run_benchmark()` below.
    # (`nargs=argparse.REMAINDER` was tried first and rejected: combined with
    # nested subparsers, it fails to capture a leading `--flag` at all --
    # a known argparse limitation, not a Forge-specific bug.)
    subparsers.add_parser(
        "benchmark",
        add_help=False,
        help="Run the Forge benchmark suite (development tool; only available from within the Forge repository)",
        description="Forwards all arguments to `python -m benchmarks` -- see that command's own --help.",
    )


def run_benchmark(bench_args: "list[str]") -> int:
    try:
        from benchmarks.run import main as benchmark_main
    except ImportError as exc:
        raise CLIError(
            "The benchmark suite is not available: the 'benchmarks' package was not found. "
            "`forge benchmark` is a development tool that only works when run from within the "
            f"Forge repository (it is not part of the installed 'forge' package). ({exc})"
        ) from exc

    benchmark_main(bench_args)
    return 0
