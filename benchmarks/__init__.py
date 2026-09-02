"""Forge benchmark subsystem (Milestone 11).

A small, self-contained performance-measurement package, deliberately kept
outside the `forge` package itself: `import forge` never imports anything
under `benchmarks/`, and the normal `pytest` correctness suite does not
execute any benchmark (see `tests/test_benchmarks.py`, which only checks
the harness's own mechanics with trivial, fast callables). Running the
suite is always an explicit, separate action:

    python -m benchmarks
    python -m benchmarks --categories forward transfer

See `docs/performance/benchmarking.md` for full methodology.
"""
