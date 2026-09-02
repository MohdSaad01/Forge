"""Benchmark suite entry point (Milestone 11).

    python -m benchmarks
    python -m benchmarks --categories forward transfer
    python -m benchmarks --output benchmarks/results/my_run.json

Not part of package import or the normal test suite: `import forge` never
touches this module, and the pytest correctness suite
(`tests/test_benchmarks.py`) only exercises the harness's own mechanics
with trivial, fast callables -- never runs this full suite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .backward_bench import run_backward_benchmarks
from .environment import collect_environment
from .ops_bench import run_forward_benchmarks
from .results import render_table, save_json
from .training_bench import run_training_benchmarks
from .transfer_bench import run_transfer_benchmarks

_CATEGORY_RUNNERS = {
    "forward": run_forward_benchmarks,
    "backward": run_backward_benchmarks,
    "transfer": run_transfer_benchmarks,
    "training": run_training_benchmarks,
}


def main(argv: "list[str] | None" = None) -> None:
    parser = argparse.ArgumentParser(description="Forge benchmark suite (see docs/performance/benchmarking.md)")
    parser.add_argument(
        "--output", default="benchmarks/results/latest.json",
        help="Path to write the JSON results file (default: benchmarks/results/latest.json).",
    )
    parser.add_argument(
        "--categories", nargs="+", default=list(_CATEGORY_RUNNERS), choices=list(_CATEGORY_RUNNERS),
        help="Which benchmark categories to run (default: all).",
    )
    args = parser.parse_args(argv)

    environment = collect_environment()
    print("Environment:")
    for key, value in environment.items():
        print(f"  {key}: {value}")
    print()

    results = []
    for category in args.categories:
        results.extend(_CATEGORY_RUNNERS[category]())

    print(render_table(results))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, environment, output_path)
    print(f"\nSaved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
