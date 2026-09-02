"""Forge's command-line interface: `python -m forge ...` / `forge ...` (Milestone 19).

A thin adapter over Forge's existing public APIs -- see
`docs/development/cli.md` for the full command reference. No command
implements new framework logic:

- `forge model inspect` / `forge checkpoint inspect` read archive metadata
  only (`forge/cli/_archive_info.py`) -- read-only, no CUDA required, no RNG
  mutation.
- `forge model convert` / `forge checkpoint convert` call
  `forge.load_model()`/`forge.save_model()`/`forge.load_checkpoint()`/
  `forge.save_checkpoint()` directly.
- `forge benchmark` forwards to the existing `benchmarks` package's own
  `argparse`-based runner.

`import forge` never imports this package (`forge/__init__.py` does not
import `forge.cli`); it is only reached via `python -m forge` or the
installed `forge` console-script entry point.
"""

from __future__ import annotations

import argparse
import sys

from ..exceptions import ForgeError
from . import benchmark, checkpoint, model
from .errors import CLIError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forge", description="Forge command-line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    model.add_parser(subparsers)
    checkpoint.add_parser(subparsers)
    benchmark.add_parser(subparsers)
    return parser


def main(argv: "list[str] | None" = None) -> int:
    parser = build_parser()
    # `parse_known_args`, not `parse_args`: `forge benchmark ...` forwards
    # every argument it doesn't recognize (all of them -- see
    # `forge/cli/benchmark.py`) to the existing `benchmarks` package's own
    # `argparse` runner, unparsed. Every other command still rejects any
    # leftover argument as a normal usage error, immediately below.
    args, extras = parser.parse_known_args(argv)
    try:
        if args.command == "benchmark":
            result = benchmark.run_benchmark(extras)
        else:
            if extras:
                parser.error(f"unrecognized arguments: {' '.join(extras)}")
            result = args.func(args)
    except CLIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ForgeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
