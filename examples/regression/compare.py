"""Compare two `examples/regression/train.py` runs (Milestone 65).

```bash
python -m examples.regression.compare artifacts/regression_history.json artifacts2/regression_history.json
```

Reads the two JSON run records `train.py` writes (see `experiment.py`) and
prints their configuration differences plus final loss/metric/epoch-count/
runtime deltas. No UI, no visualization -- text output only, per this
milestone's scope guardrails (`docs/development/m65-reproducible-training.md`).
"""

from __future__ import annotations

import argparse
import sys

try:
    from .experiment import compare_runs, load_run_record
except ImportError:  # running as a plain script
    from experiment import compare_runs, load_run_record


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a", help="Path to the first run's *_history.json record.")
    parser.add_argument("run_b", help="Path to the second run's *_history.json record.")
    return parser.parse_args(argv)


def format_report(diff: dict, label_a: str, label_b: str) -> str:
    lines = [f"Comparing:\n  A: {label_a}\n  B: {label_b}\n"]

    if diff["config_diff"]:
        lines.append("Configuration differences:")
        for key, (value_a, value_b) in sorted(diff["config_diff"].items()):
            lines.append(f"  {key}: A={value_a!r}  B={value_b!r}")
    else:
        lines.append("Configuration: identical.")

    lines.append("")
    lines.append(f"Epochs trained:     A={diff['epochs']['a']}  B={diff['epochs']['b']}")
    lines.append(f"Final train loss:   A={diff['final_train_loss']['a']}  B={diff['final_train_loss']['b']}")
    lines.append(f"Final eval loss:    A={diff['final_eval_loss']['a']}  B={diff['final_eval_loss']['b']}")
    lines.append(f"Final eval metrics: A={diff['final_eval_metrics']['a']}  B={diff['final_eval_metrics']['b']}")
    lines.append(
        f"Total runtime (s):  A={diff['total_duration_seconds']['a']}  B={diff['total_duration_seconds']['b']}"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)

    record_a = load_run_record(args.run_a)
    if record_a is None:
        print(f"error: no run record found at '{args.run_a}'", file=sys.stderr)
        return 1
    record_b = load_run_record(args.run_b)
    if record_b is None:
        print(f"error: no run record found at '{args.run_b}'", file=sys.stderr)
        return 1

    diff = compare_runs(record_a, record_b)
    print(format_report(diff, args.run_a, args.run_b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
