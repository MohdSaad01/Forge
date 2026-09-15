"""Milestone 98: evaluate every trained experiment on the same held-out set and decide.

```bash
python -m experiment.compare
python -m experiment.compare --device cuda
```

This is the "evaluate on the same held-out data, compare, decide" half of
the M98 loop. It deliberately reuses
`examples.tabular_diabetes.evaluate.evaluate_artifact()` unmodified --
exactly the composition (`forge.load_model()` + `forge.load_preprocessing()`
+ `forge.predict()` + `forge.training.Accuracy`) Milestone 97 already
established and validated -- against
`examples/tabular_diabetes/data/diabetes_holdout_eval.csv`, the same 154-row
held-out split every `experiment/run_experiment.py` run trains against
(`SPLIT_SEED = 0` in both places). No new evaluation logic was written for
this milestone: the only Forge-adjacent code this script adds is glue that
loops the existing evaluator over several artifacts and prints a comparison.

Runs entirely as its own process, independent of `run_experiment.py`, so
every evaluation here is a genuine train -> save -> **fresh process** ->
evaluate step (M98 brief Section 19/37), not a re-use of an in-memory model.

For every artifact this also hashes the file before and after evaluation to
confirm evaluation never mutates it (M98 brief Section 35).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from examples.tabular_diabetes.evaluate import evaluate_artifact

_HERE = Path(__file__).parent
_HOLDOUT_CSV = _HERE.parent / "examples" / "tabular_diabetes" / "data" / "diabetes_holdout_eval.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_results(results_dir: Path) -> "list[dict]":
    records = []
    for path in sorted(results_dir.glob("*.json")):
        records.append(json.loads(path.read_text()))
    return records


def evaluate_all(records: "list[dict]", device: "str | None" = None) -> "list[dict]":
    rows = []
    for record in records:
        artifact_path = Path(record["artifact_path"])
        before_hash = _sha256(artifact_path)
        summary = evaluate_artifact(str(artifact_path), str(_HOLDOUT_CSV), device=device)
        after_hash = _sha256(artifact_path)
        rows.append({
            **record,
            "test_accuracy": summary.accuracy,
            "test_baseline_accuracy": summary.baseline_accuracy,
            "test_improvement_pp": summary.improvement_pp,
            "artifact_unchanged": before_hash == after_hash,
        })
    return rows


def group_by_config(rows: "list[dict]") -> "dict[str, list[dict]]":
    """Group seed-repeated runs by their config identity (name with the trailing _seedN stripped)."""
    groups: "dict[str, list[dict]]" = {}
    for row in rows:
        base = row["name"].rsplit("_seed", 1)[0] if "_seed" in row["name"] else row["name"]
        groups.setdefault(base, []).append(row)
    return groups


def summarize_group(name: str, group: "list[dict]") -> dict:
    accuracies = [r["test_accuracy"] for r in group]
    mean = sum(accuracies) / len(accuracies)
    spread = (max(accuracies) - min(accuracies)) if len(accuracies) > 1 else 0.0
    return {
        "name": name,
        "n_seeds": len(group),
        "hidden": group[0]["hidden"],
        "lr": group[0]["lr"],
        "epochs": group[0]["epochs"],
        "mean_test_accuracy": mean,
        "spread": spread,
        "all_artifacts_unchanged": all(r["artifact_unchanged"] for r in group),
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(_HERE / "results"))
    parser.add_argument("--baseline", default="baseline", help="Config-group name to compare every other group against.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Device to evaluate on (default: whatever each artifact was saved from).")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    records = load_results(Path(args.results_dir))
    if not records:
        raise SystemExit(f"No experiment result JSON files found in {args.results_dir}. Run experiment.run_experiment first.")

    rows = evaluate_all(records, device=args.device)
    groups = group_by_config(rows)
    summaries = {name: summarize_group(name, group) for name, group in groups.items()}

    baseline_majority = rows[0]["test_baseline_accuracy"]
    print(f"Held-out set:        {_HOLDOUT_CSV}")
    print(f"Samples:             {rows[0]['test_samples']}")
    print(f"Majority baseline:   {baseline_majority:.1%}")
    print(f"Evaluation device:   {args.device or '(artifact default)'}")
    print()
    header = f"{'experiment':<16}{'hidden':<12}{'lr':<10}{'epochs':<8}{'seeds':<7}{'mean test acc':<16}{'spread':<10}{'immutable':<10}"
    print(header)
    print("-" * len(header))
    for name, s in sorted(summaries.items(), key=lambda kv: -kv[1]["mean_test_accuracy"]):
        print(f"{s['name']:<16}{str(s['hidden']):<12}{s['lr']:<10}{s['epochs']:<8}{s['n_seeds']:<7}"
              f"{s['mean_test_accuracy']:<16.1%}{s['spread']:<10.1%}{str(s['all_artifacts_unchanged']):<10}")

    print()
    if args.baseline not in summaries:
        print(f"(No group named '{args.baseline}' -- skipping keep/reject decisions.)")
        return

    baseline_acc = summaries[args.baseline]["mean_test_accuracy"]
    print(f"Baseline ('{args.baseline}') mean test accuracy: {baseline_acc:.1%} "
          f"({(baseline_acc - baseline_majority) * 100:+.1f}pp vs majority-class baseline)")
    print()
    for name, s in summaries.items():
        if name == args.baseline:
            continue
        diff_pp = (s["mean_test_accuracy"] - baseline_acc) * 100
        decision = "KEEP" if s["mean_test_accuracy"] > baseline_acc else "REJECT"
        print(f"{name:<16} mean test accuracy {s['mean_test_accuracy']:.1%} "
              f"({diff_pp:+.1f}pp vs baseline) -> {decision}")


if __name__ == "__main__":
    main()
