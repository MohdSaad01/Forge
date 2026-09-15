# Milestone 98 — Repeatable Model Experiment and Comparison Workflow

This directory is application/experiment code, not Forge framework code. It
answers one question using only Forge's existing public API:

> If the first model isn't good enough, can a developer change one thing,
> retrain, evaluate on the same held-out data, and decide whether the
> change actually helped?

Workload: `examples/tabular_diabetes` (real Pima Indians Diabetes dataset,
768 rows, 60/20/20 split, persisted `Compose([ReplaceValue, Normalize])`
preprocessing, `task="tabular_classification"`) — the same workload
Milestones 92/95/97 already established.

## Files

- `run_experiment.py` — trains and saves one configured variant. Every
  varying parameter (hidden layer widths, learning rate, epoch count, init
  seed) is a CLI flag; the train/val/test split seed is deliberately
  **not** a flag (hardcoded `SPLIT_SEED = 0`) so every experiment is
  evaluated against the same 154-row held-out set
  (`examples/tabular_diabetes/data/diabetes_holdout_eval.csv`).
- `compare.py` — evaluates every trained artifact against that same
  held-out CSV, in a fresh process, reusing
  `examples.tabular_diabetes.evaluate.evaluate_artifact()` unmodified (the
  exact `load_model()` + `load_preprocessing()` + `predict()` + `Accuracy`
  composition Milestone 97 validated). Also hashes each artifact before and
  after evaluation to confirm evaluation never mutates it.
- `artifacts/*.forge` — one independent artifact per experiment run, never
  overwritten.
- `results/*.json` — one structured record per run (config + training-time
  validation metric + artifact path). This is the "experiment config as
  data" file the brief asks for — no tracking database.

## Running it

```bash
# Baseline + two variants, 3 seeds each for the ones under real comparison:
python -m experiment.run_experiment --name baseline_seed0 --hidden 32 16 --lr 1e-3 --init-seed 0
python -m experiment.run_experiment --name capacity_seed0 --hidden 64 32 --lr 1e-3 --init-seed 0
python -m experiment.run_experiment --name fewer_epochs_seed0 --hidden 32 16 --lr 1e-3 --epochs 12 --init-seed 0
python -m experiment.run_experiment --name high_lr_seed0 --hidden 32 16 --lr 1e-2 --init-seed 0
# ... (seed1/seed2 variants for baseline/capacity/fewer_epochs)

python -m experiment.compare
python -m experiment.compare --device cuda
```

## Result (see the full M98 report for the complete writeup)

| experiment    | hidden   | lr    | epochs | seeds | mean test accuracy | decision |
|---------------|----------|-------|--------|-------|---------------------|----------|
| baseline      | [32, 16] | 1e-3  | 60     | 3     | 72.5%               | —        |
| capacity      | [64, 32] | 1e-3  | 60     | 3     | 71.9%               | REJECT   |
| fewer_epochs  | [32, 16] | 1e-3  | 12     | 3     | 72.9%               | KEEP (marginal — within seed spread, see report) |
| high_lr       | [32, 16] | 1e-2  | 60     | 1     | 66.2%               | REJECT   |

Majority-class baseline: 62.3%. All artifacts confirmed byte-unchanged by
evaluation; CPU and CUDA evaluation of every artifact agree exactly;
results reproduced identically (same weights, same metrics) from the
installed wheel in a clean, outside-repo consumer directory.

**Outcome: Forge changes = 0.** The experiment loop (train, save
independently, evaluate on shared held-out data from a fresh process,
compare, decide) worked end-to-end using only public
`forge.train_and_save`, `forge.nn` primitives, `forge.load_model`,
`forge.load_preprocessing`, `forge.predict`, and `forge.training.Accuracy`.
