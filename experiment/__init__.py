"""Milestone 98 experiment workflow: repeatable model-improvement loop.

Deliberately outside `forge/` -- this package is application/experiment
code, not framework code. It reuses only Forge's existing public API
(`forge.train_and_save`, `forge.nn`, `forge.optim`, `forge.training.Accuracy`,
`forge.load_model`/`load_preprocessing`/`predict`) plus
`examples.tabular_diabetes.dataset` (the same real Pima Indians Diabetes
split Milestones 92/95/97 already established) to answer one question:

    Given a baseline model, can a developer change one thing, retrain,
    evaluate on the same held-out data, and decide whether the change
    helped -- using only Forge's public API?

See `experiment/README.md` for the full writeup and results.
"""
