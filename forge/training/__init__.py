"""Forge's training engine (Milestone 6).

Orchestrates the existing `forge.data` (Dataset/DataLoader), `forge.nn`
(Module/Loss), autograd, and `forge.optim` (Optimizer) components into a
reusable training and evaluation workflow -- `Trainer` computes no
gradients, updates no parameters, and implements no loss/optimizer/batching
logic of its own. See `docs/architecture/training-engine.md`.
"""

from .metrics import Accuracy, MeanAbsoluteError, MeanSquaredError, Metric
from .trainer import EpochResult, EvaluationResult, Trainer, TrainingHistory

__all__ = [
    "Trainer",
    "TrainingHistory",
    "EpochResult",
    "EvaluationResult",
    "Metric",
    "MeanSquaredError",
    "MeanAbsoluteError",
    "Accuracy",
]
