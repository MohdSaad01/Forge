"""Forge's training engine (Milestone 6).

Orchestrates the existing `forge.data` (Dataset/DataLoader), `forge.nn`
(Module/Loss), autograd, and `forge.optim` (Optimizer) components into a
reusable training and evaluation workflow -- `Trainer` computes no
gradients, updates no parameters, and implements no loss/optimizer/batching
logic of its own. See `docs/architecture/training-engine.md`.
"""

from .inference import ClassificationPrediction, generate_sequence, interpret_classification, predict, save_and_verify
from .metrics import Accuracy, MeanAbsoluteError, MeanSquaredError, Metric
from .session import TrainingSession, start_training_session
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
    "predict",
    "save_and_verify",
    "generate_sequence",
    "interpret_classification",
    "ClassificationPrediction",
    "TrainingSession",
    "start_training_session",
]
