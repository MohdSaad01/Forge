"""Forge's training engine (Milestone 6).

Orchestrates the existing `forge.data` (Dataset/DataLoader), `forge.nn`
(Module/Loss), autograd, and `forge.optim` (Optimizer) components into a
reusable training and evaluation workflow -- `Trainer` computes no
gradients, updates no parameters, and implements no loss/optimizer/batching
logic of its own. See `docs/architecture/training-engine.md`.
"""

from .api import TrainAndSaveResult, TrainingResult, train, train_and_save
from .early_stopping import EarlyStopping
from .evaluation import ClassificationEvaluationResult, RegressionEvaluationResult
from .image_classifier import ImageClassifierResult, train_image_classifier
from .inference import (
    ArtifactPredictor,
    ClassificationPrediction,
    generate_sequence,
    interpret_classification,
    load_predictor,
    predict,
    predict_artifact,
    predict_image_artifact,
    predict_model,
    predict_sequence_artifact,
    predict_tabular_classification_artifact,
    predict_tensor_artifact,
    save_and_verify,
)
from .metrics import Accuracy, MeanAbsoluteError, MeanSquaredError, Metric
from .session import TrainingSession, start_training_session
from .tabular import (
    TabularClassificationResult,
    TabularRegressionResult,
    train_tabular_classifier,
    train_tabular_regressor,
)
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
    "train",
    "train_and_save",
    "TrainingResult",
    "TrainAndSaveResult",
    "predict",
    "save_and_verify",
    "predict_artifact",
    "predict_tensor_artifact",
    "predict_image_artifact",
    "predict_sequence_artifact",
    "predict_tabular_classification_artifact",
    "predict_model",
    "generate_sequence",
    "interpret_classification",
    "ClassificationPrediction",
    "TrainingSession",
    "start_training_session",
    "EarlyStopping",
    "ArtifactPredictor",
    "load_predictor",
    "train_image_classifier",
    "ImageClassifierResult",
    "ClassificationEvaluationResult",
    "RegressionEvaluationResult",
    "train_tabular_classifier",
    "train_tabular_regressor",
    "TabularClassificationResult",
    "TabularRegressionResult",
]
