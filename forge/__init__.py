"""Forge: a from-scratch deep-learning framework.

Forge owns its core ML machinery end-to-end -- Tensor/autograd, CPU and CUDA
execution backends, neural-network modules, optimizers, data loading,
training orchestration, and model/checkpoint persistence -- rather than
wrapping an existing framework. See `README.md` for an overview and
`docs/architecture/architecture.md` for the layered design.

Public subpackages:

- `forge.nn` -- `Module`/`Parameter` composition; layers `Linear`, `Conv2d`,
  `MaxPool2d`, `Conv1d`, `MaxPool1d`, `BatchNorm2d`, `RNNCell`, `Embedding`,
  `Dropout`, `Sequential`, `ReLU`, `Tanh`; losses `MSELoss`, `CrossEntropyLoss`.
- `forge.optim` -- `SGD`, `Adam`.
- `forge.data` -- `Dataset`/`TensorDataset`/`ImageFolder`/`DataLoader`,
  transforms, and CUDA-prefetching (`CUDAPrefetchLoader`).
- `forge.training` -- `train()` (Milestone 79), the single-call high-level
  entry point that trains a `Module` directly on a `Dataset` (building its
  own `DataLoader`(s), moving the model to `device=`, and delegating to
  `Trainer.fit()` underneath) without the caller constructing a `Trainer`
  by hand; `Trainer`, metrics, `TrainingHistory`, `predict()` -- the
  standalone post-training inference path (no `Loss`/`Optimizer`
  required, unlike `Trainer`) -- `interpret_classification()` (Milestone
  72), which turns `predict()`'s raw output plus a saved class vocabulary
  into a human-readable `ClassificationPrediction`, and
  `start_training_session()` (Milestone 73), which builds a fresh `Trainer`
  or resumes one from a checkpoint -- including the `DataLoader` shuffle-
  generator state needed for exact resume equivalence -- in one call
  (`train()` does not cover checkpoint/resume -- see `forge/training/api.py`
  for why), and `generate_sequence()` (Milestone 75), the autoregressive
  sampling loop every stepwise recurrent model (`RNNCell`/`LSTMCell`-based)
  needs to turn a trained model into new generated output; `save_and_verify()`
  (Milestone 78), which saves a model as a portable artifact and immediately
  proves it by reloading it fresh and confirming a sample prediction agrees
  with the pre-save model; `train_and_save()` (Milestone 81), which calls
  `train()` then `save_and_verify()` in one step, returning a
  `TrainAndSaveResult` (history, final validation result, reloaded model);
  `predict_artifact()` (Milestone 82), which turns a portable `.forge`
  image-classification artifact and one new image file directly into a
  `ClassificationPrediction` (or a raw class index, when no `classes=` was
  saved) -- composing `load_model()`/`load_preprocessing()`/`load_classes()`/
  `predict()`/`interpret_classification()` in one call, with no manual
  reconstruction of the training-time preprocessing/interpretation pipeline;
  and `predict_tensor_artifact()` (Milestone 83), the non-classification
  counterpart for a portable artifact whose input is a plain numeric array
  (e.g. a regression model) rather than an image file -- composing
  `load_model()`/`load_preprocessing()`/`predict()`, with preprocessing
  optional and no class-vocabulary concept, returning the model's raw
  numeric prediction `Tensor` directly; and `predict_image_artifact()`
  (Milestone 84), the third artifact shape, for a portable image-to-image
  dense-prediction artifact (e.g. `examples/segmentation`) -- composing
  `load_model()`/`load_preprocessing()`/`ImageFolder._load_image()`/
  `predict()`, then thresholding the model's raw per-pixel output into a
  `{0, 1}`-valued mask `Tensor` ready for `forge.data.save_image()`.
- `forge.serialization` -- `save_model`/`load_model`,
  `save_checkpoint`/`load_checkpoint`, `load_preprocessing` (the
  preprocessing-transform configuration optionally saved alongside a
  model, Milestone 71), `load_classes` (a classification model's saved
  class-name vocabulary, Milestone 72), `inspect_model` (Milestone 85), which
  returns a structured, read-only `ModelInfo` -- model identification,
  preprocessing, classes, format/device -- from a saved artifact without
  reconstructing a live model or requiring CUDA, so a developer holding a
  `.forge` file can answer "what is this?" before choosing which
  `predict_*_artifact()` workflow applies; and the module/optimizer/transform
  reconstruction registries -- see `docs/architecture/persistence.md`.
- `forge.backend` -- device/backend dispatch (`forge.backend.device.Device`);
  `forge.backend.cuda` holds the real CUDA execution backend.
- `forge.cuda` -- the public CUDA API: `is_cuda_available()`, streams,
  synchronization, memory statistics, pinned memory -- see that package's
  own module docstring for the full command list.
- `forge.random` -- process-global RNG seeding for reproducible parameter
  initialization.

This module also re-exports `Tensor`, `DType`, `DEFAULT_DTYPE`, `Device`,
`no_grad`, `predict`, the exception hierarchy, and the persistence/checkpoint
functions above at the top level -- see `__all__`.

Complete example workloads under `examples/` (image classification, both
from a bundled binary format and from ordinary directory-of-image-files via
`ImageFolder`, character/word-level language modeling, tabular regression,
and 1D-convolutional sequence classification) exercise this entire public
surface end-to-end -- see `examples/README.md`.
"""

from . import backend, cuda, data, nn, optim, random, serialization, training
from .autograd import no_grad
from .backend.device import Device
from .exceptions import (
    CUDAError,
    DataError,
    ForgeError,
    GradientStateError,
    LossError,
    ModuleError,
    OptimizerError,
    PersistenceError,
    ShapeMismatchError,
    TrainerError,
    UnsupportedDeviceError,
    UnsupportedDTypeError,
)
from .serialization import (
    Checkpoint,
    ModelInfo,
    inspect_model,
    load_checkpoint,
    load_classes,
    load_model,
    load_preprocessing,
    save_checkpoint,
    save_model,
)
from .tensor import DEFAULT_DTYPE, DType, Tensor
from .training import (
    ClassificationPrediction,
    TrainAndSaveResult,
    generate_sequence,
    interpret_classification,
    predict,
    predict_artifact,
    predict_image_artifact,
    predict_tensor_artifact,
    save_and_verify,
    train,
    train_and_save,
)

__version__ = "0.1.0"

__all__ = [
    "Tensor",
    "DType",
    "DEFAULT_DTYPE",
    "Device",
    "ForgeError",
    "ShapeMismatchError",
    "UnsupportedDTypeError",
    "UnsupportedDeviceError",
    "GradientStateError",
    "ModuleError",
    "LossError",
    "OptimizerError",
    "DataError",
    "TrainerError",
    "PersistenceError",
    "CUDAError",
    "no_grad",
    "nn",
    "optim",
    "random",
    "data",
    "training",
    "serialization",
    "backend",
    "cuda",
    "save_model",
    "load_model",
    "load_preprocessing",
    "load_classes",
    "inspect_model",
    "ModelInfo",
    "save_checkpoint",
    "load_checkpoint",
    "Checkpoint",
    "predict",
    "save_and_verify",
    "predict_artifact",
    "predict_tensor_artifact",
    "predict_image_artifact",
    "generate_sequence",
    "interpret_classification",
    "ClassificationPrediction",
    "train",
    "train_and_save",
    "TrainAndSaveResult",
]
