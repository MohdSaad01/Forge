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
- `forge.training` -- `Trainer`, metrics, `TrainingHistory`, and `predict()`
  -- the standalone post-training inference path (no `Loss`/`Optimizer`
  required, unlike `Trainer`).
- `forge.serialization` -- `save_model`/`load_model`,
  `save_checkpoint`/`load_checkpoint`, and the module/optimizer
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
from .serialization import Checkpoint, load_checkpoint, load_model, save_checkpoint, save_model
from .tensor import DEFAULT_DTYPE, DType, Tensor
from .training import predict

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
    "save_checkpoint",
    "load_checkpoint",
    "Checkpoint",
    "predict",
]
