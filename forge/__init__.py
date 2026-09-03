"""Forge: a from-scratch deep-learning framework.

Milestone 1 established the Tensor abstraction and the CPU execution
boundary. Milestone 2 added gradient tracking and reverse-mode autodiff on
top of that Tensor. Milestone 3 added the `nn` module/parameter composition
layer built on top of both. Milestone 4 added loss functions (`nn.MSELoss`,
`nn.CrossEntropyLoss`) and the `optim` optimizer package (`optim.SGD`) that
consumes `Parameter` gradients to update model state. Milestone 5 added the
`data` package (`Dataset`, `TensorDataset`, `DataLoader`, transforms) for
representing training data and producing model-ready batches, independent of
the model/loss/optimizer stack. Milestone 6 added the `training` package
(`Trainer`, metrics, `TrainingHistory`) that orchestrates all of the above
into a reusable training/evaluation workflow, plus `no_grad()` -- a minimal
autograd extension that suspends graph construction during evaluation.
Milestone 7 adds the `serialization` package (`save_model`, `load_model`,
`register_module`) for reconstructing a trained model's architecture and
parameter state after the training process has exited -- see
`docs/architecture/persistence.md`. Milestone 8 adds a real CUDA execution
backend (`forge.backend.cuda`) for a small forward-only operation set
(tensor transfer, `add`/`sub`/`mul`, `matmul`, `sum`), `Tensor.to(device)`
for explicit CPU<->CUDA transfer, and `CUDAError` for CUDA-specific
failures -- see `docs/architecture/cuda-backend.md`. Milestone 22 adds the
`forge.cuda` package (`memory_stats()`, `reset_peak_memory_stats()`) for
observing CUDA allocation/free lifecycle and peak memory usage. Milestone 25
adds an exact-size CUDA caching allocator sitting between `CUDAStorage` and
the driver (`forge.cuda.empty_cache()`; `memory_stats()` grows
`reserved_bytes`/`cached_bytes`/`cache_hit_count`/`cache_miss_count`) -- see
`docs/architecture/cuda-memory-allocator.md`.
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
]
