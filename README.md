# Forge

**Forge** is a deep-learning framework built from scratch in Python: its own
`Tensor`/autograd engine, CPU and CUDA execution backends, neural-network
modules, optimizers, data pipeline, training loop, and model/checkpoint
persistence. It uses NumPy for CPU array math and a hand-written CUDA
backend for GPU execution, but wraps no existing deep-learning framework --
every layer, gradient, and training step is Forge's own code.

Forge is a solo-developer, pre-release project. It is not a production
framework, is not API-stable, and is not a PyTorch/TensorFlow replacement.
It is, however, real and working: three complete model families train,
evaluate, checkpoint, persist, and reload end-to-end today (see
[Examples](#examples) below), each with hardware-verified CPU/CUDA parity.

## Architecture

```text
Public API / CLI
       |
Training & Evaluation      (forge.training)
       |
Modules / Losses / Optimizers   (forge.nn, forge.optim)
       |
Tensor + Autograd          (forge.tensor, forge.autograd)
       |
Device / Backend Abstraction
       +-- CPU  (NumPy)
       +-- CUDA (hand-written kernels, forge/backend/cuda/kernels.cu)

Data subsystem feeds Training:  forge.data: Dataset -> Transforms -> DataLoader -> Batches
Persistence crosses the model/parameter boundary: forge.serialization
```

See `docs/architecture/architecture.md` for the full design rules and
`docs/architecture/` generally for each layer's own document.

## What's currently in Forge

- **Tensor / autograd** (`forge.Tensor`) -- reverse-mode automatic
  differentiation over elementwise ops, `matmul`, `sum`/`reshape`,
  `relu`/`tanh`/`exp`/`log`/`sqrt`, `conv2d`/`max_pool2d`, `cross_entropy`,
  `embedding_lookup`, `batch_norm2d`, and a vanilla-RNN recurrence step
  (`rnn_cell`) -- all differentiable and CPU/CUDA dispatched identically.
  `no_grad()` suspends graph construction for inference/evaluation.
- **`forge.nn`** -- `Module`/`Parameter` composition; layers `Linear`,
  `Conv2d`, `MaxPool2d`, `Conv1d`, `MaxPool1d`, `BatchNorm2d`, `RNNCell`,
  `Embedding`, `Dropout`, `Sequential`, `ReLU`, `Tanh`; losses `MSELoss`,
  `CrossEntropyLoss`.
- **`forge.optim`** -- `SGD`, `Adam`.
- **`forge.data`** -- `Dataset`/`TensorDataset`, `ImageFolder` (directory-
  per-class image classification, via Pillow decoding), transforms
  (`Normalize`, `Compose`, `Reshape`, `Resize`, ...), `DataLoader` (batching,
  shuffling, `random_split`), and `CUDAPrefetchLoader` for overlapped
  host-to-device transfer.
- **`forge.training`** -- `train()`, the single-call high-level entry point
  (`forge.train(model, dataset, loss=..., optimizer=..., epochs=10)`) that
  builds its own `DataLoader`(s), moves the model to `device=`, and drives
  `Trainer.fit()` underneath -- see [Training, checkpointing, and
  persistence](#training-checkpointing-and-persistence) below; `Trainer`
  (`fit`/`evaluate`/checkpoint resume), metrics (`Accuracy`,
  `MeanAbsoluteError`, ...), `TrainingHistory`, `predict()` -- standalone
  post-training inference (`forge.predict(model, x)`), no `Loss`/`Optimizer`
  required -- `save_and_verify()`, which saves a model and immediately
  proves the file is portable by reloading it fresh and confirming a sample
  prediction agrees; `train_and_save()`, which calls `train()` then
  `save_and_verify()` in one step, returning the completed history, the
  final validation result, and the reloaded, verified model together; and
  `start_training_session()`, which builds a fresh `Trainer` or resumes one
  from a checkpoint (including the `DataLoader` shuffle-generator state
  needed for exact resume equivalence) from one call, replacing the
  resume-or-fresh-start branch every checkpoint-capable example used to
  hand-roll (`train()`/`train_and_save()` themselves have no
  checkpoint/resume -- use `start_training_session()`/`Trainer` directly
  for that).
- **`forge.serialization`** -- `save_model`/`load_model` (architecture +
  parameters, via an explicit module registry -- never arbitrary code
  execution) and `save_checkpoint`/`load_checkpoint` (adds optimizer state,
  epoch/step, and RNG state, for exact training resume). `save_model(...,
  preprocessing=...)`/`load_preprocessing()` optionally save and reconstruct
  a model's required input-preprocessing `Transform` (e.g. `Resize`/
  `Normalize`/`Compose`) alongside it, via the same explicit-registry
  principle. See `docs/architecture/persistence.md`.
- **CUDA backend** (`forge.backend.cuda`, `forge.cuda`) -- a real,
  hardware-tested backend (not simulated): device tensor storage, a caching
  memory allocator, explicit streams, pinned-memory async transfer, and
  kernels for every op above. `forge.cuda.is_cuda_available()` tells you
  whether a working CUDA device is present; every CUDA-specific example
  argument (`--device cuda`) and API (`Tensor(..., device="cuda")`,
  `Trainer(device="cuda")`) is opt-in and raises `forge.CUDAError` cleanly
  when it isn't.
- **CLI** (`forge ...` / `python -m forge ...`) -- `model inspect`/`convert`
  and `checkpoint inspect`/`convert` over the persistence format above, plus
  `forge benchmark`. See `docs/development/cli.md`.

Not in Forge (by design, not oversight): attention/Transformer layers,
convolution beyond 2D, distributed/multi-GPU training, mixed-precision
training, ONNX or any other framework's model-file interop. These are not
planned unless a real workload needs them -- see the **Project scope and
philosophy** section below.

## Installation

Requires Python >= 3.11 and NumPy (installed automatically).

```bash
git clone https://github.com/MohdSaad01/Forge.git
cd Forge
pip install -e .
```

Optional (running the test suite / example demos that use matplotlib):

```bash
pip install -e ".[dev]"
```

Verify the install:

```bash
python -c "import forge; print(forge.__version__)"
```

### CPU vs. CUDA

CPU execution requires nothing beyond the steps above and works on any
platform. CUDA execution additionally requires an NVIDIA GPU, the CUDA
Toolkit (`nvcc` on `PATH`), and on Windows, MSVC (from Visual Studio) --
Forge compiles its own small CUDA kernel library the first time a CUDA
device is actually requested (see `forge/backend/cuda/build.py`); a
CPU-only environment never needs `nvcc` and never pays this cost. Check
`forge.cuda.is_cuda_available()` before relying on CUDA in your own code.
This repository's CUDA path is hardware-verified on one specific
machine/GPU (see `docs/development/development-environment.md`); it is
expected to work on other CUDA-capable NVIDIA GPUs but has not been tested
on hardware this project does not own.

## First model

The smallest complete Forge program -- construct a model, train it, use it
-- using only public APIs:

```python
import numpy as np

import forge
from forge import Tensor
from forge.data import TensorDataset
from forge.nn import Linear
from forge.nn.loss import MSELoss
from forge.optim import SGD

forge.random.seed(0)
rng = np.random.default_rng(0)

# 1-2. Data: y = 3*x1 - 2*x2 + 1, as a Dataset (forge.train() batches it).
X = rng.uniform(-1, 1, size=(200, 2))
y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1)
dataset = TensorDataset(Tensor(X), Tensor(y))

# 3-5. Model, loss, optimizer.
model = Linear(2, 1)
optimizer = SGD(model.parameters(), lr=0.1)

# 6. Train.
forge.train(model, dataset, loss=MSELoss(), optimizer=optimizer, epochs=15, batch_size=16)

# 7. Predict.
prediction = forge.predict(model, Tensor(X[:1]))
print(prediction.numpy())
```

`forge.train()` is Forge's high-level entry point -- it builds the
`DataLoader` and drives `Trainer.fit()` underneath; construct a `DataLoader`
and a `Trainer` directly when you need more control (validation, metrics,
prefetch, checkpoint/resume) -- see [Training, checkpointing, and
persistence](#training-checkpointing-and-persistence) below.
`examples/trainer_demo.py`'s `regression_demo()` solves this exact same
problem the other way, spelling out the `DataLoader`/`Trainer` construction
`forge.train()` does on your behalf above -- read it alongside this section
to see what's happening underneath (with a second classification example
alongside it); run the full script directly:

```bash
python examples/trainer_demo.py
```

## Examples

| Example | What it shows |
|---|---|
| `examples/trainer_demo.py` | The first-model path above, runnable directly. |
| `examples/mnist/` | Image classification (CNN), a real external dataset. |
| `examples/char_rnn/` | Character-level language modeling (`RNNCell`). |
| `examples/word_rnn/` | Word-level language modeling (`Embedding` + `RNNCell`). |
| `examples/regression/` | Tabular regression (MLP over continuous features). |
| `examples/waveform_classification/` | 1D time-series classification (`Conv1d`/`MaxPool1d`). |

`mnist`, `char_rnn`/`word_rnn`, `regression`, and `waveform_classification`
are Forge's production-quality, hardware-verified workload families -- each
trains, evaluates, checkpoints/resumes, and saves/reloads a model on both
CPU and CUDA. See `examples/README.md` for the full index (what each
demonstrates, which Forge APIs it exercises) and each example's own README
for exact commands and expected numbers. Quick start:

```bash
python -m examples.regression.train --epochs 40 --device cpu
python -m examples.mnist.train --download --epochs 3 --device cpu
python -m examples.char_rnn.train --epochs 30 --device cpu
python -m examples.word_rnn.train --epochs 10 --device cpu
python -m examples.waveform_classification.train --epochs 15 --device cpu
```

Pass `--device cuda` in place of `--device cpu` on a machine where
`forge.cuda.is_cuda_available()` is `True`.

## Training, checkpointing, and persistence

`forge.train(model, dataset, loss=..., optimizer=..., epochs=...)` is the
high-level entry point for the common case: it builds a `DataLoader` from
`dataset` (or accepts one directly), moves `model` to `device=` if given,
and runs `Trainer.fit()` underneath, returning the same `TrainingHistory`.
`Trainer.fit()` itself runs the standard loop (forward -> loss -> backward
-> optimizer step, with optional validation) for callers who construct the
`DataLoader`/`Trainer` themselves -- needed for anything `train()` doesn't
expose: CUDA prefetch, a custom `DataLoader` generator, or checkpoint/resume.
`forge.save_checkpoint()`/`load_checkpoint()` capture model + optimizer
state + epoch/step + RNG state so a run can resume exactly where it left
off (`Trainer.resume()`, or `forge.training.start_training_session()` for a
fresh-or-resumed `Trainer` in one call); `forge.save_model()`/`load_model()`
save just the trained architecture and parameters for later inference,
independent of how it was trained -- `forge.save_and_verify(model, path,
sample, preprocessing=..., classes=...)` does that save and immediately
proves the file round-trips by reloading it fresh and comparing a
prediction, and `forge.train_and_save(model, dataset, loss=..., optimizer=
..., epochs=..., path=..., sample=..., preprocessing=..., classes=...)`
composes `train()` + `save_and_verify()` into the one call every
Trainer-based example's `train.py` ends with. Every example under
`examples/` demonstrates both paths -- see `docs/architecture/persistence.md`
for the file format and trust model (no arbitrary code execution on load)
and `docs/architecture/training-engine.md` for `train()`/`train_and_save()`'s
full contract and their `Trainer`/`TrainingSession` boundary.

## Testing

```bash
python -m pytest tests/            # full suite (~1,800 tests, a couple minutes)
python -m pytest tests/test_smoke.py   # fast import + minimal-model smoke check
```

CUDA-specific tests (`tests/test_cuda_*.py` and the `*_cuda_integration.py`
example tests) skip cleanly (`pytest.mark.skipif`) on a machine without a
working CUDA backend; they are hardware-verified on this project's own
reference GPU rather than assumed to pass elsewhere. CPU tests never
require CUDA.

## Where things live

- `forge/` -- the framework itself (public surface documented in
  `forge/__init__.py`'s module docstring).
- `examples/` -- runnable workloads; see `examples/README.md`.
- `tests/` -- the test suite (one file per unit under test, mirroring
  `forge/`'s layout).
- `docs/architecture/` -- per-layer design documents (Tensor, autograd,
  modules, data, CUDA backend/allocator/streams, persistence, CLI).
- `docs/development/` -- the project's milestone-by-milestone history and
  the current roadmap/workflow.
- `benchmarks/` -- performance measurement scripts (`python -m benchmarks
  --help`), separate from the test suite.

## Project scope and philosophy

Forge is built incrementally through small, working vertical slices, driven
by real workload evidence rather than speculative feature addition -- a new
`Tensor` primitive, layer, or optimizer is added only when a concrete
example genuinely needs it (see `docs/development/roadmap.md` and
`docs/development/workflow.md`). It intentionally avoids cloud services and
paid dependencies, and CUDA support is only ever claimed where it has
actually been run on real hardware -- never simulated.

## Contributing / extending Forge

This is currently a solo-developer project; read `CLAUDE.md` and the
relevant `docs/architecture/*.md` file for the layer you're touching before
changing public APIs. Every behavioral change needs a test under `tests/`
(CPU tests must not require CUDA; CUDA tests must skip cleanly without it).
To add a new example workload, follow `examples/regression/`'s structure
(`dataset.py`/`model.py`/`train.py`/`README.md`) as the most recent
precedent.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
details.
