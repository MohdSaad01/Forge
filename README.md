# Forge

[![CI](https://github.com/MohdSaad01/Forge/actions/workflows/ci.yml/badge.svg)](https://github.com/MohdSaad01/Forge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Forge** is a deep-learning framework built from scratch in Python: its own
`Tensor`/autograd engine, CPU and CUDA execution backends, neural-network
modules, optimizers, data pipeline, training loop, and model/checkpoint
persistence. It uses NumPy for CPU array math and a hand-written CUDA
backend for GPU execution, but wraps no existing deep-learning framework --
every layer, gradient, and training step is Forge's own code.

Forge is a solo-developer, pre-release project. It is not a production
framework, is not API-stable, and is not a PyTorch/TensorFlow replacement.
It is, however, real and working: twelve example workloads spanning five
task types -- image classification, dense image segmentation, sequence
generation, tabular classification, and regression (see
[Examples](#examples) below) -- train, evaluate, checkpoint, persist, and
reload end-to-end today, each with hardware-verified CPU/CUDA parity.

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
  `relu`/`tanh`/`sigmoid`/`exp`/`log`/`sqrt`, `conv2d`/`max_pool2d`/
  `upsample_nearest2d`, `cross_entropy`, `embedding_lookup`, `batch_norm2d`,
  and a vanilla-RNN recurrence step (`rnn_cell`) -- all differentiable and
  CPU/CUDA dispatched identically. `no_grad()` suspends graph construction
  for inference/evaluation.
- **`forge.nn`** -- `Module`/`Parameter` composition; layers `Linear`,
  `Conv2d`, `MaxPool2d`, `Conv1d`, `MaxPool1d`, `UpsampleNearest2d`,
  `BatchNorm2d`, `RNNCell`, `LSTMCell`, `Embedding`, `Dropout`, `Sequential`,
  `Flatten`, `ReLU`, `Tanh`; losses `MSELoss`, `CrossEntropyLoss`.
- **`forge.optim`** -- `SGD`, `Adam`.
- **`forge.data`** -- `Dataset`/`TensorDataset`/`Subset`, `ImageFolder`
  (directory-per-class image classification, via Pillow decoding),
  transforms (`Normalize`, `ReplaceValue`, `Compose`, `Reshape`, `Resize`,
  ...), `DataLoader` (batching, shuffling, `random_split`/
  `sequential_split`), `save_image()`, and `CUDAPrefetchLoader` for
  overlapped host-to-device transfer.
- **`forge.training`** -- a single-call high-level path from a `Dataset` to
  a portable, verified model: `train()` builds its own `DataLoader`(s) and
  drives `Trainer.fit()` underneath, returning a `TrainingResult`
  (history plus the trained model and final-epoch loss/metrics); `predict()`
  is standalone post-training inference with no `Loss`/`Optimizer` required;
  `train_and_save()` composes `train()` with `save_and_verify()` (which
  saves a model and immediately proves the file is portable by reloading it
  fresh and comparing a prediction) into one call returning
  `TrainAndSaveResult`. Lower-level pieces -- `Trainer` (`fit`/`evaluate`/
  checkpoint resume), metrics (`Accuracy`, `MeanAbsoluteError`, ...), and
  `start_training_session()` (fresh-or-resumed `Trainer` in one call,
  including exact `DataLoader`-shuffle resume equivalence) -- remain
  available directly for cases `train()` doesn't cover, such as
  checkpoint/resume. On the consuming side, five task-specific
  `predict_*_artifact()` functions (image classification, plain numeric
  regression, image-to-image segmentation, sequence generation, tabular
  classification) turn a saved `.forge` file into a prediction with no
  manual preprocessing/class-vocabulary reconstruction, and
  `predict_model()` picks the right one automatically from the artifact's
  own `task` metadata. See [Training, checkpointing, and
  persistence](#training-checkpointing-and-persistence) below.
- **`forge.serialization`** -- `save_model`/`load_model` (architecture +
  parameters, via an explicit module registry -- never arbitrary code
  execution) and `save_checkpoint`/`load_checkpoint` (adds optimizer state,
  epoch/step, and RNG state, for exact training resume). `save_model(...,
  preprocessing=...)`/`load_preprocessing()` optionally save and reconstruct
  a model's required input-preprocessing `Transform` alongside it, via the
  same explicit-registry principle. `save_model(..., task=...)` optionally
  declares which of `"classification"`/`"regression"`/`"segmentation"`/
  `"sequence"`/`"tabular_classification"` the artifact represents -- the
  authoritative signal `forge.predict_model()` uses to dispatch reliably.
  `inspect_model()` reads an artifact's architecture/preprocessing/classes/
  task without reconstructing a live model or requiring CUDA. See
  `docs/architecture/persistence.md`.
- **CUDA backend** (`forge.backend.cuda`, `forge.cuda`) -- a real,
  hardware-tested backend (not simulated): device tensor storage, a caching
  memory allocator, explicit streams, pinned-memory async transfer, and
  kernels for every op above. `forge.cuda.is_cuda_available()` tells you
  whether a working CUDA device is present; every CUDA-specific example
  argument (`--device cuda`) and API (`Tensor(..., device="cuda")`,
  `Trainer(device="cuda")`) is opt-in and raises `forge.CUDAError` cleanly
  when it isn't.
- **CLI** (`forge ...` / `python -m forge ...`) -- `model inspect`/`convert`
  and `checkpoint inspect`/`convert` over the persistence format above, and
  `model predict` (task-aware over all five task types above, driven by
  each artifact's own persisted `task` metadata), plus `forge benchmark`.
  See `docs/development/cli.md`.

Not in Forge (by design, not oversight): attention/Transformer layers,
convolution beyond 2D, distributed/multi-GPU training, mixed-precision
training, ONNX or any other framework's model-file interop. These are not
planned unless a real workload needs them -- see the **Project scope and
philosophy** section below.

## Installation

Requires Python >= 3.11 (NumPy and Pillow are installed automatically as
declared runtime dependencies). Forge is not yet published to PyPI, so
install directly from the source repository -- either as a real,
non-editable package (normal consumption) or as an editable checkout
(developing Forge itself).

### Normal installation (consuming Forge)

Install directly from the repository:

```bash
pip install "git+https://github.com/MohdSaad01/Forge.git"
```

or build and install the actual distribution from a local clone, rather
than pointing `pip` at the source tree:

```bash
git clone https://github.com/MohdSaad01/Forge.git
cd Forge
pip install build
python -m build              # -> dist/forge-*.whl, dist/forge-*.tar.gz
pip install dist/forge-*.whl
```

Verify the install from a directory *outside* the cloned repository (so
`import forge` can only resolve to the installed package, not a same-named
local `forge/` directory -- see [Examples](#examples) below for why this
distinction matters when running example scripts):

```bash
python -c "import forge; print(forge.__version__)"
forge --version
```

### Development from source

Working on Forge itself needs an editable install, so local edits take
effect without reinstalling:

```bash
git clone https://github.com/MohdSaad01/Forge.git
cd Forge
pip install -e .
```

Optional (running the test suite / example demos that use matplotlib):

```bash
pip install -e ".[dev]"
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

Forge has 12 example workloads under `examples/`, each with its own
README (exact commands, expected numbers, CUDA verification), plus three
small standalone demo scripts. A representative sample:

| Example | What it shows |
|---|---|
| `examples/trainer_demo.py` | The first-model path above, runnable directly. |
| `examples/mnist/` | Image classification (CNN), a real external dataset. |
| `examples/char_rnn/`, `examples/word_rnn/` | Character- and word-level language modeling (`RNNCell`/`Embedding`). |
| `examples/regression/`, `examples/tabular_diabetes/` | Tabular regression and classification (MLP), the latter on a real external dataset. |
| `examples/segmentation/` | Dense per-pixel prediction (encoder/decoder CNN). |
| `examples/image_folder_classification/` | Classifying image files on disk via a directory-per-class layout. |

Every example above is production-quality and hardware-verified on both
CPU and CUDA (trains, evaluates, checkpoints/resumes, and saves/reloads a
model). See [`examples/README.md`](examples/README.md) for the full index
-- all 12 workloads, what each demonstrates, and which Forge APIs it
exercises. Quick start:

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
and runs `Trainer.fit()` underneath, returning a `TrainingResult` -- a
`TrainingHistory` (so `len()`/indexing/iteration all still work exactly as
before) plus the trained `model` and final-epoch convenience accessors
(`final_train_loss`, `final_val_metrics`, ...). `Trainer.fit()` itself runs
the standard loop (forward -> loss -> backward
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
..., epochs=..., path=..., sample=..., preprocessing=..., classes=...,
task=...)` composes `train()` + `save_and_verify()` into the one call every
Trainer-based example's `train.py` ends with. `forge.inspect_model(path)`
answers "what is this artifact?" -- model architecture summary,
preprocessing, classes, task, format/device -- without reconstructing a live
model or requiring CUDA.

On the consuming side, five task-specific functions turn a saved `.forge`
file into a prediction with no manual reconstruction of training-time
preprocessing/interpretation: `predict_artifact()` (image classification),
`predict_tensor_artifact()` (plain numeric input, e.g. `examples/regression`),
`predict_image_artifact()` (image-to-image dense prediction, e.g.
`examples/segmentation`, returning a mask `Tensor` ready for
`forge.data.save_image()`), `predict_sequence_artifact()` (autoregressive
generation from a stepwise-recurrence model, e.g. `examples/char_rnn`), and
`predict_tabular_classification_artifact()` (numeric input, classification
output, e.g. `examples/tabular_diabetes`). `forge.predict_model(path,
input_data)` picks the right one automatically from the artifact's own
`task=` metadata (falling back to an architecture heuristic only for legacy
artifacts saved without it). See `docs/architecture/persistence.md` for the
file format and trust model (no arbitrary code execution on load) and
`docs/architecture/training-engine.md` for every function's full contract
and the `Trainer`/`TrainingSession` boundary.

## Testing

```bash
python -m pytest tests/                # full suite
python -m pytest tests/test_smoke.py   # fast import + minimal-model smoke check
```

CUDA-specific tests (`tests/test_cuda_*.py` and the `*_cuda_integration.py`
example tests) skip cleanly (`pytest.mark.skipif`) on a machine without a
working CUDA backend; they are hardware-verified on this project's own
reference GPU rather than assumed to pass elsewhere. CPU tests never
require CUDA. CI (`.github/workflows/ci.yml`) runs the CPU-visible half of
the suite plus a real wheel-build/install smoke test on every push to
`main`.

## Project status

**Forge 1.0.** The framework itself (Tensor/autograd, CPU+CUDA backends,
`nn`/`optim`/`data`/`training`/`serialization`, CLI) is feature-complete for
everything the current examples need. M105 validated it end-to-end as an
installable product, and M106 re-verified that same product surface against
the repository as it actually stands -- both flagship workflows
(`examples/mnist`, `examples/tabular_diabetes`) trained, evaluated, saved,
and predicted from freshly-built `wheel` and `sdist` distributions, in a
clean virtual environment, from a directory outside this repository, on
both CPU and (hardware-verified) CUDA -- and found no remaining
external-developer blocker. See
[`docs/development/progress.md`](docs/development/progress.md) for the
full milestone-by-milestone history and
[`docs/development/roadmap.md`](docs/development/roadmap.md) for how
future milestones get chosen.

## Where things live

- `forge/` -- the framework itself (public surface documented in
  `forge/__init__.py`'s module docstring).
- `examples/` -- runnable workloads; see `examples/README.md`.
- `experiment/` -- an application built entirely on Forge's public API for
  repeatably comparing trained models against the same held-out data; see
  `experiment/README.md`.
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
