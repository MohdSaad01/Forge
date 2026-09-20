# Forge

[![CI](https://github.com/MohdSaad01/Forge/actions/workflows/ci.yml/badge.svg)](https://github.com/MohdSaad01/Forge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Forge is a deep learning framework for building, training, evaluating,
persisting, and using machine-learning models in Python.** It provides its
own tensors, automatic differentiation, neural-network modules, optimizers,
data loading, training workflows, and model artifacts, with CPU and CUDA
execution.

## What is Forge?

Forge is written from scratch. It uses NumPy for CPU array math, Pillow for
image decoding, and a hand-written CUDA backend for GPU execution, but it does
not wrap PyTorch, TensorFlow, or any other deep learning framework. Tensors,
gradients, layers, optimizers, the training loop, and the model file format are
all Forge's own code.

The goal is to cover the whole life of a model, not just the training step:

```text
Build a model → Train it → Evaluate it → Save it → Load it later → Use it for inference
```

Forge offers two levels of API. **Low-level building blocks** (tensors,
modules, losses, optimizers, data loaders, a `Trainer`) give you control over
every part of a model and its training. **Higher-level workflows** wrap common
combinations of those pieces so that routine tasks take a few lines. Not every
kind of model has a high-level workflow yet; where one exists, it is built on
the same public building blocks you can use directly.

## What can you build with Forge?

Forge is a general framework. The repository contains runnable, tested
workloads in these areas:

- **Image models:** classification with CNNs (MNIST, a small ResNet-style
  network, classifying image files from a folder), per-pixel segmentation, and
  convolutional autoencoders.
- **Tabular models:** regression and classification with multilayer
  perceptrons, including one on a real external dataset.
- **Sequence and signal models:** character- and word-level language modeling
  with RNNs, an RNN-vs-LSTM comparison on a long-range recall task, and 1D
  convolutional classification of waveforms.

Image classification, tabular classification, and tabular regression have
one-call high-level APIs (see below). They are the demonstrated workflows, not
the scope of the project.

## A simple Forge workflow

```text
Dataset
   ↓
Data pipeline (transforms, batching)
   ↓
Model
   ↓
Training
   ↓
Evaluation
   ↓
Forge artifact (.forge file)
   ↓
Inference
```

A Forge artifact stores the model's architecture and weights, and can also
carry the preprocessing the model needs, its class names, and the kind of task
it performs. Loading an artifact later, in a different process or on a
different day, is enough to run predictions.

## Train and use a model

### Train an image classifier

For the common case of a folder of images with one subfolder per class,
`forge.train_image_classifier()` does the whole job in one call:

```python
import forge

result = forge.train_image_classifier(
    "path/to/dataset",   # dataset/cat/*.jpg, dataset/dog/*.jpg, ...
    path="model.forge",
    epochs=5,
)
print(result.val_metrics["accuracy"])
```

It loads the images with `ImageFolder`, resizes and normalizes them, splits off
a validation set, builds a CNN sized for your classes and image size (or uses a
model you pass in), trains it with cross-entropy loss and Adam, and saves and
verifies a portable artifact. Unreadable image files are reported and skipped
by default. Epochs, batch size, learning rate, image size, validation
fraction, device, and seed are all keyword arguments. A bad output `path` or a
`model=` that does not produce one score per class is reported before the first
epoch, not after training.

### Train a tabular classifier or regressor

For a table of numbers, read the file with your usual tools and pass plain
numeric arrays: `X` is `(samples, features)`, `y` one target per row. Forge has
no CSV or DataFrame layer of its own.

```python
import forge

clf = forge.train_tabular_classifier(
    X, y, path="diabetes.forge",
    classes=["no_diabetes", "diabetes"],   # optional: y holds 0/1 here
)
print(f"{clf.validation_accuracy:.1%} vs {clf.baseline_accuracy:.1%} majority baseline")

reg = forge.train_tabular_regressor(X, y, path="strength.forge")
print(f"validation MSE {reg.validation_mse:.1f} vs {reg.baseline_mse:.1f} predict-the-mean")
```

Each call validates the data, holds out a seeded validation split, fits
per-feature standardization on the **training rows only**, trains a small
multilayer perceptron (or your own `model=`) with early stopping, and saves and
verifies an artifact that carries the fitted preprocessing. The artifact then
takes raw rows, for both `predict()` and `evaluate()`:

```python
predictor = forge.load_predictor(clf.artifact_path)
predictor.predict(new_rows)[0].label
predictor.evaluate(X_test, y_test).accuracy
```

`y` for classification is class names (strings) or integer indices; the class
order is fixed (sorted names, or exactly `classes=`). Regression targets are used
in their own units, not scaled: this works well at moderate magnitudes (Concrete
strength, about 36) and degrades for very large ones (California house prices in
dollars, about 200,000, scored R² 0.67 unscaled against 0.75 standardized by
hand on a 3,000-row sample; if you do that, keep the mean and standard deviation, since
the artifact then predicts in the standardized units). NaN/Inf anywhere in `X` or
a regression `y` is rejected with a `forge.DataError`: Forge does not train
through missing values, so fill or drop them first. If a column encodes "not
measured" as a sentinel such as `0` (as the Pima diabetes data does), pass
`missing_columns=[...]` to replace it with the training-split median. Anything
predictable that would otherwise waste a run -- a wrong-shaped `model=`, an
unwritable `path`, a class with no training rows -- is reported before the
first epoch. Two ready-made artifacts and consumer scripts are in
[`models/tabular_classifier/`](models/tabular_classifier/predict.py) and
[`models/tabular_regressor/`](models/tabular_regressor/predict.py).

### Use a saved model

Any saved artifact can be loaded once and used for predictions. The
preprocessing and class names come from the file, so nothing needs to be
reconstructed by hand:

```python
predictor = forge.load_predictor("model.forge")
prediction = predictor.predict("new_photo.jpg")
print(prediction.label, prediction.confidence)
```

### Evaluate a saved model

The same loaded artifact can be scored on held-out labeled data. The
artifact's own preprocessing is applied automatically, so raw rows go in
exactly as they would for `predict()`:

```python
predictor = forge.load_predictor("examples/tabular_diabetes/artifacts/tabular_diabetes_model.forge")
result = predictor.evaluate(X_holdout, y_holdout)

print(f"{result.accuracy:.1%} vs {result.baseline_accuracy:.1%} majority baseline")
print(result.classes)             # ('no_diabetes', 'diabetes')
print(result.confusion_matrix)    # rows = true class, columns = predicted class
print(result.precision, result.recall)
```

```text
72.1% vs 62.3% majority baseline
('no_diabetes', 'diabetes')
[[78 18]
 [25 33]]
```

Classification artifacts report accuracy, loss, a confusion matrix, and
per-class precision/recall; regression artifacts report `mse`, `mae`, and
`loss`. Image classifiers are evaluated from a directory of
`class_name/image.jpg` files: `predictor.evaluate("held_out_dir")`.

The repository includes a trained model you can use straight away:

```text
models/
└── image_classifier/
    ├── image_model.forge    # trained cat/dog classifier (a Forge artifact)
    └── predict.py           # loads the artifact and classifies one image
```

```powershell
python models/image_classifier/predict.py path/to/image.jpg
```

```text
Forge Image Classifier
----------------------
Image: path/to/image.jpg
Prediction: cat
Confidence: 96.0%
```

(The output above is illustrative; the label and confidence depend on the
image.)

This model is a small example of a trained Forge model being consumed outside
the training process. It is not a benchmark or a general-purpose pretrained
model: it was trained only to tell cats from dogs, so it will answer "cat" or
"dog" for any image you give it.

The artifact was saved from a CUDA run, and Forge does not silently move a
CUDA-saved model onto the CPU. `predict.py` therefore loads it with
`device="cpu"`, so it runs on any machine, with or without a GPU. To use the
same file on CUDA, choose the device yourself (the default, `device=None`,
uses the device recorded in the file and raises `forge.PersistenceError` if
that device is unavailable):

```python
predictor = forge.load_predictor("models/image_classifier/image_model.forge", device="cpu")
# or, on a machine with a working CUDA setup:
predictor = forge.load_predictor("models/image_classifier/image_model.forge", device="cuda")
```

### Lower-level control

`forge.train_image_classifier()` is built from public pieces, and the general
training entry point, `forge.train()`, works for any model and dataset. Here is
a small regression model trained and queried with the building blocks
directly:

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

X = rng.uniform(-1, 1, size=(200, 2))
y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1)
dataset = TensorDataset(Tensor(X), Tensor(y))

model = Linear(2, 1)
optimizer = SGD(model.parameters(), lr=0.1)

forge.train(model, dataset, loss=MSELoss(), optimizer=optimizer, epochs=15, batch_size=16)
print(forge.predict(model, Tensor(X[:1])).numpy())
```

When you need more than `forge.train()` offers (validation and metrics,
CUDA prefetching, checkpoint and resume), build the `DataLoader` and `Trainer`
yourself. `forge.train_and_save()` extends `forge.train()` by saving and
verifying an artifact, and `forge.predict_model()` runs predictions from an
artifact of any supported task type.

## Current capabilities

- **Tensors and autograd:** reverse-mode automatic differentiation over
  elementwise operations, matrix multiplication, reductions, activations,
  1D/2D convolution and pooling, batch normalization, embeddings, and
  recurrent steps. `no_grad()` disables graph construction for inference.
- **Neural-network modules:** `Linear`, `Conv1d`, `Conv2d`, `MaxPool1d`,
  `MaxPool2d`, `UpsampleNearest2d`, `BatchNorm2d`, `RNNCell`, `LSTMCell`,
  `Embedding`, `Dropout`, `Flatten`, `ReLU`, `Tanh`, and `Sequential`, plus
  `MSELoss` and `CrossEntropyLoss`.
- **Optimizers:** `SGD` and `Adam`.
- **Data:** `Dataset` and `TensorDataset`, `ImageFolder` for
  directory-per-class image datasets, composable transforms (`Resize`,
  `Normalize`, `Compose`, and others), `DataLoader` with batching and
  shuffling, train/validation splitting, and a CUDA prefetching loader.
- **Training and evaluation:** `forge.train()`, `Trainer` with validation,
  metrics, and early stopping, and checkpointing with exact resume; one-call
  workflows `train_image_classifier()`, `train_tabular_classifier()`, and
  `train_tabular_regressor()`; `ArtifactPredictor.evaluate()` for saved artifacts.
- **Persistence:** `.forge` model artifacts and training checkpoints. Loading
  reconstructs models only from a registry of known Forge classes and never
  executes code from the file. `forge.inspect_model()` reports what an artifact
  contains without building the model.
- **Inference:** `forge.predict()`, `forge.load_predictor()`,
  `forge.predict_model()`, and per-task prediction functions for image
  classification, numeric regression, image-to-image segmentation, sequence
  generation, and tabular classification.
- **Command line:** `forge model inspect|convert|predict` and
  `forge checkpoint inspect|convert`.
- **Backends:** a NumPy CPU backend that works on any platform, and a CUDA
  backend with hand-written kernels, a caching memory allocator, streams, and
  pinned-memory transfers.

Not currently supported: attention and Transformer layers, 3D convolution,
distributed or multi-GPU training, mixed-precision training, and import or
export of other frameworks' model formats.

## Installation

Forge requires Python 3.11 or newer. It is not published on PyPI, so you
install it from a clone of this repository. NumPy and Pillow are installed
automatically.

```bash
git clone https://github.com/MohdSaad01/Forge.git
cd Forge
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e .
```

Check the install and run a first example:

```bash
python -c "import forge; print(forge.__version__)"
python examples/trainer_demo.py
```

To run the test suite or the examples that use matplotlib, install the
development extras with `pip install -e ".[dev]"`.

The bundled model and the examples live in the repository and are not part of
the installed package, so run them from a clone. To use Forge only as a
library, `pip install "git+https://github.com/MohdSaad01/Forge.git"` installs
the package without them.

**CUDA is optional.** CPU execution needs nothing beyond the steps above. GPU
execution needs an NVIDIA GPU, the CUDA Toolkit (`nvcc` on `PATH`), and on
Windows the MSVC compiler. Forge compiles its kernel library the first time a
CUDA device is requested. Use `forge.cuda.is_cuda_available()` to check. The
CUDA backend has been verified on one GPU (an NVIDIA GeForce 940MX, CUDA 12.6);
other CUDA-capable GPUs are expected to work but have not been tested.

## Examples

Twelve runnable workloads live under [`examples/`](examples/README.md). Each
has its own README with exact commands and expected results.

| Area | Examples |
|---|---|
| Image classification | `mnist`, `resnet`, `image_folder_classification` |
| Dense prediction and reconstruction | `segmentation`, `autoencoder` |
| Tabular | `regression`, `tabular_classification`, `tabular_diabetes` (real dataset) |
| Sequences and signals | `char_rnn`, `word_rnn`, `long_range_recall`, `waveform_classification` |

Most of these train, evaluate, checkpoint, save, and reload a model, and
have been run on both CPU and CUDA. Image-folder classification and tabular
classification/regression have one-call convenience APIs; the others use `forge.train()`,
`forge.train_and_save()`, or the `Trainer` directly, which is what makes them
useful as starting points to copy and adapt.

```bash
python -m examples.regression.train --epochs 40 --device cpu
python -m examples.mnist.train --download --epochs 3 --device cpu
python -m examples.char_rnn.train --epochs 30 --device cpu
```

Use `--device cuda` on a machine where CUDA is available.

## Project status

Forge has a substantial implemented framework surface and is usable for real
workflows: models can be built, trained, evaluated, saved, loaded, and used for
inference on CPU and CUDA, including against real external datasets. It is
developed by a single maintainer and is not a replacement for PyTorch or
TensorFlow.

Development has moved from building out the framework milestone by milestone to
maintaining it and validating it against real use. The test suite, a
wheel-build smoke test, and a real-dataset smoke test guard against
regressions, and new features are added in response to demonstrated needs, bugs,
and missing capabilities rather than to fill a backlog. See
[`docs/development/maintenance.md`](docs/development/maintenance.md) for how
this works in practice.

## What's next

More high-level workflow APIs may follow as Forge matures. The image and
tabular workflows above are a first step toward making common tasks
progressively easier without removing the lower-level building blocks
underneath. What comes next depends on the workloads and problems that turn up
as Forge is used.

## Documentation

- [`examples/README.md`](examples/README.md): index of the example workloads
  and the APIs each one exercises.
- [`docs/architecture/`](docs/architecture/): design documents for each layer
  (tensors, autograd, modules, data, training, persistence, CUDA backend).
  Start with [`architecture.md`](docs/architecture/architecture.md).
- [`docs/architecture/persistence.md`](docs/architecture/persistence.md): the
  `.forge` artifact format and how loading stays safe.
- [`docs/architecture/training-engine.md`](docs/architecture/training-engine.md):
  training and inference APIs in detail.
- [`docs/development/cli.md`](docs/development/cli.md): the command-line
  interface.
- [`docs/development/development-environment.md`](docs/development/development-environment.md):
  the hardware and CUDA setup Forge is verified on.
- [`docs/product/`](docs/product/): vision and scope.
- [`docs/development/progress.md`](docs/development/progress.md): the
  historical record of how Forge was built.

To contribute, read [`CLAUDE.md`](CLAUDE.md) and the architecture document for
the layer you are changing. Behavior changes need tests under `tests/`; CPU
tests must not require CUDA, and CUDA tests must skip cleanly without it.

## License

Forge is licensed under the MIT License. See [LICENSE](LICENSE) for details.
