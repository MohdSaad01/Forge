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
combinations of those pieces so that routine tasks take a few lines. One-call
workflows exist for image classification, tabular classification, and tabular
regression; other kinds of model use the building blocks directly. Every
high-level workflow is built on the same public building blocks you can use
yourself.

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

For a table of numbers, pass plain numeric arrays: `X` is `(samples, features)`,
`y` one target per row. A numeric CSV file with a header row is read into those
arrays by `forge.data.load_csv()` -- the target column is named, every other
column is a numeric feature; the training functions themselves take arrays and
Forge has no DataFrame layer:

```python
import forge

X, y = forge.data.load_csv("diabetes.csv", target="Outcome", labels=True)    # classification labels
clf = forge.train_tabular_classifier(
    X, y, path="diabetes.forge",
    classes=["no_diabetes", "diabetes"],   # optional: y holds 0/1 here
)
print(f"{clf.validation_accuracy:.1%} vs {clf.baseline_accuracy:.1%} majority baseline")

X, y = forge.data.load_csv("concrete.csv", target="strength")                # regression target
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
in their own units by default: this works well at moderate magnitudes (Concrete
strength, about 36) and degrades for very large ones (California house prices in
dollars, about 200,000, scored R² 0.665 on a 3,000-row sample, using all 500
epochs). For those pass `target_transform="standardize"`: the model trains on
z-scores (fitted on the training rows only, R² 0.749 on the same data in about a
quarter of the time) and the artifact stores the transform, so `predict()` and
`evaluate()` still return and compare the original units -- nothing to invert or
remember. Don't rescale `y` yourself; the artifact would then predict in the
rescaled units. NaN/Inf anywhere in `X` or
a regression `y` is rejected with a `forge.DataError`: Forge does not train
through missing values, so fill or drop them first. If a column encodes "not
measured" as a sentinel such as `0` (as the Pima diabetes data does), pass
`missing_columns=[...]` to replace it with the training-split median. Anything
predictable that would otherwise waste a run -- a wrong-shaped `model=`, an
unwritable `path`, a class with no training rows -- is reported before the
first epoch. Two ready-made artifacts and consumer scripts are in
[`models/tabular_classifier/`](models/tabular_classifier/predict.py) and
[`models/tabular_regressor/`](models/tabular_regressor/predict.py).

**Column names.** An artifact always knows how many features it takes; with
`feature_names=` it also knows *which* feature each column is, so a CSV whose columns
are in another order is matched by name instead of silently scored wrongly (on the
Pima holdout, two swapped columns took accuracy from 72.7% to 39.0% with no error):

```python
X, y, names = forge.data.load_csv("diabetes.csv", target="Outcome", labels=True, return_feature_names=True)
forge.train_tabular_classifier(X, y, path="diabetes.forge", feature_names=names)   # names are recorded in the artifact

Xn, names_n = forge.data.load_csv_features("new_patients.csv")                       # header names travel with the rows
forge.load_predictor("diabetes.forge").predict(Xn, feature_names=names_n)
```

The same names in another order are reordered into the artifact's order; a missing,
unknown, extra (an `id` column is an extra column, never dropped) or misspelled name is a
`forge.DataError` naming the columns -- Forge never guesses which column is which, and
matches names exactly (case and whitespace count). A bare NumPy array or `.npy` file has no
names, so it is checked for width only, exactly as before; an artifact trained without
names (every artifact saved before this existed) keeps working and cannot check columns.
See [`docs/development/m119-persisted-tabular-feature-schema.md`](docs/development/m119-persisted-tabular-feature-schema.md).

**Column selection.** A real CSV often carries a column that is not a feature -- an `id`,
say -- and `columns=[...]` selects and orders exactly the columns to read, leaving the rest
(the `id` included) unread and unvalidated:

```python
X, y, names = forge.data.load_csv(
    "diabetes_with_id.csv", target="Outcome", labels=True, return_feature_names=True,
    columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
)
```

`forge model predict`/`forge model evaluate` accept the identical `--columns NAME [NAME
...]` flag. There is still no automatic id/timestamp detection -- selection is always
explicit. An existing artifact that has no feature names (trained before this existed, or
from unnamed arrays) can have them attached without retraining: `forge model convert
MODEL OUTPUT --device ... --feature-names NAME ...` -- a metadata-only edit; no weight or
prediction changes. See [`docs/development/m120-tabular-api-convergence.md`](docs/development/m120-tabular-api-convergence.md).

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

The same evaluation is available from the command line, with no Python script.
Inputs are a CSV file with a header row plus `--target COLUMN`, `.npy` files
(`numpy.save()`), or an image directory for image classifiers:

```bash
forge model evaluate diabetes.forge holdout.csv --target Outcome            # readable report
forge model evaluate housing.forge held_out.csv --target median_house_value --json   # machine-readable
forge model evaluate diabetes.forge X.npy y.npy                              # the same, from .npy files
forge model evaluate pets.forge held_out_dir                                 # image classifier
```

The command only reads the files, calls `load_predictor(...).evaluate(...)`, and
prints its result, so the numbers are identical to the Python API's -- including
native-unit metrics for a regression artifact saved with
`target_transform="standardize"`. A CSV is a plain comma-delimited, UTF-8 file with
a header; every column except `--target` must be numeric, and empty cells, NaN/Inf
and text in a feature column are errors (Forge reads them, it never repairs them).
Against an artifact trained with `feature_names=`, the CSV's header names are matched to
the model's columns: a reordered file scores exactly like the ordered one and a wrong
column is one `Error:` line. `forge model predict MODEL rows.csv` (feature columns only)
follows the same rule. `--columns NAME [NAME ...]` on both commands selects and orders
exactly those header columns as features, so a file with an `id` column needs no rewriting.
See [`docs/development/cli.md`](docs/development/cli.md),
[`docs/development/m118-csv-tabular-workflow.md`](docs/development/m118-csv-tabular-workflow.md) and
[`docs/development/m120-tabular-api-convergence.md`](docs/development/m120-tabular-api-convergence.md).

Training itself is also a command, one thin adapter over `train_tabular_classifier_csv()` /
`train_tabular_regressor_csv()` / `train_image_classifier()`, chosen by `--task`:

```bash
forge model train diabetes.csv --task classification --target Outcome --output diabetes.forge
forge model train housing.csv --task regression --target median_house_value --output housing.forge \
    --target-transform standardize
forge model train petimages/ --task image-classification --output pets.forge --epochs 5
```

Each writes an ordinary `.forge` artifact, immediately usable with `model inspect|predict|evaluate`
above -- no CLI-only artifact format. See
[`docs/development/m121-tabular-csv-training.md`](docs/development/m121-tabular-csv-training.md) and
[`docs/development/m122-unified-model-training-cli.md`](docs/development/m122-unified-model-training-cli.md).

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
  `train_tabular_regressor()` (plus `train_tabular_classifier_csv()` /
  `train_tabular_regressor_csv()` for a CSV file); `ArtifactPredictor.evaluate()`
  for saved artifacts.
- **CSV and tabular artifacts:** `forge.data.load_csv()` (with column selection)
  reads a numeric CSV without pandas; tabular artifacts can persist their
  feature names, so named CSV input is matched by name, and regression
  artifacts can persist a target standardization so predictions come back in
  the target's own units.
- **Persistence:** `.forge` model artifacts and training checkpoints. Loading
  reconstructs models only from a registry of known Forge classes and never
  executes code from the file. `forge.inspect_model()` reports what an artifact
  contains without building the model.
- **Inference:** `forge.predict()`, `forge.load_predictor()`,
  `forge.predict_model()`, and per-task prediction functions for image
  classification, numeric regression, image-to-image segmentation, sequence
  generation, and tabular classification.
- **Command line:** `forge model inspect|convert|predict|evaluate|train` and
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

**Forge 1.0.0 is feature-frozen.** The package version, `forge --version`, and
the built wheel all report `1.0.0`. It is usable for real workflows: models can
be built, trained, evaluated, saved, loaded, and used for inference on CPU and
CUDA, including against real external datasets. It is developed by a single
maintainer and is not a replacement for PyTorch or TensorFlow. It is not
published on PyPI.

The closing validation (after Milestone 122) ran the full test suite (3,904
passed, 0 failed, 0 skipped on the reference machine, CUDA present) and
exercised, on both CPU and CUDA, real image data, real tabular classification
and regression data, the Python API and CLI workflows, and artifact
portability (fresh process, and CPU to CUDA). A wheel built from a clean
`git archive` installed and worked from outside the repository. CI runs the
CPU-visible tests and a packaging smoke test on Python 3.11 and 3.13. CUDA has
been hardware-verified on one GPU only (see Installation).

## What's next

No further feature work is planned. Forge is in maintenance: bug fixes,
regressions, and documentation corrections, driven by problems found in real
use. See [`docs/development/maintenance.md`](docs/development/maintenance.md).

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
