# Forge Examples

Every example below is a runnable, tested, production-quality workload built
entirely from Forge's public API (`forge`, `forge.nn`, `forge.optim`,
`forge.data`, `forge.training`, `forge.serialization`) -- none of them add
framework logic of their own. Each one has its own README with full detail
(expected numbers, CUDA verification, determinism policy, checkpoint/resume);
this page is only an index so you can find the right one quickly.

| Example | Demonstrates | Model family | Forge APIs exercised | CPU/CUDA |
|---|---|---|---|---|
| [`trainer_demo.py`](trainer_demo.py) | The minimal end-to-end path: dataset → loader → model → loss → optimizer → `Trainer.fit()` → evaluation. Start here. | `Linear` regression, `Linear`→`ReLU`→`Linear` classification | `data.TensorDataset`/`DataLoader`/`random_split`, `nn.Linear`/`ReLU`, `nn.MSELoss`/`CrossEntropyLoss`, `optim.SGD`, `training.Trainer` | CPU only (demo script) |
| [`mnist/`](mnist/README.md) | Image classification with a real convolutional network on a real external dataset -- Forge's flagship example, (Milestone 77) the second example (after `image_folder_classification`) whose saved model file carries its own preprocessing + class vocabulary, with a standalone `infer.py` proving fresh-process portability, (Milestone 79) the first real consumer of the high-level `forge.train()` entry point for its non-`--resume` path, and (Milestone 81) now `forge.train_and_save()` for that same path. | CNN (`Conv2d`, `MaxPool2d`) | + `nn.Conv2d`/`MaxPool2d`/`Flatten`, `optim.Adam`, `training.train_and_save()` (Milestone 81), checkpoint/resume, `serialization.save_model(..., preprocessing=..., classes=...)`/`load_model` | CPU and CUDA (hardware-verified) |
| [`char_rnn/`](char_rnn/README.md) | Character-level language modeling: a hand-written multi-timestep training loop over a recurrent cell. | Vanilla RNN (`RNNCell`) | + `nn.RNNCell`, `Tensor.tanh()`, model persistence | CPU and CUDA (hardware-verified) |
| [`word_rnn/`](word_rnn/README.md) | Word-level language modeling over a real (1,806-word) vocabulary. | `Embedding` → RNN | + `nn.Embedding`, `Tensor.embedding_lookup()` | CPU and CUDA (hardware-verified) |
| [`regression/`](regression/README.md) | Tabular regression: continuous features with linear, interaction, and quadratic structure -- (Milestone 83) its fresh-training path now trains and saves through `forge.train_and_save()`, with the fitted `Normalize` feature transform persisted as `preprocessing=`, and a portable artifact consumed via `forge.predict_tensor_artifact()`, the non-classification counterpart to `image_folder_classification`'s `forge.predict_artifact()`. | MLP (`Linear`/`ReLU` stack) | Same as `mnist/` (`Trainer`, `Adam`, checkpoint/resume, persistence) applied to a regression loss/metrics, plus `training.train_and_save()`/`predict_tensor_artifact()` (Milestone 83) | CPU and CUDA (hardware-verified) |
| [`waveform_classification/`](waveform_classification/README.md) | Classifying fixed-length 1D time series (noisy sine/square/sawtooth/triangle waveforms) by shape. | 1D CNN (`Conv1d`, `MaxPool1d`) | + `nn.Conv1d`/`MaxPool1d` (Milestone 62), same `Trainer`/`Adam`/checkpoint/resume/persistence pipeline as `mnist/` | CPU and CUDA (hardware-verified) |
| [`autoencoder/`](autoencoder/README.md) | Unsupervised image reconstruction through a compressed bottleneck, on real MNIST images -- the first example with no label/target beyond its own input. | Convolutional autoencoder (`Conv2d`/`MaxPool2d` encoder, `UpsampleNearest2d`/`Conv2d` decoder) | + `nn.UpsampleNearest2d` (Milestone 63), same `Trainer`/`Adam`/checkpoint/resume/persistence pipeline as `mnist/` | CPU and CUDA (hardware-verified) |
| [`long_range_recall/`](long_range_recall/README.md) | `RNNCell` vs. `LSTMCell` on a synthetic long-range-dependency task -- measures and demonstrates the vanishing-gradient gap `LSTMCell` closes. | Vanilla RNN (`RNNCell`) or LSTM (`LSTMCell`) | + `nn.LSTMCell`, `Tensor.sigmoid()` (Milestone 67), model persistence | CPU and CUDA (hardware-verified) |
| [`image_folder_classification/`](image_folder_classification/README.md) | Classifying ordinary image *files on disk* (a generated, **mixed-resolution** circle/square/triangle shape dataset) via a directory-per-class layout -- the first example whose data source isn't a bundled format or in-memory array, (Milestone 71) the first whose saved model file carries its own required input preprocessing, reconstructed automatically by a separate `infer.py` process, and (Milestone 81) its fresh-training path now trains and saves through one `forge.train_and_save()` call. | CNN (`Conv2d`/`BatchNorm2d`/`MaxPool2d`/`Dropout`) | + `data.ImageFolder`/`IMAGE_EXTENSIONS` (Milestone 69), `data.transforms.Resize` (Milestone 70), `serialization.save_model(..., preprocessing=...)`/`load_preprocessing()` (Milestone 71), `data.random_split`, `training.train_and_save()` (Milestone 81), `training.predict()` | CPU and CUDA (hardware-verified) |
| [`data_pipeline_demo.py`](data_pipeline_demo.py) | `Dataset` → `Transform` → `DataLoader` → batches feeding a model, with no training loop at all. | N/A (pipeline only) | `data.TensorDataset`, `data.transforms.Normalize`, `data.DataLoader` | CPU only (demo script) |
| [`persistence_demo.py`](persistence_demo.py) | Proof that a saved model survives a process boundary: train in one process, load and predict in a fresh subprocess. | `Linear` regression | `forge.save_model`/`load_model` | CPU only (demo script) |

## Which one should I run first?

`trainer_demo.py` -- it is the smallest complete loop (construct a model,
train it with `Trainer.fit()`, evaluate it) and finishes in well under a
second on CPU:

```bash
python examples/trainer_demo.py
```

See `README.md`'s **First model** section for a line-by-line walkthrough of
what it does.

## The "real workload" examples

`mnist`, `char_rnn`/`word_rnn`, `regression`, `waveform_classification`, and
`autoencoder` are Forge's "real workload" examples -- each is validated
end-to-end (training, evaluation, checkpoint save/resume, model
persistence, CPU/CUDA parity where applicable) at the same standard, not
toy scripts. The first three are Forge's vision-named workload families
(`docs/product/vision.md`/`docs/product/use-cases.md`'s UC1-UC3);
`waveform_classification` (Milestone 62) is UC1 again, expressed with a
model family (1D temporal convolution, `nn.Conv1d`/`nn.MaxPool1d`) none of
the others cover; `autoencoder` (Milestone 63) is a genuinely new task
shape -- unsupervised reconstruction through a compressed bottleneck,
requiring a new primitive (`nn.UpsampleNearest2d`) no prior example needed.
If you're evaluating whether Forge can support a workload shaped like
yours, read the closest match's own README first; the model/data code is
designed to be a starting point you copy and adapt, not a fixed template.

## The shared training workflow (Milestones 73/78/79/80/81/82/83)

Every `Trainer`-based example above drives shared abstractions rather than
hand-rolling its own plumbing. `mnist` (Milestone 79/81), `regression`
(Milestone 83), and `image_folder_classification` (Milestone 80/81) all
split into two branches: the common, non-`--resume` case trains through
`forge.train_and_save()` (`forge/training/api.py`, Milestone 81), the
single call that trains via `forge.train()` (building its own `DataLoader`(s)
and driving `Trainer.fit()` underneath) and then immediately saves + verifies
the result via `save_and_verify()`, while `--resume` keeps using the
lower-level API each script already needed (`mnist`: a plain `Trainer` +
`trainer.resume()` + `save_and_verify()`; `regression`/
`image_folder_classification`: `forge.training.start_training_session()`
(Milestone 73) + `save_and_verify()`, preserving Milestone 73's exact
`DataLoader`-shuffle resume-equivalence guarantee -- see `docs/architecture/
training-engine.md`'s **Single-call high-level training** and **Train,
evaluate, persist, verify in one call** sections for exactly how the fresh
path keeps that guarantee). Every `Trainer`-based example's saved model,
regardless of which of those it uses, is saved and verified through
`forge.training.save_and_verify()` (Milestone 78, called directly by the
lower-level paths above, or via `train_and_save()`), which saves the trained
model as a portable artifact and immediately proves it by reloading it
fresh and confirming a sample prediction agrees with the pre-save model --
raising `forge.PersistenceError` (not a bare `assert`) if it ever doesn't.
`mnist` and `image_folder_classification` additionally have automated,
subprocess-based tests that launch their `infer.py` as a genuinely separate
OS process (Milestone 80 -- see `docs/architecture/training-engine.md`'s
**Fresh-process verification** section); `regression` has an equivalent
subprocess-based test that calls `forge.predict_tensor_artifact()` directly
(Milestone 83, no separate `infer.py`; see `tests/
test_regression_artifact_workflow.py`). Two portable-artifact inference
functions now exist for the two materially different artifact shapes
Forge's examples produce: `forge.predict_artifact()` (Milestone 82, image
files, mandatory preprocessing, class interpretation -- `mnist`/
`image_folder_classification`) and `forge.predict_tensor_artifact()`
(Milestone 83, plain numeric input, optional preprocessing, no class
concept -- `regression`). See `docs/architecture/training-engine.md`'s
**Reusable training sessions**, **Single-call high-level training**,
**Portable-artifact save + verify**, **Train, evaluate, persist, verify in
one call**, **Portable-artifact inference in one call**, and
**Portable-artifact inference for numeric input** sections for the full
contract.

## Running the tests for these examples

Every example under `mnist/`, `char_rnn/`, `word_rnn/`, `regression/`,
`waveform_classification/`, `autoencoder/`, and `image_folder_classification/`
has a corresponding integration test (CPU) and, where applicable, a CUDA
test (skips cleanly without a working CUDA backend) under `tests/` that
exercises the same pipeline on a small synthetic/fast dataset -- see each
example's own README for the exact `pytest` invocation, or run the whole
suite with `python -m pytest tests/` from the repository root.
