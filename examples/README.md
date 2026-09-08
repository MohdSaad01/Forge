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
| [`mnist/`](mnist/README.md) | Image classification with a real convolutional network on a real external dataset. | CNN (`Conv2d`, `MaxPool2d`) | + `nn.Conv2d`/`MaxPool2d`/`Flatten`, `optim.Adam`, checkpoint/resume, `forge.save_model`/`load_model` | CPU and CUDA (hardware-verified) |
| [`char_rnn/`](char_rnn/README.md) | Character-level language modeling: a hand-written multi-timestep training loop over a recurrent cell. | Vanilla RNN (`RNNCell`) | + `nn.RNNCell`, `Tensor.tanh()`, model persistence | CPU and CUDA (hardware-verified) |
| [`word_rnn/`](word_rnn/README.md) | Word-level language modeling over a real (1,806-word) vocabulary. | `Embedding` → RNN | + `nn.Embedding`, `Tensor.embedding_lookup()` | CPU and CUDA (hardware-verified) |
| [`regression/`](regression/README.md) | Tabular regression: continuous features with linear, interaction, and quadratic structure. | MLP (`Linear`/`ReLU` stack) | Same as `mnist/` (`Trainer`, `Adam`, checkpoint/resume, persistence) applied to a regression loss/metrics | CPU and CUDA (hardware-verified) |
| [`waveform_classification/`](waveform_classification/README.md) | Classifying fixed-length 1D time series (noisy sine/square/sawtooth/triangle waveforms) by shape. | 1D CNN (`Conv1d`, `MaxPool1d`) | + `nn.Conv1d`/`MaxPool1d` (Milestone 62), same `Trainer`/`Adam`/checkpoint/resume/persistence pipeline as `mnist/` | CPU and CUDA (hardware-verified) |
| [`autoencoder/`](autoencoder/README.md) | Unsupervised image reconstruction through a compressed bottleneck, on real MNIST images -- the first example with no label/target beyond its own input. | Convolutional autoencoder (`Conv2d`/`MaxPool2d` encoder, `UpsampleNearest2d`/`Conv2d` decoder) | + `nn.UpsampleNearest2d` (Milestone 63), same `Trainer`/`Adam`/checkpoint/resume/persistence pipeline as `mnist/` | CPU and CUDA (hardware-verified) |
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

## Running the tests for these examples

Every example under `mnist/`, `char_rnn/`, `word_rnn/`, `regression/`,
`waveform_classification/`, and `autoencoder/` has a corresponding
integration test (CPU) and, where applicable, a CUDA test (skips cleanly
without a working CUDA backend) under `tests/` that exercises the same
pipeline on a small synthetic/fast dataset -- see each example's own README
for the exact `pytest` invocation, or run the whole suite with
`python -m pytest tests/` from the repository root.
