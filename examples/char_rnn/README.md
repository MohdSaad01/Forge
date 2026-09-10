# Forge Char-RNN Example (Milestone 50)

Forge's second real model family, chosen to expand beyond the existing
MNIST (image classification) and `trainer_demo.py`/`data_pipeline_demo.py`
(tabular regression/classification) examples with genuine sequence-modeling
capability:

```text
CharDataset -> DataLoader -> CharRNN (RNNCell -> Linear) -> CrossEntropyLoss -> Adam
```

A small vanilla (Elman) recurrent network trained to predict the next
character of an original, deterministically-generated synthetic corpus
(`corpus.py` -- not downloaded, not copied from any external text; see that
module's docstring). See `docs/development/m50-char-rnn.md` for the full
model-selection rationale, the existing-capability attempt, and why
`Tensor.tanh()`/`nn.RNNCell` were the one genuine blocker this model
exposed.

## Files

- `corpus.py` -- the original synthetic training text.
- `dataset.py` -- `Vocab` (character<->index mapping) and `build_dataset()`
  (fixed-length next-character-prediction sequence pairs, wrapped in the
  existing `forge.data.TensorDataset`).
- `model.py` -- `CharRNN` (`forge.nn.RNNCell` -> `forge.nn.Linear`),
  registered for persistence.
- `train.py` -- the runnable example: a hand-written multi-timestep training
  loop (not `Trainer.fit()` -- see that module's docstring for why),
  sampling, and model persistence.

## Running it

```bash
python -m examples.char_rnn.train --epochs 30 --device cpu
python -m examples.char_rnn.train --epochs 30 --device cuda
```

No download, no external dependencies beyond Forge/NumPy -- the corpus is
generated in-process. Default architecture: a single `RNNCell` with 64
hidden units over a ~30-symbol lowercase-letter/space/period vocabulary,
`seq_len=40`, `batch_size=32`, Adam (`lr=2e-2`).

### Expected approximate behavior (reference: this repository's CPU, i5-7200U)

Mean per-character cross-entropy loss starts near `ln(vocab_size) ≈ 3.4`
nats (an untrained model's uniform-guess baseline) and drops substantially
within 30 epochs as the model learns the corpus's character-level
regularities (word boundaries, common substrings, sentence-final periods).
Exact numbers vary by hardware/seed and are not a stability guarantee --
only "loss decreases well below the untrained baseline" is guaranteed
(and is what `tests/test_char_rnn_example_integration.py` checks, on a
fast synthetic stand-in corpus).

At the end of training, `train.py` samples 200 characters from the trained
model (seeded with `"a tensor"`) and prints them -- with this corpus and
architecture, expect recognizable word fragments and correct spacing/
periods rather than grammatical novel sentences (the corpus is small and
highly repetitive by design; see `corpus.py`).

## CUDA

Identical model/optimizer/data pipeline; only `build_model(..., device=
"cuda")` and each per-timestep input Tensor's `device=` differ. Hardware-
verified on the reference GeForce 940MX: CPU and CUDA runs from the same
`--seed` match within floating-point tolerance for the first several
training steps, confirming CPU/CUDA parity for `Tensor.tanh()` and
`RNNCell` (`tests/test_cuda_backend.py`/`test_cuda_consistency.py`'s new
tanh cases; `tests/test_rnn_cuda.py`'s device-parity case).

## Model persistence

`train.py` demonstrates the plain (optimizer-free) persistence path exactly
like `examples/mnist/train.py`: after training, it records one step's
logits, calls `forge.save_model()`, reloads with `forge.load_model()`, and
asserts the reloaded model reproduces the same logits. This required no
persistence changes beyond registering `CharRNN`/`RNNCell` as persistable
types (`forge/serialization/registry.py`, `model.py`) -- `RNNCell` has no
non-parameter state (no running statistics, unlike e.g. batch
normalization), so the existing parameter-only save/load format already
covers it completely.

## Determinism

`--seed` governs `forge.random` (model parameter initialization) and
`DataLoader` shuffling (its own derived `numpy.random.Generator`), matching
the two-separate-streams convention `examples/mnist/train.py` documents.
Sampling (`generate()`, now a thin wrapper over `forge.training.
generate_sequence()` -- Milestone 75) uses a third, independent
`numpy.random.Generator` seeded from `--seed + 1`.

## Framework additions this example required

- `Tensor.tanh()` (`forge/tensor/tensor.py`) plus `Backend.tanh`/
  `tanh_backward` on both `CPUBackend` and `CUDABackend` (a real CUDA
  kernel, `forge/backend/cuda/kernels.cu`) -- the vanilla RNN's
  nonlinearity. Mirrors the existing `relu`/`exp` Tensor-primitive pattern
  exactly.
- `nn.Tanh` (`forge/nn/activation.py`) -- the `Module` wrapper for
  `Tensor.tanh()`, mirroring `nn.ReLU`; not used by `RNNCell` itself
  (which calls `.tanh()` directly) but added for the same reason `ReLU`
  exists alongside `Tensor.relu()`.
- `nn.RNNCell` (`forge/nn/rnn.py`) -- one vanilla-RNN recurrence step,
  composed from two `Linear` layers and `.tanh()`. No new autograd
  machinery: weight sharing across a Python-level unroll loop was already
  handled correctly by the existing reverse-topological-order gradient
  accumulation in `forge/autograd/engine.py` (verified directly before
  writing any new code -- see `docs/development/m50-char-rnn.md`, Step 3).

No changes were needed to `forge.data`, `forge.optim`, `forge.training`, or
`forge.serialization`'s save/load *format* -- only two new registry entries
(`RNNCell`, and the example's own `CharRNN`).
