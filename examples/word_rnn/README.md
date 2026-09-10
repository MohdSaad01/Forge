# Forge Word-RNN Example (Milestone 54)

Forge's third real model family, built to expose (and then close) a
measured framework gap: recurrent language modeling over a realistic
word-level vocabulary rather than `examples/char_rnn`'s few-dozen-symbol
character vocabulary.

```text
WordDataset -> DataLoader -> WordRNN (Embedding -> RNNCell -> Linear) -> CrossEntropyLoss -> Adam
```

This script is structurally identical to `examples/char_rnn/train.py` (same
hand-written multi-timestep training loop, same reasoning for not using
`Trainer.fit()` -- see that module's docstring) with one difference:
`WordRNN` looks up each token through `nn.Embedding` instead of building a
one-hot vector. See `docs/development/m54-product-direction.md` for the
full model-selection rationale and the direct measurement (one-hot+`Linear`
is 40x-3500x more compute than a real embedding lookup at this vocabulary
size) that motivated adding `Tensor.embedding_lookup()`/`nn.Embedding`.

## Files

- `corpus.py` -- the original synthetic training text and its fixed word
  vocabulary (not downloaded, not copied from any external text).
- `dataset.py` -- `Vocab` (word<->index mapping) and `build_dataset()`
  (fixed-length next-word-prediction sequence pairs, wrapped in the
  existing `forge.data.TensorDataset`).
- `model.py` -- `WordRNN` (`forge.nn.Embedding` -> `forge.nn.RNNCell` ->
  `forge.nn.Linear`), registered for persistence.
- `train.py` -- the runnable example: a hand-written multi-timestep
  training loop, sampling, and model persistence.

## Running it

```bash
python -m examples.word_rnn.train --epochs 10 --device cpu
python -m examples.word_rnn.train --epochs 10 --device cuda
```

No download, no external dependencies beyond Forge/NumPy -- the corpus is
generated in-process. Default architecture: `Embedding(embedding_dim=32)`
feeding a single `RNNCell` with 128 hidden units, `seq_len=20`,
`batch_size=32`, Adam (`lr=2e-2`), over a synthetic corpus of 23,856 tokens
and 1,806 unique words (1,192 training sequences).

### Expected approximate behavior (reference: this repository's CPU, i5-7200U)

Mean per-word cross-entropy loss starts near `ln(1806) ≈ 7.50` nats (an
untrained model's uniform-guess baseline) and drops steadily over a handful
of epochs as the model learns the corpus's word-level positional structure.
Exact numbers vary by hardware/seed and are not a stability guarantee --
only "loss decreases well below the untrained baseline" is guaranteed (and
is what `tests/test_word_rnn_example_integration.py` checks, on a fast
synthetic stand-in corpus).

At the end of training, `train.py` samples 20 words from the trained model
(seeded with the vocabulary's first two words) and prints them -- expect
recognizable subject/verb/object/punctuation alternation rather than novel
grammatical sentences (the corpus is small and highly repetitive by design;
see `corpus.py`).

## CUDA

Identical model/optimizer/data pipeline; only `build_model(..., device=
"cuda")` and each per-timestep input Tensor's `device=` differ. Hardware
-verified on the reference GeForce 940MX: a 3-epoch CUDA run matches the
CPU run's first three epochs closely (small divergence from
floating-point-order differences in accumulation, not a correctness gap).
As of Milestone 55, the per-batch training loop runs inside an explicit
compute `Stream` (see `examples/char_rnn/train.py::train_one_epoch`'s
docstring for the full rationale) to avoid the CUDA default stream's
per-kernel-launch blocking synchronize.

## Model persistence

`train.py` demonstrates the same plain (optimizer-free) persistence path as
`examples/char_rnn/train.py`/`examples/mnist/train.py`: after training, it
records one step's logits, calls `forge.save_model()`, reloads with
`forge.load_model()`, and asserts the reloaded model reproduces the same
logits. `WordRNN`/`Embedding` needed no persistence-format changes --
`Embedding.weight` is an ordinary `Parameter` that round-trips through the
existing generic parameter save/load path.

## Determinism

`--seed` governs `forge.random` (model parameter initialization) and
`DataLoader` shuffling (its own derived `numpy.random.Generator`), matching
the two-separate-streams convention `examples/mnist/train.py` documents.
Sampling (`generate()`, now a thin wrapper over `forge.training.
generate_sequence()` -- Milestone 75, shared with `examples/char_rnn/
train.py::generate()`) uses a third, independent `numpy.random.Generator`
seeded from `--seed + 1`.

## Framework additions this example required

- `Tensor.embedding_lookup()` (`forge/tensor/tensor.py`) plus
  `Backend.embedding_lookup`/`embedding_lookup_backward` on both
  `CPUBackend` and `CUDABackend` (a real CUDA kernel pair,
  `forge/backend/cuda/kernels.cu`) -- a fused table-row-select primitive,
  needed because one-hot-plus-`Linear` scales with vocabulary size for an
  operation that is architecturally `O(1)` per token.
- `nn.Embedding` (`forge/nn/embedding.py`) -- the `Module` wrapper holding
  the lookup table as an ordinary `Parameter`.

No changes were needed to `forge.data`, `forge.optim`, `forge.training`, or
`RNNCell`/`Linear` themselves -- only one new registry entry (`WordRNN`).
