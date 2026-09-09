# Forge Long-Range-Recall Example (Milestone 67)

Forge's second recurrent cell, `nn.LSTMCell`, exercised against a synthetic
long-range-dependency task built specifically to expose the reason it was
added: `nn.RNNCell` (Forge's only recurrent cell through M50-M66) provably
loses its backpropagated gradient over long sequences.

```text
build_dataset() -> DataLoader -> RecallModel (RNNCell|LSTMCell -> Linear) -> CrossEntropyLoss -> Adam
```

## The task

Every sequence has `seq_len` timesteps, each a random one-hot `{0, 1}`
symbol, except timestep 0, which is set to the true label. A model must
predict the label from its **final** hidden state alone, after seeing
`seq_len - 1` further random (label-independent) symbols. This is the
classic "temporal order"/copy-task construction from Hochreiter &
Schmidhuber (1997) used to demonstrate the vanilla-RNN vanishing-gradient
problem.

## The evidence (`--gradient-probe`)

```bash
python -m examples.long_range_recall.train --gradient-probe --seq-len 15 30 50
```

One forward+backward pass per `(cell, seq_len)`, untrained random init --
measures `||d loss / d x_0||` vs. `||d loss / d x_{T-1}||` directly on
Forge's own autograd graph:

```text
cell    seq_len     |grad t=0|  |grad t=last|   ratio (last/first)
rnn          15      6.203e-07      2.144e-02            3.457e+04
rnn          30      1.035e-11      2.031e-02            1.962e+09
rnn          50      1.300e-18      1.980e-02            1.523e+16
lstm         15      3.828e-04      1.311e-02            3.425e+01
lstm         30      6.619e-05      1.385e-02            2.093e+02
lstm         50      7.655e-06      1.422e-02            1.857e+03
```

At `seq_len=50`, `RNNCell`'s gradient reaching timestep 0 (`1.3e-18`) is
already far below any usable floating-point/optimizer signal;
`LSTMCell`'s (`7.7e-6`) is not -- a **12-order-of-magnitude** difference at
the identical sequence length. This is `LSTMCell`'s reason for existing,
measured directly rather than assumed. See
`docs/development/m67-lstm-long-range-recall.md` for the full writeup.

## Training

```bash
python -m examples.long_range_recall.train --cell rnn  --seq-len 30
python -m examples.long_range_recall.train --cell lstm --seq-len 30
```

Both cells train to ~100% held-out accuracy (chance = 50%) at moderate
`seq_len` (tested up to 100) given enough epochs -- though not always
smoothly: both cells can show a "loss plateaus near the chance baseline for
many epochs, then breaks through" pattern rather than steady monotonic
improvement, more pronounced for `LSTMCell` in some runs. This is a known
general property of hard credit-assignment tasks trained with Adam, not
specific to either cell.

**Honest limitation**: at `seq_len >= 200`, neither cell reliably converged
with the plain hand-written training loop above in this milestone's own
testing (loss plateaus at the chance baseline for both, budget-limited by
this repository's reference hardware -- see
`docs/development/development-environment.md`). Gradient clipping alone
(tested) did not fix it. Solving that robustly would need techniques
(careful/orthogonal initialization, learning-rate warmup, curriculum
training from short to long sequences) deliberately out of scope for this
milestone -- see the linked report's Limitations/Follow-up-triggers
sections.

## Files

- `dataset.py` -- `build_dataset()`: the synthetic recall task, wrapped in
  the existing `forge.data.TensorDataset`.
- `model.py` -- `RecallModel`: a single `RNNCell` or `LSTMCell` feeding a
  `Linear` classifier head, behind a uniform `init_state()`/`step()`/
  `predict()` interface so `train.py` does not need to know which cell it
  is driving.
- `train.py` -- the runnable example: training/evaluation loop, model
  persistence, and the `--gradient-probe` diagnostic.
