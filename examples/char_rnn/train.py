"""Forge Milestone 50: a small character-level RNN language-model example.

```text
CharDataset -> DataLoader -> CharRNN (RNNCell -> Linear) -> CrossEntropyLoss -> Adam
```

Every step uses only public Forge APIs (`forge`, `forge.data`, `forge.nn`,
`forge.optim`, `forge.save_model`/`load_model`) plus the two additions this
milestone made to `forge.nn`/`forge.tensor` (`Tensor.tanh()`, `nn.RNNCell`) --
see `docs/development/m50-char-rnn.md` for why those were the one genuine
blocker found. This script adds no framework logic of its own beyond
example wiring: the corpus (`corpus.py`), vocabulary/sequence construction
(`dataset.py`), and model (`model.py`).

## Why a hand-written training loop, not `Trainer.fit()`

`forge.training.Trainer` orchestrates exactly one `forward(batch) ->
prediction -> loss` call per training step. A character-RNN's forward pass
is a *sequence* of `seq_len` recurrent steps sharing one hidden state, which
does not fit that one-call shape. Rather than extend `Trainer` with
sequence-training support no other Forge model needs (a speculative
framework addition this milestone's brief explicitly forbids), this script
uses the same hand-written `zero_grad -> forward loop -> backward -> step`
loop Milestones 1-5 always supported and MNIST/`trainer_demo.py` later
wrapped in `Trainer` only once every model in the repository shared its
per-step shape. See `docs/development/m50-char-rnn.md`'s Step 3/4 for the
full blocker-vs-workaround reasoning.

## One-hot encoding

Forge's Tensor has no embedding/gather primitive (M49 confirmed this is a
deliberate, still-uncontested scope boundary). At this example's vocabulary
size (a few dozen characters), a plain one-hot float vector built directly
in NumPy from each batch's raw character-index column is exactly as
correct and cheap as a dedicated embedding lookup would be -- `_one_hot`
below is the entire "workaround."

## Usage

```bash
python -m examples.char_rnn.train --epochs 30 --device cpu
python -m examples.char_rnn.train --epochs 30 --device cuda
```
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

import forge
import forge.cuda as cuda
from forge import Tensor, no_grad
from forge.data import DataLoader
from forge.nn import CrossEntropyLoss
from forge.optim import Adam
from forge.serialization import load_model, save_model
from forge.training import generate_sequence

try:
    from .corpus import TEXT
    from .dataset import Vocab, build_dataset
    from .model import CharRNN, build_model
except ImportError:  # running as a plain script (`python examples/char_rnn/train.py`)
    from corpus import TEXT
    from dataset import Vocab, build_dataset
    from model import CharRNN, build_model


def _one_hot(ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """`(batch,)` integer character indices -> `(batch, vocab_size)` float32 one-hot rows."""
    out = np.zeros((ids.shape[0], vocab_size), dtype=np.float32)
    out[np.arange(ids.shape[0]), ids] = 1.0
    return out


def train_one_epoch(
    model: CharRNN,
    loader: DataLoader,
    optimizer: Adam,
    loss_fn: CrossEntropyLoss,
    device: str,
    compute_stream: "cuda.Stream | None" = None,
) -> float:
    """One pass over `loader`; returns the mean per-character loss for the epoch.

    `compute_stream` (Milestone 55, `device='cuda'` only): running the whole
    batch loop on an explicit, non-default `forge.cuda.Stream` skips the
    default stream's per-kernel-launch `cudaDeviceSynchronize()` (see
    `docs/architecture/cuda-streams.md`) -- profiling this exact loop found
    that blocking sync call alone consumes over half of one epoch's
    wall-clock time on the reference 940MX (`docs/development/
    m55-post-m54-assessment.md`), since a 20-timestep unrolled sequence
    graph issues dozens of small kernel launches per batch. `Trainer`'s own
    `prefetch=True` mode already does exactly this for its batch loop
    (`_compute_stream_scope()`) -- this mirrors that, since `Trainer` itself
    does not support multi-timestep sequence training (see the module
    docstring above). `float(mean_loss.to('cpu').numpy())` below still
    correctly synchronizes just that one transfer before reading it (see
    `Tensor._data`'s pending-transfer contract) even though every compute
    kernel before it ran without a blocking host sync.
    """
    total_loss = 0.0
    total_chars = 0
    stream_scope = cuda.stream(compute_stream) if compute_stream is not None else nullcontext()
    with stream_scope:
        for input_ids, target_ids in loader:
            input_np = input_ids.numpy()
            target_np = target_ids.numpy()
            batch_size, seq_len = input_np.shape

            optimizer.zero_grad()
            h = model.init_hidden(batch_size, device=device)
            step_loss = None
            for t in range(seq_len):
                x_t = Tensor(_one_hot(input_np[:, t], model.vocab_size), device=device)
                logits_t, h = model.step(x_t, h)
                loss_t = loss_fn(logits_t, target_np[:, t])
                step_loss = loss_t if step_loss is None else step_loss + loss_t

            mean_loss = step_loss * Tensor(1.0 / seq_len, dtype=logits_t.dtype, device=device)
            mean_loss.backward()
            optimizer.step()

            total_loss += float(mean_loss.to("cpu").numpy()) * batch_size * seq_len
            total_chars += batch_size * seq_len

    return total_loss / total_chars


def generate(model: CharRNN, vocab: Vocab, seed_text: str, length: int, device: str, rng: np.random.Generator) -> str:
    """Sample `length` characters, seeded with `seed_text`, from the trained model.

    A thin wrapper over `forge.training.generate_sequence()` (Milestone 75):
    this example's own `encode`/`decode` are just one-hot encoding and
    single-character decoding, the only parts of the sampling loop that
    aren't shared with `examples/word_rnn/train.py::generate`.
    """
    generated = generate_sequence(
        model,
        seed=list(seed_text),
        encode=lambda ch: Tensor(_one_hot(vocab.encode(ch), model.vocab_size), device=device),
        decode=lambda idx: vocab.decode([idx]),
        length=length,
        device=device,
        rng=rng,
    )
    return "".join(generated)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-2, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="examples/char_rnn/artifacts")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "char_rnn_model.forge"

    forge.random.seed(args.seed)
    data_rng = np.random.default_rng(args.seed)
    sample_rng = np.random.default_rng(args.seed + 1)

    dataset, vocab = build_dataset(TEXT, seq_len=args.seq_len)
    print(f"corpus: {len(TEXT)} chars, vocab: {vocab.size} unique chars, {len(dataset)} training sequences")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=data_rng)
    model = build_model(vocab.size, hidden_size=args.hidden_size, device=args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = CrossEntropyLoss()
    compute_stream = cuda.Stream() if args.device == "cuda" else None

    start = time.perf_counter()
    losses = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, loss_fn, args.device, compute_stream)
        losses.append(loss)
        print(f"epoch {epoch:3d}/{args.epochs}: mean char loss = {loss:.4f}")
    duration = time.perf_counter() - start
    print(f"\nTrained {args.epochs} epoch(s) on '{args.device}' in {duration:.1f}s.")
    print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    print("\nSample generation (seed 'a tensor'):")
    print(generate(model, vocab, seed_text="a tensor", length=200, device=args.device, rng=sample_rng))

    save_model(model, str(model_path))
    print(f"\nSaved model -> {model_path}")

    # Model-persistence round trip: load fresh and confirm predictions match,
    # the same property `examples/mnist/train.py`/`examples/persistence_demo.py`
    # already demonstrate for their own architectures.
    with no_grad():
        h = model.init_hidden(1, device=args.device)
        x0 = Tensor(_one_hot(np.array([0]), vocab.size), device=args.device)
        pre_save_logits, _ = model.step(x0, h)
        pre_save_logits = pre_save_logits.to("cpu").numpy()

    reloaded = load_model(str(model_path), device=args.device)
    with no_grad():
        h = reloaded.init_hidden(1, device=args.device)
        post_load_logits, _ = reloaded.step(x0, h)
        post_load_logits = post_load_logits.to("cpu").numpy()

    assert np.allclose(pre_save_logits, post_load_logits, atol=1e-5), "reloaded model prediction diverged"
    print("Verified: reloaded model reproduces the pre-save prediction.")

    print("\nInspect the generated artifact with the M19 CLI:")
    print(f"  python -m forge model inspect {model_path}")


if __name__ == "__main__":
    main()
