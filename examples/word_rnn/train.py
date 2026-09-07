"""Forge Milestone 54: a small word-level RNN language-model example.

```text
WordDataset -> DataLoader -> WordRNN (Embedding -> RNNCell -> Linear) -> CrossEntropyLoss -> Adam
```

Every step uses only public Forge APIs plus the one addition this milestone
made (`nn.Embedding` / `Tensor.embedding_lookup()`) -- see
`docs/development/m54-product-direction.md` for the evidence that motivated
it. This script is structurally identical to `examples/char_rnn/train.py`
(same hand-written sequence-training loop, same reasoning for not using
`Trainer.fit()` -- see that module's docstring) with one difference: no
`_one_hot()` helper. `WordRNN.step()` takes raw integer token-id batches
directly; the embedding lookup happens inside the model.

## Usage

```bash
python -m examples.word_rnn.train --epochs 10 --device cpu
python -m examples.word_rnn.train --epochs 10 --device cuda
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

try:
    from .corpus import TEXT, VOCAB_WORDS
    from .dataset import Vocab, build_dataset
    from .model import WordRNN, build_model
except ImportError:  # running as a plain script (`python examples/word_rnn/train.py`)
    from corpus import TEXT, VOCAB_WORDS
    from dataset import Vocab, build_dataset
    from model import WordRNN, build_model


def train_one_epoch(
    model: WordRNN,
    loader: DataLoader,
    optimizer: Adam,
    loss_fn: CrossEntropyLoss,
    device: str,
    compute_stream: "cuda.Stream | None" = None,
) -> float:
    """One pass over `loader`; returns the mean per-token loss for the epoch.

    `compute_stream` (Milestone 55, `device='cuda'` only): see
    `examples/char_rnn/train.py::train_one_epoch`'s identical parameter for
    the full rationale -- running this loop's ~20-timestep unrolled sequence
    graph on an explicit, non-default stream skips the default stream's
    per-kernel-launch blocking synchronize, which profiling found consumes
    over half of one epoch's wall-clock time on the reference 940MX (see
    `docs/development/m55-post-m54-assessment.md`).
    """
    total_loss = 0.0
    total_tokens = 0
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
                x_t = Tensor(input_np[:, t].astype(np.int64), device=device)
                logits_t, h = model.step(x_t, h)
                loss_t = loss_fn(logits_t, target_np[:, t])
                step_loss = loss_t if step_loss is None else step_loss + loss_t

            mean_loss = step_loss * Tensor(1.0 / seq_len, dtype=logits_t.dtype, device=device)
            mean_loss.backward()
            optimizer.step()

            total_loss += float(mean_loss.to("cpu").numpy()) * batch_size * seq_len
            total_tokens += batch_size * seq_len

    return total_loss / total_tokens


def generate(model: WordRNN, vocab: Vocab, seed_words: "list[str]", length: int, device: str, rng: np.random.Generator) -> "list[str]":
    """Sample `length` words, seeded with `seed_words`, from the trained model.

    Inference-only: runs under `forge.no_grad()`, softmax/sampling happen in
    plain NumPy on the materialized logits -- the same pattern
    `examples/char_rnn/train.py::generate` already uses.
    """
    with no_grad():
        h = model.init_hidden(1, device=device)
        generated = list(seed_words)
        for w in seed_words[:-1]:
            x_t = Tensor(vocab.encode([w]), device=device)
            _, h = model.step(x_t, h)

        current = seed_words[-1]
        for _ in range(length):
            x_t = Tensor(vocab.encode([current]), device=device)
            logits, h = model.step(x_t, h)
            probs = np.exp(logits.to("cpu").numpy()[0])
            probs = probs / probs.sum()
            next_idx = rng.choice(model.vocab_size, p=probs)
            current = vocab.decode([next_idx])[0]
            generated.append(current)
    return generated


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-2, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="examples/word_rnn/artifacts")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "word_rnn_model.forge"

    forge.random.seed(args.seed)
    data_rng = np.random.default_rng(args.seed)
    sample_rng = np.random.default_rng(args.seed + 1)

    dataset, vocab = build_dataset(TEXT, VOCAB_WORDS, seq_len=args.seq_len)
    print(f"corpus: {len(TEXT.split())} tokens, vocab: {vocab.size} unique words, {len(dataset)} training sequences")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=data_rng)
    model = build_model(
        vocab.size, embedding_dim=args.embedding_dim, hidden_size=args.hidden_size, device=args.device
    )
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = CrossEntropyLoss()
    compute_stream = cuda.Stream() if args.device == "cuda" else None

    start = time.perf_counter()
    losses = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, loss_fn, args.device, compute_stream)
        losses.append(loss)
        print(f"epoch {epoch:3d}/{args.epochs}: mean word loss = {loss:.4f}")
    duration = time.perf_counter() - start
    print(f"\nTrained {args.epochs} epoch(s) on '{args.device}' in {duration:.1f}s.")
    print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    seed_words = vocab.words[:2]
    print(f"\nSample generation (seed {seed_words}):")
    print(generate(model, vocab, seed_words=seed_words, length=20, device=args.device, rng=sample_rng))

    save_model(model, str(model_path))
    print(f"\nSaved model -> {model_path}")

    # Model-persistence round trip: load fresh and confirm predictions match,
    # the same property `examples/char_rnn/train.py`/`examples/mnist/train.py`
    # already demonstrate for their own architectures.
    with no_grad():
        h = model.init_hidden(1, device=args.device)
        x0 = Tensor(vocab.encode([vocab.words[0]]), device=args.device)
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
