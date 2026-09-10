# M75 — `forge.training.generate_sequence()`

## 1. Investigation

The brief forbade another readiness assessment and required identifying the
smallest concrete capability that materially improves Forge's real developer
workflow, evidenced by direct inspection of `forge/data/`, `forge/training/`,
`forge/serialization/`, `forge/nn/`, the CLI, every example, and the
project's own docs — not assumed from an absent feature.

### Current workflow examined

- **Every `examples/*/train.py`** (10 real examples): grepped for
  `evaluate`/`predict`/`start_training_session`/`if args.resume` to find
  what's already shared vs. still duplicated. `start_training_session()`
  (M73) is now used by all 7 checkpoint-capable examples — M73's own
  "retrofit opportunistically" recommendation is already done, no residual
  gap there. `predict()` (M68) is used by all 6 single-forward-pass
  examples for their persistence round-trip check.
- **`forge/cli/model.py`**: `inspect`/`convert`/`predict` (M72) are each
  thin wrappers over existing public APIs; `predict` is deliberately scoped
  to single-image classification because that's the only workload with a
  saved-artifact-describable single-file input convention.
- **`examples/char_rnn/`, `examples/word_rnn/`, `examples/long_range_recall/`**:
  the three examples with a hand-written multi-timestep training loop
  (`Trainer.fit()` doesn't fit a multi-step recurrence — a deliberate,
  documented M50/M54/M67 decision, not a gap). All three share an informal
  `model.step(x, h) -> (logits, h)` / `model.init_hidden(batch_size, device)
  -> h` protocol (`word_rnn/model.py`'s own docstring: "structurally
  identical to `examples/char_rnn/model.py`'s `CharRNN`").
- **`examples/char_rnn/train.py::generate()` and `examples/word_rnn/
  train.py::generate()`**: read both in full. Byte-for-byte identical
  control flow — prime hidden state over a seed sequence, then loop
  `length` times sampling a next token from a host-side softmax over
  `model.step()`'s logits via `rng.choice`, decode, append, feed back in.
  The *only* difference is how one token becomes a model input (`char_rnn`:
  one-hot via a local `_one_hot()` helper; `word_rnn`: a raw int64 token id,
  since `Embedding` lives inside the model) and how a sampled index becomes
  a token again (`char_rnn`: `vocab.decode([idx])` → single char; `word_rnn`:
  `vocab.decode([idx])[0]` → single word). `word_rnn/train.py::generate()`'s
  own docstring states this explicitly: *"the same pattern
  `examples/char_rnn/train.py::generate` already uses."*
- **`docs/product/vision.md`/`use-cases.md`**: the core workflow ends at
  "new input → useful prediction." `predict()` (M68) covers this for every
  single-forward-pass model. No equivalent existed for a *generative*
  model's actual inference need (autoregressive sampling), despite two RNN
  language-model examples already being an accepted, precedented part of
  Forge's model portfolio (M50, M54, M67).

### Other candidate areas considered and rejected

- **Device-selection boilerplate** (`--device`, `default="cpu",
  choices=["cpu","cuda"]` repeated in all 10 examples' `argparse` setup):
  real duplication, but trivial (one line) and cosmetic — not a workflow
  block, doesn't meet the "removes a meaningful piece of boilerplate" bar
  the way a 25-line duplicated sampling loop does.
- **A generic evaluation framework** (candidate B in the brief): every
  example's "trivial baseline" (`regression`'s predict-the-mean,
  `autoencoder`'s predict-the-mean-image, `segmentation`'s
  predict-all-background) is genuinely task-specific arithmetic — forcing
  these into one shared abstraction would be exactly the kind of
  generalization-with-no-real-shared-need the brief explicitly warns
  against ("Do not add a generic evaluation framework merely for
  symmetry").
- **CLI `forge model evaluate`**: rejected for the same reason M73 rejected
  a `forge model train` command — evaluation requires a live `Dataset`
  object, and Forge has no config/YAML system to express one from the
  command line (a deliberate non-goal, `docs/product/scope.md`).
- **A formal `SequenceModel` base class / `typing.Protocol`** for the
  `step()`/`init_hidden()` shape: considered as part of designing
  `generate_sequence()`, but rejected as *unnecessary* machinery — two
  consumers already share the shape by convention (no enforcement needed),
  and Forge's own precedent (`docs/architecture/modules.md`, `Module`
  itself) is to formalize a shape only when a concrete abstraction (like
  `Trainer` needing `Loss`/`Optimizer`) actually depends on it. Duck-typing
  plus a documented docstring contract is the smaller, sufficient scope.
- **A second worked example** (candidate F): rejected — the gap here was
  found by inspection, with two already-real consumers, satisfying the
  brief's own bar for skipping straight to implementation.

### Why this direction won

It satisfies multiple of the brief's four qualifying criteria at once:
(1) removes a real, measured piece of duplicated boilerplate (an entire
25-line function, byte-for-byte identical control flow, in two files);
(3) generalizes an existing capability (`predict()`'s inference-path
pattern: eval-mode + `no_grad()` + device resolution + mode restoration)
to a second real, already-demonstrated need (autoregressive generation)
with two already-existing consumers, not a hypothetical one; and it
advances vision.md's "new input → useful prediction" pipeline for the one
model family (`RNNCell`/`LSTMCell`-based sequence models) `predict()`
itself explicitly does not cover.

## 2. Exact implementation

Added `forge.training.generate_sequence()` to `forge/training/inference.py`
(same file as `predict()`, since both share `_resolve_device()` and the
eval-mode/`no_grad()`/mode-restoration pattern):

```python
def generate_sequence(
    model: Module,
    seed: Sequence[Any],
    encode: Callable[[Any], Tensor],
    decode: Callable[[int], Any],
    length: int,
    device: "str | Device | None" = None,
    rng: "np.random.Generator | None" = None,
) -> list:
    ...
```

- Validates `model` is a `forge.nn.Module` (`TrainerError`), `seed`
  non-empty and `length >= 0` (`DataError`) — mirroring `predict()`'s own
  validation style.
- Resolves the compute device via the same private `_resolve_device()`
  helper `predict()` already uses.
- `rng` defaults to `forge.random.default_generator()` when omitted —
  `random_split()`'s own established default-argument convention.
- Puts `model` in eval mode, runs entirely inside `forge.no_grad()`,
  restores the model's prior training mode in a `finally` block — identical
  discipline to `predict()`/`Trainer.evaluate()`.
- Primes `model.init_hidden(1, device=...)` over `seed[:-1]` (discarding
  logits), then loops `length` times: `encode(current).to(target_device)`
  → `model.step(x, state)` → host-side softmax over the returned logits →
  `rng.choice(...)` → `decode(sampled_index)` → append.
- Returns `list(seed) + generated_tokens` (the seed is part of the output),
  matching both examples' prior behavior exactly.

Re-exported as `forge.training.generate_sequence` and top-level
`forge.generate_sequence`, mirroring `predict()`/`interpret_classification()`'s
own existing re-export precedent (`forge/training/__init__.py`,
`forge/__init__.py`).

`examples/char_rnn/train.py::generate()` and `examples/word_rnn/
train.py::generate()` are now thin wrappers:

```python
# char_rnn
def generate(model, vocab, seed_text, length, device, rng):
    generated = generate_sequence(
        model, seed=list(seed_text),
        encode=lambda ch: Tensor(_one_hot(vocab.encode(ch), model.vocab_size), device=device),
        decode=lambda idx: vocab.decode([idx]),
        length=length, device=device, rng=rng,
    )
    return "".join(generated)

# word_rnn
def generate(model, vocab, seed_words, length, device, rng):
    return generate_sequence(
        model, seed=list(seed_words),
        encode=lambda w: Tensor(vocab.encode([w]), device=device),
        decode=lambda idx: vocab.decode([idx])[0],
        length=length, device=device, rng=rng,
    )
```

Both keep their original public signature — every existing call site
(`train.py`'s own "Sample generation" call, both integration test suites)
needed zero changes.

## 3. Files changed

- `forge/training/inference.py` — added `generate_sequence()`.
- `forge/training/__init__.py`, `forge/__init__.py` — re-exports + docstring
  updates.
- `examples/char_rnn/train.py`, `examples/word_rnn/train.py` — `generate()`
  retrofitted to a thin wrapper; import added.
- `examples/char_rnn/README.md`, `examples/word_rnn/README.md` — Determinism
  section updated to note the shared implementation.
- `docs/architecture/training-engine.md` — new **Sequence generation**
  section documenting the design and its rejected alternatives.
- `docs/development/progress.md` — M75 entry.
- `tests/test_inference.py`, `tests/test_inference_cuda.py` — new tests (see
  below).
- `docs/development/m75-sequence-generation.md` — this report.

No `forge/data/`, `forge/nn/`, `forge/serialization/`, `forge/backend/`, or
CLI file was touched — the capability composes entirely from existing
primitives (`Module`, `no_grad`, `Tensor.to()`, `forge.random`), per the
brief's "prefer composition over introducing new internal machinery"
requirement.

## 4. Architecture impact

None to the Tensor/autograd/backend core. `forge/training/` gains one new
free function, following `predict()`'s own established shape (free
function, not a `Trainer`/`Module` method — same reasoning: no
`Loss`/`Optimizer` to own, and attaching generation directly to `Module`
would blur the compute/orchestration boundary `Trainer` itself exists to
preserve). No new base class, `Protocol`, or persistence-format change.

## 5. Public API changes

Additive only:

- `forge.training.generate_sequence(model, seed, encode, decode, length,
  device=None, rng=None) -> list` (new).
- `forge.generate_sequence` (new top-level re-export, same object).

No existing public signature changed. `examples/char_rnn/train.py::generate()`
/`examples/word_rnn/train.py::generate()` keep their exact prior signatures
and return types.

## 6. Test coverage

13 new tests, all passing:

- `tests/test_inference.py` (11 new): re-export identity consistency;
  output length includes the seed prefix; every generated token is a valid
  class index; determinism given an identical explicit `rng`; determinism
  via `forge.random`'s default generator when `rng` is omitted; training-mode
  restoration (both starting `train()` and starting `eval()`); rejection of
  a non-`Module` model, an empty seed, and a negative length (each the
  correct exception type); `length=0` returns exactly the seed; and a
  byte-for-byte match against an independently hand-written reference
  sampling loop (proves the extraction changed no behavior, not just shape).
- `tests/test_inference_cuda.py` (2 new, hardware-verified on the 940MX):
  a CPU-built `encode()` Tensor is moved to a CUDA model automatically (the
  same automatic-transfer contract `predict()` already guarantees); CPU and
  CUDA runs agree exactly given identical weights (via `Module.to()`'s
  in-place move, isolating device as the only variable) and identical `rng`
  state.

All new tests use a small local `TinyStepModel` (`RNNCell` → `Linear`, the
same `step()`/`init_hidden()` shape as `CharRNN`) rather than importing from
`examples/`, keeping `tests/` self-contained.

## 7. End-to-end verification

- `tests/test_char_rnn_example_integration.py`,
  `tests/test_word_rnn_example_integration.py`,
  `tests/test_char_rnn_example_cuda_integration.py`,
  `tests/test_word_rnn_example_cuda_integration.py` — all pass **unmodified**
  against the refactor (31 + 8 = 39 tests), proving behavioral equivalence
  between the old inlined loop and the new shared function.
- Ran both real example scripts end-to-end on CPU:
  `python -m examples.char_rnn.train --epochs 3 --device cpu` and
  `python -m examples.word_rnn.train --epochs 2 --device cpu` — both
  trained, sampled text via the retrofitted `generate()`, saved, reloaded,
  and verified the persistence round-trip, exactly as before.
- Full suite: **2,234 collected, 2,233 passed, 1 failed** (2,221 + 13 new).
  The one failure, `test_dataloader_prefetch.py::
  test_repeated_epochs_do_not_grow_cuda_or_pinned_memory`, is the same
  pre-existing allocator-measurement flake documented since M63 —
  reproduced passing cleanly in isolation in this session, unrelated to
  M75.

## 8. CPU/CUDA results

- `tests/test_inference_cuda.py` (5 tests total, including the 2 new ones)
  and `tests/test_char_rnn_example_cuda_integration.py` /
  `tests/test_word_rnn_example_cuda_integration.py` (8 tests) all pass on
  the real reference 940MX.
- CPU/CUDA parity for `generate_sequence()` itself is directly verified
  (not just inferred from existing per-op parity): identical weights via
  `Module.to("cuda")` plus identical `rng` state produce byte-for-byte
  identical generated token sequences on CPU vs. CUDA.

## 9. Persistence/compatibility impact

None. No `.forge`/checkpoint format change, no `save_model`/`load_model`
signature change. `generate_sequence()` operates purely post-load, on an
already-reconstructed `Module`.

## 10. Limitations

- Batch size is fixed at 1 (`model.init_hidden(1, ...)`) — matches both
  real consumers' actual usage (single-sequence sampling); a batched
  variant would need evidence of a real multi-sequence-at-once generation
  need, which doesn't exist yet.
- The `step()`/`init_hidden()` protocol remains informal/undocumented as a
  type (duck-typed) — a model that doesn't implement it fails with a plain
  `AttributeError` from inside the loop rather than an early, named
  validation error. Accepted as consistent with `predict()`'s own minimal
  validation (which only checks `isinstance(model, Module)`, not a callable
  `forward`), not a new gap this milestone introduced.
- No greedy/temperature/top-k decoding options — both real consumers always
  sampled from the full softmax distribution with no temperature parameter;
  adding one now would be speculative (candidate 4's "no abstraction based
  on one weak consumer" applies to unrequested decoding strategies too).

## 11. Rejected/deferred work

- A formal `SequenceModel`/`Protocol` type for `step()`/`init_hidden()` —
  deferred until a third consumer's shape actually diverges enough to need
  compile-time enforcement.
- CLI exposure (e.g. `forge model generate`) — no saved-artifact convention
  exists yet for a `Vocab`/encode-decode pair the way image preprocessing
  exists for `forge model predict`; would need its own evidence.
- Device-selection (`--device`) boilerplate deduplication, a generic
  evaluation-baseline framework, and a `forge model evaluate` CLI command —
  all investigated and explicitly rejected this milestone (see Section 1),
  not silently skipped.

## 12. Relationship to the long-term Forge vision

`docs/product/vision.md`'s core workflow ends "... → persistence →
inference." `predict()` (M68) made that concrete for every single-forward
model family Forge supports. `generate_sequence()` makes the same
"trained model → useful output on new input" step concrete for Forge's
generative/sequence-modeling family (`char_rnn`, `word_rnn`, and any future
`RNNCell`/`LSTMCell`-based model) — closing the same kind of gap `predict()`
closed, for the one model shape `predict()` structurally cannot serve.

## 13. Practical developer impact

A developer building a new stepwise recurrent model (a third RNN/LSTM-based
example, or their own) no longer needs to hand-write the "prime, then
sample-decode-feed-back" loop from scratch — they write `encode`/`decode`
for their own token representation and call
`forge.generate_sequence(model, seed, encode, decode, length)`. The two
existing consumers already prove this is a real reduction in code, not a
speculative one: both examples' `generate()` functions dropped from
~25 lines of loop logic to ~10 lines of pure encode/decode wiring.

## 14. Follow-up triggers

- If a third stepwise sequence model appears (e.g. a future GRU cell, or an
  LSTM-based language model beyond `long_range_recall`'s classification
  task) and its `step()`/`init_hidden()` shape genuinely diverges from the
  current two consumers, revisit whether a formal `Protocol` is now
  justified.
- If any consumer needs temperature scaling, top-k/top-p sampling, or
  batched (>1) generation, add it then, evidenced by that consumer's actual
  requirement — not speculatively now.
- If a future example saves a `Vocab`-like encode/decode configuration
  alongside its model file (mirroring `preprocessing=`/`classes=`), a
  `forge model generate` CLI command would become as justified as `forge
  model predict` is today.

## Suggested Commit Message

```
feat: add forge.training.generate_sequence(), extracting the identical char_rnn/word_rnn autoregressive sampling loop
```
