# Milestone 90 — Sequence-Artifact Inference

## Objective

M86-M89 built and validated a complete train -> portable `.forge` artifact ->
inspect -> predict workflow, but only for three task families: classification,
regression, segmentation. M90's brief asked for a genuinely new, realistic
workload -- materially different from those three -- attempted with existing
Forge capabilities first, so that the actual next limitation (not a
predetermined feature) drives what gets built.

## Selected workload

**Sequence / language-model generation**, using the existing
`examples/char_rnn` character-level RNN: train it, save it as a portable
`.forge` artifact, then generate text from that artifact alone, in a fresh
process, the same way `examples/segmentation`/`examples/regression`/
`examples/image_folder_classification` already support fresh-process
`predict_model()`/CLI inference.

**Why this workload.** It is the concrete "Sequence / time-series workload"
the milestone brief suggested, it is materially different from
classification/regression/segmentation (autoregressive generation from a
seed, not a single forward pass), Forge already has every primitive it
needs (`RNNCell`, `Embedding`-free one-hot encoding, `generate_sequence()`,
`save_model()`/`load_model()`), and three real examples
(`char_rnn`/`word_rnn`/`long_range_recall`) already train and locally sample
from stepwise models -- so the workload was not synthetic, and any gap found
would be a real, already-latent one, not manufactured for this milestone.

**External developer goal.** Train a char-level language model, ship the
`.forge` file, and let a teammate (or a deployment script, or a second
process days later) generate text from it without reading Forge's source or
re-deriving `char_rnn/train.py`'s one-hot-encode/`step()`/decode loop by
hand -- the same "artifact is self-sufficient" guarantee M82-M89 already
give the other three task families.

**Why existing examples don't already cover it.** `char_rnn/train.py` calls
`forge.save_model(model, path)` (no `preprocessing=`/`classes=`/`task=`) and
samples inline, in the same process, using its own in-memory `Vocab` and
model object. No `infer.py` exists for any of the three sequence examples --
generation has only ever happened where the model was trained.

## Baseline: attempting the workload with existing Forge APIs

`forge.save_model()`/`forge.load_model()` already round-trip a `CharRNN`
correctly (verified by `char_rnn/train.py`'s own pre-M90 check and
`tests/test_char_rnn_example_integration.py`). Everything past that point
failed or didn't exist:

```python
>>> import examples.char_rnn.model  # register CharRNN
>>> import forge
>>> info = forge.inspect_model("char_rnn_model.forge")
>>> info.task, info.model.module_types
(None, ('CharRNN', 'RNNCell', 'Linear', 'Linear', 'Linear'))
>>> forge.predict_model("char_rnn_model.forge", [1.0])
...
forge.exceptions.ModuleError: CharRNN does not implement forward().
```

This is not merely "unsupported" -- it is a **silent misidentification**:
`_legacy_infer_workflow()`'s `"Linear" in module_types -> "regression"` rule
(correct for `examples/regression`'s actual `Linear`-only model) also
matches `CharRNN` (`RNNCell` is itself built from two `Linear` layers, plus
`CharRNN`'s own output `Linear`), so `predict_model()` confidently picks the
wrong workflow, then fails deep inside `predict()`'s `model(x)` call with an
error that names an implementation detail (`forward()`), not the actual
problem (this artifact needs a different kind of prediction entirely).
Reproduced against a real, freshly retrained artifact (2-epoch char_rnn run)
before writing any framework code, per the milestone's required operating
model.

**What already worked:** dataset construction, model training (CPU and
CUDA), `save_model()`/`load_model()` round-tripping, `forge model inspect`
(metadata-only, no reconstruction, so it never hit the misidentification).

**What did not:** `forge.predict_model()` (misidentifies, then crashes
unclearly), `forge model predict` (would hit the same path once given a
task, but no task existed to give it), and no `infer.py` existed at all for
any of the three sequence examples -- there was nothing to reuse, only
`train.py`'s own inline, in-process sampling code.

## Findings, by severity

| Finding | Severity | Why |
|---|---|---|
| `predict_model()` silently misidentifies a stepwise sequence artifact as regression, then fails with an unrelated `ModuleError` | **Blocker** | An external developer holding a `.forge` sequence artifact has no working path to a prediction at all through the unified API -- and the failure mode actively misleads rather than saying "unsupported." |
| No `infer.py` for any sequence example | **Blocker** (for the "fresh-process, no in-memory state" goal) | Nothing to run outside the training process; generation is only ever demonstrated inline. |
| No task type/artifact-level function for sequence models | **Blocker** (root cause of the above two) | `predict_artifact()`/`predict_tensor_artifact()`/`predict_image_artifact()` all assume `predict()`'s `model(x)` calling convention, which a stepwise model never satisfies. |
| `forge model predict` has no sequence-input convention | **Significant friction** | Even once the above is fixed at the Python-API level, the CLI needs its own (necessarily narrower) tokenization convention to expose it. |
| `classes=` validation rejects whitespace-only vocabulary tokens | **Significant friction**, found *while implementing the fix* | A real char-RNN vocabulary over any corpus with word boundaries includes `" "`; reusing `classes` for a token vocabulary (the natural, minimal design) collided with a validation rule written only with classification labels in mind. |

## Problem selected: the first three (one connected root cause)

The first three findings share one root cause -- there is no artifact-level
prediction path for a stepwise-recurrence model -- and solving it end-to-end
(not just the `predict_model()` dispatch bug in isolation) is what section 9
of the milestone brief ("prefer end-to-end capability") calls for: fixing
only the misidentification would leave a developer with a correctly-raised
"unsupported" error and still no way to actually generate from the artifact.
The `classes=` whitespace issue was discovered as a direct consequence of
implementing the fix correctly (using `char_rnn`'s real vocabulary, which
contains a space) and was fixed in the same milestone since it blocked the
chosen workload from working at all, not deferred as a separate concern.

The CLI convention (fourth finding) was implemented as part of the same
change, per the brief's "extend `forge model predict`, don't add a new
command" instruction (Section 11) -- but deliberately scoped to char-level
tokenization only (see **Limitations**), since that is the one real
workload this milestone built and validated.

## Implementation

### 1. A fourth task type: `"sequence"`
`forge.serialization.model.TASK_TYPES` gained `"sequence"` alongside
`"classification"`/`"regression"`/`"segmentation"`. This is a genuinely
distinct *prediction problem* (autoregressive generation from a seed), not
"uses an RNN/LSTM layer" -- an RNN-based classifier would still be saved
with `task="classification"`. `task="sequence"` now *requires* `classes=`
(rather than forbidding it, like regression/segmentation) -- it is the
model's token vocabulary, index i -> `classes[i]`, reusing the exact
"index-to-string" concept `classes=` already establishes for classification
labels, rather than inventing a parallel `vocab=` parameter.

### 2. `classes=` validation relaxed for `task="sequence"`
`_validate_classes()` now takes the `task` being saved: for
`task="sequence"`, only the genuinely empty string `""` is rejected (a
whitespace token like `" "` is a legitimate vocabulary entry); classification
(and no-task) keeps the original, stricter "non-whitespace" rule.

### 3. `forge.predict_sequence_artifact(path, seed, length, *, device=None, rng=None)`
The fourth artifact-shape function, alongside `predict_artifact()`/
`predict_tensor_artifact()`/`predict_image_artifact()`
(`forge/training/inference.py`). Composes `load_model()` + `load_classes()`
+ `generate_sequence()` (Milestone 75, unchanged) with one-hot `encode`/
`decode` closures built from the saved vocabulary. Validates: non-empty
`seed`, every seed token present in the vocabulary, a saved vocabulary
exists, and the loaded model implements `init_hidden`/`step` -- each with a
specific `DataError`/`PersistenceError` rather than an opaque failure deep
inside the sampling loop.

### 4. `forge.predict_model()` extended with `length=`
A new, optional `length: int | None = None` keyword, used and required only
when the artifact's task is `"sequence"` (raises `DataError` if omitted for
that case); silently unused for the other three tasks, so every existing
call site is unaffected. Dispatches to `predict_sequence_artifact()`.

### 5. `forge model predict` extended for `task="sequence"`
`INPUT` is literal seed text (not a file -- the CLI's existing
`os.path.isfile(args.input)` check is skipped for this task), tokenized as
individual characters; a new `--length` flag (default 200). Prints the full
generated text (`--json` adds `{"task": "sequence", "seed": ..., "generated": ...}`).

### 6. `examples/char_rnn` retrofit
`train.py` now saves with `classes=vocab.chars, task="sequence"`. A new
`examples/char_rnn/infer.py` (mirroring `segmentation`/`image_folder_
classification`'s own `infer.py` shape) generates from the artifact alone
via `forge.predict_model()`, importing `examples.char_rnn.model` only for
its `register_module()` side effect (never `build_model()`/`Vocab`
directly) -- a `CharRNN` is a custom composite `Module`, like
`examples/resnet`/`examples/autoencoder`'s own custom classes, so it must be
registered in the loading process before `load_model()` can reconstruct it.
This is why the bare `forge model predict`/`forge model convert` CLI still
cannot load a `CharRNN` artifact directly (it never imports example model
modules) -- a pre-existing, general persistence constraint documented since
before this milestone, not something M90 introduced; `resnet`/`autoencoder`
have the identical limitation and the identical documented workaround
(a script that imports the model module first).

## Public API changes
- `forge.predict_sequence_artifact()` (new), exported from
  `forge`/`forge.training`.
- `forge.predict_model(..., length=None)` (new optional keyword; fully
  backward-compatible).
- `forge.save_model(..., task="sequence")` (new valid `task=` value).
- `forge.serialization.model.TASK_TYPES` now has 4 entries, not 3.

## CLI changes
- `forge model predict MODEL INPUT [--length N] ...` -- `--length` is new;
  `INPUT` for a sequence artifact is seed text, not a file path.

## Artifact changes
- No format-version bump (`FORMAT_VERSION` stays `2`) -- `"task"` already
  accepted an optional string; `"sequence"` is just a new valid value, and
  `classes` already existed as a metadata key. Fully backward- and
  forward-compatible, exactly like Milestones 71/72/87's own additions.

## Example changes
- `examples/char_rnn/train.py`: `save_model()` call gained `classes=`/
  `task=`; docstring and end-of-run printout updated.
- `examples/char_rnn/infer.py`: new file.
- `examples/char_rnn/README.md`: new "Portable artifact + fresh-process
  generation" section; corrected to not claim the bare CLI works for this
  custom-class artifact.
- `examples/word_rnn`/`examples/long_range_recall` were **not** retrofitted
  -- see **Limitations**.

## Files changed
- `forge/serialization/model.py` (`TASK_TYPES`, `_validate_task`,
  `_validate_classes`, `save_model()` docstring)
- `forge/training/inference.py` (`predict_sequence_artifact()`,
  `predict_model()`, module docstring)
- `forge/training/__init__.py`, `forge/__init__.py` (exports)
- `forge/cli/model.py` (`--length`, `cmd_predict()` sequence branch, module
  docstring)
- `examples/char_rnn/train.py`, `examples/char_rnn/infer.py` (new),
  `examples/char_rnn/README.md`
- `docs/architecture/persistence.md`, `docs/development/cli.md`
- `tests/test_sequence_artifact_prediction.py` (new),
  `tests/test_sequence_artifact_prediction_cuda.py` (new)

## Tests
22 new CPU tests (`test_sequence_artifact_prediction.py`): `task="sequence"`
validation (requires `classes`, whitespace-token relaxation), `predict_
sequence_artifact()` (generation shape/vocabulary, empty seed, unknown
token, missing vocabulary, non-stepwise model, determinism with an explicit
`rng`), `predict_model()` dispatch (`length=` required/passed through,
bit-for-bit match against the direct function, input-type validation),
existing-task regression check (`length=` is a no-op for regression), CLI
(`forge model predict` for sequence -- text/`--json` output, empty-seed
error, no-file-required, missing-task error), and a genuine `subprocess`
fresh-process run. 3 new CUDA tests (hardware-verified on the reference
GeForce 940MX): CUDA-saved artifact generates on CUDA by default,
`predict_model()` dispatch on CUDA, and CPU/CUDA output parity for the same
explicit `rng` seed.

## Validation

**Full suite:** all directly related existing test files re-run clean
alongside the new ones (168 passed: `test_sequence_artifact_prediction{,_cuda}.py`,
`test_unified_artifact_prediction{,_cuda}.py`, `test_cli_predict.py`,
`test_classification_metadata.py`, `test_model_inspection.py`,
`test_inference.py`), plus the pre-existing sequence-example integration
suites (`test_char_rnn_example_integration.py`,
`test_char_rnn_example_cuda_integration.py`,
`test_word_rnn_example_integration.py`,
`test_word_rnn_example_cuda_integration.py`,
`test_long_range_recall_example_integration.py` -- 37 passed, confirming
`word_rnn`/`long_range_recall` are genuinely untouched).

**End-to-end workload:** retrained `examples/char_rnn` fresh
(`python -m examples.char_rnn.train --epochs 15`), confirmed the artifact
now saves with `task="sequence"`/a 24-character vocabulary (including the
space that would have broken the pre-fix validation), then ran
`python -m examples.char_rnn.infer --model ... --seed "a tensor" --length 100`
and `python -m forge model inspect ...` as genuinely separate process
invocations against the artifact alone -- both produced correct, readable
generated text.

**Fresh-process validation:** `tests/test_sequence_artifact_prediction.py::
test_predict_sequence_artifact_works_from_a_genuinely_separate_process` runs
`forge.predict_model()` in a real `subprocess.run([sys.executable, "-c",
...])` child process.

**CPU:** all CPU tests above.

**CUDA:** hardware-verified on the reference GeForce 940MX -- a
`task="sequence"` artifact saved from a CUDA-resident model reloads and
generates on CUDA by default, `predict_model()`'s dispatch works
identically on CUDA, and CPU/CUDA outputs agree exactly for the same
explicit `rng` seed.

**Existing-workflow regression:** classification/regression/segmentation
were not touched at the logic level (only `predict_model()` gained an
unused-by-them optional keyword); `test_unified_artifact_prediction{,_cuda}.py`
and `test_cli_predict.py` re-run clean confirm this directly.

## Remaining limitations
- `forge model predict`'s sequence support tokenizes as individual
  characters only -- a word-level vocabulary (`examples/word_rnn`) needs
  `forge.predict_sequence_artifact()` called directly with a pre-tokenized
  seed list.
- `examples/word_rnn`/`examples/long_range_recall` were not retrofitted with
  `task="sequence"`/an `infer.py` -- only `examples/char_rnn`, the one
  workload actually built and validated end-to-end this milestone. The same
  pattern applies to them unchanged whenever needed (no framework work
  required).
- The bare `forge model predict`/`forge model convert` CLI cannot load a
  `CharRNN` (or any other custom-class) artifact directly -- a pre-existing,
  general persistence constraint (custom module types must be registered in
  the running process), not new to this milestone. `infer.py` is the real
  working fresh-process path.
- `predict_sequence_artifact()` covers exactly the token-vocabulary,
  one-hot-encoded shape `char_rnn`/`word_rnn` use -- a vocabulary-free
  numeric sequence model (e.g. raw-float time-series forecasting) is not
  covered and would need its own artifact-shape function.

## Explicitly deferred capabilities
Per the milestone brief's Section 20, none of the following were implemented
(no evidence any of them were a blocker for this workload): batch/directory
prediction, CSV/NDJSON input, model serving/HTTP, distributed/multi-GPU
training, additional recurrent layers, a generic tokenization/vocabulary
subsystem, or a generic "any custom class is CLI-loadable" mechanism.

## What an external developer can now do
Train a char-level language model with `examples/char_rnn/train.py`, get a
portable `.forge` artifact whose metadata declares it as a sequence
generator with its own token vocabulary, and generate text from that
artifact alone -- via `forge.predict_model(path, seed_tokens, length=N)`,
`forge.predict_sequence_artifact()` directly, `examples/char_rnn/infer.py`,
or `forge model predict model.forge "seed text" --length 200` -- in a
process that never trained the model and holds no in-memory `Vocab`, exactly
matching the guarantee already established for classification/regression/
segmentation artifacts.
