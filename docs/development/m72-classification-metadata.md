# M72 — Class-Label Metadata: From Tensor Output to Useful Prediction

## 1. Objective

M71 made a saved model self-describing about *input* preprocessing
(`save_model(..., preprocessing=...)`), closing the gap where a developer
had to remember `Resize`/pixel-scaling by hand in a fresh inference
process. That still left the workflow "framework-development-oriented": a
`.forge` file plus an image produces a raw prediction `Tensor`
(`[0.02, 0.94, 0.04]`), not something a normal developer can act on
(`"dog"`). M72's brief explicitly forbade another readiness assessment and
required inspecting the actual repository first, then implementing exactly
one justified capability that closes the smallest remaining blocker to
`train -> save -> load -> new input -> useful prediction`.

## 2. Investigation

Read/verified directly, not assumed from prior reports:

- `forge/serialization/model.py` (`save_model`/`load_model`/
  `load_preprocessing`): the `.forge` archive already carries architecture,
  weights, and (M71) preprocessing -- but nothing about output semantics.
- `docs/architecture/persistence.md`'s own **Preprocessing metadata**
  section (M71) had an explicit "Class-index-to-name vocabularies" note,
  still true: *"A model's predicted index -> label mapping is not 'how to
  prepare an input tensor' -- it is model-adjacent metadata a caller ...
  may still want to persist its own way (`examples/image_folder_
  classification/train.py` writes a plain `classes.json` sidecar for this,
  not a framework feature)."* This was the exact gap M71 deferred, named in
  writing, and M72's brief's own "likely direction" candidate.
- `examples/image_folder_classification/train.py`: confirmed the sidecar
  is real, not hypothetical -- `save_model(model, path, preprocessing=...)`
  followed by `classes_path.write_text(json.dumps(full_dataset.classes))`,
  a second, unrelated file a caller has to keep next to the model.
- `examples/image_folder_classification/infer.py`: confirmed the
  consequence -- a required `--classes path/to/classes.json` CLI argument,
  falling back to printing a bare integer index if omitted or the sidecar
  file was lost/renamed. This is precisely "the developer must remember
  and reconstruct a framework-specific detail" the milestone brief asks to
  eliminate.
- `forge/data/image_folder.py` (`ImageFolder`): already computes exactly
  the right vocabulary (`self.classes`, sorted, deterministic index
  assignment) -- nothing new needed to be invented; a caller just needed a
  place to persist `some_image_folder.classes` alongside the model.
- `forge/nn/loss.py` (`CrossEntropyLoss`): confirmed `logits` (a model's
  raw output) are treated as **unnormalized** scores -- `log_softmax` is
  computed internally. This matters because it justifies treating raw
  `predict()` output as logits and applying softmax for a "confidence"
  value later, rather than inventing an unjustified display heuristic.
- `forge/training/inference.py` (`predict()`, M68): stops deliberately at
  a raw `Tensor` -- confirmed it has, and should have, no opinion about
  what the output means; the gap is downstream of it, not inside it.
- `forge/cli/model.py`/`forge/cli/main.py` (M19 CLI): confirmed a mature,
  working `argparse`-subcommand structure (`model inspect`/`convert`,
  `checkpoint inspect`/`convert`, `benchmark`) with an established
  read-only-vs-mutating split and a single error-handling chokepoint --
  a `predict` subcommand fits this shape directly, with no new CLI
  framework needed.
- Searched every other example (`mnist`, `regression`, `resnet`,
  `segmentation`, `autoencoder`, `waveform_classification`, `char_rnn`,
  `word_rnn`) for a similar class-vocabulary pattern: none exists.
  `image_folder_classification` is currently the *only* Forge workflow with
  a file-based single-input inference story and a class vocabulary to map
  onto -- confirming the capability's real consumer without overclaiming
  breadth it doesn't have yet.

**Established directly, not assumed:**
- Forge cannot validate a `classes` list against a model's actual output
  width at *save* time without introspecting an arbitrary module tree's
  architecture -- something Forge deliberately never does elsewhere in the
  persistence layer (`save_model()` never inspects `Linear.out_features`
  generically either; only `get_config()`/`from_config()` do, per-type).
  The honest place for that check is at *interpretation* time, once a real
  output `Tensor` with an actual, already-produced width exists.
- A raw `Tensor.softmax()` primitive is not required: the only place a
  probability reading is needed is a non-differentiable, post-`no_grad()`,
  about-to-be-displayed value -- the same "host-side reduction, not a new
  core Tensor op" precedent `Metric`/`predict()`'s own batch concatenation
  already established (M49's capability-assessment discipline).

## 3. Decision

Implemented, in order of dependency:

1. **`save_model(model, path, classes=[...])`** -- a new optional kwarg,
   mirroring `preprocessing=` exactly in shape and compatibility policy.
2. **`load_classes(path)`** -- mirrors `load_preprocessing()` exactly.
3. **`forge.training.interpret_classification(output, classes)`** -- the
   "tensor output" -> "useful prediction" step, returning
   `ClassificationPrediction(label, index, confidence)`.
4. **`forge model predict MODEL --image IMAGE`** CLI command -- a thin
   wrapper composing 1-3 with `load_model()`/`load_preprocessing()`/
   `predict()`, scoped to image classification only.
5. **`examples/image_folder_classification/train.py`/`infer.py`** updated
   as the real consumer: the `classes.json` sidecar is gone entirely.

### Rejected alternatives
- **A generalized `metadata={...}` dict with an arbitrary schema.**
  Rejected per the brief's own explicit instruction and Forge's existing
  precedent (`preprocessing=` is its own dedicated kwarg, not folded into
  a generic blob) -- `classes=` is a second dedicated kwarg for a second
  concrete concern, not a step toward an open-ended metadata system.
- **A `"task"` field (e.g. `"image_classification"`).** No consumer needs
  to distinguish task types today; `classes` means the same thing
  regardless of what produced the scores. Adding an unused field would be
  exactly the "giant generalized metadata framework without evidence" the
  brief warned against.
- **Validating `classes` against model output width at save time.** Would
  require introspecting an arbitrary `Module` tree's architecture -- a
  capability Forge does not have and should not invent for this. Deferred
  to `interpret_classification()`, where the check is both possible and
  honest.
- **A generic `forge predict` CLI command spanning every Forge model
  family.** Every other example workload (tabular regression, autoencoder
  reconstruction, sequence generation) has a different natural input shape
  with no established single-file convention yet -- building one generic
  command would mean inventing that convention speculatively, which the
  brief explicitly warns against ("Do NOT force metadata if repository
  evidence points elsewhere" applied here to CLI scope, not just
  metadata). `forge model predict` is scoped to exactly the one workflow
  the artifact can already fully describe end to end.
- **Folding class labels into the `preprocessing` mechanism.** Explicitly
  rejected by `docs/architecture/persistence.md`'s own existing
  architecture: preprocessing is about preparing an *input*; class labels
  are about interpreting an *output* -- keeping them as separate sibling
  metadata entries (mirroring how buffers stay separate from parameters)
  preserves that boundary rather than blurring it for convenience.

## 4. Implementation

- `forge/serialization/model.py`: `_validate_classes()` (non-empty list,
  non-empty/unique string elements, `PersistenceError` before any write);
  `save_model(..., classes=None)` writes `metadata["classes"]`;
  `load_classes(path)` mirrors `load_preprocessing()`.
- `forge/serialization/__init__.py`, `forge/__init__.py`: export
  `load_classes` alongside the existing persistence API, at both levels
  `load_preprocessing` already is.
- `forge/training/inference.py`: `ClassificationPrediction` (frozen
  dataclass: `label: str`, `index: int`, `confidence: float`);
  `interpret_classification(output, classes)` -- shape validation
  (`TrainerError` for non-2-D output or `output.shape[1] != len(classes)`),
  numerically stable host-side softmax (max-subtracted before `exp`,
  mirroring `CrossEntropyLoss`'s own log-sum-exp trick), one result per
  batch row via `np.argmax`.
- `forge/training/__init__.py`, `forge/__init__.py`: export
  `interpret_classification`/`ClassificationPrediction` alongside `predict`.
- `forge/exceptions.py`: `TrainerError`'s docstring extended with the new
  failure modes.
- `forge/cli/model.py`: `cmd_inspect` now reports `has_preprocessing`/
  `classes` (text and `--json`); new `cmd_predict` (`forge model predict`)
  -- loads preprocessing (required, clear `CLIError` if absent, mirroring
  `infer.py`'s own existing policy), model, and classes (optional);
  decodes `--image` via `ImageFolder._load_image`; runs `predict()`; prints
  `Predicted class: dog` / `Confidence: 94.2%` if classes are present,
  else `Predicted class index: N` with an explanatory note.
- `examples/image_folder_classification/train.py`: `save_model(model,
  path, preprocessing=..., classes=full_dataset.classes)`; the
  `classes.json` sidecar file is deleted entirely, along with the `json`
  import that only existed to write it; the end-of-run demo now calls
  `load_classes()` + `interpret_classification()` instead of indexing
  `full_dataset.classes` with `np.argmax` by hand, and prints a confidence
  percentage.
- `examples/image_folder_classification/infer.py`: `--classes` argument
  removed; `run()` calls `load_classes()` and, if present, returns a
  `ClassificationPrediction` via `interpret_classification()` (falls back
  to a raw index only if the model was saved without `classes=`).
- `forge/cli/model.py`'s `cmd_convert` (`forge model convert`): fixed in
  passing to carry `preprocessing`/`classes` through a device conversion
  (`load_preprocessing()`/`load_classes()` on the input, passed to
  `save_model()` for the output). This was a real pre-existing gap since
  M71 introduced `preprocessing=` -- `model convert` silently dropped it,
  untested -- that would otherwise have silently defeated M72's own
  self-describing-artifact story the moment a converted copy was made;
  fixed here since it is a one-line, low-risk change directly in the file
  this milestone already had open, not a separate scope expansion.

## 5. Architecture Impact

None to `Tensor`, autograd, `nn`, `optim`, `data`, or the CUDA backend.
`Module` gained no new state -- `classes` is a sibling top-level metadata
entry alongside `"root"`/`"preprocessing"`, exactly like M71's own
`preprocessing` addition, preserving the "model state, preprocessing
config, and descriptive metadata stay conceptually distinct" boundary the
brief required. `interpret_classification()` is a new small addition to
`forge/training`, not a new subsystem -- it composes existing pieces
(`Tensor.to("cpu").numpy()`, plain NumPy) rather than adding a new Tensor
op or autograd-aware computation.

## 6. Persistence / Compatibility

No `FORMAT_VERSION` bump. `"classes"` is optional (`null`/absent both mean
"no vocabulary saved," handled identically by `load_classes()`) --
forward-compatible (an older Forge build never reads the key) and
backward-compatible (a pre-M72 file with no `"classes"` key at all loads
unchanged through both `load_model()` and `load_classes()`), verified
directly against a hand-edited archive with the key deleted, mirroring
`test_preprocessing_persistence.py::
test_backward_compatible_file_has_no_preprocessing_key`'s own method.

## 7. Real Consumer

`examples/image_folder_classification/` end to end:
`generate_dataset() -> ImageFolder -> Resize/Normalize -> Trainer -> CNN ->
save(model + preprocessing + classes) -> [fresh process] -> load_model() +
load_preprocessing() + load_classes() -> predict() ->
interpret_classification() -> "Prediction: dog / Confidence: 94.2%"`,
demonstrated both via `infer.py` (a Python script) and `forge model
predict` (the CLI), against the same artifact, with no `classes.json`
sidecar file anywhere in the pipeline.

## 8. Tests

- `tests/test_classification_metadata.py` (30 tests): `save_model(...,
  classes=...)`/`load_classes()` round trip; every documented invalid-
  `classes` case (empty list, non-list, non-string element, empty/
  whitespace-only string, duplicate labels) raising `PersistenceError`
  before any file is written; backward compatibility against a hand-edited
  legacy archive with the `"classes"` key deleted; malformed-metadata
  rejection on load; `interpret_classification()` correctness (including a
  manual-softmax numerical cross-check), the output-dimension-mismatch
  error, non-2-D-output rejection, and empty-classes rejection; the full
  `forge model predict`/`forge model inspect` CLI surface (label+confidence
  printed when classes are present, index-only fallback when absent, clear
  failure when preprocessing is absent, missing-file handling); and a real
  `ImageFolder` -> train -> save -> fresh-process `infer.py` + CLI
  end-to-end workflow; plus one test confirming `forge model convert`
  preserves preprocessing/classes across a device conversion (the fix
  described in **Implementation** above).
- `tests/test_classification_metadata_cuda_integration.py` (2 tests,
  hardware-verified on the reference 940MX): a CUDA-loaded model's output
  interpreted correctly via classes saved alongside it, and CPU-loaded vs.
  CUDA-loaded copies of the same model agreeing on the interpreted label
  and confidence.

## 9. Full-Suite Result

`tests/test_preprocessing_persistence.py`, `tests/test_cli.py`, `tests/
test_serialization.py`, `tests/test_image_folder_classification_
integration.py` (the suites most directly touched by this milestone's
changes): 116/116 passed, confirming no regression to preprocessing
persistence, the CLI, general serialization, or the existing image-folder
workflow. Full `pytest tests/`: **2,199 collected, 2,198 passed, 1 failed**
(2,167 pre-M72 + 30 CPU + 2 CUDA new here, on this CUDA-equipped machine;
2,197 collected on a CPU-only one). The one failure is
`test_dataloader_prefetch.py::test_repeated_epochs_do_not_grow_cuda_or_pinned_memory`
-- the same pre-existing allocator-measurement flake documented since M63
(`docs/development/progress.md`), reproduced passing cleanly in isolation
(`pytest tests/test_dataloader_prefetch.py::test_repeated_epochs_do_not_grow_cuda_or_pinned_memory`
-> 1 passed) -- not an M72 regression.

## 10. CPU/CUDA Verification

Both new test files pass; the CUDA file was run directly against the
reference GeForce 940MX (not merely collected/skipped) and confirms a
CUDA-loaded model's `predict()` output, interpreted via
`interpret_classification()`, agrees with a CPU-loaded copy of the same
model to `atol=1e-4` on both the predicted label and its confidence.

## 11. Limitations

- `classes` is a flat list of strings only -- no multi-label, no
  hierarchical taxonomy, no per-class extra data, no task-type field.
- Never validated against the model's actual output width at save time --
  only at first `interpret_classification()` call (a deliberate choice,
  not an oversight; see **Rejected alternatives**).
- `forge model predict` (and `interpret_classification()`'s own framing)
  assumes a classification model whose output is `(batch, num_classes)`
  raw scores -- it is not, and does not try to be, a generic
  "interpret any Forge model's output" mechanism.
- `save_checkpoint()`/`load_checkpoint()` do not carry `classes=` (the same
  scoping M71 already applied to `preprocessing=`) -- a resumed training
  run reconstructs its own class vocabulary from its own training script,
  exactly as before.

## 12. How This Advances Forge's Product Vision

```text
Dataset -> Data preprocessing -> Model definition -> Training -> Evaluation
    -> Model artifact -> Inference -> Prediction interpretation -> User-facing workflow
```
Before M72: everything through "Inference" was solid (M68/M71), but
"Prediction interpretation" did not exist as a framework concept at all --
every example that needed it (so far, only one) reinvented it by hand with
a sidecar file. M72 makes that layer real, narrow, and reusable: a second
classification example built tomorrow gets `classes=`/
`interpret_classification()` for free, with no new sidecar-file pattern to
invent. "User-facing workflow" gained its first genuinely thin,
non-Python-required surface (`forge model predict`) for the one workflow
where the artifact now fully supports it.

**Now genuinely usable end-to-end, no example-specific knowledge required:**
train an image classifier, save it, and get `"dog"`/confidence back from a
completely fresh process or from the command line -- no Python script,
sidecar file, or memorized preprocessing/class-order convention needed.

**Still future work, not started here:**
- No equivalent "prediction interpretation" story exists yet for
  regression (`examples/regression/`), reconstruction
  (`autoencoder`/`segmentation`), or generation (`char_rnn`/`word_rnn`) --
  each would need its own, differently-shaped interpretation step (e.g. a
  regression output is often already the "useful" value; a generative
  model's useful output is a decoded sequence, not a class index). None
  should be built speculatively; each needs its own real consumer first,
  the same discipline this milestone followed.
- No `forge train`/broader `forge predict` CLI surface beyond image
  classification.
- No model registry, serving infrastructure, or multi-model artifact
  management -- explicitly out of scope, per the brief.

## 13. Recommended Next Direction

Not chosen here, per the brief's own instruction not to force a next
milestone speculatively -- but the clearest evidence-backed candidate this
investigation surfaced is: pick one *non-classification* example (most
naturally `examples/regression/`, since it already has the most mature
reproducible-training workflow per M65) and determine, by direct
execution, whether it has an analogous "tensor output is not yet a useful
result" gap worth its own small interpretation mechanism -- rather than
assuming classification's `interpret_classification()` shape generalizes
without evidence.

## 14. Suggested Commit Message
```
feat: persist class-label metadata alongside a saved model, closing tensor-output-to-prediction gap

save_model(..., classes=...) / load_classes() / interpret_classification() /
`forge model predict` CLI -- image_folder_classification's classes.json
sidecar is gone; a .forge file now fully describes how to turn a new image
into "Prediction: dog / Confidence: 94.2%" from a fresh process or the CLI.
```
