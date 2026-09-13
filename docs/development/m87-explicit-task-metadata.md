# M87 — Explicit Artifact Task Metadata and Reliable Prediction Dispatch

## 1. Executive summary

M86 built `forge.predict_model()`, a unified dispatcher over the three
task-specific portable-artifact inference functions
(`predict_artifact()`/`predict_tensor_artifact()`/`predict_image_artifact()`),
but its own report explicitly flagged a real, tested limitation: a
classification artifact saved with `classes=None` (a valid, documented state)
is architecturally indistinguishable from a regression model -- both are
`Linear`-terminated with no saved `classes` -- so `_determine_workflow()`
misidentified it as regression. M86 correctly declined to "fix" this with a
more elaborate architecture heuristic, since no amount of tensor-shape or
module-name inspection can honestly resolve an ambiguity the architecture
itself does not encode.

M87 closes this gap the right way: by making the `.forge` artifact
**semantically self-describing**. `forge.save_model(..., task=...)` lets a
caller declare, explicitly, which of `"classification"`/`"regression"`/
`"segmentation"` an artifact represents. `forge.predict_model()` now checks
this declaration first, and only falls back to the old (renamed, isolated)
architecture heuristic for artifacts that never declared one. This is a
strictly additive, backward-compatible change -- no `FORMAT_VERSION` bump, no
change to any existing function's behavior for a file that omits `task=`.

## 2. Final public API

```python
forge.save_model(model, path, preprocessing=None, classes=None, task=None)
forge.save_and_verify(model, path, sample, preprocessing=None, classes=None, task=None, ...)
forge.train_and_save(model, dataset, ..., path=..., sample=..., preprocessing=None, classes=None, task=None, ...)

info = forge.inspect_model(path)
info.task   # "classification" / "regression" / "segmentation" / None

result = forge.predict_model(path, input_data)   # unchanged signature; dispatch behavior improved
```

No existing function's signature lost a parameter or changed a default; the
new `task=` keyword is optional and defaults to `None` everywhere it appears.

## 3. Exact task vocabulary

```python
forge.serialization.model.TASK_TYPES == ("classification", "regression", "segmentation")
```

A small, fixed, closed vocabulary -- one entry per existing
`predict_*_artifact()` function, not an open-ended or user-extensible
registry. `save_model(task=...)` raises `PersistenceError` immediately for
any other value, before anything is written.

## 4. Artifact metadata representation

`save_model()` writes one new, optional, JSON-safe top-level metadata key,
a sibling of `"preprocessing"`/`"classes"`:

```json
{
    "forge_format_version": 2,
    "device": "cpu",
    "root": { ... },
    "preprocessing": { ... } | null,
    "classes": ["cat", "dog"] | null,
    "task": "classification" | "regression" | "segmentation" | null
}
```

Omitting `task=` (the default) writes `"task": null` -- byte-for-byte
equivalent, in this respect, to every file saved before Milestone 87.

**Validation performed at save time** (`_validate_task()`,
`forge/serialization/model.py`):
- `task` must be `None` or one of `TASK_TYPES`, else `PersistenceError`.
- `task in ("regression", "segmentation")` combined with a non-`None`
  `classes=` raises `PersistenceError` -- neither workflow has a
  class-vocabulary concept, so the combination would describe an artifact
  whose two metadata entries disagree.
- `task="classification"` places **no** restriction on `classes` --
  `classes=None` remains valid, and is precisely the state this milestone
  exists to identify correctly.
- `task` is never validated against `model`'s actual architecture (Forge
  does not introspect a module tree to guess or check its task, mirroring
  `classes`'s own "never validated against output width at save time"
  precedent) -- an inaccurate `task` is accepted at save time and only
  matters the next time `predict_model()` dispatches on it.

## 5. Format-version decision and compatibility behavior

**No `FORMAT_VERSION` bump.** Per `docs/architecture/persistence.md`'s own
existing policy (established by `"preprocessing"`/Milestone 71 and
`"classes"`/Milestone 72, and explicitly *not* bumped for either): a version
bump is reserved for a change to the *required* shape of every module node
(e.g. Milestone 53's new required `"buffers"` key, which really did break
old readers). Adding a new **optional** top-level key that is:

- forward-compatible (a pre-M87 Forge build's `load_model()`/`inspect_model()`
  never reads `"task"` at all, so an M87-or-later file loads on an older
  build exactly as before), and
- backward-compatible (`inspect_model()` treats a missing key exactly like an
  explicit `null`, returning `info.task = None`, not raising)

...requires no version bump, by that same existing policy. This decision was
verified directly, not just asserted: `tests/test_task_metadata.py::
test_inspect_model_task_is_none_for_legacy_no_task_key` constructs a genuine
pre-M87-shaped archive (the `"task"` key deleted entirely, via the same
metadata-tampering helper `test_model_inspection.py`/
`test_classification_metadata.py` already use) and confirms it still loads
and inspects cleanly.

## 6. How `inspect_model()` exposes task

```python
info = forge.inspect_model(path)
info.task   # "classification" / "regression" / "segmentation" / None
print(info) # includes a "Task: <value>" line, or
            # "Task: unknown (legacy artifact, saved before Milestone 87)"
```

`ModelInfo.task` is a new field on the existing frozen dataclass (no new
type introduced). `inspect_model()` validates the raw `"task"` value against
`TASK_TYPES` (or `None`) and raises `PersistenceError` for anything else --
a malformed/tampered file, since `save_model()` itself never writes any
other value there. Reading `task` requires no CUDA and reconstructs no live
`Module`, exactly like every other `ModelInfo` field.

## 7. How `predict_model()` dispatches

`_determine_workflow()` (`forge/training/inference.py`) was restructured:

```python
def _determine_workflow(info):
    if info.task is not None:
        return info.task
    return _legacy_infer_workflow(info)
```

`_legacy_infer_workflow()` is the exact, unmodified M86 heuristic (`classes`
presence -> classification; else `Linear`/`Conv2d` module-type signal),
renamed and given its own docstring explaining precisely what it can and
cannot safely tell apart, and why it survives unchanged. It is reached
**only** when `info.task is None`.

This means: for any artifact saved with an explicit `task=`, dispatch is now
O(1) and metadata-only -- no `classes`/`module_types` inspection occurs at
all. `predict_model()`'s own delegation (`predict_artifact()`/
`predict_tensor_artifact()`/`predict_image_artifact()`, called unchanged) is
untouched.

## 8. Legacy artifact behavior

Tested directly against all four states M86's own report enumerated
(`tests/test_task_metadata.py`):

| State | Legacy (no `task`) behavior | With explicit `task=` |
|---|---|---|
| classification + classes | Correctly identified (unambiguous: `classes` present) | Correctly identified |
| classification, no classes | **Misidentified as regression** (unchanged M86 limitation -- architecturally indistinguishable from a legacy regression artifact) | **Correctly identified** -- the specific M86 bug, fixed |
| regression | Correctly identified (`Linear`, no `classes`) | Correctly identified |
| segmentation | Correctly identified (`Conv2d`, no `Linear`, no `classes`) | Correctly identified |

The one persisting gap (classification-without-classes, no `task`) is
structural, not an oversight: with no `task` key and no `classes`, such an
artifact is byte-for-byte indistinguishable, at the metadata level, from a
real legacy regression artifact that must keep working (`examples/
regression`'s own pre-M87 saved files). There is no metadata left to decide
correctly without the caller re-saving the file with `task=` -- exactly the
capability M87 adds. `tests/test_task_metadata.py::
test_predict_model_legacy_classification_without_classes_still_misdispatches`
pins this down explicitly, mirroring `test_unified_artifact_prediction.py`'s
own M86-era pin for the same state, so a future change surfaces here rather
than as a silent regression either way.

An artifact whose `task` is present but not one of `TASK_TYPES` (a
malformed/tampered file) raises `PersistenceError` from `inspect_model()`
before `predict_model()` ever reaches dispatch logic.

## 9. Classification-without-classes behavior

This is the headline fix. `task="classification", classes=None` is now:

- Accepted at save time (`test_save_model_classification_task_allows_classes_none`).
- Reported correctly by `inspect_model()` (`info.task == "classification"`).
- Dispatched correctly by `predict_model()` to `predict_artifact()`, which
  returns the raw predicted class index as an `int` (unchanged
  `predict_artifact()` behavior for a `classes=None` artifact) --
  `test_predict_model_dispatches_classification_without_classes_via_explicit_task`.
- Proven to override the architecture ambiguity directly:
  `test_predict_model_explicit_task_overrides_architecture_ambiguity` saves
  exactly this artifact, confirms its architecture *does* contain a `Linear`
  layer (the same signal that would say "regression"), confirms `info.task`
  is `"classification"` anyway, and confirms `predict_model()` rejects a
  `Tensor`/accepts an image path -- proving it really did dispatch through
  `predict_artifact()`, not `predict_tensor_artifact()`.

## 10. Which example producers were updated

All four required producers, verified by retraining each end-to-end (small
smoke configurations) and inspecting the resulting artifact's recorded task
via `forge model inspect --json`:

| Example | `task=` | Verified |
|---|---|---|
| `examples/mnist/train.py` | `"classification"` (both `--resume` and fresh branches) | `task= classification` |
| `examples/image_folder_classification/train.py` | `"classification"` (both branches) | `task= classification` |
| `examples/regression/train.py` | `"regression"` (both branches) | `task= regression` |
| `examples/segmentation/train.py` | `"segmentation"` | `task= segmentation` |

Two additional classification examples were updated mechanically, per the
brief's "if the change can be made mechanically without broadening scope"
allowance (both already save real classification artifacts, no new save
call introduced):

- `examples/waveform_classification/train.py` -- `task="classification"`
  added to its existing `save_and_verify()` call.
- `examples/resnet/train.py` -- `task="classification"` added to its
  existing `save_and_verify()` call (alongside its existing `classes=`).

`examples/autoencoder/train.py` and the three stepwise-recurrence examples
(`char_rnn`/`word_rnn`/`long_range_recall`) were **not** touched -- none of
them produce an artifact matching one of the three `TASK_TYPES` (autoencoder
reconstruction and sequence generation are neither classification,
regression, nor segmentation in `predict_model()`'s sense), so assigning any
task string to them would be inaccurate metadata, not a mechanical update.

## 11. Whether `train_and_save()` was updated

Yes. `train_and_save()` (`forge/training/api.py`) gained a `task=None`
keyword, passed straight through to `save_and_verify()` (itself passing it
straight through to `save_model()`) -- no new validation or logic beyond
that pass-through. This is the recommended modern producer path:

```python
forge.train_and_save(model, dataset, ..., path="model.forge", task="classification")
```

`train()` itself was not touched -- it has no persistence step to attach
`task=` to.

## 12. CLI changes

- `forge model inspect` -- new "Task" line in text output (`"unknown (legacy
  artifact, saved before Milestone 87)"` when absent), and a `"task"` field
  in `--json` output (`null` when absent). Both delegate to
  `inspect_model()`'s own `ModelInfo.task`, matching Milestone 85's
  "Preprocessing detail" precedent of delegating to the public API rather
  than re-deriving the value.
- `forge model convert` -- now also preserves `task` across a device
  conversion (`inspect_model(...).task` read, then passed to the output
  `save_model()` call), alongside the `preprocessing`/`classes` it already
  preserved -- a converted file remains a complete, self-describing
  artifact.
- `forge model predict` -- **unchanged**. Evaluated again (the module
  docstring in `forge/cli/model.py` documents the reasoning): this command's
  own real, passing test
  (`test_classification_metadata.py::test_cli_predict_without_classes_prints_index`)
  exercises a classification artifact saved with `classes=None` and **no**
  `task=` either -- a genuinely legacy-shaped artifact by this milestone's
  own definition. Routing this command through `predict_model()` would
  therefore still hit the legacy fallback and still misdispatch, so there
  is no real gain to weigh against the risk of silently changing this
  command's already-correct behavior. `predict` stays a thin,
  `--image`-only wrapper directly over `predict_artifact()`.

## 13. Files changed

- `forge/serialization/model.py` -- `TASK_TYPES`, `_validate_task()`;
  `save_model(..., task=...)`; `ModelInfo.task` field + updated `__str__()`;
  `inspect_model()` reads/validates/returns `task`; updated docstrings;
  `TASK_TYPES` added to `__all__`.
- `forge/serialization/__init__.py` -- exports `TASK_TYPES`; updated module
  docstring.
- `forge/__init__.py` -- updated module docstring (no new top-level
  re-export of `TASK_TYPES` -- reachable via `forge.serialization.TASK_TYPES`,
  consistent with `FORMAT_VERSION`'s own non-top-level precedent).
- `forge/training/inference.py` -- `save_and_verify(..., task=...)`;
  `_legacy_infer_workflow()` (renamed/isolated from the old
  `_determine_workflow()` body); new `_determine_workflow()` that checks
  `info.task` first; updated docstrings.
- `forge/training/api.py` -- `train_and_save(..., task=...)`, passed through
  to `save_and_verify()`; updated docstrings and usage example.
- `forge/cli/model.py` -- "Task" line/JSON field in `cmd_inspect()`; `task`
  preserved in `cmd_convert()`; updated module docstring.
- `examples/mnist/train.py`, `examples/image_folder_classification/train.py`,
  `examples/regression/train.py`, `examples/segmentation/train.py` --
  `task=` added to their `save_and_verify()`/`train_and_save()` calls.
- `examples/waveform_classification/train.py`, `examples/resnet/train.py` --
  `task="classification"` added mechanically.
- `docs/architecture/persistence.md` -- new **Task metadata (Milestone 87)**
  section; updated **Model inspection** section, `ModelInfo` public-API
  block, package-layout header, **Errors**, and **Known limitations**.
- `docs/architecture/training-engine.md` -- updated **Unified
  portable-artifact prediction** section (dispatch signal, documented
  limitation, real consumers, rejected alternatives), package-layout header,
  file title, `save_and_verify()`/`train_and_save()` usage examples.
- `README.md` -- updated `forge.serialization` bullet and the
  train/save/predict paragraph to mention `task=`/`predict_model()`/
  `inspect_model()`.
- `examples/mnist/README.md`, `examples/regression/README.md`,
  `examples/segmentation/README.md`,
  `examples/image_folder_classification/README.md` -- one-paragraph
  mentions of the new `task=` metadata in context.
- `docs/development/progress.md` -- new M87 entry.
- `docs/development/m87-explicit-task-metadata.md` -- this report.
- `tests/test_task_metadata.py` (new, 37 CPU tests).
- `tests/test_task_metadata_cuda.py` (new, 3 CUDA tests, hardware-verified).
- `tests/test_inference.py`, `tests/test_training_api.py` -- `task=`
  round-trip tests added for `save_and_verify()`/`train_and_save()`.

## 14. Architecture impact

None to `Tensor`, autograd, CUDA kernels, `Module`, `Trainer`,
`TrainingSession`, `DataLoader`, or the archive file format
(ZIP(json + `.npy`), unchanged). `task` is a plain JSON string, a sibling
metadata entry exactly like `preprocessing`/`classes` -- no new persisted
data shape, no new archive layout, no new registry, no model-architecture
inference engine, no generic metadata dictionary. The entire change lives in
existing metadata-handling code paths already established by Milestones
71/72/85/86.

## 15. Tests added

`tests/test_task_metadata.py` (37 CPU tests):
- `TASK_TYPES` vocabulary shape; each valid task accepted and round-trips
  through `inspect_model()`; omitted/explicit-`None` task both default to
  `None`; invalid task values (wrong string, wrong type) rejected before
  writing.
- `task="classification"` with `classes=None` and with `classes=[...]`, both
  valid; `task="regression"`/`"segmentation"` combined with `classes=[...]`
  rejected; `task="regression"`/`"segmentation"` alone (no `classes`)
  accepted.
- `inspect_model()`: `task is None` for a genuinely legacy (tampered,
  key-deleted) archive; `PersistenceError` for a malformed `"task"` value;
  `ModelInfo.__str__()` includes the "Task:" line for both an explicit task
  and the "unknown (legacy artifact...)" case.
- CLI: "Task" line in text output, `"task"` field in `--json` output
  (present and `null`-for-legacy cases), `task` preserved across `forge
  model convert`.
- `predict_model()` dispatch: all three workflows dispatched via explicit
  `task=`, including the headline classification-without-classes fix;
  explicit task proven to override the architecture-ambiguity signal
  directly; unified result compared against the direct `predict_artifact()`
  call.
- Legacy fallback: unchanged M86 behavior for a real legacy
  classification-with-classes artifact; the documented,
  still-not-retroactively-fixable classification-without-classes-and-no-task
  misdispatch, pinned explicitly; the undeterminable-artifact
  `PersistenceError` (updated message wording, since it is now reached only
  via the legacy path).
- Fresh process: one genuine `subprocess.run([sys.executable, "-c", ...])`
  call dispatching an explicit-task classification artifact, compared
  against this process's own `predict_model()` call.
- Basic re-export consistency.

`tests/test_task_metadata_cuda.py` (3 tests, hardware-verified on the
940MX): `inspect_model()` reads `task` from a CUDA-recorded artifact with
CUDA explicitly monkeypatched unavailable (no CUDA requirement for
inspection, mirroring Milestone 85's own contract); `predict_model()`
restores a CUDA-saved, explicit-task classification artifact onto CUDA by
default; CPU-saved and CUDA-saved explicit-task regression artifacts agree
numerically.

`tests/test_inference.py`/`tests/test_training_api.py`: one `task=`
pass-through test and one no-`task=`-default test each, for
`save_and_verify()`/`train_and_save()` respectively.

## 16. Full-suite result

```text
python -m pytest tests/
2,453 collected, 2,452 passed, 1 failed
```

The one failure is
`test_dataloader_prefetch.py::test_repeated_epochs_do_not_grow_cuda_or_pinned_memory`
-- the same pre-existing CUDA allocator-measurement flake documented since
M63 (confirmed again here: `288` vs `320` bytes on this run). Re-run in
isolation immediately after:

```text
python -m pytest tests/test_dataloader_prefetch.py::test_repeated_epochs_do_not_grow_cuda_or_pinned_memory
1 passed in 1.05s
```

Confirming it is unrelated to M87. Every pre-existing persistence/inference/
CLI test suite (`test_serialization.py`, `test_preprocessing_persistence.py`,
`test_checkpoint.py`, `test_cli.py`, `test_model_inspection.py`,
`test_classification_metadata.py`, `test_unified_artifact_prediction{,_cuda}.py`)
re-ran green, unchanged.

## 17. Fresh-process verification

`tests/test_task_metadata.py::
test_predict_model_dispatches_via_explicit_task_from_a_genuinely_separate_process`
saves a classification artifact with `classes=None, task="classification"`,
launches a real `subprocess.run([sys.executable, "-c", ...])` process that
imports only `forge`, calls `forge.predict_model()` on it, and compares the
result against this test process's own call to the same file -- proving the
explicit-task dispatch path works with no source-level knowledge of the
model class, and no reliance on any state the training/saving process left
behind. `tests/test_unified_artifact_prediction.py`'s own pre-existing
fresh-process test (covering all three artifact shapes via the legacy path,
since its fixtures never pass `task=`) continues to pass unmodified,
confirming both dispatch paths remain fresh-process-safe.

## 18. CPU/CUDA verification

Task metadata itself is a plain JSON value with no device dependency.
`tests/test_task_metadata_cuda.py` (hardware-verified on the 940MX, CUDA
12.6) confirms: (1) a CUDA-recorded artifact's `task` is readable via
`inspect_model()` with CUDA explicitly made unavailable via monkeypatching,
proving no CUDA requirement leaked in; (2) `predict_model(device=None)` on a
CUDA-saved, explicit-task classification artifact restores onto CUDA by
default and produces a correct result, exactly as `load_model()`'s own
device policy already guarantees; (3) CPU-saved and CUDA-saved explicit-task
regression artifacts of the same weights agree numerically
(`rtol=atol=1e-4`, the same tolerance `test_unified_artifact_prediction_cuda.py`
already established). No CUDA kernel was added or modified.

## 19. Real end-to-end workflows demonstrated

All three workflows demonstrated with real, retrained artifacts (not
artificial test-only models), via the four required examples:

- **Classification**: `examples/mnist/train.py` (both `--resume` and fresh
  paths) and `examples/image_folder_classification/train.py` (both paths)
  retrained with small smoke configurations; `forge model inspect --json`
  confirmed `"task": "classification"` on the resulting `.forge` files; each
  script's own existing fresh-process/in-process inference demo continued to
  produce correct predictions.
- **Regression**: `examples/regression/train.py` retrained; confirmed
  `"task": "regression"`; the script's own `forge.predict_model()` end-of-run
  demo (now dispatching via the explicit task rather than the `Linear`
  heuristic) produced a correct prediction on a brand-new raw feature vector.
- **Segmentation**: `examples/segmentation/train.py` retrained; confirmed
  `"task": "segmentation"`; the script's existing `predict_image_artifact()`
  demo and `infer.py`'s `predict_model()` call (dispatching via the explicit
  task) both continued to produce a correct predicted mask.

## 20. Limitations

- **Legacy classification-without-classes artifacts remain ambiguous.** An
  artifact saved before Milestone 87 (or with `task=` deliberately omitted)
  that is a classification model with `classes=None` is still
  architecturally indistinguishable from a legacy regression artifact and is
  still misidentified by the legacy fallback -- this is a structural
  limitation of metadata that was never recorded, not something Milestone 87
  could resolve retroactively. Documented and pinned by a dedicated test
  (Section 8).
- **`forge model predict` still does not use `predict_model()`/`task`** --
  unchanged from Milestone 86, for the reason in Section 12. It remains a
  correct, narrower, `--image`-only classification command.
- **No task validation against architecture.** `save_model()` accepts an
  inaccurate `task` (e.g. `task="regression"` on a model that is actually a
  classifier) without complaint -- Forge does not introspect architecture to
  cross-check a caller's explicit declaration, matching the existing
  `classes`-vs-output-width precedent. An inaccurate declaration only
  surfaces as a wrong `predict_model()` dispatch, not a save-time error.
- **No `load_task()` public function.** Unlike `load_preprocessing()`/
  `load_classes()`, `task` has no standalone loader function -- it is
  reachable only via `inspect_model().task`. This was a deliberate scope
  choice (see Section 21) since no real consumer needs to fetch `task`
  independently of the rest of `ModelInfo`.

## 21. Rejected/deferred alternatives

- **A standalone `load_task()` function**, mirroring `load_preprocessing()`/
  `load_classes()`. Rejected: no real consumer needs `task` independent of
  `inspect_model()`'s other fields (unlike `preprocessing`/`classes`, which
  `predict_artifact()`/`predict_tensor_artifact()`/`predict_image_artifact()`
  each fetch standalone before calling `predict()`). Adding it would be
  speculative API surface with no demonstrated need; `inspect_model().task`
  is one attribute access away regardless.
- **Retroactively "fixing" the legacy classification-without-classes
  misdispatch** by, e.g., treating an ambiguous legacy artifact as an error
  rather than a guess. Rejected: doing so would break real, already-passing
  behavior for genuine legacy regression artifacts (architecturally
  identical, and needing to keep working) -- there is no metadata signal
  left to decide correctly for either case without the caller re-saving with
  `task=`. Explicitly permitted by this milestone's own brief ("if backward
  compatibility requires retaining the M86 heuristic... isolate it
  explicitly as a legacy fallback").
- **A `FORMAT_VERSION` bump for the new `"task"` key.** Rejected per this
  document's own Section 5 -- the existing, already-established persistence
  versioning policy reserves a bump for required-shape changes, not new
  optional top-level keys with full compatibility in both directions.
- **Retrofitting `forge model predict` onto `predict_model()`.** Re-evaluated
  and again rejected -- see Section 12; the CLI's own real test exercises
  exactly the artifact shape (`classes=None`, no `task=`) this milestone
  could not retroactively disambiguate.
- **Adding `task=` to every other example that calls `save_model()`**
  (`autoencoder`, `char_rnn`, `word_rnn`, `long_range_recall`). Rejected:
  none of these produce an artifact matching one of the three `TASK_TYPES`
  -- assigning any task string would be inaccurate metadata, not a
  mechanical update, and `predict_model()` was never meant to cover these
  workflows (M82-84's own scope).
- **A generic task-plugin/registry system, or automatic architecture-based
  task inference for newly saved artifacts.** Both explicitly out of scope
  per the milestone's own Architecture Guardrails; neither was implemented
  or considered as a real alternative.

## 22. What an external Forge developer can now do

Before M87:

```python
forge.save_model(model, "model.forge", classes=my_classes)
# no way to declare "this is a classification artifact" independent of
# whether classes= happened to be given
result = forge.predict_model("model.forge", input_data)
# could silently misidentify a classes=None classification artifact as regression
```

After M87:

```python
forge.train_and_save(
    model, dataset, loss=..., optimizer=..., epochs=...,
    path="model.forge", sample=..., task="classification",
)
info = forge.inspect_model("model.forge")
print(info.task)   # "classification" -- reliable, explicit, self-describing

result = forge.predict_model("model.forge", "new_image.png")
# dispatches correctly regardless of whether classes= was also saved
```

The `.forge` artifact now carries its own semantic contract -- "I am a
classification/regression/segmentation model" -- rather than requiring Forge
(or a developer) to infer that from module names, layer types, or the
presence of other metadata. This is a genuine reliability improvement for
`forge.predict_model()`, and establishes the metadata field any future
artifact-consuming API can build on.

## 23. Suggested commit message

```
feat: add explicit forge.save_model(task=...) metadata and make forge.predict_model() dispatch on it reliably
```
