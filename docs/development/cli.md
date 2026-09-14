# Forge Command-Line Interface (Milestone 19, extended in 72)

## Purpose
A thin command-line adapter over Forge's existing public persistence and
benchmark APIs -- `forge model ...`, `forge checkpoint ...`, `forge
benchmark ...`. It exposes operations a Python caller could already do with
`forge.save_model()`/`forge.load_model()`/`forge.save_checkpoint()`/
`forge.load_checkpoint()` and the `benchmarks` package, for use without
writing a script. It implements no new framework logic: every command either
reads existing archive metadata read-only, or calls one of the functions
above directly. See `forge/cli/main.py`'s module docstring for the package
layout.

## Installation / entry point
```bash
python -m forge ...          # always available once `forge` is importable
forge ...                    # available after `pip install -e .` (or a
                              # normal install), via the `forge` console-script
                              # entry point declared in pyproject.toml
```
`import forge` never imports `forge.cli` -- the CLI package is only reached
through one of the two entry points above, so a plain library import pays no
CLI-related cost.

## Help
```bash
forge --help
forge model --help
forge model inspect --help
forge model convert --help
forge model predict --help
forge checkpoint --help
forge checkpoint inspect --help
forge checkpoint convert --help
forge benchmark --help        # forwarded to `python -m benchmarks --help`
```

## Model inspection
```bash
forge model inspect PATH [--json]
```
Reports the saved format version, the module hierarchy (types and dotted
names), every parameter's name/shape/dtype, total parameter count, the
recorded device, the saved training/eval mode, whether a preprocessing
pipeline was saved (`forge.save_model(..., preprocessing=...)`, Milestone
71), and the saved class-name vocabulary if any (`forge.save_model(...,
classes=...)`, Milestone 72). Never prints tensor values.

The module tree and parameter list (text and `--json`) are always ordered by
construction order (e.g. a `Sequential`'s children as `0, 1, 2, ..., 12`, not
`0, 1, 10, 11, 12, 2, ...`) regardless of the archive's on-disk key order --
see Milestone 89.

**Never requires CUDA, regardless of the model's recorded device**, and
never reconstructs a live `Module` or runs any computation: it reads only
`metadata.json` from the archive (via `forge.serialization.archive
.read_archive`, the same primitive `load_model()` itself uses internally),
so a CUDA-saved model can be inspected on a CPU-only machine. This also
means `inspect` does not require the saved module types to be registered in
the current process -- unlike `load_model()`/`model convert`, which do.

## Checkpoint inspection
```bash
forge checkpoint inspect PATH [--json]
```
Reports the checkpoint format version, the model architecture (same as
`model inspect`), the optimizer type and hyperparameter configuration, epoch
and global step, the recorded device, whether RNG state is present, and how
many of the optimizer's parameters have saved per-parameter state (e.g. Adam
`m`/`v`) -- a count, never the state values themselves.

Like `model inspect`, this reads only `metadata.json` and never requires
CUDA. It deliberately never calls `forge.load_checkpoint()`: that function's
documented contract includes **overwriting `forge.random`'s process-global
RNG state** as part of restoring a checkpoint (the mechanism that makes
resumed training deterministic) -- exactly the kind of state mutation an
`inspect` command must never trigger. `checkpoint convert` (below), which is
not a read-only operation, does call `load_checkpoint()`/`save_checkpoint()`
directly and inherits that documented RNG side effect, same as any Python
caller of those functions would.

## Model conversion
```bash
forge model convert MODEL --device {cpu,cuda} --output OUTPUT
```
Loads `MODEL` explicitly onto `--device` (`forge.load_model(MODEL,
device=...)`) and saves the result to `OUTPUT` (`forge.save_model(...)`).
Performs no computation beyond the transfer `load_model`/`save_model`
already do. `--device` is required -- there is no default and no "use CUDA
if convenient" behavior; requesting `--device cuda` with no CUDA backend
available fails with a clear error and a non-zero exit status rather than
silently falling back to CPU.

## Model prediction (Milestone 72, made task-aware in Milestone 88, extended to sequence generation in Milestone 90 and tabular classification in Milestone 91)
```bash
forge model predict MODEL INPUT [--device {cpu,cuda}] [--output PATH] [--length N] [--json]
```
Predicts from a saved `.forge` artifact using only what the file itself
already carries -- the developer never has to know or pass which of
classification/regression/segmentation/sequence/tabular_classification
`MODEL` is. This command
reads the artifact's own persisted `task` metadata (`forge.save_model(...,
task=...)`, Milestone 87, via `forge.inspect_model()`) and delegates
straight to `forge.predict_model()` (Milestone 86), which dispatches to the
matching task-specific function unchanged. No new inference logic lives
here.

**`INPUT`'s shape depends on the artifact's task:**

- **classification** -- `INPUT` is an image file path, decoded the same way
  `ImageFolder` does. With a saved class vocabulary (`classes=...`):
  ```text
  Prediction: dog
  Confidence: 94.2%
  ```
  Without one, falls back to the raw index (no confidence is claimed, since
  there is no label to attach it to):
  ```text
  Prediction: class index 3 (no class-name vocabulary was saved with this model)
  ```
- **regression** -- `INPUT` is a path to a JSON file of numeric data: a flat
  list (`[1.2, 3.4, 5.6, 7.8]`) is treated as one unbatched sample and given
  a leading batch dimension; a nested list (`[[1.2, 3.4], [5.6, 7.8]]`) is
  treated as already batched. A leading UTF-8 byte-order mark (BOM) is
  tolerated and stripped (Milestone 89) -- common on Windows, where
  `Out-File`/`>`/Notepad's "UTF-8" option all write one by default. Prints
  the raw numeric prediction:
  ```text
  Prediction: [[0.8134]]
  ```
- **segmentation** -- `INPUT` is an image file path; `--output PATH` is
  **required** and receives the predicted mask, written via
  `forge.data.save_image()`:
  ```text
  Predicted mask saved to: mask.png
  ```
- **sequence** (Milestone 90) -- `INPUT` is literal seed text on the command
  line, not a file path (there is nothing to decode); tokenized as
  individual characters (`list(INPUT)`), matching the char-level vocabulary
  `examples/char_rnn` uses. `--length N` (default 200) is the number of new
  characters to generate. Prints the full generated text, seed included:
  ```text
  Generated: a tensor produces a sample...
  ```
  A word-level (or other custom-tokenized) sequence artifact is not
  representable through this command's char-level convention -- call
  `forge.predict_sequence_artifact()` directly with a pre-tokenized seed
  list instead.
- **tabular_classification** (Milestone 91) -- `INPUT` is a path to a JSON
  file of numeric data, parsed exactly like **regression**'s (a flat list is
  one sample; a nested list is already batched), but the output is
  classification-shaped -- one prediction per input row, since the input may
  legitimately be more than one sample:
  ```text
  Prediction: warning
  Confidence: 87.1%
  ```
  A multi-row input prints one numbered block per row instead:
  ```text
  Sample 0: Prediction: normal
  Sample 0: Confidence: 91.0%
  Sample 1: Prediction: fault
  Sample 1: Confidence: 76.4%
  ```
  Without a saved class vocabulary, falls back to raw indices per row, the
  same as **classification**. See `docs/development/
  m91-tabular-classification-artifact-inference.md` for why this needed its
  own task value: `task="classification"`'s `INPUT` has always meant "an
  image file path" (`predict_artifact()`), and a tabular classification
  artifact's input is an already-batched numeric feature vector instead --
  before this task existed, this exact artifact shape (numeric input,
  `task="classification"`) failed with `predict_artifact() requires image to
  be a file path (str or os.PathLike), got ndarray`.

`--json` prints a stable, machine-readable result instead of the text above,
e.g. `{"task": "classification", "class": "dog", "confidence": 0.942}` (or
`{"task": "regression", "prediction": [[0.8134]]}` /
`{"task": "segmentation", "output_path": "mask.png"}` /
`{"task": "sequence", "seed": "a tensor", "generated": "a tensor produces..."}` /
`{"task": "tabular_classification", "predictions": [{"class": "warning", "confidence": 0.871}]}`).

Requires `MODEL` to have been saved with `preprocessing=...` for
classification/segmentation -- there is nothing to reproduce automatically
otherwise, and this command never guesses or silently skips preprocessing;
it fails with a clear error instead. A sequence artifact has no
preprocessing concept; it requires `classes=...` instead (the saved token
vocabulary -- see **Model prediction** limitations below). A
tabular_classification artifact's `preprocessing=`/`classes=` are both
optional, exactly like regression's/classification's own.

**Custom model classes.** `MODEL` must be built entirely from module types
already registered in the running `forge` CLI process (Forge's built-ins --
`Linear`, `Conv2d`, `Sequential`, etc. -- are pre-registered; a hand-written
composite class is not, per `docs/architecture/persistence.md`'s **Known
limitations**). This is a pre-existing, general persistence requirement, not
specific to any one task -- it already applied to `model convert` before
Milestone 90. It does mean the bare CLI cannot `predict`/`convert` a
custom-class artifact like `examples/char_rnn`'s `CharRNN`: that example's
`infer.py` imports its own `model.py` (registering `CharRNN` as a side
effect) before calling the same `forge.predict_model()` this command uses
internally -- see that script and `examples/resnet`/`examples/autoencoder`'s
READMEs, which document the identical constraint for their own custom
classes.

**Legacy artifacts (no `task=`, saved before Milestone 87).** This command
never falls back to an architecture-based guess to pick a task: doing so
could silently misidentify a classification artifact saved with
`classes=None` as a regression artifact (both are `Linear`-terminated with
no saved `classes` -- the exact ambiguity Milestones 86/87 documented and
refused to ship a CLI dispatcher around). A legacy artifact with no `task=`
fails clearly instead:
```text
Error: artifact does not declare a task.
Use the task-specific prediction API or resave the model with task metadata.
```
Resave the model with `forge.save_model(..., task=...)` (or
`train_and_save()`/`save_and_verify()`'s own `task=`), or call the
task-specific Python API directly (`forge.predict_artifact()`/
`forge.predict_tensor_artifact()`/`forge.predict_image_artifact()`/
`forge.predict_tabular_classification_artifact()`), which are unaffected by
this limitation.

This is a thin adapter over exactly the same Python sequence
`examples/image_folder_classification/infer.py`, `examples/segmentation/
infer.py`, and `examples/regression/train.py`'s own demo each already
demonstrate as standalone scripts.

## Checkpoint conversion
```bash
forge checkpoint convert CHECKPOINT --device {cpu,cuda} --output OUTPUT
```
Loads `CHECKPOINT` explicitly onto `--device` (`forge.load_checkpoint()`)
and saves the reconstructed model, optimizer (type, hyperparameters, and
per-parameter state), and training progress (epoch, global step) back out
(`forge.save_checkpoint()`), preserving everything the checkpoint format
itself preserves. Same explicit-device/no-fallback policy as `model
convert`.

## Benchmark invocation
```bash
forge benchmark [any argument accepted by `python -m benchmarks`]
```
A pure pass-through: every argument after `benchmark` is forwarded unparsed
to `benchmarks.run.main()`, so `--categories`/`--output`/benchmark selection
logic lives in exactly one place (`benchmarks/run.py`), not duplicated here.
See `docs/performance/benchmarking.md` for the underlying suite.

`benchmarks/` is a top-level package outside `forge`, deliberately excluded
from `forge`'s own installation (`import forge` never touches it, matching
Milestone 11's original design). `forge benchmark` therefore **only works
when run from within the Forge repository** -- against an installed `forge`
package with no `benchmarks/` directory alongside it, it fails with a clear
"benchmark suite is not available" error rather than a raw `ImportError`.

## Device behavior
Every device-affecting command that changes a model's device (`model
convert`, `checkpoint convert`) requires an explicit `--device {cpu,cuda}`
-- Forge's existing "no implicit device fallback" policy
(`docs/architecture/persistence.md`), unchanged by the CLI. `model predict`'s
`--device` is optional, matching `forge.load_model()`'s/`forge.predict()`'s
own default (the device the model was saved from) -- prediction does not
*convert* the file, so there is no ambiguity to force an explicit choice
about. Inspection commands never touch a device or a backend at all.

## CUDA requirements
- `model inspect` / `checkpoint inspect`: never require CUDA, for any input.
- `model convert` / `checkpoint convert` with `--device cpu`: never require
  CUDA, even for a CUDA-recorded input file.
- `model convert` / `checkpoint convert` with `--device cuda`: require a
  real CUDA backend; if unavailable, fail with a clear error and exit
  status 1 (never a silent CPU fallback).
- `model predict`: requires CUDA only if `--device cuda` is passed explicitly,
  or if omitted and the model was saved from `"cuda"` -- identical policy to
  `forge.load_model()`'s own (**Device semantics**,
  `docs/architecture/persistence.md`).
- `benchmark`: CUDA-dependent categories (`transfer`, and the CUDA half of
  `forward`/`backward`/`training`) produce no results when CUDA is
  unavailable, exactly as `python -m benchmarks` already behaves
  unmodified.

## Error behavior
All CLI-facing errors -- missing file, malformed/corrupt archive,
unsupported format version, invalid `--device` choice, unavailable CUDA,
unknown command, missing required argument -- print a concise `Error: ...`
message to stderr and exit non-zero. Argument-parsing errors (an invalid
`--device` choice, a missing required flag, an unknown subcommand) are
reported by `argparse` itself (exit status 2, its own convention); every
other user-facing failure is a `forge.exceptions.ForgeError` (most commonly
`PersistenceError`) or a CLI-specific `CLIError`, both caught by
`forge/cli/main.py`'s single handler and printed without a Python traceback.
An unexpected internal exception (not one of the above) is left to propagate
with its full traceback, for diagnosability during development -- it is
never silently swallowed.

## Limitations
- **No automatic training resume.** The checkpoint format does not capture
  enough to reconstruct an arbitrary caller's dataset/model-construction
  code, and the CLI deliberately does not invent a second training engine to
  paper over that gap (Milestone 19's explicit non-goal). `forge checkpoint
  inspect`/`convert` are the CLI's checkpoint story; resuming an actual
  training run remains a Python `forge.load_checkpoint()` call followed by
  the caller's own `Trainer`/training-loop code, exactly as before this
  milestone.
- **`forge benchmark` is a development-only command** (see **Benchmark
  invocation** above) -- it is not available from a `pip`-installed `forge`
  package used outside the Forge repository.
- No experiment tracking, config/YAML system, hyperparameter sweeps,
  distributed-training commands, cloud storage, dataset downloading, job
  scheduling, daemon/server mode, model serving, or REST API -- all
  explicitly out of scope for this milestone.
- **`model predict` supports exactly the four tasks `forge.predict_model()`
  does** (Milestones 88/90): classification, regression, segmentation,
  sequence. There is no generic `forge predict` command spanning every
  Forge workload (autoencoders, ...) -- those have a different natural
  input shape with no single saved-artifact-describable file convention yet.
- **`model predict`'s sequence support is char-level-tokenization-only**
  (Milestone 90): the CLI splits `INPUT` into individual characters; a
  vocabulary tokenized a different way (e.g. `examples/word_rnn`'s
  whole-word vocabulary) needs `forge.predict_sequence_artifact()` called
  directly with a pre-tokenized seed list -- see **Model prediction** above.
- **`model predict` requires explicit `task=` metadata** (Milestone 88): a
  legacy artifact saved before Milestone 87 (no `task=`) is not predictable
  through this command -- see **Model prediction** above for why, and
  `docs/development/m88-unified-artifact-prediction-cli.md` for the full
  reasoning.
