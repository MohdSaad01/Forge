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

## Model prediction (Milestone 72)
```bash
forge model predict MODEL --image IMAGE [--device {cpu,cuda}]
```
Classifies one image file against a saved model, using only what the `.forge`
file itself already carries: `forge.load_preprocessing(MODEL)` (Milestone 71)
to prepare `IMAGE` the same way training did, then `forge.load_model()` +
`forge.predict()` + `forge.load_classes(MODEL)` (Milestone 72) +
`forge.interpret_classification()` to turn the raw output into a label +
confidence:
```text
Predicted class: dog
Confidence: 94.2%
```
Requires `MODEL` to have been saved with `preprocessing=...` -- there is
nothing to reproduce automatically otherwise, and this command never
guesses or silently skips preprocessing; it fails with a clear error
instead. A model saved without `classes=...` still predicts, falling back to
`Predicted class index: N` (no confidence is claimed either, since there is
no label to attach it to). This is a thin adapter over exactly the same
Python sequence `examples/image_folder_classification/infer.py` demonstrates
as a standalone script -- no new inference logic lives here. Scoped to a
single image file: unlike `model inspect`/`convert`, this is the CLI's first
command that actually runs model computation, and image classification is
currently the only Forge workflow where a saved artifact fully describes how
to turn one input *file* into a prediction end to end (tabular/sequence
workloads have no equivalent single-file input convention yet -- see
`docs/development/m72-classification-metadata.md`'s **Rejected
alternatives**).

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
- **`model predict` is image-classification-only** (Milestone 72): it always
  decodes `--image` via `ImageFolder._load_image` and always interprets the
  model's output as per-class classification scores. There is no generic
  `forge predict` command spanning every Forge workload (tabular regression,
  autoencoders, sequence models, ...) -- each has a different natural input
  shape with no single saved-artifact-describable file convention yet; see
  `docs/development/m72-classification-metadata.md`.
