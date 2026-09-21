"""`forge model inspect` / `forge model convert` / `forge model predict`.

`inspect` reads archive metadata only (`forge/cli/_archive_info.py`) -- it
never reconstructs a live `Module`, never requires CUDA, and never mutates
anything. Its "Preprocessing detail" line (Milestone 85) and "Task" line
(Milestone 87) both delegate to `forge.inspect_model()` -- the public,
structured inspection API -- rather than re-deriving that information of its
own; the rest of this command's per-module/per-parameter dump remains this
file's own lower-level archive walk (`_archive_info.py`), which stays
intentionally more detailed than `inspect_model()`'s product-level `ModelInfo`
contract. `convert` is a real device conversion and goes straight through
`forge.load_model()` / `forge.save_model()`, exactly as a Python caller
would -- no separate conversion logic lives here; it preserves
preprocessing/classes/task metadata across the conversion (Milestone 87 added
`task` to what it carries over) since a converted file is still meant to be a
complete, self-describing artifact.

`predict` (Milestone 72, made task-aware in **Milestone 88**) is a thin CLI
wrapper over `forge.predict_model()` (Milestone 86) -- the same unified
dispatcher a Python caller would use. Milestones 86/87 deliberately did
*not* route this command through `predict_model()` yet: `predict` used to be
hard-wired to "classify one image" (`predict_artifact()` directly), and a
classification artifact saved with `classes=None` and no `task=` is
architecturally indistinguishable from a legacy regression artifact (both
`Linear`-terminated) -- routing through `predict_model()`'s architecture-based
legacy fallback would have silently misidentified it as regression. Milestone
87's explicit `task=` metadata is what finally makes a *safe* generic
dispatcher possible: this command now reads `forge.inspect_model(model).task`
itself and uses it as the sole routing signal --

- `task` present -> delegate straight to `forge.predict_model()`, which
  returns the matching workflow immediately (no architecture inspection at
  all, since `task` is already authoritative -- see
  `forge/training/inference.py::_determine_workflow()`).
- `task` absent (a genuinely legacy artifact, predating Milestone 87) ->
  fail clearly rather than guess. This command never falls back to
  `predict_model()`'s own `_legacy_infer_workflow()` architecture heuristic
  -- doing so would reintroduce exactly the `classes=None` misdispatch
  Milestones 86/87 documented and refused to ship. A legacy artifact must be
  resaved with `task=`, or predicted via the task-specific Python API
  (`predict_artifact()`/`predict_tensor_artifact()`/`predict_image_artifact()`)
  directly.

This keeps the dispatch logic itself in exactly one place
(`forge.training.inference._determine_workflow()`) -- this command never
re-implements or duplicates it, only decides whether it is safe to call at
all.

Each task's input/output is otherwise unchanged from its established
single-task shape: classification/segmentation take an image file path
(decoded via the same `ImageFolder._load_image()` `predict_artifact()`/
`predict_image_artifact()` already use); regression takes a JSON file of
numeric data, parsed here and handed to `predict_tensor_artifact()` as a
`Tensor`-convertible batch; segmentation additionally requires `--output`
to save the predicted mask via `forge.data.save_image()`. See
`docs/development/cli.md` and `docs/development/
m88-unified-artifact-prediction-cli.md` for the full command reference.

`sequence` (Milestone 90) takes the seed as literal text on the command
line, not a file -- unlike classification/regression/segmentation, there is
no file to decode; `--length` (default 200) controls how many new tokens to
generate. This command tokenizes the seed as individual characters
(`list(seed)`), matching the char-level vocabulary convention
`examples/char_rnn` uses -- the one real sequence workload this milestone
built end-to-end. A different tokenization convention (e.g.
`examples/word_rnn`'s whole-word vocabulary) is not something this command
can infer from the artifact alone; call `forge.predict_sequence_artifact()`
directly with a pre-tokenized seed list for that case.

`tabular_classification` (Milestone 91) shares `regression`'s "JSON file of
numeric data" input parsing (`_parse_numeric_input()`, generalized from what
used to be `_parse_regression_input()` for this exact reason) but produces
classification-shaped results -- one per input row (`_print_classification_
results()`), since a JSON input file may legitimately batch more than one
sample (a flat list is one sample; a nested list is already batched, exactly
like `regression`'s own input convention). A single-row input (the common
case) prints identically to `task == "classification"`'s own single-image
text output; a genuinely multi-row input prints one numbered "Sample N:"
block per row rather than silently reporting only the first. See
`forge.predict_tabular_classification_artifact()`'s own docstring and
`docs/development/m91-tabular-classification-artifact-inference.md` for why
this needed its own task value rather than reusing `"classification"`: that
task's input has always meant "an image file path," and a tabular
classification artifact's input is an already-batched numeric feature
vector instead -- the exact same distinction `"regression"` already makes
from `"classification"`.

`evaluate` (**Milestone 117**) is the command-line door to
`forge.load_predictor(path).evaluate(X, y)` (Milestone 113): `forge model
evaluate MODEL INPUT [TARGETS]`. It is an interface layer and nothing else --
it reads `INPUT`/`TARGETS` from disk (`.npy` files, or -- **Milestone 118** -- one
`.csv` file plus `--target COLUMN`, for the numeric tasks; an `ImageFolder`-layout
directory for image classification), calls
`load_predictor()` once and `ArtifactPredictor.evaluate()` once, and prints the
frozen result dataclass it gets back. It computes no metric, applies no
preprocessing, knows nothing about class order or target transforms
(`ArtifactPredictor` applies the artifact's own, so a Milestone 116
`target_transform="standardize"` artifact reports native-unit metrics with no
CLI involvement), and its JSON is generated from the result dataclass's own
fields (`dataclasses.fields()`), so the two cannot drift apart. As with
`predict`, the artifact's persisted `task` is the only routing signal -- and it
is used solely to decide how to *read* `INPUT`, never how to evaluate it. See
`docs/development/m117-artifact-evaluation-cli.md`.

The `.csv` reader is `forge.data.load_csv()` (Milestone 118), a boundary that only turns
the file into the same `(X, y)` arrays a `.npy` pair would be: the extension picks the
reader, and the CLI still makes exactly one `load_predictor()` and one `evaluate()` call
on those arrays. A regression artifact's `--target` column is read as numbers, a
classification one as class labels (integer indices or text names) -- and that is all
the CSV path knows about the artifact: never its class order, its preprocessing or its
target transform. See `docs/development/m118-csv-tabular-workflow.md`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import zipfile

import numpy as np

from ..data import load_csv, save_image
from ..serialization import (
    inspect_model, load_classes, load_model, load_preprocessing, load_target_transform, save_model,
)
from ..training import (
    ClassificationEvaluationResult, ClassificationPrediction, RegressionEvaluationResult, load_predictor,
    predict_model,
)
from ._archive_info import count_elements, module_training_state, read_model_metadata, walk_modules, walk_parameters
from .errors import CLIError


def add_parser(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser("model", help="Inspect or convert a saved Forge model")
    sub = parser.add_subparsers(dest="model_command", required=True)

    inspect_parser = sub.add_parser("inspect", help="Report a saved model's architecture and metadata")
    inspect_parser.add_argument("path", help="Path to a model file saved with forge.save_model()")
    inspect_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text")
    inspect_parser.set_defaults(func=cmd_inspect)

    convert_parser = sub.add_parser("convert", help="Load a model onto a device and save it back out")
    convert_parser.add_argument("model", help="Path to a model file saved with forge.save_model()")
    convert_parser.add_argument(
        "--device", required=True, choices=["cpu", "cuda"],
        help="Target device to load and re-save the model on -- always explicit, never a fallback.",
    )
    convert_parser.add_argument("--output", required=True, help="Path to write the converted model to")
    convert_parser.set_defaults(func=cmd_convert)

    predict_parser = sub.add_parser(
        "predict",
        help="Predict from a saved model artifact, using its own persisted task metadata to choose the workflow",
    )
    predict_parser.add_argument(
        "model", help="Path to a .forge model file (must declare task metadata -- see forge.save_model(..., task=...))"
    )
    predict_parser.add_argument(
        "input",
        help="Input for the prediction: an image file for classification/segmentation, "
        "a JSON file of numeric data for regression/tabular_classification, "
        "or literal seed text for sequence generation",
    )
    predict_parser.add_argument(
        "--device", default=None, choices=["cpu", "cuda"],
        help="Device to load the model onto (default: whatever device it was saved from)",
    )
    predict_parser.add_argument(
        "--output", default=None, help="Where to save the predicted mask (required for segmentation artifacts)"
    )
    predict_parser.add_argument(
        "--length", type=int, default=200,
        help="Number of new tokens to generate (sequence artifacts only; default 200)",
    )
    predict_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text")
    predict_parser.set_defaults(func=cmd_predict)

    evaluate_parser = sub.add_parser(
        "evaluate",
        help="Score a saved model artifact on labeled data, using its own persisted preprocessing",
    )
    evaluate_parser.add_argument(
        "model", help="Path to a .forge model file (must declare task metadata -- see forge.save_model(..., task=...))"
    )
    evaluate_parser.add_argument(
        "input",
        help="Evaluation inputs: a .csv file (header row; needs --target) or a .npy file of shape (samples, features) "
        "for tabular_classification/regression, or a directory laid out as root/<class_name>/<image files> for "
        "image classification",
    )
    evaluate_parser.add_argument(
        "targets", nargs="?", default=None,
        help="For a .npy INPUT: a .npy file with one target per sample (class names, class indices, or regression "
        "targets in native units). Not accepted for a .csv INPUT (use --target) or for image classification",
    )
    evaluate_parser.add_argument(
        "--target", default=None, metavar="COLUMN",
        help="For a .csv INPUT: the header name of the target column (every other column is a numeric feature, "
        "in file order). Not accepted for a .npy or image INPUT",
    )
    evaluate_parser.add_argument(
        "--device", default=None, choices=["cpu", "cuda"],
        help="Device to load the model onto (default: whatever device it was saved from)",
    )
    evaluate_parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Samples evaluated per batch; only bounds memory (default: the evaluation API's own default)",
    )
    evaluate_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text")
    evaluate_parser.set_defaults(func=cmd_evaluate)


def cmd_inspect(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.path):
        raise CLIError(f"Cannot inspect model '{args.path}': file not found.")

    metadata = read_model_metadata(args.path)
    root = metadata["root"]
    modules = list(walk_modules(root))
    parameters = list(walk_parameters(root))
    total_params = sum(count_elements(meta.get("shape", [])) for _, meta in parameters)
    training = module_training_state(root)
    has_preprocessing = metadata.get("preprocessing") is not None
    classes = metadata.get("classes")

    # Milestone 85: the public inspection contract (forge.inspect_model())
    # is reused here for the preprocessing description rather than
    # re-deriving it -- "CLI must delegate to the public inspection API."
    # Milestone 87: the same call also exposes info.task.
    info = inspect_model(args.path)
    preprocessing_description = info.preprocessing.description if info.preprocessing is not None else None
    task = info.task
    # Milestone 101: `info.input_schema` is `None` for any artifact this
    # milestone builds no feature-count contract for (no task="regression"/
    # "tabular_classification", or an architecture the contract can't be
    # derived from) -- reported as `null`/"n/a", never fabricated.
    input_feature_count = info.input_schema.feature_count if info.input_schema is not None else None
    # Milestone 116: `null` for every artifact without a target transform.
    target_transform = info.target_transform.to_config() if info.target_transform is not None else None

    if args.json:
        payload = {
            "path": args.path,
            "format_version": metadata["forge_format_version"],
            "device": metadata["device"],
            "training": "train" if training else "eval",
            "has_preprocessing": has_preprocessing,
            "preprocessing_description": preprocessing_description,
            "classes": classes,
            "task": task,
            "input_feature_count": input_feature_count,
            "target_transform": target_transform,
            "modules": [{"name": name, "type": type_name} for name, type_name in modules],
            "parameters": [
                {
                    "name": name,
                    "shape": list(meta.get("shape", [])),
                    "dtype": meta.get("dtype"),
                    "requires_grad": bool(meta.get("requires_grad", True)),
                }
                for name, meta in parameters
            ],
            "total_parameters": total_params,
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Model: {args.path}")
    print(f"Format version: {metadata['forge_format_version']}")
    print(f"Device: {metadata['device']}")
    print(f"Task: {task if task is not None else 'unknown (legacy artifact, saved before Milestone 87)'}")
    if input_feature_count is not None:
        print(f"Input: {input_feature_count} feature(s)")
    print(f"Training: {'train' if training else 'eval'}")
    print(f"Preprocessing: {'yes' if has_preprocessing else 'no'}")
    if preprocessing_description is not None:
        print(f"Preprocessing detail: {preprocessing_description}")
    if info.target_transform is not None:
        print(f"Target transform: {info.target_transform!r} (predictions are returned in native units)")
    print(f"Classes: {', '.join(classes) if classes else '(none)'}")
    print()
    print("Modules:")
    for name, type_name in modules:
        depth = 0 if name == "(root)" else name.count(".") + 1
        print(f"  {'  ' * depth}{name}: {type_name}")
    print()
    print("Parameters:")
    if parameters:
        name_width = max(len(name) for name, _ in parameters)
        for name, meta in parameters:
            shape = tuple(meta.get("shape", []))
            dtype = meta.get("dtype")
            print(f"  {name.ljust(name_width)}   shape={shape}  dtype={dtype}")
    else:
        print("  (none)")
    print()
    print(f"Total parameters: {total_params}")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.model):
        raise CLIError(f"Cannot convert model '{args.model}': file not found.")
    output_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    if not os.path.isdir(output_dir):
        raise CLIError(f"Cannot write to '{args.output}': directory '{output_dir}' does not exist.")

    model = load_model(args.model, device=args.device)
    # Preserve preprocessing (Milestone 71)/classes (Milestone 72)/task
    # (Milestone 87) metadata across a device conversion -- a converted file
    # is still meant to be a complete, self-describing artifact, not a bare
    # weights-only copy.
    preprocessing = load_preprocessing(args.model)
    classes = load_classes(args.model)
    task = inspect_model(args.model).task
    # Milestone 116: a regression artifact's target transform is part of what its
    # predictions mean; dropping it here would silently turn native-unit predictions
    # into the model's raw training-space output.
    target_transform = load_target_transform(args.model)
    save_model(
        model, args.output, preprocessing=preprocessing, classes=classes, task=task,
        target_transform=target_transform,
    )
    print(f"Converted '{args.model}' -> '{args.output}' (device={args.device}).")
    return 0


def _parse_numeric_input(path: str) -> np.ndarray:
    """Parse a JSON file of numeric data into a batched NumPy array (Milestone 88;
    generalized to `task="tabular_classification"` in Milestone 91).

    A flat list (`[1.2, 3.4, 5.6, 7.8]`) is treated as one unbatched sample
    and given a leading batch dimension; a nested list (`[[1.2, 3.4], [5.6,
    7.8]]`) is treated as already batched and passed through as-is -- the
    same convention documented in `predict_tensor_artifact()`'s/
    `predict_tabular_classification_artifact()`'s docstrings for a raw NumPy
    array. Anything that is not valid JSON, or whose values are not all
    numeric, raises `CLIError` with one clear message -- covering malformed
    JSON, non-numeric JSON, and a non-JSON file (e.g. an image) given where
    numeric input is expected, all identically. Used for both `task=
    "regression"` and `task="tabular_classification"` (Milestone 91) -- the
    two tasks share the exact same "already-batched numeric array" input
    shape; only what `predict_model()` does with the parsed array differs.
    """
    try:
        # "utf-8-sig" transparently strips a leading UTF-8 BOM if present and
        # behaves identically to "utf-8" otherwise -- Windows tools (e.g.
        # PowerShell's `Out-File`/`>`, Notepad's "UTF-8" save option) commonly
        # write a BOM, and without this a numerically valid JSON file failed
        # with the same misleading "not numeric JSON" error as truly malformed
        # input (discovered during Milestone 89 external-workflow validation).
        with open(path, "r", encoding="utf-8-sig") as fh:
            raw = json.load(fh)
    except (OSError, ValueError, UnicodeDecodeError):
        # ValueError covers json.JSONDecodeError; UnicodeDecodeError covers a
        # binary file (e.g. an image) given where JSON text is expected.
        raise CLIError("input must contain numeric JSON data.")

    try:
        array = np.array(raw, dtype=np.float32)
    except (TypeError, ValueError):
        raise CLIError("input must contain numeric JSON data.")

    if array.dtype == object or array.size == 0:
        raise CLIError("input must contain numeric JSON data.")

    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim != 2:
        raise CLIError("input must be a flat list or a 2-D list of numbers.")

    return array


def _print_classification_result(result: "ClassificationPrediction | int", as_json: bool) -> None:
    if isinstance(result, ClassificationPrediction):
        if as_json:
            print(json.dumps(
                {"task": "classification", "class": result.label, "confidence": result.confidence}, indent=2
            ))
        else:
            print(f"Prediction: {result.label}")
            print(f"Confidence: {result.confidence:.1%}")
    else:
        if as_json:
            print(json.dumps({"task": "classification", "class": None, "index": result, "confidence": None}, indent=2))
        else:
            print(f"Prediction: class index {result} "
                  "(no class-name vocabulary was saved with this model)")


def _print_classification_results(results: "list", as_json: bool, task: str) -> None:
    """Print a *list* of classification-shaped results (Milestone 91).

    `task="tabular_classification"`'s input is already-batched numeric data
    (`predict_tabular_classification_artifact()`'s own "one result per row,
    not one result overall" contract -- see that function's docstring), so
    `predict_model()` always returns a list here, unlike `task=
    "classification"`'s single-image `_print_classification_result()` above.
    The common case (one input row) prints identically to that function; a
    genuinely multi-row JSON input prints one numbered block per row instead
    of silently reporting only the first.
    """
    if as_json:
        predictions = []
        for result in results:
            if isinstance(result, ClassificationPrediction):
                predictions.append({"class": result.label, "confidence": result.confidence})
            else:
                predictions.append({"class": None, "index": result, "confidence": None})
        print(json.dumps({"task": task, "predictions": predictions}, indent=2))
        return

    multi = len(results) > 1
    for i, result in enumerate(results):
        prefix = f"Sample {i}: " if multi else ""
        if isinstance(result, ClassificationPrediction):
            print(f"{prefix}Prediction: {result.label}")
            print(f"{prefix}Confidence: {result.confidence:.1%}")
        else:
            print(f"{prefix}Prediction: class index {result} "
                  "(no class-name vocabulary was saved with this model)")


def cmd_predict(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.model):
        raise CLIError(f"artifact not found: {args.model}")

    # Milestone 88: the artifact's own persisted task metadata is the sole
    # routing signal -- see this module's own docstring for why a missing
    # task fails clearly here rather than falling back to
    # predict_model()'s architecture-based legacy guess.
    task = inspect_model(args.model).task
    if task is None:
        raise CLIError(
            "artifact does not declare a task.\n"
            "Use the task-specific prediction API or resave the model with task metadata."
        )

    if task == "sequence":
        # Milestone 90: unlike the other three tasks, the input is literal
        # seed text on the command line, not a file -- see this module's own
        # docstring for the char-level tokenization convention this applies.
        seed_tokens = list(args.input)
        if not seed_tokens:
            raise CLIError("sequence prediction requires a non-empty seed string.")
        generated = predict_model(args.model, seed_tokens, device=args.device, length=args.length)
        text = "".join(generated)
        if args.json:
            print(json.dumps({"task": "sequence", "seed": args.input, "generated": text}, indent=2))
        else:
            print(f"Generated: {text}")
        return 0

    if not os.path.isfile(args.input):
        raise CLIError(f"input file not found: {args.input}")

    if task == "classification":
        result = predict_model(args.model, args.input, device=args.device)
        _print_classification_result(result, args.json)
        return 0

    if task == "regression":
        input_data = _parse_numeric_input(args.input)
        result = predict_model(args.model, input_data, device=args.device)
        values = result.numpy().tolist()
        if args.json:
            print(json.dumps({"task": "regression", "prediction": values}, indent=2))
        else:
            print(f"Prediction: {values}")
        return 0

    if task == "tabular_classification":
        # Milestone 91: shares regression's "already-batched numeric JSON"
        # input parsing, but the result is a classification prediction
        # (label/confidence, or a raw index with no saved classes=) -- the
        # exact same output shape/printing `task == "classification"` above
        # already uses, since both delegate to _print_classification_result().
        input_data = _parse_numeric_input(args.input)
        results = predict_model(args.model, input_data, device=args.device)
        _print_classification_results(results, args.json, task="tabular_classification")
        return 0

    if task == "segmentation":
        if not args.output:
            raise CLIError("segmentation prediction requires --output <path> to save the predicted mask.")
        output_dir = os.path.dirname(os.path.abspath(args.output)) or "."
        if not os.path.isdir(output_dir):
            raise CLIError(f"cannot write to '{args.output}': directory '{output_dir}' does not exist.")

        result = predict_model(args.model, args.input, device=args.device)
        save_image(result, args.output)
        if args.json:
            print(json.dumps({"task": "segmentation", "output_path": args.output}, indent=2))
        else:
            print(f"Predicted mask saved to: {args.output}")
        return 0

    # Defensive only: inspect_model() already rejects any "task" value
    # outside forge.serialization.model.TASK_TYPES, so a real artifact can
    # never reach this branch today. Kept so a future TASK_TYPES addition
    # this command has not yet been taught to handle fails clearly here
    # rather than falling through silently.
    raise CLIError(f"unsupported artifact task: {task}")


# -- evaluate (Milestone 117) ---------------------------------------------------------------------------------
#
# Which tasks read `.npy`/`.csv` files and which read an ImageFolder directory. This is routing of *file formats* only
# (how to read INPUT/TARGETS) -- the same role the per-task branches of `cmd_predict()` play -- and nothing else
# about evaluation is decided here. A task in neither tuple has no evaluation semantics (`ArtifactPredictor.
# evaluate()` refuses it too) and is reported before any data is read.
_NUMERIC_EVALUATION_TASKS = ("tabular_classification", "regression")
_IMAGE_EVALUATION_TASKS = ("classification",)


def _is_csv(path: str) -> bool:
    """Whether INPUT is read as CSV: decided by the file extension alone, never by sniffing its content."""
    return os.path.splitext(path)[1].lower() == ".csv"


def _load_npy(path: str, role: str) -> np.ndarray:
    """Read one `.npy` array (`role` is "input" or "targets", for error messages).

    `allow_pickle=False`: an evaluation file is data, never code. A missing file, a directory, something that is
    not `.npy` at all (a CSV, a pickled object array), an `.npz` archive or a truncated `.npy` is one `CLIError`.
    Nothing about the *contents* is validated here -- shape, dtype, finiteness and label vocabulary belong to
    `ArtifactPredictor.evaluate()` and reach the user as its `DataError`.
    """
    if not os.path.isfile(path):
        raise CLIError(f"{role} file not found: {path}")
    try:
        with open(path, "rb") as fh:
            array = np.load(fh, allow_pickle=False)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        raise CLIError(f"{role} file '{path}' is not a readable .npy file (an array saved with numpy.save()).")
    if not isinstance(array, np.ndarray):
        if hasattr(array, "close"):
            array.close()  # an .npz archive: np.load returns a lazy container, not an array
        raise CLIError(f"{role} file '{path}' is not a single .npy array (an .npz archive was given).")
    return array


def _jsonable(value):
    """`value` as plain JSON types: NumPy arrays/scalars and tuples become lists / Python numbers."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _evaluation_payload(result: "ClassificationEvaluationResult | RegressionEvaluationResult") -> dict:
    """Every field of the result dataclass, verbatim -- no field list to keep in sync with `evaluation.py`."""
    return {f.name: _jsonable(getattr(result, f.name)) for f in dataclasses.fields(result)}


def _table(header: "list[str]", rows: "list[list[str]]") -> "list[str]":
    """Left-align the first column, right-align the rest, two spaces between columns."""
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    return [
        "  ".join(cell.ljust(w) if i == 0 else cell.rjust(w) for i, (cell, w) in enumerate(zip(row, widths)))
        for row in [header, *rows]
    ]


def _print_evaluation(result: "ClassificationEvaluationResult | RegressionEvaluationResult") -> None:
    print(f"Task: {result.task}")
    print(f"Samples: {result.samples}")
    if isinstance(result, ClassificationEvaluationResult):
        print(f"Accuracy: {result.accuracy:.2%}")
        print(f"Baseline accuracy: {result.baseline_accuracy:.2%}")
        print(f"Loss: {result.loss:.4f}")
        print()
        print("Classes:")
        rows = [
            [name, f"{p:.4f}", f"{r:.4f}", str(s)]
            for name, p, r, s in zip(result.classes, result.precision, result.recall, result.support)
        ]
        for line in _table(["class", "precision", "recall", "support"], rows):
            print(f"  {line}")
        print()
        print("Confusion matrix (rows = true class, columns = predicted class):")
        rows = [[name, *(str(int(v)) for v in row)] for name, row in zip(result.classes, result.confusion_matrix)]
        for line in _table(["", *result.classes], rows):
            print(f"  {line}")
    else:
        print(f"MSE: {result.mse:.6g}")
        print(f"MAE: {result.mae:.6g}")
        print(f"Baseline MSE: {result.baseline_mse:.6g}")
        print(f"Loss: {result.loss:.6g}")


def cmd_evaluate(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.model):
        raise CLIError(f"artifact not found: {args.model}")

    # The artifact's own persisted task decides how INPUT is read -- see this module's docstring, and
    # `cmd_predict()` for why a task-less (pre-Milestone-87) artifact is refused rather than guessed at.
    task = inspect_model(args.model).task
    if task is None:
        raise CLIError(
            "artifact does not declare a task.\n"
            "Resave the model with task metadata, or evaluate it through the Python API."
        )
    if task not in _NUMERIC_EVALUATION_TASKS + _IMAGE_EVALUATION_TASKS:
        raise CLIError(
            f"evaluation is not supported for task '{task}' -- it is defined only for "
            f"{', '.join(_IMAGE_EVALUATION_TASKS + _NUMERIC_EVALUATION_TASKS)} artifacts."
        )

    if task in _IMAGE_EVALUATION_TASKS:
        if args.target is not None:
            raise CLIError(
                "--target names a column of a .csv INPUT; an image-classification artifact reads its labels from "
                "the directory names (INPUT/<class_name>/<image>)."
            )
        if args.targets is not None:
            raise CLIError(
                "an image-classification artifact takes no TARGETS file: the labels are the directory names "
                "(INPUT/<class_name>/<image>)."
            )
        if not os.path.isdir(args.input):
            raise CLIError(
                f"input '{args.input}' is not a directory -- image evaluation needs root/<class_name>/<image files>."
            )
        features, targets = args.input, None
    elif _is_csv(args.input):
        # The file extension picks the reader (no content sniffing); the artifact's task only says what the
        # target column must hold. Everything after the two arrays exist is the same `evaluate()` call.
        if args.targets is not None:
            raise CLIError(
                "a .csv INPUT carries its own targets: name the target column with --target COLUMN instead of "
                "giving a TARGETS file."
            )
        if args.target is None:
            raise CLIError(f"a .csv INPUT needs --target COLUMN, the header name of its target column: "
                           f"forge model evaluate MODEL {args.input} --target COLUMN")
        features, targets = load_csv(
            args.input, target=args.target, labels=task == "tabular_classification",
        )
    else:
        if args.target is not None:
            raise CLIError(
                "--target names a column of a .csv INPUT; a .npy INPUT takes its targets as a TARGETS file."
            )
        if args.targets is None:
            raise CLIError(f"a {task} artifact needs a TARGETS file: forge model evaluate MODEL INPUT TARGETS")
        features, targets = _load_npy(args.input, "input"), _load_npy(args.targets, "targets")

    options = {} if args.batch_size is None else {"batch_size": args.batch_size}
    # One load, one evaluate call: everything else -- input check, preprocessing, target transform, class order,
    # metrics -- happens inside the predictor. Nothing is written anywhere.
    result = load_predictor(args.model, device=args.device).evaluate(features, targets, **options)

    if args.json:
        try:
            text = json.dumps(_evaluation_payload(result), indent=2, allow_nan=False)
        except ValueError:
            raise CLIError("evaluation produced a non-finite metric, which JSON cannot represent; run without --json.")
        print(text)
    else:
        _print_evaluation(result)
    return 0
