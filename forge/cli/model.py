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
line, not a file -- unlike the other three tasks, there is no file to
decode; `--length` (default 200) controls how many new tokens to generate.
This command tokenizes the seed as individual characters (`list(seed)`),
matching the char-level vocabulary convention `examples/char_rnn` uses --
the one real sequence workload this milestone built end-to-end. A
different tokenization convention (e.g. `examples/word_rnn`'s whole-word
vocabulary) is not something this command can infer from the artifact
alone; call `forge.predict_sequence_artifact()` directly with a
pre-tokenized seed list for that case.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..data import save_image
from ..serialization import inspect_model, load_classes, load_model, load_preprocessing, save_model
from ..training import ClassificationPrediction, predict_model
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
        "a JSON file of numeric data for regression, or literal seed text for sequence generation",
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
    print(f"Training: {'train' if training else 'eval'}")
    print(f"Preprocessing: {'yes' if has_preprocessing else 'no'}")
    if preprocessing_description is not None:
        print(f"Preprocessing detail: {preprocessing_description}")
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
    save_model(model, args.output, preprocessing=preprocessing, classes=classes, task=task)
    print(f"Converted '{args.model}' -> '{args.output}' (device={args.device}).")
    return 0


def _parse_regression_input(path: str) -> np.ndarray:
    """Parse a JSON file of numeric data into a batched NumPy array (Milestone 88).

    A flat list (`[1.2, 3.4, 5.6, 7.8]`) is treated as one unbatched sample
    and given a leading batch dimension; a nested list (`[[1.2, 3.4], [5.6,
    7.8]]`) is treated as already batched and passed through as-is -- the
    same convention documented in `predict_tensor_artifact()`'s docstring for
    a raw NumPy array. Anything that is not valid JSON, or whose values are
    not all numeric, raises `CLIError` with one clear message -- covering
    malformed JSON, non-numeric JSON, and a non-JSON file (e.g. an image)
    given where a regression artifact expects numeric input, all identically.
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
        raise CLIError("regression input must contain numeric JSON data.")

    try:
        array = np.array(raw, dtype=np.float32)
    except (TypeError, ValueError):
        raise CLIError("regression input must contain numeric JSON data.")

    if array.dtype == object or array.size == 0:
        raise CLIError("regression input must contain numeric JSON data.")

    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim != 2:
        raise CLIError("regression input must be a flat list or a 2-D list of numbers.")

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
        input_data = _parse_regression_input(args.input)
        result = predict_model(args.model, input_data, device=args.device)
        values = result.numpy().tolist()
        if args.json:
            print(json.dumps({"task": "regression", "prediction": values}, indent=2))
        else:
            print(f"Prediction: {values}")
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
