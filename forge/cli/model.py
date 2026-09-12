"""`forge model inspect` / `forge model convert` / `forge model predict`.

`inspect` reads archive metadata only (`forge/cli/_archive_info.py`) -- it
never reconstructs a live `Module`, never requires CUDA, and never mutates
anything. Its "Preprocessing detail" line (Milestone 85) delegates to
`forge.inspect_model()` -- the public, structured inspection API -- rather
than re-deriving a preprocessing description of its own; the rest of this
command's per-module/per-parameter dump remains this file's own lower-level
archive walk (`_archive_info.py`), which stays intentionally more detailed
than `inspect_model()`'s product-level `ModelInfo` contract. `convert` is a real device conversion and goes straight through
`forge.load_model()` / `forge.save_model()`, exactly as a Python caller
would -- no separate conversion logic lives here. `predict` (Milestone 72)
is a thin CLI wrapper over `forge.training.predict_artifact()` (Milestone
82) -- the same "`load_model()` / `load_preprocessing()` / `load_classes()`
/ decode image / `predict()` / `interpret_classification()`" sequence
`examples/image_folder_classification/infer.py` and this command
independently hand-wrote until Milestone 82 extracted it into one function
both now share. Scoped to a single image file: Forge has no generic "input
format" concept spanning its example workloads (images, tabular rows, raw
sequences all shape differently), so this command only claims the one
concrete input shape a saved artifact can already fully describe end-to-end.
"""

from __future__ import annotations

import argparse
import json
import os

from ..serialization import inspect_model, load_classes, load_model, load_preprocessing, save_model
from ..training import ClassificationPrediction, predict_artifact
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
        "predict", help="Classify one image with a saved model (requires preprocessing=... at save time)"
    )
    predict_parser.add_argument("model", help="Path to a model file saved with forge.save_model()")
    predict_parser.add_argument("--image", required=True, help="Path to one image file to classify")
    predict_parser.add_argument(
        "--device", default=None, choices=["cpu", "cuda"],
        help="Device to load the model onto (default: whatever device it was saved from)",
    )
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
    info = inspect_model(args.path)
    preprocessing_description = info.preprocessing.description if info.preprocessing is not None else None

    if args.json:
        payload = {
            "path": args.path,
            "format_version": metadata["forge_format_version"],
            "device": metadata["device"],
            "training": "train" if training else "eval",
            "has_preprocessing": has_preprocessing,
            "preprocessing_description": preprocessing_description,
            "classes": classes,
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
    # Preserve preprocessing (Milestone 71)/classes (Milestone 72) metadata
    # across a device conversion -- a converted file is still meant to be a
    # complete, self-describing artifact, not a bare weights-only copy.
    preprocessing = load_preprocessing(args.model)
    classes = load_classes(args.model)
    save_model(model, args.output, preprocessing=preprocessing, classes=classes)
    print(f"Converted '{args.model}' -> '{args.output}' (device={args.device}).")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.model):
        raise CLIError(f"Cannot predict with model '{args.model}': file not found.")
    if not os.path.isfile(args.image):
        raise CLIError(f"Cannot predict: image '{args.image}' not found.")

    # Milestone 82: forge.training.predict_artifact() is the same
    # "load_preprocessing() -> load_model() -> decode image -> preprocess ->
    # predict() -> load_classes() -> interpret_classification()" sequence
    # this command used to hand-roll -- see this module's own docstring.
    result = predict_artifact(args.model, args.image, device=args.device)
    if isinstance(result, ClassificationPrediction):
        print(f"Predicted class: {result.label}")
        print(f"Confidence: {result.confidence:.1%}")
    else:
        print(f"Predicted class index: {result}")
        print("(No class-name vocabulary was saved with this model -- see "
              "forge.save_model(..., classes=...) -- so only the raw index is available.)")
    return 0
