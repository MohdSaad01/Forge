"""`forge model inspect` / `forge model convert` / `forge model predict`.

`inspect` reads archive metadata only (`forge/cli/_archive_info.py`) -- it
never reconstructs a live `Module`, never requires CUDA, and never mutates
anything. `convert` is a real device conversion and goes straight through
`forge.load_model()` / `forge.save_model()`, exactly as a Python caller
would -- no separate conversion logic lives here. `predict` (Milestone 72)
is a thin CLI wrapper over `forge.load_model()` / `forge.load_preprocessing()`
/ `forge.load_classes()` / `forge.predict()` /
`forge.interpret_classification()` -- the exact same sequence
`examples/image_folder_classification/infer.py` already demonstrates as a
Python script, exposed as a command now that a `.forge` file can carry
everything that sequence needs (preprocessing since Milestone 71, class
labels since Milestone 72). Scoped to a single image file: Forge has no
generic "input format" concept spanning its example workloads (images,
tabular rows, raw sequences all shape differently), so this command only
claims the one concrete input shape a saved artifact can already fully
describe end-to-end.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ..data.image_folder import ImageFolder
from ..serialization import load_classes, load_model, load_preprocessing, save_model
from ..training import interpret_classification, predict
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

    if args.json:
        payload = {
            "path": args.path,
            "format_version": metadata["forge_format_version"],
            "device": metadata["device"],
            "training": "train" if training else "eval",
            "has_preprocessing": has_preprocessing,
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

    preprocessing = load_preprocessing(args.model)
    if preprocessing is None:
        raise CLIError(
            f"'{args.model}' was saved with no preprocessing configuration (see "
            "forge.save_model(..., preprocessing=...)) -- 'forge model predict' has no "
            "automatic way to prepare the input image for this model."
        )
    model = load_model(args.model, device=args.device)

    raw = ImageFolder._load_image(Path(args.image))
    prepared = preprocessing(raw)
    batch = prepared.reshape(1, *prepared.shape)
    output = predict(model, batch)

    classes = load_classes(args.model)
    if classes is not None:
        result = interpret_classification(output, classes)[0]
        print(f"Predicted class: {result.label}")
        print(f"Confidence: {result.confidence:.1%}")
    else:
        predicted_idx = int(np.argmax(output.numpy(), axis=1)[0])
        print(f"Predicted class index: {predicted_idx}")
        print("(No class-name vocabulary was saved with this model -- see "
              "forge.save_model(..., classes=...) -- so only the raw index is available.)")
    return 0
