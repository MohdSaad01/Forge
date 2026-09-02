"""`forge checkpoint inspect` / `forge checkpoint convert`.

`inspect` reads archive metadata only (`forge/cli/_archive_info.py`) --
deliberately never `forge.load_checkpoint()`, since that function's documented
contract includes overwriting `forge.random`'s process-global RNG state, a
side effect an inspect command must never trigger. `convert` is a real
device conversion and goes straight through `forge.load_checkpoint()` /
`forge.save_checkpoint()`, exactly as a Python caller would.
"""

from __future__ import annotations

import argparse
import json
import os

from ..serialization import load_checkpoint, save_checkpoint
from ._archive_info import count_elements, read_checkpoint_metadata, walk_modules, walk_parameters
from .errors import CLIError


def add_parser(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser("checkpoint", help="Inspect or convert a saved Forge training checkpoint")
    sub = parser.add_subparsers(dest="checkpoint_command", required=True)

    inspect_parser = sub.add_parser("inspect", help="Report a saved checkpoint's model, optimizer, and progress")
    inspect_parser.add_argument("path", help="Path to a checkpoint file saved with forge.save_checkpoint()")
    inspect_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text")
    inspect_parser.set_defaults(func=cmd_inspect)

    convert_parser = sub.add_parser("convert", help="Load a checkpoint onto a device and save it back out")
    convert_parser.add_argument("checkpoint", help="Path to a checkpoint file saved with forge.save_checkpoint()")
    convert_parser.add_argument(
        "--device", required=True, choices=["cpu", "cuda"],
        help="Target device to load and re-save the checkpoint on -- always explicit, never a fallback.",
    )
    convert_parser.add_argument("--output", required=True, help="Path to write the converted checkpoint to")
    convert_parser.set_defaults(func=cmd_convert)


def cmd_inspect(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.path):
        raise CLIError(f"Cannot inspect checkpoint '{args.path}': file not found.")

    metadata = read_checkpoint_metadata(args.path)
    model_root = metadata["model"]
    optimizer_meta = metadata["optimizer"]
    progress = metadata["training_progress"]
    rng_meta = metadata.get("rng")
    rng_present = isinstance(rng_meta, dict) and "default_generator_state" in rng_meta

    modules = list(walk_modules(model_root))
    parameters = list(walk_parameters(model_root))
    total_params = sum(count_elements(meta.get("shape", [])) for _, meta in parameters)

    opt_type = optimizer_meta.get("type")
    opt_config = optimizer_meta.get("config", {}) or {}
    opt_params = optimizer_meta.get("parameters", []) or []
    opt_param_state = optimizer_meta.get("param_state", {}) or {}

    if args.json:
        payload = {
            "path": args.path,
            "format_version": metadata["forge_checkpoint_format_version"],
            "device": metadata["device"],
            "epoch": progress.get("epoch"),
            "global_step": progress.get("global_step"),
            "rng_state_present": rng_present,
            "optimizer": {"type": opt_type, "config": opt_config},
            "modules": [{"name": name, "type": type_name} for name, type_name in modules],
            "parameters": [
                {"name": name, "shape": list(meta.get("shape", [])), "dtype": meta.get("dtype")}
                for name, meta in parameters
            ],
            "total_parameters": total_params,
            "optimizer_parameters_total": len(opt_params),
            "optimizer_parameters_with_state": len(opt_param_state),
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Checkpoint: {args.path}")
    print(f"Format version: {metadata['forge_checkpoint_format_version']}")
    print(f"Device: {metadata['device']}")
    print(f"Epoch: {progress.get('epoch')}")
    print(f"Global step: {progress.get('global_step')}")
    print(f"RNG state: {'present' if rng_present else 'absent'}")
    print()
    print(f"Optimizer: {opt_type}")
    for key, value in opt_config.items():
        print(f"  {key}: {value}")
    print()
    print("Model:")
    for name, type_name in modules:
        depth = 0 if name == "(root)" else name.count(".") + 1
        print(f"  {'  ' * depth}{name}: {type_name}")
    print()
    print(f"Optimizer parameters: {len(opt_params)} total, {len(opt_param_state)} with saved state")
    print(f"Total parameters: {total_params}")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.checkpoint):
        raise CLIError(f"Cannot convert checkpoint '{args.checkpoint}': file not found.")
    output_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    if not os.path.isdir(output_dir):
        raise CLIError(f"Cannot write to '{args.output}': directory '{output_dir}' does not exist.")

    checkpoint = load_checkpoint(args.checkpoint, device=args.device)
    save_checkpoint(
        args.output,
        checkpoint.model,
        checkpoint.optimizer,
        epoch=checkpoint.epoch,
        global_step=checkpoint.global_step,
        extra=checkpoint.extra,
    )
    print(f"Converted '{args.checkpoint}' -> '{args.output}' (device={args.device}).")
    return 0
