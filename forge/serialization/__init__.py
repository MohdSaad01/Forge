"""Model persistence (Milestone 7).

`save_model()`/`load_model()` serialize a `forge.nn.Module` tree to/from a
structured, inspectable archive -- see `docs/architecture/persistence.md`
for the file format, versioning, and security/trust model. Reconstruction
goes through an explicit registry of supported module types
(`register_module()`); Forge's built-in `Linear`/`ReLU` are pre-registered,
and a custom `Module` subclass must be registered before it can be
saved/loaded. Nothing in this package executes arbitrary code found in a
model file.
"""

from .model import load_model, save_model
from .registry import register_module

__all__ = ["save_model", "load_model", "register_module"]
