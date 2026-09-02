"""Model + checkpoint persistence (Milestones 7, 13, 18).

`save_model()`/`load_model()` serialize a `forge.nn.Module` tree to/from a
structured, inspectable archive -- see `docs/architecture/persistence.md`
for the file format, versioning, and security/trust model. Reconstruction
goes through an explicit registry of supported module types
(`register_module()`); Forge's built-in module types are pre-registered,
and a custom `Module` subclass must be registered before it can be
saved/loaded. Nothing in this package executes arbitrary code found in a
model file.

`save_checkpoint()`/`load_checkpoint()` (Milestone 18) serialize *training*
state -- model state plus optimizer type/hyperparameters/per-parameter
state plus training progress -- in a separate, independently versioned
format; see `forge.serialization.checkpoint`'s module docstring.
Reconstruction goes through the same explicit-registry principle
(`optimizer_registry.register_optimizer()`); `SGD`/`Adam` are pre-registered.
"""

from .checkpoint import Checkpoint, CHECKPOINT_FORMAT_VERSION, load_checkpoint, save_checkpoint
from .model import load_model, save_model
from .optimizer_registry import register_optimizer
from .registry import register_module

__all__ = [
    "save_model", "load_model", "register_module",
    "save_checkpoint", "load_checkpoint", "Checkpoint", "CHECKPOINT_FORMAT_VERSION",
    "register_optimizer",
]
