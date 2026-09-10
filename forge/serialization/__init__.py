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

`save_model(..., preprocessing=...)`/`load_preprocessing()` (Milestone 71)
optionally save and reconstruct the `forge.data.transforms.Transform`
pipeline a model's inputs are expected to already have gone through, as a
JSON-safe sibling entry in the same model file -- see
`forge.serialization.transforms`'s module docstring and
`docs/architecture/persistence.md`'s **Preprocessing metadata** section.

`save_model(..., classes=...)`/`load_classes()` (Milestone 72) optionally
save and reconstruct a classification model's ordered class-name
vocabulary the same way -- see `forge.training.interpret_classification()`
for turning a raw prediction `Tensor` plus this vocabulary into a
human-readable label.
"""

from .checkpoint import Checkpoint, CHECKPOINT_FORMAT_VERSION, load_checkpoint, save_checkpoint
from .model import load_classes, load_model, load_preprocessing, save_model
from .optimizer_registry import register_optimizer
from .registry import register_module
from .transforms import register_transform

__all__ = [
    "save_model", "load_model", "load_preprocessing", "load_classes", "register_module",
    "save_checkpoint", "load_checkpoint", "Checkpoint", "CHECKPOINT_FORMAT_VERSION",
    "register_optimizer", "register_transform",
]
