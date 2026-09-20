# Model Persistence (Milestone 7; CUDA persistence in Milestone 13; training checkpoints in Milestone 18)

## Package layout
```
forge/
    serialization/
        registry.py             ModuleSpec, register_module(), spec_for_class/spec_for_name,
                                 built-in registration of Linear/ReLU/Conv2d/MaxPool2d/Sequential/Flatten/Dropout
        optimizer_registry.py   OptimizerSpec, register_optimizer(), spec_for_class/spec_for_name,
                                 built-in registration of SGD/Adam (Milestone 18)
        transforms.py            TransformSpec, register_transform(), serialize_transform()/
                                 deserialize_transform() -- preprocessing-transform configuration (Milestone 71)
        archive.py               write_archive/read_archive -- the generic ZIP(json + .npy) file format
        model.py                  save_model(), load_model(), load_preprocessing(), load_classes(),
                                 inspect_model() -- tree walk, validation, reconstruction (Milestone 85: read-only
                                 summary; Milestone 87: explicit task= metadata, TASK_TYPES)
        checkpoint.py             save_checkpoint(), load_checkpoint() -- training-state persistence (Milestone 18)
```
`forge.serialization` is exposed as a submodule of `forge`, alongside
`forge.nn`/`forge.optim`/`forge.data`/`forge.training`. `save_model`,
`load_model`, `save_checkpoint`, `load_checkpoint`, `Checkpoint`,
`inspect_model`, `ModelInfo`, and `PersistenceError` are also exposed at the
top level (`forge.save_model`, `forge.load_model`, `forge.save_checkpoint`,
`forge.load_checkpoint`, `forge.Checkpoint`, `forge.inspect_model`,
`forge.ModelInfo`, `forge.PersistenceError`).

## Public API
```python
forge.save_model(model, path)
loaded = forge.load_model(path)
loaded = forge.load_model(path, device="cpu")    # explicit override (M13)
loaded = forge.load_model(path, device="cuda")   # explicit override (M13)
```
Free functions rather than `Module.save()`/`Module.load()` methods --
persistence is a distinct concern layered *over* `Module`, not a
responsibility `Module` itself needs to know about (matching the
architecture's "preserve clear boundaries... between... serialization...
and backends" rule). `Module` gained no new methods or state for this
milestone.

## What gets saved
```text
model metadata (format version, device)
    v
module type / configuration      (forge.serialization.registry)
    v
child modules (recursive, by attribute name)
    v
parameter state (name, shape, dtype, requires_grad, values)
    v
buffer state (name, shape, dtype, values -- Milestone 53)
```
Concretely, for each module in the tree (self, then every `_modules` child,
recursively): its registered type name, its architecture **configuration**
(constructor keyword arguments -- not weights), its `.training` flag, its
own `_parameters`' shapes/dtypes/`requires_grad`/values, and (as of
Milestone 53) its own `_buffers`' shapes/dtypes/values. Parameter and buffer
*names* are the dotted path used elsewhere in Forge (`fc1.weight`,
`bn1.running_mean`), matching `Module.named_parameters()`/`named_buffers()`.

**Not saved:** `.grad` on any parameter, any autograd graph (`grad_fn`),
and optimizer state -- see **Autograd state** and **Optimizer state** below.

## Buffer state (Milestone 53)
Buffers (`Module.register_buffer()`, `docs/architecture/modules.md`) round
-trip the same way parameters do -- a values array plus shape/dtype metadata
per dotted name -- just with no `requires_grad` field (a buffer is never
differentiable, enforced at registration). A buffer registered as `None`
(an unset optional buffer) is recorded as `null` in the metadata with no
array, so loading can tell "no buffer data" apart from "buffer data
present" without guessing. `FORMAT_VERSION` was bumped `1 -> 2` for this
change (a new required `"buffers"` key per module node) -- per this
document's own **Versioning** policy, a version bump is a deliberate,
documented breaking change: a Forge build from before Milestone 53 cannot
load a Milestone-53-or-later archive and vice versa, exactly like every
other `FORMAT_VERSION` bump. `CHECKPOINT_FORMAT_VERSION` was bumped
`1 -> 2` in lockstep, since a checkpoint's embedded model node has the same
new shape.

## Architecture reconstruction: the module registry
Forge does not serialize Python callables, class paths, or constructor
code. A saved file's `"type"` field is a plain string (e.g. `"Linear"`)
that is used **only** as a lookup key into
`forge.serialization.registry`'s in-process registry
(`forge/serialization/registry.py`) -- never `eval`'d, never resolved via
dynamic import or attribute lookup. If the key is not registered, loading
fails with `PersistenceError` before anything is constructed.

```python
forge.serialization.register_module(
    type_name="Linear",
    cls=Linear,
    get_config=lambda m: {
        "in_features": m.in_features,
        "out_features": m.out_features,
        "bias": m.bias is not None,
    },
)
```
`Linear` and `ReLU` are registered this way at import time --
Forge's only two built-in supported module types as of this milestone.
`get_config(instance)` extracts a JSON-safe configuration dict (never
weights); `from_config(config)` (defaulting to `cls(**config)` if not
given explicitly) reconstructs a **bare** instance from that dict, which
`load_model()` then overwrites with the file's saved parameter values.

**RNG isolation (Milestone 81).** A "bare" instance is still constructed
through the class's ordinary `__init__` -- for `Linear`/`Conv2d`/`Embedding`/
etc. this draws a real initial-weights sample from `forge.random.
default_generator()`, discarded a moment later when `load_model()` overwrites
it with the archived values. That draw still *advances* the global
generator, though, which is an observable side effect a caller reloading a
model mid-run (e.g. `forge.training.save_and_verify()`) never asked for and
would not expect -- found when it silently broke a resume-equivalence
guarantee in `examples/image_folder_classification/train.py` (see
`docs/development/m81-train-to-verified-artifact-workflow.md`). `load_model()`
now snapshots `forge.random.get_state()` before reconstructing the tree and
restores it in a `finally` immediately after (the same mechanism `forge.
serialization.checkpoint` uses for exact training resume, **RNG /
determinism policy** below) -- reconstruction's wasted draws never escape
`load_model()`'s own call.

### Custom/composite modules
A hand-written `Module` subclass such as
```python
class MLP(Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.fc1 = Linear(in_features, hidden)
        self.relu = ReLU()
        self.fc2 = Linear(hidden, out_features)
```
is **not** automatically persistable -- Forge has no general mechanism for
reflecting a `__init__`'s constructor signature back out of an instance,
and building one would mean either executing arbitrary code paths or
guessing at a class's construction contract, both explicitly out of scope
for M7. Such a class must opt in with the same `register_module()` call
shown above (see `tests/test_serialization.py::MLP` for a complete
example); an unregistered type raises `PersistenceError` immediately,
during `save_model()`, with the offending class's fully-qualified name --
never a silent fallback to executing anything found in a file. This is the
"explicit registry" mechanism described in the milestone brief: reflection
is deliberately avoided in favor of an explicit, auditable opt-in list.
Because `from_config` for a registered composite type is a real call to
that class's own `__init__`, its own children (`fc1`, `relu`, `fc2` above)
are recreated automatically as part of that call; `load_model()` then
recurses into the file's saved `"children"` metadata and replaces each one
with its own freshly reconstructed, correctly-valued instance.

## Parameter/tensor state
Each parameter is saved with: dotted name, shape, Forge dtype name (e.g.
`"float32"`), `requires_grad`, and its numeric values. On load, every
saved value is validated against its own declared shape/dtype **and**
against the shape/dtype the array actually deserializes to, before being
wrapped in a fresh `Parameter(array, dtype=..., device="cpu",
requires_grad=...)` and attached to the reconstructed module by name --
mismatches raise `PersistenceError` identifying the exact parameter and
the discrepancy (see **Errors** below).

## Device semantics
As of Milestone 13, a model may be saved from and loaded onto either
`"cpu"` or `"cuda"` (see `docs/architecture/cuda-backend.md`'s **CUDA model
persistence** section for the CUDA-specific mechanics). `Module.device`
(Milestone 9) already requires a device-coherent module tree -- `save_model()`
calls it once up front, so a manually-assembled mixed-device tree raises
`ModuleError` before anything is written, exactly as `Module.device` itself
does; a module tree with no `Parameter`s anywhere records `"cpu"`.

### Recorded metadata
The archive's top-level `"device"` field records the whole tree's device at
save time (`"cpu"` or `"cuda"`) -- Forge assumes one coherent device per
model, matching `Module.device`'s own contract, so no per-parameter device
field is needed. A file whose `"device"` is anything else (an unrecognized
string, or a tampered file) raises `PersistenceError` before any module is
constructed.

### Loading policy
```python
load_model(path)                 # restore onto the recorded device, if available
load_model(path, device="cpu")   # explicit override: always available
load_model(path, device="cuda")  # explicit override: requires CUDA
```
- **`device=None` (default).** Restore onto the device recorded in the
  archive -- but *only* when that device is actually available right now.
  A `"cuda"`-recorded file loaded with no CUDA backend present raises
  `PersistenceError` explaining that CUDA is required; Forge never silently
  falls back to CPU or pretends CUDA execution occurred. `is_cuda_available()`
  is checked lazily -- `forge.backend.cuda` is only imported at all when the
  recorded (or requested) device is `"cuda"`, so a CPU-only environment
  loading a CPU-recorded file never touches CUDA in any way.
- **`device="cpu"`.** Explicit override: always succeeds regardless of the
  recorded device (a deliberate CUDA -> CPU conversion at load time). Every
  saved parameter's bytes are already host-resident in the archive (see
  **Parameter/tensor state** below), so this override needs no CUDA backend
  at all -- it works even on a machine with no CUDA toolchain.
- **`device="cuda"`.** Explicit override: restores onto CUDA regardless of
  the recorded device (a deliberate CPU -> CUDA conversion). Still requires
  CUDA to actually be available -- an unavailable explicit `device="cuda"`
  fails with the same clear `PersistenceError`, never a silent CPU fallback.
- **Any other `device` value** (not `None`, `"cpu"`, or `"cuda"`) raises
  `PersistenceError` immediately.

Every parameter in the reconstructed tree lands on the same resolved target
device -- `load_model()` never produces a mixed-device tree, matching the
same coherence `Module.device`/`save_model()` require.

### How a CUDA `Parameter` is saved and restored
```text
Saving:   CUDA Parameter -> Backend.to_numpy() (device-to-host copy) -> .npy/archive
Loading:  .npy/archive -> host NumPy array -> Parameter(array, device="cuda", ...)
                                                       |
                                                       v
                                        CUDABackend.from_array(): real
                                        cudaMalloc + host-to-device memcpy
```
`Backend.to_numpy()`/`Parameter(..., device=...)` are the same primitives
`Tensor.to()`/`Module.to()` already use elsewhere in Forge -- persistence
introduces no second CUDA transfer code path, and no CUDA `Parameter` is
ever produced by relabeling a NumPy array. These are *persistence
transfers*: `save_model()`/`load_model()` never run a forward or backward
pass, on CPU or CUDA, as part of saving or loading -- see
`tests/test_cuda_persistence.py::test_save_load_cuda_model_never_calls_cpu_backend_compute_ops`
for the structural proof (no CPU-side computation occurs either, even for a
CUDA model's save/load).

## Autograd semantics
A saved model captures **state**, not a computation graph: no `grad_fn`,
no `.grad`, no reference to whatever forward/backward pass happened to be
in flight when `save_model()` was called. Every parameter `load_model()`
produces is a fresh leaf `Parameter` (`is_leaf is True`, `grad_fn is
None`, `.grad is None`), exactly as if it had just been constructed
directly. A subsequent forward pass through the loaded model builds an
entirely new autograd graph under the same rules as any other Forge model
(`docs/architecture/autograd.md`); nothing about having been loaded from a
file makes the graph behave differently.

## Training/evaluation mode semantics
**Decision:** each module's `.training` flag, as it existed at save time,
is recorded per-module and restored exactly on load -- `load_model()` does
not force the result into `eval()` or `train()` regardless of what was
saved. This is a deliberate, per-module (not merely per-tree) record:
`Module.eval()`/`train()` normally propagate uniformly to every
descendant, but nothing prevents a caller from later calling `.train()`/
`.eval()` on an individual child module afterward, producing a tree with
divergent modes -- persistence preserves that exact per-module state
rather than assuming uniformity. Callers who want a predictable inference
mode should call `loaded_model.eval()` explicitly after `load_model()`
(the common case, since most saved models exist for inference), exactly
as they would call it on any freshly constructed model before running it
without training semantics.

## Optimizer-state limitations
`save_model()`/`load_model()` persist model (architecture + parameter)
state only. No `Optimizer` state -- learning rate, momentum buffers, step
count, or anything else an optimizer might accumulate -- is saved or
restored, and this is deliberate, not merely an unimplemented gap: a
`load_model()`-ed model is meant for inference, and `save_model()` must stay
optimizer-state-free even now that checkpointing exists (see
**Checkpointing** below and its **Isolation from `save_model()`**
subsection) -- the two APIs serve different purposes and their formats are
not merged. Training-resume with identical optimizer dynamics is
`save_checkpoint()`/`load_checkpoint()`'s job, a distinct capability with
its own format/versioning (Milestone 18).

## Checkpointing (Milestone 18)
`forge.serialization.checkpoint` (`save_checkpoint()`/`load_checkpoint()`,
also exposed as `forge.save_checkpoint`/`forge.load_checkpoint`) adds a
second, separate persistence API for *training* state, layered on the same
`forge/serialization/archive.py` ZIP(json + `.npy`) primitives `save_model`/
`load_model` use, but in its own independently versioned format:
```text
save_model()      -> model state only          (this file's sections above)
save_checkpoint() -> training state             (this section)
                       = model state (identical tree walk to save_model())
                       + optimizer type + hyperparameters + per-parameter state
                       + training progress (epoch, global_step)
                       + Forge's default RNG state
                       + caller-supplied JSON-safe `extra`
```

### Public API
```python
forge.save_checkpoint(path, model, optimizer, epoch=0, global_step=0, extra=None)
checkpoint = forge.load_checkpoint(path)                # restore recorded device
checkpoint = forge.load_checkpoint(path, device="cpu")  # explicit override
checkpoint = forge.load_checkpoint(path, device="cuda") # explicit override
# checkpoint.model, checkpoint.optimizer, checkpoint.epoch,
# checkpoint.global_step, checkpoint.extra
```
`Trainer.save_checkpoint(path, extra=None)` / `Trainer.resume(checkpoint)`
are thin wrappers around these two functions using the Trainer's own
`model`/`optimizer`/`epoch`/`global_step` -- see
`docs/architecture/training-engine.md`'s **Checkpointing and resume**
section for the Trainer-level contract; nothing below duplicates that.

### Parameter-identity association
`Adam`'s runtime state (`forge/optim/adam.py`) keys `optimizer.state` by
`Parameter` **object identity** -- which archive serialization cannot
preserve. `save_checkpoint()` therefore records, alongside each
optimizer-owned parameter's state, its dotted name from
`model.named_parameters()` (the same naming `save_model()` already uses),
plus the ordered list of dotted names the optimizer was constructed from
(so an optimizer covering only *some* of `model`'s parameters -- e.g. a
frozen layer excluded on purpose -- round-trips correctly, not just the
common "optimizer over every parameter" case). `load_checkpoint()`
reconstructs the model first (an ordinary `_build_load_node` tree walk,
identical to `load_model()`'s), giving a fresh name -> `Parameter` mapping;
builds the optimizer from that same ordered name list; and re-associates
each parameter's state by name, installing it into the *new* optimizer's
`state` dict by the *new* `Parameter` objects' identity. Adam's own
identity-keyed lookup mechanism is never changed -- only bridged across the
archive boundary. An optimizer holding a `Parameter` not reachable from the
given `model`'s tree raises `PersistenceError` at save time, since state can
only be associated by a name relative to that model.

### Optimizer registry
Exactly the same principle as the module registry (**Architecture
reconstruction** above): `load_checkpoint()` never instantiates an
optimizer class from a string read out of the archive.
`forge.serialization.optimizer_registry` maps a stable `type_name` (e.g.
`"Adam"`) to an `OptimizerSpec` bundling `get_config`/`from_config` (JSON-safe
hyperparameters, and reconstruction from them) plus `get_param_state`/
`set_param_state` (per-parameter state extraction/installation, using
`get_backend(...).to_numpy()`/`from_array()` for any array-valued field --
never a NumPy array relabeled as CUDA storage). `SGD` and `Adam` are
registered by default (`weight_decay 0.0` counts as an explicit
hyperparameter here too -- never silently substituted); any other optimizer
type needs `register_optimizer()` before it can be checkpointed, the same
opt-in requirement `register_module()` places on custom `Module` subclasses.
An unregistered/unknown `"optimizer"."type"` string raises
`PersistenceError` before anything is constructed -- never `eval`, dynamic
import, or `pickle`.

### Device semantics
Identical policy to `load_model()`'s (**Device semantics** above):
`device=None` restores the recorded device if available, `device="cpu"`/
`"cuda"` explicitly convert (still requiring CUDA to actually be present for
a `"cuda"` target), and an unrecognized value raises `PersistenceError`
immediately. Both the reconstructed model's `Parameter`s and the
reconstructed optimizer's per-parameter arrays (Adam's `m`/`v`) always end
up on the *same* resolved target device -- restoring onto `"cuda"` allocates
genuine `CUDAStorage` for both, via the same `Backend.from_array()`
primitive `load_model()` already uses for parameters.

### RNG / determinism policy -- **Policy A**
A checkpoint captures `forge.random.get_state()` (the process-global default
generator's exact bit-generator state, JSON-safe) and `load_checkpoint()`
restores it via `forge.random.set_state()`. This is the one generator
`Dropout` draws from by default -- directly on CPU, and as the host-side
per-call seed for `CUDABackend.dropout_mask` on CUDA
(`docs/architecture/cuda-backend.md`'s **CUDA Dropout** section) -- so
restoring it reproduces the exact sequence of future Dropout draws (CPU mask
values, or CUDA per-call seeds) that would have followed at save time.
Training resumed from a checkpoint is therefore bitwise-deterministic for a
model whose only randomness is default-generator `Dropout`, given the same
sequence of subsequent operations.

**Not covered:** a `Dropout(..., generator=...)` built with an explicit
`numpy.random.Generator`, or any other caller-owned generator (e.g. a
`DataLoader`'s own `shuffle` generator) -- neither lives inside any
`Module`/`Optimizer` state a checkpoint inspects; reproducing those remains
the caller's own responsibility. Forge does not build a general-purpose
RNG-tracking framework for this (`forge/random.py` stays a single default
generator plus `get_state()`/`set_state()`).

**Worked example (Milestone 65):** `save_checkpoint(..., extra=...)`'s
existing caller-defined JSON-safe dict is sufficient for a caller to fulfill
this responsibility itself, with no framework change -- `numpy.random.
Generator.bit_generator.state` is already a JSON-safe dict, so a caller can
save it into `extra` and restore it into the same generator object on
resume. `examples/regression/train.py` does exactly this for its
`DataLoader`'s shuffle generator; see
`docs/development/m65-reproducible-training.md`.

### Training progress
`epoch`/`global_step` are plain non-negative integers the caller supplies
(`Trainer.save_checkpoint()` fills them in from `Trainer.epoch`/
`Trainer.global_step`, its own persistent progress counters -- see
`docs/architecture/training-engine.md`). No distributed/multi-worker state,
no automatic periodic scheduling, no early-stopping bookkeeping -- Forge
does not invent a training-state model beyond these two counters.
`TrainingHistory`/`EpochResult` are explicitly **not** part of the
checkpoint contract: they are a record of past `fit()` calls, not state
required to correctly continue training, so nothing about them is
serialized; a resumed `fit()` call returns a fresh `TrainingHistory`
starting at the checkpoint's epoch. `extra` is an optional, caller-supplied
JSON-safe dict for any small additional state (e.g. best validation loss so
far) -- validated as JSON-safe at save time, never arbitrary Python objects.

### Isolation from `save_model()`
`save_model()` is completely unaffected by this milestone: it still writes
no `"optimizer"`, `"training_progress"`, `"rng"`, or
`"forge_checkpoint_format_version"` key, and a checkpoint file is not a
valid `save_model()`/`load_model()` file (their format-version keys --
`"forge_format_version"` vs. `"forge_checkpoint_format_version"` -- are
different fields entirely, checked independently). This split is
regression-tested directly (`tests/test_checkpoint.py`).

### File layout
Checkpoints reuse `forge/serialization/archive.py`'s generic
`write_archive`/`read_archive` (the same ZIP(json + `.npy`) primitives
`save_model`/`load_model` use, generalized in Milestone 18 to accept
arbitrary caller-chosen array paths rather than a single hard-coded
`parameters/` directory):
```text
metadata.json                                        -- everything above, as JSON
model/parameters/<dotted.parameter.name>.npy          -- model parameter values
optimizer/state/<dotted.parameter.name>/<field>.npy   -- per-parameter optimizer state (e.g. m, v)
```
`save_model()`/`load_model()` still produce/consume the original
`parameters/<dotted.name>.npy` layout unchanged (`model.py` adds/strips that
prefix itself around the shared, now-generic archive functions) -- existing
model files remain fully readable.

### Versioning
`"forge_checkpoint_format_version"` (`forge.serialization.checkpoint.CHECKPOINT_FORMAT_VERSION`,
currently `2` -- bumped `1 -> 2` in Milestone 53 alongside `FORMAT_VERSION`,
since a checkpoint's embedded model node gained the same new `"buffers"`
key) is checked independently of `save_model()`'s `"forge_format_version"`
-- a missing, wrong, or malformed value raises `PersistenceError` naming
both the found and supported values, with no forward/backward-compatibility
shim, exactly matching `load_model()`'s own versioning policy.

### Security
Same trust model as model persistence (**Security / trust model** below):
no `pickle`, no `eval`/`exec`, no dynamic import; every array loads with
`numpy.load(..., allow_pickle=False)`; a checkpoint's `"model"."type"` and
`"optimizer"."type"` strings are only ever dictionary lookups against their
respective in-process registries. A malicious/unknown value for either
raises `PersistenceError` before anything is constructed
(`tests/test_checkpoint.py`'s security tests cover both).

## Preprocessing metadata (Milestone 71)
`save_model()`/`load_model()` persist the model itself; they said nothing
about what a caller was supposed to do to an input *before* handing it to
the model. Every image-based example built through Milestone 70 (see
`docs/development/m70-image-preprocessing.md`'s Section 16/17 "Future
Pressure Identified") therefore left preprocessing (e.g. `Resize`, pixel
scaling) as external knowledge: a developer had to remember and manually
reproduce it at inference time, in a possibly-separate process, with
nothing in the saved file to check against or reconstruct from.

### Public API
```python
forge.save_model(model, path, preprocessing=some_transform)  # optional kwarg, default None
transform = forge.load_preprocessing(path)                   # -> Transform, or None
```
`preprocessing` is an ordinary `forge.data.transforms.Transform` (most
often a `Compose` pipeline) -- the *same object* a caller would otherwise
pass as `ImageFolder(..., transform=...)` or apply by hand before calling
`forge.predict()`. `load_preprocessing()` is a separate free function, not
a second return value of `load_model()`: `load_model()`'s signature and
return type are completely unchanged, so every existing caller keeps
working without modification; a caller that also wants the preprocessing
configuration asks for it explicitly, from the same file, independently.

### What is (and is not) saved
Only a JSON-safe **configuration** -- the transform's constructor
arguments -- travels through the file, via a small explicit registry
(`forge/serialization/transforms.py`, `forge.serialization.transforms.
register_transform()`) mirroring the module registry described above
exactly: a `type_name` string is looked up against transform types this
process has already imported and opted in, never `eval`'d, pickled, or
dynamically imported. Built-in registered transforms: `Resize`, `Compose`
(recursively, over its own child transforms), `Normalize`, `ReplaceValue`
(Milestone 92 -- replaces a sentinel value with a fixed per-column fill on
specific columns, e.g. imputing a dataset's sentinel-coded missing values;
see `docs/development/m92-real-dataset-ingestion.md`), `ToTensor`,
`Reshape`, `Flatten`. **`Lambda` is deliberately never registered** -- it
wraps an arbitrary Python callable (frequently a closure) with no safe
general representation short of serializing executable code, which
Milestone 71's brief explicitly ruled out. Attempting to save a `Lambda`
(directly, or nested inside a `Compose`) raises `PersistenceError`
immediately, before anything is written -- the same "not registered,
here's how to fix it" failure `spec_for_class()` already gives for an
unregistered `Module`. A pipeline needing a `Lambda`-shaped step (e.g.
pixel scaling) should use `Normalize` instead where the same computation
is expressible that way -- `examples/image_folder_classification/train.py`
does exactly this (`Normalize(mean=0.0, std=255.0)` in place of
`Lambda(lambda x: x / 255.0)`).

What is explicitly **not** part of this mechanism:
- **`ImageFolder` itself.** A `Dataset` is a data *source* (file paths on
  a particular machine); persisting it would couple a saved model to one
  filesystem layout for no benefit `forge.predict()` needs. Only the
  *transform* a `Dataset`'s samples were passed through is preprocessing
  in the sense this mechanism cares about.
- **Class-index-to-name vocabularies.** A model's predicted index -> label
  mapping is not "how to prepare an input tensor" -- it is a separate,
  output-side concept with its own mechanism, `save_model(...,
  classes=...)` (**Milestone 72**, see **Class-label metadata** below), not
  folded into `preprocessing`.
- **`Module` state.** `preprocessing` is stored as a sibling top-level
  metadata entry alongside `"root"`/`"device"`, never as an attribute of
  the `Module` tree itself -- a model's parameters and its input
  preprocessing remain conceptually distinct, matching this document's own
  **Optimizer-state limitations** precedent (a different concern, kept in
  a different place, rather than folded into `Module`).

### Compatibility: no format-version change
`"preprocessing"` is a new, *optional* top-level metadata key -- `null`
(equivalently, absent) when `save_model()` is called without
`preprocessing=`, exactly matching every pre-Milestone-71 call site. This
required no `FORMAT_VERSION` bump:
- **Forward-compatible.** A pre-Milestone-71 Forge build's `load_model()`
  never reads `metadata["preprocessing"]` at all (it only ever inspected
  `"root"`), so a Milestone-71-or-later file -- with or without a real
  preprocessing configuration -- loads on an older build exactly as before
  (parameters and architecture only; the preprocessing key is silently
  present but unused).
- **Backward-compatible.** A file saved before Milestone 71 has no
  `"preprocessing"` key at all; `load_preprocessing()` (`metadata.get(
  "preprocessing")`) treats a missing key exactly like an explicit `null`
  and returns `None` -- an ordinary, documented outcome, not an error.

`tests/test_preprocessing_persistence.py::
test_backward_compatible_file_has_no_preprocessing_key` verifies both
directions directly (a `.forge` archive with the `"preprocessing"` key
deleted entirely still loads through both `load_model()` and
`load_preprocessing()`).

**Not just for images.** `preprocessing=` was designed against
`image_folder_classification`'s `Resize`/`Normalize` pipeline, but nothing
about the mechanism itself is image-specific -- `Normalize` alone (no
`Resize`) is exactly the fitted feature-standardization transform
`examples/regression/train.py` saves alongside its model as of **Milestone
83**, consumed via `forge.predict_tensor_artifact()`
(`docs/architecture/training-engine.md`'s **Portable-artifact inference for
numeric input** section) rather than `predict_artifact()`'s image-file-
specific consumption path.

## Class-label metadata (Milestone 72)
`save_model()`/`load_model()`/`load_preprocessing()` together get a caller
from a saved artifact to a raw prediction `Tensor` -- but for a
classification model, a raw `Tensor` like `[0.02, 0.94, 0.04]` is not yet a
*useful* prediction. Milestone 71 explicitly deferred this exact gap (see
**Preprocessing metadata** above's "Class-index-to-name vocabularies" note,
and `examples/image_folder_classification/train.py`'s pre-Milestone-72
`classes.json` sidecar file): nothing in a saved model file recorded what a
predicted index *meant*, so a caller had to keep a separate file (or
in-memory list) in sync with the model by hand, with nothing to check
consistency against.

### Public API
```python
forge.save_model(model, path, classes=["cat", "dog"])  # optional kwarg, default None
classes = forge.load_classes(path)                      # -> list[str], or None
```
`classes[i]` means the same thing `forge.data.ImageFolder.classes[i]`
already means: `output[..., i]` is that class's score. This is a
deliberate, narrow convention (not an arbitrary metadata blob) -- a caller
training against `ImageFolder` passes `some_image_folder.classes` directly,
with no manual index bookkeeping. `load_classes()` is a separate free
function, not a second return value of `load_model()`, mirroring
`load_preprocessing()`'s own shape exactly and for the same reason:
`load_model()`'s signature and return type stay completely unchanged.

### What is (and is not) saved
Only a plain JSON list of strings -- never a dict, never per-class extra
data, never an inferred count. `save_model()` validates `classes` itself,
before writing anything: it must be a non-empty list, every element a
non-empty string, and no duplicate labels -- violating any of these raises
`PersistenceError` immediately (the same "fail before writing" behavior
`preprocessing=`'s `Lambda` rejection already has). What `save_model()`
deliberately does **not** do is validate `classes` against `model`'s actual
output width: Forge does not introspect an arbitrary module tree to guess
its output-class count (the same reasoning that keeps `save_model()` free
of any other architecture-inference machinery), so a `classes` list of the
wrong length is accepted at save time and only surfaces the first time it is
actually used to interpret a real prediction -- see
`forge.training.interpret_classification()`, which raises `TrainerError`
for that mismatch, with the two failure modes deliberately separated:
*malformed* class metadata fails at save time, *inconsistent-with-the-model*
class metadata fails at first interpretation.

### `forge.training.interpret_classification()`
```python
output = forge.predict(model, batch)                          # raw Tensor(batch, num_classes)
results = forge.interpret_classification(output, classes)      # list[ClassificationPrediction]
results[0].label        # "dog"
results[0].index         # 1
results[0].confidence    # 0.942
```
The "tensor output" -> "useful prediction" step: `output` is treated as
unnormalized per-class scores (logits) -- the same assumption `nn.
CrossEntropyLoss` already makes about its own `logits` argument
(`forge/nn/loss.py`), since every Forge classification model is trained
against exactly that loss. `confidence` is therefore that row's numerically
stable softmax probability of the predicted class, computed in plain
host-side NumPy on already-materialized (`no_grad()`-produced, about to be
printed) data -- not a new differentiable `Tensor.softmax()` primitive, and
not through the autograd graph. This is a legitimate probability reading of
a real training-time-consistent quantity, not an unjustified confidence
claim manufactured for display purposes.

### Compatibility: no format-version change
`"classes"` is a new, optional top-level metadata key -- `null`
(equivalently, absent) when `save_model()` is called without `classes=`,
exactly matching every pre-Milestone-72 call site and mirroring
`"preprocessing"`'s own Milestone 71 compatibility story exactly:
forward-compatible (an older Forge build never reads the key at all) and
backward-compatible (`load_classes()` treats a missing key the same as an
explicit `null`, returning `None`, not raising). No `FORMAT_VERSION` bump.

## Task metadata (Milestone 87, extended to `"sequence"` in Milestone 90 and `"tabular_classification"` in Milestone 91)
Milestone 86's `forge.predict_model()` had to guess which of the three
portable-artifact workflows (classification/regression/segmentation) a
`.forge` file represented, from `ModelInfo.classes`/`ModelInfo.model.
module_types` -- and that guess had a real, documented gap: a classification
model saved with `classes=None` (a valid state, see **Class-label metadata**
above) is `Linear`-terminated exactly like a regression model, with no saved
`classes` to disambiguate it, so it was misidentified as regression. Task
metadata closes this gap by letting a caller declare the artifact's intended
workflow explicitly, rather than Forge inferring it from architecture.
Milestone 90 added a fourth value, `"sequence"`, when a real char-RNN
workload found that a stepwise-recurrence model (`model.step()`/`model.
init_hidden()`, no `forward()`) has no architecture-based guess at all to
fall back on -- see **Sequence-artifact prediction** below. Milestone 91
added a fifth value, `"tabular_classification"`, when a real tabular
(non-image) classification workload found that `task="classification"` had
always meant "input is an image file path" -- see **Tabular-classification-
artifact prediction** below.

### Public API
```python
forge.save_model(model, path, task="classification")  # optional kwarg, default None
forge.save_model(model, path, task="regression")
forge.save_model(model, path, task="segmentation")
forge.save_model(model, path, task="sequence", classes=vocab)  # requires classes=
forge.save_model(model, path, task="tabular_classification", classes=labels)  # classes optional
info = forge.inspect_model(path)
info.task                                              # one of forge.serialization.model.TASK_TYPES, or None
```
`task`, when given, must be one of `forge.serialization.model.TASK_TYPES` --
`"classification"`, `"regression"`, `"segmentation"`, `"sequence"`,
`"tabular_classification"` -- a small, fixed vocabulary matching Forge's
five existing portable-artifact inference workflows exactly
(`predict_artifact()`/`predict_tensor_artifact()`/`predict_image_artifact()`/
`predict_sequence_artifact()`/`predict_tabular_classification_artifact()`,
Milestones 82-84/90/91), not an open-ended task registry. Any other value
raises `PersistenceError` before anything is written, the same "fail before
writing" behavior `preprocessing=`/`classes=` already have. `"sequence"` is
a genuinely distinct *prediction problem* (autoregressive next-token
generation from a seed), not merely "this model uses an RNN/LSTM layer" --
an RNN-based classifier is still saved with `task="classification"`.
`"tabular_classification"` is a genuinely distinct *input modality* from
`"classification"`, not a variant of it -- see below.

### What is (and is not) validated
`task="regression"` or `task="segmentation"` combined with a non-`None`
`classes=` raises `PersistenceError` immediately -- neither workflow has a
class-vocabulary concept, so the two pieces of metadata would disagree about
what the artifact is. `task="classification"`/`task="tabular_classification"`
place **no** such restriction on `classes`: `classes=None` remains a real,
valid state for either (see **Class-label metadata** above), and this is
precisely the state `task=` now lets `predict_model()` recognize correctly
instead of misidentifying as regression. `task="sequence"` is the opposite of
regression/segmentation: it *requires* a non-`None` `classes=` (the token
vocabulary `predict_sequence_artifact()` needs) and raises `PersistenceError`
if omitted -- there is no way to encode/decode tokens without one. `task` is
never validated against `model`'s actual architecture (Forge does not
introspect a module tree to guess its task, matching `classes`'s own "never
validated against output width at save time" precedent above) -- an
inaccurate `task` value is accepted at save time and only affects behavior
the next time `predict_model()` dispatches on it.

### Relationship to `classes`/`preprocessing`
`task` is a third, independent sibling metadata entry alongside
`"preprocessing"`/`"classes"` -- not merged into either. It describes the
artifact's **intended use** ("this is a classification model"), while
`classes` describes how to interpret a classification model's *output*
indices (or, for `task="sequence"`, the model's token vocabulary -- the same
"index i names entry i" concept), and `preprocessing` describes how to
prepare its *input*. A classification artifact may combine all three; a
regression or segmentation artifact combines `task` with `preprocessing`
only (never `classes`); a sequence artifact combines `task` with `classes`
only (no `preprocessing` concept -- there is no file to decode, only a seed
token sequence the caller has already tokenized).

### Compatibility: no format-version change
Exactly the same story as `"preprocessing"` (Milestone 71) and `"classes"`
(Milestone 72): `"task"` is a new, optional top-level metadata key -- `null`
(equivalently, absent) when `save_model()` is called without `task=`, so
files saved before Milestone 87 and files saved with no task metadata are
byte-for-byte equivalent in this respect. Forward-compatible (an older Forge
build never reads the key at all) and backward-compatible (`inspect_model()`
treats a missing key the same as an explicit `null`, returning `None`, not
raising). No `FORMAT_VERSION` bump -- consistent with this document's own
**Versioning** policy below, which reserves a bump for changes to the
*required* shape of every module node (e.g. Milestone 53's `"buffers"` key),
not for a new optional top-level key with full forward/backward compatibility
in both directions.

### Legacy artifacts and `forge.predict_model()`'s fallback
`info.task is None` means exactly "this artifact never declared an explicit
task" -- `inspect_model()` never guesses one from architecture. `forge.
predict_model()` (`forge/training/inference.py`) uses `info.task` as its
primary, authoritative dispatch signal when present, going straight to the
matching workflow with no architecture inspection at all -- this is what
finally lets a classification artifact saved with `classes=None` dispatch
correctly. Only when `task` is absent does it fall back to `_legacy_infer_
workflow()`, the exact Milestone 86 heuristic, kept isolated in its own
function and documented as a legacy-only mechanism: a genuinely pre-Milestone
-87 classification artifact saved with `classes=None` is still
architecturally indistinguishable from a legacy regression artifact (both are
`Linear`-terminated with no saved `classes`), so that specific ambiguity
persists for artifacts with no explicit task -- it cannot be resolved
retroactively without the caller re-saving with `task=`. See
`docs/development/m87-explicit-task-metadata.md` for the full reasoning.

### CLI
`forge model inspect model.forge` reports a "Task" line (`"unknown (legacy
artifact, saved before Milestone 87)"` when absent) sourced from `inspect_
model()`'s own `ModelInfo.task`; `--json` mode adds the same value under
`"task"` (`null` when absent). `forge model convert` preserves `task` across
a device conversion, alongside `preprocessing`/`classes`. `forge model
predict` (made task-aware in Milestone 88) reads `info.task` as its sole
routing signal and delegates straight to `forge.predict_model()` -- see
`forge/cli/model.py`'s own module docstring, and `docs/development/
cli.md`'s **Model prediction** section, for the full per-task input/output
behavior (including `"sequence"`, Milestone 90, and `"tabular_classification"`,
Milestone 91).

## Sequence-artifact prediction (Milestone 90)
Milestones 82-84 gave every ordinary `forward(x) -> output` model an
artifact-level prediction function (`predict_artifact()`/`predict_tensor_
artifact()`/`predict_image_artifact()`). A stepwise-recurrence model
(`examples/char_rnn`'s `CharRNN`, and every other sequence example in this
repo) does not fit that shape: it has no `forward()` at all, only `model.
step(x, state) -> (logits, state)` and `model.init_hidden(batch_size,
device=...)` -- the protocol `forge.training.generate_sequence()`
(Milestone 75) already samples from for an **in-memory** model. Nothing
connected that protocol to a saved `.forge` file: `predict()`/`predict_
tensor_artifact()` call `model(x)` directly, which raises `ModuleError`
(`"... does not implement forward()"`) for a stepwise model, and `save_and_
verify()`'s own docstring already documented this as explicitly out of its
scope, for exactly this reason.

### Public API
```python
forge.save_model(model, path, classes=vocab.chars, task="sequence")
generated = forge.predict_sequence_artifact(path, seed=list("a tensor"), length=200)
"".join(generated)
```
`predict_sequence_artifact(path, seed, length, *, device=None, rng=None)`
composes `load_model()` + `load_classes()` (the saved token vocabulary) +
`generate_sequence()`, building one-hot `encode`/`decode` closures over the
vocabulary -- the same closure every stepwise sequence example in this repo
already hand-wrote identically before this milestone
(`examples/char_rnn/train.py::_one_hot`). `seed` is a non-empty sequence of
tokens already split the way the saved vocabulary tokenizes (individual
characters for a char-level vocabulary; whatever `classes` lists otherwise);
this function does no tokenization of its own, mirroring `predict_tensor_
artifact()`'s "input must already be batched" contract. An unknown seed
token raises `DataError`; an artifact saved with no vocabulary, or whose
loaded model does not implement `init_hidden`/`step`, raises
`PersistenceError` with a clear, specific message -- never the raw
`ModuleError`/`AttributeError` that would otherwise surface deep inside
`generate_sequence()`.

`forge.predict_model()` dispatches to it when `info.task == "sequence"`,
via a new optional `length: int | None = None` keyword -- required (raises
`DataError` if omitted) only for a sequence artifact, ignored for the other
three. `forge model predict` (CLI) takes the seed as literal text on the
command line rather than a file (there is no file to decode), tokenized as
individual characters, plus a `--length` flag -- see `docs/development/
cli.md`'s **Model prediction** section for the exact behavior and its
documented char-level-only limitation.

### A related, pre-existing constraint surfaced by this workload
`classes=` validation used to reject any string that is empty after
`.strip()` -- correct for a *classification label* (a label of `" "` is
almost always a mistake), but wrong for a *token vocabulary*: a whitespace
character (most commonly a space) is one of the most ordinary tokens a text
vocabulary contains (`examples/char_rnn`'s own `Vocab.chars`, built from any
corpus with word boundaries, includes `" "`). `_validate_classes()` now
takes the `task` being saved and only requires "non-empty" (not
"non-whitespace") for `task="sequence"` -- classification's stricter check
is unchanged. See `docs/development/m90-sequence-artifact-inference.md` for
how this was found (the very first artifact this milestone tried to save).

See `docs/development/m90-sequence-artifact-inference.md` for the full
workload-driven writeup: the workload attempted, the concrete blocker found
(a real `.forge` artifact silently misidentified as regression, then
failing with `ModuleError: CharRNN does not implement forward()`), and why
`classes` (not a new `vocab=` parameter) was reused for the token
vocabulary.

## Tabular-classification-artifact prediction (Milestone 91)
Milestones 82-84 gave classification a single artifact-level prediction
function, `predict_artifact()` -- but its `image` argument has always
required a file path, decoded via `ImageFolder._load_image()`. A tabular
(non-image) classification model -- numeric features in, a class label out,
trainable end-to-end today with `Trainer`/`DataLoader`/`CrossEntropyLoss`/
`classes=`, with no framework change needed -- had no artifact-level
prediction path at all: `forge.predict_model()` routed every
`task="classification"` artifact to `predict_artifact()` unconditionally,
which raises `DataError` immediately for anything that is not a file path.
This was discovered, not hypothesized: `examples/tabular_classification`
trained a real classifier (82% test accuracy against a 25% trivial baseline)
and then failed at the very next step, trying to predict on a brand-new raw
feature vector via `forge.predict_model()`.

### Public API
```python
forge.save_model(model, path, preprocessing=normalize, classes=labels, task="tabular_classification")
results = forge.predict_tabular_classification_artifact(path, raw_features)
results[0].label, results[0].confidence
```
`predict_tabular_classification_artifact(path, input_data, *, device=None)`
composes `load_model()` + `load_preprocessing()` (optional, applied when
present -- e.g. a fitted `Normalize` feature-standardization transform,
mirroring `predict_tensor_artifact()`) + `predict()` + `load_classes()`
(optional -- a raw predicted class index is returned when absent, mirroring
`predict_artifact()`) + `interpret_classification()`. `input_data` must
already be batched exactly like `predict_tensor_artifact()`'s own
`input_data` -- a `Tensor`, NumPy array, or nested list/tuple, with a
leading batch dimension.

**Returns one result per input row, not one result overall.** This is the
one deliberate difference from `predict_artifact()` (always exactly one
image in, one prediction out): `input_data` follows `predict_tensor_
artifact()`'s "already batched, any batch size" convention, not `predict_
artifact()`'s "always exactly one file" convention, so silently keeping
only the first row's result would silently discard real caller data for any
batch size greater than one. A single-sample call (the common case) still
returns a length-1 list.

`forge.predict_model()` dispatches to it when `info.task ==
"tabular_classification"`, exactly like the other four tasks. There is no
legacy-architecture fallback for this task (mirroring `"sequence"`'s own
policy): `task="tabular_classification"` is architecturally identical to
`task="classification"` (both may be `Linear`-terminated with a `classes=`
vocabulary), so no guess from `module_types` could ever safely tell them
apart -- an explicit declaration is the only reliable signal. `forge model
predict` (CLI) shares `"regression"`'s JSON-numeric-file input parsing but
prints classification-shaped output (one block per input row for a
multi-row JSON file) -- see `docs/development/cli.md`'s **Model prediction**
section.

See `docs/development/m91-tabular-classification-artifact-inference.md` for
the full workload-driven writeup: the workload attempted (a synthetic
device-telemetry health classifier), the concrete blocker found and its
exact error text, and why a new task value (rather than reusing
`"classification"` or generalizing `predict_artifact()`) was the smallest
complete fix.

## Model inspection (Milestone 85)
`save_model()`/`load_model()`/`load_preprocessing()`/`load_classes()` give a
caller everything needed to *use* a saved artifact -- but a developer who
only just received a `.forge` file has no way to answer "what is this?"
without already knowing which of `forge.predict_artifact()`/
`predict_tensor_artifact()`/`predict_image_artifact()` applies, or opening
the archive by hand. `inspect_model()` closes that gap: a single read-only
call that turns the metadata `save_model()` already wrote into a structured,
programmatically usable summary. As of **Milestone 86**, a caller no longer
even has to read `ModelInfo` themselves to pick the right function --
`forge.predict_model()` (`docs/architecture/training-engine.md`'s own
**Unified portable-artifact prediction** section) calls `inspect_model()`
internally and dispatches on `ModelInfo.classes`/`ModelInfo.model.
module_types` for them. As of **Milestone 87**, `ModelInfo.task` -- when
present -- is consulted first and is authoritative; the `classes`/
`module_types` heuristic is now an isolated legacy fallback only (see **Task
metadata** above).

### Public API
```python
info = forge.inspect_model(path)   # -> ModelInfo
print(info)

info.model.type              # "Sequential"
info.model.module_types      # ("Sequential", "Conv2d", "ReLU", "MaxPool2d", "Linear")
info.model.parameter_count   # 12345
info.preprocessing           # PreprocessingInfo, or None
info.preprocessing.description  # "Resize(size=(64, 64)) -> Normalize(mean=0.0, std=255.0)"
info.preprocessing.transform    # the reconstructed Transform instance itself
info.classes                 # ["cat", "dog"], or None
info.task                    # "classification" / "regression" / "segmentation" / None
info.format_version          # 2
info.device                  # "cpu" or "cuda"
```
`ModelInfo`/`ModelSummary`/`PreprocessingInfo` are plain frozen dataclasses
(`forge.serialization.model`), not a dict -- fields are meant to be read
programmatically (`if info.classes: ...`), not just printed.

### What `inspect_model()` deliberately does and does not do
Reads only `metadata.json` via the same `read_archive()` primitive
`load_model()`/`load_preprocessing()`/`load_classes()` already use
internally -- it never reconstructs a live `Module` (no module-type registry
required), never requires CUDA regardless of the device the artifact was
saved for, and never touches `forge.random`'s state. This makes it cheap
relative to `load_model()` and safe to call before deciding whether, or how,
to load the model at all -- mirroring exactly why `forge/cli/_archive_info.py`
already reads metadata this way for `forge model inspect`, rather than
through `load_model()`/`load_checkpoint()`.

Reconstructing the preprocessing pipeline (when present) does go through
`deserialize_transform()`, the same reconstruction `load_preprocessing()`
performs -- so a saved transform type must be registered in this process,
exactly as `load_preprocessing()` already requires (a far smaller registry
than `load_model()`'s full module-type one).

`ModelInfo` deliberately omits per-parameter shapes/dtypes and dotted module
names -- that lower-level detail remains `forge model inspect`'s own report
(`walk_modules`/`walk_parameters` in `forge/cli/_archive_info.py`), which the
CLI command still produces unchanged. `inspect_model()`'s contract is
intentionally the smaller, product-level subset: enough to *decide* how to
use an artifact, not a dump of the archive's internal representation.

### Compatibility
Works identically on artifacts saved before Milestones 71/72/87 existed:
`info.preprocessing`/`info.classes`/`info.task` are simply `None`, exactly
matching `load_preprocessing()`/`load_classes()`'s own backward-compatible
behavior for a missing key. No `FORMAT_VERSION` change, and no new persisted
data -- `inspect_model()` only reads metadata `save_model()` already wrote.

### CLI
`forge model inspect model.forge` reports a "Preprocessing detail" line
sourced from `inspect_model()`'s own `PreprocessingInfo.description` (in
addition to its existing "Preprocessing: yes/no" line); `--json` mode adds
the same string under `"preprocessing_description"`. The command's per-module
and per-parameter listing is unchanged.

## Portable input-contract validation (Milestone 101)

`predict_tensor_artifact()`/`predict_tabular_classification_artifact()`
already composed `load_model()`/`load_preprocessing()`/`predict()` correctly
-- but neither ever checked that a caller's raw input actually matched the
shape the saved model expects before handing it to `preprocessing`/`Linear`.
Two real, reproduced failure modes followed:

```text
too few / too many features
    -> reaches Normalize/Linear anyway
    -> ShapeMismatchError deep inside preprocessing or the model
       ("Cannot apply '-' to shapes (1, 7) and (8,): not broadcastable.")

same feature count, columns swapped (e.g. Glucose/Pregnancies)
    -> the tensor shape is correct
    -> the model executes successfully
    -> a plausible-looking but wrong prediction, with no error at all
```

The first case was already an error, just a confusing, internals-leaking
one, raised well after the input crossed the artifact's public boundary. The
second case produced no error whatsoever -- this is the more serious of the
two, and it is what this section is really about.

### What is actually knowable

`forge.train()`/`forge.train_and_save()` receive a `forge.data.Dataset` and
a `Module` -- never a per-column name. The one thing Forge's training path
*does* reliably fix, for a plain `Sequential` MLP, is the width of the first
`Linear` layer's input (`in_features`) -- already present, unmodified, in
every saved artifact's `metadata.json` `"root"` tree (`Linear`'s own
registered `get_config()`, since Milestone 7). No new information needs to
be captured at training time, and no new metadata is written by
`save_model()` -- **Milestone 101 required no `FORMAT_VERSION` change.**

### `InputSchema`

```python
info = forge.inspect_model("model.forge")
info.input_schema                # InputSchema(feature_count=8), or None
info.input_schema.feature_count  # 8
```

`forge.serialization.InputSchema` (a frozen dataclass, mirroring
`ModelSummary`/`PreprocessingInfo`) has exactly one field: `feature_count`.
`ModelInfo.input_schema` is populated only when **both**:

1. the artifact was saved with `task="regression"` or
   `task="tabular_classification"` (`forge.save_model(..., task=...)`,
   Milestones 87/91) -- the two workflows whose input genuinely is "an
   already-batched fixed-width numeric feature vector"
   (`predict_tensor_artifact()`/`predict_tabular_classification_artifact()`);
   and
2. the saved architecture is a plain `Sequential` container whose first
   child is a `Linear` layer (`_leading_linear_in_features()`,
   `forge/serialization/model.py`) -- every real fixed-width-vector workload
   in this repo (`examples/regression`, `examples/tabular_diabetes`,
   `examples/tabular_classification`) is exactly this shape.

Otherwise `input_schema` is `None` -- **never guessed**. In particular:

- `task="classification"`/`"segmentation"` (an image file path input) and
  `task="sequence"` (a token-vocabulary seed, no fixed-width vector concept
  at all) never get an `input_schema`, even if a `Linear` classifier head
  exists somewhere deep in a CNN -- gating on `task` first means a
  `Conv2d`-first architecture is never even inspected for this.
- A legacy artifact saved with no `task=` at all gets no `input_schema`
  either, mirroring `_legacy_infer_workflow()`'s own "an absent `task` is
  never a license to guess" stance for the unrelated classification/
  regression ambiguity Milestone 87 closed. **Do not invent metadata for an
  artifact that never declared what kind of workflow it is.**
- A custom architecture whose raw input is not immediately consumed by a
  `Linear` (any container type other than `Sequential`, or a `Sequential`
  whose first child is not `Linear`) also gets `None` -- rather than a wrong
  guess about which layer "really" receives the input.

### Validation boundary

`predict_tensor_artifact()`/`predict_tabular_classification_artifact()`
each call `inspect_model(path)` and, when `input_schema` is not `None`,
check `input_data`'s last-axis width against `feature_count` -- **before**
`load_preprocessing()`'s transform or the model ever run:

```python
result = forge.predict_tensor_artifact("model.forge", seven_values)
# forge.DataError: predict_tensor_artifact() expected 8 input feature(s), received 7.
```

This is deliberately the *raw* input, not a post-preprocessing shape: every
registered tabular preprocessing step today (`ReplaceValue`, `Normalize`)
operates elementwise/per-column and never changes the feature-axis width,
so checking before preprocessing runs is equivalent to checking just before
the model, but fails immediately with a clear, artifact-level message
instead of surfacing from inside `Normalize`'s broadcast arithmetic. Both a
single unbatched sample (`(feature_count,)`) and a batch
(`(N, feature_count)`) remain valid -- exactly the two shapes `Linear`/
`predict()` already accept; only `feature_count` itself is checked, never
the batch dimension. Any other rank is left untouched, falling through to
whatever error the model/preprocessing already raises for it.

`predict_model()` needs no separate change: it already delegates to these
two functions unchanged, so the validation applies automatically to every
caller, including `forge model predict` (`forge/cli/model.py`).

### What this explicitly does not validate

**Feature ordering/semantics.** A same-width input whose columns have been
permuted (the diabetes dataset's real
`[Pregnancies, Glucose, BloodPressure, ...]` reordered to
`[Glucose, Pregnancies, BloodPressure, ...]`) passes this check --
`InputSchema` only ever recorded a count, because a count is the only thing
`forge.train()`'s `Dataset`-in, `Module`-in calling convention ever gave
Forge to record. This was verified empirically, not assumed: reordering two
real feature columns on the real `examples/tabular_diabetes` artifact
changed the model's reported confidence but produced no error and no shape
difference. Detecting this would require an explicit, per-column feature-
name declaration Forge's current public data/training API has no natural
place to express (`forge.data.Dataset`/`TensorDataset` carry no column
names) -- deliberately not built here; see **Out of scope** below.

**Dtype.** No dtype contract is recorded or checked -- `Tensor(input_data)`
already normalizes any numeric NumPy dtype into the model's own parameter
dtype at construction time (the same conversion `predict_tensor_artifact()`
already relied on before this milestone), so there is nothing this
milestone needed to add here.

**Sequence artifacts.** `task="sequence"` inputs are a seed *token
sequence* against a saved vocabulary, not a fixed-width numeric vector --
`predict_sequence_artifact()` already validates the one thing that actually
matters for this shape (every seed token must be a member of the saved
vocabulary, `forge.DataError` otherwise, Milestone 90) and gets no
`InputSchema` at all; forcing a `feature_count` concept onto it would
misrepresent what a stepwise-recurrence model's input actually is.

**Image artifacts.** `task="classification"`/`"segmentation"` already have
an input-adaptation mechanism (`_expected_image_channels()`, Milestone 94)
that *converts* a mismatched grayscale/RGB input rather than rejecting it --
a fundamentally different, already-solved problem (a real image with the
wrong channel count can be losslessly converted; a tabular row with the
wrong feature count or order cannot be "converted" into a correct one).
Investigated for a demonstrated gap and found none: Milestone 94's behavior
is unchanged by Milestone 101.

### Out of scope

No dataframe/CSV/feature-store/schema-registry/Pandera-style capability was
built, and no automatic feature-name inference was added. If a future
milestone gives `forge.train()`'s `Dataset`/`DataLoader` path a natural,
optional place for a caller to declare per-column names, a genuinely
semantic (order-aware) contract could be built on top of this same
`InputSchema` mechanism -- deferred, not attempted here, since no such place
exists in the current public data API and inventing one was explicitly out
of this milestone's scope.

### Compatibility

No `FORMAT_VERSION` change (still `2`); no new bytes written by
`save_model()`. Every existing artifact -- including files saved before
Milestone 87 introduced `task=` -- remains loadable exactly as before.
`ModelInfo.input_schema` is computed purely from `inspect_model()`'s
existing metadata read, so it is available immediately for every artifact
already saved with `task="regression"`/`task="tabular_classification"` and
a `Sequential([Linear, ...])`-shaped architecture, with no re-save required.

### CLI

`forge model inspect model.forge` prints an `Input: N feature(s)` line
(omitted entirely when `input_schema` is `None`, matching the existing
"Preprocessing detail" line's own omit-when-absent convention);
`--json` mode adds `"input_feature_count"` (an int, or `null`).

## Reusable artifact inference: `forge.load_predictor()` (Milestone 102)

Every function above `predict_model()` composes -- `predict_artifact()`,
`predict_tensor_artifact()`, `predict_image_artifact()`, `predict_sequence_
artifact()`, `predict_tabular_classification_artifact()` -- reopens and
reconstructs the entire `.forge` artifact on every single call:
`inspect_model()`, `load_model()`, `load_preprocessing()`, `load_classes()`,
all over again. That is the right tradeoff for a genuinely one-off
prediction, but it means an application making many predictions from the
same artifact (a batch job, a loop over incoming rows, a long-running
process) repeats the same disk read, archive parsing, and model
reconstruction for no reason -- the artifact never changes between calls.

```python
predictor = forge.load_predictor("model.forge")   # loads once

result_1 = predictor.predict(input_1)
result_2 = predictor.predict(input_2)
result_3 = predictor.predict(input_3)
```

`forge.load_predictor(path, device=None)` performs exactly the loading work
`predict_model()` performs on every call, once, and returns an
`ArtifactPredictor` (`forge/training/inference.py`) retaining:

- the loaded `Module` (`.model`);
- the reconstructed preprocessing pipeline, if any;
- the class/vocabulary list, if any (`.classes`);
- the resolved `InputSchema`, if any (`.input_schema`, Milestone 101);
- which of the five workflows `_determine_workflow()` resolved the artifact
  to (`.task`).

`predictor.predict(input_data, *, length=None, rng=None, threshold=0.5)`
then reruns only the genuinely per-call work -- input validation,
preprocessing, the forward pass -- by delegating to the same
artifact-independent core functions (`_classify_image_core()`,
`_predict_tensor_core()`, `_segment_image_core()`, `_generate_sequence_
core()`) the five one-shot functions themselves call. **This is not a
second inference engine**: a prediction through `ArtifactPredictor` and the
equivalent one-shot call agree exactly, for the same artifact/input/device,
because both run the identical code after loading.

### What moves to load time

For a `"classification"`/`"segmentation"` artifact, a missing preprocessing
configuration -- previously an error raised by `predict_artifact()`/
`predict_image_artifact()` on every call -- is now raised once, by
`load_predictor()` itself. For a `"sequence"` artifact, a missing vocabulary
or a model that does not implement the stepwise-recurrence protocol
(`init_hidden`/`step`) are checked the same way. `InputSchema` validation
(Milestone 101) still runs on every `predict()` call, because it is
genuinely a per-input check -- but it validates against the `InputSchema`
already cached on the predictor, never by calling `inspect_model()` again.

### Task support

All five workflows `predict_model()` supports are supported here unchanged:
`classification`, `regression`, `segmentation`, `sequence`,
`tabular_classification`. `predict()`'s calling convention mirrors
`predict_model()`'s own -- an image file path, a `Tensor`/NumPy array/
nested list, or a non-empty token sequence with `length=` -- so a caller
switching from `predict_model()` to `load_predictor()` for the same
artifact changes nothing about how it calls it, only how many times loading
happens.

### Lifetime, ownership, and concurrency

An `ArtifactPredictor` owns its loaded state for exactly as long as the
Python object exists -- ordinary object lifetime, no context manager, no
global cache, no model registry. Forge never tracks or reuses predictor
instances on a caller's behalf. `predict()` mutates no state on `self`
beyond what `predict()`/`generate_sequence()` themselves already do
(`model.eval()`/`no_grad()`/mode restoration), so repeated sequential reuse
from one thread is safe by construction; concurrent calls from multiple
threads carry whatever thread-safety guarantee (or lack of one) calling
`predict()` directly on a shared model from multiple threads already has --
this milestone adds no new concurrency infrastructure and makes no new
thread-safety claim.

### Not a model server

`ArtifactPredictor` is an in-process Python object, not a serving layer: no
HTTP/gRPC/socket interface, no request queue, no worker pool, no dynamic
batching, no global cache. `forge.predict_model()` remains the right choice
for a single, one-off prediction; `load_predictor()` is the natural next
step once an application makes more than one prediction from the same
artifact.

### Compatibility

No `FORMAT_VERSION` change (still `2`) and no new bytes written by
`save_model()` -- `load_predictor()` composes existing, unchanged reads.
`predict_model()` and the five task-specific `predict_*_artifact()`
functions are themselves unmodified in behavior (only refactored internally
to share their core logic with `ArtifactPredictor.predict()` -- see
`forge/training/inference.py`'s own Milestone 102 module docstring
paragraph) and remain fully supported.

## Evaluating a saved artifact: `ArtifactPredictor.evaluate()` (Milestone 113)

```python
predictor = forge.load_predictor("model.forge")
result = predictor.evaluate(X_test, y_test)      # raw held-out rows, labels
result.accuracy, result.baseline_accuracy, result.confusion_matrix
```

`evaluate()` scores a loaded artifact on held-out labeled data. The evaluation
path *is* the prediction path -- raw `X` -> `InputSchema` check -> the
artifact's persisted `preprocessing` -> the model -> metrics -- so the caller
never re-creates `Normalize`/`ReplaceValue`/`Resize` outside the artifact.
That is the point: `Trainer.evaluate()` has no hook for persisted
preprocessing and silently returned 37.0% on raw Pima rows for an artifact
that scores 72.1% (Milestone 97). Each batch runs through the same
`_predict_tensor_core()` that `predict()` uses; the loaded model,
preprocessing, and classes are reused, never reloaded.

| Task | `X` | `y` | Result |
|---|---|---|---|
| `tabular_classification` | `(n, ...)` numeric array/list/`Tensor` | class **names** (in `predictor.classes`) or integer indices | `ClassificationEvaluationResult` |
| `regression` | same | `(n,)`, `(n, 1)` or `(n, outputs)` | `RegressionEvaluationResult` |
| `classification` (image) | a directory in `ImageFolder` layout | omitted -- labels are the folder names, matched to `predictor.classes` **by name** | `ClassificationEvaluationResult` |
| `segmentation`, `sequence` | -- | -- | `DataError`: no evaluation semantics yet |

**Input dtype (Milestone 114).** A NumPy array or list given to `predict()`/`evaluate()`
(or any numeric `predict_*_artifact()`) is converted to Forge's default float32 before
preprocessing; an explicit `Tensor` is used as given. Previously an array kept its own
dtype, so the float64 array `np.genfromtxt` returns -- or an integer column -- was
rejected by a CUDA-loaded artifact (`CUDA 'matmul' requires matching dtypes`), while
CPU silently promoted and returned float64. CPU and CUDA now behave the same.

Both result types are frozen dataclasses exported from `forge.training`.
`ClassificationEvaluationResult`: `task`, `samples`, `loss` (mean
`CrossEntropyLoss`), `accuracy`, `baseline_accuracy`, `classes`,
`confusion_matrix` (read-only `(k, k)`, **rows = true class, columns =
predicted class**, indexed by the artifact's persisted class order), and the
parallel tuples `precision`, `recall`, `support`. A class the model never
predicted has `precision == 0.0`; a class with no true samples has
`recall == 0.0` (scikit-learn's `zero_division=0` convention) -- never NaN.
`RegressionEvaluationResult`: `task`, `samples`, `loss` (`MSELoss`), `mse`,
`mae`, `baseline_mse`. Each also has a `metrics` dict using the
`accuracy`/`mse`/`mae` keys `TrainingResult.val_metrics` uses. Metric
definitions live in `forge/training/evaluation.py`'s module docstring; there
is no new `Metric` class or registry.

**Baselines are computed on the evaluated data**, not read from the artifact
(which does not record its training distribution): `baseline_accuracy` is the
share of the evaluated data's majority class, `baseline_mse` the MSE of always
predicting the evaluated targets' mean. They are floors to beat, not claims
about the training set. (Pima holdout: 72.1% against a 62.3% baseline -- not
the 65.1% full-dataset majority share.)

**Errors.** Invalid input raises `DataError` naming the problem -- missing or
mismatched `y`, an unknown class name or out-of-range index, wrong feature
count, wrong rank, non-finite values in `X` (after preprocessing) or `y`, an
input the model itself rejects (surfaced from `ShapeMismatchError`), a
non-finite model output, a bad `batch_size`. A classification artifact saved
without `classes=`, or whose class count disagrees with its model's output
width, raises `PersistenceError`: the confusion matrix is indexed by the
persisted order and never by one inferred from `y`.

**Read-only.** Nothing is written; the model's parameters, its train/eval
mode (restored by `predict()`), the preprocessing, and the class/task
metadata are untouched, no RNG is consumed, and repeated calls return
identical results. `batch_size` (default 256) bounds memory only: metrics are
computed once over the concatenated outputs. It runs on whichever device the
predictor was loaded onto. No `FORMAT_VERSION` change and no new bytes are
written by `save_model()`.

## Custom-module limitations
See **Custom/composite modules** above: only module types registered via
`forge.serialization.register_module()` in the *loading* process can be
saved or reconstructed. Forge ships two built-in registrations (`Linear`,
`ReLU`); any other class -- including every composite model a user writes
-- needs its own explicit registration before `save_model()`/`load_model()`
will handle it. There is no reflection-based or pickle-based fallback for
unregistered types.

## File format
A single file: a ZIP archive (`forge/serialization/archive.py`) containing
```text
metadata.json                        -- module tree, config, parameter metadata
parameters/<dotted.parameter.name>.npy  -- one NumPy array per parameter
```
Chosen over a bespoke binary format or a full-object `pickle` dump because:
- **Inspectable without Forge.** `metadata.json` is readable text; each
  `.npy` is standard NumPy array storage; the whole thing opens with any
  ZIP tool.
- **Not executable by construction.** `json.loads` cannot produce a
  callable or arbitrary object graph; loading `.npy` payloads always
  passes `allow_pickle=False`, so a value array can never smuggle in a
  pickled Python object -- see **Security/trust model**.
- **No new binary dependency.** ZIP and `.npy` are both already reachable
  from the standard library / NumPy, matching ADR-001's numerical-
  foundation boundary and the "no large binary dependencies" constraint.
- **Efficient enough for M7's models.** `.npy` stores raw contiguous
  array bytes (plus a small header) with optional DEFLATE compression via
  the surrounding ZIP -- appropriate for the small CPU models this
  milestone targets (see `docs/development/development-environment.md`).

## Versioning
`metadata.json`'s `"forge_format_version"` field is checked against this
build's `forge.serialization.model.FORMAT_VERSION` (`2`, as of Milestone 53
-- see below). A mismatch of any kind -- older, newer, missing, or
malformed -- raises `PersistenceError` naming both the found and supported
values. There is no forward- or backward-compatibility shim in this
milestone: a version change is a breaking change to the format until a
later milestone implements migration, and Forge does not claim otherwise.

**Milestone 53 bumped `FORMAT_VERSION` `1 -> 2`** for the new required
`"buffers"` key per module node (**Buffer state**, above) -- the first
bump since the format's introduction.

**Milestone 13 did not bump `FORMAT_VERSION`.** CUDA persistence needed no
new metadata field or archive layout -- only the `"device"` field's set of
legal values (`"cpu"` and now `"cuda"`) and `load_model()`'s own runtime
policy for that value changed, both handled entirely in Python without
touching the wire format. An M7-M12 CPU-only file (`"device": "cpu"`) is
still valid version-`1` metadata and loads unmodified; `load_model()`
applies exactly the same version check to every file regardless of which
device it names.

## Atomicity and file safety
`save_model()` writes the full archive to a temporary file in the
destination's own directory, then `os.replace()`s it into place only after
writing succeeds completely (`forge/serialization/archive.py:write_archive`).
`os.replace` is an atomic rename on both POSIX and Windows for a
same-volume destination, so a reader can never observe a partially written
file, and a failed save (disk full, permission error, an unregistered
module type discovered mid-tree) leaves no file at the destination path at
all -- the temporary file is removed and `PersistenceError` is raised.
Because `save_model()` builds the entire in-memory metadata tree (and
validates every module type against the registry) *before* writing
anything, an unsupported module type anywhere in the tree is caught before
the archive write even begins.

## Security / trust model
Model files are treated as **untrusted input**. Loading a file:
- Never calls `eval`, `exec`, `pickle.load`/`pickle.loads`, or dynamic
  import (`importlib`) on anything read from the file.
- Never constructs a class that was not already imported and explicitly
  registered with `register_module()` by trusted, already-running Forge
  code -- a file's `"type"` string is only ever a dictionary key, looked
  up against that fixed, in-process registry.
- Loads every numeric array with `numpy.load(..., allow_pickle=False)`,
  which raises rather than deserializing an object-dtype array (NumPy's
  own object-array pickling is the one part of the `.npy`/`.npz` format
  that *can* run arbitrary code -- explicitly disabled here).
- Validates format version, device, module-tree structure, and every
  parameter's shape/dtype before trusting any of it, raising
  `PersistenceError` with specific context on the first inconsistency
  found rather than partially reconstructing a model from a bad file.

The one privileged input is the **registry itself**, populated only by
code already running in the loading process (Forge's own built-ins, plus
whatever a user's own trusted code registers) -- never by the file being
loaded. A file cannot expand what it is able to cause Forge to construct.

## Errors
All persistence failures raise `forge.exceptions.PersistenceError`,
covering: a non-`Module` passed to `save_model()`, a module type not
registered for persistence (`save_model()` or `load_model()`), an invalid
save destination (missing parent directory, OS-level write failure), a
missing model file, a corrupt/non-ZIP file, missing or malformed
`metadata.json`, an unsupported format version, an unrecognized recorded
device, an invalid `device=` override passed to `load_model()`, a
CUDA-recorded (or explicitly `device="cuda"`-requested) load with no CUDA
backend available, a missing parameter's data, a parameter shape/dtype
mismatch between metadata and the actual array, corrupted parameter bytes,
and a structural inconsistency between a file's declared parameters/children
and what the registered constructor actually produced ("inconsistent model
state"). `save_model(..., preprocessing=...)`/`load_preprocessing()`
(Milestone 71) raise the same `PersistenceError` for: an unregistered
transform type anywhere in the given `preprocessing` (most notably a
`Lambda`, see **Preprocessing metadata** above), a malformed
`"preprocessing"` metadata node, and invalid configuration for a registered
transform type (e.g. a saved `Resize` config with a non-positive
dimension). `save_model(..., classes=...)`/`load_classes()` (Milestone 72)
raise the same `PersistenceError` for: a non-list `classes`, an empty list,
a non-string or empty-string element, duplicate labels, and a malformed
`"classes"` metadata entry on load (see **Class-label metadata** above);
`forge.training.interpret_classification()` raises `TrainerError` instead,
for a non-2-D output or an output whose class-score dimension does not
match `len(classes)` (a *model/vocabulary* inconsistency discovered at
interpretation time, not a persistence-format problem). `save_model(...,
task=...)`/`inspect_model()` (Milestone 87, extended in Milestone 90) raise
the same `PersistenceError` for: a `task` value not in `TASK_TYPES`,
`task="regression"`/`task="segmentation"` combined with a non-`None`
`classes=`, `task="sequence"` combined with `classes=None`, and a malformed
`"task"` metadata entry on load (see **Task metadata** above); `forge.
predict_model()` raises `PersistenceError` when neither an explicit `task`
nor the legacy architecture heuristic can determine a supported workflow (see
`_legacy_infer_workflow()`'s own error message). `forge.predict_sequence_
artifact()` (Milestone 90) raises the same `PersistenceError` for an
artifact with no saved vocabulary or a loaded model missing the `step`/
`init_hidden` protocol, and `DataError` for an empty seed or a seed token
outside the saved vocabulary (see **Sequence-artifact prediction** above). A
mixed-device
module tree passed to `save_model()` raises
`ModuleError` (from `Module.device`), not `PersistenceError` -- the same
error that operation already raises everywhere else in Forge. Low-level
exceptions (`zipfile.BadZipFile`, `json.JSONDecodeError`, raw `OSError`s)
are always caught and re-raised as `PersistenceError` with added context,
never surfaced directly to callers.

`save_checkpoint()`/`load_checkpoint()` (Milestone 18) raise the same
`PersistenceError` for the checkpoint-specific equivalents: a non-`Module`
model or non-`Optimizer` optimizer, an unregistered/unsupported/malicious
optimizer type, an optimizer holding a `Parameter` not reachable from the
given model, an unsupported checkpoint format version, missing/malformed
`"optimizer"`/`"training_progress"`/`"rng"`/`"extra"` metadata, a missing
optimizer-state array, and non-JSON-safe `extra` data -- always via the same
explicit-registry / no-arbitrary-construction mechanism as `load_model()`,
never a raw exception surfaced to callers.

## Known limitations
- Only `Linear` and `ReLU` are built-in registered module types; every
  other class (including any composite model) requires an explicit
  `register_module()` call before it can be saved/loaded -- see **Custom
  module limitations**.
- `save_model()`/`load_model()` remain optimizer-state-free by design (see
  **Optimizer-state limitations**) -- training-resume checkpointing is
  `save_checkpoint()`/`load_checkpoint()`'s separate job (Milestone 18, see
  **Checkpointing** above), not something `save_model()` grew.
- CPU and CUDA only, one device per model tree: `"device"` is `"cpu"` or
  `"cuda"` (Milestone 13); a file recording anything else fails to load
  rather than silently running on CPU or pretending to run on that device.
  No multi-GPU-aware serialization -- CUDA persistence is bound by the same
  single-GPU (index 0) restriction as the rest of the CUDA backend (see
  `docs/architecture/cuda-backend.md`). Checkpoints inherit the same
  restriction.
- No forward/backward format-version compatibility or migration, for either
  model files or checkpoints.
- No compression tuning, encryption, or model/checkpoint signing (all
  explicitly out of scope for this milestone).
- No CLI (`forge` command-line save/load entry points) yet -- `save_model`/
  `load_model`/`save_checkpoint`/`load_checkpoint` are Python API only.
- `save_checkpoint()`/`load_checkpoint()` do not carry `preprocessing=`
  (Milestone 71's mechanism is `save_model()`-only) -- a resumed training
  run is expected to reconstruct its own `Dataset`/transform pipeline from
  its own training script, exactly as before; only the trained-model-for-
  inference path gained persisted preprocessing.
- Preprocessing persistence covers only the transforms registered with
  `forge.serialization.transforms.register_transform()` (`Resize`,
  `Compose`, `Normalize`, `ReplaceValue`, `ToTensor`, `Reshape`, `Flatten`) -- `Lambda` and
  any other custom `Transform` subclass must be registered the same way a
  custom `Module` subclass must be, or expressed using a registered
  transform instead; there is no reflection/pickle-based fallback, by the
  same design choice as the module registry.
- Class-label metadata (Milestone 72) is a flat list of strings only -- no
  hierarchical/multi-label taxonomy, no per-class extra data (e.g. a
  description or color) (a separate, explicit task-type field was added in
  Milestone 87 -- see **Task metadata** above). It is also
  never validated against `model`'s actual output width at save time (see
  **Class-label metadata**'s own note on why) -- only at first
  `interpret_classification()` call.
- Task metadata (Milestones 87/90) is a fixed, closed four-value vocabulary
  (`TASK_TYPES`) -- no generic/open-ended task registry, and no automatic
  detection from a model's architecture for newly saved artifacts (see **Task
  metadata**'s own **Architecture Guardrails**). It is never validated
  against `model`'s actual architecture at save time, the same "trust the
  caller, validate only structure" precedent `classes` already set. A
  genuinely legacy artifact (saved before Milestone 87, or with `task`
  deliberately omitted) that is a classification model saved with
  `classes=None` remains indistinguishable from a legacy regression artifact
  by `forge.predict_model()`'s fallback heuristic -- this cannot be resolved
  retroactively without re-saving the artifact with an explicit `task=`. A
  `task="sequence"` artifact has no equivalent legacy fallback at all --
  there is no architecture-based guess for a stepwise-recurrence model, so
  it always requires an explicit `task="sequence"` (see
  **Sequence-artifact prediction**).
- Sequence-artifact prediction (Milestone 90) is scoped to a saved token
  vocabulary (`classes`) and one-hot encode/decode over it -- the same shape
  `examples/char_rnn`/`examples/word_rnn` already use. A vocabulary-free
  numeric sequence model (forecasting raw floats, not a fixed token
  vocabulary) is not covered; it would need its own artifact-shape function.
  `forge model predict`'s CLI tokenizes its seed as individual characters
  only -- a word-level vocabulary (`examples/word_rnn`) needs `forge.
  predict_sequence_artifact()` called directly with a pre-tokenized seed
  list. Neither `examples/word_rnn` nor `examples/long_range_recall` were
  retrofitted with `task="sequence"`/an `infer.py` by this milestone -- only
  `examples/char_rnn`, the one workload this milestone actually built and
  validated end-to-end; the same `save_model(classes=vocab, task="sequence")`
  + `infer.py` pattern applies to them unchanged, whenever needed.
- Checkpointing (Milestone 18) is itself further scoped down: only `SGD` and
  `Adam` are built-in registered optimizer types (any other type needs
  `register_optimizer()`, as `register_module()` requires for a custom
  `Module`); `TrainingHistory` is never serialized (see **Checkpointing**'s
  **Training progress** subsection); RNG determinism covers only Forge's
  process-global default generator, not a caller-supplied `Dropout(...,
  generator=...)` or a `DataLoader`'s own shuffling generator (see
  **Checkpointing**'s **RNG / determinism policy** subsection); and there is
  no distributed/multi-worker/sharded/async checkpointing, no automatic
  periodic checkpoint scheduling, and no early-stopping integration.
