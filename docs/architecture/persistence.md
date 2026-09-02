# Model Persistence (Milestone 7; CUDA persistence in Milestone 13)

## Package layout
```
forge/
    serialization/
        registry.py    ModuleSpec, register_module(), spec_for_class/spec_for_name,
                        built-in registration of Linear/ReLU
        archive.py      write_archive/read_archive -- the ZIP(json + .npy) file format
        model.py         save_model(), load_model() -- tree walk, validation, reconstruction
```
`forge.serialization` is exposed as a submodule of `forge`, alongside
`forge.nn`/`forge.optim`/`forge.data`/`forge.training`. `save_model`,
`load_model`, and the new `PersistenceError` are also exposed at the top
level (`forge.save_model`, `forge.load_model`, `forge.PersistenceError`).

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
```
Concretely, for each module in the tree (self, then every `_modules` child,
recursively): its registered type name, its architecture **configuration**
(constructor keyword arguments -- not weights), its `.training` flag, and
its own `_parameters`' shapes/dtypes/`requires_grad`/values. Parameter
*names* are the dotted path used elsewhere in Forge (`fc1.weight`), matching
`Module.named_parameters()`.

**Not saved:** `.grad` on any parameter, any autograd graph (`grad_fn`),
and optimizer state -- see **Autograd state** and **Optimizer state** below.

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
restored. Consequently a `load_model()`-ed model can be used for inference
immediately, but resuming *training* from a saved checkpoint with
identical optimizer dynamics is not supported by this milestone. This is
an explicit scope boundary (see `docs/development/roadmap.md`), not an
oversight: training-resume checkpointing is a distinct capability with its
own format/versioning concerns and is left for a later milestone.

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
build's `forge.serialization.model.FORMAT_VERSION` (`1`). A mismatch of
any kind -- older, newer, missing, or malformed -- raises
`PersistenceError` naming both the found and supported values. There is no
forward- or backward-compatibility shim in this milestone: a version
change is a breaking change to the format until a later milestone
implements migration, and Forge does not claim otherwise.

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
state"). A mixed-device module tree passed to `save_model()` raises
`ModuleError` (from `Module.device`), not `PersistenceError` -- the same
error that operation already raises everywhere else in Forge. Low-level
exceptions (`zipfile.BadZipFile`, `json.JSONDecodeError`, raw `OSError`s)
are always caught and re-raised as `PersistenceError` with added context,
never surfaced directly to callers.

## Known limitations
- Only `Linear` and `ReLU` are built-in registered module types; every
  other class (including any composite model) requires an explicit
  `register_module()` call before it can be saved/loaded -- see **Custom
  module limitations**.
- No optimizer-state checkpointing or training-resume support (see
  **Optimizer-state limitations**).
- CPU and CUDA only, one device per model tree: `"device"` is `"cpu"` or
  `"cuda"` (Milestone 13); a file recording anything else fails to load
  rather than silently running on CPU or pretending to run on that device.
  No multi-GPU-aware serialization -- CUDA persistence is bound by the same
  single-GPU (index 0) restriction as the rest of the CUDA backend (see
  `docs/architecture/cuda-backend.md`).
- No forward/backward format-version compatibility or migration.
- No compression tuning, encryption, or model signing (all explicitly out
  of scope for this milestone).
- No CLI (`forge` command-line save/load entry points) yet -- `save_model`/
  `load_model` are Python API only.
