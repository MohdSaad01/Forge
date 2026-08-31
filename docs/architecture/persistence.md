# Model Persistence (Milestone 7)

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
M7 remains CPU-only (see `docs/development/development-environment.md`).
The archive records a single top-level `"device": "cpu"` field; every
parameter loaded is explicitly constructed with `device="cpu"`.
`load_model()` on a file whose `"device"` is anything else (e.g. a
hypothetical future `"cuda"` value, or a tampered file) raises
`PersistenceError` naming the unsupported device -- Forge never silently
loads a CUDA-tagged model onto CPU or pretends CUDA execution occurred.

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
`metadata.json`, an unsupported format version, an unsupported device, a
missing parameter's data, a parameter shape/dtype mismatch between
metadata and the actual array, corrupted parameter bytes, and a
structural inconsistency between a file's declared parameters/children and
what the registered constructor actually produced ("inconsistent model
state"). Low-level exceptions (`zipfile.BadZipFile`, `json.JSONDecodeError`,
raw `OSError`s) are always caught and re-raised as `PersistenceError` with
added context, never surfaced directly to callers.

## Known limitations
- Only `Linear` and `ReLU` are built-in registered module types; every
  other class (including any composite model) requires an explicit
  `register_module()` call before it can be saved/loaded -- see **Custom
  module limitations**.
- No optimizer-state checkpointing or training-resume support (see
  **Optimizer-state limitations**).
- CPU-only: saved files always declare `"device": "cpu"`; no CUDA
  serialization exists yet, and a file claiming another device fails to
  load rather than silently running on CPU or pretending to run on that
  device.
- No forward/backward format-version compatibility or migration.
- No compression tuning, encryption, or model signing (all explicitly out
  of scope for this milestone).
- No CLI (`forge` command-line save/load entry points) yet -- `save_model`/
  `load_model` are Python API only.
