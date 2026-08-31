# ADR-003: Model Persistence Format and Reconstruction Mechanism

## Status
Accepted

## Decision
Persist a trained Forge model as a ZIP archive containing a JSON
architecture/configuration manifest plus one `.npy` file per parameter,
loaded with `numpy.load(..., allow_pickle=False)`. Reconstruct module
instances only through an explicit, in-process registry of supported
module types (`forge.serialization.register_module`) keyed by a plain
string name -- never through `pickle`, `eval`/`exec`, or dynamic
import/attribute resolution driven by data read from the file.

## Rationale
A model file is untrusted input by default: it may be shared, copied, or
tampered with outside Forge's control. `pickle` (Python's default general
object serializer, and what a naive "just pickle the model" approach would
use) can execute arbitrary code during deserialization by design, which is
incompatible with treating saved files as untrusted -- the milestone
explicitly rules this out. A structured, inspectable text-plus-array format
avoids that risk entirely: JSON cannot encode a callable, and NumPy's
`allow_pickle=False` load path cannot deserialize an object array (the one
part of `.npy`/`.npz` that can run code).

That leaves the harder problem of *reconstructing* a Python `Module`
subclass from data alone. The alternatives considered:
1. **Full object pickling.** Rejected outright -- arbitrary code execution
   on load, explicitly prohibited by the milestone brief and by Forge's
   security requirements generally.
2. **Reflection-based reconstruction** (introspect `__init__`'s signature,
   guess constructor arguments from instance attributes). Rejected: fragile
   for any class with non-trivial construction logic, and still amounts to
   calling code shaped by data taken from the file, which is a smaller but
   real version of the same trust problem.
3. **An explicit registry of known-safe types** (the chosen approach).
   Every class `load_model()` can construct was already imported and
   opted in by trusted, already-running code before the file was ever
   read -- the file's `"type"` string only ever selects among constructors
   that already exist in the process, never introduces a new one.

The registry additionally lets Forge state its persistence guarantees
honestly at this milestone's scope: two built-in types (`Linear`, `ReLU`)
are supported out of the box, and everything else -- including any
composite/custom model a user writes -- is explicitly unsupported until
registered, with a clear `PersistenceError` rather than a silent failure
or an unsafe fallback.

## Consequences
- Saved files are human/tool-inspectable (a ZIP viewer, a text editor, and
  `numpy.load` are sufficient) without depending on Forge at all to
  *read* the metadata, even though reconstructing a live model still
  requires Forge's registry.
- A custom `Module` subclass is not persistable until its author calls
  `register_module()` for it. This is a real limitation (see
  `docs/architecture/persistence.md`'s **Custom-module limitations**), not
  a temporary gap expected to be silently patched by a future fallback to
  pickling or reflection -- any future expansion of what's persistable
  must go through the same explicit-registration model, not around it.
- The format is versioned (`forge_format_version`) independently of
  Forge's own package version, so the reconstruction *mechanism* (this
  ADR) and the *wire format* it produces can each evolve without being
  conflated -- a version bump is a deliberate, documented breaking change
  in `load_model()`, never silently tolerated.
- Optimizer state and CUDA device state are out of scope for this format
  by design (see `docs/architecture/persistence.md`); extending the format
  to cover either is a distinct future decision, not implied by this one.
