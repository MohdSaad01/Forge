# Forge Maintenance Model (Milestone 110)

Forge reached 1.0 at Milestone 106 (`docs/development/progress.md`): the
framework itself is feature-complete for everything the current examples
need. Milestone 110 closes the milestone-driven feature-accumulation model
(`docs/development/roadmap.md`/`workflow.md`) and replaces it with a
maintenance model for the phase Forge is now in.

```text
FORGE 1.x MAINTENANCE MODE
```

This does not replace `roadmap.md`/`workflow.md` -- it governs what happens
*after* a real problem is identified, in place of picking the next
milestone from a backlog. Nothing here changes any public API.

## 1. Maintenance boundary

What Forge 1.x currently supports -- extending this set is a Minor/Major
change (see **Release discipline** below), not an ordinary bug fix:

| Area | Supported surface | Reference |
|---|---|---|
| CPU | All ops, NumPy-backed, any platform | `docs/architecture/backend-architecture.md` |
| CUDA | Hardware-verified on one reference GPU (see **Compatibility**); same op surface as CPU | `docs/architecture/cuda-backend.md`, `cuda-memory-allocator.md`, `cuda-streams.md`, `cuda-transfers.md` |
| Model APIs | `forge.Tensor`/autograd; `forge.nn` layers/losses listed in `README.md`'s **What's currently in Forge** | `docs/architecture/tensor-api.md`, `autograd.md`, `modules.md` |
| Data APIs | `Dataset`/`TensorDataset`/`Subset`/`ImageFolder`, transforms, `DataLoader`, `CUDAPrefetchLoader` | `docs/architecture/data-system.md`, `data-pipeline.md` |
| Training | `Trainer`, `forge.train()`/`train_and_save()`, `start_training_session()`, metrics, early stopping | `docs/architecture/training-engine.md` |
| Artifacts | `save_model`/`load_model`, `save_checkpoint`/`load_checkpoint`, `inspect_model()`, `InputSchema` | `docs/architecture/persistence.md` |
| Inference | `predict()`, the five `predict_*_artifact()` functions, `predict_model()`, `load_predictor()`/`ArtifactPredictor` | `docs/architecture/training-engine.md` |
| High-level image classification | `forge.train_image_classifier()`, `ImageFolder(on_error=)` | `forge/training/image_classifier.py` module docstring |
| High-level tabular classification/regression | `forge.train_tabular_classifier()`, `forge.train_tabular_regressor()`, `ArtifactPredictor.evaluate()` | `forge/training/tabular.py` module docstring, `docs/development/m114-tabular-workflows.md` |
| Packaging | `pip install`/wheel+sdist build, CLI entry point | M93, M105, M106 in `progress.md` |

Not supported, and not implicitly promised by any of the above: attention/
Transformer layers, convolution beyond 2D, distributed/multi-GPU training,
mixed-precision training, ONNX or other framework interop (see `README.md`'s
**Project scope and philosophy**).

## 2. Regression baseline

Run before and after any change:

```bash
python -m pytest tests/ -q
```

Baseline recorded at Milestone 115 (2026-09-20, reference hardware, CUDA
present; 2802 at Issue I1, plus 43 tests for M113's `evaluate()`, 164 for
M114's tabular workflows and 24 for M115's image-classifier preflight):

```text
Total:   3033
Passed:  3033
Failed:  0
Skipped: 0
```

A change that drops this to fewer passing tests, or that changes this
number without a corresponding `tests/` change explaining why, is a
regression by definition (see **Bug classification** below) -- diagnose
before proceeding, regardless of what triggered the change.

CI (`.github/workflows/ci.yml`) runs the CPU-visible half of this suite
(CUDA tests skip cleanly without a GPU) plus a wheel-build/install smoke
test on every push to `main`, on Python 3.11 and 3.13. The full number
above additionally exercises every CUDA-specific test, and is only
reproducible on the reference GPU.

## 3. Real-world smoke test

`tests/real_world/petimages_smoke.py` is the maintenance-mode real-world
acceptance check:

```text
ImageFolder -> train_image_classifier() -> artifact -> inference
```

It is a tracked, standalone script -- **local validation, not CI, and not
part of `pytest`** (`tests/` collects only `test_*.py`, so neither
`python -m pytest tests/` nor `.github/workflows/ci.yml` ever runs it). What
it needs and where it lives:

- **Data:** a real cat/dog dataset in the ~25,000-image "PetImages" layout
  (`<root>/<class>/<n>.jpg`, two classes) that Milestone 107 validated
  against. The dataset is not part of Forge and is never committed; keep it
  anywhere outside the repository and pass its directory as the first
  argument, or set `FORGE_PETIMAGES_DIR`.
- **What it does:** copies a deterministic 160-image subset (the first 80
  files of each class in sorted filename order) to a temporary directory,
  trains 2 epochs on the CPU (~10s on the reference machine), asserts the
  artifact saves/verifies and reloads with the right classes, and asserts
  that inference on one held-out image per class returns a valid
  `ClassificationPrediction`. It is not an accuracy benchmark -- the subset
  and epoch count are chosen for speed, not for a meaningful accuracy number.
- **Without the dataset:** it prints `SKIP: ...` and exits 0. A skip is not
  a pass -- if the change you are validating touches the chain below, run it
  where the dataset is available.

```bash
python tests/real_world/petimages_smoke.py /path/to/petimages
```

Run this after any change that plausibly touches the chain above (image
decoding, `ImageFolder`, `train_image_classifier`, artifact save/load,
`predict_model`/`predict_artifact`), in addition to the full regression
suite. The full ~25,000-image run is not repeated for ordinary changes; it
stays a manual, occasional exercise:

```python
forge.train_image_classifier("/path/to/petimages", path="model.forge", epochs=5, device="cuda")
```

## 4. Bug classification

| Class | Definition |
|---|---|
| **Bug** | Existing documented behavior is incorrect. |
| **Regression** | Previously working behavior has broken. |
| **UX issue** | The functionality works but is unnecessarily difficult to use. |
| **Feature request** | New capability not currently promised by Forge. |
| **Unsupported use case** | Outside the current contract (see **Maintenance boundary**). |

Classify before acting. A Feature request or Unsupported use case does not
get emergency framework development -- it goes through **Issue-driven
development** below like anything else.

## 5. Release discipline

```text
Patch  -- bug fixes, regressions, documentation corrections.
           No public API change. Example: a serialization round-trip bug,
           a stale README code sample.

Minor  -- backward-compatible post-1.0 capabilities.
           Additive only -- an existing call signature/return type/file
           format never changes meaning. Example: a new predict_*_artifact
           variant, a new opt-in Dataset/transform.

Major  -- breaking API or architectural changes.
           Anything that changes an existing public call's behavior/
           signature, or bumps FORGE_FORMAT_VERSION (see ADR-003). Requires
           the same explicit, documented justification any prior breaking
           change in progress.md already needed.
```

## 6. Compatibility

Only what has actually been tested is claimed:

- **Python:** `>=3.11` (`pyproject.toml`); CI runs 3.11 and 3.13 on every
  push. Local development is on 3.13.5.
- **Operating systems:** developed and hardware-verified on Windows 10;
  CI's CPU suite and packaging smoke test run on `ubuntu-latest`. No other
  OS has been tested.
- **CUDA:** hardware-verified on exactly one reference GPU/driver/toolkit
  combination -- NVIDIA GeForce 940MX (Compute Capability 5.0), CUDA
  Toolkit 12.6, driver 582.53, MSVC 19.44 (`docs/development/
  development-environment.md`). Expected to work on other CUDA-capable
  NVIDIA GPUs but not tested on hardware this project does not own; never
  claim CUDA correctness without hardware verification (`CLAUDE.md`).
- **Dependencies:** `numpy>=1.26`, `Pillow>=10.0` (runtime); `pytest>=8.0`,
  `matplotlib>=3.8` (dev only, `.[dev]`).

## 7. Issue-driven development

> No significant Forge feature should be added without a demonstrated
> requirement, workload, or repeated user problem.

Sources: your own projects, external developers, bug reports, real
datasets, deployment requirements, compatibility requirements -- the same
evidence standard every milestone since M52/M59 already used
(`docs/development/roadmap.md`). No speculative feature backlog is
maintained to keep development active.

## 8. Maintenance workflow

```text
real problem
     |
reproduce
     |
classify              (section 4)
     |
determine whether Forge should solve it   (section 7)
     |
smallest appropriate change
     |
regression test        (section 2)
     |
real-world validation where relevant   (section 3)
     |
release                (section 5)
```

This replaces the milestone-driven feature-accumulation model
`docs/development/workflow.md`'s control loop describes for the
pre-1.0 phase; `workflow.md`'s AI-assisted-development/scope-control/git
rules still apply unchanged to whatever change this workflow produces.

## 9. Documentation maintenance

Update documentation only when the behavior it describes changes, as part
of that same change -- no documentation work independent of an actual code
change (`CLAUDE.md`). A documentation-only correction (stale example, wrong
default) is itself a Patch (section 5).
