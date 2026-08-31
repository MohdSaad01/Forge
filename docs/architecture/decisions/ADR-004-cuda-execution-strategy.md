# ADR-004: CUDA Execution Strategy -- `nvcc`-compiled kernels loaded via `ctypes`

## Status
Accepted

## Context
Milestone 8 requires a real CUDA execution backend on the verified
development GPU (NVIDIA GeForce 940MX, Compute Capability 5.0, CUDA Toolkit
12.6, MSVC 2022), without delegating CUDA execution to another deep-learning
framework (PyTorch, TensorFlow, CuPy, and JAX are explicitly excluded by the
project's constraints) and without introducing a paid/cloud dependency.

Candidate strategies considered:
1. **`nvcc`-compiled `.cu` kernels, `extern "C"` exports, loaded via the
   standard-library `ctypes`.**
2. **A hand-written CUDA C++ extension bound with `pybind11`.** More
   idiomatic for a larger native-extension surface, but adds a build-system
   dependency (a `setup.py`/`scikit-build` native build step, a pinned
   `pybind11` version, C++ ABI compatibility with the exact Python build)
   for no benefit at this milestone's scale (six kernels, plain data types).
2. **`numba.cuda`** (JIT-compiles Python functions annotated `@cuda.jit` to
   PTX). Not one of the explicitly excluded frameworks, and avoids needing
   `nvcc`/MSVC at all. Rejected because it is a third-party JIT dependency
   the project would need to adopt as Forge's actual GPU execution
   mechanism -- more machinery than "a low-level CUDA Python binding or
   extension mechanism" the milestone brief anticipates evaluating
   carefully, and it was not needed: this environment's `nvcc`+MSVC
   toolchain was already confirmed working before this milestone started.
3. **`cupy`**: explicitly excluded by the project's constraints.

## Decision
Write kernels directly in CUDA C++ (`forge/backend/cuda/kernels.cu`), with
every entry point declared `extern "C"` (no C++ name mangling), compile them
with `nvcc` into a plain shared library (`.dll`), and call the exported
functions from Python via the standard-library `ctypes.CDLL` -- no
third-party binding generator, no build-system plugin, no additional
dependency beyond the CUDA Toolkit the hardware verification already
required.

The build step (`forge/backend/cuda/build.py`) runs lazily, the first time a
CUDA device is actually requested, and is never invoked at `import forge`
time -- a CPU-only environment/CI machine never needs `nvcc` on PATH.

## Rationale
- **No new dependency.** `ctypes` is in the Python standard library. The
  only external tool required is `nvcc` itself, which this milestone's
  premise already assumes is present and working.
- **Matches the "no other framework" constraint precisely.** Every line of
  kernel code and every line of dispatch code lives inside Forge; nothing
  is delegated to a third party's CUDA execution engine.
- **`extern "C"` + `ctypes` avoids the hardest part of a native-extension
  build.** A `pybind11`/`nanobind` extension must be built against the
  exact CPython ABI in use (version, platform tag) and typically needs a
  packaging-level build step (`setup.py build_ext`, `scikit-build-core`,
  etc.). A plain `nvcc -shared` DLL with C linkage has no such ABI
  coupling -- `ctypes.CDLL` loads it the same way regardless of the Python
  build that loads it.
- **Verified end-to-end before adoption.** Before writing any Forge code
  for this milestone, this exact pipeline (write a `.cu` kernel -> compile
  with `nvcc -arch=sm_50 -shared` -> load via `ctypes.CDLL` -> launch ->
  synchronize -> read back) was manually confirmed to execute correctly on
  the actual 940MX, including locating MSVC's `cl.exe` (nvcc's required
  host compiler on Windows) via `vswhere.exe` when it is not already on
  `PATH`.

## Consequences
- Forge's CUDA backend requires `nvcc` and, on Windows, an MSVC installation
  with the "Desktop development with C++" (or equivalent VC++ Tools)
  workload, at build/first-use time on whatever machine runs CUDA tensors.
  This is already implied by the milestone's own premise (the verified
  development environment) and is documented in
  `docs/architecture/cuda-backend.md`.
- Kernel code is C++, not Python -- adding a new CUDA kernel means editing
  `kernels.cu` and recompiling, rather than writing a `@cuda.jit`-annotated
  Python function. Acceptable for this milestone's deliberately small,
  stable operation set (see `docs/architecture/cuda-backend.md`'s operation
  table); a much larger future kernel surface might revisit this ADR.
- All CUDA compute kernels are written by hand (no cuBLAS/cuDNN), matching
  the milestone's explicit non-goals; a future performance milestone
  (M11) could introduce cuBLAS for `matmul` specifically if justified by
  benchmarks, without needing to revisit this ADR's overall strategy.
- The compiled `.dll` is a per-machine build artifact (targets one
  `-arch=sm_50` and links a specific CUDA Toolkit's static runtime) and is
  not committed to version control; each machine compiles its own copy on
  first CUDA use, cached thereafter.
