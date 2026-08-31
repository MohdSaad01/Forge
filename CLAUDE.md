# Forge — Claude Code Instructions

## Project identity
Forge is a from-scratch deep-learning framework. It owns the ML abstractions and training machinery; it is not a wrapper around PyTorch/TensorFlow.

## Current objective
Build Forge incrementally through small, working vertical slices. Read the relevant `docs/` files before changing architecture or public APIs.

## Core principles
- Prefer simple, explicit abstractions over speculative framework machinery.
- Keep high-level model code independent of the execution backend.
- CPU must remain independently testable.
- CUDA support must be real and hardware-tested; never simulate GPU behavior.
- Use NumPy or other numerical infrastructure where it reduces unnecessary low-level reinvention, but keep tensors, autograd, model abstractions, training, and optimization inside Forge.
- Preserve clear boundaries between tensor computation, autograd, neural-network modules, data, training, serialization, and backends.
- Do not silently expand milestone scope.
- Do not introduce cloud services or paid dependencies.
- Do not commit, push, or rewrite Git history.

## Environment constraints
Development and primary verification environment: Windows, Python 3.13.5, i5-7200U, 8 GB RAM, NVIDIA 940MX (2 GB VRAM, CC 5.0), CUDA 12.6.
Forge should remain generally usable on more capable hardware; development workloads must simply remain practical on this machine.

## Testing
Every implementation change must have appropriate tests. Prefer deterministic, small tests. CPU tests must not require CUDA. CUDA-specific tests should skip cleanly when CUDA is unavailable and must be explicitly hardware-verified when the milestone requires it.

## Documentation
Update only documentation affected by meaningful architectural changes. Do not create speculative documents. Keep public API decisions documented.

## Architecture uncertainty
If a normal implementation choice can be resolved by engineering judgment, decide and proceed. Stop and report only when a choice materially changes the architecture/product and cannot reasonably be resolved from existing documentation.

## Development Environment
Forge must remain practical on the primary development environment.
See `docs/development/development-environment.md` for the verified
hardware, CUDA environment, and development constraints.

## Reporting
At milestone completion report:
1. Implemented
2. Files Changed
3. Architecture
4. Tests
5. Verification
6. Limitations
7. Next Step
8. Suggested Commit Message

Do not create the Git commit yourself.
