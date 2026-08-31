# ADR-001: Numerical Foundation

## Status
Accepted

## Decision
Use established numerical infrastructure such as NumPy where it provides reliable host-side array operations, while implementing Forge's tensor semantics, autograd, neural-network abstractions, training, optimization, and backend contracts inside Forge.

## Rationale
Reimplementing every low-level numerical primitive would add complexity without improving the framework's learning value. The core intellectual responsibility should remain in Forge.

## Consequences
- NumPy may be a foundational dependency for CPU-side numerical storage/operations where appropriate.
- Forge must not delegate the core training/autograd system to PyTorch/TensorFlow.
- Dependency boundaries should remain documented as architecture evolves.
