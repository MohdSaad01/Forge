# Forge Testing Strategy

## Principles
- Test behavior, not implementation line counts.
- Keep core tests deterministic and small.
- CPU correctness is the baseline.
- CUDA tests compare semantics within appropriate tolerances.
- Tests should expose regressions at the lowest useful layer.

## Levels

### Unit
Tensor operations, shape/dtype validation, autograd rules, layers, losses, optimizers, transforms.

### Component
Model parameter registration, DataLoader behavior, serialization components, backend dispatch.

### Integration
Training a tiny model through the complete stack.

### End-to-end
User-level workflows such as train → evaluate → save → load → predict.

### Hardware
CUDA compilation/execution and CPU/CUDA consistency on supported hardware.

## Numerical testing
Use analytical gradients where practical and numerical finite-difference checks for selected operations. Define tolerances explicitly rather than requiring bit-for-bit equality.

## Resource-aware testing
Use tiny datasets/models suitable for 8 GB RAM and a 2 GB GPU. Avoid tests that require large downloads or cloud resources.
