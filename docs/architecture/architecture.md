# Forge Architecture

## Architectural layers

```text
Public API / CLI
       ↓
Training & Evaluation
       ↓
Models / Modules / Losses / Optimizers
       ↓
Tensor + Autograd
       ↓
Device / Backend Abstraction
       ├── CPU
       └── CUDA

Data subsystem feeds Training:
Dataset → Transforms → DataLoader → Batches

Persistence crosses the model/parameter boundary.
Benchmarking measures lower layers and end-to-end workloads.
```

## Design rules
1. High-level APIs must not contain backend-specific tensor implementation details.
2. Tensor operations define the computation surface used by autograd and higher layers.
3. Autograd is attached to differentiable tensor operations, not to individual model classes.
4. Parameters are explicit trainable state.
5. Training orchestrates models, losses, gradients, optimizers, data, metrics, and devices.
6. Data loading is independent from model implementation.
7. Serialization must preserve model structure/configuration and parameter state required for reconstruction.
8. CLI commands delegate to public framework services.
9. Benchmarks must execute real code and record metadata.
10. Abstractions must be introduced only where they enforce a real boundary.

## Initial implementation strategy
Start with a minimal end-to-end CPU slice. Establish interfaces that can later host CUDA without duplicating the public framework. Add CUDA only after the CPU semantics and tests are trustworthy.
